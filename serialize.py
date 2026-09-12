import argparse
import features
import math
import model as M
import struct
import torch
import io
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from functools import reduce
import operator
import numpy as np
from numba import njit
import os

def to_numpy(value):
  """Return a NumPy view/copy suitable for serialization and diagnostics."""
  if isinstance(value, torch.Tensor):
    return value.detach().cpu().numpy()
  return np.asarray(value)

def ascii_hist(name, x, bins=6):
  N,X = np.histogram(x, bins=bins)
  total = 1.0*len(x)
  width = 50
  nmax = N.max()

  print(name)
  for (xi, n) in zip(X,N):
    bar = '#'*int(n*1.0*width/nmax)
    xi = '{0: <8.4g}'.format(xi).ljust(10)
    print('{0}| {1}'.format(xi,bar))

@njit
def encode_leb_128_array(arr):
  res = []
  for v in arr:
    while True:
      byte = v & 0x7f
      v = v >> 7
      if (v == 0 and byte & 0x40 == 0) or (v == -1 and byte & 0x40 != 0):
        res.append(byte)
        break
      res.append(byte | 0x80)
  return res

@njit
def decode_leb_128_array(arr, n):
  ints = np.zeros(n)
  k = 0
  for i in range(n):
    r = 0
    shift = 0
    while True:
      byte = arr[k]
      k = k + 1
      r |= (byte & 0x7f) << shift
      shift += 7
      if (byte & 0x80) == 0:
        ints[i] = r if (byte & 0x40) == 0 else r | ~((1 << shift) - 1)
        break
  return ints

# hardcoded for now
VERSION = 0x7AF32F16
DEFAULT_DESCRIPTION = "HalfKA-KSDG3_FM-1280"
COMPACT_DESCRIPTION = "HalfKA-KSDG3_FM-1280-L2x160-NoAbsSqr"
COMPACT_FC_HASH_XOR = 0x00600000
PHASE5_DESCRIPTION = "HalfKA-KSDG3_FM-1280-L2x160-NoAbsSqr-Phase5"
PHASE5_FC1X64_DESCRIPTION = PHASE5_DESCRIPTION + "-FC1x64"
PHASE5_FC_HASH_XOR = 0x00050000
CROSS24_DESCRIPTION = PHASE5_FC1X64_DESCRIPTION + "-Cross24"
CROSS24_FC_HASH_XOR = 0x00240000
COMPACT128_DESCRIPTION = (
    "HalfKA-KSDG3_FM-1280-L2x128-NoAbsSqr-Phase5-FC1x64-"
    "Cross16-FMDiff24-FMAbsRaw24"
)
COMPACT128_FC_HASH_XOR = 0x00128010
LCA24_FC_HASH_XOR = 0x00CA2400
LCA16_FC_HASH_XOR = 0x00CA1600
LCA24_DESCRIPTION = COMPACT128_DESCRIPTION + "-LCAx24"
LCA16_DESCRIPTION = COMPACT128_DESCRIPTION + "-LCAx16"
COMPACT128_FM_DIFF_UNITS = (
    2, 10, 14, 13, 8, 6, 5, 28, 11, 3, 1, 15,
    7, 9, 12, 4, 0, 23, 27, 24, 20, 16, 22, 17,
)
COMPACT128_FM_ABS_RAW_UNITS = (
    10, 20, 28, 21, 8, 15, 4, 9, 19, 13, 17, 18,
    3, 1, 6, 25, 24, 0, 14, 12, 2, 22, 5, 31,
)
SFNN_OUTER_HASH = 0x3C203B32
SFNN_FEATURE_TRANSFORMER_HASH = 0x5F134AB8

class NNUEWriter():
  """
  All values are stored in little endian.
  """
  def __init__(self, model, description=None, ft_compression='none'):
    is_fc1x64 = getattr(model.layer_stacks, 'l3_dimensions', M.L3) == 64
    cross_width = getattr(model.layer_stacks, 'cross_output_dimensions', 32)
    diff_units = tuple(getattr(
        model.layer_stacks, 'l2_fm_diff_indices', tuple(range(32))))
    abs_units = tuple(getattr(
        model.layer_stacks, 'l2_fm_abs_raw_indices', tuple(range(32))))
    lca_qk_units = tuple(getattr(
        model.layer_stacks, 'lca_qk_indices', tuple(range(32))))
    lca_value_units = tuple(getattr(
        model.layer_stacks, 'lca_value_indices', tuple(range(32))))
    lca_width = len(lca_qk_units)
    if lca_width not in (16, 24, 32) or len(lca_value_units) != lca_width:
        raise ValueError("serializer supports matching LCA Q/K/V widths 16, 24 or 32")
    is_compact128 = (
        getattr(model.layer_stacks, 'l2_in_total', None) == 128
        and cross_width == 16
        and diff_units == COMPACT128_FM_DIFF_UNITS
        and abs_units == COMPACT128_FM_ABS_RAW_UNITS
    )
    if cross_width not in (16, 24, 32):
        raise ValueError(
            f"serializer supports Cross widths 16, 24 and 32, got {cross_width}")
    if cross_width == 16 and not is_compact128:
        raise ValueError(
            "Cross16 serialization is reserved for the fixed compact128 "
            "unit ordering")
    if description is None:
        phase5_description = (
            PHASE5_FC1X64_DESCRIPTION
            if is_fc1x64
            else PHASE5_DESCRIPTION
        )
        description = (
            LCA24_DESCRIPTION if is_compact128 and lca_width == 24 else
            LCA16_DESCRIPTION if is_compact128 and lca_width == 16 else
            COMPACT128_DESCRIPTION
            if is_compact128
            else
            CROSS24_DESCRIPTION
            if cross_width == 24
            else phase5_description
            if getattr(model, 'phase_output_dimensions', 6) == 5
            else COMPACT_DESCRIPTION
            if getattr(model, 'remove_abs_sqr_l2', False)
            else DEFAULT_DESCRIPTION
        )
    description_is_fc1x64 = (
        PHASE5_FC1X64_DESCRIPTION in description
        or description in (COMPACT128_DESCRIPTION, LCA24_DESCRIPTION,
                           LCA16_DESCRIPTION)
    )
    if is_fc1x64 != description_is_fc1x64:
        raise ValueError(
            "description/fc1 width mismatch: 64-wide models require "
            f"{PHASE5_FC1X64_DESCRIPTION!r}, and 96-wide models must not use it"
        )
    if (cross_width == 24) != (CROSS24_DESCRIPTION in description):
        raise ValueError(
            "description/Cross width mismatch: 24-wide models require "
            f"{CROSS24_DESCRIPTION!r}, and 32-wide models must not use it"
        )
    if is_compact128 != (description in (
            COMPACT128_DESCRIPTION, LCA24_DESCRIPTION, LCA16_DESCRIPTION)):
        raise ValueError(
            "description/compact128 mismatch: the 128-input fixed-order model "
            f"requires {COMPACT128_DESCRIPTION!r}"
        )
    expected_lca_description = (
        LCA24_DESCRIPTION if lca_width == 24 else
        LCA16_DESCRIPTION if lca_width == 16 else None)
    if ((expected_lca_description is not None and description != expected_lca_description)
        or (expected_lca_description is None and description in (
            LCA24_DESCRIPTION, LCA16_DESCRIPTION))):
        raise ValueError("description/LCA width mismatch")

    self.buf = bytearray()

    # NOTE: model._clip_weights() should probably be called here. It's not necessary now
    # because it doesn't have more restrictive bounds than these defined by quantization,
    # but it might be necessary in the future.
    fc_hash = self.fc_hash(model)
    self.write_header(model, fc_hash, description)
    self.int32(SFNN_FEATURE_TRANSFORMER_HASH)
    self.write_feature_transformer(model, ft_compression)

    # --- [追加] Router Layer は共通で1個だけ書き出す (ループの外) ---
    print(f"Router Layer START [Pos: {len(self.buf)}]")
    self.write_fc_layer(model, model.layer_stacks.router)
    print(f"Router Layer END [Pos: {len(self.buf)}]")

    for (l1, diff_b, abs_b, cross_p, l2, output
         , bucket_blend, lca_q, lca_k, lca_v, lca_temp_val, phase_p) in model.layer_stacks.get_coalesced_layer_stacks():

      self.int32(fc_hash)

      print(f"Main Path START [Pos: {len(self.buf)}]")
      self.write_fc_layer(model, l1)
      print(f"Main Path END [Pos: {len(self.buf)}]")

      print(f"FM Diff Path START [Pos: {len(self.buf)}]")
      self.write_fc_layer(model, diff_b)
      print(f"FM Diff Path END [Pos: {len(self.buf)}]")

      print(f"FM Abs Path START [Pos: {len(self.buf)}]")
      self.write_fc_layer(model, abs_b)
      print(f"FM Abs Path END [Pos: {len(self.buf)}]")

      print(f"LCA Q/K/V Projection START [Pos: {len(self.buf)}]")
      # Compact LCA rows are physical rows, not SIMD padding. Input columns
      # remain padded by write_fc_layer as required by the C++ affine layer.
      self.write_fc_layer(model, lca_q, pad_output=False)
      self.write_fc_layer(model, lca_k, pad_output=False)
      self.write_fc_layer(model, lca_v, pad_output=False)
      print(f"LCA Q/K/V Projection END [Pos: {len(self.buf)}]")

      print(f"LCA Temperature: {lca_temp_val}")
      self.float32(lca_temp_val)

      print(f"Phase Proj START [Pos: {len(self.buf)}]")
      self.write_fc_layer(model, phase_p)
      print(f"Phase Proj END [Pos: {len(self.buf)}]")

      print(f"Cross Proj START [Pos: {len(self.buf)}]")
      # Cross24 is a real 24-output layer on disk and in C++.  Unlike Phase5,
      # it is not padded back to 32 output rows; the architecture string/hash
      # prevents a Cross32 binary from interpreting this shorter payload.
      self.write_fc_layer(model, cross_p, pad_output=False)
      print(f"Cross Proj END [Pos: {len(self.buf)}]")

      print(f"L2 Layer START [Pos: {len(self.buf)}]")
      self.write_fc_layer(model, l2)
      print(f"L2 Layer END [Pos: {len(self.buf)}]")

      print(f"Output Layer START [Pos: {len(self.buf)}]")
      self.write_fc_layer(model, output, is_output=True)
      print(f"Output Layer END [Pos: {len(self.buf)}]")

      print(f"Bucket Blend Value: {bucket_blend}")
      self.write_blend_param(bucket_blend)
      print(f"Bucket Blend END [Pos: {len(self.buf)}]")

  def write_blend_param(self, val):
    # シグモイド適用済みの値を 0～16384 の整数で保存
    alpha = torch.sigmoid(torch.tensor(val)).item()
    int_alpha = int(alpha * 16384)
    self.int32(int_alpha)
    print(f"Alpha saved as: {int_alpha} / 16384")

  @staticmethod
  def fc_hash(model):
    prev_hash = 0xEC42E90D ^ (M.L1 * 2)
    layers = [
      model.layer_stacks.l1,
      model.layer_stacks.l2,
      model.layer_stacks.output
    ]
    for layer in layers:
      layer_hash = 0xCC03DAE4
      out_dims = layer.out_features // model.num_ls_buckets
      layer_hash += out_dims
      layer_hash ^= prev_hash >> 1
      layer_hash ^= (prev_hash << 31) & 0xFFFFFFFF
      if out_dims != 1:
        layer_hash = (layer_hash + 0x538D24C7) & 0xFFFFFFFF
      prev_hash = layer_hash
    # The legacy hash algorithm does not include input dimensions.  Preserve
    # its exact 192-input result, while assigning a distinct serialized
    # architecture hash to the 160-input NoAbsSqr model.
    if getattr(model, 'remove_abs_sqr_l2', False):
      layer_hash ^= COMPACT_FC_HASH_XOR
    if getattr(model, 'phase_output_dimensions', 6) == 5:
      layer_hash ^= PHASE5_FC_HASH_XOR
    if getattr(model.layer_stacks, 'cross_output_dimensions', 32) == 24:
      layer_hash ^= CROSS24_FC_HASH_XOR
    if getattr(model.layer_stacks, 'l2_in_total', None) == 128:
      layer_hash ^= COMPACT128_FC_HASH_XOR
    lca_width = len(getattr(
        model.layer_stacks, 'lca_qk_indices', tuple(range(32))))
    if lca_width == 24:
      layer_hash ^= LCA24_FC_HASH_XOR
    elif lca_width == 16:
      layer_hash ^= LCA16_FC_HASH_XOR
    return layer_hash

  def write_header(self, model, fc_hash, description):
    self.int32(VERSION) # version
    # SFNNwoPSQT uses a fixed outer architecture hash in the C++ loader.
    self.int32(SFNN_OUTER_HASH)
    encoded_description = description.encode('utf-8')
    self.int32(len(encoded_description)) # Network definition
    self.buf.extend(encoded_description)

  def write_leb_128_array(self, arr):
    buf = encode_leb_128_array(arr)
    self.int32(len(buf))
    self.buf.extend(buf)

  def write_tensor(self, arr, compression='none'):
    print(f"[Pos: {len(self.buf)}]")

    if compression == 'none':
      self.buf.extend(arr.tobytes())
    elif compression == 'leb128':
      self.buf.extend('COMPRESSED_LEB128'.encode('utf-8'))
      self.write_leb_128_array(arr)
    else:
      raise Exception('Invalid compression method.')

    print(f"[Pos: {len(self.buf)}]")

  def write_feature_transformer(self, model, ft_compression):
    layer = model.input

    # --- 1. Bias ---
    bias_data = layer.bias.data
    bias_quantized = bias_data.mul(model.quantized_one).round().to(torch.int16)
    print(f"FT Bias bytes: {bias_quantized.nbytes} (Shape: {bias_quantized.shape})")

    # --- 2. Weight ---
    all_weight = M.coalesce_ft_weights(model, layer.weight.data)
    weight_quantized = all_weight.mul(model.quantized_one).round().to(torch.int16)
    print(f"FT Weight bytes: {weight_quantized.nbytes} (Shape: {weight_quantized.shape})")

    # --- 3. v (因子ベクトル) ---
    all_v = M.coalesce_ft_weights(model, layer.v.data) 

    # スケールを model.quantized_one(127) から 2032 (127*16) に引き上げる
    V_CUSTOM_SCALE = model.quantized_one * 16.0 
    v_quantized = all_v.mul(V_CUSTOM_SCALE).round().to(torch.int16)

    print(f"FT V Weight (High Precision) bytes: {v_quantized.nbytes} (Shape: {v_quantized.shape})")

    # --- 4. pair_weight ---
    # pw_raw: [4, 640, 3] (0:序盤, 1:中盤1, 2:中盤2, 3:終盤)
    pw_raw = model.pair_weights.data 

    # nn.binから学習parameterを復元するための4 phaseデータ
    # Softmax は dim=2 (Mul/Diff/Sum の次元) でかける
    pw_softmax = torch.softmax(pw_raw, dim=2) 

    # 14bit (16384) スケールで量子化
    W_SCALE = 16384.0 
    pw_quantized = pw_softmax.mul(W_SCALE).round().to(torch.int16)

    # --- 各フェーズ・各ペアごとに丸め誤差の補正 ---
    # pw_quantized.shape は [4, 640, 3]
    # 合計が 16384 になるように調整
    for p in range(4):
        # [640]
        diffs = 16384 - pw_quantized[p].sum(dim=1, dtype=torch.int32)
        for i in range(640):
            if diffs[i] != 0:
                # そのペアの中で最大値を持つインデックスで調整
                max_idx = torch.argmax(pw_quantized[p, i])
                pw_quantized[p, i, max_idx] += diffs[i].item()

    # --- 転置エクスポートの準備 ---
    # C++側で読みやすいように [Phase, Type, Dimensions] に並び替える
    # 最終的な並び: [P0_Mul][P0_Diff][P0_Sum]...[P3_Sum]
    # .permute(0, 2, 1) で [4, 3, 640] に変換
    pw_exported = pw_quantized.permute(0, 2, 1).contiguous()

    print(f"FT Pair Weight (4-Phase x 3-terms) bytes: {pw_exported.nbytes}")
    print(f"Shape: {pw_exported.shape} (Phase, Type[M/D/S], 640)")

    # C++推論用の12 bucketデータ。
    # model.pyのforwardと同じく、4 phaseのraw logitsを先に補間してから
    # Mul/Diff/Sum次元にsoftmaxを適用する。
    bucket_indices = torch.arange(
        12,
        device=pw_raw.device,
        dtype=pw_raw.dtype,
    ).view(12, 1, 1)
    pf = bucket_indices / 11.0
    p3 = pf * 3.0
    w0 = torch.clamp(1.0 - p3, min=0.0)
    w1 = torch.clamp(1.0 - torch.abs(p3 - 1.0), min=0.0)
    w2 = torch.clamp(1.0 - torch.abs(p3 - 2.0), min=0.0)
    w3 = torch.clamp(p3 - 2.0, min=0.0)

    pw_bucket_logits = (
        w0 * pw_raw[0]
        + w1 * pw_raw[1]
        + w2 * pw_raw[2]
        + w3 * pw_raw[3]
    )
    pw_bucket_softmax = torch.softmax(pw_bucket_logits, dim=2)
    pw_bucket_quantized = pw_bucket_softmax.mul(W_SCALE).round().to(torch.int16)

    # 各bucket・各channelのMul/Diff/Sum合計を16384に補正する。
    for b in range(12):
        diffs = 16384 - pw_bucket_quantized[b].sum(dim=1, dtype=torch.int32)
        for i in range(640):
            if diffs[i] != 0:
                max_idx = torch.argmax(pw_bucket_quantized[b, i])
                pw_bucket_quantized[b, i, max_idx] += diffs[i].item()

    # [12, 640, 3] -> [12, 3, 640]
    pw_bucket_exported = pw_bucket_quantized.permute(0, 2, 1).contiguous()

    print(f"FT Pair Weight (12-Bucket x 3-terms) bytes: {pw_bucket_exported.nbytes}")
    print(f"Shape: {pw_bucket_exported.shape} (Bucket, Type[M/D/S], 640)")

    # --- ヒストグラム表示 ---
    for p in range(4):
        phase_names = ["OPEN", "MID1", "MID2", "END"]
        print(f"--- Phase {p}: {phase_names[p]} ---")
        ascii_hist(f'P{p} MUL :', to_numpy(pw_softmax[p, :, 0]))
        ascii_hist(f'P{p} DIFF:', to_numpy(pw_softmax[p, :, 1]))
        ascii_hist(f'P{p} SUM :', to_numpy(pw_softmax[p, :, 2]))

    # --- 書き出し ---
    self.write_tensor(to_numpy(bias_quantized.flatten()), ft_compression)
    self.write_tensor(to_numpy(weight_quantized.flatten()), ft_compression)
    self.write_tensor(to_numpy(v_quantized.flatten()), ft_compression)
    # 4 phase復元用、12 bucket推論用の順で書き出す
    self.write_tensor(to_numpy(pw_exported.flatten()), ft_compression)
    self.write_tensor(to_numpy(pw_bucket_exported.flatten()), ft_compression)

  def write_fc_layer(self, model, layer, is_output=False, pad_output=True):
    # FC layers are stored as int8 weights, and int32 biases
    kWeightScaleHidden = model.weight_scale_hidden
    kWeightScaleOut = model.nnue2score * model.weight_scale_out / model.quantized_one
    kWeightScale = kWeightScaleOut if is_output else kWeightScaleHidden
    kBiasScaleOut = model.weight_scale_out * model.nnue2score
    kBiasScaleHidden = model.weight_scale_hidden * model.quantized_one
    kBiasScale = kBiasScaleOut if is_output else kBiasScaleHidden
    kMaxWeight = model.quantized_one / kWeightScale

    print(f"kBiasScale={kBiasScale}")
    print(f"kWeightScale={kWeightScale}")

    bias = layer.bias.data
    bias = bias.mul(kBiasScale).round().to(torch.int32)

    weight = layer.weight.data
    clipped = torch.count_nonzero(weight.clamp(-kMaxWeight, kMaxWeight) - weight)
    total_elements = torch.numel(weight)
    clipped_max = torch.max(torch.abs(weight.clamp(-kMaxWeight, kMaxWeight) - weight))

    weight = weight.clamp(-kMaxWeight, kMaxWeight).mul(kWeightScale).round().to(torch.int8)

    # --- パディング (Bias と Weight の行数) ---
    num_output = weight.shape[0]
    if pad_output and num_output != 1 and num_output % 32 != 0:
        padded_output = num_output + (32 - (num_output % 32))

        # Biasを0でパディング
        new_b = torch.zeros(
            padded_output, dtype=torch.int32, device=bias.device)
        new_b[:num_output] = bias
        bias = new_b

        # Weightの行を0でパディング
        new_w = torch.zeros(
            padded_output,
            weight.shape[1],
            dtype=torch.int8,
            device=weight.device)
        new_w[:num_output, :] = weight
        weight = new_w
        print(f"Padding Output: {num_output} -> {padded_output}")

    ascii_hist('fc bias:', to_numpy(bias))
    print("layer has {}/{} clipped weights. Exceeding by {} the maximum {}.".format(clipped, total_elements, clipped_max, kMaxWeight))
    ascii_hist('fc weight:', to_numpy(weight))

    # FC inputs are padded to 32 elements by spec.
    num_input = weight.shape[1]
    if num_input % 32 != 0:
      num_input += 32 - (num_input % 32)
      new_w = torch.zeros(
          weight.shape[0], num_input, dtype=torch.int8, device=weight.device)
      new_w[:, :weight.shape[1]] = weight
      weight = new_w

    print(f"FC Bias bytes: {bias.nbytes} (Shape: {bias.shape})")
    print(f"FC Weight bytes: {weight.nbytes} (Shape: {weight.shape})")

    self.buf.extend(to_numpy(bias.flatten()).tobytes())
    self.buf.extend(to_numpy(weight.flatten()).tobytes())


  def int32(self, v):
    self.buf.extend(struct.pack("<I", v))

  def float32(self, val):
    # 'f' は float (4 bytes), '<' はリトルエンディアン
    self.buf.extend(struct.pack('<f', val))


class NNUEReader():
  def __init__(self, f, feature_set):
    self.f = f
    self.feature_set = feature_set

    # Read enough of the header to select the physical fc_1 input shape before
    # allocating the PyTorch model.  192-input files retain their old hash;
    # compact files carry the input-dimension discriminator above.
    version = self.read_int32()
    network_hash = self.read_int32()
    desc_len = self.read_int32()
    self.description = self.f.read(desc_len).decode('utf-8')
    if version != VERSION:
      raise Exception('Unsupported NNUE version: 0x%08x' % version)

    is_compact128 = self.description in (
        COMPACT128_DESCRIPTION, LCA24_DESCRIPTION, LCA16_DESCRIPTION)
    lca_qk_indices = (
        (0, 16, 28, 22, 3, 13, 19, 29, 10, 20, 7, 5, 23, 6, 2, 31,
         14, 30, 4, 8, 24, 11, 21, 15)
        if self.description == LCA24_DESCRIPTION else
        (0, 16, 28, 22, 3, 13, 19, 29, 10, 20, 7, 5, 23, 6, 2, 31)
        if self.description == LCA16_DESCRIPTION else tuple(range(32)))
    lca_value_indices = (
        (15, 9, 1, 2, 7, 5, 8, 14, 3, 11, 28, 6, 4, 24, 20, 31,
         0, 16, 23, 29, 13, 17, 27, 22)
        if self.description == LCA24_DESCRIPTION else
        (15, 9, 1, 2, 7, 5, 8, 14, 3, 11, 28, 6, 4, 24, 20, 31)
        if self.description == LCA16_DESCRIPTION else tuple(range(32)))
    remove_abs_sqr_l2 = COMPACT_DESCRIPTION in self.description or is_compact128
    phase_output_dimensions = (
        M.PHASE_CHANNELS_NO_ABS_SQR
        if PHASE5_DESCRIPTION in self.description or is_compact128
        else M.PHASE_CHANNELS_LEGACY
    )
    l3_dimensions = (
        M.L3
        if PHASE5_FC1X64_DESCRIPTION in self.description or is_compact128
        else M.L3_LEGACY
    )
    cross_output_dimensions = (
        16 if is_compact128 else
        24 if CROSS24_DESCRIPTION in self.description else 32
    )
    self.model = M.NNUE(
        feature_set, remove_abs_sqr_l2=remove_abs_sqr_l2,
        phase_output_dimensions=phase_output_dimensions,
        l3_dimensions=l3_dimensions,
        cross_output_dimensions=cross_output_dimensions,
        l2_fm_diff_indices=(
            COMPACT128_FM_DIFF_UNITS if is_compact128 else None),
        l2_fm_abs_raw_indices=(
            COMPACT128_FM_ABS_RAW_UNITS if is_compact128 else None),
        lca_qk_indices=lca_qk_indices,
        lca_value_indices=lca_value_indices)
    fc_hash = NNUEWriter.fc_hash(self.model)
    expected_network_hash = fc_hash ^ feature_set.hash ^ (M.L1 * 2)
    # Accept legacy serializer output as well as the C++ SFNN fixed hash.
    if network_hash not in (SFNN_OUTER_HASH, expected_network_hash):
      raise Exception(
          'NNUE architecture hash mismatch: expected 0x%08x, got 0x%08x'
          % (expected_network_hash, network_hash))
    self.read_int32(SFNN_FEATURE_TRANSFORMER_HASH)
    self.read_feature_transformer(self.model.input)

    self.model.layer_stacks.l1_fact.weight.data.fill_(0.0)
    self.model.layer_stacks.l1_fact.bias.data.fill_(0.0)


    # --- [追加] Router Layer は共通で1個だけ読み込む (ループの外) ---
    # パディングで12は32になる
    router_p_tmp = nn.Linear(384, 32)
    self.read_fc_layer(router_p_tmp)
    self.model.layer_stacks.router.weight.data = router_p_tmp.weight.data[:12, :]
    self.model.layer_stacks.router.bias.data   = router_p_tmp.bias.data[:12]


    for i in range(self.model.num_ls_buckets):
      # --- 1. 一時レイヤーの定義 ---
      l1_tmp      = nn.Linear(M.L1_MAIN, 32)
      diff_b_tmp  = nn.Linear(128, 64) 
      abs_b_tmp   = nn.Linear(128, 64)

      # LCA Projection用
      lca_q_tmp   = nn.Linear(31, len(lca_qk_indices)) # Query
      lca_k_tmp   = nn.Linear(64, len(lca_qk_indices)) # Key
      lca_v_tmp   = nn.Linear(64, len(lca_value_indices)) # Value

      # Phase Gate 用
      # Phase5/Phase6 are both padded to 32 physical output rows on disk.
      phase_p_tmp = nn.Linear(384, 32)

      # cross_proj用
      cross_p_tmp = nn.Linear(
          self.model.layer_stacks.cross_dim * 2,
          self.model.layer_stacks.cross_output_dimensions,
          bias=True)

      l3_dimensions = self.model.layer_stacks.l3_dimensions
      l2_tmp      = nn.Linear(self.model.layer_stacks.l2_in_total, l3_dimensions)
      output_tmp  = nn.Linear(l3_dimensions, 1)

      # --- 2. バイナリからの読み込み ---
      self.read_int32(fc_hash)
      self.read_fc_layer(l1_tmp)      # Main Path
      self.read_fc_layer(diff_b_tmp)  # FM Diff Path
      self.read_fc_layer(abs_b_tmp)   # FM Abs Path

      # --- LCA 関連
      self.read_fc_layer(lca_q_tmp)   # LCA Query
      self.read_fc_layer(lca_k_tmp)   # LCA Key
      self.read_fc_layer(lca_v_tmp)   # LCA Value

      # LCA Temperature
      import struct
      lca_temp_raw = self.f.read(4)
      lca_temp_val = struct.unpack('<f', lca_temp_raw)[0]

      # Phase Gate
      self.read_fc_layer(phase_p_tmp)

      # Cross Projection
      self.read_fc_layer(cross_p_tmp)

      # L2 Layer, Output Layer
      self.read_fc_layer(l2_tmp)
      self.read_fc_layer(output_tmp, is_output=True)

      # Blend Parameter (alpha)
      int_alpha = self.read_int32() 
      bucket_blend_val = float(int_alpha) / 16384.0
      
      # --- 3. モデルの各パラメータ (バケット別) へ分配 ---
      # MainPath は 32次元単位
      l1_s, l1_e = i * 32, (i + 1) * 32
      # FMPath は 64次元単位 (Gate 32 + Value 32)
      fm_s, fm_e = i * 64, (i + 1) * 64
      
      # L1 Main
      self.model.layer_stacks.l1.weight.data[l1_s:l1_e, :] = l1_tmp.weight.data
      self.model.layer_stacks.l1.bias.data[l1_s:l1_e] = l1_tmp.bias.data

      # FM Diff
      self.model.layer_stacks.fm_diff.weight.data[fm_s:fm_e, :] = diff_b_tmp.weight.data
      self.model.layer_stacks.fm_diff.bias.data[fm_s:fm_e] = diff_b_tmp.bias.data

      # FM Abs
      self.model.layer_stacks.fm_abs.weight.data[fm_s:fm_e, :] = abs_b_tmp.weight.data
      self.model.layer_stacks.fm_abs.bias.data[fm_s:fm_e] = abs_b_tmp.bias.data

      # Cross Projection
      cross_dims = self.model.layer_stacks.cross_output_dimensions
      cross_s, cross_e = i * cross_dims, (i + 1) * cross_dims
      self.model.layer_stacks.cross_proj.weight.data[cross_s:cross_e, :] = \
          cross_p_tmp.weight.data[:cross_dims, :]
      self.model.layer_stacks.cross_proj.bias.data[cross_s:cross_e] = \
          cross_p_tmp.bias.data[:cross_dims]

      # LCA パラメータの分配 (バケット共通だがセット)
      self.model.layer_stacks.q_proj.weight.data = lca_q_tmp.weight.data
      self.model.layer_stacks.q_proj.bias.data   = lca_q_tmp.bias.data
      self.model.layer_stacks.k_proj.weight.data = lca_k_tmp.weight.data
      self.model.layer_stacks.k_proj.bias.data   = lca_k_tmp.bias.data
      self.model.layer_stacks.v_proj.weight.data = lca_v_tmp.weight.data
      self.model.layer_stacks.v_proj.bias.data   = lca_v_tmp.bias.data
      self.model.layer_stacks.lca_temp.data      = torch.tensor(lca_temp_val)

      # Phase Gate パラメータの分配 (全バケット共通だが、最新の値をセット)
      phase_dims = self.model.layer_stacks.phase_output_dimensions
      self.model.layer_stacks.phase_proj.weight.data = phase_p_tmp.weight.data[:phase_dims, :]
      self.model.layer_stacks.phase_proj.bias.data   = phase_p_tmp.bias.data[:phase_dims]

      # Blend Parameter (alpha)
      eps = 1e-6
      safe_val = max(eps, min(1.0 - eps, bucket_blend_val))
      self.model.layer_stacks.blend.data[i] = torch.tensor(safe_val).logit()

      # L2 & Output
      l2_s_idx = i * l3_dimensions
      l2_e_idx = (i + 1) * l3_dimensions
      self.model.layer_stacks.l2.weight.data[l2_s_idx:l2_e_idx, :] = l2_tmp.weight.data
      self.model.layer_stacks.l2.bias.data[l2_s_idx:l2_e_idx] = l2_tmp.bias.data
      
      self.model.layer_stacks.output.weight.data[i:(i+1), :] = output_tmp.weight.data
      self.model.layer_stacks.output.bias.data[i:(i+1)] = output_tmp.bias.data


  def read_header(self, feature_set, fc_hash):
    self.read_int32(VERSION) # version
    self.read_int32(fc_hash ^ feature_set.hash ^ (M.L1*2))
    desc_len = self.read_int32()
    self.description = self.f.read(desc_len).decode('utf-8')

  def read_leb_128_array(self, dtype, shape):
    l = self.read_int32()
    d = self.f.read(l)
    if len(d) != l:
      raise Exception('Unexpected end of file when reading compressed data.')

    res = torch.FloatTensor(decode_leb_128_array(d, reduce(operator.mul, shape, 1)))
    res = res.reshape(shape)
    return res

  def peek(self, length=1):
    pos = self.f.tell()
    data = self.f.read(length)
    self.f.seek(pos)
    return data

  def determine_compression(self):
    leb128_magic = b'COMPRESSED_LEB128'
    if self.peek(len(leb128_magic)) == leb128_magic:
      self.f.read(len(leb128_magic)) # actually advance the file pointer
      return 'leb128'
    else:
      return 'none'

  def tensor(self, dtype, shape):
    compression = self.determine_compression()

    if compression == 'none':
      d = np.fromfile(self.f, dtype, reduce(operator.mul, shape, 1))
      d = torch.from_numpy(d.astype(np.float32))
    else:
      d = self.read_leb_128_array(dtype, shape)
    d = d.reshape(shape)

    print(f"Current file position: {self.f.tell()} bytes")
    return d

  def read_feature_transformer(self, layer):
    # --- bias
    bias = self.tensor(np.int16, [layer.bias.shape[0]]).divide(self.model.quantized_one)
    print(f"read_feature_transformer bias END")
    
    # --- weights
    shape = layer.weight.shape
    weights = self.tensor(np.int16, [self.feature_set.num_real_features, shape[1]]).divide(self.model.quantized_one)
    print(f"read_feature_transformer weights END")

    # --- v_weights
    # 保存時に 16倍 (V_CUSTOM_SCALE) したので、読み込み時も 16倍で割る
    V_CUSTOM_SCALE = self.model.quantized_one * 16.0
    v_weights = self.tensor(np.int16, [self.feature_set.num_real_features, layer.factor_dim]).divide(V_CUSTOM_SCALE)
    print(f"read_feature_transformer v_weights END")

    # --- 4. Pair Weight
    # 保存時（エクスポート時）の形状は [4, 3, 640]
    # (Phase, Type[M/D/S], 640)
    W_SCALE = 16384.0
    
    # 1. 保存時の形状 [4, 3, 640] で読み込み
    # tensor関数の引数に [4, 3, 640] を指定
    pw_exported = self.tensor(np.int16, [4, 3, 640]).divide(W_SCALE)
    print(f"read_feature_transformer pair_weights END")

    # C++推論用の12 bucketデータ。PyTorchモデルの復元には使用しないが、
    # 後続のrouter/network weightを正しい位置から読むために読み飛ばす。
    self.tensor(np.int16, [12, 3, 640])
    print(f"read_feature_transformer inference_pair_weights END")
    
    # 2. モデルの Parameter 形状 [4, 640, 3] に戻す
    # [4, 3, 640] -> [4, 640, 3] へ軸を入れ替え
    pair_w_restored = pw_exported.permute(0, 2, 1).contiguous()
    
    # 3. 対数変換 (Logits復元)
    # Forward時に Softmax を通すため、逆操作として log をとる
    # 0があると log(0) で -inf になるため、非常に小さい値 eps で clamp
    eps = 1e-6
    pair_w_clamped = torch.clamp(pair_w_restored, eps, 1.0)
    pair_w_logits = torch.log(pair_w_clamped)

    # --- 各レイヤーへの代入 ---
    layer.bias.data = bias
    layer.weight.data = weights
    layer.v.data = v_weights

    # モデル側の Parameter に 4段階の Logits を代入
    # self.model.pair_weights.shape は [4, 640, 3]
    self.model.pair_weights.data = pair_w_logits


  def read_fc_layer(self, layer, is_output=False):
    kWeightScaleHidden = self.model.weight_scale_hidden
    kWeightScaleOut = self.model.nnue2score * self.model.weight_scale_out / self.model.quantized_one
    kWeightScale = kWeightScaleOut if is_output else kWeightScaleHidden
    kBiasScaleOut = self.model.weight_scale_out * self.model.nnue2score
    kBiasScaleHidden = self.model.weight_scale_hidden * self.model.quantized_one
    kBiasScale = kBiasScaleOut if is_output else kBiasScaleHidden
    kMaxWeight = self.model.quantized_one / kWeightScale

    # FC inputs are padded to 32 elements by spec.
    non_padded_shape = layer.weight.shape
    padded_shape = (non_padded_shape[0], ((non_padded_shape[1]+31)//32)*32)

    layer.bias.data = self.tensor(np.int32, layer.bias.shape).divide(kBiasScale)
    layer.weight.data = self.tensor(np.int8, padded_shape).divide(kWeightScale)

    # Strip padding.
    layer.weight.data = layer.weight.data[:non_padded_shape[0], :non_padded_shape[1]]

  def read_int32(self, expected=None):
    v = struct.unpack("<I", self.f.read(4))[0]
    #if expected is not None and v != expected:
    #  raise Exception("Expected: %x, got %x" % (expected, v))
    return v

def main():
  parser = argparse.ArgumentParser(description="Converts files between ckpt and NNUE network formats.")
  parser.add_argument("source", help="Source file (can be .ckpt, .pt, .nnue or .bin)")
  parser.add_argument("target", help="Target file (can be .pt, .nnue or .bin)")
  parser.add_argument("--description", default=None, type=str, dest='description', help="The description string to include in the network. Only works when serializing into a .nnue or .bin file.")
  parser.add_argument("--ft_compression", default='leb128', type=str, dest='ft_compression', help="Compression method to use for FT weights and biases. Either 'none' or 'leb128'. Only allowed if saving to .nnue or .bin.")
  parser.add_argument("--ft_perm", default=None, type=str, dest='ft_perm', help="Path to a file that defines the permutation to use on the feature transformer.")
  parser.add_argument("--ft_optimize", action='store_true', dest='ft_optimize', help="Whether to perform full feature transformer optimization (ftperm.py) on the resulting network. This process is very time consuming.")
  parser.add_argument("--ft_optimize_data", default=None, type=str, dest='ft_optimize_data', help="Path to the dataset to use for FT optimization.")
  parser.add_argument("--ft_optimize_count", default=10000, type=int, dest='ft_optimize_count', help="Number of positions to use for FT optimization.")
  features.add_argparse_args(parser)
  args = parser.parse_args()

  feature_set = features.get_feature_set_from_name(args.features)

  print('Converting %s to %s' % (args.source, args.target))

  if args.source.endswith('.ckpt'):
    nnue = M.NNUE.load_from_checkpoint(args.source, feature_set=feature_set)
    nnue.eval()
  elif args.source.endswith('.pt'):
      saved = torch.load(args.source, map_location='cpu', weights_only=False)
      if isinstance(saved, M.NNUE):
        nnue = saved
      elif isinstance(saved, dict) and 'state_dict' in saved:
        architecture = saved.get('architecture', {})
        l2_input_physical = architecture.get('l2_input_physical')
        l3_dimensions = int(architecture.get('fc1_output_dimensions', M.L3))
        cross_output_dimensions = int(
            architecture.get('cross_output_dimensions', 32))
        l2_fm_diff_indices = tuple(
            architecture.get('fm_diff_kept_source_units', range(32)))
        l2_fm_abs_raw_indices = tuple(
            architecture.get('fm_abs_raw_kept_source_units', range(32)))
        lca_qk_indices = tuple(
            architecture.get('lca_qk_kept_source_units', range(32)))
        lca_value_indices = tuple(
            architecture.get('lca_value_kept_source_units', range(32)))
        if l3_dimensions not in (M.L3, M.L3_LEGACY):
          raise Exception(
              'Unsupported .pt fc1 output architecture: %r'
              % (l3_dimensions,))
        expected_l2_input = (
            M.L2_IN_TOTAL_WITHOUT_ABS_SQR
            - (32 - cross_output_dimensions)
            - (32 - len(l2_fm_diff_indices))
            - (32 - len(l2_fm_abs_raw_indices))
        )
        if cross_output_dimensions not in (16, 24, 32):
          raise Exception(
              'Unsupported .pt Cross output architecture: %r'
              % (cross_output_dimensions,))
        if cross_output_dimensions == 16 and (
            l2_fm_diff_indices != COMPACT128_FM_DIFF_UNITS
            or l2_fm_abs_raw_indices != COMPACT128_FM_ABS_RAW_UNITS
            or l2_input_physical != 128):
          raise Exception(
              'Cross16 .pt must use the fixed compact128 FM unit ordering')
        if l2_input_physical not in (M.L2_IN_TOTAL,
                                     M.L2_IN_TOTAL_WITHOUT_ABS_SQR,
                                     expected_l2_input):
          raise Exception(
              'Unsupported or missing .pt L2 architecture: %r'
              % (l2_input_physical,))
        nnue = M.NNUE(
            feature_set,
            remove_abs_sqr_l2=(
                l2_input_physical != M.L2_IN_TOTAL),
            phase_output_dimensions=int(architecture.get(
                'phase_output_dimensions',
                M.PHASE_CHANNELS_NO_ABS_SQR
                if l2_input_physical != M.L2_IN_TOTAL
                else M.PHASE_CHANNELS_LEGACY)),
            l3_dimensions=l3_dimensions,
            cross_output_dimensions=cross_output_dimensions,
            l2_fm_diff_indices=l2_fm_diff_indices,
            l2_fm_abs_raw_indices=l2_fm_abs_raw_indices,
            lca_qk_indices=lca_qk_indices,
            lca_value_indices=lca_value_indices)
        state_dict = saved['state_dict']
        if l2_input_physical != M.L2_IN_TOTAL:
          M.migrate_phase_proj_state_dict_to_five(state_dict)
        nnue.load_state_dict(state_dict, strict=True)
      else:
        raise Exception(
            '.pt source must contain an NNUE model or a state_dict package')
      nnue.eval()
  elif args.source.endswith(('.nnue', '.bin')):
    with open(args.source, 'rb') as f:
      reader = NNUEReader(f, feature_set)
      nnue = reader.model
      if args.description is None:
        args.description = reader.description
  """
    else:
    raise Exception('Invalid network input format.')
  """

  if args.ft_compression != 'none' and not args.target.endswith(('.nnue', '.bin')):
    args.ft_compression = 'none'
    # raise Exception('Compression only allowed for .nnue or .bin target.')

  # The compact production architecture has no FM AbsSqr L2 consumer.  When
  # exporting an older Phase6 160-input checkpoint/object, move old Cross row
  # 5 to row 4 and discard only the provably unused old AbsSqr row 4.
  if (args.target.endswith(('.nnue', '.bin'))
      and getattr(nnue, 'remove_abs_sqr_l2', False)
      and getattr(nnue.layer_stacks.phase_proj, 'out_features', 6) == 6):
    M.migrate_model_phase_to_five(nnue)
    if args.description in (None, COMPACT_DESCRIPTION):
      args.description = (
          PHASE5_FC1X64_DESCRIPTION
          if nnue.layer_stacks.l3_dimensions == 64
          else PHASE5_DESCRIPTION
      )

  if args.ft_compression not in ['none', 'leb128']:
    raise Exception('Invalid compression method.')

  if args.ft_optimize and args.ft_perm is not None:
    raise Exception('Options --ft_perm and --ft_optimize are mutually exclusive.')

  if args.ft_perm is not None:
    import ftperm
    ftperm.ft_permute(nnue, args.ft_perm)

  if args.ft_optimize:
    import ftperm
    if args.ft_optimize_data is None:
      raise Exception('Invalid dataset path for FT optimization. (--ft_optimize_data)')
    if args.ft_optimize_count is None or args.ft_optimize_count < 1:
      raise Exception('Invalid number of positions to optimize FT with. (--ft_optimize_count)')

    ftperm.ft_optimize(nnue, args.ft_optimize_data, args.ft_optimize_count)

  if args.target.endswith('.ckpt'):
    raise Exception('Cannot convert into .ckpt')
  elif args.target.endswith('.pt'):
    torch.save(nnue, args.target)
  elif args.target.endswith(('.nnue', '.bin')):
    os.makedirs(os.path.dirname(args.target), exist_ok=True)
    writer = NNUEWriter(nnue, args.description, ft_compression=args.ft_compression)
    with open(args.target, 'wb') as f:
      f.write(writer.buf)
  else:
    raise Exception('Invalid network output format.')

if __name__ == '__main__':
  main()
