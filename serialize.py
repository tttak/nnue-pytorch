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

class NNUEWriter():
  """
  All values are stored in little endian.
  """
  def __init__(self, model, description=None, ft_compression='none'):
    if description is None:
        description = DEFAULT_DESCRIPTION

    self.buf = bytearray()

    # NOTE: model._clip_weights() should probably be called here. It's not necessary now
    # because it doesn't have more restrictive bounds than these defined by quantization,
    # but it might be necessary in the future.
    fc_hash = self.fc_hash(model)
    self.write_header(model, fc_hash, description)
    self.int32(model.feature_set.hash ^ (M.L1*2)) # Feature transformer hash
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
      self.write_fc_layer(model, lca_q)
      self.write_fc_layer(model, lca_k)
      self.write_fc_layer(model, lca_v)
      print(f"LCA Q/K/V Projection END [Pos: {len(self.buf)}]")

      print(f"LCA Temperature: {lca_temp_val}")
      self.float32(lca_temp_val)

      print(f"Phase Proj START [Pos: {len(self.buf)}]")
      self.write_fc_layer(model, phase_p)
      print(f"Phase Proj END [Pos: {len(self.buf)}]")

      print(f"Cross Proj START [Pos: {len(self.buf)}]")
      self.write_fc_layer(model, cross_p) 
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
    return layer_hash

  def write_header(self, model, fc_hash, description):
    self.int32(VERSION) # version
    self.int32(fc_hash ^ model.feature_set.hash ^ (M.L1*2)) # halfkp network hash
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

    # --- ヒストグラム表示 ---
    for p in range(4):
        phase_names = ["OPEN", "MID1", "MID2", "END"]
        print(f"--- Phase {p}: {phase_names[p]} ---")
        ascii_hist(f'P{p} MUL :', pw_softmax[p, :, 0].cpu().numpy())
        ascii_hist(f'P{p} DIFF:', pw_softmax[p, :, 1].cpu().numpy())
        ascii_hist(f'P{p} SUM :', pw_softmax[p, :, 2].cpu().numpy())

    # --- 書き出し ---
    self.write_tensor(bias_quantized.flatten().numpy(), ft_compression)
    self.write_tensor(weight_quantized.flatten().numpy(), ft_compression)
    self.write_tensor(v_quantized.flatten().numpy(), ft_compression)
    # 4段階になった pair_weights を書き出し
    self.write_tensor(pw_exported.flatten().cpu().numpy(), ft_compression)

  def write_fc_layer(self, model, layer, is_output=False):
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
    if num_output != 1 and num_output % 32 != 0:
        padded_output = num_output + (32 - (num_output % 32))

        # Biasを0でパディング
        new_b = torch.zeros(padded_output, dtype=torch.int32)
        new_b[:num_output] = bias
        bias = new_b

        # Weightの行を0でパディング
        new_w = torch.zeros(padded_output, weight.shape[1], dtype=torch.int8)
        new_w[:num_output, :] = weight
        weight = new_w
        print(f"Padding Output: {num_output} -> {padded_output}")

    ascii_hist('fc bias:', bias.numpy())
    print("layer has {}/{} clipped weights. Exceeding by {} the maximum {}.".format(clipped, total_elements, clipped_max, kMaxWeight))
    ascii_hist('fc weight:', weight.numpy())

    # FC inputs are padded to 32 elements by spec.
    num_input = weight.shape[1]
    if num_input % 32 != 0:
      num_input += 32 - (num_input % 32)
      new_w = torch.zeros(weight.shape[0], num_input, dtype=torch.int8)
      new_w[:, :weight.shape[1]] = weight
      weight = new_w

    print(f"FC Bias bytes: {bias.nbytes} (Shape: {bias.shape})")
    print(f"FC Weight bytes: {weight.nbytes} (Shape: {weight.shape})")

    self.buf.extend(bias.flatten().numpy().tobytes())
    self.buf.extend(weight.flatten().numpy().tobytes())


  def int32(self, v):
    self.buf.extend(struct.pack("<I", v))

  def float32(self, val):
    # 'f' は float (4 bytes), '<' はリトルエンディアン
    self.buf.extend(struct.pack('<f', val))


class NNUEReader():
  def __init__(self, f, feature_set):
    self.f = f
    self.feature_set = feature_set
    self.model = M.NNUE(feature_set)
    fc_hash = NNUEWriter.fc_hash(self.model)

    self.read_header(feature_set, fc_hash)
    self.read_int32(feature_set.hash ^ (M.L1*2)) # Feature transformer hash
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
      lca_q_tmp   = nn.Linear(31, 32) # Query
      lca_k_tmp   = nn.Linear(64, 32) # Key
      lca_v_tmp   = nn.Linear(64, 32) # Value

      # Phase Gate 用
      # パディングで6は32になる
      #phase_p_tmp = nn.Linear(384, 6)
      phase_p_tmp = nn.Linear(384, 32)

      # cross_proj用
      cross_p_tmp = nn.Linear(self.model.layer_stacks.cross_dim * 2, 32, bias=True)

      l2_tmp      = nn.Linear(M.L2_IN_TOTAL, M.L3)
      output_tmp  = nn.Linear(M.L3, 1)

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
      self.model.layer_stacks.cross_proj.weight.data[l1_s:l1_e, :] = cross_p_tmp.weight.data
      self.model.layer_stacks.cross_proj.bias.data[l1_s:l1_e] = cross_p_tmp.bias.data

      # LCA パラメータの分配 (バケット共通だがセット)
      self.model.layer_stacks.q_proj.weight.data = lca_q_tmp.weight.data
      self.model.layer_stacks.q_proj.bias.data   = lca_q_tmp.bias.data
      self.model.layer_stacks.k_proj.weight.data = lca_k_tmp.weight.data
      self.model.layer_stacks.k_proj.bias.data   = lca_k_tmp.bias.data
      self.model.layer_stacks.v_proj.weight.data = lca_v_tmp.weight.data
      self.model.layer_stacks.v_proj.bias.data   = lca_v_tmp.bias.data
      self.model.layer_stacks.lca_temp.data      = torch.tensor(lca_temp_val)

      # Phase Gate パラメータの分配 (全バケット共通だが、最新の値をセット)
      self.model.layer_stacks.phase_proj.weight.data = phase_p_tmp.weight.data[:6, :]
      self.model.layer_stacks.phase_proj.bias.data   = phase_p_tmp.bias.data[:6]

      # Blend Parameter (alpha)
      eps = 1e-6
      safe_val = max(eps, min(1.0 - eps, bucket_blend_val))
      self.model.layer_stacks.blend.data[i] = torch.tensor(safe_val).logit()

      # L2 & Output
      l2_s_idx, l2_e_idx = i * M.L3, (i + 1) * M.L3
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
  parser = argparse.ArgumentParser(description="Converts files between ckpt and nnue format.")
  parser.add_argument("source", help="Source file (can be .ckpt, .pt or .nnue)")
  parser.add_argument("target", help="Target file (can be .pt or .nnue)")
  parser.add_argument("--description", default=None, type=str, dest='description', help="The description string to include in the network. Only works when serializing into a .nnue file.")
  parser.add_argument("--ft_compression", default='leb128', type=str, dest='ft_compression', help="Compression method to use for FT weights and biases. Either 'none' or 'leb128'. Only allowed if saving to .nnue.")
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
      nnue = torch.load(args.source)
  elif args.source.endswith('.nnue'):
    with open(args.source, 'rb') as f:
      reader = NNUEReader(f, feature_set)
      nnue = reader.model
      if args.description is None:
        args.description = reader.description
  """
    else:
    raise Exception('Invalid network input format.')
  """

  if args.ft_compression != 'none' and not args.target.endswith('.nnue'):
    args.ft_compression = 'none'
    # raise Exception('Compression only allowed for .nnue target.')

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
  elif args.target.endswith('.nnue'):
    os.makedirs(os.path.dirname(args.target), exist_ok=True)
    writer = NNUEWriter(nnue, args.description, ft_compression=args.ft_compression)
    with open(args.target, 'wb') as f:
      f.write(writer.buf)
  else:
    raise Exception('Invalid network output format.')

if __name__ == '__main__':
  main()
