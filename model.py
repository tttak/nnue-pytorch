import chess
import ranger
import ranger21
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
from feature_transformer import DoubleFeatureTransformerSlice

import bitsandbytes as bnb

# --- 定数定義 ---
L1_MAIN = 1280
FM_DIM = 32
L1 = L1_MAIN
L2 = 31
L3 = 96

L2_IN_TOTAL = 192
NUM_LS_BUCKETS = 12

def coalesce_ft_weights(model, data):
  weight = data
  indices = model.feature_set.get_virtual_to_real_features_gather_indices()
  weight_coalesced = weight.new_zeros((model.feature_set.num_real_features, weight.shape[1]))
  for i_real, is_virtual in enumerate(indices):
    if len(is_virtual) > 0:
      virt_idx = torch.tensor(is_virtual, device=weight.device)
      weight_coalesced[i_real, :] = weight.index_select(0, virt_idx).sum(dim=0)
  return weight_coalesced

def get_parameters(layers):
  return [p for layer in layers for p in layer.parameters()]

class LayerStacks(nn.Module):
  def __init__(self, count):
    super(LayerStacks, self).__init__()
    self.count = count

    #--- l1, l1_fact
    L1_OUT_PER_BUCKET = 32 
    TOTAL_L1_OUT = L1_OUT_PER_BUCKET * count
    self.l1 = nn.Linear(L1_MAIN, TOTAL_L1_OUT)
    self.l1_fact = nn.Linear(L1_MAIN, L1_OUT_PER_BUCKET, bias=True)

    #--- fm_diff, fm_abs
    FM_GLU_OUT_PER_BUCKET = 64 
    TOTAL_FM_GLU_OUT = FM_GLU_OUT_PER_BUCKET * count
    self.fm_diff = nn.Linear(128, TOTAL_FM_GLU_OUT)
    self.fm_abs = nn.Linear(128, TOTAL_FM_GLU_OUT)

    #--- l2, output
    self.l2 = nn.Linear(L2_IN_TOTAL, L3 * count)
    self.output = nn.Linear(L3, 1 * count)

    #--- blend
    self.blend = nn.Parameter(torch.zeros(count)) 

    #--- cross_proj
    self.cross_dim = 16
    self.cross_proj = nn.Linear(self.cross_dim * 2, 32 * count) 

    # --- Lightweight Cross-Attention Layers ---
    # MainPath(31次元) から FMPath(32次元) への注目度を計算
    self.q_proj = nn.Linear(31, 32) # Query
    self.k_proj = nn.Linear(64, 32) # Key
    self.v_proj = nn.Linear(64, 32) # Value

    # Temperature パラメータ (初期値 0.7 = やや鋭めからスタート)
    self.lca_temp = nn.Parameter(torch.tensor(0.7))

    #--- Phase Gate
    self.phase_proj = nn.Linear(384, 6)

    #--- lossへの加算用
    self.current_phase_for_loss = None

    #--- ログ出力用
    self.last_gate_d = None
    self.last_gate_a = None
    self.last_att_score = None
    self.last_lca_temp = None
    self.last_phase = None

    # Cached helper tensor for choosing outputs by bucket indices.
    # Initialized lazily in forward.
    self.idx_offset = None
    self._init_layers()

  def _init_layers(self):
    with torch.no_grad():
      self.l1_fact.weight.fill_(0.0)
      self.l1_fact.bias.fill_(0.0)
      self.fm_diff.weight.fill_(0.0)
      self.fm_diff.bias.fill_(0.0)
      self.fm_abs.weight.fill_(0.0)
      self.fm_abs.bias.fill_(0.0)
      self.output.bias.fill_(0.0)
      self.cross_proj.bias.fill_(0.0)

      nn.init.normal_(self.phase_proj.weight, std=0.01)
      nn.init.constant_(self.phase_proj.bias, 0.0)

      for i in range(1, self.count):
        # L1 Main (32次元)
        s1, e1 = i * 32, (i + 1) * 32
        self.l1.weight.data[s1:e1, :].copy_(self.l1.weight.data[0:32, :])
        self.l1.bias.data[s1:e1].copy_(self.l1.bias.data[0:32])
        
        # FM Paths (64次元単位でコピー)
        sf, ef = i * 64, (i + 1) * 64
        self.fm_diff.weight.data[sf:ef, :].copy_(self.fm_diff.weight.data[0:64, :])
        self.fm_diff.bias.data[sf:ef].copy_(self.fm_diff.bias.data[0:64])
        self.fm_abs.weight.data[sf:ef, :].copy_(self.fm_abs.weight.data[0:64, :])
        self.fm_abs.bias.data[sf:ef].copy_(self.fm_abs.bias.data[0:64])
        
        # cross_proj のコピー (32次元単位)
        self.cross_proj.weight.data[s1:e1, :].copy_(self.cross_proj.weight.data[0:32, :])
        self.cross_proj.bias.data[s1:e1].copy_(self.cross_proj.bias.data[0:32])

        # blend のコピー
        # すべてのバケットに bucket[0] のブレンド率を適用
        self.blend.data[i] = self.blend.data[0]

        # L2/Output
        self.l2.weight.data[i*L3:(i+1)*L3, :].copy_(self.l2.weight.data[0:L3, :])
        self.l2.bias.data[i*L3:(i+1)*L3].copy_(self.l2.bias.data[0:L3])
        self.output.weight.data[i:i+1, :].copy_(self.output.weight.data[0:1, :])
        self.output.bias.data[i:i+1].copy_(self.output.bias.data[0:1])

  def forward(self, l1_main, diff_in, abs_in, ls_indices):

    # --- PHASE 0: 初期化とインデックス準備 ---
    with torch.no_grad():
      # バッチサイズやデバイス変更に追従してオフセットを再生成
      if self.idx_offset is None or self.idx_offset.shape[0] != l1_main.shape[0] or self.idx_offset.device != ls_indices.device:
          self.idx_offset = torch.arange(0, l1_main.shape[0]*self.count, self.count, device=ls_indices.device).long()
      indices = ls_indices.flatten() + self.idx_offset


    # --- PHASE 1: PhaseGate (適応的重み付け) の計算 ---
    # 1. 特徴入力の正規化
    p_abs = torch.clamp(abs_in - 0.5, 0.0, 1.0) * 2.0
    p_abs_modified = p_abs.clone()

    # バケット情報(0-11)を最後の次元に埋め込み
    bucket_info = ls_indices.float() / 11.0
    p_abs_modified[:, 127] = bucket_info

    # 2. 特徴結合 (Abs + Diff + Mainの一部) からフェーズ判定
    p_combined = torch.cat([p_abs_modified, diff_in], dim=1) 
    main_sub = l1_main[:, :128]
    p_extra_combined = torch.cat([p_combined, main_sub], dim=1)

    # 3. 6系統のゲート(phase)を生成: [MainSqr, MainRaw, Diff, AbsR, AbsS, Cross]
    phase_logit = (self.phase_proj(p_extra_combined) * 3.0) + 1.0
    phase = 0.1 + 0.9 * torch.sigmoid(phase_logit)

    # [統計・ログ用] 勾配を切って各系統の活動状況を記録
    p_detached = phase.detach()
    phase_names = ["MainSqr", "MainRaw", "FM_Diff", "FM_AbsR", "FM_AbsS", "Cross"]
    channel_stats = []
    for i in range(6):
        p_ch = p_detached[:, i]
        stats = {
            'name': phase_names[i],
            'mean': p_ch.mean().item(),
            'std':  p_ch.std().item(),
            'min':  p_ch.min().item(),
            'max':  p_ch.max().item(),
            'low': (p_ch < 0.2).float().mean().item() * 100,
            'high':(p_ch > 0.8).float().mean().item() * 100
        }
        channel_stats.append(stats)

    if self.training:
        self.last_phase = phase.detach()
        self.current_phase_for_loss = phase # detachせずに保持


    # --- PHASE 2: FMPath (差分・絶対値特徴) の抽出 ---
    # 1. 差分(Diff)と絶対値(Abs)の特徴変換
    l1c_diff_raw = self.fm_diff(diff_in).reshape((-1, self.count, 64)).view(-1, 64)[indices]
    l1c_abs_raw = self.fm_abs(abs_in).reshape((-1, self.count, 64)).view(-1, 64)[indices]

    # 2. GLU / RMSNorm による情報選別
    gate_d, val_d = l1c_diff_raw.chunk(2, dim=-1)
    gate_a, val_a = l1c_abs_raw.chunk(2, dim=-1)

    rms_d = torch.rsqrt(val_d.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
    val_d_normed = val_d * rms_d
    l1c_diff_gated = val_d_normed
    l1c_abs_gated = val_a * torch.sigmoid(gate_a)

    # 3. 非線形変換 (ClippedReLU / Square)
    l1_diff_l2 = torch.clamp(l1c_diff_gated * 0.2 + 0.5, 0.0, 1.0)
    l1_abs_raw = torch.clamp(l1c_abs_gated * 0.05 + 0.6, 0.0, 1.0)
    l1_abs_sqr = l1_abs_raw.pow(2.0)

    # 統計用に保存
    if self.training:
        self.last_gate_d = gate_d.detach().clone()
        self.last_gate_a = gate_a.detach().clone()


    # --- PHASE 3: MainPath & Attention 制御 ---
    # 1. 盤面情報の取得と Factorized 特徴の加算
    l1c_main = self.l1(l1_main).reshape((-1, self.count, 32)).view(-1, 32)[indices]
    l1f_     = self.l1_fact(l1_main) 
    l1_combined = l1c_main + l1f_ 

    # 2. DiffゲートによるMainPathの動的抑制
    gate_d = gate_d - 0.3
    l1_combined = l1_combined * (0.5 + 0.5 * torch.sigmoid(gate_d))

    # Main: 31次元(L2へ) + 1次元(Bypassへ)
    l1_val, l1_main_bp = l1_combined.split([31, 1], dim=1)

    # L2入力用：ClippedReLU
    l1_main_sqr = torch.clamp(l1_val, 0.0, 1.0).pow(2.0) * (127/128)
    l1_main_raw = torch.clamp(l1_val, 0.0, 1.0)

    # 3. Lightweight Cross-Attention (LCA)
    # MainPath(Q) が FMPath(K,V) から必要な情報を引き出す
    q = self.q_proj(l1_main_raw)
    fm_cat = torch.cat([l1_diff_l2, l1_abs_raw], dim=1)
    k = self.k_proj(fm_cat)
    v = self.v_proj(fm_cat)

    # スコア計算 (Logit)
    logit = (q * k).sum(dim=-1, keepdim=True) / 5.656

    # Temperature の適用 (0.125以下にならないようガード)
    safe_temp = torch.clamp(self.lca_temp, min=0.125)
    att_score = torch.sigmoid(logit / safe_temp)

    # FM成分に Attention 結果を反映
    v_clamped = torch.clamp(v * 0.4 + 0.5, 0.0, 1.0)
    l1_diff_l2 = l1_diff_l2 * (1 - att_score) + v_clamped * att_score

    # 統計用に保存
    if self.training:
      self.last_att_score = att_score.detach()
      self.last_lca_temp = safe_temp.detach() # ログ用


    # --- PHASE 4: CrossFeat ---
    # Cross特徴 (Main × FM の積)
    k = self.cross_dim  # 先頭 k 次元だけ使う
    cross_diff = l1_main_sqr[:, :k] * l1_diff_l2[:, :k]
    cross_abs  = l1_main_raw[:, :k] * l1_abs_raw[:, :k]
    cross_cat = torch.cat([cross_diff, cross_abs], dim=1)  # [B, 2k]

    cross_feat = self.cross_proj(cross_cat).reshape((-1, self.count, 32)).view(-1, 32)[indices]
    cross_feat = torch.clamp(cross_feat, 0.0, 1.0) 


    # --- PHASE 5: L2 Input 構築 ---
    l2_main_sqr_weighted = l1_main_sqr * (0.5 + phase[:, 0:1] * 0.5) * 1.3
    l2_main_raw_weighted = l1_main_raw * (0.5 + phase[:, 1:2] * 0.5) * 1.5
    l2_diff_weighted     = l1_diff_l2  * (0.5 + phase[:, 2:3] * 0.5) * 1.0
    l1_abs_raw_weighted  = l1_abs_raw  * (0.5 + phase[:, 3:4] * 0.5) * 0.7
    l1_abs_sqr_weighted  = l1_abs_sqr  * (0.5 + phase[:, 4:5] * 0.5) * 0.88
    l2_cross_weighted    = cross_feat  * (0.5 + phase[:, 5:6] * 0.5) * 1.5

    l2_padding = torch.zeros((l1_main.shape[0], 2), device=l1_main.device)
    l2_input = torch.cat([
        l2_main_sqr_weighted, # [  0: 30]
        l2_main_raw_weighted, # [ 31: 61]
        l2_diff_weighted,     # [ 62: 93]
        l1_abs_raw_weighted,  # [ 94:125]
        l1_abs_sqr_weighted,  # [126:157]
        l2_cross_weighted,    # [158:189]
        l2_padding            # [190:191]
    ], dim=1)

    l2_input = torch.clamp(l2_input, 0.0, 1.0)


    # --- PHASE 6: Output (DeepPath vs Bypass) ---
    l2c_ = self.l2(l2_input).reshape((-1, self.count, L3)).view(-1, L3)[indices]
    l2x_ = torch.clamp(l2c_, 0.0, 1.0)
    l3c_ = self.output(l2x_).reshape((-1, self.count, 1)).view(-1, 1)[indices]

    # バケット別の学習パラメータ alpha で L3(Deep) と L1(Bypass) をブレンド
    bucket_blend = self.blend[ls_indices].reshape(-1, 1)
    alpha = torch.sigmoid(bucket_blend)

    l3c_ = l3c_ * alpha
    l1_main_bp = l1_main_bp * (1.0 - alpha)
    final_output = l3c_ + l1_main_bp

    return final_output, l3c_, l1_main_bp, l2_input, l1c_diff_gated, l1c_abs_gated, gate_d, gate_a, channel_stats


  def get_coalesced_layer_stacks(self):
      for i in range(self.count):
          with torch.no_grad():
              # --- 1. 推論用レイヤーの器 (Instance) の定義 ---
              l1 = nn.Linear(L1_MAIN, 32)
              diff_b = nn.Linear(128, 64)
              abs_b = nn.Linear(128, 64)
              l2 = nn.Linear(L2_IN_TOTAL, L3) 
              output = nn.Linear(L3, 1)
              cross_p = nn.Linear(self.cross_dim * 2, 32)

              # --- 2. 共通パラメータ (Shared Parameters) の取得 ---
              # これらはバケットに依存しないが、便宜上ループ内で参照を渡す
              lca_q = self.q_proj
              lca_k = self.k_proj
              lca_v = self.v_proj
              lca_temp_val = torch.clamp(self.lca_temp, min=0.125).item()
              phase_p = self.phase_proj

              # --- 3. バケット固有の重み抽出 (Slicing & Coalescing) ---

              # [Main Path] バケット別の重みに Factorized(共通)重みを合成
              l1.weight.data = self.l1.weight.data[i*32:(i+1)*32, :] + self.l1_fact.weight.data
              l1.bias.data = self.l1.bias.data[i*32:(i+1)*32] + self.l1_fact.bias.data

              # [FM Paths] インデックスを 64 単位で抽出 (GLUのGate/Valueを含む)
              diff_b.weight.data = self.fm_diff.weight.data[i*64:(i+1)*64, :]
              diff_b.bias.data = self.fm_diff.bias.data[i*64:(i+1)*64]
              abs_b.weight.data = self.fm_abs.weight.data[i*64:(i+1)*64, :]
              abs_b.bias.data = self.fm_abs.bias.data[i*64:(i+1)*64]

              # [Cross Projection] 32次元の出力をバケット単位でスライス
              s_c, e_c = i * 32, (i + 1) * 32
              cross_p.weight.data = self.cross_proj.weight.data[s_c:e_c, :]
              cross_p.bias.data = self.cross_proj.bias.data[s_c:e_c]

              # [L2 / Mid Layer] 全体の流入制御用。i番目のバケットセグメントを抽出
              l2.weight.data = self.l2.weight.data[i*L3:(i+1)*L3, :L2_IN_TOTAL]
              l2.bias.data = self.l2.bias.data[i*L3:(i+1)*L3]

              # [Output / Blend] 最終出力の重みと、バケット別ブレンド係数
              output.weight.data = self.output.weight.data[i:i+1, :]
              output.bias.data = self.output.bias.data[i:i+1]
              bucket_blend = self.blend.data[i].item()

              # --- 4. 生成したセットを yield ---
              yield (l1, diff_b, abs_b, cross_p, l2, output
                     , bucket_blend, lca_q, lca_k, lca_v, lca_temp_val, phase_p)


class NNUE(pl.LightningModule):
  """
  This model attempts to directly represent the nodchip Stockfish trainer methodology.

  lambda_ = 0.0 - purely based on game results
  lambda_ = 1.0 - purely based on search scores

  It is not ideal for training a Pytorch quantized model directly.
  """
  def __init__(self, feature_set, start_lambda=1.0, end_lambda=1.0, max_epoch=800, gamma=0.992, lr=8.75e-4, epoch_size=100_000_000, batch_size=16384, in_scaling=240, out_scaling=280, offset=270, offset1=270, offset2=270, adjust_loss=0.1):
    super(NNUE, self).__init__()
    self.num_ls_buckets = NUM_LS_BUCKETS

    # FT層: 合計 (L1_MAIN + FM_DIM * 2) 次元を出力
    self.input = DoubleFeatureTransformerSlice(feature_set.num_features, L1_MAIN, FM_DIM)

    # pair_weights
    self.pair_weights = nn.Parameter(torch.zeros(4, 640, 3))

    self.feature_set = feature_set
    self.layer_stacks = LayerStacks(self.num_ls_buckets)
    self.start_lambda = start_lambda
    self.end_lambda = end_lambda
    self.gamma = gamma
    self.lr = lr
    self.nnue2score = 600.0
    self.weight_scale_hidden = 64.0
    self.weight_scale_out = 16.0
    self.quantized_one = 127.0
    self.max_epoch = max_epoch
    self.epoch_size = epoch_size
    self.batch_size = batch_size
    self.in_scaling = in_scaling
    self.out_scaling = out_scaling
    self.offset = offset
    self.offset1 = offset1
    self.offset2 = offset2
    self.adjust_loss = adjust_loss
    self.last_bucket_losses = None

    # --- 重みクリッピングの設定 ---
    max_hidden_weight = self.quantized_one / self.weight_scale_hidden # 約1.98
    max_out_weight = (self.quantized_one * self.quantized_one) / (self.nnue2score * self.weight_scale_out)

    self.weight_clipping = [
      {'params' : [self.input.v], 'min_weight' : -max_hidden_weight, 'max_weight' : max_hidden_weight },
      {'params' : [self.layer_stacks.l1.weight], 'min_weight' : -max_hidden_weight, 'max_weight' : max_hidden_weight, 'virtual_params' : self.layer_stacks.l1_fact.weight },
      {'params' : [self.layer_stacks.fm_diff.weight], 'min_weight' : -max_hidden_weight, 'max_weight' : max_hidden_weight },
      {'params' : [self.layer_stacks.fm_abs.weight], 'min_weight' : -max_hidden_weight, 'max_weight' : max_hidden_weight },
      {'params' : [self.layer_stacks.cross_proj.weight], 'min_weight' : -max_hidden_weight, 'max_weight' : max_hidden_weight },
      {'params' : [self.layer_stacks.q_proj.weight], 'min_weight' : -max_hidden_weight, 'max_weight' : max_hidden_weight },
      {'params' : [self.layer_stacks.k_proj.weight], 'min_weight' : -max_hidden_weight, 'max_weight' : max_hidden_weight },
      {'params' : [self.layer_stacks.v_proj.weight], 'min_weight' : -max_hidden_weight, 'max_weight' : max_hidden_weight },
      {'params' : [self.layer_stacks.phase_proj.weight], 'min_weight' : -max_hidden_weight, 'max_weight' : max_hidden_weight },
      {'params' : [self.layer_stacks.l2.weight], 'min_weight' : -max_hidden_weight, 'max_weight' : max_hidden_weight },
      {'params' : [self.layer_stacks.output.weight], 'min_weight' : -max_out_weight, 'max_weight' : max_out_weight },
    ]

    self._zero_virtual_feature_weights()

  '''
  We zero all virtual feature weights because during serialization to .nnue
  we compute weights for each real feature as being the sum of the weights for
  the real feature in question and the virtual features it can be factored to.
  This means that if we didn't initialize the virtual feature weights to zero
  we would end up with the real features having effectively unexpected values
  at initialization - following the bell curve based on how many factors there are.
  '''
  def _zero_virtual_feature_weights(self):
    weights = self.input.weight
    v_weights = self.input.v
    with torch.no_grad():
      for a, b in self.feature_set.get_virtual_feature_ranges():
        # メインの線形パスの重みをゼロに
        weights[a:b, :] = 0.0
        # 因子ベクトル (FM項) の重みもゼロに
        v_weights[a:b, :] = 0.0

    self.input.weight = nn.Parameter(weights)
    self.input.v = nn.Parameter(v_weights)


  '''
  Clips the weights of the model based on the min/max values allowed
  by the quantization scheme.
  '''
  def _clip_weights(self):
    for group in self.weight_clipping:
      for p in group['params']:
        if 'min_weight' in group or 'max_weight' in group:
          p_data_fp32 = p.data
          min_weight = group['min_weight']
          max_weight = group['max_weight']
          if 'virtual_params' in group:
            virtual_params = group['virtual_params']
            xs = p_data_fp32.shape[0] // virtual_params.shape[0]
            ys = p_data_fp32.shape[1] // virtual_params.shape[1]
            expanded_virtual_layer = virtual_params.repeat(xs, ys)
            if min_weight is not None:
              min_weight_t = p_data_fp32.new_full(p_data_fp32.shape, min_weight) - expanded_virtual_layer
              p_data_fp32 = torch.max(p_data_fp32, min_weight_t)
            if max_weight is not None:
              max_weight_t = p_data_fp32.new_full(p_data_fp32.shape, max_weight) - expanded_virtual_layer
              p_data_fp32 = torch.min(p_data_fp32, max_weight_t)
          else:
            if min_weight is not None and max_weight is not None:
              p_data_fp32.clamp_(min_weight, max_weight)
            else:
              raise Exception('Not supported.')
          p.data.copy_(p_data_fp32)


  '''
  This method attempts to convert the model from using the self.feature_set
  to new_feature_set.
  '''
  def set_feature_set(self, new_feature_set):
    if self.feature_set.name == new_feature_set.name:
      return

    # TODO: Implement this for more complicated conversions.
    #       Currently we support only a single feature block.
    if len(self.feature_set.features) > 1:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

    # Currently we only support conversion for feature sets with
    # one feature block each so we'll dig the feature blocks directly
    # and forget about the set.
    old_feature_block = self.feature_set.features[0]
    new_feature_block = new_feature_set.features[0]

    # next(iter(new_feature_block.factors)) is the way to get the
    # first item in a OrderedDict. (the ordered dict being str : int
    # mapping of the factor name to its size).
    # It is our new_feature_factor_name.
    # For example old_feature_block.name == "HalfKP"
    # and new_feature_factor_name == "HalfKP^"
    # We assume here that the "^" denotes factorized feature block
    # and we would like feature block implementers to follow this convention.
    # So if our current feature_set matches the first factor in the new_feature_set
    # we only have to add the virtual feature on top of the already existing real ones.
    if old_feature_block.name == next(iter(new_feature_block.factors)):
      # We can just extend with zeros since it's unfactorized -> factorized
      weights = self.input.weight
      padding = weights.new_zeros((new_feature_block.num_virtual_features, weights.shape[1]))
      weights = torch.cat([weights, padding], dim=0)
      self.input.weight = nn.Parameter(weights)
      self.feature_set = new_feature_set
    else:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

  def forward(self, us, them, white_indices, white_values, black_indices, black_values, layer_stack_indices):

    # --- PHASE 1: 入力埋め込みと特徴量分離 ---
    # 1. 蓄積計算用ベクトル(t)とFM用埋め込み(v)を全入力から取得
    t_w, t_b, v_w, v_b = self.input(white_indices, white_values, black_indices, black_values)

    # 2. 特徴量IDの境界定義 (HalfKA/KSDG系 と Factorized系 を分ける)
    H_START = 12672
    H_END   = 214210 # KSDG3_Fact の開始点までが H系

    def process_fm(v_feat, indices):
        # 特徴量IDに基づいてマスクを作成 (H系 vs それ以外)
        mask_h = (indices >= H_START) & (indices < H_END)
        mask_k = (indices < H_START) | (indices >= H_END)

        # 有効な特徴量のみを抽出
        vh = v_feat * mask_h.unsqueeze(-1)
        vk = v_feat * mask_k.unsqueeze(-1)

        # 合計(Sum)と交互作用(Interaction)を計算
        sh, sk = torch.sum(vh, dim=1), torch.sum(vk, dim=1)
        ih = 0.5 * (sh**2 - torch.sum(vh**2, dim=1))
        ik = 0.5 * (sk**2 - torch.sum(vk**2, dim=1))
        return ih, ik, sh, sk

    # 自軍・敵軍それぞれのFM要素 [ih, ik, sh, sk] を抽出
    ih_w, ik_w, sh_w, sk_w = process_fm(v_w, white_indices)
    ih_b, ik_b, sh_b, sk_b = process_fm(v_b, black_indices)


    # --- PHASE 2: Main Path (蓄積計算) の処理 ---
    # 1. 視点(us/them)を考慮して1280次元(640ペア)を生成し、Clippingを適用
    l0_raw = (us * torch.cat([t_w, t_b], dim=1)) + (them * torch.cat([t_b, t_w], dim=1))
    l0_clipped = torch.clamp(l0_raw, 0.0, 1.0)
    l0_s = torch.split(l0_clipped, 640, dim=1)
    
    # 2. フェーズ進行度 (0.0=序盤 ～ 1.0=終盤) に基づくテント関数の計算
    # w0: 0.0でピーク, w1: 0.33, w2: 0.66, w3: 1.0
    pf = layer_stack_indices.view(-1, 1, 1).float() / 11.0
    w0 = torch.clamp(1.0 - pf * 3.0, min=0.0)
    w1 = torch.clamp(1.0 - torch.abs(pf * 3.0 - 1.0), min=0.0)
    w2 = torch.clamp(1.0 - torch.abs(pf * 3.0 - 2.0), min=0.0)
    w3 = torch.clamp(pf * 3.0 - 2.0, min=0.0)

    # 3. ペアごとの演算比率(Mul/Diff/Sum)をフェーズでブレンド
    mixed_weights = (w0 * self.pair_weights[0] + 
                     w1 * self.pair_weights[1] + 
                     w2 * self.pair_weights[2] + 
                     w3 * self.pair_weights[3])
    weights = torch.softmax(mixed_weights, dim=2)

    # 4. メインパスの計算
    # 前半部分
    term_mul = l0_s[0] * l0_s[1]
    term_diff_sq = torch.pow(l0_s[0] - l0_s[1], 2)
    term_sum = (l0_s[0] + l0_s[1]) * 0.5

    # 配合比率を適用
    l1_main_part1 = (weights[:, :, 0] * term_mul + 
                     weights[:, :, 1] * term_diff_sq + 
                     weights[:, :, 2] * term_sum)

    # 後半部分
    term_mul2 = l0_s[2] * l0_s[3]
    term_diff_sq2 = torch.pow(l0_s[2] - l0_s[3], 2)
    term_sum2 = (l0_s[2] + l0_s[3]) * 0.5

    l1_main_part2 = (weights[:, :, 0] * term_mul2 + 
                     weights[:, :, 1] * term_diff_sq2 + 
                     weights[:, :, 2] * term_sum2)

    # 前半と後半をcat
    l1_main_input = torch.cat([l1_main_part1, l1_main_part2], dim=1) * (127/128)


    # --- PHASE 3: FM項 (Diff / Abs) のスケーリング ---
    # 1. FM成分を統合し、視点を考慮した「差(Diff)」と「和(Abs)」を生成
    v_w_all = torch.stack([ih_w, ik_w, sh_w, sk_w], dim=1)
    v_b_all = torch.stack([ih_b, ik_b, sh_b, sk_b], dim=1)
    us_3d = us.view(-1, 1, 1)
    them_3d = them.view(-1, 1, 1)

    raw_diff = (us_3d * (v_w_all - v_b_all)) + (them_3d * (v_b_all - v_w_all))
    raw_abs  = (us_3d * v_w_all) + (them_3d * v_b_all)

    # 2. 成分別正規化
    # [Inter_H, Inter_K, Sum_H, Sum_K] の順でスケールを適用
    norm_diff = torch.tensor([0.01, 0.01, 0.05, 0.05], device=raw_diff.device).view(1, 4, 1)
    norm_abs  = torch.tensor([0.004, 0.004, 0.02, 0.02], device=raw_abs.device).view(1, 4, 1)

    diff_input_scaled = torch.clamp(raw_diff * norm_diff + 0.5, 0.0, 1.0)
    abs_input_scaled  = torch.clamp(raw_abs * norm_abs + 0.5, 0.0, 1.0)

    # 最終的に LayerStacks に渡すために 128次元に flatten
    diff_input = diff_input_scaled.view(-1, 128)
    abs_input  = abs_input_scaled.view(-1, 128)


    # --- PHASE 4: LayerStacks による深層処理 ---
    # 構築した3つのパス (Main:1280, Diff:128, Abs:128) を後続層へ
    final_output, l3_out, l1_main_bp, l2_input, diff_gated, abs_gated, gate_d, gate_a, channel_stats = self.layer_stacks(
        l1_main_input, diff_input, abs_input, layer_stack_indices
    )


    # --- 統計ログの呼び出し ---
    if self.training:
        self._log_detailed_stats(
            us, white_indices, l1_main_input,
            diff_input, abs_input,
            diff_gated, abs_gated,
            layer_stack_indices, final_output, l3_out, l1_main_bp, l2_input
            , gate_d, gate_a, channel_stats
        )

    return final_output


  def _log_detailed_stats(self, us, white_indices, l1_main, 
                         diff_l1, abs_l1,
                         diff_gated, abs_gated,
                         layer_stack_indices, final_output, l3_out, l1_main_bp, l2_input
                         , gate_d, gate_a, channel_stats):

    # --- 実行頻度制御 (500ステップに1回) ---
    if not hasattr(self, 'dbg_cnt'): self.dbg_cnt = 0
    self.dbg_cnt += 1
    if self.dbg_cnt % 500 != 0: return

    with torch.no_grad():
      print(f"\n[FM Detailed Debug Step {self.dbg_cnt}]")

      # --- SECTION 1: 基本統計 (特徴量のアクティブ数と入力信号の分布) ---
      _b_size = us.shape[0]

      # 0以上の要素のみをカウント
      _num_active_total = torch.sum(white_indices >= 0).item()
      _num_act_avg = _num_active_total / _b_size
      _unique_active = torch.unique(white_indices[white_indices >= 0]).numel()

      print(f" Features   | ActiveAvg:{_num_act_avg:5.1f}, Unique:{_unique_active}")
      print(f" LayerStacks Entry Analysis")
      print(f"   MainPath | mean:{l1_main.mean():7.4f}, std:{l1_main.std():7.4f}, min:{l1_main.min():7.4f}, max:{l1_main.max():7.4f}")
      print(f"   FM Diff  | mean:{diff_l1.mean():7.4f}, std:{diff_l1.std():7.4f}, min:{diff_l1.min():7.4f}, max:{diff_l1.max():7.4f}")
      print(f"   FM Abs   | mean:{abs_l1.mean():7.4f}, std:{abs_l1.std():7.4f}, min:{abs_l1.min():7.4f}, max:{abs_l1.max():7.4f}")


      # --- SECTION 2: 出力構成分析 (DeepPath vs MainBypass の寄与比率) ---
      l3_mag = l3_out.abs().mean().item()
      m_bp_mag = l1_main_bp.abs().mean().item()

      total_mag = l3_mag + m_bp_mag + 1e-9
      l3_ratio = (l3_mag / total_mag) * 100
      bp_ratio = (m_bp_mag / total_mag) * 100

      print(f" Output Composition | L3(Deep): {l3_mag:7.4f} ({l3_ratio:4.1f}%) | MainBypass: {m_bp_mag:7.4f} ({bp_ratio:4.1f}%)")
      print(f" Final Score Range  | Min: {final_output.min():7.2f}, Max: {final_output.max():7.2f}, Mean: {final_output.mean():7.2f}")


      # --- SECTION 3: Blend Alpha の可視化 (DeepPath をどの程度採用しているか) ---
      alphas = torch.sigmoid(self.layer_stacks.blend).cpu().numpy()
      alpha_pct = alphas * 100

      print(f" Blend Alpha (DeepPath Ratio):")
      print(f"   Mean : {alpha_pct.mean():.2f}%")
      print(f"   Min  : {alpha_pct.min():.2f}%")
      print(f"   Max  : {alpha_pct.max():.2f}%")


      # --- SECTION 4: FM 4成分の詳細分析 (Interaction/Sum × H/K 系統) ---
      print("-" * 110)
      print(f"{'FM Component':19} | {'Raw min / mean  / max    / std':33} | {'Zero%':>7} | {'High%':>7}")
      print("-" * 110)

      # 飽和率計算用ヘルパー
      def get_sat_zero(t):
          return (t <= 0.01).float().mean().item() * 100
      def get_sat_high(t):
          return (t >= 0.98).float().mean().item() * 100

      # L1段階 (128次元) を成分ごとに分解
      r_d_4ch = diff_l1.view(-1, 4, 32)
      r_a_4ch = abs_l1.view(-1, 4, 32)
      
      names = ["Inter HalfKA", "Inter KSDG3", "SumV HalfKA", "SumV KSDG3"]

      for i, name in enumerate(names):
          # --- Diff成分 ---
          d_comp = r_d_4ch[:, i, :]
          d_min, d_m, d_max, d_std = d_comp.min(), d_comp.mean(), d_comp.max(), d_comp.std()
          sat_zero_l1_d = get_sat_zero(d_comp)
          sat_high_l1_d = get_sat_high(d_comp)
          print(f" {name+'(Diff)':18} | {d_min:6.3f} / {d_m:6.3f} / {d_max:6.3f} / {d_std:6.3f} | {sat_zero_l1_d:6.2f}% | {sat_high_l1_d:6.2f}%")
          
          # --- Abs成分 ---
          a_comp = r_a_4ch[:, i, :]
          a_min, a_m, a_max, a_std = a_comp.min(), a_comp.mean(), a_comp.max(), a_comp.std()
          sat_zero_l1_a = get_sat_zero(a_comp)
          sat_high_l1_a = get_sat_high(a_comp)
          print(f" {name+'(Abs)':18} | {a_min:6.3f} / {a_m:6.3f} / {a_max:6.3f} / {a_std:6.3f} | {sat_zero_l1_a:6.2f}% | {sat_high_l1_a:6.2f}%")


      # --- SECTION 5: ゲート適用直後の詳細分析 (スライス後の raw 信号強度) ---
      print("-" * 115)
      print(f"{'Component':38} | {'Min':>6} / {'Mean':>6} / {'Max':>6} / {'Std':>6} ")
      print("-" * 115)

      # --- Diff Gated (RMSNorm直後) ---
      d_comp = diff_gated.detach()
      print(f" {'FM Diff (RMSNormed, before scaling)':37} | {d_comp.min():6.2f} / {d_comp.mean():6.2f} / {d_comp.max():6.2f} / {d_comp.std():6.2f}")

      # --- Abs Gated (Gate適用後) ---
      a_comp = abs_gated.detach()
      print(f" {'FM Abs  (Gated, before scaling)':37} | {a_comp.min():6.2f} / {a_comp.mean():6.2f} / {a_comp.max():6.2f} / {a_comp.std():6.2f}")


      # --- SECTION 6: L2入力信号のセクション別強度 (どのパスが支配的か) ---
      print("-" * 110)
      print(f"[Signal Strength (L2 Input)]")
      main_sqr_part = l2_input[:, 0:31].abs().mean().item()
      main_raw_part = l2_input[:, 31:62].abs().mean().item()
      fm_diff_part  = l2_input[:, 62:94].abs().mean().item()
      fm_abs_part   = l2_input[:, 94:158].abs().mean().item()
      cross_feat    = l2_input[:, 158:190].abs().mean().item()

      print(f" L2 In | Main(Sqr): {main_sqr_part:.4f} | Main(Raw): {main_raw_part:.4f} | FM(Diff): {fm_diff_part:.4f} | FM(Abs): {fm_abs_part:.4f}  | cross_feat: {cross_feat:.4f}")


      # --- SECTION 7: L2入力信号の統計詳細テーブル ---
      print("-" * 110)
      print(f"{'Section':12} | {'Mean':>7} | {'AbsMean':>7} | {'Std':>7} | {'Max':>7} | {'Min':>7} | {'Zero%':>7} | {'High%':>7}")
      print("-" * 110)
      
      def log_stats(name, tensor):
          t = tensor.detach().float()
          t_abs = t.abs()
          sparsity = (t_abs < 0.01).float().mean().item() * 100
          high_signal = (t > 0.95).float().mean().item() * 100 # 1.0に近いものをカウント
          print(f"{name:12} | {t.mean():7.3f} | {t_abs.mean():7.3f} | {t.std():7.3f} | {t.max():7.2f} | {t.min():7.2f} | {sparsity:6.1f}% | {high_signal:6.1f}%")
      
      # l2_inputの構造に基づいた切り出し
      log_stats("Main(Sqr)",    l2_input[:, 0:31])
      log_stats("Main(Raw)",    l2_input[:, 31:62])
      log_stats("FM(Diff)",     l2_input[:, 62:94])
      log_stats("FM(Abs_Raw)",  l2_input[:, 94:126])
      log_stats("FM(Abs_Sqr)",  l2_input[:, 126:158])
      log_stats("cross_feat",   l2_input[:, 158:190])


      # --- SECTION 8: 勾配統計 (G_Ratio: FM係数とメイン重みの学習バランス) ---
      if self.input.v.grad is not None:
          # 1要素あたりの平均絶対勾配
          v_grad_mean = self.input.v.grad.abs().mean().item()
          m_grad_mean = self.input.weight.grad.abs().mean().item()
          g_ratio = v_grad_mean / (m_grad_mean + 1e-9)
          print(f"[Gradient] MeanAbs_V:{v_grad_mean:8.2e}, MeanAbs_M:{m_grad_mean:8.2e}, G_Ratio:{g_ratio:8.4f}")


      # --- SECTION 9: パス別の重みノルム (正則化やスケーリングの確認) ---
      print(f"[Weights]  Diff_W:{self.layer_stacks.fm_diff.weight.norm():6.2f}, "
            f"Abs_W:{self.layer_stacks.fm_abs.weight.norm():6.2f}, "
            f"L2_W:{self.layer_stacks.l2.weight.norm():6.2f}")


      # --- SECTION 10: 各レイヤー・パーツ別の詳細重み/勾配統計 ---
      w_input_c = self.input.weight.detach().cpu()
      g_input_c = self.input.weight.grad.detach().cpu() if self.input.weight.grad is not None else None
      b_input_c = self.input.bias.detach().cpu()

      v_input_c = self.input.v.detach().cpu()
      vg_input_c = self.input.v.grad.detach().cpu() if self.input.v.grad is not None else None

      pw_w = self.pair_weights.detach().cpu()
      pw_g = self.pair_weights.grad.detach().cpu() if self.pair_weights.grad is not None else None
      
      # 実際の混合比率を Softmax で計算 [640, 3]
      # 0:積, 1:差, 2:和
      pw_softmax = torch.softmax(pw_w, dim=2) 

      # 特徴量分割点 (現在の構成に合わせて調整してください)
      S0, S1, S2 = 0, 12672, 203670


      # --- [ADD] L1_Main と L1_Fact の合成重み・勾配の計算 ---
      l1_main_w = self.layer_stacks.l1.weight.detach().cpu()
      l1_fact_w = self.layer_stacks.l1_fact.weight.detach().cpu()
      
      l1_main_g = self.layer_stacks.l1.weight.grad.detach().cpu() if self.layer_stacks.l1.weight.grad is not None else None
      l1_fact_g = self.layer_stacks.l1_fact.weight.grad.detach().cpu() if self.layer_stacks.l1_fact.weight.grad is not None else None

      # 先に安全にテンソルをCPUに取得する
      l1_main_b = self.layer_stacks.l1.bias.detach().cpu() if self.layer_stacks.l1.bias is not None else None
      l1_fact_b = self.layer_stacks.l1_fact.bias.detach().cpu() if self.layer_stacks.l1_fact.bias is not None else None
      
      # クリッピング時と同じルールで拡張して足し算（重み）
      xs = l1_main_w.shape[0] // l1_fact_w.shape[0]
      ys = l1_main_w.shape[1] // l1_fact_w.shape[1]
      expanded_fact_w = l1_fact_w.repeat(xs, ys)
      l1_combined_w = l1_main_w + expanded_fact_w

      # 勾配も同様に合成
      if l1_main_g is not None and l1_fact_g is not None:
          l1_combined_g = l1_main_g + l1_fact_g.repeat(xs, ys)
      else:
          l1_combined_g = l1_main_g if l1_main_g is not None else None

      # バイアスも加算（取得した変数を使ってリピート拡張）
      if l1_main_b is not None and l1_fact_b is not None:
          bias_xs = l1_main_b.shape[0] // l1_fact_b.shape[0]
          expanded_fact_b = l1_fact_b.repeat(bias_xs)
          l1_combined_b = l1_main_b + expanded_fact_b
      else:
          l1_combined_b = l1_main_b if l1_main_b is not None else l1_fact_b


      # --- Layer Parts List ---
      parts = [
          ("W_input (All)   ", w_input_c, g_input_c, b_input_c),
          ("W_KSDG3 (Part)  ", w_input_c[S0:S1, :], g_input_c[S0:S1, :] if g_input_c is not None else None, b_input_c),
          ("W_HalfKA (Part) ", w_input_c[S1:S2, :], g_input_c[S1:S2, :] if g_input_c is not None else None, b_input_c),
          ("V_Factor (FM)   ", v_input_c, vg_input_c, None),
          ("Pair_W (Raw)      ", pw_w, pw_g, None),
          ("L1_Main (Linear)", self.layer_stacks.l1.weight.detach().cpu(), self.layer_stacks.l1.weight.grad.detach().cpu() if self.layer_stacks.l1.weight.grad is not None else None, self.layer_stacks.l1.bias.detach().cpu()),
          ("L1_Fact         ", self.layer_stacks.l1_fact.weight.detach().cpu(), self.layer_stacks.l1_fact.weight.grad.detach().cpu() if self.layer_stacks.l1_fact.weight.grad is not None else None, self.layer_stacks.l1_fact.bias.detach().cpu()),

          # 【ADD】合成後のレイヤー統計
          ("L1_Combined (Sum)", l1_combined_w, l1_combined_g, l1_combined_b),

          ("FM_Diff_Path    ", self.layer_stacks.fm_diff.weight.detach().cpu(), self.layer_stacks.fm_diff.weight.grad.detach().cpu() if self.layer_stacks.fm_diff.weight.grad is not None else None, self.layer_stacks.fm_diff.bias.detach().cpu()),
          ("FM_Abs_Path     ", self.layer_stacks.fm_abs.weight.detach().cpu(), self.layer_stacks.fm_abs.weight.grad.detach().cpu() if self.layer_stacks.fm_abs.weight.grad is not None else None, self.layer_stacks.fm_abs.bias.detach().cpu()),
          ("cross_proj ", self.layer_stacks.cross_proj.weight.detach().cpu(), self.layer_stacks.cross_proj.weight.grad.detach().cpu() if self.layer_stacks.cross_proj.weight.grad is not None else None, self.layer_stacks.cross_proj.bias.detach().cpu()),
          ("q_proj ", self.layer_stacks.q_proj.weight.detach().cpu(), self.layer_stacks.q_proj.weight.grad.detach().cpu() if self.layer_stacks.q_proj.weight.grad is not None else None, self.layer_stacks.q_proj.bias.detach().cpu()),
          ("k_proj ", self.layer_stacks.k_proj.weight.detach().cpu(), self.layer_stacks.k_proj.weight.grad.detach().cpu() if self.layer_stacks.k_proj.weight.grad is not None else None, self.layer_stacks.k_proj.bias.detach().cpu()),
          ("v_proj ", self.layer_stacks.v_proj.weight.detach().cpu(), self.layer_stacks.v_proj.weight.grad.detach().cpu() if self.layer_stacks.v_proj.weight.grad is not None else None, self.layer_stacks.v_proj.bias.detach().cpu()),
          ("phase_proj ", self.layer_stacks.phase_proj.weight.detach().cpu(), self.layer_stacks.phase_proj.weight.grad.detach().cpu() if self.layer_stacks.phase_proj.weight.grad is not None else None, self.layer_stacks.phase_proj.bias.detach().cpu()),
          ("L2_Weight (Sum) ", self.layer_stacks.l2.weight.detach().cpu(), self.layer_stacks.l2.weight.grad.detach().cpu() if self.layer_stacks.l2.weight.grad is not None else None, self.layer_stacks.l2.bias.detach().cpu()),
          ("Output_Weight   ", self.layer_stacks.output.weight.detach().cpu(), self.layer_stacks.output.weight.grad.detach().cpu() if self.layer_stacks.output.weight.grad is not None else None, self.layer_stacks.output.bias.detach().cpu())
      ]

      # 各フェーズごとの選ばれ方を統計に追加
      phase_names = ["Open", "Mid1", "Mid2", "End "]
      for p in range(4):
          parts.append((f"P_{phase_names[p]}_Mul   ", pw_softmax[p, :, 0], pw_g[p, :, 0], None))
          parts.append((f"P_{phase_names[p]}_Diff  ", pw_softmax[p, :, 1], pw_g[p, :, 1], None))
          parts.append((f"P_{phase_names[p]}_Sum   ", pw_softmax[p, :, 2], pw_g[p, :, 2], None))

      print("-" * 120)
      print(f"{'Layer Name':<18} | {'Grad Mean':<12} {'Active':<8} | {'W_Mean':<8} {'W_Min':<8} {'W_Max':<9} {'W_Std':<7} | {'B_Mean':<8} {'B_Min':<8} {'B_Max':<9} {'B_Std':<7}")
      print("-" * 120)

      for name, w, g, b in parts:
          # 勾配の統計 (要素数によるスケーリング)
          gm = (g.norm().item() / (g.numel()**0.5 + 1e-9)) if g is not None else 0.0
          ga = (g != 0).sum().item() if g is not None else 0

          # 重みの統計
          wm, wmin, wmax, ws = w.mean().item(), w.min().item(), w.max().item(), w.std().item()

          # バイアスの統計
          if b is not None and b.numel() > 0:
              bm, bmin, bmax = b.mean().item(), b.min().item(), b.max().item()
              bs = b.std().item() if b.numel() > 1 else 0.0
          else:
              bm, bmin, bmax, bs = 0.0, 0.0, 0.0, 0.0

          print(f"{name:<18} | {gm:12.10f} {ga:<8} | {wm:+8.5f} {wmin:+8.5f} {wmax:+8.5f} {ws:8.5f} | {bm:+8.5f} {bmin:+8.5f} {bmax:+8.5f} {bs:8.5f}")
      print("-" * 120)


      # --- SECTION 11: Attention 状態 (LCA: Lightweight Cross-Attention) ---
      att = self.layer_stacks.last_att_score
      att_mean, att_min, att_max, att_std = att.mean().item(), att.min().item(), att.max().item(), att.std().item()

      print(f"[Attention Status] dynamic_scale (FM-Filter)")
      print(f"  Mean: {att_mean:.4f} | Min: {att_min:.4f} | Max: {att_max:.4f} | Std: {att_std:.4f}")
      # どの程度「極端に」絞っているかの分布（例：0.2以下、0.8以上）
      low_rate = (att < 0.2).float().mean().item() * 100
      high_rate = (att > 0.8).float().mean().item() * 100
      print(f"  Distribution: Low(<0.2): {low_rate:.1f}% | High(>0.8): {high_rate:.1f}%")
      print(f"  Current Temp (T) : {self.layer_stacks.last_lca_temp.item():.4f}")


      # --- SECTION 12: LCA Meta-Learning (動的な温度パラメータの学習方向) ---
      if hasattr(self.layer_stacks, 'lca_temp'):
          temp_param = self.layer_stacks.lca_temp
          t_val = temp_param.item()
          t_grad = temp_param.grad.item() if temp_param.grad is not None else 0.0
          
          # 勾配の向きの解釈: 
          # T = T - lr * grad なので
          # Grad > 0 => Tは減少する方向 => 判断を鋭く(0/1)したい
          # Grad < 0 => Tは増加する方向 => 判断をマイルド(0.5)にしたい
          direction = "Sharper(0/1) ↓" if t_grad > 0 else "Milder(0.5) ↑"
          print(f"[LCA Meta-Learning]")
          print(f"  Current Temp (T) : {t_val:.4f}")
          print(f"  Temp Grad        : {t_grad:+.2e} [{direction}]")


      # --- SECTION 13: 演算ブレンド戦略の詳細統計 (Mul/Diff/Sum の分布) ---
      def get_simple_stats(t):
          return {
              'avg': t.mean().item(),
              'std': t.std().item(),
              'min': t.min().item(),
              'max': t.max().item()
          }

      s_mul  = get_simple_stats(pw_softmax[:, :, 0])
      s_diff = get_simple_stats(pw_softmax[:, :, 1])
      s_sum  = get_simple_stats(pw_softmax[:, :, 2])

      print(f"[Blend Strategy Detailed]")
      print(f"  - Mul  | Avg: {s_mul['avg']:.1%} | Std: {s_mul['std']:.3f} | Range: [{s_mul['min']:.1%} - {s_mul['max']:.1%}]")
      print(f"  - Diff | Avg: {s_diff['avg']:.1%} | Std: {s_diff['std']:.3f} | Range: [{s_diff['min']:.1%} - {s_diff['max']:.1%}]")
      print(f"  - Sum  | Avg: {s_sum['avg']:.1%} | Std: {s_sum['std']:.3f} | Range: [{s_sum['min']:.1%} - {s_sum['max']:.1%}]")

      # 各フェーズごとの選ばれ方を統計に追加
      phase_names = ["Open", "Mid1", "Mid2", "End "]
      for p in range(4):
          avg_m = pw_softmax[p, :, 0].mean().item()
          avg_d = pw_softmax[p, :, 1].mean().item()
          avg_s = pw_softmax[p, :, 2].mean().item()
          print(f"  Phase {phase_names[p]} Mix Ratio -> Mul: {avg_m:.3f}, Diff: {avg_d:.3f}, Sum: {avg_s:.3f}")


      # --- SECTION 14: バケット別詳細統計 (Gate vs Value 分離) ---
      def get_layer_stats(w, g, b):
          # w, g, b は特定のバケットの特定の32次元スライス
          wm, wmin, wmax, ws = w.mean().item(), w.min().item(), w.max().item(), w.std().item()
          gm = (g.norm().item() / (g.numel()**0.5 + 1e-9)) if g is not None else 0.0
          ga = (g != 0).sum().item() if g is not None else 0
          bm = b.mean().item() if b is not None else 0
          return gm, ga, wm, wmin, wmax, ws, bm

      print("-" * 115)
      print(f"{'Layer (Bucket)':<22} | {'Grad Mean':<12} {'Active':<8} | {'W_Mean':<8} {'W_Min':<8} {'W_Max':<9} {'W_Std':<7} | {'B_Mean':<8}")
      print("-" * 115)
      
      # 対象バケット
      target_buckets = [0, 11]
      
      for b_idx in target_buckets:
          # 64次元周期 (Gate: 0-31, Val: 32-63)
          base = b_idx * 64

          # fm_diff と fm_abs をループ
          for name, layer in [("FM_Diff", self.layer_stacks.fm_diff), ("FM_Abs", self.layer_stacks.fm_abs)]:
              w_all = layer.weight.detach().cpu()
              g_all = layer.weight.grad.detach().cpu() if layer.weight.grad is not None else None
              b_all = layer.bias.detach().cpu()

              # Gate (前半32次元)
              p_gate = get_layer_stats(w_all[base : base+32], 
                                 g_all[base : base+32] if g_all is not None else None,
                                 b_all[base : base+32])
              # Val (後半32次元)
              p_val = get_layer_stats(w_all[base+32 : base+64], 
                                g_all[base+32 : base+64] if g_all is not None else None,
                                b_all[base+32 : base+64])

              # 表示
              print(f"{name+'_Gate(B'+str(b_idx)+')':<22} | {p_gate[0]:12.10f} {p_gate[1]:<8} | {p_gate[2]:+8.5f} {p_gate[3]:+8.5f} {p_gate[4]:+8.5f} {p_gate[5]:8.5f} | {p_gate[6]:+8.5f}")
              print(f"{name+'_Val (B'+str(b_idx)+')':<22} | {p_val[0]:12.10f} {p_val[1]:<8} | {p_val[2]:+8.5f} {p_val[3]:+8.5f} {p_val[4]:+8.5f} {p_val[5]:8.5f} | {p_val[6]:+8.5f}")

          if b_idx == 0: print("-" * 115) # バケット間の区切り
      print("-" * 115)


      # --- SECTION 15: Inter-Gating Status (実効的な門戸開放率) ---
      # 1. AbsPath への門戸開放率 (実効値)
      open_to_abs = torch.sigmoid(gate_a).mean().item() * 100.0
      
      # 2. MainPath への門戸開放率 (実効値： 0.5 + 0.5 * sigmoid)
      eff_sigmoid_d = 0.5 + 0.5 * torch.sigmoid(gate_d)
      open_to_main = eff_sigmoid_d.mean().item() * 100.0

      # 3. ゲートのsharpness
      sharpness_d = torch.sigmoid(gate_d).var().item()
      sharpness_a = torch.sigmoid(gate_a).var().item()

      print(f"--- Inter-Gating Status (Effective) ---")
      print(f"Abs (Filtered by Abs-Gate) Open: {open_to_abs:.2f}% (sharp:{sharpness_a:.3f})")
      print(f"Main (Filtered by Diff-Gate) Open: {open_to_main:.2f}% (sharp:{sharpness_d:.3f})")


      # --- SECTION 16: 6-Channel Phase Gate (適応制御の状態) ---
      print(f"--- 6-Channel Phase Gate Status (Adaptive Control) ---")
      print(f"{'Name':<8} | {'Mean':<5} | {'Std':<5} | {'Range':<11} | {'Low%':<5} | {'High%':<5}")
      print("-" * 62)

      for s in channel_stats:
          name = s['name']
          mean = s['mean']
          std  = s['std']
          r_min = s['min']
          r_max = s['max']
          low  = s['low']
          high = s['high']

          print(f"{name:<8} | {mean:.3f} | {std:.3f} | [{r_min:.2f}-{r_max:.2f}] | {low:>4.1f}% | {high:>5.1f}%")


      # --- SECTION 17: 個別バケット詳細ログの呼び出し ---
      self._log_bucket_stats(l3_out, l1_main_bp, diff_gated, layer_stack_indices, gate_d, gate_a)


  def _log_bucket_stats(self, l3_out, l1_main_bp, diff_gated, layer_stack_indices, gate_d, gate_a):
    with torch.no_grad():
      # --- SECTION 1: バケット分布と最終出力の計算 ---
      batch_size = layer_stack_indices.size(0)
      bucket_counts = torch.bincount(layer_stack_indices, minlength=12)
      bucket_ratios = (bucket_counts.float() / batch_size) * 100.0

      # 最終評価値の算出 (DeepPath + MainBypass)
      final_output = l3_out + l1_main_bp
      # センチポーン(cp)に変換 (self.nnue2score = 600.0)
      final_cp = final_output * self.nnue2score

      # 全バケットの Blend Alpha (DeepPath採用率) を取得
      all_alphas_pct = torch.sigmoid(self.layer_stacks.blend).cpu().numpy() * 100

      print(f"[Bucket-wise FM Value & Gate Analysis]")
      print("-" * 110)
      header = f"{'B_ID':<4} | Samples% | {'Eval(cp)':<9} | {'L1_Main':<8} | {'FM_Diff_V':<12} | {'FM_Abs_V':<12} | {'AbsOpen(GatebyAbs)%':<10} | {'MainOpen(GatebyDiff)%':<10} | {'L2_Layer':<8} | {'L3(Deep)%':<8} | {'Blend(Alpha)%':<10}"
      print(header)
      print("-" * 110)

      # 重みと勾配の事前取得 (統計計算用)
      w1 = self.layer_stacks.l1.weight.detach().cpu()
      wd = self.layer_stacks.fm_diff.weight.detach().cpu()
      wa = self.layer_stacks.fm_abs.weight.detach().cpu()

      # 勾配がある場合は取得、なければゼロ埋め
      gd_grad = self.layer_stacks.fm_diff.weight.grad.detach().cpu() if self.layer_stacks.fm_diff.weight.grad is not None else torch.zeros_like(wd)
      ga_grad = self.layer_stacks.fm_abs.weight.grad.detach().cpu() if self.layer_stacks.fm_abs.weight.grad is not None else torch.zeros_like(wa)

      w2 = self.layer_stacks.l2.weight.detach().cpu()


      # --- SECTION 2: バケットごとの詳細ループ ---
      for i in range(self.layer_stacks.count):
        mask = (layer_stack_indices == i)
        if not mask.any():
            continue

        # 出現頻度
        s_ratio = bucket_ratios[i].item()

        # インデックス計算 (L1:32次元周期, FM:64次元周期(GLU), L2:96次元周期)
        s1, e1 = i * 32, (i + 1) * 32
        sf, ef = i * 64, (i + 1) * 64
        s2, e2 = i * 96, (i + 1) * 96

        # FM Value側の重み(w)と勾配(g)の統計 (GLUの後半32次元がValueに相当)
        md_w = wd[sf+32:ef].abs().mean().item()
        md_g = gd_grad[sf+32:ef].abs().mean().item()

        ma_w = wa[sf+32:ef].abs().mean().item()
        ma_g = ga_grad[sf+32:ef].abs().mean().item()

        b_mask = (layer_stack_indices == i)

        if b_mask.sum() > 0:
            # このバケットに属する局面の平均 Eval
            avg_cp = final_cp[b_mask].mean().item()
            abs_cp = final_cp[b_mask].abs().mean().item()

            # ゲートの開放率 (実際にどの程度の信号を通しているか)
            # AbsGate (Absを通す率): sigmoid(gate_a)
            open_abs = torch.sigmoid(gate_a[b_mask]).mean().item() * 100
            # DiffGate (Mainを通す率): 0.5 + 0.5 * sigmoid(gate_d)
            open_main = (0.5 + 0.5 * torch.sigmoid(gate_d[b_mask])).mean().item() * 100

            # DeepPath(L3)が最終出力に占める寄与度
            fm_r = (l3_out[b_mask].abs().mean().item() / 
                   (l3_out[b_mask].abs().mean().item() + l1_main_bp[b_mask].abs().mean().item() + 1e-9)) * 100
        else:
            avg_cp, abs_cp, open_abs, open_main, fm_r = 0.0, 0.0, 0.0

        # このバケットのブレンド係数
        current_alpha = all_alphas_pct[i]

        # 出力
        print(f"B{i:02d} | {s_ratio:5.1f}% | {avg_cp:+6.1f}({abs_cp:6.1f}) | {w1[s1:e1].abs().mean():.3f} | {md_w:.3f}|{md_g:.1e} | {ma_w:.3f}|{ma_g:.1e} | {open_abs:5.1f}% | {open_main:5.1f}% | {w2[s2:e2].abs().mean():.3f}   | {fm_r:5.1f}% | {current_alpha:5.1f}%")
      print("-" * 110)


      # --- SECTION 3: Bucket-wise Phase Detail (6チャンネル適応制御) ---
      print("\n[Bucket-wise Phase Gate Analysis (6-Channel)]")
      print("-" * 110)
      print(f"{'B_ID':<4} | {'MSqr':<5} | {'MRaw':<5} | {'Diff':<5} | {'AbsR':<5} | {'AbsS':<5} | {'Cross':<5} | {'Low%':<5} | {'High%':<6} | {'Samples%':<8} | {'AttScore(mean/std)':<18} | {'Loss':<8} ")
      print("-" * 110)

      # バッチ全体のサンプル数を取得
      total_samples = layer_stack_indices.size(0)

      for i in range(12):
          mask = (layer_stack_indices == i)
          count = mask.sum().item()
          if count == 0: continue

          # このバケットに属する局面の 6ch ゲート平均値
          p_batch = self.layer_stacks.last_phase[mask]
          p_means = p_batch.mean(dim=0) # [6]

          # 飽和統計 (どれくらい 0/1 に張り付いているか)
          low_r  = (p_batch < 0.2).float().mean().item() * 100
          high_r = (p_batch > 0.8).float().mean().item() * 100
          s_ratio = (count / total_samples) * 100

          # 6つの信号 (MainSqr, MainRaw, Diff, AbsRaw, AbsSqr, Cross) の平均値
          m_sq, m_ra, f_di, f_ar, f_as, crs = p_means.tolist()

          # LCA (Cross-Attention) スコアの統計
          att = self.layer_stacks.last_att_score[mask]

          # Loss
          if torch.is_tensor(self.last_bucket_losses):
              loss_val = self.last_bucket_losses[i].cpu().item() # ループ内で個別に取る場合
          else:
              loss_val = self.last_bucket_losses[i] # すでにリスト化されている場合

          print(f"B{i:02d}  | {m_sq:.3f} | {m_ra:.3f} | {f_di:.3f} | {f_ar:.3f} | {f_as:.3f} | {crs:.3f} | {low_r:>4.1f}% | {high_r:>5.1f}% | {s_ratio:>7.1f}% | {att.mean().item():.3f} / {att.std().item():.3f} | {loss_val:>7.5f} ")
      print("-" * 110)


  def step_(self, batch, batch_idx, loss_type):
      # --- SECTION 1: データ展開とパラメータの準備 ---
      self._clip_weights()
      us, them, white_indices, white_values, black_indices, black_values, outcome, score, layer_stack_indices, material, kif_group_id = batch

      # --- SECTION 2: Lambda の決定 ---
      actual_lambda = self._get_actual_lambda(loss_type)

      # --- SECTION 3: ネットワーク出力の計算と勝率変換 ---
      scorenet = self(us, them, white_indices, white_values, black_indices, black_values, layer_stack_indices)
      scorenet = scorenet * self.nnue2score

      # シグモイド関数を用いて勝率(0.0~1.0)に変換
      q  = ( scorenet - self.offset1) / self.in_scaling
      qm = (-scorenet - self.offset2) / self.in_scaling
      qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())

      # 教師スコア(Search Score)も勝率空間へ変換
      p  = ( score - self.offset1) / self.out_scaling
      pm = (-score - self.offset2) / self.out_scaling
      pf = 0.5 * (1.0 + p.sigmoid() - pm.sigmoid())

      # ターゲット作成 (教師勝率 + 実際の対局結果)
      pt = pf * actual_lambda + outcome * (1.0 - actual_lambda)

      # --- SECTION 4: 各種損失の計算 ---
      kif_group_id_flat = kif_group_id.view(-1)

      # 1. Base Loss (ID: 1, 2 対象)
      base_loss = self._compute_base_loss(pt, qf, pf, kif_group_id_flat)

      # Pairwise / Listwise ソート用の共通データ準備 (ID: 3 対象)
      pairwise_mask = (kif_group_id_flat == 3)
      n_pairwise = pairwise_mask.sum().item()

      sorted_data = self._prepare_sorted_data(
          pairwise_mask, n_pairwise, pt, qf, score, scorenet, layer_stack_indices, material
      )

      # 2. Pairwise Loss (ID: 3 対象)
      pairwise_loss, pair_metrics = self._compute_pairwise_loss(sorted_data, n_pairwise, pt.device)

      # 3. Listwise Loss (ID: 3 対象)
      listwise_loss, pt_range = self._compute_listwise_loss(sorted_data, n_pairwise, pt.device)

      # --- SECTION 5: 最終損失の結合 ---
      pairwise_weight = 0.010
      listwise_weight = 0.030
      loss = base_loss + (pairwise_weight * pairwise_loss) + (listwise_weight * listwise_loss)

      # 4. Phase Gate (適応制御) 正則化ペナルティの加算
      phase_penalty = self._compute_phase_penalty()
      # 必要に応じて適用
      loss = loss + 0.002 * phase_penalty

      # --- SECTION 6: [検証用] バケット別損失の統計集計 ---
      self._update_bucket_stats(pt, qf, layer_stack_indices, loss_type)

      # --- SECTION 7: ログ出力・デバッグ ---
      self._log_debug_info(
          loss_type, loss, base_loss, pairwise_loss, listwise_loss,
          pt, qf, score, scorenet, layer_stack_indices, kif_group_id_flat, actual_lambda,
          pair_metrics, pt_range
      )

      return loss

  # =========================================================================
  #  Helper Methods
  # =========================================================================

  def _get_actual_lambda(self, loss_type):
      """Lambda の動的／固定設定を取得"""
      lambda_dict = {
          'val_loss_lambda1.0': 1.0,
          'val_loss_lambda0.0': 0.0,
          'val_loss_lambda0.1': 0.1,
          'val_loss_lambda0.5': 0.5,
          'val_loss_lambda0.8': 0.8,
      }
      if loss_type in lambda_dict:
          return lambda_dict[loss_type]
      return self.start_lambda + (self.end_lambda - self.start_lambda) * (self.current_epoch / self.max_epoch)

  def _compute_base_loss(self, pt, qf, pf, kif_group_id_flat):
      """ベースとなる点推定損失の計算 (kif_group_id == 1, 2)"""
      loss_elements = torch.pow(torch.abs(pt - qf), 2.5)
      loss_elements = loss_elements * ((qf > pt) * self.adjust_loss + 1)

      weights = 1 + (2.0**1.2 - 1) * torch.pow((pf - 0.5) ** 2 * pf * (1 - pf), 0.8)
      weights_flat = weights.view(-1)

      # Base Loss は 1 と 2 のみ
      base_loss_mask = (kif_group_id_flat == 1) | (kif_group_id_flat == 2)

      if base_loss_mask.any():
          return (loss_elements.view(-1)[base_loss_mask] * weights_flat[base_loss_mask]).sum() / weights_flat[base_loss_mask].sum()
      return torch.tensor(0.0, device=pt.device)

  def _prepare_sorted_data(self, pairwise_mask, n_pairwise, pt, qf, score, scorenet, layer_stack_indices, material):
      """Pairwise/Listwise計算用にデータを抽出し、ソートして返す (kif_group_id == 3 対象)"""
      if n_pairwise <= 1:
          return None

      pt_p = pt.view(-1)[pairwise_mask]
      qf_p = qf.view(-1)[pairwise_mask]
      score_p = score.view(-1)[pairwise_mask]
      scorenet_p = scorenet.view(-1)[pairwise_mask]
      lsind_p = layer_stack_indices.view(-1)[pairwise_mask]
      material_p = material.view(-1)[pairwise_mask]

      if self.training and self.global_step % 500 == 0:
          print(f"\n[DEBUG RANGE] Step: {self.global_step}")
          print(f"  [material_p] Min: {material_p.min().item():.1f} | Max: {material_p.max().item():.1f} | Mean: {material_p.mean().item():.1f}")
          print(f"  [pt_p]       Min: {pt_p.min().item():.4f} | Max: {pt_p.max().item():.4f}")

      # ソートキーは「material, pt」
      sort_key = material_p + pt_p
      sorted_indices = torch.argsort(sort_key)

      return {
          'pt': pt_p[sorted_indices],
          'qf': qf_p[sorted_indices],
          'score': score_p[sorted_indices],
          'scorenet': scorenet_p[sorted_indices],
          'lsind': lsind_p[sorted_indices],
          'material': material_p[sorted_indices],
      }

  def _compute_pairwise_loss(self, sorted_data, n_pairwise, device):
      """ミニバッチ内擬似ペアワイズ損失の計算"""
      default_metrics = {
          'total_valid_pairs': 0, 'all_pred_diffs': [], 'all_target_directions': [],
          'all_value_gaps': [], 'all_num_equals': 0, 'all_valid_lsinds': [], 'all_diff_abs': [],
          'max_possible_pairs': 0
      }

      window_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 50, 100, 200, 300, 500, 1000, 2000]

      if sorted_data is None:
          return torch.tensor(0.0, device=device), default_metrics

      pt_s, qf_s = sorted_data['pt'], sorted_data['qf']
      score_s, scorenet_s = sorted_data['score'], sorted_data['scorenet']
      lsind_s, mat_s = sorted_data['lsind'], sorted_data['material']

      total_pairwise_loss = torch.tensor(0.0, device=device)
      total_valid_pairs = 0
      all_pred_diffs, all_target_directions, all_value_gaps = [], [], []
      all_valid_lsinds, all_diff_abs = [], []
      all_num_equals = 0

      for w in window_sizes:
          if w >= n_pairwise:
              continue

          pt_A, pt_B = pt_s[:-w], pt_s[w:]
          qf_A, qf_B = qf_s[:-w], qf_s[w:]
          cp_true_A, cp_true_B = score_s[:-w], score_s[w:]
          cp_pred_A, cp_pred_B = scorenet_s[:-w], scorenet_s[w:]
          lsind_true_A = lsind_s[:-w]
          material_A, material_B = mat_s[:-w], mat_s[w:]

          diff_abs = (pt_A - pt_B).abs()

          tol = torch.where(lsind_true_A < 4, 400, torch.where(lsind_true_A < 8, 200, 100))
          valid_pair_mask = (diff_abs > 0.003) & (diff_abs <= 0.10) & ((material_A - material_B).abs() <= tol)

          if valid_pair_mask.any():
              target_direction = torch.sign(pt_A - pt_B)
              num_equal = (target_direction[valid_pair_mask] == 0).sum().item()

              pred_diff = qf_A - qf_B
              pair_weight_curve = torch.sigmoid((diff_abs - 0.005) * 150) * torch.sigmoid((0.05 - diff_abs) * 120)
              scale = 2.0 + 3.0 * torch.exp(-(diff_abs / 0.02)**2)

              # 1. [方向の損失]
              direction_loss = -F.logsigmoid(target_direction * pred_diff * scale)

              # 2. [値の損失]
              true_cp_compressed = torch.tanh((cp_true_A - cp_true_B) / 400.0)
              pred_cp_compressed = torch.tanh((cp_pred_A - cp_pred_B) / 400.0)
              value_loss_cp = F.smooth_l1_loss(pred_cp_compressed, true_cp_compressed, reduction="none", beta=0.1)

              # 3. [ハイブリッド結合]
              raw_pairwise = (direction_loss + 0.6 * value_loss_cp) * pair_weight_curve
              equal_penalty = pred_diff.pow(2) * 1.0
              pairwise_loss_all = torch.where(target_direction != 0, raw_pairwise, equal_penalty)

              # 集計
              total_pairwise_loss += pairwise_loss_all[valid_pair_mask].sum()
              valid_cnt = valid_pair_mask.sum().item()
              total_valid_pairs += valid_cnt

              # 評価メトリクス用
              all_pred_diffs.append(pred_diff[valid_pair_mask].detach())
              all_target_directions.append(target_direction[valid_pair_mask].detach())
              all_num_equals += num_equal
              all_value_gaps.append(((cp_pred_A - cp_pred_B) - (cp_true_A - cp_true_B)).abs()[valid_pair_mask].detach())
              all_valid_lsinds.append(lsind_true_A[valid_pair_mask].detach())
              all_diff_abs.append(diff_abs[valid_pair_mask].detach())

      pairwise_loss = (total_pairwise_loss / total_valid_pairs) if total_valid_pairs > 0 else torch.tensor(0.0, device=device)

      metrics = {
          'total_valid_pairs': total_valid_pairs,
          'all_pred_diffs': all_pred_diffs,
          'all_target_directions': all_target_directions,
          'all_value_gaps': all_value_gaps,
          'all_num_equals': all_num_equals,
          'all_valid_lsinds': all_valid_lsinds,
          'all_diff_abs': all_diff_abs,
          'max_possible_pairs': sum([max(0, n_pairwise - w) for w in window_sizes])
      }

      return pairwise_loss, metrics

  def _compute_listwise_loss(self, sorted_data, n_pairwise, device):
      """マルチウィンドウ Listwise 損失の計算"""
      if sorted_data is None:
          return torch.tensor(0.0, device=device), None

      list_size = 6
      temperature = 0.15
      stride_sizes = [1, 2, 3, 4, 5, 6, 7, 8]

      pt_s, qf_s = sorted_data['pt'], sorted_data['qf']
      mat_s, lsind_s = sorted_data['material'], sorted_data['lsind']

      total_listwise_loss = torch.tensor(0.0, device=device)
      total_valid_groups = 0
      all_listwise_ranges = []

      for stride in stride_sizes:
          block_size = list_size * stride
          if block_size > n_pairwise:
              continue

          pt_blocks = pt_s.unfold(0, block_size, 1)
          qf_blocks = qf_s.unfold(0, block_size, 1)
          mat_blocks = mat_s.unfold(0, block_size, 1)
          lsind_blocks = lsind_s.unfold(0, block_size, 1)

          pt_lists = pt_blocks[:, ::stride]
          qf_lists = qf_blocks[:, ::stride]
          mat_lists = mat_blocks[:, ::stride]
          lsind_lists = lsind_blocks[:, ::stride]

          # フィルタリング判定
          lsind_ref = lsind_lists[:, 0]
          tol = torch.where(lsind_ref < 4, 400, torch.where(lsind_ref < 8, 200, 100))
          material_mask = (mat_lists.max(dim=1).values - mat_lists.min(dim=1).values) <= tol

          pt_range_current = pt_lists.max(dim=1).values - pt_lists.min(dim=1).values
          range_mask = (pt_range_current > 0.003) & (pt_range_current <= 0.35)

          valid_group_mask = material_mask & range_mask

          if valid_group_mask.any():
              pt_filtered = pt_lists[valid_group_mask]
              qf_filtered = qf_lists[valid_group_mask]

              all_listwise_ranges.append(pt_range_current[valid_group_mask].detach())

              true_dist = torch.softmax(pt_filtered / temperature, dim=-1)
              pred_log_dist = torch.log_softmax(qf_filtered / temperature, dim=-1)

              total_listwise_loss += F.kl_div(pred_log_dist, true_dist, reduction="sum")
              total_valid_groups += valid_group_mask.sum().item()

      if total_valid_groups > 0:
          listwise_loss = total_listwise_loss / total_valid_groups
          pt_range = torch.cat(all_listwise_ranges, dim=0)
          return listwise_loss, pt_range

      return torch.tensor(0.0, device=device), None

  def _compute_phase_penalty(self):
      """Phase Gate (適応制御) への正則化ペナルティの計算"""
      phase = getattr(self.layer_stacks, 'current_phase_for_loss', None)
      if phase is None:
          return 0.0

      p_means = phase.mean(dim=0)
      p_stds  = phase.std(dim=0)

      # 1. 全体平均を 0.5 に近づける制約
      overall_mean = p_means.mean() 
      mean_penalty = (overall_mean - 0.5) ** 2

      # 2. 境界値ペナルティ: 各チャンネルの平均が極端（0.05未満 or 0.95超）になるのを防ぐ
      mean_bounds_penalty = torch.mean(
          torch.clamp(p_means - 0.95, min=0) ** 2
          + torch.clamp(0.05 - p_means, min=0) ** 2
      )

      # 3. 分散ペナルティ: 局面に応じて「開閉」の変化（多様性）を促す
      std_penalty = torch.mean(torch.clamp(0.15 - p_stds, min=0) ** 2)

      return mean_penalty + mean_bounds_penalty + std_penalty

  def _update_bucket_stats(self, pt, qf, layer_stack_indices, loss_type):
      """検証フェーズにおけるバケット(Bucket)別損失の統計更新"""
      with torch.no_grad():
          loss_per_sample = torch.pow(torch.abs(pt - qf), 2.5).detach().squeeze()

          if not self.training and loss_type == 'val_loss_actual_lambda':
              if not hasattr(self, 'bucket_stats'):
                  self.bucket_stats = {
                      'loss_sum': torch.zeros(12, device=pt.device),
                      'count': torch.zeros(12, device=pt.device)
                  }

              indices = layer_stack_indices.squeeze()
              self.bucket_stats['loss_sum'].index_add_(0, indices, loss_per_sample)
              self.bucket_stats['count'].index_add_(0, indices, torch.ones_like(loss_per_sample))

  def _log_debug_info(self, loss_type, loss, base_loss, pairwise_loss, listwise_loss,
                      pt, qf, score, scorenet, layer_stack_indices, kif_group_id_flat, actual_lambda,
                      pair_metrics, pt_range):
      """コンソール出力および TensorBoard/Lightning ログ記録"""
      mean_base = base_loss.item()
      mean_pair = pairwise_loss.item()
      mean_list = listwise_loss.item()
      mean_total = loss.item()

      if self.training:
          self.log("train/base_loss", mean_base, prog_bar=False)
          self.log("train/pairwise_loss", mean_pair, prog_bar=False)
          self.log("train/listwise_loss", mean_list, prog_bar=False)

      # 500ステップ毎のコンソール詳細出力
      if self.training and (self.global_step % 500 == 0):
          valid_count = pair_metrics['total_valid_pairs']
          max_possible = pair_metrics['max_possible_pairs']

          print(f"\n[DEBUG LOSS] Total: {mean_total:.6f} | Base: {mean_base:.6f} | Pairwise: {mean_pair:.6f} | Listwise: {mean_list:.6f} (Valid Pairs: {valid_count}/{max_possible})")
          print(f"  [PT Distribution] Min: {pt.min().item():.4f} | Median: {pt.median().item():.4f} | Max: {pt.max().item():.4f} | Std: {pt.std().item():.4f}")

          with torch.no_grad():
              counts = torch.bincount(kif_group_id_flat, minlength=4)
              print(f"  [Kif Group Counts] ID_1: {counts[1].item()} | ID_2: {counts[2].item()} | ID_3: {counts[3].item()} (Batch Total: {pt.size(0)})")

          if valid_count > 0 and len(pair_metrics['all_pred_diffs']) > 0:
              flat_pred_diff = torch.cat(pair_metrics['all_pred_diffs'], dim=0)
              flat_targets = torch.cat(pair_metrics['all_target_directions'], dim=0)
              flat_diff_abs = torch.cat(pair_metrics['all_diff_abs'], dim=0)

              print(f"  equal={pair_metrics['all_num_equals'] / valid_count:.3%}")
              pd_abs_mean = flat_pred_diff.abs().mean().item()
              print(f"  pred_diff_abs_mean={pd_abs_mean:.6f}")

              pair_acc = ((flat_targets * flat_pred_diff) > 0).float().mean()
              print(f"  pair_acc={pair_acc.item():.4f}")

              # 範囲別集計
              bins = [(0.000, 0.002), (0.002, 0.005), (0.005, 0.010), (0.010, 0.020), (0.020, 0.030),
                      (0.030, 0.050), (0.050, 0.070), (0.070, 0.100), (0.100, 0.150)]
              
              print(f"  [Pairwise Detail per Range]")
              for start, end in bins:
                  label = f"{start*100:.1f}%-{end*100:.1f}%"
                  mask = (flat_diff_abs >= start) & (flat_diff_abs < end)
                  count = mask.sum().item()

                  if count > 0:
                      sub_t = flat_targets[mask]
                      sub_pd = flat_pred_diff[mask]
                      acc = ((sub_t * sub_pd) > 0).float().mean().item()
                      s_mean = (sub_t * sub_pd).mean().item()
                      pd_abs_m = sub_pd.abs().mean().item()
                      print(f"    {label:<11}: {count:>6}t | acc={acc:.4f} | signed_m={s_mean:+.5f} | pred_diff_abs_m={pd_abs_m:.6f}")

                      if self.training:
                          self.log(f"pair_acc_range/{label}", acc, prog_bar=False)
                          self.log(f"pair_signed_mean_range/{label}", s_mean, prog_bar=False)
                          self.log(f"pair_pred_diff_abs_range/{label}", pd_abs_m, prog_bar=False)
                  else:
                      print(f"    {label:<11}:      0t | acc=0.0000 | signed_m=+0.00000 | pred_diff_abs_m=0.000000")

              signed_mean = (flat_targets * flat_pred_diff).mean().item()
              print(f"  signed_mean={signed_mean:.5f}")

              flat_gaps = torch.cat(pair_metrics['all_value_gaps'], dim=0)
              value_gap_mean = flat_gaps.mean().item()
              print(f"  value_diff_cp_mae={value_gap_mean:.3f} cp")

              if len(pair_metrics['all_valid_lsinds']) > 0:
                  with torch.no_grad():
                      flat_lsinds = torch.cat(pair_metrics['all_valid_lsinds'], dim=0).long()
                      lsind_counts = torch.bincount(flat_lsinds, minlength=12)
                      counts_str = " | ".join([f"L{i}: {lsind_counts[i].item()}" for i in range(12)])
                      print(f"  [Pairwise Valid Pairs per Layer] {counts_str}")

              if self.training:
                  self.log("train/pair_acc", pair_acc, prog_bar=False)
                  self.log("train/pred_diff_abs_mean", pd_abs_mean, prog_bar=False)
                  self.log("train/value_diff_mae", value_gap_mean, prog_bar=False)

          if pt_range is not None and pt_range.numel() > 0:
              print(
                  f"[Listwise Range] (Total Active Groups: {pt_range.numel()})\n"
                  f"  Mean: {pt_range.mean():.4f} | Std: {pt_range.std():.4f} | Median: {pt_range.median():.4f} | Min: {pt_range.min():.4f} | Max: {pt_range.max():.4f}\n"
                  f"  [Histogram]\n"
                  f"    0.00-0.01 : {torch.sum((pt_range >= 0.00) & (pt_range < 0.01)).item():5d}t\n"
                  f"    0.01-0.02 : {torch.sum((pt_range >= 0.01) & (pt_range < 0.02)).item():5d}t\n"
                  f"    0.02-0.05 : {torch.sum((pt_range >= 0.02) & (pt_range < 0.05)).item():5d}t\n"
                  f"    0.05-0.10 : {torch.sum((pt_range >= 0.05) & (pt_range < 0.10)).item():5d}t\n"
                  f"    0.10-0.20 : {torch.sum((pt_range >= 0.10) & (pt_range < 0.20)).item():5d}t\n"
                  f"    0.20-0.30 : {torch.sum((pt_range >= 0.20) & (pt_range < 0.30)).item():5d}t\n"
                  f"    >0.30     : {torch.sum(pt_range >= 0.30).item():5d}t"
              )

      # 100ステップ毎の TensorBoard ヒストグラム出力
      if self.training and (self.global_step % 100 == 0) and hasattr(self, "logger") and self.logger is not None:
          self.logger.experiment.add_histogram("Distribution/PT_Target", pt.view(-1), global_step=self.global_step)
          self.logger.experiment.add_histogram("Distribution/QF_Prediction", qf.view(-1), global_step=self.global_step)
          self.logger.experiment.add_histogram("Distribution/score_Target", score.view(-1), global_step=self.global_step)
          self.logger.experiment.add_histogram("Distribution/scorenet", scorenet.view(-1), global_step=self.global_step)

          score_flat = score.view(-1)
          scorenet_flat = scorenet.view(-1)
          for i in range(12):
              layer_mask = (layer_stack_indices.view(-1) == i)
              if layer_mask.any():
                  self.logger.experiment.add_histogram(f"Distribution_Layer/L{i:02d}_score_Target", score_flat[layer_mask], global_step=self.global_step)
                  self.logger.experiment.add_histogram(f"Distribution_Layer/L{i:02d}_scorenet", scorenet_flat[layer_mask], global_step=self.global_step)

      # 検証時（Val）のログ出力
      if loss_type == 'val_loss_actual_lambda' and self.global_step > 1:
          self.log('actual_lambda', actual_lambda)
          self.log("val_loss/base_loss", mean_base, prog_bar=False)
          self.log("val_loss/pairwise_loss", mean_pair, prog_bar=False)
          self.log("val_loss/listwise_loss", mean_list, prog_bar=False)

          if pair_metrics['total_valid_pairs'] > 0 and len(pair_metrics['all_pred_diffs']) > 0:
              flat_pd = torch.cat(pair_metrics['all_pred_diffs'], dim=0)
              flat_t = torch.cat(pair_metrics['all_target_directions'], dim=0)
              val_pair_acc = ((flat_t * flat_pd) > 0).float().mean()
              val_value_gap_mean = torch.cat(pair_metrics['all_value_gaps'], dim=0).mean().item()

              self.log("val_loss/pair_acc", val_pair_acc, prog_bar=False)
              self.log("val_loss/value_diff_cp_mae", val_value_gap_mean, prog_bar=False)
          else:
              self.log("val_loss/pair_acc", 0.0, prog_bar=False)
              self.log("val_loss/value_diff_cp_mae", 0.0, prog_bar=False)

      self.log(loss_type, loss)

  # =========================================================================


  def training_step(self, batch, batch_idx):
    return self.step_(batch, batch_idx, 'train_loss')

  def validation_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'val_loss_actual_lambda')
    self.step_(batch, batch_idx, 'val_loss_lambda1.0')
    self.step_(batch, batch_idx, 'val_loss_lambda0.0')
    self.step_(batch, batch_idx, 'val_loss_lambda0.1')
    self.step_(batch, batch_idx, 'val_loss_lambda0.5')
    self.step_(batch, batch_idx, 'val_loss_lambda0.8')

    # 学習率のトラッキング
    optimizer = self.optimizers()
    current_lr = optimizer.param_groups[0]['lr']
    self.log('current_lr', current_lr, on_step=False, on_epoch=True)

  def test_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'test_loss')

  def on_validation_epoch_end(self):
      """検証エポック終了時：バケットごとの損失統計を確定させる"""
      if hasattr(self, 'bucket_stats'):
          s = self.bucket_stats['loss_sum']
          c = self.bucket_stats['count']

          # --- 1. 平均損失の算出 (ゼロ除算を考慮) ---
          # 各バケットの累積損失をサンプル数で割り、デバッグ表示用のプロパティへ格納
          avg_losses = (s / (c + 1e-9)).detach().cpu().numpy()
          self.last_bucket_losses = avg_losses

          # --- 2. 統計のリセット ---
          # 次のエポックで新鮮な集計を行うためにゼロ埋め
          self.bucket_stats['loss_sum'].zero_()
          self.bucket_stats['count'].zero_()


  def on_train_batch_end(self, outputs, batch, batch_idx):
    """学習バッチ終了時：100ステップごとに内部状態(ゲートや重み)を可視化"""
    if self.global_step % 100 == 0:
        with torch.no_grad():
            # --- 1. ゲート制御信号の取得 ---
            # Forwardパスで保存された最新のゲート値 (Raw Logits)
            gate_d = self.layer_stacks.last_gate_d
            gate_a = self.layer_stacks.last_gate_a

            sig_d = torch.sigmoid(gate_d)
            sig_a = torch.sigmoid(gate_a)

            # --- 2. 実効的な開放率(Open Rate)の計算 ---
            # AbsPath: 0.0(全閉) ～ 1.0(全開) の範囲
            eff_open_abs = sig_a.mean() * 100.0
            
            # MainPath: 0.5(半開) ～ 1.0(全開) の範囲にスケーリング
            # ※構造上、Mainは最低でも50%の信号を流す仕様を反映
            eff_open_main = (0.5 + 0.5 * sig_d).mean() * 100.0

            # --- 3. 指標のログ記録 (Scalars) ---
            self.log("gate/open_rate_abs_to_abs", eff_open_abs)
            self.log("gate/open_rate_diff_to_main", eff_open_main)
            self.log("gate/sharpness_diff", sig_d.var())
            self.log("gate/sharpness_abs", sig_a.var())

            # --- 4. 分布の記録 (Histograms) ---
            tensorboard = self.logger.experiment
            # Absゲートの分布
            tensorboard.add_histogram("gate_dist/abs_to_abs_effective", sig_a, self.global_step)
            # Mainゲートの実効値(0.5~1.0)分布
            eff_sig_d = 0.5 + 0.5 * sig_d
            tensorboard.add_histogram("gate_dist/diff_to_main_effective", eff_sig_d, self.global_step)

            # --- 5. pair_weights(Mul/Diff/Sum) 演算ブレンド戦略の記録 ---
            pw = self.pair_weights.detach().cpu() # [4フェーズ, 640特徴量, 3演算]
            # 生の重み分布を確認
            tensorboard.add_histogram("pair_w/overall_raw", pw, self.global_step)
            # Softmaxによる実効比率の算出 (3演算の合計が1.0になるよう正規化)
            pw_softmax = torch.softmax(pw, dim=2) 

            # --- 6. フェーズ別(序・中1・中2・終)の戦略比率 ---
            phase_labels = ["Open", "Mid1", "Mid2", "End"]
            for i, label in enumerate(phase_labels):
                # 戦略ごとの比率ヒストグラム (フェーズごと)
                tensorboard.add_histogram(f"pair_w_ratio_{label}/mul",  pw_softmax[i, :, 0], self.global_step)
                tensorboard.add_histogram(f"pair_w_ratio_{label}/diff", pw_softmax[i, :, 1], self.global_step)
                tensorboard.add_histogram(f"pair_w_ratio_{label}/sum",  pw_softmax[i, :, 2], self.global_step)

                # スカラー値としての平均比率 (折れ線グラフで推移が見やすくなる)
                self.log(f"pair_w_avg_{label}/mul_ratio",  pw_softmax[i, :, 0].mean())
                self.log(f"pair_w_avg_{label}/diff_ratio", pw_softmax[i, :, 1].mean())
                self.log(f"pair_w_avg_{label}/sum_ratio",  pw_softmax[i, :, 2].mean())

            # 全フェーズを通じた総合的な演算傾向
            self.log("pair_w_avg_total/mul",  pw_softmax[:, :, 0].mean())
            self.log("pair_w_avg_total/diff", pw_softmax[:, :, 1].mean())
            self.log("pair_w_avg_total/sum",  pw_softmax[:, :, 2].mean())


  def configure_optimizers(self):
    LR = self.lr

    # --- SECTION 1: パラメータグループ別の学習戦略設定 ---
    # 各コンポーネントの役割に応じて LR倍率や Weight Decay を微調整
    train_params = [
      # 入力層 (Feature Transformer)
      {'params' : [self.input.weight], 'lr' : LR * 1.0, 'weight_decay': 0.0 }, 
      {'params' : [self.input.bias]  , 'lr' : LR * 1.0, 'weight_decay': 0.0 }, 

      # FM（Factorization Machine）因子ベクトル
      {'params' : [self.input.v]     , 'lr' : LR * 1.5, 'weight_decay': 0.0 }, 

      # pair_weights（mul/diff/sum）
      {'params' : [self.pair_weights], 'lr' : LR * 1.0, 'weight_decay': 1e-5 }, 

      # LayerStacks Main Path
      {'params' : [self.layer_stacks.l1.weight]      , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.l1.bias]        , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.l1_fact.weight] , 'lr' : LR * 0.5 , 'weight_decay': 1e-5 },
      {'params' : [self.layer_stacks.l1_fact.bias]   , 'lr' : LR * 0.5 , 'weight_decay': 1e-5 },

      # FM Diff Path
      {'params' : [self.layer_stacks.fm_diff.weight] , 'lr' : LR * 1.0, 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.fm_diff.bias]   , 'lr' : LR * 1.0, 'weight_decay': 0.0 },

      # FM Abs Path
      {'params' : [self.layer_stacks.fm_abs.weight]  , 'lr' : LR * 1.0, 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.fm_abs.bias]    , 'lr' : LR * 1.0, 'weight_decay': 0.0 },

      # L2 / Output Layer
      {'params' : [self.layer_stacks.l2.weight]      , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.l2.bias]        , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.output.weight]  , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.output.bias]    , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },

      # L3とバイパスのブレンド
      {'params' : [self.layer_stacks.blend]          , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },

      # cross_proj
      {'params' : [self.layer_stacks.cross_proj.weight]  , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.cross_proj.bias]    , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },

      # LCA
      {'params' : [self.layer_stacks.q_proj.weight]      , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.q_proj.bias]        , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.k_proj.weight]      , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.k_proj.bias]        , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.v_proj.weight]      , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.v_proj.bias]        , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.lca_temp]           , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },

      # phase_proj
      {'params' : [self.layer_stacks.phase_proj.weight]  , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.phase_proj.bias]    , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
    ]

    # Increasing the eps leads to less saturated nets with a few dead neurons.
    # Gradient localisation appears slightly harmful.
    #optimizer = ranger.Ranger(
    #  train_params, betas=(0.9, 0.999), eps=1.0e-7, gc_loc=False, use_gc=False
    #)
    """
    optimizer = ranger21.Ranger21(train_params,
      lr=1.0, betas=(.9, 0.999), eps=1.0e-7,
      using_gc=False, using_normgc=False,
      weight_decay=0.0,
      num_batches_per_epoch=int(self.epoch_size / self.batch_size), num_epochs=self.max_epoch,
      warmdown_active=False, use_warmup=False,
      use_adaptive_gradient_clipping=False,
      softplus=False,
      pnm_momentum_factor=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(
      optimizer, step_size=1, gamma=self.gamma
    )
    return [optimizer], [scheduler]
    """

    # --- SECTION 2: オプティマイザの構築 (AdamW 8-bit) ---
    # メモリ節約と学習速度向上のため 8-bit AdamW を採用。
    optimizer = bnb.optim.AdamW8bit(
        train_params,
        lr=LR,
        betas=(0.9, 0.995),
        eps=1e-7,
        weight_decay=1e-6,
        min_8bit_size=16384
    )

    # --- SECTION 3: スケジューラの設定 ---
    scheduler = {
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, threshold=1e-8),
        'monitor': 'val_loss_lambda1.0',
        'interval': 'epoch',
        'frequency': 1
    }
    return [optimizer], [scheduler]

  def get_layers(self, filt):
    """
    Returns a list of layers.
    filt: Return true to include the given layer.
    """
    for i in self.children():
      if filt(i):
        if isinstance(i, nn.Linear):
          for p in i.parameters():
            if p.requires_grad:
              yield p
