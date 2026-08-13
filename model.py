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

    # --- router層
    self.router = nn.Linear(384, count)

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
    self.q_proj = nn.Linear(31, 32) # Query
    self.k_proj = nn.Linear(64, 32) # Key
    self.v_proj = nn.Linear(64, 32) # Value

    # Temperature パラメータ (初期値 0.7 = やや鋭めからスタート)
    self.lca_temp = nn.Parameter(torch.tensor(0.7))

    #--- Phase Gate
    self.phase_proj = nn.Linear(384, 6)

    #--- lossへの加算用
    self.current_phase_for_loss = None

    #--- ログ・ルーティング用
    self.last_router_logits = None
    self.last_router_probs = None
    self.last_routing_weights = None
    self.last_routing_indices = None  # ルーターが実際に選んだインデックス

    self.last_gate_d = None
    self.last_gate_a = None
    self.last_att_score = None
    self.last_lca_temp = None
    self.last_phase = None

    self.idx_offset = None
    self._init_layers()

    self.step_counter = 0


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

      nn.init.normal_(self.router.weight, std=0.01)
      nn.init.constant_(self.router.bias, 0.0)

      for i in range(1, self.count):
        s1, e1 = i * 32, (i + 1) * 32
        self.l1.weight.data[s1:e1, :].copy_(self.l1.weight.data[0:32, :])
        self.l1.bias.data[s1:e1].copy_(self.l1.bias.data[0:32])
        
        sf, ef = i * 64, (i + 1) * 64
        self.fm_diff.weight.data[sf:ef, :].copy_(self.fm_diff.weight.data[0:64, :])
        self.fm_diff.bias.data[sf:ef].copy_(self.fm_diff.bias.data[0:64])
        self.fm_abs.weight.data[sf:ef, :].copy_(self.fm_abs.weight.data[0:64, :])
        self.fm_abs.bias.data[sf:ef].copy_(self.fm_abs.bias.data[0:64])
        
        self.cross_proj.weight.data[s1:e1, :].copy_(self.cross_proj.weight.data[0:32, :])
        self.cross_proj.bias.data[s1:e1].copy_(self.cross_proj.bias.data[0:32])

        self.blend.data[i] = self.blend.data[0]

        self.l2.weight.data[i*L3:(i+1)*L3, :].copy_(self.l2.weight.data[0:L3, :])
        self.l2.bias.data[i*L3:(i+1)*L3].copy_(self.l2.bias.data[0:L3])
        self.output.weight.data[i:i+1, :].copy_(self.output.weight.data[0:1, :])
        self.output.bias.data[i:i+1].copy_(self.output.bias.data[0:1])


  def forward(self, l1_main, diff_in, abs_in, ls_indices=None):
        if self.training:
            self.step_counter += 1

        # --- PHASE 0: Router による動的バケット選択 (384 -> 12) ---
        p_abs_base = torch.clamp(abs_in - 0.5, 0.0, 1.0) * 2.0   # [B, 128]
        main_sub = l1_main[:, :128]                               # [B, 128]

        router_input = torch.cat([p_abs_base, diff_in, main_sub], dim=1) # [B, 384]
        #router_input = l1_main[:, :384]

        router_logits = self.router(router_input)  # [B, count]

        # 1. 学習時（self.training == True）のみジッター（ノイズ）を追加
        if self.training:
            #jitter_std = 0.1
            # Router Logits の広がりに応じて動的にジッターを決定
            current_std = router_logits.std().detach().clamp(min=1e-3)
            jitter_std = current_std * 0.01

            noise = torch.randn_like(router_logits) * jitter_std
            logits_for_routing = router_logits + noise

            #-----
            raw_indices = router_logits.argmax(dim=-1)
            noisy_indices = logits_for_routing.argmax(dim=-1)

            # ノイズによって選択が変わった割合 (Flip Rate)
            if self.step_counter % 500 == 0:
                flip_rate = (raw_indices != noisy_indices).float().mean().item()
                print(f"[Router Check] Logits Mean: {router_logits.mean().item():.4f}, Std: {router_logits.std().item():.4f}")
                print(f"[Jitter Check] Flip Rate: {flip_rate:.1%}")
            #-----

        else:
            logits_for_routing = router_logits

        # 2. Straight-Through One-Hot による Hard Routing 化
        probs = F.softmax(logits_for_routing, dim=-1)
        router_indices = logits_for_routing.argmax(dim=-1)

        hard_one_hot = F.one_hot(router_indices, num_classes=self.count).float()
        routing_weights = (hard_one_hot - probs).detach() + probs

        self.last_routing_indices = router_indices.detach()


        # 3. [loss算出用、統計・ログ用] 
        self.last_router_logits = router_logits
        self.last_router_probs = F.softmax(router_logits, dim=-1)
        self.last_router_probs_log = self.last_router_probs.detach()
        self.last_router_weights = routing_weights.detach()

        # 4. PhaseGate 埋め込み用の bucket_info (0.0 ~ 1.0)
        bucket_scale = torch.linspace(0.0, 1.0, self.count, device=l1_main.device)
        bucket_info = (routing_weights * bucket_scale).sum(dim=-1, keepdim=True) # [B, 1]


        # --- PHASE 1: PhaseGate (適応的重み付け) の計算 ---
        p_abs_modified = p_abs_base.clone()
        p_abs_modified[:, 127:128] = bucket_info

        p_combined = torch.cat([p_abs_modified, diff_in], dim=1) 
        p_extra_combined = torch.cat([p_combined, main_sub], dim=1)

        phase_logit = (self.phase_proj(p_extra_combined) * 3.0) + 1.0
        phase = 0.1 + 0.9 * torch.sigmoid(phase_logit) # [B, 6]

        p_detached = phase.detach()
        phase_names = ["MainSqr", "MainRaw", "FM_Diff", "FM_AbsR", "FM_AbsS", "Cross"]
        channel_stats = []
        if self.training and (self.step_counter % 100 == 0):
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
            self.current_phase_for_loss = phase


        # --- PHASE 2: FMPath (全12バケット完全並列抽出) ---
        l1c_diff_all = self.fm_diff(diff_in).reshape(-1, self.count, 64) # [B, 12, 64]
        l1c_abs_all  = self.fm_abs(abs_in).reshape(-1, self.count, 64)   # [B, 12, 64]

        gate_d_all, val_d_all = l1c_diff_all.chunk(2, dim=-1) # 各 [B, 12, 32]
        gate_a_all, val_a_all = l1c_abs_all.chunk(2, dim=-1)  # 各 [B, 12, 32]

        rms_d_all = torch.rsqrt(val_d_all.pow(2).mean(dim=-1, keepdim=True) + 1e-8) # [B, 12, 1]
        val_d_normed_all = val_d_all * rms_d_all
        l1c_diff_gated_all = val_d_normed_all
        l1c_abs_gated_all  = val_a_all * torch.sigmoid(gate_a_all)

        l1_diff_l2_all = torch.clamp(l1c_diff_gated_all * 0.2 + 0.5, 0.0, 1.0) # [B, 12, 32]
        l1_abs_raw_all = torch.clamp(l1c_abs_gated_all * 0.05 + 0.6, 0.0, 1.0) # [B, 12, 32]
        l1_abs_sqr_all = l1_abs_raw_all.pow(2.0)                               # [B, 12, 32]


        # --- PHASE 3: MainPath & Attention 制御 (全12バケット完全独立) ---
        l1c_main_all = self.l1(l1_main).reshape(-1, self.count, 32) # [B, 12, 32]
        l1f_ = self.l1_fact(l1_main)                                # [B, 32]
        l1_combined_all = l1c_main_all + l1f_.unsqueeze(1)          # [B, 12, 32]

        gate_d_sub_all = gate_d_all - 0.3
        l1_combined_all = l1_combined_all * (0.5 + 0.5 * torch.sigmoid(gate_d_sub_all))

        l1_val_all, l1_main_bp_all = l1_combined_all.split([31, 1], dim=-1)

        l1_main_sqr_all = torch.clamp(l1_val_all, 0.0, 1.0).pow(2.0) * (127/128) # [B, 12, 31]
        l1_main_raw_all = torch.clamp(l1_val_all, 0.0, 1.0)                      # [B, 12, 31]

        q_all = self.q_proj(l1_main_raw_all)                                     # [B, 12, q_dim]
        fm_cat_all = torch.cat([l1_diff_l2_all, l1_abs_raw_all], dim=-1)         # [B, 12, 64]
        k_all = self.k_proj(fm_cat_all)
        v_all = self.v_proj(fm_cat_all)

        logit_all = (q_all * k_all).sum(dim=-1, keepdim=True) / 5.656            # [B, 12, 1]
        safe_temp = torch.clamp(self.lca_temp, min=0.125)
        att_score_all = torch.sigmoid(logit_all / safe_temp)

        v_clamped_all = torch.clamp(v_all * 0.4 + 0.5, 0.0, 1.0)
        l1_diff_l2_all = l1_diff_l2_all * (1 - att_score_all) + v_clamped_all * att_score_all # [B, 12, 32]


        # --- PHASE 4: CrossFeat (einsum による最適化) ---
        k_dim = self.cross_dim
        cross_diff_all = l1_main_sqr_all[:, :, :k_dim] * l1_diff_l2_all[:, :, :k_dim] # [B, 12, k_dim]
        cross_abs_all  = l1_main_raw_all[:, :, :k_dim] * l1_abs_raw_all[:, :, :k_dim] # [B, 12, k_dim]
        cross_cat_all  = torch.cat([cross_diff_all, cross_abs_all], dim=-1)           # [B, 12, 2*k_dim]

        # ★ einsum による12バケット個別の全結合演算
        W_cross = self.cross_proj.weight.view(self.count, 32, -1)  # [12, 32, 2*k_dim]
        b_cross = self.cross_proj.bias.view(self.count, 32)        # [12, 32]
        cross_feat_all = torch.einsum("bci,coi->bco", cross_cat_all, W_cross) + b_cross # [B, 12, 32]
        cross_feat_all = torch.clamp(cross_feat_all, 0.0, 1.0)


        # --- PHASE 5: L2 Input 構築 (全12バケット完全独立) ---
        p0 = phase[:, 0:1].unsqueeze(1) # [B, 1, 1]
        p1 = phase[:, 1:2].unsqueeze(1)
        p2 = phase[:, 2:3].unsqueeze(1)
        p3 = phase[:, 3:4].unsqueeze(1)
        p4 = phase[:, 4:5].unsqueeze(1)
        p5 = phase[:, 5:6].unsqueeze(1)

        l2_main_sqr_weighted_all = l1_main_sqr_all * (0.5 + p0 * 0.5) * 1.3
        l2_main_raw_weighted_all = l1_main_raw_all * (0.5 + p1 * 0.5) * 1.5
        l2_diff_weighted_all     = l1_diff_l2_all  * (0.5 + p2 * 0.5) * 1.0
        l1_abs_raw_weighted_all  = l1_abs_raw_all  * (0.5 + p3 * 0.5) * 0.7
        l1_abs_sqr_weighted_all  = l1_abs_sqr_all  * (0.5 + p4 * 0.5) * 0.88
        l2_cross_weighted_all    = cross_feat_all  * (0.5 + p5 * 0.5) * 1.5

        l2_padding_all = torch.zeros((l1_main.shape[0], self.count, 2), device=l1_main.device)
        l2_input_all = torch.cat([
            l2_main_sqr_weighted_all,
            l2_main_raw_weighted_all,
            l2_diff_weighted_all,   
            l1_abs_raw_weighted_all, 
            l1_abs_sqr_weighted_all, 
            l2_cross_weighted_all,   
            l2_padding_all            
        ], dim=-1) # Shape: [B, 12, L2_IN_DIM]

        l2_input_all = torch.clamp(l2_input_all, 0.0, 1.0)


        # --- PHASE 6: Output (einsum による L2 & Output の高速一括計算) ---
        # ★ einsum による L2層の計算 [B, 12, L2_IN_DIM] -> [B, 12, L3]
        W_l2 = self.l2.weight.view(self.count, L3, -1)   # [12, L3, L2_IN_DIM]
        b_l2 = self.l2.bias.view(self.count, L3)         # [12, L3]
        l2c_all = torch.einsum("bci,coi->bco", l2_input_all, W_l2) + b_l2
        l2x_all = torch.clamp(l2c_all, 0.0, 1.0)

        # ★ einsum による Output層の計算 [B, 12, L3] -> [B, 12, 1]
        W_out = self.output.weight.view(self.count, 1, -1) # [12, 1, L3]
        b_out = self.output.bias.view(self.count, 1)       # [12, 1]
        l3c_all = torch.einsum("bci,coi->bco", l2x_all, W_out) + b_out # [B, 12, 1]

        # 1. 各バケット個別の alpha (blend): Shape [1, count, 1]
        alpha_all = torch.sigmoid(self.blend).view(1, self.count, 1)

        # 2. ★ 真の Oracle 用出力（他バケットの混ざり物ゼロの12バケット個別の最終出力）
        # Shape: [B, count]
        all_final_outputs = (l3c_all * alpha_all + l1_main_bp_all * (1.0 - alpha_all)).squeeze(-1)

        # 3. Router の選択重み（routing_weights: [B, count]）を掛けて本命の出力を算出: Shape [B, 1]
        final_output = (all_final_outputs * routing_weights).sum(dim=-1, keepdim=True)


        # --- PHASE 7: ログ・デバッグ用変数の抽出（ルーティング選択された1本のストリーム） ---
        w_expand = routing_weights.unsqueeze(-1) # [B, 12, 1]

        l3c_selected         = (l3c_all.squeeze(-1) * routing_weights).sum(dim=-1, keepdim=True)
        l1_main_bp_selected  = (l1_main_bp_all.squeeze(-1) * routing_weights).sum(dim=-1, keepdim=True)
        l2_input_selected    = (l2_input_all * w_expand).sum(dim=1)
        l1c_diff_gated_sel   = (l1c_diff_gated_all * w_expand).sum(dim=1)
        l1c_abs_gated_sel    = (l1c_abs_gated_all * w_expand).sum(dim=1)
        gate_d_selected       = (gate_d_all * w_expand).sum(dim=1)
        gate_a_selected       = (gate_a_all * w_expand).sum(dim=1)

        if self.training:
            self.last_gate_d = gate_d_selected.detach().clone()
            self.last_gate_a = gate_a_selected.detach().clone()
            self.last_att_score = (att_score_all.squeeze(-1) * routing_weights).sum(dim=-1, keepdim=True).detach()
            self.last_lca_temp = safe_temp.detach()

        return (
            final_output,         # [B, 1] Router選択後のメイン出力
            l3c_selected,         # [B, 1] 選択された DeepPath 出力
            l1_main_bp_selected,  # [B, 1] 選択された Bypass 出力
            l2_input_selected,    # [B, L2_IN_DIM] 選択された L2入力
            l1c_diff_gated_sel,   # [B, 32]
            l1c_abs_gated_sel,    # [B, 32]
            gate_d_selected,      # [B, 32]
            gate_a_selected,      # [B, 32]
            channel_stats,        # PhaseGate 統計
            router_indices,       # [B] 選択インデックス
            router_logits,        # [B, count] Router CE Loss 用
            all_final_outputs     # [B, count] ★ 完全独立計算された真の Oracle 評価値
        )


  def get_coalesced_layer_stacks(self):
      for i in range(self.count):
          with torch.no_grad():
              l1 = nn.Linear(L1_MAIN, 32)
              diff_b = nn.Linear(128, 64)
              abs_b = nn.Linear(128, 64)
              l2 = nn.Linear(L2_IN_TOTAL, L3) 
              output = nn.Linear(L3, 1)
              cross_p = nn.Linear(self.cross_dim * 2, 32)

              lca_q = self.q_proj
              lca_k = self.k_proj
              lca_v = self.v_proj
              lca_temp_val = torch.clamp(self.lca_temp, min=0.125).item()
              phase_p = self.phase_proj

              l1.weight.data = self.l1.weight.data[i*32:(i+1)*32, :] + self.l1_fact.weight.data
              l1.bias.data = self.l1.bias.data[i*32:(i+1)*32] + self.l1_fact.bias.data

              diff_b.weight.data = self.fm_diff.weight.data[i*64:(i+1)*64, :]
              diff_b.bias.data = self.fm_diff.bias.data[i*64:(i+1)*64]
              abs_b.weight.data = self.fm_abs.weight.data[i*64:(i+1)*64, :]
              abs_b.bias.data = self.fm_abs.bias.data[i*64:(i+1)*64]

              s_c, e_c = i * 32, (i + 1) * 32
              cross_p.weight.data = self.cross_proj.weight.data[s_c:e_c, :]
              cross_p.bias.data = self.cross_proj.bias.data[s_c:e_c]

              l2.weight.data = self.l2.weight.data[i*L3:(i+1)*L3, :L2_IN_TOTAL]
              l2.bias.data = self.l2.bias.data[i*L3:(i+1)*L3]

              output.weight.data = self.output.weight.data[i:i+1, :]
              output.bias.data = self.output.bias.data[i:i+1]
              bucket_blend = self.blend.data[i].item()

              yield (l1, diff_b, abs_b, cross_p, l2, output
                     , bucket_blend, lca_q, lca_k, lca_v, lca_temp_val, phase_p)


class NNUE(pl.LightningModule):
  def __init__(self, feature_set, start_lambda=1.0, end_lambda=1.0, max_epoch=800, gamma=0.992, lr=8.75e-4, epoch_size=100_000_000, batch_size=16384, in_scaling=240, out_scaling=280, offset=270, offset1=270, offset2=270, adjust_loss=0.1):
    super(NNUE, self).__init__()
    self.num_ls_buckets = NUM_LS_BUCKETS

    self.input = DoubleFeatureTransformerSlice(feature_set.num_features, L1_MAIN, FM_DIM)
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

    max_hidden_weight = self.quantized_one / self.weight_scale_hidden
    max_out_weight = (self.quantized_one * self.quantized_one) / (self.nnue2score * self.weight_scale_out)

    self.weight_clipping = [
      {'params' : [self.input.v], 'min_weight' : -max_hidden_weight, 'max_weight' : max_hidden_weight },
      {'params' : [self.layer_stacks.l1.weight], 'min_weight' : -max_hidden_weight, 'max_weight' : max_hidden_weight, 'virtual_params' : self.layer_stacks.l1_fact.weight },
      {'params' : [self.layer_stacks.router.weight], 'min_weight' : -max_hidden_weight, 'max_weight' : max_hidden_weight },
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


  def _zero_virtual_feature_weights(self):
    weights = self.input.weight
    v_weights = self.input.v
    with torch.no_grad():
      for a, b in self.feature_set.get_virtual_feature_ranges():
        weights[a:b, :] = 0.0
        v_weights[a:b, :] = 0.0

    self.input.weight = nn.Parameter(weights)
    self.input.v = nn.Parameter(v_weights)

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

  def set_feature_set(self, new_feature_set):
    if self.feature_set.name == new_feature_set.name:
      return

    if len(self.feature_set.features) > 1:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

    old_feature_block = self.feature_set.features[0]
    new_feature_block = new_feature_set.features[0]

    if old_feature_block.name == next(iter(new_feature_block.factors)):
      weights = self.input.weight
      padding = weights.new_zeros((new_feature_block.num_virtual_features, weights.shape[1]))
      weights = torch.cat([weights, padding], dim=0)
      self.input.weight = nn.Parameter(weights)
      self.feature_set = new_feature_set
    else:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

  def forward(self, us, them, white_indices, white_values, black_indices, black_values, layer_stack_indices):

    # --- PHASE 1: 入力埋め込みと特徴量分離 ---
    t_w, t_b, v_w, v_b = self.input(white_indices, white_values, black_indices, black_values)

    H_START = 12672
    H_END   = 214210

    def process_fm(v_feat, indices):
        mask_h = (indices >= H_START) & (indices < H_END)
        mask_k = (indices < H_START) | (indices >= H_END)

        vh = v_feat * mask_h.unsqueeze(-1)
        vk = v_feat * mask_k.unsqueeze(-1)

        sh, sk = torch.sum(vh, dim=1), torch.sum(vk, dim=1)
        ih = 0.5 * (sh**2 - torch.sum(vh**2, dim=1))
        ik = 0.5 * (sk**2 - torch.sum(vk**2, dim=1))
        return ih, ik, sh, sk

    ih_w, ik_w, sh_w, sk_w = process_fm(v_w, white_indices)
    ih_b, ik_b, sh_b, sk_b = process_fm(v_b, black_indices)


    # --- PHASE 2: Main Path (蓄積計算) の処理 ---
    # ※最初は layer_stack_indicesを使用
    l0_raw = (us * torch.cat([t_w, t_b], dim=1)) + (them * torch.cat([t_b, t_w], dim=1))
    l0_clipped = torch.clamp(l0_raw, 0.0, 1.0)
    l0_s = torch.split(l0_clipped, 640, dim=1)
    
    pf = layer_stack_indices.view(-1, 1, 1).float() / 11.0
    w0 = torch.clamp(1.0 - pf * 3.0, min=0.0)
    w1 = torch.clamp(1.0 - torch.abs(pf * 3.0 - 1.0), min=0.0)
    w2 = torch.clamp(1.0 - torch.abs(pf * 3.0 - 2.0), min=0.0)
    w3 = torch.clamp(pf * 3.0 - 2.0, min=0.0)

    mixed_weights = (w0 * self.pair_weights[0] + 
                     w1 * self.pair_weights[1] + 
                     w2 * self.pair_weights[2] + 
                     w3 * self.pair_weights[3])
    weights = torch.softmax(mixed_weights, dim=2)

    # --- Branch Dropout (学習時のみ適用) ---
    branch_drop_prob = 0.01
    if self.training and branch_drop_prob > 0.0:
        # mixed_weightsと同じ形状 [batch, channels, 3] でマスク生成
        drop_mask = (torch.rand_like(mixed_weights) >= branch_drop_prob).float()
        
        # 3ブランチ全てがdropした場合は全復元（無効な重みを防止）
        all_dropped = (drop_mask.sum(dim=2, keepdim=True) == 0)
        drop_mask = torch.where(all_dropped, torch.ones_like(drop_mask), drop_mask)
        
        # （例: 500ステップごとに表示する場合）
        if self.global_step % 500 == 0:
            mul_drop_pct   = (1.0 - drop_mask[:, :, 0].mean()).item() * 100
            diff2_drop_pct = (1.0 - drop_mask[:, :, 1].mean()).item() * 100
            sum_drop_pct   = (1.0 - drop_mask[:, :, 2].mean()).item() * 100
            all_keep_pct   = (drop_mask.sum(dim=2) == 3).float().mean().item() * 100
            two_keep_pct   = (drop_mask.sum(dim=2) == 2).float().mean().item() * 100
            one_keep_pct   = (drop_mask.sum(dim=2) == 1).float().mean().item() * 100

            print(f"Branch Drop:\n"
                  f"  mul_drop   {mul_drop_pct:5.1f}%\n"
                  f"  diff2_drop {diff2_drop_pct:5.1f}%\n"
                  f"  sum_drop   {sum_drop_pct:5.1f}%\n"
                  f"  all_keep   {all_keep_pct:5.1f}%\n"
                  f"  two_keep   {two_keep_pct:5.1f}%\n"
                  f"  one_keep   {one_keep_pct:5.1f}%")

        # マスク適用後、残ったブランチの合計が 1.0 になるよう再正規化
        masked_weights = weights * drop_mask
        weights = masked_weights / (masked_weights.sum(dim=2, keepdim=True) + 1e-8)

    # --- Part 1 (l0_s[0] & l0_s[1]) ---
    term_mul = l0_s[0] * l0_s[1]
    term_diff_sq = torch.pow(l0_s[0] - l0_s[1], 2)
    term_sum = (l0_s[0] + l0_s[1]) * 0.5

    l1_main_part1 = (weights[:, :, 0] * term_mul + 
                     weights[:, :, 1] * term_diff_sq + 
                     weights[:, :, 2] * term_sum)

    # --- Part 2 (l0_s[2] & l0_s[3]) ---
    term_mul2 = l0_s[2] * l0_s[3]
    term_diff_sq2 = torch.pow(l0_s[2] - l0_s[3], 2)
    term_sum2 = (l0_s[2] + l0_s[3]) * 0.5

    l1_main_part2 = (weights[:, :, 0] * term_mul2 + 
                     weights[:, :, 1] * term_diff_sq2 + 
                     weights[:, :, 2] * term_sum2)

    l1_main_input = torch.cat([l1_main_part1, l1_main_part2], dim=1) * (127/128)


    # --- PHASE 3: FM項 (Diff / Abs) のスケーリング ---
    v_w_all = torch.stack([ih_w, ik_w, sh_w, sk_w], dim=1)
    v_b_all = torch.stack([ih_b, ik_b, sh_b, sk_b], dim=1)
    us_3d = us.view(-1, 1, 1)
    them_3d = them.view(-1, 1, 1)

    raw_diff = (us_3d * (v_w_all - v_b_all)) + (them_3d * (v_b_all - v_w_all))
    raw_abs  = (us_3d * v_w_all) + (them_3d * v_b_all)

    norm_diff = torch.tensor([0.01, 0.01, 0.05, 0.05], device=raw_diff.device).view(1, 4, 1)
    norm_abs  = torch.tensor([0.004, 0.004, 0.02, 0.02], device=raw_abs.device).view(1, 4, 1)

    diff_input_scaled = torch.clamp(raw_diff * norm_diff + 0.5, 0.0, 1.0)
    abs_input_scaled  = torch.clamp(raw_abs * norm_abs + 0.5, 0.0, 1.0)

    diff_input = diff_input_scaled.view(-1, 128)
    abs_input  = abs_input_scaled.view(-1, 128)


    # --- PHASE 4: LayerStacks による深層処理 ---
    # ★ 途中から router の結果 (router_indices) を受け取る
    final_output, l3_out, l1_main_bp, l2_input, diff_gated, abs_gated, gate_d, gate_a, channel_stats, router_indices, router_logits, all_final_outputs = self.layer_stacks(
        l1_main_input, diff_input, abs_input, ls_indices=layer_stack_indices
    )

    # --- 統計ログの呼び出し (router_indices を渡してログ出力) ---
    if self.training:
        self._log_detailed_stats(
            us, white_indices, l1_main_input,
            diff_input, abs_input,
            diff_gated, abs_gated,
            router_indices,  # ★ ログ出力時には router の結果を使用
            final_output, l3_out, l1_main_bp, l2_input,
            gate_d, gate_a, channel_stats
        )

    return final_output, router_logits, all_final_outputs


  def _log_detailed_stats(self, us, white_indices, l1_main, 
                         diff_l1, abs_l1,
                         diff_gated, abs_gated,
                         layer_stack_indices, final_output, l3_out, l1_main_bp, l2_input,
                         gate_d, gate_a, channel_stats):

    if not hasattr(self, 'dbg_cnt'): self.dbg_cnt = 0
    self.dbg_cnt += 1
    if self.dbg_cnt % 500 != 0: return

    with torch.no_grad():
      print(f"\n[FM Detailed Debug Step {self.dbg_cnt}]")

      _b_size = us.shape[0]

      _num_active_total = torch.sum(white_indices >= 0).item()
      _num_act_avg = _num_active_total / _b_size
      _unique_active = torch.unique(white_indices[white_indices >= 0]).numel()

      print(f" Features   | ActiveAvg:{_num_act_avg:5.1f}, Unique:{_unique_active}")
      print(f" LayerStacks Entry Analysis")
      print(f"   MainPath | mean:{l1_main.mean():7.4f}, std:{l1_main.std():7.4f}, min:{l1_main.min():7.4f}, max:{l1_main.max():7.4f}")
      print(f"   FM Diff  | mean:{diff_l1.mean():7.4f}, std:{diff_l1.std():7.4f}, min:{diff_l1.min():7.4f}, max:{diff_l1.max():7.4f}")
      print(f"   FM Abs   | mean:{abs_l1.mean():7.4f}, std:{abs_l1.std():7.4f}, min:{abs_l1.min():7.4f}, max:{abs_l1.max():7.4f}")

      l3_mag = l3_out.abs().mean().item()
      m_bp_mag = l1_main_bp.abs().mean().item()

      total_mag = l3_mag + m_bp_mag + 1e-9
      l3_ratio = (l3_mag / total_mag) * 100
      bp_ratio = (m_bp_mag / total_mag) * 100

      print(f" Output Composition | L3(Deep): {l3_mag:7.4f} ({l3_ratio:4.1f}%) | MainBypass: {m_bp_mag:7.4f} ({bp_ratio:4.1f}%)")
      print(f" Final Score Range  | Min: {final_output.min():7.2f}, Max: {final_output.max():7.2f}, Mean: {final_output.mean():7.2f}")

      alphas = torch.sigmoid(self.layer_stacks.blend).cpu().numpy()
      alpha_pct = alphas * 100

      print(f" Blend Alpha (DeepPath Ratio):")
      print(f"   Mean : {alpha_pct.mean():.2f}%")
      print(f"   Min  : {alpha_pct.min():.2f}%")
      print(f"   Max  : {alpha_pct.max():.2f}%")

      print("-" * 110)
      print(f"{'FM Component':19} | {'Raw min / mean  / max    / std':33} | {'Zero%':>7} | {'High%':>7}")
      print("-" * 110)

      def get_sat_zero(t):
          return (t <= 0.01).float().mean().item() * 100
      def get_sat_high(t):
          return (t >= 0.98).float().mean().item() * 100

      r_d_4ch = diff_l1.view(-1, 4, 32)
      r_a_4ch = abs_l1.view(-1, 4, 32)
      
      names = ["Inter HalfKA", "Inter KSDG3", "SumV HalfKA", "SumV KSDG3"]

      for i, name in enumerate(names):
          d_comp = r_d_4ch[:, i, :]
          d_min, d_m, d_max, d_std = d_comp.min(), d_comp.mean(), d_comp.max(), d_comp.std()
          sat_zero_l1_d = get_sat_zero(d_comp)
          sat_high_l1_d = get_sat_high(d_comp)
          print(f" {name+'(Diff)':18} | {d_min:6.3f} / {d_m:6.3f} / {d_max:6.3f} / {d_std:6.3f} | {sat_zero_l1_d:6.2f}% | {sat_high_l1_d:6.2f}%")
          
          a_comp = r_a_4ch[:, i, :]
          a_min, a_m, a_max, a_std = a_comp.min(), a_comp.mean(), a_comp.max(), a_comp.std()
          sat_zero_l1_a = get_sat_zero(a_comp)
          sat_high_l1_a = get_sat_high(a_comp)
          print(f" {name+'(Abs)':18} | {a_min:6.3f} / {a_m:6.3f} / {a_max:6.3f} / {a_std:6.3f} | {sat_zero_l1_a:6.2f}% | {sat_high_l1_a:6.2f}%")

      print("-" * 115)
      print(f"{'Component':38} | {'Min':>6} / {'Mean':>6} / {'Max':>6} / {'Std':>6} ")
      print("-" * 115)

      d_comp = diff_gated.detach()
      print(f" {'FM Diff (RMSNormed, before scaling)':37} | {d_comp.min():6.2f} / {d_comp.mean():6.2f} / {d_comp.max():6.2f} / {d_comp.std():6.2f}")

      a_comp = abs_gated.detach()
      print(f" {'FM Abs  (Gated, before scaling)':37} | {a_comp.min():6.2f} / {a_comp.mean():6.2f} / {a_comp.max():6.2f} / {a_comp.std():6.2f}")

      print("-" * 110)
      print(f"[Signal Strength (L2 Input)]")
      main_sqr_part = l2_input[:, 0:31].abs().mean().item()
      main_raw_part = l2_input[:, 31:62].abs().mean().item()
      fm_diff_part  = l2_input[:, 62:94].abs().mean().item()
      fm_abs_part   = l2_input[:, 94:158].abs().mean().item()
      cross_feat    = l2_input[:, 158:190].abs().mean().item()

      print(f" L2 In | Main(Sqr): {main_sqr_part:.4f} | Main(Raw): {main_raw_part:.4f} | FM(Diff): {fm_diff_part:.4f} | FM(Abs): {fm_abs_part:.4f}  | cross_feat: {cross_feat:.4f}")

      print("-" * 110)
      print(f"{'Section':12} | {'Mean':>7} | {'AbsMean':>7} | {'Std':>7} | {'Max':>7} | {'Min':>7} | {'Zero%':>7} | {'High%':>7}")
      print("-" * 110)
      
      def log_stats(name, tensor):
          t = tensor.detach().float()
          t_abs = t.abs()
          sparsity = (t_abs < 0.01).float().mean().item() * 100
          high_signal = (t > 0.95).float().mean().item() * 100
          print(f"{name:12} | {t.mean():7.3f} | {t_abs.mean():7.3f} | {t.std():7.3f} | {t.max():7.2f} | {t.min():7.2f} | {sparsity:6.1f}% | {high_signal:6.1f}%")
      
      log_stats("Main(Sqr)",    l2_input[:, 0:31])
      log_stats("Main(Raw)",    l2_input[:, 31:62])
      log_stats("FM(Diff)",     l2_input[:, 62:94])
      log_stats("FM(Abs_Raw)",  l2_input[:, 94:126])
      log_stats("FM(Abs_Sqr)",  l2_input[:, 126:158])
      log_stats("cross_feat",   l2_input[:, 158:190])

      if self.input.v.grad is not None:
          v_grad_mean = self.input.v.grad.abs().mean().item()
          m_grad_mean = self.input.weight.grad.abs().mean().item()
          g_ratio = v_grad_mean / (m_grad_mean + 1e-9)
          print(f"[Gradient] MeanAbs_V:{v_grad_mean:8.2e}, MeanAbs_M:{m_grad_mean:8.2e}, G_Ratio:{g_ratio:8.4f}")

      print(f"[Weights]  Diff_W:{self.layer_stacks.fm_diff.weight.norm():6.2f}, "
            f"Abs_W:{self.layer_stacks.fm_abs.weight.norm():6.2f}, "
            f"L2_W:{self.layer_stacks.l2.weight.norm():6.2f}")

      w_input_c = self.input.weight.detach().cpu()
      g_input_c = self.input.weight.grad.detach().cpu() if self.input.weight.grad is not None else None
      b_input_c = self.input.bias.detach().cpu()

      v_input_c = self.input.v.detach().cpu()
      vg_input_c = self.input.v.grad.detach().cpu() if self.input.v.grad is not None else None

      pw_w = self.pair_weights.detach().cpu()
      pw_g = self.pair_weights.grad.detach().cpu() if self.pair_weights.grad is not None else None
      
      pw_softmax = torch.softmax(pw_w, dim=2) 

      S0, S1, S2 = 0, 12672, 203670

      l1_main_w = self.layer_stacks.l1.weight.detach().cpu()
      l1_fact_w = self.layer_stacks.l1_fact.weight.detach().cpu()
      
      l1_main_g = self.layer_stacks.l1.weight.grad.detach().cpu() if self.layer_stacks.l1.weight.grad is not None else None
      l1_fact_g = self.layer_stacks.l1_fact.weight.grad.detach().cpu() if self.layer_stacks.l1_fact.weight.grad is not None else None

      l1_main_b = self.layer_stacks.l1.bias.detach().cpu() if self.layer_stacks.l1.bias is not None else None
      l1_fact_b = self.layer_stacks.l1_fact.bias.detach().cpu() if self.layer_stacks.l1_fact.bias is not None else None
      
      xs = l1_main_w.shape[0] // l1_fact_w.shape[0]
      ys = l1_main_w.shape[1] // l1_fact_w.shape[1]
      expanded_fact_w = l1_fact_w.repeat(xs, ys)
      l1_combined_w = l1_main_w + expanded_fact_w

      if l1_main_g is not None and l1_fact_g is not None:
          l1_combined_g = l1_main_g + l1_fact_g.repeat(xs, ys)
      else:
          l1_combined_g = l1_main_g if l1_main_g is not None else None

      if l1_main_b is not None and l1_fact_b is not None:
          bias_xs = l1_main_b.shape[0] // l1_fact_b.shape[0]
          expanded_fact_b = l1_fact_b.repeat(bias_xs)
          l1_combined_b = l1_main_b + expanded_fact_b
      else:
          l1_combined_b = l1_main_b if l1_main_b is not None else l1_fact_b

      parts = [
          ("W_input (All)   ", w_input_c, g_input_c, b_input_c),
          ("W_KSDG3 (Part)  ", w_input_c[S0:S1, :], g_input_c[S0:S1, :] if g_input_c is not None else None, b_input_c),
          ("W_HalfKA (Part) ", w_input_c[S1:S2, :], g_input_c[S1:S2, :] if g_input_c is not None else None, b_input_c),
          ("V_Factor (FM)   ", v_input_c, vg_input_c, None),
          ("Pair_W (Raw)      ", pw_w, pw_g, None),
          ("router ", self.layer_stacks.router.weight.detach().cpu(), self.layer_stacks.router.weight.grad.detach().cpu() if self.layer_stacks.router.weight.grad is not None else None, self.layer_stacks.router.bias.detach().cpu()),
          ("L1_Main (Linear)", self.layer_stacks.l1.weight.detach().cpu(), self.layer_stacks.l1.weight.grad.detach().cpu() if self.layer_stacks.l1.weight.grad is not None else None, self.layer_stacks.l1.bias.detach().cpu()),
          ("L1_Fact         ", self.layer_stacks.l1_fact.weight.detach().cpu(), self.layer_stacks.l1_fact.weight.grad.detach().cpu() if self.layer_stacks.l1_fact.weight.grad is not None else None, self.layer_stacks.l1_fact.bias.detach().cpu()),
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

      phase_names = ["Open", "Mid1", "Mid2", "End "]
      for p in range(4):
          parts.append((f"P_{phase_names[p]}_Mul   ", pw_softmax[p, :, 0], pw_g[p, :, 0], None))
          parts.append((f"P_{phase_names[p]}_Diff  ", pw_softmax[p, :, 1], pw_g[p, :, 1], None))
          parts.append((f"P_{phase_names[p]}_Sum   ", pw_softmax[p, :, 2], pw_g[p, :, 2], None))

      print("-" * 120)
      print(f"{'Layer Name':<18} | {'Grad Mean':<12} {'Active':<8} | {'W_Mean':<8} {'W_Min':<8} {'W_Max':<9} {'W_Std':<7} | {'B_Mean':<8} {'B_Min':<8} {'B_Max':<9} {'B_Std':<7}")
      print("-" * 120)

      for name, w, g, b in parts:
          gm = (g.norm().item() / (g.numel()**0.5 + 1e-9)) if g is not None else 0.0
          ga = (g != 0).sum().item() if g is not None else 0

          wm, wmin, wmax, ws = w.mean().item(), w.min().item(), w.max().item(), w.std().item()

          if b is not None and b.numel() > 0:
              bm, bmin, bmax = b.mean().item(), b.min().item(), b.max().item()
              bs = b.std().item() if b.numel() > 1 else 0.0
          else:
              bm, bmin, bmax, bs = 0.0, 0.0, 0.0, 0.0

          print(f"{name:<18} | {gm:12.10f} {ga:<8} | {wm:+8.5f} {wmin:+8.5f} {wmax:+8.5f} {ws:8.5f} | {bm:+8.5f} {bmin:+8.5f} {bmax:+8.5f} {bs:8.5f}")
      print("-" * 120)

      att = self.layer_stacks.last_att_score
      att_mean, att_min, att_max, att_std = att.mean().item(), att.min().item(), att.max().item(), att.std().item()

      print(f"[Attention Status] dynamic_scale (FM-Filter)")
      print(f"  Mean: {att_mean:.4f} | Min: {att_min:.4f} | Max: {att_max:.4f} | Std: {att_std:.4f}")
      low_rate = (att < 0.2).float().mean().item() * 100
      high_rate = (att > 0.8).float().mean().item() * 100
      print(f"  Distribution: Low(<0.2): {low_rate:.1f}% | High(>0.8): {high_rate:.1f}%")
      print(f"  Current Temp (T) : {self.layer_stacks.last_lca_temp.item():.4f}")

      if hasattr(self.layer_stacks, 'lca_temp'):
          temp_param = self.layer_stacks.lca_temp
          t_val = temp_param.item()
          t_grad = temp_param.grad.item() if temp_param.grad is not None else 0.0
          
          direction = "Sharper(0/1) ↓" if t_grad > 0 else "Milder(0.5) ↑"
          print(f"[LCA Meta-Learning]")
          print(f"  Current Temp (T) : {t_val:.4f}")
          print(f"  Temp Grad        : {t_grad:+.2e} [{direction}]")

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

      phase_names = ["Open", "Mid1", "Mid2", "End "]
      for p in range(4):
          avg_m = pw_softmax[p, :, 0].mean().item()
          avg_d = pw_softmax[p, :, 1].mean().item()
          avg_s = pw_softmax[p, :, 2].mean().item()
          print(f"  Phase {phase_names[p]} Mix Ratio -> Mul: {avg_m:.3f}, Diff: {avg_d:.3f}, Sum: {avg_s:.3f}")

      def get_layer_stats(w, g, b):
          wm, wmin, wmax, ws = w.mean().item(), w.min().item(), w.max().item(), w.std().item()
          gm = (g.norm().item() / (g.numel()**0.5 + 1e-9)) if g is not None else 0.0
          ga = (g != 0).sum().item() if g is not None else 0
          bm = b.mean().item() if b is not None else 0
          return gm, ga, wm, wmin, wmax, ws, bm

      print("-" * 115)
      print(f"{'Layer (Bucket)':<22} | {'Grad Mean':<12} {'Active':<8} | {'W_Mean':<8} {'W_Min':<8} {'W_Max':<9} {'W_Std':<7} | {'B_Mean':<8}")
      print("-" * 115)
      
      target_buckets = [0, 11]
      
      for b_idx in target_buckets:
          base = b_idx * 64

          for name, layer in [("FM_Diff", self.layer_stacks.fm_diff), ("FM_Abs", self.layer_stacks.fm_abs)]:
              w_all = layer.weight.detach().cpu()
              g_all = layer.weight.grad.detach().cpu() if layer.weight.grad is not None else None
              b_all = layer.bias.detach().cpu()

              p_gate = get_layer_stats(w_all[base : base+32], 
                                 g_all[base : base+32] if g_all is not None else None,
                                 b_all[base : base+32])
              p_val = get_layer_stats(w_all[base+32 : base+64], 
                                g_all[base+32 : base+64] if g_all is not None else None,
                                b_all[base+32 : base+64])

              print(f"{name+'_Gate(B'+str(b_idx)+')':<22} | {p_gate[0]:12.10f} {p_gate[1]:<8} | {p_gate[2]:+8.5f} {p_gate[3]:+8.5f} {p_gate[4]:+8.5f} {p_gate[5]:8.5f} | {p_gate[6]:+8.5f}")
              print(f"{name+'_Val (B'+str(b_idx)+')':<22} | {p_val[0]:12.10f} {p_val[1]:<8} | {p_val[2]:+8.5f} {p_val[3]:+8.5f} {p_val[4]:+8.5f} {p_val[5]:8.5f} | {p_val[6]:+8.5f}")

          if b_idx == 0: print("-" * 115)
      print("-" * 115)

      open_to_abs = torch.sigmoid(gate_a).mean().item() * 100.0
      eff_sigmoid_d = 0.5 + 0.5 * torch.sigmoid(gate_d)
      open_to_main = eff_sigmoid_d.mean().item() * 100.0

      sharpness_d = torch.sigmoid(gate_d).var().item()
      sharpness_a = torch.sigmoid(gate_a).var().item()

      print(f"--- Inter-Gating Status (Effective) ---")
      print(f"Abs (Filtered by Abs-Gate) Open: {open_to_abs:.2f}% (sharp:{sharpness_a:.3f})")
      print(f"Main (Filtered by Diff-Gate) Open: {open_to_main:.2f}% (sharp:{sharpness_d:.3f})")

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

      # ★ router_indices を使用してバケット別ログを出力
      self._log_bucket_stats(l3_out, l1_main_bp, diff_gated, layer_stack_indices, gate_d, gate_a)


  def _log_bucket_stats(self, l3_out, l1_main_bp, diff_gated, layer_stack_indices, gate_d, gate_a):
    with torch.no_grad():
      batch_size = layer_stack_indices.size(0)
      bucket_counts = torch.bincount(layer_stack_indices, minlength=12)
      bucket_ratios = (bucket_counts.float() / batch_size) * 100.0

      final_output = l3_out + l1_main_bp
      final_cp = final_output * self.nnue2score

      all_alphas_pct = torch.sigmoid(self.layer_stacks.blend).cpu().numpy() * 100

      print(f"[Bucket-wise FM Value & Gate Analysis (Router-based)]")
      print("-" * 110)
      header = f"{'B_ID':<4} | Samples% | {'Eval(avg_cp,abs_cp)':<9} | {'L1_Main':<8} | {'FM_Diff_V':<12} | {'FM_Abs_V':<12} | {'AbsOpen(GatebyAbs)%':<10} | {'MainOpen(GatebyDiff)%':<10} | {'L2_Layer':<8} | {'L3(Deep)%':<8} | {'Blend(Alpha)%':<10}"
      print(header)
      print("-" * 110)

      w1 = self.layer_stacks.l1.weight.detach().cpu()
      wd = self.layer_stacks.fm_diff.weight.detach().cpu()
      wa = self.layer_stacks.fm_abs.weight.detach().cpu()

      gd_grad = self.layer_stacks.fm_diff.weight.grad.detach().cpu() if self.layer_stacks.fm_diff.weight.grad is not None else torch.zeros_like(wd)
      ga_grad = self.layer_stacks.fm_abs.weight.grad.detach().cpu() if self.layer_stacks.fm_abs.weight.grad is not None else torch.zeros_like(wa)

      w2 = self.layer_stacks.l2.weight.detach().cpu()

      for i in range(self.layer_stacks.count):
        mask = (layer_stack_indices == i)
        if not mask.any():
            continue

        s_ratio = bucket_ratios[i].item()

        s1, e1 = i * 32, (i + 1) * 32
        sf, ef = i * 64, (i + 1) * 64
        s2, e2 = i * 96, (i + 1) * 96

        md_w = wd[sf+32:ef].abs().mean().item()
        md_g = gd_grad[sf+32:ef].abs().mean().item()

        ma_w = wa[sf+32:ef].abs().mean().item()
        ma_g = ga_grad[sf+32:ef].abs().mean().item()

        b_mask = (layer_stack_indices == i)

        if b_mask.sum() > 0:
            avg_cp = final_cp[b_mask].mean().item()
            abs_cp = final_cp[b_mask].abs().mean().item()

            open_abs = torch.sigmoid(gate_a[b_mask]).mean().item() * 100
            open_main = (0.5 + 0.5 * torch.sigmoid(gate_d[b_mask])).mean().item() * 100

            fm_r = (l3_out[b_mask].abs().mean().item() / 
                   (l3_out[b_mask].abs().mean().item() + l1_main_bp[b_mask].abs().mean().item() + 1e-9)) * 100
        else:
            avg_cp, abs_cp, open_abs, open_main, fm_r = 0.0, 0.0, 0.0, 0.0, 0.0

        current_alpha = all_alphas_pct[i]

        print(f"B{i:02d} | {s_ratio:5.1f}% | {avg_cp:+7.1f}({abs_cp:6.1f}) | {w1[s1:e1].abs().mean():.3f} | {md_w:.3f}|{md_g:.1e} | {ma_w:.3f}|{ma_g:.1e} | {open_abs:5.1f}% | {open_main:5.1f}% | {w2[s2:e2].abs().mean():.3f}   | {fm_r:5.1f}% | {current_alpha:5.1f}%")
      print("-" * 110)

      print("\n[Bucket-wise Phase Gate Analysis (6-Channel Router-based)]")
      print("-" * 110)
      print(f"{'B_ID':<4} | {'MSqr':<5} | {'MRaw':<5} | {'Diff':<5} | {'AbsR':<5} | {'AbsS':<5} | {'Cross':<5} | {'Low%':<5} | {'High%':<6} | {'Samples%':<8} | {'AttScore(mean/std)':<18} | {'ValBaseLoss':<8} ")
      print("-" * 110)

      total_samples = layer_stack_indices.size(0)

      for i in range(12):
          mask = (layer_stack_indices == i)
          count = mask.sum().item()
          if count == 0: continue

          p_batch = self.layer_stacks.last_phase[mask]
          p_means = p_batch.mean(dim=0)

          low_r  = (p_batch < 0.2).float().mean().item() * 100
          high_r = (p_batch > 0.8).float().mean().item() * 100
          s_ratio = (count / total_samples) * 100

          m_sq, m_ra, f_di, f_ar, f_as, crs = p_means.tolist()

          att = self.layer_stacks.last_att_score[mask]

          if torch.is_tensor(self.last_bucket_losses):
              loss_val = self.last_bucket_losses[i].cpu().item()
          else:
              loss_val = self.last_bucket_losses[i] if self.last_bucket_losses is not None else 0.0

          print(f"B{i:02d}  | {m_sq:.3f} | {m_ra:.3f} | {f_di:.3f} | {f_ar:.3f} | {f_as:.3f} | {crs:.3f} | {low_r:>4.1f}% | {high_r:>5.1f}% | {s_ratio:>7.1f}% | {att.mean().item():.3f} / {att.std().item():.3f} | {loss_val:>7.5f} ")
      print("-" * 110)


  def step_(self, batch, batch_idx, loss_type):
    self.print_mem("step_() start")

    self._clip_weights()
    self.print_mem("After _clip_weights")

    (
        us,
        them,
        white_indices,
        white_values,
        black_indices,
        black_values,
        outcome,
        score,
        layer_stack_indices,
        material,
        kif_group_id,
    ) = batch
    self.print_mem("After batch")

    # ==========================================
    # Phase 1: Forward推論 & 基本変数の準備
    # ==========================================
    actual_lambda = self._get_actual_lambda(loss_type)
    kif_group_id_flat = kif_group_id.view(-1)

    self.print_mem("Before Student Forward")

    # Studentモデル推論
    scorenet, router_logits, all_final_outputs = self(
        us,
        them,
        white_indices,
        white_values,
        black_indices,
        black_values,
        layer_stack_indices,
    )
    self.print_mem("After Student Forward")  # 活性化値（中間テンソル）の保持量を計測

    scorenet = scorenet * self.nnue2score

    # 実際のルーティング結果インデックスを取得
    active_indices = getattr(
        self.layer_stacks, "last_routing_indices", layer_stack_indices
    )

    # ==========================================
    # Phase 2: スコアから勝率(qf, pf, pt)への変換
    # ==========================================
    # 選ばれたバケットの予測勝率 (Student)
    q = (scorenet - self.offset1) / self.in_scaling
    qm = (-scorenet - self.offset2) / self.in_scaling
    qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())

    # 教師データの目標勝率 (Target)
    p = (score - self.offset1) / self.out_scaling
    pm = (-score - self.offset2) / self.out_scaling
    pf = 0.5 * (1.0 + p.sigmoid() - pm.sigmoid())

    pt = pf * actual_lambda + outcome * (1.0 - actual_lambda)

    # ==========================================
    # Phase 3: 各種 Loss の計算
    # ==========================================

    # --- 3-1. メイン損失 ---
    base_loss = self._compute_base_loss(pt, qf, pf, kif_group_id_flat)
    self.print_mem("After Base Loss")


    # --- 3-2. EMA蒸留 ---
    ema_distill_loss = self._compute_ema_loss(
        scorenet,
        router_logits,       # 追加
        all_final_outputs,   # 追加
        us,
        them,
        white_indices,
        white_values,
        black_indices,
        black_values,
        layer_stack_indices,
    )


    # --- 3-3. Router 関連損失 ---
    router_ce_loss, r_info = self._compute_router_teacher_loss(all_final_outputs, router_logits, pt)

    router_load_loss, router_margin_loss = self._compute_router_loss(
        best_bucket_indices=r_info['best_bucket_indices'],
        oracle_gaps=r_info['oracle_gaps'],
        margin_base=0.05,
        margin_scale=0.25,
        margin_min=0.02,
        margin_max=0.25,
        soft_incorrect_weight=0.2,
        use_logit_margin=False
    )


    # --- 3-4. Bucket Distillation ---
    bucket_distill_loss = self._compute_bucket_distill_loss(
        all_final_outputs=all_final_outputs,
        pt=pt,
        active_indices=active_indices,
        oracle_top_k=2,
    )
    self.print_mem("After Bucket Distill Loss")


    # --- 3-5. ランキング損失 (Pairwise / Listwise) ---
    pairwise_mask = kif_group_id_flat == 3
    n_pairwise = pairwise_mask.sum().item()

    sorted_data = self._prepare_sorted_data(
        pairwise_mask,
        n_pairwise,
        pt,
        qf,
        score,
        scorenet,
        active_indices,
        material,
    )
    self.print_mem("After Prepare Sorted Data")

    pairwise_loss, pair_metrics = self._compute_pairwise_loss(
        sorted_data, n_pairwise, pt.device
    )
    self.print_mem("After Pairwise Loss")

    listwise_loss, pt_range = self._compute_listwise_loss(
        sorted_data, n_pairwise, pt.device
    )
    self.print_mem("After Listwise Loss")


    # --- 3-6. 正則化・ペナルティ損失 ---
    phase_penalty = self._compute_phase_penalty()
    self.print_mem("After Phase Penalty")

    ortho_loss = self._compute_ortho_loss(threshold=0.2)
    self.print_mem("After Ortho Loss")

    # ==========================================
    # Phase 4: Total Loss の計算
    # ==========================================
    weights = {
        "pairwise"      : 0.0100,
        "listwise"      : 0.0300,
        "phase"         : 0.0020,
        "router_load"   : 0.0030,
        "router_margin" : 0.0030,
        "router_ce"     : 0.0030,
        "ortho"         : 0.0010,
        "ema_distill"   : 0.0070,
        "bucket_distill": 0.0030, # 未選択bucketへの弱い蒸留
    }

    loss = (
        base_loss
        + (weights["pairwise"] * pairwise_loss)
        + (weights["listwise"] * listwise_loss)
        + (weights["phase"] * phase_penalty)
        + (weights["router_load"] * router_load_loss)
        + (weights["router_margin"] * router_margin_loss)
        + (weights["router_ce"] * router_ce_loss)
        + (weights["ortho"] * ortho_loss)
        + (weights["ema_distill"] * ema_distill_loss)
        + (weights["bucket_distill"] * bucket_distill_loss)
    )
    self.print_mem("After Total Loss")

    # ==========================================
    # Phase 5: 統計更新 & ログ出力
    # ==========================================
    if self.training and (self.global_step % 500 == 0):
        self._print_router_bucket_debug(
            r_info["best_bucket_indices"],
            r_info["pred_bucket_indices"],
            r_info["oracle_gaps"],
            router_logits,
        )

    self._update_bucket_stats(pt, qf, active_indices, loss_type)

    self._log_debug_info(
        loss_type,
        loss,
        base_loss,
        pairwise_loss,
        listwise_loss,
        pt,
        qf,
        score,
        scorenet,
        active_indices,
        kif_group_id_flat,
        actual_lambda,
        pair_metrics,
        pt_range,
        router_load_loss,
        phase_penalty,
        ortho_loss,
        router_margin_loss,
        ema_distill_loss,
        router_ce_loss,
        bucket_distill_loss,
        r_info["router_acc"],
        r_info["gap_mean"],
        r_info["gap_median"],
        r_info["gap_max"],
        r_info["gw_mean"],
        r_info["gw_median"],
        r_info["gw_gt_05"],
        r_info["gw_gt_08"],
        r_info["router_acc_high"],
        weights,
    )

    self._log_debug_gpu_info()
    self.print_mem("step_() end")

    return loss


  def _get_actual_lambda(self, loss_type):
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
      loss_elements = torch.pow(torch.abs(pt - qf), 2.5)
      loss_elements = loss_elements * ((qf > pt) * self.adjust_loss + 1)

      weights = 1 + (2.0**1.2 - 1) * torch.pow((pf - 0.5) ** 2 * pf * (1 - pf), 0.8)
      weights_flat = weights.view(-1)

      base_loss_mask = (kif_group_id_flat == 1) | (kif_group_id_flat == 2)

      if base_loss_mask.any():
          return (loss_elements.view(-1)[base_loss_mask] * weights_flat[base_loss_mask]).sum() / weights_flat[base_loss_mask].sum()
      return torch.tensor(0.0, device=pt.device)


  def _compute_ema_loss(
      self,
      scorenet: torch.Tensor,
      router_logits: torch.Tensor,
      all_final_outputs: torch.Tensor,
      us,
      them,
      white_indices,
      white_values,
      black_indices,
      black_values,
      layer_stack_indices,
  ) -> torch.Tensor:
      """EMA (Teacher) モデルとの Consistency Loss を計算する。"""
      
      # EMAネット (Teacher) の推論
      with torch.no_grad():
          self.ema_model.eval()
          scorenet_ema, router_logits_ema, all_outputs_ema = self.ema_model(
              us,
              them,
              white_indices,
              white_values,
              black_indices,
              black_values,
              layer_stack_indices,
          )
          # ★ cp 単位への変換 (* self.nnue2score) は EMA Loss の計算では行わない
  
      # 1. 主出力 (全バケット) の Consistency Loss (生出力のまま計算)
      # cpスケールで beta=10.0 だった場合、生出力空間では beta=0.1 ~ 0.5 程度が目安です
      score_ema_loss = F.smooth_l1_loss(all_final_outputs, all_outputs_ema, beta=0.1)
  
      # 2. Router の Consistency Loss (KL Divergence)
      p_student = F.log_softmax(router_logits, dim=-1)
      q_teacher = F.softmax(router_logits_ema, dim=-1)
      router_ema_loss = F.kl_div(p_student, q_teacher, reduction='batchmean')
  
      # 3. 統合 (スケールが揃うため 0.1 ~ 1.0 程度でバランスが取れるようになります)
      w_router = 0.5  # ログを見ながら 0.1 ~ 1.0 で調整
      total_ema_loss = score_ema_loss + w_router * router_ema_loss
  
      # 定期デバッグ出力
      if self.training and (self.global_step % 500 == 0):
          with torch.no_grad():
              # scorenet はすでに cp 単位、scorenet_ema は生出力なので EMA 側にだけ nnue2score を掛ける
              scorenet_ema_cp = scorenet_ema * self.nnue2score
              cp_diff = torch.abs(scorenet - scorenet_ema_cp)
  
              print("[EMA DEBUG]")
              print(f"  Student CP : mean={scorenet.abs().mean():.2f}")
              print(f"  EMA CP     : mean={scorenet_ema_cp.abs().mean():.2f}")
              print(f"  CP diff    : mean={cp_diff.mean():.2f} / median={cp_diff.median():.2f}")
              print("[EMA Loss Breakdown]")
              print(f"  Score EMA Loss (Raw) : {score_ema_loss.item():.5f}")
              print(f"  Router KL Loss       : {router_ema_loss.item():.5f} (Weighted: {w_router * router_ema_loss.item():.5f})")
  
      return total_ema_loss


  def _compute_router_teacher_loss(self, all_final_outputs, router_logits, pt):
    """
    全バケット出力からOracleターゲットを算出し、教師誘導 Router Loss (Hard CE + Soft KL) 
    および各種メトリクスを計算する。
    """
    with torch.no_grad():
        # 全12バケットの出力を評価値スケールに変換: [B, 12]
        all_scorenet = all_final_outputs * self.nnue2score
        
        # 全12バケットの勝率 (all_qf) を一括計算: [B, 12]
        all_q  = ( all_scorenet - self.offset1) / self.in_scaling
        all_qm = (-all_scorenet - self.offset2) / self.in_scaling
        all_qf = 0.5 * (1.0 + all_q.sigmoid() - all_qm.sigmoid())

        # 教師ターゲット pt: [B, 1] と全バケット勝率: [B, 12] の絶対誤差を比較
        bucket_errors = torch.abs(all_qf - pt.view(-1, 1))

        # Top1 と Top2 の誤差差 (Gap) を算出
        sorted_errors, _ = torch.sort(bucket_errors, dim=-1)
        oracle_gaps = sorted_errors[:, 1] - sorted_errors[:, 0]

        # 最も誤差が小さかったバケット (Oracle) および予測バケットの取得
        best_bucket_indices = torch.argmin(bucket_errors, dim=-1)
        pred_bucket_indices = torch.argmax(router_logits, dim=-1)
        router_acc = (pred_bucket_indices == best_bucket_indices).float().mean()

    # 1. Gap Weight (Sigmoid による連続的・滑らかな重み付け)
    raw_gap_weight = torch.sigmoid((oracle_gaps - 0.008) / 0.003)
    gap_weight = raw_gap_weight * 0.95 + 0.05

    # 2. Hard Teacher (Gap 重み付き Cross-Entropy)
    ce_per_sample = F.cross_entropy(router_logits, best_bucket_indices, reduction='none')
    gap_weight_sum = gap_weight.sum()
    if gap_weight_sum > 1e-6:
        router_ce_loss = (ce_per_sample * gap_weight).sum() / gap_weight_sum
    else:
        router_ce_loss = ce_per_sample.mean()

    # 3. Soft Teacher (誤差ベースの KL Divergence)
    temp = 0.005
    relative_errors = bucket_errors - bucket_errors.min(dim=1, keepdim=True).values
    soft_targets = F.softmax(-relative_errors / temp, dim=-1)

    router_logprob = F.log_softmax(router_logits, dim=-1)
    router_kl_loss = F.kl_div(router_logprob, soft_targets, reduction='batchmean')

    # 4. 損失の統合 (Hard CE を主軸にしたハイブリッド化)
    alpha, beta = 0.7, 0.3
    router_ce_loss = alpha * router_ce_loss + beta * router_kl_loss

    # 辞書の初期化時にデフォルト値を設定しておく
    metrics = {
        'router_acc': router_acc,
        'gap_mean': oracle_gaps.mean().item(),
        'gap_median': oracle_gaps.median().item(),
        'gap_max': oracle_gaps.max().item(),
        'best_bucket_indices': best_bucket_indices,
        'pred_bucket_indices': pred_bucket_indices,
        'oracle_gaps': oracle_gaps,
        'gap_weight': gap_weight,
        # デフォルト値を設定 (KeyError 防止)
        'gw_mean': 0.0,
        'gw_median': 0.0,
        'gw_gt_05': 0.0,
        'gw_gt_08': 0.0,
        'router_acc_high': 0.0,
    }

    # 100ステップ毎のみ実際の値を上書き計算
    if self.training and (self.global_step % 100 == 0):
        with torch.no_grad():
            metrics['gw_mean'] = gap_weight.mean().item()
            metrics['gw_median'] = gap_weight.median().item()
            metrics['gw_gt_05'] = (gap_weight > 0.5).float().mean().item() * 100
            metrics['gw_gt_08'] = (gap_weight > 0.8).float().mean().item() * 100

            high_gap_mask = gap_weight > 0.5
            if high_gap_mask.sum() > 0:
                metrics['router_acc_high'] = (pred_bucket_indices[high_gap_mask] == best_bucket_indices[high_gap_mask]).float().mean().item() * 100
            else:
                metrics['router_acc_high'] = 0.0

    return router_ce_loss, metrics


  def _compute_router_loss(self,
                         best_bucket_indices,
                         oracle_gaps,
                         margin_base=0.05,
                         margin_scale=0.25,
                         margin_min=0.02,
                         margin_max=0.3,
                         soft_incorrect_weight=0.2,
                         use_logit_margin=False):  # デフォルトは確率空間(False)を推奨
    """
    Returns: (router_load_loss, router_margin_loss)
    """
    if not hasattr(self.layer_stacks, 'last_router_probs') or self.layer_stacks.last_router_probs is None:
        zero = torch.tensor(0.0, device=self.device)
        return zero, zero

    probs = self.layer_stacks.last_router_probs  # [B, num_buckets]
    device = probs.device
    best_bucket_indices = best_bucket_indices.to(device)
    oracle_gaps = oracle_gaps.to(device)

    num_buckets = probs.shape[-1]

    # 1) Load balancing loss
    chosen_buckets = torch.argmax(probs, dim=-1)
    hard_fractions = F.one_hot(chosen_buckets, num_classes=num_buckets).float().mean(dim=0)
    mean_probs = probs.mean(dim=0)
    router_load_loss = num_buckets * torch.sum(hard_fractions * mean_probs)

    # 2) Target prob / logit と Other max の差分計算
    target_probs = torch.gather(probs, dim=-1, index=best_bucket_indices.unsqueeze(-1)).squeeze(-1)

    masked_probs = probs.clone()
    masked_probs.scatter_(dim=-1, index=best_bucket_indices.unsqueeze(-1), value=-1.0)
    max_other_probs, _ = torch.max(masked_probs, dim=-1)

    if use_logit_margin and hasattr(self.layer_stacks, 'last_router_logits'):
        logits = self.layer_stacks.last_router_logits.to(device)
        target_logits = torch.gather(logits, dim=-1, index=best_bucket_indices.unsqueeze(-1)).squeeze(-1)
        
        masked_logits = logits.clone()
        masked_logits.scatter_(dim=-1, index=best_bucket_indices.unsqueeze(-1), value=-1e9)
        max_other_logits, _ = torch.max(masked_logits, dim=-1)
        
        margin_diff = target_logits - max_other_logits
        
        # Logit 空間用の Target Margin スケール (例: 0.2 ~ 1.2)
        raw_margin_target = 0.2 + 1.0 * torch.sigmoid((oracle_gaps - 0.008) / 0.003)
        margin_target = raw_margin_target.clamp(min=0.1, max=1.5)
    else:
        margin_diff = target_probs - max_other_probs
        
        # 確率空間用の Target Margin スケール (0.05 ~ 0.30)
        raw_margin_target = margin_base + margin_scale * torch.sigmoid((oracle_gaps - 0.008) / 0.003)
        margin_target = raw_margin_target.clamp(min=margin_min, max=margin_max)

    # 3) apply_mask: 正解時 1.0 / 不正解時は確信度に応じたソフトな重み
    pred_bucket_indices = torch.argmax(probs, dim=-1)
    correct_mask = (pred_bucket_indices == best_bucket_indices).float()
    top1_conf = torch.gather(probs, dim=-1, index=pred_bucket_indices.unsqueeze(-1)).squeeze(-1)
    
    apply_mask = correct_mask + (1.0 - correct_mask) * (soft_incorrect_weight * top1_conf)
    apply_mask = apply_mask.clamp(0.0, 1.0)

    # 4) Hinge loss & Weighted average
    raw_margin_loss = F.relu(margin_target - margin_diff)
    weighted_sum = (raw_margin_loss * apply_mask).sum()
    denom = apply_mask.sum().clamp(min=1e-6)
    router_margin_loss = weighted_sum / denom

    return router_load_loss, router_margin_loss


  def _compute_bucket_distill_loss(
      self,
      all_final_outputs,
      pt,
      active_indices,
      oracle_top_k=2,
  ):
      """
      Oracleに近い未選択bucketだけをptへ蒸留する。
      """
  
      all_scorenet = all_final_outputs * self.nnue2score
  
      all_q = (all_scorenet - self.offset1) / self.in_scaling
      all_qm = (-all_scorenet - self.offset2) / self.in_scaling
  
      all_qf = 0.5 * (
          1.0
          + all_q.sigmoid()
          - all_qm.sigmoid()
      )
  
      pt_target = pt.view(-1, 1).detach()
  
      # 各bucketの教師との距離
      errors = torch.abs(all_qf - pt_target)
  
      # Oracle Top-K
      _, oracle_indices = torch.topk(
          errors,
          k=oracle_top_k,
          dim=1,
          largest=False,
      )
  
      num_buckets = all_final_outputs.shape[1]
  
      bucket_ids = torch.arange(
          num_buckets,
          device=all_final_outputs.device,
      ).view(1, -1)
  
      active = active_indices.view(-1, 1)
  
      # Oracle Top-Kに入っているbucket
      oracle_mask = torch.zeros_like(errors, dtype=torch.bool)
  
      oracle_mask.scatter_(
          1,
          oracle_indices,
          True,
      )
  
      # ただし選択済みbucketは除外
      unselected_mask = bucket_ids != active
  
      mask = oracle_mask & unselected_mask
  
      batch_size = all_final_outputs.shape[0]

      if not mask.any():
          distill_loss = all_final_outputs.sum() * 0.0
          valid_count = 0
      else:
          distill_loss = F.smooth_l1_loss(
              all_qf[mask],
              pt_target.expand_as(all_qf)[mask],
              beta=0.01,
              reduction="mean",
          )
          # 少なくとも1つの未選択Oracleバケットが割り当たったサンプル数
          valid_count = mask.any(dim=1).sum().item()

      # ログ表示
      if self.training and (self.global_step % 500 == 0):
          bucket_stats = []
          for i in range(num_buckets):
              m_i = mask[:, i]
              cnt = m_i.sum().item()
              if cnt > 0:
                  # Top-2に入った局面における平均絶対誤差 (Abs Error)
                  avg_err = errors[:, i][m_i].mean().item()
                  bucket_stats.append(f"B{i:02d}: count={cnt:<4d} error={avg_err:.3f}")
              else:
                  bucket_stats.append(f"B{i:02d}: count=0    error=0.000")
  
          print("[BUCKET DISTILL]")
          print(f"Loss: {distill_loss.item():.5f}")
          print(f"Valid: {valid_count} / {batch_size}")
          print("Target Bucket Detail:")
          # 見やすく4バケットずつ改行して表示
          for b in range(0, num_buckets, 4):
              print("  " + " | ".join(bucket_stats[b : b + 4]))

      return distill_loss


  def _prepare_sorted_data(self, pairwise_mask, n_pairwise, pt, qf, score, scorenet, active_indices, material):
      if n_pairwise <= 1:
          return None

      pt_p = pt.view(-1)[pairwise_mask]
      qf_p = qf.view(-1)[pairwise_mask]
      score_p = score.view(-1)[pairwise_mask]
      scorenet_p = scorenet.view(-1)[pairwise_mask]
      lsind_p = active_indices.view(-1)[pairwise_mask]
      material_p = material.view(-1)[pairwise_mask]

      if self.training and self.global_step % 500 == 0:
          print(f"\n[DEBUG RANGE] Step: {self.global_step}")
          print(f"  [material_p] Min: {material_p.min().item():.1f} | Max: {material_p.max().item():.1f} | Mean: {material_p.mean().item():.1f}")
          print(f"  [pt_p]       Min: {pt_p.min().item():.4f} | Max: {pt_p.max().item():.4f}")

      #sort_key = material_p + pt_p
      #sort_key = pt_p

      material_bin = torch.round(material_p / 200.0)
      sort_key = material_bin * 10.0 + pt_p

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

          #tol = torch.where(lsind_true_A < 4, 400, torch.where(lsind_true_A < 8, 200, 100))
          tol = 10000
          valid_pair_mask = (diff_abs > 0.003) & (diff_abs <= 0.10) & ((material_A - material_B).abs() <= tol)

          if valid_pair_mask.any():
              target_direction = torch.sign(pt_A - pt_B)
              num_equal = (target_direction[valid_pair_mask] == 0).sum().item()

              pred_diff = qf_A - qf_B
              pair_weight_curve = torch.sigmoid((diff_abs - 0.005) * 150) * torch.sigmoid((0.05 - diff_abs) * 120)
              scale = 2.0 + 3.0 * torch.exp(-(diff_abs / 0.02)**2)

              direction_loss = -F.logsigmoid(target_direction * pred_diff * scale)

              true_cp_compressed = torch.tanh((cp_true_A - cp_true_B) / 400.0)
              pred_cp_compressed = torch.tanh((cp_pred_A - cp_pred_B) / 400.0)
              value_loss_cp = F.smooth_l1_loss(pred_cp_compressed, true_cp_compressed, reduction="none", beta=0.1)

              raw_pairwise = (direction_loss + 0.6 * value_loss_cp) * pair_weight_curve
              equal_penalty = pred_diff.pow(2) * 1.0
              pairwise_loss_all = torch.where(target_direction != 0, raw_pairwise, equal_penalty)

              total_pairwise_loss += pairwise_loss_all[valid_pair_mask].sum()
              valid_cnt = valid_pair_mask.sum().item()
              total_valid_pairs += valid_cnt

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
      if sorted_data is None:
          return torch.tensor(0.0, device=device), None

      list_size = 6
      temperature = 0.15

      #stride_sizes = [1, 2, 3, 4, 5, 6, 7, 8]
      stride_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 50, 100, 200, 300, 500, 1000, 2000]

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

          lsind_ref = lsind_lists[:, 0]
          #tol = torch.where(lsind_ref < 4, 400, torch.where(lsind_ref < 8, 200, 100))
          tol = 10000
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
      phase = getattr(self.layer_stacks, 'current_phase_for_loss', None)
      if phase is None:
          return 0.0

      p_means = phase.mean(dim=0)
      p_stds  = phase.std(dim=0)

      overall_mean = p_means.mean() 
      mean_penalty = (overall_mean - 0.5) ** 2

      mean_bounds_penalty = torch.mean(
          torch.clamp(p_means - 0.95, min=0) ** 2
          + torch.clamp(0.05 - p_means, min=0) ** 2
      )

      std_penalty = torch.mean(torch.clamp(0.15 - p_stds, min=0) ** 2)

      return mean_penalty + mean_bounds_penalty + std_penalty


  def _compute_ortho_loss(self, threshold=0.2):
        """
        順序に依存しない Soft Orthogonality Loss。
        バケット間の類似度が threshold (デフォルト 0.2) 以下であればペナルティをかけず、
        適度な共通表現（滑らかさ）を残しながら過度な重複を防ぎます。
        """
        count = self.layer_stacks.count
        device = self.device

        bucket_vectors = []
        for i in range(count):
            w_l1   = self.layer_stacks.l1.weight[i * 32 : (i + 1) * 32].reshape(-1)
            w_fd   = self.layer_stacks.fm_diff.weight[i * 64 : (i + 1) * 64].reshape(-1)
            w_fa   = self.layer_stacks.fm_abs.weight[i * 64 : (i + 1) * 64].reshape(-1)
            w_cross= self.layer_stacks.cross_proj.weight[i * 32 : (i + 1) * 32].reshape(-1)
            w_l2   = self.layer_stacks.l2.weight[i * L3 : (i + 1) * L3].reshape(-1)
            w_out  = self.layer_stacks.output.weight[i : i + 1].reshape(-1)

            v_i = torch.cat([w_l1, w_fd, w_fa, w_cross, w_l2, w_out], dim=0)
            bucket_vectors.append(v_i)

        V = torch.stack(bucket_vectors, dim=0)
        V_norm = F.normalize(V, p=2, dim=1)
        G = torch.mm(V_norm, V_norm.t()) # コサイン類似度行列 [count, count]

        # 自分自身との類似度(1.0)を除外するため、対角成分を 0 にしたマスクを作成
        I = torch.eye(count, device=device)
        off_diagonal_sim = G * (1.0 - I)

        # 類似度が threshold (例: 0.2) を超えた部分だけにペナルティを発生させる
        # (0.2 以下なら F.relu により 0 になるためロスは発生しない)
        excess_sim = F.relu(off_diagonal_sim - threshold)
        ortho_loss = torch.mean(excess_sim ** 2)

        return ortho_loss


  def _print_router_bucket_debug(self, best_bucket_indices, pred_bucket_indices, oracle_gaps, router_logits):
        """定期ログ用: 詳細なバケット別 Gap / 精度解析を出力する"""
        with torch.no_grad():
            num_buckets = self.layer_stacks.count

            # --- Router Softmax 確率の計算 ---
            router_probs = F.softmax(router_logits, dim=-1) # [B, num_buckets]

            oracle_counts = torch.bincount(best_bucket_indices, minlength=num_buckets)
            pred_counts   = torch.bincount(pred_bucket_indices, minlength=num_buckets)

            is_correct = (pred_bucket_indices == best_bucket_indices)
            correct_counts = torch.bincount(best_bucket_indices[is_correct], minlength=num_buckets)
            per_bucket_acc = (correct_counts.float() / (oracle_counts.float() + 1e-8) * 100)

            best_gap_means, pred_gap_means, oracle_prob_means = [], [], []
            for b in range(num_buckets):
                b_mask = (best_bucket_indices == b)
                b_gap = oracle_gaps[b_mask].mean().item() if b_mask.any() else float('nan')
                best_gap_means.append(round(b_gap, 6))

                # --- 変更点: * 100 して % 表記（小数第1位まで）にする ---
                b_prob = (router_probs[b_mask, b].mean().item() * 100) if b_mask.any() else float('nan')
                oracle_prob_means.append(round(b_prob, 1))

                p_mask = (pred_bucket_indices == b)
                p_gap = oracle_gaps[p_mask].mean().item() if p_mask.any() else float('nan')
                pred_gap_means.append(round(p_gap, 6))

            pred_cond_acc = []
            for b in range(num_buckets):
                p_mask = (pred_bucket_indices == b)
                correct_when_pred = (best_bucket_indices[p_mask] == b).float().mean().item() * 100 if p_mask.any() else float('nan')
                pred_cond_acc.append(round(correct_when_pred, 2))

            high_gap_thr = 0.01
            high_gap_mask = oracle_gaps > high_gap_thr
            high_gap_acc_per_bucket = []
            for b in range(num_buckets):
                mask = (best_bucket_indices == b) & high_gap_mask
                acc = (pred_bucket_indices[mask] == best_bucket_indices[mask]).float().mean().item() * 100 if mask.any() else float('nan')
                high_gap_acc_per_bucket.append(round(acc, 2))

            pred_gap_correct, pred_gap_wrong = [], []
            for b in range(num_buckets):
                p_mask = (pred_bucket_indices == b)
                if p_mask.any():
                    correct_mask = p_mask & (best_bucket_indices == pred_bucket_indices)
                    wrong_mask = p_mask & (best_bucket_indices != pred_bucket_indices)
                    mean_correct = oracle_gaps[correct_mask].mean().item() if correct_mask.any() else float('nan')
                    mean_wrong = oracle_gaps[wrong_mask].mean().item() if wrong_mask.any() else float('nan')
                else:
                    mean_correct, mean_wrong = float('nan'), float('nan')
                pred_gap_correct.append(round(mean_correct, 6))
                pred_gap_wrong.append(round(mean_wrong, 6))

            conf_mat = torch.zeros((num_buckets, num_buckets), dtype=torch.int64)
            for o, p in zip(best_bucket_indices.tolist(), pred_bucket_indices.tolist()):
                conf_mat[o, p] += 1

        print("\n[BUCKET DISTRIBUTION DEBUG]")
        print(f"  Oracle Counts (正解件数) : {oracle_counts.tolist()}")
        print(f"  Pred Counts   (予測件数) : {pred_counts.tolist()}")
        print(f"  Bucket Accs   (精度 % )  : {[round(a,1) for a in per_bucket_acc.tolist()]}")
        print(f"  Oracle Target Probs (正解確率 %): {oracle_prob_means}")  # <- % 表記
        print(f"  Best Gaps     (正解時Gap): {best_gap_means}")
        print(f"  Pred Gaps     (予測時Gap): {pred_gap_means}")
        print(f"  Pred-Cond Acc (% when Pred=b): {pred_cond_acc}")
        print(f"  HighGap Acc (gap>{high_gap_thr}) : {high_gap_acc_per_bucket}")
        print(f"  Pred Gap Mean (when correct) : {pred_gap_correct}")
        print(f"  Pred Gap Mean (when wrong)   : {pred_gap_wrong}")
        print("\n  Confusion Matrix (rows=Oracle, cols=Pred):")
        for i in range(num_buckets):
            row = conf_mat[i].tolist()
            print(f"    B{i:02d}: {row}")


  def _update_bucket_stats(self, pt, qf, active_indices, loss_type):
      with torch.no_grad():
          loss_per_sample = torch.pow(torch.abs(pt - qf), 2.5).detach().squeeze()

          if not self.training and loss_type == 'val_loss_actual_lambda':
              if not hasattr(self, 'bucket_stats'):
                  self.bucket_stats = {
                      'loss_sum': torch.zeros(12, device=pt.device),
                      'count': torch.zeros(12, device=pt.device)
                  }

              indices = active_indices.squeeze()
              self.bucket_stats['loss_sum'].index_add_(0, indices, loss_per_sample)
              self.bucket_stats['count'].index_add_(0, indices, torch.ones_like(loss_per_sample))

  def _log_debug_info(
        self,
        loss_type, loss, base_loss, pairwise_loss, listwise_loss,
        pt, qf, score, scorenet, active_indices, kif_group_id_flat, actual_lambda,
        pair_metrics, pt_range, router_load_loss, phase_penalty, ortho_loss, router_margin_loss, ema_distill_loss,
        router_ce_loss, bucket_distill_loss, router_acc, gap_mean, gap_median, gap_max
        , gw_mean, gw_median, gw_gt_05, gw_gt_08, router_acc_high
        , weights
    ):

        def _to_float(val):
            if val is None:
                return 0.0
            if hasattr(val, "item"):
                return val.item()
            return float(val)

        mean_base   = _to_float(base_loss)
        mean_pair   = _to_float(pairwise_loss)
        mean_list   = _to_float(listwise_loss)
        mean_router = _to_float(router_load_loss)
        mean_margin = _to_float(router_margin_loss)
        mean_ce     = _to_float(router_ce_loss) # ★ 追加
        mean_acc = _to_float(router_acc) # ★ 追加
        mean_phase  = _to_float(phase_penalty)
        mean_ortho  = _to_float(ortho_loss)
        mean_ema_distill  = _to_float(ema_distill_loss)
        mean_bucket_distill = _to_float(bucket_distill_loss)
        mean_total  = _to_float(loss)

        w_base   = mean_base
        w_pair   = mean_pair   * weights["pairwise"]
        w_list   = mean_list   * weights["listwise"]
        w_phase  = mean_phase  * weights["phase"]
        w_router = mean_router * weights["router_load"]
        w_margin = mean_margin * weights["router_margin"]
        w_ce     = mean_ce     * weights["router_ce"]
        w_ortho  = mean_ortho  * weights["ortho"]
        w_ema_distill = mean_ema_distill * weights["ema_distill"]
        w_bucket_distill = mean_bucket_distill * weights["bucket_distill"]

        if self.training:
            self.log("train/base_loss", mean_base, prog_bar=False)
            self.log("train/pairwise_loss", mean_pair, prog_bar=False)
            self.log("train/listwise_loss", mean_list, prog_bar=False)
            if router_load_loss is not None:
                self.log("train/router_load_loss", mean_router, prog_bar=False)
            if router_margin_loss is not None:
                self.log("train/router_margin_loss", mean_margin, prog_bar=False)
            if router_ce_loss is not None:
                self.log("train/router_ce_loss", mean_ce, prog_bar=False) # ★ 追加
                self.log("train/router_acc", mean_acc, prog_bar=False) # ★ 追加 (Routerの一致率)
            if phase_penalty is not None:
                self.log("train/phase_penalty", mean_phase, prog_bar=False)
            if ortho_loss is not None:
                self.log("train/ortho_loss", mean_ortho, prog_bar=False)
            if ema_distill_loss is not None:
                self.log("train/ema_distill_loss", mean_ema_distill, prog_bar=False)
            if bucket_distill_loss is not None:
                self.log("train/bucket_distill_loss", mean_bucket_distill, prog_bar=False)

        if self.training and (self.global_step % 500 == 0):
            valid_count = pair_metrics['total_valid_pairs']
            max_possible = pair_metrics['max_possible_pairs']

            print(f"\n[DEBUG LOSS] Total: {mean_total:.6f} | "
                  f"Base: {mean_base:.6f} | "
                  f"Pairwise: {mean_pair:.6f} (w: {w_pair:.6f}) | "
                  f"Listwise: {mean_list:.6f} (w: {w_list:.6f}) | "
                  f"Router_Load: {mean_router:.6f} (w: {w_router:.6f}) | "
                  f"R_Margin: {mean_margin:.6f} (w: {w_margin:.6f}) | "
                  f"R_CE: {mean_ce:.6f} (w: {w_ce:.6f}) (Acc: {mean_acc:.2%}) | " # ★ (Acc: xx.xx%) で表示！
                  f"Oracle Gap (Mean: {gap_mean:.6f}, Med: {gap_median:.6f}, Max: {gap_max:.6f}) | " # ★ 追加
                  f"Phase: {mean_phase:.6f} (w: {w_phase:.6f}) | "
                  f"Ortho: {mean_ortho:.6f} (w: {w_ortho:.6f}) | "
                  f"Ema_Distill: {mean_ema_distill:.6f} (w: {w_ema_distill:.6f}) | "
                  f"Bucket_Distill: {mean_bucket_distill:.6f} (w: {w_bucket_distill:.6f}) | "
                  f"(Valid Pairs: {valid_count}/{max_possible})")

            print(f"[GAP WEIGHT DEBUG] "
                  f"GW Mean: {gw_mean:.3f} | Med: {gw_median:.3f} | "
                  f">0.5: {gw_gt_05:.1f}% | >0.8: {gw_gt_08:.1f}% || "
                  f"Acc(All): {router_acc.item()*100:.1f}% | Acc(HighGap): {router_acc_high:.1f}%")


            print(f"  [PT Distribution] Min: {pt.min().item():.4f} | Median: {pt.median().item():.4f} | Max: {pt.max().item():.4f} | Std: {pt.std().item():.4f}")
            
            target_model = getattr(self, "layer_stacks", self)
            if hasattr(target_model, "last_routing_weights") and target_model.last_routing_weights is not None:
                with torch.no_grad():
                    r_counts = target_model.last_routing_weights.sum(dim=0).long()
                    total_samples = pt.size(0)
                    
                    counts_str = " | ".join([f"L{i:02d}: {r_counts[i].item():>5d}" for i in range(target_model.count)])
                    pcts_str   = " | ".join([f"L{i:02d}: {r_counts[i].item() / total_samples:>5.1%}" for i in range(target_model.count)])
                    
                    print(f"  [Router Bucket Counts] {counts_str} (Total: {total_samples})")
                    print(f"  [Router Bucket Share ] {pcts_str}")

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
                        counts_str = " | ".join([f"L{i:02d}: {lsind_counts[i].item()}" for i in range(12)])
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

        if self.training and (self.global_step % 100 == 0) and hasattr(self, "logger") and self.logger is not None:
            self.logger.experiment.add_histogram("Distribution/PT_Target", pt.view(-1), global_step=self.global_step)
            self.logger.experiment.add_histogram("Distribution/QF_Prediction", qf.view(-1), global_step=self.global_step)
            self.logger.experiment.add_histogram("Distribution/score_Target", score.view(-1), global_step=self.global_step)
            self.logger.experiment.add_histogram("Distribution/scorenet", scorenet.view(-1), global_step=self.global_step)

            target_model = getattr(self, "layer_stacks", self)
            if hasattr(target_model, "last_router_probs_log") and target_model.last_router_probs_log is not None:
                self.logger.experiment.add_histogram("Distribution/Router_Probs", target_model.last_router_probs_log, global_step=self.global_step)

            score_flat = score.view(-1)
            scorenet_flat = scorenet.view(-1)
            for i in range(12):
                layer_mask = (active_indices.view(-1) == i)
                if layer_mask.any():
                    self.logger.experiment.add_histogram(f"Distribution_Layer/L{i:02d}_score_Target", score_flat[layer_mask], global_step=self.global_step)
                    self.logger.experiment.add_histogram(f"Distribution_Layer/L{i:02d}_scorenet", scorenet_flat[layer_mask], global_step=self.global_step)

        if loss_type == 'val_loss_actual_lambda' and self.global_step > 1:
            self.log('actual_lambda', actual_lambda)
            self.log("val_loss/base_loss", mean_base, prog_bar=False)
            self.log("val_loss/pairwise_loss", mean_pair, prog_bar=False)
            self.log("val_loss/listwise_loss", mean_list, prog_bar=False)
            if router_load_loss is not None:
                self.log("val_loss/router_load_loss", mean_router, prog_bar=False)
            if router_ce_loss is not None:
                self.log("val_loss/router_ce_loss", mean_ce, prog_bar=False) # ★ 追加

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


  def _log_debug_gpu_info(self):
      if self.training and (self.global_step % 500 == 0):
          print(f"\n[_log_debug_gpu_info] Step: {self.global_step}")
          #print(torch.cuda.memory_summary())
          print("allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
          print("reserved:",  torch.cuda.memory_reserved() / 1024**3, "GB")
          print("max allocated:", torch.cuda.max_memory_allocated() / 1024**3, "GB")

          """
          import gc
          for obj in gc.get_objects():
              try:
                  if torch.is_tensor(obj) and obj.is_cuda:
                      print(
                          obj.shape,
                          obj.dtype,
                          obj.numel() * obj.element_size() / 1024**3,
                          obj.device
                      )
              except:
                  pass
          """

      if self.training and (self.global_step % 500 == 0):
          import gc
          gc.collect()
          torch.cuda.empty_cache()

          print("torch.cuda.empty_cache()!!")
          print("allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
          print("reserved:",  torch.cuda.memory_reserved() / 1024**3, "GB")
          print("max allocated:", torch.cuda.max_memory_allocated() / 1024**3, "GB")


  def print_mem(self, tag):
      if self.training and (self.global_step % 500 == 1):
          mb = torch.cuda.memory_allocated() / (1024**2)
          print(f"[{tag}] Allocated Memory: {mb:.2f} MB")


  def training_step(self, batch, batch_idx):
    return self.step_(batch, batch_idx, 'train_loss')

  def validation_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'val_loss_actual_lambda')
    self.step_(batch, batch_idx, 'val_loss_lambda1.0')
    self.step_(batch, batch_idx, 'val_loss_lambda0.0')
    self.step_(batch, batch_idx, 'val_loss_lambda0.1')
    self.step_(batch, batch_idx, 'val_loss_lambda0.5')
    self.step_(batch, batch_idx, 'val_loss_lambda0.8')

    optimizer = self.optimizers()
    current_lr = optimizer.param_groups[0]['lr']
    self.log('current_lr', current_lr, on_step=False, on_epoch=True)

  def test_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'test_loss')

  def on_validation_epoch_end(self):
      if hasattr(self, 'bucket_stats'):
          s = self.bucket_stats['loss_sum']
          c = self.bucket_stats['count']

          avg_losses = (s / (c + 1e-9)).detach().cpu().numpy()
          self.last_bucket_losses = avg_losses

          self.bucket_stats['loss_sum'].zero_()
          self.bucket_stats['count'].zero_()

  def on_train_batch_end(self, outputs, batch, batch_idx):
    if self.global_step % 100 == 0:
        with torch.no_grad():
            gate_d = self.layer_stacks.last_gate_d
            gate_a = self.layer_stacks.last_gate_a

            sig_d = torch.sigmoid(gate_d)
            sig_a = torch.sigmoid(gate_a)

            eff_open_abs = sig_a.mean() * 100.0
            eff_open_main = (0.5 + 0.5 * sig_d).mean() * 100.0

            self.log("gate/open_rate_abs_to_abs", eff_open_abs)
            self.log("gate/open_rate_diff_to_main", eff_open_main)
            self.log("gate/sharpness_diff", sig_d.var())
            self.log("gate/sharpness_abs", sig_a.var())

            tensorboard = self.logger.experiment
            tensorboard.add_histogram("gate_dist/abs_to_abs_effective", sig_a, self.global_step)
            eff_sig_d = 0.5 + 0.5 * sig_d
            tensorboard.add_histogram("gate_dist/diff_to_main_effective", eff_sig_d, self.global_step)

            pw = self.pair_weights.detach().cpu()
            tensorboard.add_histogram("pair_w/overall_raw", pw, self.global_step)
            pw_softmax = torch.softmax(pw, dim=2) 

            phase_labels = ["Open", "Mid1", "Mid2", "End"]
            for i, label in enumerate(phase_labels):
                tensorboard.add_histogram(f"pair_w_ratio_{label}/mul",  pw_softmax[i, :, 0], self.global_step)
                tensorboard.add_histogram(f"pair_w_ratio_{label}/diff", pw_softmax[i, :, 1], self.global_step)
                tensorboard.add_histogram(f"pair_w_ratio_{label}/sum",  pw_softmax[i, :, 2], self.global_step)

                self.log(f"pair_w_avg_{label}/mul_ratio",  pw_softmax[i, :, 0].mean())
                self.log(f"pair_w_avg_{label}/diff_ratio", pw_softmax[i, :, 1].mean())
                self.log(f"pair_w_avg_{label}/sum_ratio",  pw_softmax[i, :, 2].mean())

            self.log("pair_w_avg_total/mul",  pw_softmax[:, :, 0].mean())
            self.log("pair_w_avg_total/diff", pw_softmax[:, :, 1].mean())
            self.log("pair_w_avg_total/sum",  pw_softmax[:, :, 2].mean())


    """バッチ学習終了後にEMAモデルの重みを更新"""
    if self.training:
        with torch.no_grad():
            decay = 0.9995
            # Student (self) の重みを Teacher (self.ema_model) に EMA更新
            for p_student, p_ema in zip(self.parameters(), self.ema_model.parameters()):
                p_ema.data.mul_(decay).add_(p_student.data, alpha=1.0 - decay)


  def on_fit_start(self):
    if not hasattr(self, 'ema_model') or self.ema_model is None:
        self.ema_model = NNUE(
            feature_set=self.feature_set,
            start_lambda=self.start_lambda,
            end_lambda=self.end_lambda,
            max_epoch=self.max_epoch,
            gamma=self.gamma,
            lr=self.lr,
            epoch_size=self.epoch_size,
            batch_size=self.batch_size,
            in_scaling=self.in_scaling,
            out_scaling=self.out_scaling,
            offset=self.offset,
            offset1=self.offset1,
            offset2=self.offset2,
            adjust_loss=self.adjust_loss
        ).to(self.device)
        
        # strict=False を追加して不一致キーを無視
        self.ema_model.load_state_dict(self.state_dict(), strict=False)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False


  def on_load_checkpoint(self, checkpoint):
    """過去の余分な EMA キーを安全に削除"""
    state_dict = checkpoint.get("state_dict", {})
    for k in [k for k in state_dict.keys() if k.startswith("ema_model.")]:
      del state_dict[k]



  def configure_optimizers(self):
    # =========================================================
    # 【一時的】Routerの重みとバイアスを再初期化
    # =========================================================
    """
    with torch.no_grad():
      nn.init.normal_(self.layer_stacks.router.weight, std=0.01)
      nn.init.constant_(self.layer_stacks.router.bias, 0.0)
      print("Routerの重みとバイアスを再初期化")
    """
    # =========================================================


    LR = self.lr

    train_params = [
      {'params' : [self.input.weight], 'lr' : LR * 1.0, 'weight_decay': 0.0 }, 
      {'params' : [self.input.bias]  , 'lr' : LR * 1.0, 'weight_decay': 0.0 }, 
      {'params' : [self.input.v]     , 'lr' : LR * 1.5, 'weight_decay': 0.0 }, 
      {'params' : [self.pair_weights], 'lr' : LR * 1.0, 'weight_decay': 1e-5 }, 
      {'params' : [self.layer_stacks.router.weight]  , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.router.bias]    , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.l1.weight]      , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.l1.bias]        , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.l1_fact.weight] , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.l1_fact.bias]   , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.fm_diff.weight] , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.fm_diff.bias]   , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.fm_abs.weight]  , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.fm_abs.bias]    , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.cross_proj.weight], 'lr' : LR * 1.0, 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.cross_proj.bias]  , 'lr' : LR * 1.0, 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.q_proj.weight]  , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.k_proj.weight]  , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.v_proj.weight]  , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.lca_temp]      , 'lr' : LR * 0.1 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.phase_proj.weight], 'lr': LR * 1.0, 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.phase_proj.bias]  , 'lr': LR * 1.0, 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.l2.weight]      , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.l2.bias]        , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.output.weight]  , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.output.bias]    , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
      {'params' : [self.layer_stacks.blend]          , 'lr' : LR * 1.0 , 'weight_decay': 0.0 },
    ]


    """
    optimizer = ranger.Ranger(train_params, lr=LR, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)

    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
        }
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

