"""Canonical HalfKA_hm2/no-DG SFNN baseline for Experiment 85.

This module is deliberately independent from model.NNUE.  Shared training
semantics are small, explicit functions here; no complex Router/FM/Cross/LCA
state can accidentally enter a simple checkpoint.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

import model as complex_model

SIMPLE_SCHEMA_VERSION = 2
ARCHITECTURE_TYPE = "halfka_hm2_simple"
FEATURE_NAME = "HalfKA_HM2_NoDG"
FT_INPUTS = 73_305
FT_WIDTH = 1_536
LAYER_STACKS = 9
NNUE_TO_SCORE = 127.0 * 64.0 / 16.0
HIDDEN_WEIGHT_SCALE = 64.0
MAX_HIDDEN_WEIGHT = 127.0 / HIDDEN_WEIGHT_SCALE
FT_SERIALIZER_SCALE = 127.0
BUCKET_STAT_NAMES = (
    "count", "weight", "loss", "prob_mae", "cp_mae",
    "pt_sum", "pt_sq", "pf_sum", "pf_sq", "qf_sum", "qf_sq",
    "error_sum", "error_abs", "error_sq",
    "ply_sum", "ply_sq", "material_sum", "material_sq",
    "score_sum", "score_sq", "pred_sum", "pred_sq")


class SimpleFeatureTransformer(nn.Module):
    def __init__(self, num_inputs: int = FT_INPUTS, width: int = FT_WIDTH):
        super().__init__()
        sigma = math.sqrt(1.0 / num_inputs)
        self.weight = nn.Parameter(torch.empty(num_inputs, width))
        self.bias = nn.Parameter(torch.empty(width))
        nn.init.uniform_(self.weight, -sigma, sigma)
        nn.init.uniform_(self.bias, -sigma, sigma)

    def forward(self, white_indices, white_values, black_indices, black_values):
        # The complex model's generated CuPy kernel is tuned and cached for its
        # 1280-wide FT.  Generating the first 1536-wide kernel takes minutes on
        # the supported Windows toolchain.  embedding_bag provides the same
        # sparse weighted row sum using a native PyTorch kernel, without a
        # multi-gigabyte [batch, active, width] intermediate.
        def accumulate(indices, values):
            batch, active = indices.shape
            valid = indices >= 0
            safe_indices = indices.masked_fill(~valid, 0).reshape(-1).long()
            safe_values = values.masked_fill(~valid, 0.0).reshape(-1)
            offsets = torch.arange(
                0, batch * active, active, dtype=torch.long,
                device=indices.device)
            return F.embedding_bag(
                safe_indices, self.weight, offsets,
                per_sample_weights=safe_values, mode="sum") + self.bias

        return (accumulate(white_indices, white_values),
                accumulate(black_indices, black_values))


class SimpleStack(nn.Module):
    """1536 -> 16; (sqr15, relu15) -> 32 -> 1 + shortcut."""

    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(FT_WIDTH, 16)
        self.fc1 = nn.Linear(30, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, collect_diagnostics=False):
        h0 = self.fc0(x)
        hidden_pre = h0[:, :15]
        # Canonical SFNN squares the raw affine output before clipping.
        hidden2 = torch.clamp(
            hidden_pre.square() * (127.0 / 128.0), 0.0, 1.0)
        hidden = torch.clamp(hidden_pre, 0.0, 1.0)
        h1 = torch.clamp(
            self.fc1(torch.cat((hidden2, hidden), dim=1)), 0.0, 1.0)
        deep = self.fc2(h1)
        shortcut = h0[:, 15:16]
        output = deep + shortcut
        if collect_diagnostics:
            return output, {
                "fc0_pre": h0[:, :15],
                "clipped": hidden,
                "squared": hidden2,
                "fc1_activation": h1,
                "deep": deep,
                "shortcut": shortcut,
                "final": output,
            }
        return output


class SimpleHalfKAHM2NNUE(pl.LightningModule):
    architecture_type = ARCHITECTURE_TYPE

    def __init__(
        self,
        feature_set,
        start_lambda=1.0,
        end_lambda=1.0,
        max_epoch=800,
        lr=1e-3,
        gamma=0.992,
        in_scaling=380.0,
        out_scaling=380.0,
        offset=0.0,
        offset1=None,
        offset2=None,
        adjust_loss=0.0,
        epoch_size=50_000_000,
        batch_size=16_384,
        simple_debug_log_interval=500,
        **unused,
    ):
        super().__init__()
        if feature_set.name != FEATURE_NAME or feature_set.num_features != FT_INPUTS:
            raise ValueError(
                f"simple architecture requires {FEATURE_NAME}/{FT_INPUTS}, got "
                f"{feature_set.name}/{feature_set.num_features}")
        self.feature_set = feature_set
        self.input = SimpleFeatureTransformer()
        self.layer_stacks = nn.ModuleList(SimpleStack() for _ in range(LAYER_STACKS))
        self.start_lambda = float(start_lambda)
        self.end_lambda = float(end_lambda)
        self.max_epoch = int(max_epoch)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.in_scaling = float(in_scaling)
        self.out_scaling = float(out_scaling)
        self.offset = float(offset)
        self.offset1 = float(offset if offset1 is None else offset1)
        self.offset2 = float(offset if offset2 is None else offset2)
        self.adjust_loss = float(adjust_loss)
        self.epoch_size = int(epoch_size)
        self.batch_size = int(batch_size)
        self.simple_debug_log_interval = int(simple_debug_log_interval)
        self._simple_debug_capture = False
        self._simple_debug_snapshot = None
        self._simple_debug_touched_rows = None
        self._simple_debug_ft_before = None
        self._simple_debug_layer_stats = None
        self._simple_clip_stats_pending = None
        self._simple_debug_display_step = None
        self.nnue2score = NNUE_TO_SCORE
        self.weight_clipping = [
            {
                "name": "FC0",
                "params": [stack.fc0.weight for stack in self.layer_stacks],
                "min_weight": -MAX_HIDDEN_WEIGHT,
                "max_weight": MAX_HIDDEN_WEIGHT,
            },
            {
                "name": "FC1",
                "params": [stack.fc1.weight for stack in self.layer_stacks],
                "min_weight": -MAX_HIDDEN_WEIGHT,
                "max_weight": MAX_HIDDEN_WEIGHT,
            },
            {
                "name": "Output",
                "params": [stack.fc2.weight for stack in self.layer_stacks],
                "min_weight": -MAX_HIDDEN_WEIGHT,
                "max_weight": MAX_HIDDEN_WEIGHT,
            },
        ]
        # The ranking objectives are architecture-independent.  Reuse the
        # production implementation verbatim, but allow the full gradient to
        # reach the simple FT because there is no complex-path gradient
        # balancing to preserve here.
        self.pairwise_ft_grad_scale = 1.0
        self.listwise_ft_grad_scale = 1.0
        self.ranking_disagreement_weight = 1.0
        self.save_hyperparameters(ignore=("feature_set", "unused"))

    @staticmethod
    def architecture_metadata(transplant_source=None, mapping_version=None):
        return {
            "architecture_type": ARCHITECTURE_TYPE,
            "feature": FEATURE_NAME,
            "ft_width": FT_WIDTH,
            "layer_stack_count": LAYER_STACKS,
            "bucket_scheme": "k3k3",
            "distinguish_golds": False,
            "long_effect_required": False,
            "simple_schema_version": SIMPLE_SCHEMA_VERSION,
            "transplant_source": transplant_source,
            "transplant_mapping_version": mapping_version,
        }

    def set_feature_set(self, feature_set):
        if feature_set.name != FEATURE_NAME:
            raise ValueError(f"simple checkpoint requires {FEATURE_NAME}")
        self.feature_set = feature_set

    def forward(self, us, them, white_indices, white_values, black_indices,
                black_values, layer_stack_indices, **unused):
        tw, tb = self.input(
            white_indices, white_values, black_indices, black_values)
        us = us.view(-1, 1)
        them = them.view(-1, 1)
        # The loader exposes BLACK in white_* and WHITE in black_*; us/them
        # select the side-to-move ordering exactly as the complex model does.
        # Each 1536-wide accumulator contains two 768-wide factor lanes.
        # Element-wise multiplication emits 768 values per perspective, then
        # side-to-move ordering concatenates friend and enemy perspectives.
        def ewm(t):
            a, b = t[:, :FT_WIDTH // 2], t[:, FT_WIDTH // 2:]
            return (torch.clamp(a, 0.0, 2.0)
                    * torch.clamp(b, 0.0, 2.0)
                    * (127.0 / 128.0))
        ew, eb = ewm(tw), ewm(tb)
        transformed = us * torch.cat((ew, eb), dim=1) \
            + them * torch.cat((eb, ew), dim=1)
        buckets = layer_stack_indices.view(-1).long()
        if torch.any((buckets < 0) | (buckets >= LAYER_STACKS)):
            raise RuntimeError("HalfKA_hm2 simple bucket must be in [0,8]")
        collect = bool(self.training and self._simple_debug_capture)
        out = transformed.new_empty((transformed.shape[0], 1))
        diagnostic_parts = {} if collect else None
        for bucket, stack in enumerate(self.layer_stacks):
            mask = buckets == bucket
            if mask.any():
                if collect:
                    bucket_out, bucket_diagnostics = stack(
                        transformed[mask], collect_diagnostics=True)
                    out[mask] = bucket_out
                    for name, value in bucket_diagnostics.items():
                        diagnostic_parts.setdefault(name, []).append(value)
                else:
                    out[mask] = stack(transformed[mask])
        if collect:
            activation_tensors = {
                name: torch.cat(parts, dim=0)
                for name, parts in diagnostic_parts.items()
            }
            self._simple_debug_snapshot = {
                "activations": self._activation_snapshot(
                    transformed, activation_tensors),
            }
        return out

    @staticmethod
    def _tensor_summary(value, activation=False, high_source=None):
        value = value.detach().float()
        result = {
            "mean": value.mean(),
            "std": value.std(unbiased=False),
            "min": value.min(),
            "max": value.max(),
        }
        if activation:
            result["zero_pct"] = (value == 0).float().mean() * 100.0
            high_value = value if high_source is None else high_source.detach()
            result["high_pct"] = (high_value >= 1.0).float().mean() * 100.0
        return result

    def _activation_snapshot(self, transformed, tensors):
        deep = tensors["deep"].detach().float()
        shortcut = tensors["shortcut"].detach().float()
        final = tensors["final"].detach().float()
        denominator = deep.abs() + shortcut.abs()
        valid = denominator > 0
        deep_share = torch.where(
            valid, deep.abs() / denominator.clamp_min(1e-12),
            torch.zeros_like(denominator))
        shortcut_share = torch.where(
            valid, shortcut.abs() / denominator.clamp_min(1e-12),
            torch.zeros_like(denominator))
        return {
            "ft": self._tensor_summary(transformed),
            "fc0_pre": self._tensor_summary(tensors["fc0_pre"]),
            "clipped": self._tensor_summary(
                tensors["clipped"], activation=True),
            "squared": self._tensor_summary(
                tensors["squared"], activation=True,
                high_source=tensors["clipped"]),
            "fc1": self._tensor_summary(
                tensors["fc1_activation"], activation=True),
            "shortcut": self._tensor_summary(shortcut),
            "output": {
                "deep_abs_mean": deep.abs().mean(),
                "shortcut_abs_mean": shortcut.abs().mean(),
                "final_mean": final.mean(),
                "final_abs_mean": final.abs().mean(),
                "final_min": final.min(),
                "final_max": final.max(),
                "deep_share": deep_share.mean(),
                "shortcut_share": shortcut_share.mean(),
            },
        }

    def _lambda(self):
        progress = min(max(self.current_epoch / max(self.max_epoch, 1), 0.0), 1.0)
        return self.start_lambda + (self.end_lambda - self.start_lambda) * progress

    # Shared, architecture-independent production ranking implementation.
    # Router/FM/LCA/Phase losses remain absent from this model.
    _prepare_sorted_data = complex_model.NNUE._prepare_sorted_data
    _compute_pairwise_loss = complex_model.NNUE._compute_pairwise_loss
    _compute_listwise_loss = complex_model.NNUE._compute_listwise_loss

    def _step(self, batch, stage):
        (us, them, wi, wv, bi, bv, outcome, score, bucket, material,
         group, ply, *optional) = batch
        pred_cp = self(us, them, wi, wv, bi, bv, bucket).view(-1) * self.nnue2score
        score = score.view(-1)
        outcome = outcome.view(-1)
        group = group.view(-1)
        material = material.view(-1)
        ply = ply.view(-1).float()
        ranking_score = optional[0].view(-1) if optional else score
        q = (pred_cp - self.offset1) / self.in_scaling
        qm = (-pred_cp - self.offset2) / self.in_scaling
        qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())
        p = (score - self.offset1) / self.out_scaling
        pm = (-score - self.offset2) / self.out_scaling
        pf = 0.5 * (1.0 + p.sigmoid() - pm.sigmoid())
        target = pf * self._lambda() + outcome * (1.0 - self._lambda())
        ranking_p = (ranking_score - self.offset1) / self.out_scaling
        ranking_pm = (-ranking_score - self.offset2) / self.out_scaling
        ranking_pf = 0.5 * (1.0 + ranking_p.sigmoid() - ranking_pm.sigmoid())
        ranking_target = (ranking_pf * self._lambda()
                          + outcome * (1.0 - self._lambda()))
        mask = (group == 1) | (group == 2)
        err = (target - qf).abs().pow(2.5)
        weights = 1.0 + 0.5 * torch.exp(-(pf - 0.5).abs() / 0.15)
        err = err * (1.0 + self.adjust_loss * (qf > target))
        base_loss = (err[mask] * weights[mask]).sum() / weights[mask].sum().clamp_min(1e-9)

        ranking_indices = torch.nonzero(group == 3, as_tuple=False).flatten()
        sorted_data = self._prepare_sorted_data(
            ranking_indices, target, qf, score, ranking_target,
            ranking_score, pred_cp, bucket.view(-1).long(), material, ply)
        pair_loss, pair_metrics = self._compute_pairwise_loss(
            sorted_data, ranking_indices.numel(), pred_cp.device,
            collect_metrics=(not self.training or self._simple_debug_capture))
        listwise_loss, pt_range = self._compute_listwise_loss(
            sorted_data, ranking_indices.numel(), pred_cp.device)
        pair_acc = pred_cp.new_tensor(float("nan"))
        if pair_metrics["all_pred_diffs"]:
            diffs = torch.cat(pair_metrics["all_pred_diffs"])
            directions = torch.cat(pair_metrics["all_target_directions"])
            non_equal = directions != 0
            if non_equal.any():
                pair_acc = ((diffs[non_equal] > 0)
                            == (directions[non_equal] > 0)).float().mean()
        total = base_loss + 0.01 * pair_loss + 0.01 * listwise_loss

        if stage == "val":
            # Keep a stable score-only validation objective even when a future
            # run uses a win-rate blend (start/end lambda != 1).  Reuse the
            # already computed forward result; only the lightweight ranking
            # objectives are evaluated again with lambda fixed to 1.0.
            lambda1_target = pf
            lambda1_err = (lambda1_target - qf).abs().pow(2.5)
            lambda1_err = lambda1_err * (
                1.0 + self.adjust_loss * (qf > lambda1_target))
            lambda1_base = (
                (lambda1_err[mask] * weights[mask]).sum()
                / weights[mask].sum().clamp_min(1e-9))
            lambda1_sorted = self._prepare_sorted_data(
                ranking_indices, lambda1_target, qf, score, ranking_pf,
                ranking_score, pred_cp, bucket.view(-1).long(), material, ply)
            lambda1_pair, _ = self._compute_pairwise_loss(
                lambda1_sorted, ranking_indices.numel(), pred_cp.device,
                collect_metrics=False)
            lambda1_listwise, _ = self._compute_listwise_loss(
                lambda1_sorted, ranking_indices.numel(), pred_cp.device)
            lambda1_total = (
                lambda1_base + 0.01 * lambda1_pair
                + 0.01 * lambda1_listwise)
            self.log(
                "val_loss_lambda1.0", lambda1_total,
                on_step=False, on_epoch=True, batch_size=score.numel())

        if (stage == "train"
                and int(getattr(self, "simple_debug_log_interval", 500)) > 0):
            # Accumulate stable score-only diagnostics over 500 training
            # steps.  This remains stdout-only and synchronizes the GPU only
            # at the print boundary.
            with torch.no_grad():
                lambda1_err = (pf - qf).abs().pow(2.5)
                lambda1_err = lambda1_err * (
                    1.0 + self.adjust_loss * (qf > pf))
                self._accumulate_training_bucket_stats(
                    bucket.view(-1).long()[mask],
                    weights[mask],
                    lambda1_err[mask] * weights[mask],
                    target[mask], pf[mask], qf[mask],
                    (pf - qf).abs()[mask],
                    (pred_cp - score).abs()[mask],
                    ply[mask], material[mask], score[mask], pred_cp[mask])

            if self._simple_debug_capture:
                self._capture_current_batch_diagnostics(
                    wi, wv, bi, bv, target, pf, qf, score, pred_cp,
                    material, ply, group, total, base_loss, pair_loss,
                    listwise_loss, pair_metrics, pt_range)
        self.log(
            f"{stage}/total_loss",
            total,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=stage == "train",
            batch_size=score.numel())
        self.log(f"{stage}/base_loss", base_loss, on_step=False, on_epoch=True,
                 batch_size=score.numel())
        self.log(f"{stage}/pairwise_loss", pair_loss, on_step=False, on_epoch=True,
                 batch_size=score.numel())
        self.log(f"{stage}/listwise_loss", listwise_loss, on_step=False, on_epoch=True,
                 batch_size=score.numel())
        self.log(f"{stage}/value_mae_cp", (pred_cp - score).abs().mean(),
                 on_step=False, on_epoch=True, batch_size=score.numel())
        if torch.isfinite(pair_acc):
            self.log(f"{stage}/pair_accuracy", pair_acc, on_step=False, on_epoch=True,
                     batch_size=score.numel())
        return total

    def training_step(self, batch, batch_idx):
        interval = int(getattr(self, "simple_debug_log_interval", 500))
        display_step = int(self.global_step) + 1
        self._simple_debug_capture = (
            interval > 0 and display_step % interval == 0)
        self._simple_debug_display_step = (
            display_step if self._simple_debug_capture else None)
        self._clip_simple_weights(capture=self._simple_debug_capture)
        return self._step(batch, "train")

    @torch.no_grad()
    def _clip_simple_weights(self, capture=False):
        """Apply the dense int8 serializer range before the forward pass."""
        captured = {}
        for group in self.weight_clipping:
            low = float(group["min_weight"])
            high = float(group["max_weight"])
            if capture:
                flat = torch.cat(
                    [p.detach().reshape(-1) for p in group["params"]])
                outside_low = flat < low
                outside_high = flat > high
            for parameter in group["params"]:
                parameter.clamp_(low, high)
            if capture:
                clipped = flat.clamp(low, high)
                eps = 0.5 / HIDDEN_WEIGHT_SCALE
                captured[group["name"]] = {
                    "w_min": clipped.min(),
                    "w_max": clipped.max(),
                    "clip_low_pct": outside_low.float().mean() * 100.0,
                    "clip_high_pct": outside_high.float().mean() * 100.0,
                    "boundary_low": (clipped <= low + eps).sum(),
                    "boundary_high": (clipped >= high - eps).sum(),
                    "elements": clipped.numel(),
                }
        if capture:
            # FT uses int16 at scale 127. Upstream does not clamp it to the
            # dense int8 range, so only report serializer-bound violations.
            ft_limit = 32767.0 / FT_SERIALIZER_SCALE
            ft = self.input.weight.detach()
            captured["FT(int16 contract)"] = {
                "w_min": ft.min(),
                "w_max": ft.max(),
                "clip_low_pct": (ft < -ft_limit).float().mean() * 100.0,
                "clip_high_pct": (ft > ft_limit).float().mean() * 100.0,
                "boundary_low": (ft <= -ft_limit).sum(),
                "boundary_high": (ft >= ft_limit).sum(),
                "elements": ft.numel(),
            }
            self._simple_clip_stats_pending = captured

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def _new_bucket_stats(self, device):
        return torch.zeros(
            (LAYER_STACKS, len(BUCKET_STAT_NAMES)),
            device=device, dtype=torch.float32)

    def _accumulate_training_bucket_stats(
            self, bucket, weight, loss, pt, pf, qf, prob_mae, cp_mae,
            ply, material, score, prediction):
        stats = getattr(self, "_simple_train_bucket_stats", None)
        if stats is None or stats.device != bucket.device:
            stats = self._new_bucket_stats(bucket.device)
            self._simple_train_bucket_stats = stats
            self._simple_train_bucket_steps = 0
        self._simple_train_bucket_steps += 1
        error = qf - pt
        # One compact scatter replaces one index_add kernel per statistic.
        # Float32 is sufficient for at most one logging interval (normally
        # 500 * 16,384 samples) and materially reduces diagnostic overhead.
        rows = torch.stack((
            torch.ones_like(prob_mae), weight, loss, prob_mae, cp_mae,
            pt, pt.square(), pf, pf.square(), qf, qf.square(),
            error, error.abs(), error.square(),
            ply, ply.square(), material, material.square(),
            score, score.square(), prediction, prediction.square()), dim=1)
        stats.index_add_(0, bucket, rows.float())

    def _capture_current_batch_diagnostics(
            self, wi, wv, bi, bv, pt, pf, qf, score, pred_cp,
            material, ply, group, total, base_loss, pair_loss,
            listwise_loss, pair_metrics, pt_range):
        snapshot = self._simple_debug_snapshot or {}
        with torch.no_grad():
            white_active = (wi >= 0) & (wv != 0)
            black_active = (bi >= 0) & (bv != 0)
            active_indices = torch.cat((
                wi[white_active].long(), bi[black_active].long()))
            unique_rows = torch.unique(active_indices)
            occurrences = int(active_indices.numel())
            perspectives = max(int(wi.shape[0]) * 2, 1)
            snapshot["features"] = {
                "active_avg": occurrences / perspectives,
                "unique": int(unique_rows.numel()),
                "unique_ratio": unique_rows.numel() / FT_INPUTS,
                "occurrences": occurrences,
                "occurrences_per_unique": (
                    occurrences / max(int(unique_rows.numel()), 1)),
            }
            self._simple_debug_touched_rows = unique_rows
            snapshot["loss"] = {
                "total": total.detach(),
                "base": base_loss.detach(),
                "pairwise": pair_loss.detach(),
                "listwise": listwise_loss.detach(),
                "valid_pairs": int(pair_metrics["total_valid_pairs"]),
                "active_listwise_groups": (
                    0 if pt_range is None else int(pt_range.numel())),
                "actual_lambda": float(self._lambda()),
            }
            snapshot["distribution"] = {
                "pt": pt.detach(), "pf": pf.detach(), "qf": qf.detach(),
                "score": score.detach(), "pred": pred_cp.detach(),
                "material": material.detach(), "ply": ply.detach(),
                "group": group.detach(),
            }
            snapshot["pair_metrics"] = pair_metrics
            snapshot["pt_range"] = (
                None if pt_range is None else pt_range.detach())
        self._simple_debug_snapshot = snapshot

    def _print_training_bucket_stats(self, display_step):
        stats = self._simple_train_bucket_stats
        host_tensor = stats.detach().cpu().double()
        host = {
            name: host_tensor[:, index]
            for index, name in enumerate(BUCKET_STAT_NAMES)
        }
        accumulated_steps = int(self._simple_train_bucket_steps)
        print(
            f"[Simple Bucket Detailed Stats](Step {display_step}, "
            f"last {accumulated_steps} steps aggregate)")
        print("  Bucket |      count | lambda1 loss | prob MAE |    cp MAE")
        print("  " + "-" * 61)
        total_samples = max(int(host["count"].sum().item()), 1)
        for index in range(LAYER_STACKS):
            count = int(host["count"][index].item())
            if count == 0:
                print(f"  B{index:02d}    | {count:10d} |          N/A |      N/A |       N/A")
                continue
            loss = (host["loss"][index] / host["weight"][index]).item()
            prob_mae = (host["prob_mae"][index] / count).item()
            cp_mae = (host["cp_mae"][index] / count).item()
            print(
                f"  B{index:02d}    | {count:10d} | {loss:12.8f} | "
                f"{prob_mae:8.6f} | {cp_mae:9.3f}")
            def mean_std(name):
                mean = (host[f"{name}_sum"][index] / count).item()
                variance = max(
                    (host[f"{name}_sq"][index] / count).item()
                    - mean * mean, 0.0)
                return mean, math.sqrt(variance)
            pt_mean, pt_std = mean_std("pt")
            pf_mean, pf_std = mean_std("pf")
            qf_mean, qf_std = mean_std("qf")
            ply_mean, ply_std = mean_std("ply")
            material_mean, material_std = mean_std("material")
            score_mean, score_std = mean_std("score")
            pred_mean, pred_std = mean_std("pred")
            error_mean = (host["error_sum"][index] / count).item()
            error_abs = (host["error_abs"][index] / count).item()
            error_rmse = math.sqrt(
                (host["error_sq"][index] / count).item())
            print(
                f"         share={count / total_samples:7.3%} | "
                f"pt={pt_mean:.4f}+/-{pt_std:.4f} "
                f"pf={pf_mean:.4f}+/-{pf_std:.4f} "
                f"qf={qf_mean:.4f}+/-{qf_std:.4f}")
            print(
                f"         qf-pt={error_mean:+.5f} "
                f"|qf-pt|={error_abs:.5f} rmse={error_rmse:.5f} | "
                f"ply={ply_mean:.1f}+/-{ply_std:.1f} "
                f"material={material_mean:.1f}+/-{material_std:.1f}")
            print(
                f"         teacher_cp={score_mean:+.1f}+/-{score_std:.1f} | "
                f"pred_cp={pred_mean:+.1f}+/-{pred_std:.1f}")
        total_count = int(host["count"].sum().item())
        if total_count:
            total_loss = (host["loss"].sum() / host["weight"].sum()).item()
            total_prob = (host["prob_mae"].sum() / total_count).item()
            total_cp = (host["cp_mae"].sum() / total_count).item()
            print("  " + "-" * 61)
            print(
                f"  ALL    | {total_count:10d} | {total_loss:12.8f} | "
                f"{total_prob:8.6f} | {total_cp:9.3f}")
        stats.zero_()
        self._simple_train_bucket_steps = 0

    @staticmethod
    def _layer_stat_row(weight, bias, weight_grad=None, bias_grad=None,
                        active=None):
        weight = weight.detach().float().reshape(-1)
        bias = bias.detach().float().reshape(-1)
        gradients = []
        if weight_grad is not None:
            gradients.append(weight_grad.detach().float().reshape(-1))
        if bias_grad is not None:
            gradients.append(bias_grad.detach().float().reshape(-1))
        grad_mean = (
            torch.cat(gradients).abs().mean()
            if gradients else weight.new_zeros(()))
        return {
            "grad_mean": grad_mean,
            "active": int(weight.numel() + bias.numel()
                          if active is None else active),
            "w_mean": weight.mean(), "w_min": weight.min(),
            "w_max": weight.max(), "w_std": weight.std(unbiased=False),
            "b_mean": bias.mean(), "b_min": bias.min(),
            "b_max": bias.max(), "b_std": bias.std(unbiased=False),
        }

    def on_before_optimizer_step(self, optimizer):
        if not self._simple_debug_capture:
            return
        rows = self._simple_debug_touched_rows
        if rows is None or rows.numel() == 0:
            return
        with torch.no_grad():
            ft_weight = self.input.weight.index_select(0, rows)
            self._simple_debug_ft_before = ft_weight.detach().clone()
            ft_grad = (
                None if self.input.weight.grad is None
                else self.input.weight.grad.index_select(0, rows))
            layer_stats = {
                "FT_HalfKA_HM2": self._layer_stat_row(
                    ft_weight, self.input.bias, ft_grad,
                    self.input.bias.grad,
                    active=int(ft_weight.numel() + self.input.bias.numel()))
            }
            for label, attribute in (
                    ("FC0_1536x16", "fc0"),
                    ("FC1_30x32", "fc1"),
                    ("Output_32x1", "fc2")):
                layers = [getattr(stack, attribute)
                          for stack in self.layer_stacks]
                weights = torch.cat([layer.weight.detach().reshape(-1)
                                     for layer in layers])
                biases = torch.cat([layer.bias.detach().reshape(-1)
                                    for layer in layers])
                weight_grads = [
                    layer.weight.grad.detach().reshape(-1)
                    for layer in layers if layer.weight.grad is not None]
                bias_grads = [
                    layer.bias.grad.detach().reshape(-1)
                    for layer in layers if layer.bias.grad is not None]
                layer_stats[label] = self._layer_stat_row(
                    weights, biases,
                    torch.cat(weight_grads) if weight_grads else None,
                    torch.cat(bias_grads) if bias_grads else None)
            self._simple_debug_layer_stats = layer_stats
            snapshot = self._simple_debug_snapshot
            snapshot["ft_pre_step"] = {
                "grad_norm": (
                    ft_grad.float().norm() if ft_grad is not None
                    else ft_weight.new_zeros(())),
                "weight_norm": ft_weight.float().norm(),
                "touched": int(rows.numel()),
            }

    def _finish_ft_update_stats(self):
        rows = self._simple_debug_touched_rows
        before = self._simple_debug_ft_before
        if rows is None or before is None:
            return
        with torch.no_grad():
            after = self.input.weight.index_select(0, rows).detach()
            update = after.float() - before.float()
            before_float = before.float()
            row_relative = (
                update.norm(dim=1)
                / before_float.norm(dim=1).clamp_min(1e-12))
            snapshot = self._simple_debug_snapshot["ft_pre_step"]
            snapshot.update({
                "update_rms": update.square().mean().sqrt(),
                "update_max": update.abs().max(),
                "relative_update": (
                    update.norm() / before_float.norm().clamp_min(1e-12)),
                "row_mean": row_relative.mean(),
                "row_median": row_relative.median(),
                "row_p90": torch.quantile(row_relative, 0.90),
                "row_p99": torch.quantile(row_relative, 0.99),
            })

    @staticmethod
    def _format_summary(name, stats, activation=False):
        text = (
            f"  {name:<20}: mean={stats['mean']:+.6e} "
            f"std={stats['std']:.6e} min={stats['min']:+.6e} "
            f"max={stats['max']:+.6e}")
        if activation:
            text += (f" zero={stats['zero_pct']:.2f}%"
                     f" high={stats['high_pct']:.2f}%")
        return text

    def _print_current_batch_diagnostics(self, display_step):
        snapshot = self._simple_debug_snapshot
        if not snapshot:
            return

        # Transfer only compact scalar summaries and diagnostic vectors at the
        # 500-step boundary. No host synchronization occurs on ordinary steps.
        def host_scalar(value):
            return float(value.detach().cpu()) if torch.is_tensor(value) else value

        feature = snapshot["features"]
        print(f"[Simple Feature Stats](Step {display_step}, current batch snapshot)")
        print(
            f"  Features | ActiveAvg: {feature['active_avg']:.2f} "
            f"(per position-perspective) | Unique: {feature['unique']:,} / "
            f"{FT_INPUTS:,} ({feature['unique_ratio']:.2%})")
        print(
            f"  occurrences={feature['occurrences']:,} | "
            f"occurrences/unique={feature['occurrences_per_unique']:.2f}")

        activations = {
            name: {key: host_scalar(value) for key, value in values.items()}
            for name, values in snapshot["activations"].items()
            if name != "output"
        }
        print(f"[Simple Network Activations](Step {display_step}, current batch snapshot)")
        print(self._format_summary("FT output", activations["ft"]))
        print(self._format_summary("FC0 hidden pre-act", activations["fc0_pre"]))
        print(self._format_summary("ClippedReLU", activations["clipped"], True))
        print(self._format_summary("SqrClippedReLU", activations["squared"], True))
        print(self._format_summary("FC1 activation", activations["fc1"], True))
        print(self._format_summary("Shortcut", activations["shortcut"]))

        output = {
            key: host_scalar(value)
            for key, value in snapshot["activations"]["output"].items()
        }
        print(f"[Simple Output Composition](Step {display_step}, current batch snapshot)")
        print(f"  Deep output     : mean abs={output['deep_abs_mean']:.6e}")
        print(f"  Direct shortcut : mean abs={output['shortcut_abs_mean']:.6e}")
        print(
            f"  Final output    : mean={output['final_mean']:+.6e} "
            f"absmean={output['final_abs_mean']:.6e} "
            f"min={output['final_min']:+.6e} max={output['final_max']:+.6e}")
        print(
            f"  Absolute share  : Deep={output['deep_share']:.2%} "
            f"Shortcut={output['shortcut_share']:.2%}")

        print(f"[Simple FT Stats] step={display_step}")
        if "ft_pre_step" in snapshot and all(
                key in snapshot["ft_pre_step"] for key in (
                    "update_rms", "update_max", "relative_update",
                    "row_mean", "row_median", "row_p90", "row_p99")):
            ft = {key: host_scalar(value)
                  for key, value in snapshot["ft_pre_step"].items()}
            print(
                f"  HalfKA_HM2: touched={int(ft['touched']):6d} "
                f"ratio={ft['touched'] / FT_INPUTS:6.2%} "
                f"grad={ft['grad_norm']:.4e} weight={ft['weight_norm']:.4e} "
                f"rms={ft['update_rms']:.4e} max={ft['update_max']:.4e} "
                f"rel={ft['relative_update']:.4e} "
                f"row_mean={ft['row_mean']:.4e} row_med={ft['row_median']:.4e} "
                f"p90={ft['row_p90']:.4e} p99={ft['row_p99']:.4e}")
        else:
            print("  HalfKA_HM2: optimizer update statistics unavailable")

        print("[Simple Layer/Gradient Stats]")
        print("  Layer Name       | Grad Mean    Active   | W_Mean   W_Min    W_Max     W_Std   | B_Mean   B_Min    B_Max     B_Std")
        print("  " + "-" * 116)
        for name, values in (self._simple_debug_layer_stats or {}).items():
            row = {key: host_scalar(value) for key, value in values.items()}
            print(
                f"  {name:<16} | {row['grad_mean']:11.4e} "
                f"{int(row['active']):8d} | {row['w_mean']:+.5f} "
                f"{row['w_min']:+.5f} {row['w_max']:+.5f} {row['w_std']:.5f} | "
                f"{row['b_mean']:+.5f} {row['b_min']:+.5f} "
                f"{row['b_max']:+.5f} {row['b_std']:.5f}")

        clip_stats = self._simple_clip_stats_pending or {}
        print(f"[Simple Fixed-Point Weight Clipping](Step {display_step})")
        print("  Layer              | W_Min       W_Max       | clip_low  clip_high | boundary low/high")
        print("  " + "-" * 91)
        for name, values in clip_stats.items():
            row = {key: host_scalar(value) for key, value in values.items()}
            print(
                f"  {name:<18} | {row['w_min']:+.7f} {row['w_max']:+.7f} | "
                f"{row['clip_low_pct']:8.5f}% {row['clip_high_pct']:9.5f}% | "
                f"{int(row['boundary_low']):7d}/{int(row['boundary_high']):7d} "
                f"of {int(row['elements']):,}")

        loss = {key: host_scalar(value) for key, value in snapshot["loss"].items()}
        print(f"[Simple DEBUG LOSS](Step {display_step}, current batch snapshot)")
        print(
            f"  Total={loss['total']:.8f} Base={loss['base']:.8f} "
            f"Pairwise={loss['pairwise']:.8f} Listwise={loss['listwise']:.8f}")
        print(
            f"  Valid Pairs={int(loss['valid_pairs']):,} | "
            f"Active Listwise Groups={int(loss['active_listwise_groups']):,} | "
            f"actual_lambda={loss['actual_lambda']:.6f}")

        self._print_pairwise_listwise_diagnostics(display_step, snapshot)
        self._print_distribution_diagnostics(display_step, snapshot["distribution"])
        self._print_training_bucket_stats(display_step)

        if torch.cuda.is_available():
            print(f"[Simple GPU Memory](Step {display_step})")
            print(f"  allocated    : {torch.cuda.memory_allocated() / 1024**3:7.4f} GB")
            print(f"  reserved     : {torch.cuda.memory_reserved() / 1024**3:7.4f} GB")
            print(f"  max allocated: {torch.cuda.max_memory_allocated() / 1024**3:7.4f} GB")

    @staticmethod
    def _print_pairwise_listwise_diagnostics(display_step, snapshot):
        pair_metrics = snapshot.get("pair_metrics", {})
        pred_parts = pair_metrics.get("all_pred_diffs", [])
        target_parts = pair_metrics.get("all_target_directions", [])
        valid_count = int(pair_metrics.get("total_valid_pairs", 0))
        if valid_count > 0 and pred_parts and target_parts:
            pred_diff = torch.cat(pred_parts).detach().float().cpu()
            target_direction = torch.cat(target_parts).detach().float().cpu()
            diff_parts = pair_metrics.get("all_diff_abs", [])
            diff_abs = (torch.cat(diff_parts).detach().float().cpu()
                        if diff_parts else torch.zeros_like(pred_diff))
            value_parts = pair_metrics.get("all_value_gaps", [])
            value_gap = (torch.cat(value_parts).detach().float().cpu()
                         if value_parts else torch.empty(0))
            correct = (target_direction * pred_diff) > 0
            equals = int(pair_metrics.get("all_num_equals", 0))
            print(f"[Simple Pairwise Detail](Step {display_step}, current batch snapshot)")
            print(
                f"  pair_acc={correct.float().mean().item():.4f} | "
                f"valid_pairs={valid_count:,} | equal={equals / valid_count:.3%}")
            print(
                f"  pred_diff_abs_mean={pred_diff.abs().mean().item():.6f} | "
                f"signed_mean={(target_direction * pred_diff).mean().item():+.6f} | "
                f"value_diff_cp_mae="
                f"{value_gap.mean().item() if value_gap.numel() else float('nan'):.3f}")

            bins = (
                (0.000, 0.002), (0.002, 0.005), (0.005, 0.010),
                (0.010, 0.020), (0.020, 0.030), (0.030, 0.050),
                (0.050, 0.070), (0.070, 0.100), (0.100, 0.150))
            print(f"[Simple Pairwise per Range](Step {display_step})")
            for start, end in bins:
                selected = (diff_abs >= start) & (diff_abs < end)
                count = int(selected.sum().item())
                label = f"{start * 100:.1f}%-{end * 100:.1f}%"
                if count:
                    sub_product = target_direction[selected] * pred_diff[selected]
                    print(
                        f"  {label:<11}: {count:6d} | "
                        f"acc={(sub_product > 0).float().mean().item():.4f} | "
                        f"signed={sub_product.mean().item():+.6f} | "
                        f"pred_abs={pred_diff[selected].abs().mean().item():.6f}")
                else:
                    print(f"  {label:<11}:      0 | N/A")

            layer_parts = pair_metrics.get("all_valid_lsinds", [])
            if layer_parts:
                layer_indices = torch.cat(layer_parts).detach().long().cpu()
                counts = torch.bincount(layer_indices, minlength=LAYER_STACKS)
                text = " | ".join(
                    f"B{index:02d}: {int(counts[index]):6d}"
                    for index in range(LAYER_STACKS))
                print(f"[Simple Pairwise Valid Pairs per Layer] {text}")
        else:
            print(f"[Simple Pairwise Detail](Step {display_step}) no valid pairs")

        pt_range = snapshot.get("pt_range")
        if pt_range is None or not pt_range.numel():
            print(f"[Simple Listwise Detail](Step {display_step}) no active groups")
            return
        values = pt_range.detach().float().cpu()
        print(
            f"[Simple Listwise Detail](Step {display_step}) "
            f"active_groups={values.numel():,}")
        print(
            f"  range mean={values.mean().item():.6f} "
            f"std={values.std(unbiased=False).item():.6f} "
            f"median={values.median().item():.6f} "
            f"min={values.min().item():.6f} max={values.max().item():.6f}")
        bins = (
            (0.00, 0.01, "0.00-0.01"),
            (0.01, 0.02, "0.01-0.02"),
            (0.02, 0.05, "0.02-0.05"),
            (0.05, 0.10, "0.05-0.10"),
            (0.10, 0.20, "0.10-0.20"),
            (0.20, 0.30, "0.20-0.30"))
        histogram = [
            f"{label}: {int(((values >= low) & (values < high)).sum()):5d}"
            for low, high, label in bins]
        histogram.append(f">=0.30: {int((values >= 0.30).sum()):5d}")
        print("  histogram | " + " | ".join(histogram))

    @staticmethod
    def _print_distribution_diagnostics(display_step, distribution):
        host = {
            name: value.detach().float().view(-1).cpu()
            for name, value in distribution.items()
        }
        pt, pf, qf = host["pt"], host["pf"], host["qf"]
        ply = host["ply"]
        material = host["material"]
        group = host["group"].long()
        print(f"[Simple PT Distribution](Step {display_step}, current batch snapshot)")
        print(
            f"  mean={pt.mean().item():.6f} std={pt.std(unbiased=False).item():.6f} "
            f"median={pt.median().item():.6f} min={pt.min().item():.6f} "
            f"max={pt.max().item():.6f}")
        pt_bins = []
        for index in range(10):
            low, high = index / 10.0, (index + 1) / 10.0
            selected = ((pt >= low) & (pt < high)) if index < 9 else (pt >= low)
            pt_bins.append(f"{low:.1f}-{high:.1f}: {int(selected.sum()):5d}")
        print("  histogram | " + " | ".join(pt_bins))
        group_counts = torch.bincount(group.clamp_min(0), minlength=4)
        print(
            f"[Simple Kif Group Counts] ID_1: {int(group_counts[1])} | "
            f"ID_2: {int(group_counts[2])} | ID_3: {int(group_counts[3])} "
            f"(Batch Total: {pt.numel()})")

        print(f"[Simple ply Distribution](Step {display_step}, current batch snapshot)")
        print(
            f"  mean={ply.mean().item():.3f} std={ply.std(unbiased=False).item():.3f} "
            f"median={ply.median().item():.1f} min={ply.min().item():.1f} "
            f"max={ply.max().item():.1f} | "
            f"ply=0 ratio={(ply == 0).float().mean().item():.3%}")
        print("  Ply range | Count |  qf_m  |  pf_m  |  pt_m  | |qf-.5| | |pf-.5| | |pt-.5| | Mat mean | |Mat| mean")
        print("  " + "-" * 108)
        ply_bins = (
            (0, 1, "0"), (1, 21, "1-20"), (21, 41, "21-40"),
            (41, 61, "41-60"), (61, 81, "61-80"),
            (81, 101, "81-100"), (101, 121, "101-120"),
            (121, 161, "121-160"), (161, float("inf"), "161+"))
        for low, high, label in ply_bins:
            selected = (ply >= low) & (ply < high)
            count = int(selected.sum().item())
            if not count:
                continue
            print(
                f"  {label:>9} | {count:5d} | {qf[selected].mean().item():.4f} | "
                f"{pf[selected].mean().item():.4f} | {pt[selected].mean().item():.4f} | "
                f"{(qf[selected] - .5).abs().mean().item():.4f} | "
                f"{(pf[selected] - .5).abs().mean().item():.4f} | "
                f"{(pt[selected] - .5).abs().mean().item():.4f} | "
                f"{material[selected].mean().item():+8.1f} | "
                f"{material[selected].abs().mean().item():8.1f}")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self._simple_debug_capture:
            return
        display_step = int(
            self._simple_debug_display_step
            if self._simple_debug_display_step is not None
            else self.global_step)
        self._finish_ft_update_stats()
        self._print_current_batch_diagnostics(display_step)
        self._simple_debug_capture = False
        self._simple_debug_snapshot = None
        self._simple_debug_touched_rows = None
        self._simple_debug_ft_before = None
        self._simple_debug_layer_stats = None
        self._simple_clip_stats_pending = None
        self._simple_debug_display_step = None

    def on_validation_epoch_start(self):
        optimizer = self.optimizers()
        param_groups = optimizer.param_groups
        if not param_groups:
            return
        current_lr = float(param_groups[0]["lr"])
        print("[Simple optimizer learning rates]")
        group_names = ("HalfKA_HM2 FT", "9 bucket layer stacks")
        for index, group in enumerate(param_groups):
            name = group_names[index] if index < len(group_names) else "unknown"
            parameters = sum(parameter.numel() for parameter in group["params"])
            print(
                f"  group {index:02d} [{name}]: "
                f"lr={float(group['lr']):.12g}, params={parameters:,}")

        experiment = getattr(self.logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "add_scalar"):
            experiment.add_scalar(
                "current_lr", current_lr, global_step=self.global_step)

    def configure_optimizers(self):
        # A fresh optimizer is intentional for transplanted checkpoints.
        optimizer = torch.optim.AdamW(
            [
                {"params": list(self.input.parameters()), "lr": self.lr},
                {"params": list(self.layer_stacks.parameters()), "lr": self.lr},
            ],
            betas=(0.9, 0.995), eps=1e-7, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        metadata = self.architecture_metadata(
            getattr(self, "transplant_source", None),
            getattr(self, "transplant_mapping_version", None))
        checkpoint["architecture"] = metadata
        checkpoint["nnue_architecture"] = metadata

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        metadata = checkpoint.get("architecture") or checkpoint.get("nnue_architecture")
        if not metadata or metadata.get("architecture_type") != ARCHITECTURE_TYPE:
            raise ValueError("checkpoint is not a HalfKA_hm2 simple checkpoint")
        if int(metadata.get("simple_schema_version", -1)) != SIMPLE_SCHEMA_VERSION:
            raise ValueError("HalfKA_hm2 simple schema mismatch")
        self.transplant_source = metadata.get("transplant_source")
        self.transplant_mapping_version = metadata.get("transplant_mapping_version")
