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
# Simple v2 fixed-point contract.  These values mirror the production C++
# inference implementation and serialize_halfka_hm2_simple.py; QAT must not
# silently inherit constants from Stockfish or from the Complex network.
FT_QUANT_SCALE = 127.0
FT_QUANT_MIN = -32768.0
FT_QUANT_MAX = 32767.0
FT_LOAD_MULTIPLIER = 2.0
DENSE_WEIGHT_SCALE = 64.0
DENSE_WEIGHT_MIN = -127.0
DENSE_WEIGHT_MAX = 127.0
DENSE_BIAS_SCALE = DENSE_WEIGHT_SCALE * FT_QUANT_SCALE  # 8128
ACCUMULATOR_PACK_MIN = 0.0
ACCUMULATOR_PACK_MAX = 254.0
ACTIVATION_QUANT_MIN = 0.0
ACTIVATION_QUANT_MAX = 127.0
UINT8_MAX = 255.0
INT16_MIN = -32768.0
INT16_MAX = 32767.0
DENSE_ACTIVATION_SHIFT = 6
SQUARED_ACTIVATION_SHIFT = 19
EWM_SHIFT = 9
NNUE_TO_SCORE = FT_QUANT_SCALE * DENSE_WEIGHT_SCALE / 16.0
MAX_HIDDEN_WEIGHT = DENSE_WEIGHT_MAX / DENSE_WEIGHT_SCALE

# Compatibility aliases used by existing diagnostics and external scripts.
HIDDEN_WEIGHT_SCALE = DENSE_WEIGHT_SCALE
FT_SERIALIZER_SCALE = FT_QUANT_SCALE
SIMPLE_SIDE_INPUT_TYPE = "ply_material_v1"
SIMPLE_SIDE_INPUT_DIM = 4
SIMPLE_DIRECT_SIDE_INPUT_TYPE = "ply_material_direct_v1"
SIMPLE_DIRECT_SIDE_INPUT_DIM = 2
SIMPLE_SHARED_PSQT_TYPE = "halfka_hm2_shared_psqt_v1"
SIMPLE_QAT_MODES = ("off", "weights", "weight_activation", "full")
SIMPLE_QAT_RECOMMENDED_MODE = "full"
BUCKET_STAT_NAMES = (
    "count", "weight", "loss", "prob_mae", "cp_mae",
    "importance_weighted_loss",
    "pt_sum", "pt_sq", "pf_sum", "pf_sq", "qf_sum", "qf_sq",
    "error_sum", "error_abs", "error_sq",
    "ply_sum", "ply_sq", "material_sum", "material_sq",
    "score_sum", "score_sq", "pred_sum", "pred_sq")

# Experiment 92: fixed natural-training-distribution frequencies.  These are
# aggregated from 2,120 disjoint 500-step windows in the v27 production log
# (13,527,794,634 base-regression samples), never from the current mini-batch.
SIMPLE_BUCKET_TRAIN_COUNTS = (
    111_119_684, 48_838_898, 81_167_259, 50_293_752, 91_517_365,
    481_820_651, 96_587_244, 577_071_342, 11_989_378_439)
SIMPLE_BUCKET_TRAIN_FREQUENCIES = tuple(
    count / sum(SIMPLE_BUCKET_TRAIN_COUNTS)
    for count in SIMPLE_BUCKET_TRAIN_COUNTS)
SIMPLE_BUCKET_IMPORTANCE_RAW = tuple(
    min(frequency ** -0.25, 4.0)
    for frequency in SIMPLE_BUCKET_TRAIN_FREQUENCIES)
_SIMPLE_BUCKET_IMPORTANCE_MEAN = sum(
    frequency * weight for frequency, weight in zip(
        SIMPLE_BUCKET_TRAIN_FREQUENCIES, SIMPLE_BUCKET_IMPORTANCE_RAW))
SIMPLE_BUCKET_IMPORTANCE_NORMALIZED = tuple(
    weight / _SIMPLE_BUCKET_IMPORTANCE_MEAN
    for weight in SIMPLE_BUCKET_IMPORTANCE_RAW)


def normalize_simple_side_input(ply, material_stm, dtype=None):
    """Normalize the two Simple side inputs without changing perspective.

    ``training_data_loader.cpp`` already emits ``material`` as
    ``Eval::material(pos) * (side_to_move == BLACK ? 1 : -1)``.  It is thus
    side-to-move-relative (friend minus enemy), exactly like the transformed
    NNUE input. Applying another color/sign conversion here would be wrong.
    """
    if dtype is None:
        dtype = torch.float32
    ply_norm = torch.clamp(
        ply.view(-1).to(dtype=dtype) / 200.0, 0.0, 2.0)
    material_norm = torch.clamp(
        material_stm.view(-1).to(dtype=dtype) / 4000.0, -2.0, 2.0)
    return torch.stack((ply_norm, material_norm), dim=1)


def _ste(soft, hard):
    """Use ``hard`` in forward while preserving the gradient of ``soft``."""
    return soft + (hard - soft).detach()


def _fake_quantize(parameter, scale, low, high):
    """Fake-quantize onto a C++ storage grid with identity STE backward."""
    quantized = torch.clamp(torch.round(parameter * scale), low, high) / scale
    return _ste(parameter, quantized)


def _fake_raw_grid(value):
    """Use the common dense raw unit (1/8128) in forward, STE in backward."""
    return _ste(value, torch.round(value * DENSE_BIAS_SCALE)
                / DENSE_BIAS_SCALE)


def _fake_dense_parameters(layer):
    # Exactly serialize_halfka_hm2_simple._write_fc(): symmetric int8
    # weights at scale 64 and int32 biases at scale 64*127.
    weight = _fake_quantize(
        layer.weight, DENSE_WEIGHT_SCALE,
        DENSE_WEIGHT_MIN, DENSE_WEIGHT_MAX)
    bias = _fake_raw_grid(layer.bias)
    return weight, bias


def _simple_qat_ewm(value, qat_mode):
    """Element-wise multiplication for one fixed-perspective FT accumulator.

    Float modes preserve the historical Simple training expression.  The
    activation/full modes reproduce the production C++ ``scale_weights(true)``
    path: q127 lanes are doubled, uint8-packed, multiplied and shifted by 9.
    """
    a, b = value[:, :FT_WIDTH // 2], value[:, FT_WIDTH // 2:]
    soft = (torch.clamp(a, 0.0, 2.0)
            * torch.clamp(b, 0.0, 2.0)
            * (FT_QUANT_SCALE / 128.0))
    if qat_mode not in ("weight_activation", "full"):
        return soft
    q0 = torch.clamp(
        torch.round(a * FT_QUANT_SCALE) * FT_LOAD_MULTIPLIER,
        ACCUMULATOR_PACK_MIN, ACCUMULATOR_PACK_MAX)
    q1 = torch.clamp(
        torch.round(b * FT_QUANT_SCALE) * FT_LOAD_MULTIPLIER,
        ACCUMULATOR_PACK_MIN, ACCUMULATOR_PACK_MAX)
    hard = torch.clamp(
        torch.floor(q0 * q1 / float(1 << EWM_SHIFT)),
        ACTIVATION_QUANT_MIN, UINT8_MAX) / FT_QUANT_SCALE
    return _ste(soft, hard)


def _simple_qat_activation(pre_activation, squared=False):
    """Apply the Simple v2 uint8 activation contract with STE backward."""
    raw = torch.clamp(
        torch.round(pre_activation * DENSE_BIAS_SCALE),
        INT16_MIN, INT16_MAX)
    if squared:
        hard_u8 = torch.clamp(
            torch.floor(raw.square() / float(1 << SQUARED_ACTIVATION_SHIFT)),
            ACTIVATION_QUANT_MIN, ACTIVATION_QUANT_MAX)
        soft = torch.clamp(
            pre_activation.square() * (FT_QUANT_SCALE / 128.0), 0.0, 1.0)
    else:
        hard_u8 = torch.clamp(
            torch.floor(raw / float(1 << DENSE_ACTIVATION_SHIFT)),
            ACTIVATION_QUANT_MIN, ACTIVATION_QUANT_MAX)
        soft = torch.clamp(pre_activation, 0.0, 1.0)
    return _ste(soft, hard_u8 / FT_QUANT_SCALE)


class SimpleFeatureTransformer(nn.Module):
    def __init__(self, num_inputs: int = FT_INPUTS, width: int = FT_WIDTH):
        super().__init__()
        sigma = math.sqrt(1.0 / num_inputs)
        self.weight = nn.Parameter(torch.empty(num_inputs, width))
        self.bias = nn.Parameter(torch.empty(width))
        nn.init.uniform_(self.weight, -sigma, sigma)
        nn.init.uniform_(self.bias, -sigma, sigma)
        self._qat_eval_cache = None

    def train(self, mode: bool = True):
        # A validation cache is valid only until parameters may change again.
        # Lightning switches the module back to train mode before optimization,
        # so clearing here prevents a later validation epoch from seeing stale
        # fake-quantized FT tensors.
        if mode:
            self._qat_eval_cache = None
        return super().train(mode)

    def forward(self, white_indices, white_values, black_indices, black_values,
                qat_mode="off"):
        # The complex model's generated CuPy kernel is tuned and cached for its
        # 1280-wide FT.  Generating the first 1536-wide kernel takes minutes on
        # the supported Windows toolchain.  embedding_bag provides the same
        # sparse weighted row sum using a native PyTorch kernel, without a
        # multi-gigabyte [batch, active, width] intermediate.
        if qat_mode == "off":
            weight, bias = self.weight, self.bias
        elif not self.training and self._qat_eval_cache is not None:
            weight, bias = self._qat_eval_cache
        else:
            # nn.bin stores both FT tensors as int16 at scale 127. C++ doubles
            # them after loading; the exact EWM path below accounts for that.
            weight = _fake_quantize(
                self.weight, FT_QUANT_SCALE, FT_QUANT_MIN, FT_QUANT_MAX)
            bias = _fake_quantize(
                self.bias, FT_QUANT_SCALE, FT_QUANT_MIN, FT_QUANT_MAX)
            if not self.training:
                # Validation has many batches but no parameter updates. Avoid
                # requantizing the 112.6M-entry FT for every batch.
                self._qat_eval_cache = (weight.detach(), bias.detach())

        def accumulate(indices, values):
            batch, active = indices.shape
            valid = indices >= 0
            safe_indices = indices.masked_fill(~valid, 0).reshape(-1).long()
            safe_values = values.masked_fill(~valid, 0.0).reshape(-1)
            offsets = torch.arange(
                0, batch * active, active, dtype=torch.long,
                device=indices.device)
            return F.embedding_bag(
                safe_indices, weight, offsets,
                per_sample_weights=safe_values, mode="sum") + bias

        return (accumulate(white_indices, white_values),
                accumulate(black_indices, black_values))


class SimpleSharedPsqt(nn.Module):
    """One learned scalar per HalfKA_HM2 feature, shared by all buckets."""

    def __init__(self, num_inputs: int = FT_INPUTS):
        super().__init__()
        # Zero rows make baseline -> PSQT migration exactly output-neutral.
        self.weight = nn.Parameter(torch.zeros(num_inputs, 1))
        # Keep the scale explicit and trainable.  Initializing it to one lets
        # the zero rows receive gradient on the first optimizer step.
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, white_indices, white_values,
                black_indices, black_values):
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
                per_sample_weights=safe_values, mode="sum").view(-1)

        return (accumulate(white_indices, white_values),
                accumulate(black_indices, black_values))


class SimpleStack(nn.Module):
    """1536 -> 16; (sqr15, relu15) -> 32 -> 1 + shortcut."""

    def __init__(self, side_input_dim: int = 0):
        super().__init__()
        self.side_input_dim = int(side_input_dim)
        self.fc0 = nn.Linear(FT_WIDTH, 16)
        self.fc1 = nn.Linear(30 + self.side_input_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        if self.side_input_dim:
            # Preserve the old network exactly at architecture-migration time.
            # The original 30 columns keep their normal initialization/load;
            # only the newly attached side columns start with no contribution.
            with torch.no_grad():
                self.fc1.weight[:, 30:].zero_()

    def forward(self, x, side_input=None, collect_diagnostics=False,
                qat_mode="off"):
        quantize_weights = qat_mode != "off"
        quantize_activations = qat_mode in ("weight_activation", "full")

        def affine(layer, value):
            if not quantize_weights:
                return layer(value)
            weight, bias = _fake_dense_parameters(layer)
            return F.linear(value, weight, bias)

        h0 = affine(self.fc0, x)
        if quantize_activations:
            h0 = _fake_raw_grid(h0)
        hidden_pre = h0[:, :15]
        if quantize_activations:
            hidden = _simple_qat_activation(hidden_pre, squared=False)
            hidden2 = _simple_qat_activation(hidden_pre, squared=True)
        else:
            # Canonical SFNN squares the raw affine output before clipping.
            hidden2 = torch.clamp(
                hidden_pre.square() * (FT_QUANT_SCALE / 128.0), 0.0, 1.0)
            hidden = torch.clamp(hidden_pre, 0.0, 1.0)
        main_h = torch.cat((hidden2, hidden), dim=1)
        if self.side_input_dim:
            if side_input is None:
                raise ValueError("enabled simple side input requires side features")
            fc1_input = torch.cat((main_h, side_input), dim=1)
        else:
            fc1_input = main_h
        fc1_pre = affine(self.fc1, fc1_input)
        if quantize_activations:
            fc1_pre = _fake_raw_grid(fc1_pre)
            h1 = _simple_qat_activation(fc1_pre, squared=False)
        else:
            h1 = torch.clamp(fc1_pre, 0.0, 1.0)
        deep = affine(self.fc2, h1)
        shortcut = h0[:, 15:16]
        output = deep + shortcut
        if qat_mode == "full":
            # C++ adds deep and shortcut in their common raw 1/8128 domain.
            deep_raw = torch.round(deep * DENSE_BIAS_SCALE)
            shortcut_raw = torch.round(shortcut * DENSE_BIAS_SCALE)
            output = _ste(
                output, (deep_raw + shortcut_raw) / DENSE_BIAS_SCALE)
        if collect_diagnostics:
            diagnostics = {
                "fc0_pre": h0[:, :15],
                "clipped": hidden,
                "squared": hidden2,
                "fc1_pre": fc1_pre,
                "fc1_activation": h1,
                "deep": deep,
                "shortcut": shortcut,
                "final": output,
            }
            if self.side_input_dim:
                diagnostics.update({
                    "side_projection": side_input,
                    "main_fc1_contribution": F.linear(
                        main_h, self.fc1.weight[:, :30], None),
                    "side_fc1_contribution": F.linear(
                        side_input, self.fc1.weight[:, 30:], None),
                })
                if self.side_input_dim == SIMPLE_DIRECT_SIDE_INPUT_DIM:
                    diagnostics.update({
                        "ply_fc1_contribution": (
                            side_input[:, 0:1] * self.fc1.weight[None, :, 30]),
                        "material_fc1_contribution": (
                            side_input[:, 1:2] * self.fc1.weight[None, :, 31]),
                    })
            return output, diagnostics
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
        use_side_input=False,
        use_direct_side_input=False,
        use_shared_psqt=False,
        use_bucket_importance_base_loss=False,
        simple_qat_mode="off",
        **unused,
    ):
        super().__init__()
        if feature_set.name != FEATURE_NAME or feature_set.num_features != FT_INPUTS:
            raise ValueError(
                f"simple architecture requires {FEATURE_NAME}/{FT_INPUTS}, got "
                f"{feature_set.name}/{feature_set.num_features}")
        self.feature_set = feature_set
        if use_side_input and use_direct_side_input:
            raise ValueError(
                "projected and direct Simple side inputs are mutually exclusive")
        self.use_projected_side_input = bool(use_side_input)
        self.use_direct_side_input = bool(use_direct_side_input)
        self.use_side_input = bool(use_side_input or use_direct_side_input)
        self.use_shared_psqt = bool(use_shared_psqt)
        self.simple_side_input_type = (
            (SIMPLE_DIRECT_SIDE_INPUT_TYPE if self.use_direct_side_input
             else SIMPLE_SIDE_INPUT_TYPE)
            if self.use_side_input else "none")
        self.input = SimpleFeatureTransformer()
        self.shared_psqt = (
            SimpleSharedPsqt() if self.use_shared_psqt else None)
        self.side_proj = (
            nn.Sequential(nn.Linear(2, SIMPLE_SIDE_INPUT_DIM), nn.ReLU())
            if self.use_projected_side_input else None)
        side_dim = (
            SIMPLE_DIRECT_SIDE_INPUT_DIM if self.use_direct_side_input
            else SIMPLE_SIDE_INPUT_DIM if self.use_projected_side_input
            else 0)
        self.layer_stacks = nn.ModuleList(
            SimpleStack(side_dim)
            for _ in range(LAYER_STACKS))
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
        self.use_bucket_importance_base_loss = bool(
            use_bucket_importance_base_loss)
        self.simple_qat_mode = str(simple_qat_mode)
        if self.simple_qat_mode not in SIMPLE_QAT_MODES:
            raise ValueError(
                f"simple_qat_mode must be one of {SIMPLE_QAT_MODES}, got "
                f"{self.simple_qat_mode!r}")
        self.register_buffer(
            "_bucket_importance_weights",
            torch.tensor(SIMPLE_BUCKET_IMPORTANCE_NORMALIZED),
            persistent=False)
        self._simple_debug_capture = False
        self._simple_debug_snapshot = None
        self._simple_debug_touched_rows = None
        self._simple_debug_ft_before = None
        self._simple_debug_layer_stats = None
        self._simple_clip_stats_pending = None
        self._simple_debug_display_step = None
        self.simple_validation_cohort_report = False
        self._simple_validation_cohorts = None
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

    def architecture_metadata(self, transplant_source=None, mapping_version=None):
        use_side_input = bool(getattr(self, "use_side_input", False))
        return {
            "architecture_type": ARCHITECTURE_TYPE,
            "feature": FEATURE_NAME,
            "ft_width": FT_WIDTH,
            "layer_stack_count": LAYER_STACKS,
            "bucket_scheme": "k3k3",
            "distinguish_golds": False,
            "long_effect_required": False,
            "simple_schema_version": SIMPLE_SCHEMA_VERSION,
            "use_side_input": use_side_input,
            "simple_side_input_type": (
                getattr(self, "simple_side_input_type", "none")
                if use_side_input else "none"),
            "simple_side_input_projection_dim": (
                (SIMPLE_DIRECT_SIDE_INPUT_DIM
                 if getattr(self, "use_direct_side_input", False)
                 else SIMPLE_SIDE_INPUT_DIM)
                if use_side_input else 0),
            "simple_side_input_fusion": (
                ("fc1_direct_concat"
                 if getattr(self, "use_direct_side_input", False)
                 else "fc1_concat")
                if use_side_input else "none"),
            "simple_side_input_normalization": (
                "ply_clamp_200_0_2+material_clamp_4000_m2_2"
                if use_side_input else "none"),
            "use_shared_psqt": bool(self.use_shared_psqt),
            "simple_psqt_type": (
                SIMPLE_SHARED_PSQT_TYPE if self.use_shared_psqt else "none"),
            "simple_psqt_scale": (
                "learnable_scalar_init_1" if self.use_shared_psqt else "none"),
            "transplant_source": transplant_source,
            "transplant_mapping_version": mapping_version,
        }

    def set_feature_set(self, feature_set):
        if feature_set.name != FEATURE_NAME:
            raise ValueError(f"simple checkpoint requires {FEATURE_NAME}")
        self.feature_set = feature_set

    def forward(self, us, them, white_indices, white_values, black_indices,
                black_values, layer_stack_indices, ply=None, material=None,
                **unused):
        tw, tb = self.input(
            white_indices, white_values, black_indices, black_values,
            qat_mode=self.simple_qat_mode)
        us = us.view(-1, 1)
        them = them.view(-1, 1)
        # The loader exposes BLACK in white_* and WHITE in black_*; us/them
        # select the side-to-move ordering exactly as the complex model does.
        # Each 1536-wide accumulator contains two 768-wide factor lanes.
        # Element-wise multiplication emits 768 values per perspective, then
        # side-to-move ordering concatenates friend and enemy perspectives.
        ew = _simple_qat_ewm(tw, self.simple_qat_mode)
        eb = _simple_qat_ewm(tb, self.simple_qat_mode)
        transformed = us * torch.cat((ew, eb), dim=1) \
            + them * torch.cat((eb, ew), dim=1)
        side_raw = None
        side_h = None
        if self.use_side_input:
            if ply is None or material is None:
                raise ValueError(
                    "ply/material are required when simple side input is enabled")
            side_raw = normalize_simple_side_input(
                ply, material, dtype=transformed.dtype)
            side_h = (side_raw if self.use_direct_side_input
                      else self.side_proj(side_raw))
        buckets = layer_stack_indices.view(-1).long()
        if torch.any((buckets < 0) | (buckets >= LAYER_STACKS)):
            raise RuntimeError("HalfKA_hm2 simple bucket must be in [0,8]")
        collect = bool(self.training and self._simple_debug_capture)
        out = transformed.new_empty((transformed.shape[0], 1))
        diagnostic_parts = {} if collect else None
        psqt_deep = (transformed.new_empty(transformed.shape[0])
                     if collect and self.use_shared_psqt else None)
        psqt_shortcut = (transformed.new_empty(transformed.shape[0])
                         if collect and self.use_shared_psqt else None)
        for bucket, stack in enumerate(self.layer_stacks):
            mask = buckets == bucket
            if mask.any():
                if collect:
                    bucket_out, bucket_diagnostics = stack(
                        transformed[mask],
                        None if side_h is None else side_h[mask],
                        collect_diagnostics=True,
                        qat_mode=self.simple_qat_mode)
                    out[mask] = bucket_out
                    if self.use_shared_psqt:
                        psqt_deep[mask] = bucket_diagnostics["deep"].view(-1)
                        psqt_shortcut[mask] = (
                            bucket_diagnostics["shortcut"].view(-1))
                    for name, value in bucket_diagnostics.items():
                        diagnostic_parts.setdefault(name, []).append(value)
                else:
                    out[mask] = stack(
                        transformed[mask],
                        None if side_h is None else side_h[mask],
                        qat_mode=self.simple_qat_mode)
        psqt_snapshot = None
        if self.use_shared_psqt:
            white_psqt, black_psqt = self.shared_psqt(
                white_indices, white_values, black_indices, black_values)
            # white_* is the fixed BLACK-perspective accumulator and black_*
            # the fixed WHITE-perspective accumulator.  Apply the same
            # side-to-move friend/enemy selection as the FT path above.
            raw_psqt = (
                us.view(-1) * (white_psqt - black_psqt)
                + them.view(-1) * (black_psqt - white_psqt))
            psqt_value = self.shared_psqt.scale * raw_psqt
            existing_output = out.view(-1)
            out = out + psqt_value.view(-1, 1)
            if collect:
                psqt_snapshot = {
                    "raw": raw_psqt.detach(),
                    "value": psqt_value.detach(),
                    "existing_output": existing_output.detach(),
                    "final_output": out.view(-1).detach(),
                    "deep": psqt_deep.detach(),
                    "shortcut": psqt_shortcut.detach(),
                }
        if collect:
            activation_tensors = {
                name: torch.cat(parts, dim=0)
                for name, parts in diagnostic_parts.items()
            }
            self._simple_debug_snapshot = {
                "activations": self._activation_snapshot(
                    transformed, activation_tensors, side_raw),
            }
            if psqt_snapshot is not None:
                self._simple_debug_snapshot["psqt"] = psqt_snapshot
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

    def _activation_snapshot(self, transformed, tensors, side_raw=None):
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
        result = {
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
        if self.use_side_input:
            side_contribution = tensors["side_fc1_contribution"].detach().float()
            main_contribution = tensors["main_fc1_contribution"].detach().float()
            total_pre = tensors["fc1_pre"].detach().float()
            result["side_input"] = {
                "ply_norm": self._tensor_summary(side_raw[:, 0]),
                "material_norm": self._tensor_summary(side_raw[:, 1]),
                "side_projection": self._tensor_summary(
                    tensors["side_projection"]),
                "side_contribution_abs_mean": side_contribution.abs().mean(),
                "main_contribution_abs_mean": main_contribution.abs().mean(),
                "total_pre_abs_mean": total_pre.abs().mean(),
                "side_to_total_ratio": (
                    side_contribution.abs().mean()
                    / total_pre.abs().mean().clamp_min(1e-12)),
            }
            if self.use_direct_side_input:
                ply_contribution = tensors["ply_fc1_contribution"].detach().float()
                material_contribution = (
                    tensors["material_fc1_contribution"].detach().float())
                ply_weights = torch.cat([
                    stack.fc1.weight[:, 30].detach().float().reshape(-1)
                    for stack in self.layer_stacks])
                material_weights = torch.cat([
                    stack.fc1.weight[:, 31].detach().float().reshape(-1)
                    for stack in self.layer_stacks])
                result["side_input"].update({
                    "ply_contribution_abs_mean": ply_contribution.abs().mean(),
                    "material_contribution_abs_mean": (
                        material_contribution.abs().mean()),
                    "ply_weight_norm": ply_weights.norm(),
                    "material_weight_norm": material_weights.norm(),
                })
        return result

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
        pred_cp = self(
            us, them, wi, wv, bi, bv, bucket,
            ply=ply, material=material).view(-1) * self.nnue2score
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
        base_loss_numerator = err[mask] * weights[mask]
        if stage == "train" and self.use_bucket_importance_base_loss:
            # Keep the old denominator.  Since E_natural[importance] == 1,
            # this preserves the expected global gradient scale while changing
            # only the selected-bucket allocation of base-regression gradient.
            bucket_importance = self._bucket_importance_weights.index_select(
                0, bucket.view(-1).long()[mask])
            base_loss_numerator = base_loss_numerator * bucket_importance
        base_loss = (base_loss_numerator.sum()
                     / weights[mask].sum().clamp_min(1e-9))

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
            if self.simple_validation_cohort_report:
                self._accumulate_validation_cohort_stats(
                    bucket.view(-1).long(), (qf - pf).abs(),
                    (pred_cp - score).abs(), ply, material)
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
                    (lambda1_err[mask] * weights[mask]
                     * (self._bucket_importance_weights.index_select(
                         0, bucket.view(-1).long()[mask])
                        if self.use_bucket_importance_base_loss else 1.0)),
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
            ft_limit = FT_QUANT_MAX / FT_SERIALIZER_SCALE
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
            self, bucket, weight, loss, importance_weighted_loss,
            pt, pf, qf, prob_mae, cp_mae,
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
            importance_weighted_loss,
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
            if self.use_shared_psqt and "psqt" in snapshot:
                psqt = snapshot["psqt"]

                def correlation(left, right):
                    left = left.detach().float().view(-1)
                    right = right.detach().float().view(-1)
                    left = left - left.mean()
                    right = right - right.mean()
                    denominator = left.square().sum().sqrt() * \
                        right.square().sum().sqrt()
                    return ((left * right).sum()
                            / denominator.clamp_min(1e-12))

                teacher_output = score.detach().float() / self.nnue2score
                residual = teacher_output - psqt["existing_output"].float()
                contribution = psqt["value"].float()
                psqt["corr_teacher_residual"] = correlation(
                    contribution, residual)
                psqt["corr_shortcut"] = correlation(
                    contribution, psqt["shortcut"])
                psqt["corr_deep"] = correlation(
                    contribution, psqt["deep"])
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
        if self.use_bucket_importance_base_loss:
            weighted_numerator = host["importance_weighted_loss"]
            total_weighted = weighted_numerator.sum().item()
            print(
                f"[Simple Bucket Base Importance](Step {display_step}, "
                f"fixed-frequency, exponent=-0.25, clip=4.0, E[w]=1)")
            print("  Bucket | sample freq | raw weight | norm weight | weighted loss contribution")
            print("  " + "-" * 83)
            for index in range(LAYER_STACKS):
                contribution = (
                    weighted_numerator[index].item() / total_weighted
                    if total_weighted else 0.0)
                print(
                    f"  B{index:02d}    | "
                    f"{SIMPLE_BUCKET_TRAIN_FREQUENCIES[index]:11.7%} | "
                    f"{SIMPLE_BUCKET_IMPORTANCE_RAW[index]:10.6f} | "
                    f"{SIMPLE_BUCKET_IMPORTANCE_NORMALIZED[index]:11.6f} | "
                    f"{contribution:26.6%}")
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
                    ((f"FC1_{30 + self.layer_stacks[0].side_input_dim}x32"
                      if self.use_side_input else "FC1_30x32"),
                     "fc1"),
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
            if self.use_projected_side_input:
                side_linear = self.side_proj[0]
                layer_stats["SideProj_2x4"] = self._layer_stat_row(
                    side_linear.weight, side_linear.bias,
                    side_linear.weight.grad, side_linear.bias.grad)
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

        if self.use_shared_psqt:
            psqt = snapshot["psqt"]
            value = psqt["value"].detach().float()
            raw = psqt["raw"].detach().float()
            existing = psqt["existing_output"].detach().float()
            final = psqt["final_output"].detach().float()
            weight = self.shared_psqt.weight.detach().float().view(-1)
            print(f"[PSQT Stats](Step {display_step}, current batch snapshot)")
            print(
                "  psqt output mean/std/min/max       : "
                f"{value.mean().item():+.6e} / {value.std(unbiased=False).item():.6e} / "
                f"{value.min().item():+.6e} / {value.max().item():+.6e}")
            print(f"  psqt abs mean                     : {value.abs().mean().item():.6e}")
            print(f"  existing output abs mean          : {existing.abs().mean().item():.6e}")
            print(
                "  psqt / final-output abs ratio      : "
                f"{(value.abs().mean() / final.abs().mean().clamp_min(1e-12)).item():.6e}")
            print(
                "  psqt weight mean/std/min/max       : "
                f"{weight.mean().item():+.6e} / {weight.std(unbiased=False).item():.6e} / "
                f"{weight.min().item():+.6e} / {weight.max().item():+.6e}")
            print(f"  psqt learnable scale              : {self.shared_psqt.scale.item():+.6e}")
            print(
                "  active raw contribution mean/std  : "
                f"{raw.mean().item():+.6e} / {raw.std(unbiased=False).item():.6e}")
            print(
                "  corr(psqt, teacher residual)       : "
                f"{host_scalar(psqt['corr_teacher_residual']):+.6f}")
            print(
                "  corr(psqt, shortcut / deep)        : "
                f"{host_scalar(psqt['corr_shortcut']):+.6f} / "
                f"{host_scalar(psqt['corr_deep']):+.6f}")

        if self.use_side_input:
            side = {
                key: ({sub_key: host_scalar(sub_value)
                       for sub_key, sub_value in value.items()}
                      if isinstance(value, dict) else host_scalar(value))
                for key, value in snapshot["activations"]["side_input"].items()
            }
            heading = ("Direct Side Input Stats" if self.use_direct_side_input
                       else "Side Input Stats")
            print(f"[{heading}](Step {display_step}, current batch snapshot)")
            print(self._format_summary("ply_norm", side["ply_norm"]))
            print(self._format_summary(
                "material_norm", side["material_norm"]))
            if self.use_projected_side_input:
                print(self._format_summary(
                    "side_proj", side["side_projection"]))
            else:
                print(
                    "  ply FC1 contribution mean abs  : "
                    f"{side['ply_contribution_abs_mean']:.6e}")
                print(
                    "  material FC1 contribution mean abs: "
                    f"{side['material_contribution_abs_mean']:.6e}")
                print(
                    "  ply/material FC1 weight norm   : "
                    f"{side['ply_weight_norm']:.6e} / "
                    f"{side['material_weight_norm']:.6e}")
            print(
                "  side FC1 contribution mean abs : "
                f"{side['side_contribution_abs_mean']:.6e}")
            print(
                "  main FC1 contribution mean abs : "
                f"{side['main_contribution_abs_mean']:.6e}")
            print(
                "  total FC1 pre-act mean abs      : "
                f"{side['total_pre_abs_mean']:.6e}")
            print(
                "  side / total FC1 pre-act ratio  : "
                f"{side['side_to_total_ratio']:.6e} "
                f"({side['side_to_total_ratio']:.4%})")

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
        print("  Ply range | Count | prob MAE |  cp MAE |  qf_m  |  pf_m  | Mat mean | |Mat| mean")
        print("  " + "-" * 91)
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
                f"  {label:>9} | {count:5d} | "
                f"{(qf[selected] - pf[selected]).abs().mean().item():9.6f} | "
                f"{(host['pred'][selected] - host['score'][selected]).abs().mean().item():8.1f} | "
                f"{qf[selected].mean().item():.4f} | "
                f"{pf[selected].mean().item():.4f} | "
                f"{material[selected].mean().item():+8.1f} | "
                f"{material[selected].abs().mean().item():8.1f}")

        print(f"[Simple Error by |material|](Step {display_step}, current batch snapshot)")
        print("  |material| range | Count | prob MAE |  cp MAE | ply mean")
        print("  " + "-" * 63)
        abs_material = material.abs()
        material_bins = (
            (0, 300, "0-299"), (300, 1000, "300-999"),
            (1000, 2000, "1000-1999"),
            (2000, 4000, "2000-3999"),
            (4000, float("inf"), "4000+"))
        for low, high, label in material_bins:
            selected = (abs_material >= low) & (abs_material < high)
            count = int(selected.sum().item())
            if not count:
                continue
            print(
                f"  {label:>16} | {count:5d} | "
                f"{(qf[selected] - pf[selected]).abs().mean().item():9.6f} | "
                f"{(host['pred'][selected] - host['score'][selected]).abs().mean().item():8.1f} | "
                f"{ply[selected].mean().item():8.1f}")

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
        if self.simple_validation_cohort_report:
            self._simple_validation_cohorts = None
        optimizer = self.optimizers()
        param_groups = optimizer.param_groups
        if not param_groups:
            return
        current_lr = float(param_groups[0]["lr"])
        print("[Simple optimizer learning rates]")
        group_names = (
            "HalfKA_HM2 FT",
             ("9 bucket layer stacks + side projection"
              if self.use_projected_side_input else
              "9 bucket layer stacks + direct side"
              if self.use_direct_side_input else "9 bucket layer stacks"))
        if self.use_shared_psqt:
            group_names = (
                group_names[0], group_names[1] + " + shared PSQT")
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

    @staticmethod
    def _cohort_add(table, indices, prob_error, cp_error):
        rows = torch.stack((
            torch.ones_like(prob_error), prob_error, cp_error), dim=1)
        table.index_add_(0, indices, rows.float())

    def _accumulate_validation_cohort_stats(
            self, bucket, prob_error, cp_error, ply, material):
        if self._simple_validation_cohorts is None:
            device = bucket.device
            self._simple_validation_cohorts = {
                "bucket": torch.zeros((9, 3), device=device),
                "ply": torch.zeros((6, 3), device=device),
                "material": torch.zeros((5, 3), device=device),
            }
        stats = self._simple_validation_cohorts
        ply_index = torch.where(
            ply <= 20, 0, torch.where(
                ply <= 40, 1, torch.where(
                    ply <= 60, 2, torch.where(
                        ply <= 80, 3, torch.where(ply <= 100, 4, 5))))).long()
        absolute_material = material.abs()
        material_index = torch.where(
            absolute_material < 300, 0, torch.where(
                absolute_material < 1000, 1, torch.where(
                    absolute_material < 2000, 2, torch.where(
                        absolute_material < 4000, 3, 4)))).long()
        self._cohort_add(stats["bucket"], bucket, prob_error, cp_error)
        self._cohort_add(stats["ply"], ply_index, prob_error, cp_error)
        self._cohort_add(
            stats["material"], material_index, prob_error, cp_error)

    def on_validation_epoch_end(self):
        if not self.simple_validation_cohort_report \
                or self._simple_validation_cohorts is None:
            return
        labels = {
            "bucket": [f"B{index:02d}" for index in range(9)],
            "ply": ["1-20", "21-40", "41-60", "61-80", "81-100", "101+"],
            "material": ["0-299", "300-999", "1000-1999", "2000-3999", "4000+"],
        }
        print(f"[Simple Validation Cohorts](Step {int(self.global_step)})")
        for group in ("bucket", "ply", "material"):
            table = self._simple_validation_cohorts[group].detach().cpu().double()
            print(f"  {group}:")
            for index, label in enumerate(labels[group]):
                count = int(table[index, 0].item())
                if count:
                    print(
                        f"    {label}: n={count} "
                        f"prob_mae={table[index, 1].item() / count:.9f} "
                        f"cp_mae={table[index, 2].item() / count:.6f}")
        bucket = self._simple_validation_cohorts["bucket"].detach().cpu().double()
        non_count = int(bucket[:8, 0].sum().item())
        b08_count = int(bucket[8, 0].item())
        print(
            f"  non-B08: n={non_count} "
            f"prob_mae={bucket[:8, 1].sum().item() / max(non_count, 1):.9f} "
            f"cp_mae={bucket[:8, 2].sum().item() / max(non_count, 1):.6f}")
        print(
            f"  B08: n={b08_count} "
            f"prob_mae={bucket[8, 1].item() / max(b08_count, 1):.9f} "
            f"cp_mae={bucket[8, 2].item() / max(b08_count, 1):.6f}")

    def configure_optimizers(self):
        # A fresh optimizer is intentional for transplanted checkpoints.
        downstream = list(self.layer_stacks.parameters())
        if self.side_proj is not None:
            downstream.extend(self.side_proj.parameters())
        if self.shared_psqt is not None:
            downstream.extend(self.shared_psqt.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": list(self.input.parameters()), "lr": self.lr},
                {"params": downstream, "lr": self.lr},
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
        saved_side_input = bool(metadata.get("use_side_input", False))
        saved_side_type = metadata.get("simple_side_input_type", "none")
        saved_shared_psqt = bool(metadata.get("use_shared_psqt", False))
        if saved_side_input and saved_side_type != self.simple_side_input_type:
            raise ValueError(
                "Simple side-input type mismatch: "
                f"checkpoint={saved_side_type}, model={self.simple_side_input_type}")
        if saved_side_input and not self.use_side_input:
            raise ValueError(
                "side-input checkpoint cannot be loaded into the baseline "
                "Simple architecture")
        if not saved_side_input and self.use_side_input:
            # Weight-only baseline -> side migration. Lightning calls this hook
            # before load_state_dict(), so expand the nine FC1 tensors here.
            # The trainer rejects this path for full optimizer-state resume.
            state = checkpoint.get("state_dict", {})
            current = self.state_dict()
            for index in range(LAYER_STACKS):
                key = f"layer_stacks.{index}.fc1.weight"
                old = state.get(key)
                if old is None or tuple(old.shape) != (32, 30):
                    raise ValueError(
                        f"cannot migrate baseline FC1 tensor {key}: "
                        f"shape={None if old is None else tuple(old.shape)}")
                expanded = current[key].detach().clone()
                expanded[:, :30].copy_(old)
                expanded[:, 30:].zero_()
                state[key] = expanded
            if self.use_projected_side_input:
                for key in ("side_proj.0.weight", "side_proj.0.bias"):
                    state[key] = current[key].detach().clone()
        if saved_shared_psqt and not self.use_shared_psqt:
            raise ValueError(
                "shared-PSQT checkpoint cannot be loaded into the baseline "
                "Simple architecture")
        if not saved_shared_psqt and self.use_shared_psqt:
            # Weight-only baseline -> PSQT migration.  Preserve the exact
            # baseline at step zero while introducing the new trainable path.
            state = checkpoint.get("state_dict", {})
            current = self.state_dict()
            state["shared_psqt.weight"] = (
                current["shared_psqt.weight"].detach().clone())
            state["shared_psqt.scale"] = (
                current["shared_psqt.scale"].detach().clone())
        self.transplant_source = metadata.get("transplant_source")
        self.transplant_mapping_version = metadata.get("transplant_mapping_version")
