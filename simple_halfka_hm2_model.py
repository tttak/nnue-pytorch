"""Canonical HalfKA_hm2/no-DG SFNN baseline for Experiment 85.

This module is deliberately independent from model.NNUE.  Shared training
semantics are small, explicit functions here; no complex Router/FM/Cross/LCA
state can accidentally enter a simple checkpoint.
"""

from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from simple_bucket_execution import DEFAULT_MODE, METADATA_ATTRIBUTE, validate_mode, execute_heads
from simple_frequency_aware_optimizer import FrequencyAwareAdamW
from simple_quant_boundary import (
    MODES as QUANT_BOUNDARY_MODES, FORMULA_VERSION as QUANT_BOUNDARY_VERSION,
    CALIBRATED_PRESETS, boundary_penalty, touched_ft_rows, ft_regularization_due,
)
from simple_quant_hysteresis import (
    HysteresisAdamW, FrequencyHysteresisAdamW, VERSION as HYSTERESIS_VERSION,
    DEFAULT_COOLDOWN, DEFAULT_BAND, DEFAULT_MARGIN,
)
from simple_training_compatibility import validate_hysteresis_options

import model as complex_model
from simple_pp3wide import (
    PP3WIDE_FEATURES, PP3WIDE_INIT_MODES, PP3WIDE_MAPPING_VERSION,
    PP3WIDE_QUANT_SCALE, PP3WIDE_SCHEMA_VERSION, PP3WIDE_TYPE,
    PP3WIDE64_PROJECTION_OUTPUTS, PP3WIDE64_SCHEMA_VERSION,
    PP3WIDE64_TRANSFORMED, PP3WIDE64_TYPE, PP3WIDE64_WIDTH,
    PP3WIDE64_INIT_TABLE_QUANTUM,
)
from simple_local_pair64 import (
    LOCALPAIR64_FEATURES, LOCALPAIR64_INIT_TABLE_QUANTUM,
    LOCALPAIR64_MAPPING_VERSION, LOCALPAIR64_PROJECTION_OUTPUTS,
    LOCALPAIR64_SCHEMA_VERSION, LOCALPAIR64_TRANSFORMED, LOCALPAIR64_TYPE,
    LOCALPAIR64_WIDTH,
)
from simple_ksg_local_pair64 import (
    KSG_LOCALPAIR64_FEATURES, KSG_LOCALPAIR64_INIT_TABLE_QUANTUM,
    KSG_LOCALPAIR64_MAPPING_VERSION, KSG_LOCALPAIR64_PROJECTION_OUTPUTS,
    KSG_LOCALPAIR64_SCHEMA_VERSION, KSG_LOCALPAIR64_TRANSFORMED,
    KSG_LOCALPAIR64_TYPE, KSG_LOCALPAIR64_WIDTH,
)
from simple_gs_local_pair64 import (
    GS_LOCALPAIR64_FEATURES, GS_LOCALPAIR64_INIT_TABLE_QUANTUM,
    GS_LOCALPAIR64_MAPPING_VERSION, GS_LOCALPAIR64_PROJECTION_OUTPUTS,
    GS_LOCALPAIR64_SCHEMA_VERSION, GS_LOCALPAIR64_TRANSFORMED,
    GS_LOCALPAIR64_TYPE, GS_LOCALPAIR64_WIDTH,
)
from simple_gs_local_pair32 import (
    GS_LOCALPAIR32_FEATURES, GS_LOCALPAIR32_INIT_TABLE_QUANTUM,
    GS_LOCALPAIR32_MAPPING_VERSION, GS_LOCALPAIR32_PROJECTION_OUTPUTS,
    GS_LOCALPAIR32_SCHEMA_VERSION, GS_LOCALPAIR32_TRANSFORMED,
    GS_LOCALPAIR32_TYPE, GS_LOCALPAIR32_WIDTH,
)
from simple_gs_local_pair32_d1 import (
    GS_LOCALPAIR32_D1_FEATURES, GS_LOCALPAIR32_D1_INIT_TABLE_QUANTUM,
    GS_LOCALPAIR32_D1_MAPPING_VERSION, GS_LOCALPAIR32_D1_PROJECTION_OUTPUTS,
    GS_LOCALPAIR32_D1_SCHEMA_VERSION, GS_LOCALPAIR32_D1_TRANSFORMED,
    GS_LOCALPAIR32_D1_TYPE, GS_LOCALPAIR32_D1_WIDTH,
)

SIMPLE_SCHEMA_VERSION = 2
ARCHITECTURE_TYPE = "halfka_hm2_simple"
FEATURE_NAME = "HalfKA_HM2_NoDG"
FT_INPUTS = 73_305
FT_WIDTH = 1_536
HALFKA_HM2_KING_BUCKETS = 45
HALFKA_HM2_PLANES = 1_629
FT_VIRTUAL_FACTORIZATION_MODES = ("off", "shared")
FT_VIRTUAL_MAPPING_VERSION = "halfka_hm2_makeindex_mod1629_v1"
FT_FREQUENCY_LR_MODES = ("off", "mild", "medium")
FT_FREQUENCY_LR_PRESETS = {
    "mild": (0.125, 0.75, 1.50),
    "medium": (0.25, 0.50, 2.00),
}
LAYER_STACKS = 9
# Simple v2 fixed-point contract.  These values mirror the production C++
# inference implementation and serialize_halfka_hm2_simple.py; QAT must not
# silently inherit constants from Stockfish or from the Complex network.
FT_QUANT_SCALE = 127.0
FT_INIT_MODES = ("legacy", "qat_safe")
FT_INIT_VERSION = "legacy_rescale_v1"
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
SIMPLE_LOCAL_PAIR_FEATURES = (
    "off", PP3WIDE_TYPE, PP3WIDE64_TYPE, LOCALPAIR64_TYPE,
    KSG_LOCALPAIR64_TYPE, GS_LOCALPAIR64_TYPE, GS_LOCALPAIR32_TYPE,
    GS_LOCALPAIR32_D1_TYPE)
LOCALPAIR64_TYPES = (
    LOCALPAIR64_TYPE, KSG_LOCALPAIR64_TYPE, GS_LOCALPAIR64_TYPE)
LOCALPAIR_TYPES = (
    *LOCALPAIR64_TYPES, GS_LOCALPAIR32_TYPE, GS_LOCALPAIR32_D1_TYPE)
LOCALPAIR_SCHEMA = {
    LOCALPAIR64_TYPE: LOCALPAIR64_SCHEMA_VERSION,
    KSG_LOCALPAIR64_TYPE: KSG_LOCALPAIR64_SCHEMA_VERSION,
    GS_LOCALPAIR64_TYPE: GS_LOCALPAIR64_SCHEMA_VERSION,
    GS_LOCALPAIR32_TYPE: GS_LOCALPAIR32_SCHEMA_VERSION,
    GS_LOCALPAIR32_D1_TYPE: GS_LOCALPAIR32_D1_SCHEMA_VERSION,
}
LOCALPAIR_MAPPING = {
    LOCALPAIR64_TYPE: LOCALPAIR64_MAPPING_VERSION,
    KSG_LOCALPAIR64_TYPE: KSG_LOCALPAIR64_MAPPING_VERSION,
    GS_LOCALPAIR64_TYPE: GS_LOCALPAIR64_MAPPING_VERSION,
    GS_LOCALPAIR32_TYPE: GS_LOCALPAIR32_MAPPING_VERSION,
    GS_LOCALPAIR32_D1_TYPE: GS_LOCALPAIR32_D1_MAPPING_VERSION,
}
LOCALPAIR_FEATURE_COUNT = {
    LOCALPAIR64_TYPE: LOCALPAIR64_FEATURES,
    KSG_LOCALPAIR64_TYPE: KSG_LOCALPAIR64_FEATURES,
    GS_LOCALPAIR64_TYPE: GS_LOCALPAIR64_FEATURES,
    GS_LOCALPAIR32_TYPE: GS_LOCALPAIR32_FEATURES,
    GS_LOCALPAIR32_D1_TYPE: GS_LOCALPAIR32_D1_FEATURES,
}
LOCALPAIR_WIDTH = {
    LOCALPAIR64_TYPE: LOCALPAIR64_WIDTH,
    KSG_LOCALPAIR64_TYPE: KSG_LOCALPAIR64_WIDTH,
    GS_LOCALPAIR64_TYPE: GS_LOCALPAIR64_WIDTH,
    GS_LOCALPAIR32_TYPE: GS_LOCALPAIR32_WIDTH,
    GS_LOCALPAIR32_D1_TYPE: GS_LOCALPAIR32_D1_WIDTH,
}
LOCALPAIR_TABLE_QUANTUM = {
    LOCALPAIR64_TYPE: LOCALPAIR64_INIT_TABLE_QUANTUM,
    KSG_LOCALPAIR64_TYPE: KSG_LOCALPAIR64_INIT_TABLE_QUANTUM,
    GS_LOCALPAIR64_TYPE: GS_LOCALPAIR64_INIT_TABLE_QUANTUM,
    GS_LOCALPAIR32_TYPE: GS_LOCALPAIR32_INIT_TABLE_QUANTUM,
    GS_LOCALPAIR32_D1_TYPE: GS_LOCALPAIR32_D1_INIT_TABLE_QUANTUM,
}
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
    if value.shape[1] % 2:
        raise ValueError("EWM input width must be even")
    half = value.shape[1] // 2
    a, b = value[:, :half], value[:, half:]
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
    def __init__(self, num_inputs: int = FT_INPUTS, width: int = FT_WIDTH,
                 virtual_factorization: str = "off", initialization: str = "legacy"):
        super().__init__()
        if initialization not in FT_INIT_MODES:
            raise ValueError(f"FT initialization must be one of {FT_INIT_MODES}")
        self.initialization = initialization
        if virtual_factorization not in FT_VIRTUAL_FACTORIZATION_MODES:
            raise ValueError(
                "virtual_factorization must be one of "
                f"{FT_VIRTUAL_FACTORIZATION_MODES}, got "
                f"{virtual_factorization!r}")
        if num_inputs == FT_INPUTS and (
                FT_INPUTS != HALFKA_HM2_KING_BUCKETS * HALFKA_HM2_PLANES):
            raise AssertionError("HalfKA_HM2 factorization layout changed")
        self.virtual_factorization = virtual_factorization
        sigma = math.sqrt(1.0 / num_inputs)
        self.weight = nn.Parameter(torch.empty(num_inputs, width))
        self.bias = nn.Parameter(torch.empty(width))
        if virtual_factorization == "shared":
            self.virtual_weight = nn.Parameter(
                torch.zeros(HALFKA_HM2_PLANES, width))
            self.register_buffer(
                "virtual_index",
                torch.arange(num_inputs, dtype=torch.long).remainder(
                    HALFKA_HM2_PLANES),
                persistent=False)
        else:
            self.register_parameter("virtual_weight", None)
            self.register_buffer("virtual_index", None, persistent=False)
        nn.init.uniform_(self.weight, -sigma, sigma)
        # Preserve the legacy random draws, signs, and all following RNG state.
        # Only newly constructed scratch weights use this wider q127 range.
        self.initialization_width = (1.0 / FT_QUANT_SCALE
                                     if initialization == "qat_safe" else sigma)
        if initialization == "qat_safe":
            with torch.no_grad():
                self.weight.mul_(self.initialization_width / sigma)
        nn.init.uniform_(self.bias, -sigma, sigma)
        self._qat_eval_cache = None

    def effective_weight(self):
        """Return the deployment FT table without mutating training weights."""
        if self.virtual_weight is None:
            return self.weight
        return self.weight + self.virtual_weight.index_select(
            0, self.virtual_index)

    @torch.no_grad()
    def mean_decompose_(self, source_weight):
        """Function-preserving unfactorized -> shared migration.

        Every virtual group contains the 45 serialized king-bucket rows.  The
        decomposition deliberately includes rows that may be unreachable in a
        legal position: they are part of the existing inference table and this
        makes migration deterministic and reversible without corpus-dependent
        metadata.
        """
        if self.virtual_weight is None:
            raise RuntimeError("mean_decompose_ requires shared factorization")
        if tuple(source_weight.shape) != tuple(self.weight.shape):
            raise ValueError(
                f"FT shape mismatch: {tuple(source_weight.shape)} vs "
                f"{tuple(self.weight.shape)}")
        source = source_weight.to(device=self.weight.device,
                                  dtype=self.weight.dtype)
        grouped = source.view(
            HALFKA_HM2_KING_BUCKETS, HALFKA_HM2_PLANES, -1)
        virtual = grouped.mean(dim=0)
        specific = source - virtual.index_select(0, self.virtual_index)
        # Correct the last-bit cancellation where possible.  This keeps the
        # coalesced float table bit-identical for normal trained weights and,
        # more importantly, always preserves the production q127 table.
        for _ in range(2):
            reconstructed = specific + virtual.index_select(
                0, self.virtual_index)
            specific = specific + (source - reconstructed)
        # Addition/subtraction can still miss the source by one ULP.  Nudge
        # only those residual elements by one representable float toward the
        # value whose subsequent S+V addition rounds exactly to the source.
        expanded_virtual = virtual.index_select(0, self.virtual_index)
        for _ in range(4):
            reconstructed = specific + expanded_virtual
            mismatch = reconstructed != source
            if not mismatch.any():
                break
            direction = torch.where(
                reconstructed < source,
                torch.full_like(specific, float("inf")),
                torch.full_like(specific, float("-inf")))
            nudged = torch.nextafter(specific, direction)
            specific = torch.where(mismatch, nudged, specific)
        # The deployment contract is the q127 table.  Values exactly at a
        # rounding boundary can change bin after the one-ULP reconstruction
        # error above; nudge those rare elements toward the source bin.
        desired_q = torch.round(source * FT_QUANT_SCALE)
        for _ in range(4):
            reconstructed = specific + expanded_virtual
            actual_q = torch.round(reconstructed * FT_QUANT_SCALE)
            mismatch = actual_q != desired_q
            if not mismatch.any():
                break
            direction = torch.where(
                actual_q < desired_q,
                torch.full_like(specific, float("inf")),
                torch.full_like(specific, float("-inf")))
            specific = torch.where(
                mismatch, torch.nextafter(specific, direction), specific)
        self.virtual_weight.copy_(virtual)
        self.weight.copy_(specific)
        self._qat_eval_cache = None
        return self

    @torch.no_grad()
    def load_coalesced_(self, source_weight):
        """Load a deployment table into either parameterization."""
        if self.virtual_weight is None:
            self.weight.copy_(source_weight)
        else:
            self.mean_decompose_(source_weight)
        self._qat_eval_cache = None
        return self

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
            weight, bias = self.effective_weight(), self.bias
        elif not self.training and self._qat_eval_cache is not None:
            weight, bias = self._qat_eval_cache
        else:
            # nn.bin stores both FT tensors as int16 at scale 127. C++ doubles
            # them after loading; the exact EWM path below accounts for that.
            # QAT is applied after coalescing.  Quantizing S and V separately
            # would not match the single q127 FT table stored in nn.bin.
            weight = _fake_quantize(
                self.effective_weight(), FT_QUANT_SCALE,
                FT_QUANT_MIN, FT_QUANT_MAX)
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


class SimplePp3Wide(nn.Module):
    """Experiment 120 board-only unpromoted pawn/lance local-pair FT."""

    def __init__(self, width=FT_WIDTH, initialization="zero",
                 nonzero_rate=0.05, seed=120,
                 feature_count=PP3WIDE_FEATURES):
        super().__init__()
        if initialization not in PP3WIDE_INIT_MODES:
            raise ValueError(
                f"PP3Wide init must be one of {PP3WIDE_INIT_MODES}")
        if not 0.0 <= float(nonzero_rate) <= 1.0:
            raise ValueError("PP3Wide nonzero rate must be in [0,1]")
        self.initialization = initialization
        self.nonzero_rate = float(nonzero_rate)
        self.seed = int(seed)
        self.weight = nn.Parameter(torch.zeros(feature_count, width))
        if initialization == "quantized_random":
            # Construct directly on the exported int8 grid.  A local CPU RNG
            # keeps model initialization from perturbing the data-stream RNG.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed)
            with torch.no_grad():
                active = torch.rand(
                    self.weight.shape, generator=generator) < self.nonzero_rate
                signs = torch.randint(
                    0, 2, self.weight.shape, generator=generator,
                    dtype=torch.int8).float().mul_(2).sub_(1)
                self.weight.copy_(
                    active.float() * signs / PP3WIDE_QUANT_SCALE)
        self._qat_eval_cache = None

    def train(self, mode: bool = True):
        if mode:
            self._qat_eval_cache = None
        return super().train(mode)

    def forward(self, indices, batch_indices, batch_size, qat_mode="off"):
        if qat_mode == "off":
            weight = self.weight
        elif not self.training and self._qat_eval_cache is not None:
            weight = self._qat_eval_cache
        else:
            # int8 storage, scale 127.  C++ sign-extends q8 and doubles it
            # before merging with the q127*2 main accumulator.
            weight = _fake_quantize(
                self.weight, PP3WIDE_QUANT_SCALE, -127.0, 127.0)
            if not self.training:
                self._qat_eval_cache = weight.detach()
        if indices.numel() == 0:
            return weight.new_zeros((batch_size, weight.shape[1]))
        counts = torch.bincount(batch_indices, minlength=batch_size)
        offsets = torch.cat((
            counts.new_zeros(1), counts.cumsum(dim=0))).long()
        return F.embedding_bag(
            indices.long(), weight, offsets, mode="sum",
            include_last_offset=True)


class SimplePp3Wide64(SimplePp3Wide):
    """Experiment 121: 64-wide PP accumulator plus shared 64->16 projection."""

    def __init__(self, initialization="zero", table_nonzero_rate=0.05,
                 projection_nonzero_rate=0.05, seed=121,
                 feature_count=PP3WIDE_FEATURES,
                 table_quantum=PP3WIDE64_INIT_TABLE_QUANTUM,
                 latent_width=PP3WIDE64_WIDTH):
        super().__init__(
            width=latent_width, initialization="zero",
            nonzero_rate=table_nonzero_rate, seed=seed,
            feature_count=feature_count)
        if initialization not in PP3WIDE_INIT_MODES:
            raise ValueError(
                f"PP3Wide64 init must be one of {PP3WIDE_INIT_MODES}")
        if not 0.0 <= float(projection_nonzero_rate) <= 1.0:
            raise ValueError("PP3Wide64 projection nonzero rate must be in [0,1]")
        self.initialization = str(initialization)
        self.nonzero_rate = float(table_nonzero_rate)
        self.projection_nonzero_rate = float(projection_nonzero_rate)
        self.seed = int(seed)
        if latent_width <= 0 or latent_width % 2:
            raise ValueError("pair latent width must be positive and even")
        self.latent_width = int(latent_width)
        self.projection = nn.Linear(
            self.latent_width, PP3WIDE64_PROJECTION_OUTPUTS, bias=False)
        nn.init.zeros_(self.projection.weight)
        if initialization == "quantized_random":
            # Initialize both halves directly on their deployment grids. Local
            # generators keep the training-data RNG identical to PP OFF.
            table_generator = torch.Generator(device="cpu")
            table_generator.manual_seed(self.seed)
            projection_generator = torch.Generator(device="cpu")
            projection_generator.manual_seed(self.seed + 1)
            with torch.no_grad():
                table_active = torch.rand(
                    self.weight.shape, generator=table_generator) \
                    < self.nonzero_rate
                table_signs = torch.randint(
                    0, 2, self.weight.shape, generator=table_generator,
                    dtype=torch.int8).float().mul_(2).sub_(1)
                self.weight.copy_(
                    table_active.float() * table_signs
                    * table_quantum
                    / PP3WIDE_QUANT_SCALE)
                proj_active = torch.rand(
                    self.projection.weight.shape,
                    generator=projection_generator) \
                    < self.projection_nonzero_rate
                proj_signs = torch.randint(
                    0, 2, self.projection.weight.shape,
                    generator=projection_generator,
                    dtype=torch.int8).float().mul_(2).sub_(1)
                self.projection.weight.copy_(
                    proj_active.float() * proj_signs / DENSE_WEIGHT_SCALE)

    def project(self, white_accumulator, black_accumulator, us, them,
                qat_mode="off"):
        white = _simple_qat_ewm(white_accumulator, qat_mode)
        black = _simple_qat_ewm(black_accumulator, qat_mode)
        transformed = us * torch.cat((white, black), dim=1) \
            + them * torch.cat((black, white), dim=1)
        if qat_mode == "off":
            projected = F.linear(transformed, self.projection.weight, None)
        else:
            weight = _fake_quantize(
                self.projection.weight, DENSE_WEIGHT_SCALE,
                DENSE_WEIGHT_MIN, DENSE_WEIGHT_MAX)
            projected = F.linear(transformed, weight, None)
            if qat_mode in ("weight_activation", "full"):
                projected = _fake_raw_grid(projected)
        return projected, transformed


class SimpleLocalPair64(SimplePp3Wide64):
    """Experiment 122 L4 Chebyshev-distance-2 local piece-pair branch."""

    def __init__(self, initialization="zero", table_nonzero_rate=0.05,
                 projection_nonzero_rate=0.05, seed=122):
        super().__init__(
            initialization=initialization,
            table_nonzero_rate=table_nonzero_rate,
            projection_nonzero_rate=projection_nonzero_rate,
            seed=seed,
            feature_count=LOCALPAIR64_FEATURES,
            table_quantum=LOCALPAIR64_INIT_TABLE_QUANTUM)


class SimpleKsgLocalPair64(SimplePp3Wide64):
    """Experiment 123 R2 knight/silver/gold-like local-pair branch."""

    def __init__(self, initialization="zero", table_nonzero_rate=0.05,
                 projection_nonzero_rate=0.05, seed=123):
        super().__init__(
            initialization=initialization,
            table_nonzero_rate=table_nonzero_rate,
            projection_nonzero_rate=projection_nonzero_rate,
            seed=seed,
            feature_count=KSG_LOCALPAIR64_FEATURES,
            table_quantum=KSG_LOCALPAIR64_INIT_TABLE_QUANTUM)


class SimpleGsLocalPair64(SimplePp3Wide64):
    """Experiment 124 R5 silver/gold-like local-pair branch."""

    def __init__(self, initialization="zero", table_nonzero_rate=0.05,
                 projection_nonzero_rate=0.05, seed=124):
        super().__init__(
            initialization=initialization,
            table_nonzero_rate=table_nonzero_rate,
            projection_nonzero_rate=projection_nonzero_rate,
            seed=seed,
            feature_count=GS_LOCALPAIR64_FEATURES,
            table_quantum=GS_LOCALPAIR64_INIT_TABLE_QUANTUM)


class SimpleGsLocalPair32(SimplePp3Wide64):
    """Experiment 125: the exact R5 relation set with latent width 32."""

    def __init__(self, initialization="zero", table_nonzero_rate=0.05,
                 projection_nonzero_rate=0.05, seed=125):
        super().__init__(
            initialization=initialization,
            table_nonzero_rate=table_nonzero_rate,
            projection_nonzero_rate=projection_nonzero_rate,
            seed=seed,
            feature_count=GS_LOCALPAIR32_FEATURES,
            table_quantum=GS_LOCALPAIR32_INIT_TABLE_QUANTUM,
            latent_width=GS_LOCALPAIR32_WIDTH)


class SimpleGsLocalPair32D1(SimplePp3Wide64):
    """Experiment 126: R5 latent32 restricted to Chebyshev radius 1."""

    def __init__(self, initialization="zero", table_nonzero_rate=0.05,
                 projection_nonzero_rate=0.05, seed=126):
        super().__init__(
            initialization=initialization,
            table_nonzero_rate=table_nonzero_rate,
            projection_nonzero_rate=projection_nonzero_rate,
            seed=seed,
            feature_count=GS_LOCALPAIR32_D1_FEATURES,
            table_quantum=GS_LOCALPAIR32_D1_INIT_TABLE_QUANTUM,
            latent_width=GS_LOCALPAIR32_D1_WIDTH)


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

    def forward(self, x, side_input=None, fc0_residual=None,
                collect_diagnostics=False, qat_mode="off"):
        quantize_weights = qat_mode != "off"
        quantize_activations = qat_mode in ("weight_activation", "full")

        def affine(layer, value):
            if not quantize_weights:
                return layer(value)
            weight, bias = _fake_dense_parameters(layer)
            return F.linear(value, weight, bias)

        h0 = affine(self.fc0, x)
        main_fc0 = h0
        if fc0_residual is not None:
            h0 = h0 + fc0_residual
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
            if fc0_residual is not None:
                diagnostics.update({
                    "main_fc0_pre": main_fc0,
                    "pp_fc0_residual": fc0_residual,
                    "merged_fc0_pre": h0,
                })
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
        simple_quant_boundary_reg="off",
        simple_quant_boundary_band=None,
        simple_quant_boundary_ft_interval=None,
        simple_quant_boundary_ft_weight=0.0,
        simple_quant_boundary_dense_weight=0.0,
        simple_quant_boundary_fc0_weight=None,
        simple_quant_boundary_fc1_weight=None,
        simple_quant_boundary_output_weight=None,
        enforce_quant_boundary_resume_match=False,
        simple_qat_hysteresis="off",
        simple_qat_hysteresis_cooldown=DEFAULT_COOLDOWN,
        simple_qat_hysteresis_band=DEFAULT_BAND,
        simple_qat_hysteresis_restore_margin=DEFAULT_MARGIN,
        enforce_hysteresis_resume_match=False,
        simple_ft_virtual_factorization="off",
        simple_ft_init="legacy",
        simple_local_pair_feature="off",
        simple_bucket_mode="k3k3",
        simple_bucket_execution=DEFAULT_MODE,
        simple_ft_frequency_lr="off",
        simple_ft_frequency_table=None,
        simple_pp3wide_init="zero",
        simple_pp3wide_nonzero_rate=0.05,
        simple_pp3wide_seed=120,
        simple_pp3wide64_table_nonzero_rate=0.05,
        simple_pp3wide64_proj_nonzero_rate=0.05,
        simple_pp3wide64_seed=121,
        simple_localpair64_table_nonzero_rate=0.05,
        simple_localpair64_proj_nonzero_rate=0.05,
        simple_localpair64_seed=122,
        **unused,
    ):
        super().__init__()
        self.set_bucket_execution(simple_bucket_execution)
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
        self.simple_ft_virtual_factorization = str(
            simple_ft_virtual_factorization)
        if self.simple_ft_virtual_factorization not in \
                FT_VIRTUAL_FACTORIZATION_MODES:
            raise ValueError(
                "simple_ft_virtual_factorization must be one of "
                f"{FT_VIRTUAL_FACTORIZATION_MODES}")
        self.input = SimpleFeatureTransformer(
            virtual_factorization=self.simple_ft_virtual_factorization,
            initialization=simple_ft_init)
        self.simple_ft_init = self.input.initialization
        self.simple_ft_frequency_lr = str(simple_ft_frequency_lr)
        if self.simple_ft_frequency_lr not in FT_FREQUENCY_LR_MODES:
            raise ValueError(
                f"simple_ft_frequency_lr must be one of {FT_FREQUENCY_LR_MODES}")
        self.simple_ft_frequency_table = (
            str(simple_ft_frequency_table) if simple_ft_frequency_table else None)
        self.simple_ft_frequency_checksum = None
        self.simple_ft_frequency_alpha = None
        self.simple_ft_frequency_scale_min = None
        self.simple_ft_frequency_scale_max = None
        self.simple_ft_frequency_normalization = "off"
        self.register_buffer("simple_ft_frequency_scale", None,
                             persistent=False)
        if self.simple_ft_frequency_lr != "off":
            if not self.simple_ft_frequency_table:
                raise ValueError(
                    "--simple-ft-frequency-table is required when frequency LR is enabled")
            table_path = Path(self.simple_ft_frequency_table)
            raw_bytes = table_path.read_bytes()
            frequency = np.load(table_path).astype(np.float64)
            if frequency.shape != (FT_INPUTS,):
                raise ValueError(
                    f"frequency table shape must be ({FT_INPUTS},), got {frequency.shape}")
            alpha, scale_min, scale_max = FT_FREQUENCY_LR_PRESETS[
                self.simple_ft_frequency_lr]
            positive = frequency[frequency > 0]
            if not positive.size or frequency.sum() <= 0:
                raise ValueError("frequency table has no positive occurrences")
            reference = float(np.median(positive))
            raw_scale = (reference / (frequency + 1.0)) ** alpha
            left, right = 1e-6, 1e6
            for _ in range(100):
                normalizer = (left + right) * 0.5
                scale = np.clip(raw_scale / normalizer, scale_min, scale_max)
                weighted_mean = float(
                    np.dot(scale, frequency) / frequency.sum())
                if weighted_mean > 1.0:
                    left = normalizer
                else:
                    right = normalizer
            scale = np.clip(
                raw_scale / ((left + right) * 0.5), scale_min, scale_max)
            self.simple_ft_frequency_scale = torch.from_numpy(
                scale.astype(np.float32))
            self.simple_ft_frequency_checksum = hashlib.sha256(raw_bytes).hexdigest()
            self.simple_ft_frequency_alpha = alpha
            self.simple_ft_frequency_scale_min = scale_min
            self.simple_ft_frequency_scale_max = scale_max
            self.simple_ft_frequency_normalization = "occurrence_weighted_mean_1"
        self._frequency_optimizer = None
        self.simple_local_pair_feature = str(simple_local_pair_feature)
        self.simple_bucket_mode = str(simple_bucket_mode)
        if self.simple_bucket_mode not in ("k3k3", "phase9", "kingfree_tree"):
            raise ValueError("simple_bucket_mode must be k3k3, phase9, or kingfree_tree")
        if self.simple_local_pair_feature not in SIMPLE_LOCAL_PAIR_FEATURES:
            raise ValueError(
                "simple_local_pair_feature must be one of "
                f"{SIMPLE_LOCAL_PAIR_FEATURES}")
        # Stable diagnostic label for freshly constructed and migrated models.
        # Older pickled models happened to carry this attribute dynamically;
        # constructor-created models must define it explicitly as well.
        self.pp3wide_type = self.simple_local_pair_feature
        if self.simple_local_pair_feature == PP3WIDE_TYPE:
            self.pp3wide = SimplePp3Wide(
                initialization=simple_pp3wide_init,
                nonzero_rate=simple_pp3wide_nonzero_rate,
                seed=simple_pp3wide_seed)
        elif self.simple_local_pair_feature == PP3WIDE64_TYPE:
            self.pp3wide = SimplePp3Wide64(
                initialization=simple_pp3wide_init,
                table_nonzero_rate=simple_pp3wide64_table_nonzero_rate,
                projection_nonzero_rate=simple_pp3wide64_proj_nonzero_rate,
                seed=simple_pp3wide64_seed)
        elif self.simple_local_pair_feature == LOCALPAIR64_TYPE:
            self.pp3wide = SimpleLocalPair64(
                initialization=simple_pp3wide_init,
                table_nonzero_rate=simple_localpair64_table_nonzero_rate,
                projection_nonzero_rate=simple_localpair64_proj_nonzero_rate,
                seed=simple_localpair64_seed)
        elif self.simple_local_pair_feature == KSG_LOCALPAIR64_TYPE:
            self.pp3wide = SimpleKsgLocalPair64(
                initialization=simple_pp3wide_init,
                table_nonzero_rate=simple_localpair64_table_nonzero_rate,
                projection_nonzero_rate=simple_localpair64_proj_nonzero_rate,
                seed=simple_localpair64_seed)
        elif self.simple_local_pair_feature == GS_LOCALPAIR64_TYPE:
            self.pp3wide = SimpleGsLocalPair64(
                initialization=simple_pp3wide_init,
                table_nonzero_rate=simple_localpair64_table_nonzero_rate,
                projection_nonzero_rate=simple_localpair64_proj_nonzero_rate,
                seed=simple_localpair64_seed)
        elif self.simple_local_pair_feature == GS_LOCALPAIR32_TYPE:
            self.pp3wide = SimpleGsLocalPair32(
                initialization=simple_pp3wide_init,
                table_nonzero_rate=simple_localpair64_table_nonzero_rate,
                projection_nonzero_rate=simple_localpair64_proj_nonzero_rate,
                seed=simple_localpair64_seed)
        elif self.simple_local_pair_feature == GS_LOCALPAIR32_D1_TYPE:
            self.pp3wide = SimpleGsLocalPair32D1(
                initialization=simple_pp3wide_init,
                table_nonzero_rate=simple_localpair64_table_nonzero_rate,
                projection_nonzero_rate=simple_localpair64_proj_nonzero_rate,
                seed=simple_localpair64_seed)
        else:
            self.pp3wide = None
        self.simple_pp3wide_init = str(simple_pp3wide_init)
        self.simple_pp3wide_nonzero_rate = float(simple_pp3wide_nonzero_rate)
        self.simple_pp3wide_seed = int(simple_pp3wide_seed)
        self.simple_pp3wide64_table_nonzero_rate = float(
            simple_pp3wide64_table_nonzero_rate)
        self.simple_pp3wide64_proj_nonzero_rate = float(
            simple_pp3wide64_proj_nonzero_rate)
        self.simple_pp3wide64_seed = int(simple_pp3wide64_seed)
        self.simple_localpair64_table_nonzero_rate = float(
            simple_localpair64_table_nonzero_rate)
        self.simple_localpair64_proj_nonzero_rate = float(
            simple_localpair64_proj_nonzero_rate)
        self.simple_localpair64_seed = int(simple_localpair64_seed)
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
        self.simple_quant_boundary_reg = str(simple_quant_boundary_reg)
        if self.simple_quant_boundary_reg not in QUANT_BOUNDARY_MODES:
            raise ValueError("unknown Simple quantization boundary mode")
        default_band = .025 if self.simple_quant_boundary_reg == "weak" else .05
        if simple_quant_boundary_band is None:
            simple_quant_boundary_band = default_band
        if simple_quant_boundary_ft_interval is None:
            simple_quant_boundary_ft_interval = (
                8 if self.simple_quant_boundary_reg == "weak" else 1)
        ft_regularization_due(0, simple_quant_boundary_ft_interval)
        self.simple_quant_boundary_ft_interval = int(simple_quant_boundary_ft_interval)
        if self.simple_quant_boundary_reg != "off" and self.simple_qat_mode != "full":
            raise ValueError("quantization boundary regularization requires full QAT")
        if (self.simple_quant_boundary_reg != "off" and
                (simple_ft_virtual_factorization != "off" or
                 simple_ft_frequency_lr != "off" or
                 simple_local_pair_feature != "off")):
            raise ValueError("boundary experiment requires virtual/frequency/LocalPair OFF")
        if (self.simple_quant_boundary_reg in CALIBRATED_PRESETS and
                not simple_quant_boundary_ft_weight and
                not simple_quant_boundary_dense_weight and
                all(x is None for x in (
                    simple_quant_boundary_fc0_weight,
                    simple_quant_boundary_fc1_weight,
                    simple_quant_boundary_output_weight))):
            if float(simple_quant_boundary_band) != default_band:
                raise ValueError(
                    "boundary band override requires explicit calibrated weights")
            preset = CALIBRATED_PRESETS[self.simple_quant_boundary_reg]
            simple_quant_boundary_ft_weight = preset["ft"]
            simple_quant_boundary_fc0_weight = preset["fc0"]
            simple_quant_boundary_fc1_weight = preset["fc1"]
            simple_quant_boundary_output_weight = preset["fc2"]
        self.simple_quant_boundary_band = float(simple_quant_boundary_band)
        self.simple_quant_boundary_ft_weight = float(simple_quant_boundary_ft_weight)
        self.simple_quant_boundary_dense_weight = float(simple_quant_boundary_dense_weight)
        self.simple_quant_boundary_layer_weights = {
            name: (self.simple_quant_boundary_dense_weight if value is None
                   else float(value))
            for name, value in (
                ("fc0", simple_quant_boundary_fc0_weight),
                ("fc1", simple_quant_boundary_fc1_weight),
                ("fc2", simple_quant_boundary_output_weight))}
        self.enforce_quant_boundary_resume_match = bool(
            enforce_quant_boundary_resume_match)
        if not 0.0 < self.simple_quant_boundary_band < 0.5:
            raise ValueError("boundary band must be in (0, 0.5)")
        if (self.simple_quant_boundary_ft_weight < 0.0 or
                self.simple_quant_boundary_dense_weight < 0.0 or
                any(x < 0.0 for x in self.simple_quant_boundary_layer_weights.values())):
            raise ValueError("boundary weights must be nonnegative")
        if (self.simple_quant_boundary_reg != "off" and
                not (self.simple_quant_boundary_ft_weight or
                     any(self.simple_quant_boundary_layer_weights.values()))):
            raise ValueError("enabled boundary mode requires calibrated weights")
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
        self.simple_qat_hysteresis = str(simple_qat_hysteresis)
        self.simple_qat_hysteresis_cooldown = int(simple_qat_hysteresis_cooldown)
        self.simple_qat_hysteresis_band = float(simple_qat_hysteresis_band)
        self.simple_qat_hysteresis_restore_margin = float(simple_qat_hysteresis_restore_margin)
        self.enforce_hysteresis_resume_match = bool(enforce_hysteresis_resume_match)
        self._hysteresis_rows = None
        if self.simple_qat_hysteresis not in ("off", "anti_flip"):
            raise ValueError("unknown Simple QAT hysteresis mode")
        if self.simple_qat_hysteresis != "off":
            validate_hysteresis_options(mode=self.simple_qat_hysteresis,
                qat_mode=self.simple_qat_mode, boundary=self.simple_quant_boundary_reg,
                factor=self.simple_ft_virtual_factorization, local_pair=self.simple_local_pair_feature,
                side=self.use_side_input, psqt=self.use_shared_psqt,
                bucket_importance=self.use_bucket_importance_base_loss)
            if not 1 <= self.simple_qat_hysteresis_cooldown <= 63:
                raise ValueError("hysteresis cooldown must be 1..63")
            if not 0 < self.simple_qat_hysteresis_band < .5 or not 0 < self.simple_qat_hysteresis_restore_margin < .5:
                raise ValueError("hysteresis band/margin must be in (0,.5)")
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
        if self.simple_local_pair_feature in (
                PP3WIDE64_TYPE, *LOCALPAIR_TYPES):
            self.weight_clipping.append({
                "name": ("LocalPair64Projection"
                         if self.simple_local_pair_feature in LOCALPAIR_TYPES
                         else "PP64Projection"),
                "params": [self.pp3wide.projection.weight],
                "min_weight": -MAX_HIDDEN_WEIGHT,
                "max_weight": MAX_HIDDEN_WEIGHT,
            })
        # The ranking objectives are architecture-independent.  Reuse the
        # production implementation verbatim, but allow the full gradient to
        # reach the simple FT because there is no complex-path gradient
        # balancing to preserve here.
        self.pairwise_ft_grad_scale = 1.0
        self.listwise_ft_grad_scale = 1.0
        self.ranking_disagreement_weight = 1.0
        self.save_hyperparameters(ignore=("feature_set", "unused", "simple_bucket_execution"))
        if self.simple_qat_hysteresis == "off":
            for key in ("simple_qat_hysteresis", "simple_qat_hysteresis_cooldown",
                        "simple_qat_hysteresis_band", "simple_qat_hysteresis_restore_margin",
                        "enforce_hysteresis_resume_match"):
                self.hparams.pop(key, None)
        if self.simple_quant_boundary_reg == "off":
            # Preserve the historical OFF checkpoint hyperparameter payload.
            for key in (
                    "simple_quant_boundary_reg", "simple_quant_boundary_band",
                    "simple_quant_boundary_ft_interval",
                    "simple_quant_boundary_ft_weight",
                    "simple_quant_boundary_dense_weight",
                    "simple_quant_boundary_fc0_weight",
                    "simple_quant_boundary_fc1_weight",
                    "simple_quant_boundary_output_weight",
                    "enforce_quant_boundary_resume_match"):
                self.hparams.pop(key, None)

    def architecture_metadata(self, transplant_source=None, mapping_version=None):
        use_side_input = bool(getattr(self, "use_side_input", False))
        metadata = {
            "architecture_type": ARCHITECTURE_TYPE,
            "feature": FEATURE_NAME,
            "ft_width": FT_WIDTH,
            "layer_stack_count": LAYER_STACKS,
            "bucket_scheme": self.simple_bucket_mode,
            "simple_bucket_mode": self.simple_bucket_mode,
            "distinguish_golds": False,
            "long_effect_required": False,
            "simple_schema_version": SIMPLE_SCHEMA_VERSION,
            # Training-only parameterization.  It does not participate in the
            # C++ architecture/hash/schema because export coalesces the table.
            "simple_ft_virtual_factorization": (
                self.simple_ft_virtual_factorization),
            # Initialization provenance only; never part of the runtime hash.
            "simple_ft_init": getattr(self, "simple_ft_init", "legacy"),
            "simple_ft_init_width": (1.0 / FT_QUANT_SCALE
                if getattr(self, "simple_ft_init", "legacy") == "qat_safe"
                else math.sqrt(1.0 / FT_INPUTS)),
            "simple_ft_init_formula_version": FT_INIT_VERSION,
            "simple_ft_virtual_mapping_version": (
                FT_VIRTUAL_MAPPING_VERSION
                if self.simple_ft_virtual_factorization == "shared" else None),
            "simple_ft_frequency_lr": self.simple_ft_frequency_lr,
            "simple_ft_frequency_table_checksum": self.simple_ft_frequency_checksum,
            "simple_ft_frequency_alpha": self.simple_ft_frequency_alpha,
            "simple_ft_frequency_scale_min": self.simple_ft_frequency_scale_min,
            "simple_ft_frequency_scale_max": self.simple_ft_frequency_scale_max,
            "simple_ft_frequency_normalization": self.simple_ft_frequency_normalization,
            "simple_local_pair_feature": self.simple_local_pair_feature,
            "simple_pp3wide_schema_version": (
                (LOCALPAIR_SCHEMA[self.simple_local_pair_feature]
                 if self.simple_local_pair_feature in LOCALPAIR_TYPES
                 else PP3WIDE64_SCHEMA_VERSION
                 if self.simple_local_pair_feature == PP3WIDE64_TYPE
                 else PP3WIDE_SCHEMA_VERSION)
                if self.pp3wide is not None else None),
            "simple_pp3wide_mapping_version": (
                (LOCALPAIR_MAPPING[self.simple_local_pair_feature]
                 if self.simple_local_pair_feature in LOCALPAIR_TYPES
                 else PP3WIDE_MAPPING_VERSION)
                if self.pp3wide is not None else None),
            "simple_pp3wide_dimensions": (
                (LOCALPAIR_FEATURE_COUNT[self.simple_local_pair_feature]
                 if self.simple_local_pair_feature in LOCALPAIR_TYPES
                 else PP3WIDE_FEATURES)
                if self.pp3wide is not None else 0),
            "simple_pp3wide_export_dtype": (
                "int8_q127" if self.pp3wide is not None else "none"),
            "simple_pp3wide_width": (
                LOCALPAIR_WIDTH[self.simple_local_pair_feature]
                if self.simple_local_pair_feature in LOCALPAIR_TYPES
                else PP3WIDE64_WIDTH
                if self.simple_local_pair_feature == PP3WIDE64_TYPE
                else FT_WIDTH if self.pp3wide is not None else 0),
            "simple_pp3wide_projection": (
                f"shared_{self.pp3wide.projection.in_features}x16_int8_q64_no_bias"
                if self.simple_local_pair_feature in (
                    PP3WIDE64_TYPE, *LOCALPAIR_TYPES)
                else "none"),
            "simple_pp3wide_init": (
                self.simple_pp3wide_init if self.pp3wide is not None else None),
            "simple_pp3wide_nonzero_rate": (
                self.simple_pp3wide_nonzero_rate
                if self.pp3wide is not None else 0.0),
            "simple_pp3wide64_table_nonzero_rate": (
                self.simple_pp3wide64_table_nonzero_rate
                if self.simple_local_pair_feature == PP3WIDE64_TYPE else 0.0),
            "simple_pp3wide64_proj_nonzero_rate": (
                self.simple_pp3wide64_proj_nonzero_rate
                if self.simple_local_pair_feature == PP3WIDE64_TYPE else 0.0),
            "simple_pp3wide64_seed": (
                self.simple_pp3wide64_seed
                if self.simple_local_pair_feature == PP3WIDE64_TYPE else None),
            "simple_pp3wide64_init_table_quantum": (
                PP3WIDE64_INIT_TABLE_QUANTUM
                if self.simple_local_pair_feature == PP3WIDE64_TYPE else None),
            "simple_localpair64_table_nonzero_rate": (
                self.simple_localpair64_table_nonzero_rate
                if self.simple_local_pair_feature in LOCALPAIR_TYPES else 0.0),
            "simple_localpair64_proj_nonzero_rate": (
                self.simple_localpair64_proj_nonzero_rate
                if self.simple_local_pair_feature in LOCALPAIR_TYPES else 0.0),
            "simple_localpair64_seed": (
                self.simple_localpair64_seed
                if self.simple_local_pair_feature in LOCALPAIR_TYPES else None),
            "simple_localpair64_init_table_quantum": (
                LOCALPAIR_TABLE_QUANTUM[self.simple_local_pair_feature]
                if self.simple_local_pair_feature in LOCALPAIR_TYPES else None),
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
        if self.simple_quant_boundary_reg != "off":
            metadata.update({
                "simple_quant_boundary_reg": self.simple_quant_boundary_reg,
                "simple_quant_boundary_band": self.simple_quant_boundary_band,
                "simple_quant_boundary_ft_interval": self.simple_quant_boundary_ft_interval,
                "simple_quant_boundary_ft_weight": self.simple_quant_boundary_ft_weight,
                "simple_quant_boundary_dense_weight": self.simple_quant_boundary_dense_weight,
                "simple_quant_boundary_layer_weights":
                    self.simple_quant_boundary_layer_weights,
                "simple_quant_boundary_formula": QUANT_BOUNDARY_VERSION,
            })
        return metadata

    def set_feature_set(self, feature_set):
        if feature_set.name != FEATURE_NAME:
            raise ValueError(f"simple checkpoint requires {FEATURE_NAME}")
        self.feature_set = feature_set

    def set_bucket_execution(self, mode):
        # Runtime policy only: deliberately absent from hparams/state/schema.
        self.simple_bucket_execution = validate_mode(mode)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Metadata belongs to this batch's bucket tensor, not mutable model state.
        metadata = getattr(batch[8], METADATA_ATTRIBUTE, None)
        result = super().transfer_batch_to_device(batch, device, dataloader_idx)
        if metadata is not None:
            setattr(result[8], METADATA_ATTRIBUTE, metadata.to(device))
        return result

    def forward(self, us, them, white_indices, white_values, black_indices,
                black_values, layer_stack_indices, ply=None, material=None,
                pp3wide_white_indices=None,
                pp3wide_white_batch_indices=None,
                pp3wide_black_indices=None,
                pp3wide_black_batch_indices=None,
                **unused):
        tw, tb = self.input(
            white_indices, white_values, black_indices, black_values,
            qat_mode=self.simple_qat_mode)
        us = us.view(-1, 1)
        them = them.view(-1, 1)
        pp_white = pp_black = None
        pp_fc0_residual = pp_transformed = None
        if self.pp3wide is not None:
            required = (
                pp3wide_white_indices, pp3wide_white_batch_indices,
                pp3wide_black_indices, pp3wide_black_batch_indices)
            if any(value is None for value in required):
                raise ValueError("PP3Wide enabled model requires sparse PP rows")
            batch_size = white_indices.shape[0]
            pp_white = self.pp3wide(
                pp3wide_white_indices, pp3wide_white_batch_indices,
                batch_size, self.simple_qat_mode)
            pp_black = self.pp3wide(
                pp3wide_black_indices, pp3wide_black_batch_indices,
                batch_size, self.simple_qat_mode)
            if self.simple_local_pair_feature == PP3WIDE_TYPE:
                tw = tw + pp_white
                tb = tb + pp_black
            else:
                pp_fc0_residual, pp_transformed = self.pp3wide.project(
                    pp_white, pp_black, us, them, self.simple_qat_mode)
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
        metadata = getattr(layer_stack_indices, METADATA_ATTRIBUTE, None)
        if getattr(self, "simple_bucket_execution", DEFAULT_MODE) == "index_reuse" and metadata is not None:
            out = execute_heads(self, transformed, side_h, pp_fc0_residual,
                                metadata, collect, out, diagnostic_parts,
                                psqt_deep, psqt_shortcut)
        else:
            # Exact legacy path; metadata-free callers safely keep this path.
            for bucket, stack in enumerate(self.layer_stacks):
                mask = buckets == bucket
                if mask.any():
                    if collect:
                        bucket_out, bucket_diagnostics = stack(
                            transformed[mask],
                            None if side_h is None else side_h[mask],
                            None if pp_fc0_residual is None
                            else pp_fc0_residual[mask],
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
                            None if pp_fc0_residual is None
                            else pp_fc0_residual[mask],
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
            if self.pp3wide is not None:
                pp_cat = torch.cat((pp_white, pp_black)).float()
                pp_indices = torch.cat((
                    pp3wide_white_indices.long(),
                    pp3wide_black_indices.long()))
                self._simple_debug_snapshot["pp3wide"] = {
                    "variant": self.simple_local_pair_feature,
                    "accumulator_rms": pp_cat.square().mean().sqrt(),
                    "mean_abs": pp_cat.abs().mean(),
                    "merged_ratio": (
                        pp_cat.square().mean().sqrt()
                        / torch.cat((tw, tb)).float().square().mean().sqrt()
                          .clamp_min(1e-12))
                    if self.simple_local_pair_feature == PP3WIDE_TYPE
                    else pp_fc0_residual.float().abs().mean()
                         / activation_tensors["fc0_pre"].float().abs().mean()
                           .clamp_min(1e-12),
                    "active_mean": (
                        (pp3wide_white_indices.numel()
                         + pp3wide_black_indices.numel())
                        / max(2 * white_indices.shape[0], 1)),
                    "active_total": int(pp_indices.numel()),
                    "unique_rows": int(torch.unique(pp_indices).numel()),
                    "feature_count": int(self.pp3wide.weight.shape[0]),
                    "nonzero_weight_ratio": (
                        (torch.round(self.pp3wide.weight.detach()
                                     * PP3WIDE_QUANT_SCALE) != 0)
                        .float().mean()),
                }
                if self.simple_local_pair_feature in (
                        PP3WIDE64_TYPE, *LOCALPAIR_TYPES):
                    self._simple_debug_snapshot["pp3wide"].update({
                        "transformed_abs_mean": pp_transformed.float().abs().mean(),
                        "projection_abs_mean": pp_fc0_residual.float().abs().mean(),
                        "projection_nonzero_weight_ratio": (
                            (torch.round(
                                self.pp3wide.projection.weight.detach()
                                * DENSE_WEIGHT_SCALE) != 0).float().mean()),
                    })
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
        if stage == "train" and self.simple_qat_hysteresis != "off":
            self._hysteresis_rows = touched_ft_rows(wi, wv, bi, bv)
        if stage == "train" and self.simple_ft_frequency_lr != "off":
            touched = torch.unique(torch.cat((wi[wi >= 0], bi[bi >= 0])))
            if self._frequency_optimizer is None:
                raise RuntimeError("frequency-aware optimizer is not configured")
            self._frequency_optimizer.set_touched_rows(touched)
        pp_kwargs = {}
        if self.pp3wide is not None:
            if len(optional) < 4:
                raise ValueError("PP3Wide batch is missing four sparse tensors")
            pp = optional[-4:]
            optional = optional[:-4]
            pp_kwargs = {
                "pp3wide_white_indices": pp[0],
                "pp3wide_white_batch_indices": pp[1],
                "pp3wide_black_indices": pp[2],
                "pp3wide_black_batch_indices": pp[3],
            }
        pred_cp = self(
            us, them, wi, wv, bi, bv, bucket,
            ply=ply, material=material, **pp_kwargs).view(-1) * self.nnue2score
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
        if stage == "train" and self.simple_quant_boundary_reg != "off":
            boundary_total, ft_reg, dense_regs, ft_applied = self.quant_boundary_loss(
                wi, wv, bi, bv, self.global_step)
            total = total + boundary_total
            self.log("train/quant_boundary_ft_applied", float(ft_applied),
                     on_step=True, on_epoch=False, batch_size=score.numel())
            self.log("train/quant_boundary_ft", ft_reg.detach(), on_step=True,
                     on_epoch=False, batch_size=score.numel())
            for name, reg in zip(("fc0", "fc1", "output"), dense_regs):
                self.log(f"train/quant_boundary_{name}", reg.detach(),
                         on_step=True, on_epoch=False,
                         batch_size=score.numel())
            self.log("train/quant_boundary_total", boundary_total.detach(),
                     on_step=True, on_epoch=False,
                     batch_size=score.numel())

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

    def quant_boundary_loss(self, wi, wv, bi, bv, global_step):
        """Training-only penalty; skipped FT steps do no unique/index selection.

        Callers supply optimizer global_step explicitly for offline probes.
        No interval multiplier is applied to lambda on execution steps.
        """
        ft_reg = self.input.weight.new_zeros(())
        ft_applied = (self.simple_quant_boundary_reg != "off" and
                      self.simple_quant_boundary_ft_weight != 0 and
                      ft_regularization_due(global_step, self.simple_quant_boundary_ft_interval))
        if ft_applied:
            touched = touched_ft_rows(wi, wv, bi, bv)
            if touched.numel():
                ft_reg = boundary_penalty(
                    self.input.weight.index_select(0, touched),
                    FT_QUANT_SCALE, FT_QUANT_MIN, FT_QUANT_MAX,
                    self.simple_quant_boundary_band)
        dense_regs = []
        for name in ("fc0", "fc1", "fc2"):
            if self.simple_quant_boundary_reg == "weak":
                # Equal-size bucket tensors: one concatenated mean is the
                # same margin objective as mean(per-bucket means). Fewer
                # small CUDA launches keeps skipped-FT steps inexpensive.
                weights = torch.cat([
                    getattr(stack, name).weight.reshape(-1)
                    for stack in self.layer_stacks])
                dense_regs.append(boundary_penalty(
                    weights, DENSE_WEIGHT_SCALE, DENSE_WEIGHT_MIN, DENSE_WEIGHT_MAX,
                    self.simple_quant_boundary_band))
                continue
            dense_regs.append(torch.stack([
                boundary_penalty(getattr(stack, name).weight,
                                 DENSE_WEIGHT_SCALE, DENSE_WEIGHT_MIN, DENSE_WEIGHT_MAX,
                                 self.simple_quant_boundary_band)
                for stack in self.layer_stacks]).mean())
        total = self.simple_quant_boundary_ft_weight * ft_reg + sum(
            self.simple_quant_boundary_layer_weights[name] * reg
            for name, reg in zip(("fc0", "fc1", "fc2"), dense_regs))
        return total, ft_reg, dense_regs, ft_applied

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
        gradient = (
            gradients[0] if len(gradients) == 1
            else torch.cat(gradients) if gradients
            else None)
        grad_mean = (gradient.abs().mean()
                     if gradient is not None else weight.new_zeros(()))
        grad_norm = (gradient.norm()
                     if gradient is not None else weight.new_zeros(()))
        return {
            "grad_mean": grad_mean, "grad_norm": grad_norm,
            "active": int(weight.numel() + bias.numel()
                          if active is None else active),
            "w_norm": weight.norm(),
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
            if self.pp3wide is not None:
                pair_label = (
                    ("KSGLocalPair64_Table"
                     if self.simple_local_pair_feature == KSG_LOCALPAIR64_TYPE
                     else "LocalPair64_Table")
                    if self.simple_local_pair_feature in LOCALPAIR_TYPES
                    else "PP3Wide64_Table"
                    if self.simple_local_pair_feature == PP3WIDE64_TYPE
                    else "PP3Wide_Table")
                pair_weight = self.pp3wide.weight
                pair_zero_bias = pair_weight.new_zeros(1)
                layer_stats[pair_label] = self._layer_stat_row(
                    pair_weight, pair_zero_bias,
                    pair_weight.grad, None,
                    active=int(pair_weight.numel()))
                if self.simple_local_pair_feature in (
                        PP3WIDE64_TYPE, *LOCALPAIR_TYPES):
                    projection = self.pp3wide.projection.weight
                    projection_label = (
                        ("KSGLocalPair64_Proj"
                         if self.simple_local_pair_feature == KSG_LOCALPAIR64_TYPE
                         else "LocalPair64_Proj")
                        if self.simple_local_pair_feature in LOCALPAIR_TYPES
                        else "PP3Wide64_Proj")
                    layer_stats[projection_label] = self._layer_stat_row(
                        projection, projection.new_zeros(1),
                        projection.grad, None,
                        active=int(projection.numel()))
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

        if self.pp3wide is not None and "pp3wide" in snapshot:
            pp = snapshot["pp3wide"]
            pair_heading = (
                ("KSG LocalPair64 Stats"
                 if pp.get("variant") == KSG_LOCALPAIR64_TYPE
                 else "LocalPair64 Stats")
                if pp.get("variant") in LOCALPAIR_TYPES
                else "PP3Wide64 Stats"
                if pp.get("variant") == PP3WIDE64_TYPE
                else "PP3Wide Stats")
            print(f"[{pair_heading}](Step {display_step}, current batch snapshot)")
            print(
                "  accumulator RMS / mean abs : "
                f"{host_scalar(pp['accumulator_rms']):.6e} / "
                f"{host_scalar(pp['mean_abs']):.6e}")
            print(
                "  PP / merged FT RMS ratio   : "
                f"{host_scalar(pp['merged_ratio']):.6e}")
            print(
                "  active features/perspective: "
                f"{float(pp['active_mean']):.3f}")
            print(
                "  active total / unique rows : "
                f"{int(pp['active_total']):,} / {int(pp['unique_rows']):,} "
                f"({int(pp['unique_rows']) / max(int(pp['feature_count']), 1):.4%} "
                "of table)")
            print(
                "  quantized nonzero weights   : "
                f"{host_scalar(pp['nonzero_weight_ratio']):.4%}")
            if pp.get("variant") in (PP3WIDE64_TYPE, *LOCALPAIR_TYPES):
                print(
                    "  EWM transformed abs mean    : "
                    f"{host_scalar(pp['transformed_abs_mean']):.6e}")
                print(
                    "  projected residual abs mean : "
                    f"{host_scalar(pp['projection_abs_mean']):.6e}")
                print(
                    "  residual / FC0 pre-act ratio: "
                    f"{host_scalar(pp['merged_ratio']):.6e}")
                print(
                    "  projection q-nonzero weights: "
                    f"{host_scalar(pp['projection_nonzero_weight_ratio']):.4%}")
                table_key = (
                    ("KSGLocalPair64_Table"
                     if pp.get("variant") == KSG_LOCALPAIR64_TYPE
                     else "LocalPair64_Table")
                    if pp.get("variant") in LOCALPAIR_TYPES
                    else "PP3Wide64_Table")
                projection_key = (
                    ("KSGLocalPair64_Proj"
                     if pp.get("variant") == KSG_LOCALPAIR64_TYPE
                     else "LocalPair64_Proj")
                    if pp.get("variant") in LOCALPAIR_TYPES
                    else "PP3Wide64_Proj")
                layer_stats = self._simple_debug_layer_stats or {}
                if table_key in layer_stats and projection_key in layer_stats:
                    table_stats = layer_stats[table_key]
                    projection_stats = layer_stats[projection_key]
                    print(
                        "  table weight / grad norm    : "
                        f"{host_scalar(table_stats['w_norm']):.6e} / "
                        f"{host_scalar(table_stats['grad_norm']):.6e}")
                    print(
                        "  projection weight/grad norm : "
                        f"{host_scalar(projection_stats['w_norm']):.6e} / "
                        f"{host_scalar(projection_stats['grad_norm']):.6e}")

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
        ft_group_name = "HalfKA_HM2 FT"
        if self.pp3wide is not None:
            ft_group_name += f" + {self.pp3wide_type} branch"
        group_names = (
            ft_group_name,
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
        ft_parameters = list(self.input.parameters())
        if self.pp3wide is not None:
            ft_parameters.extend(self.pp3wide.parameters())
        optimizer_class = (FrequencyAwareAdamW
                           if self.simple_ft_frequency_lr != "off"
                           else torch.optim.AdamW)
        optimizer_kwargs = {}
        if self.simple_qat_hysteresis == "anti_flip":
            optimizer_class = (FrequencyHysteresisAdamW
                if self.simple_ft_frequency_lr != "off" else HysteresisAdamW)
            targets = [("FT", self.input.weight, FT_QUANT_SCALE, FT_QUANT_MIN, FT_QUANT_MAX)]
            for i, stack in enumerate(self.layer_stacks):
                for name in ("fc0", "fc1", "fc2"):
                    targets.append((f"{name}/B{i:02d}", getattr(stack, name).weight,
                                    DENSE_WEIGHT_SCALE, DENSE_WEIGHT_MIN, DENSE_WEIGHT_MAX))
            optimizer_kwargs.update(targets=targets, row_provider=lambda: self._hysteresis_rows,
                cooldown=self.simple_qat_hysteresis_cooldown, band=self.simple_qat_hysteresis_band,
                restore_margin=self.simple_qat_hysteresis_restore_margin)
        if issubclass(optimizer_class, FrequencyAwareAdamW):
            optimizer_kwargs.update(ft_weight=self.input.weight,
                                    row_scale=self.simple_ft_frequency_scale)
        optimizer = optimizer_class(
            [
                {"params": ft_parameters, "lr": self.lr},
                {"params": downstream, "lr": self.lr},
            ],
            betas=(0.9, 0.995), eps=1e-7, weight_decay=1e-6,
            **optimizer_kwargs)
        if isinstance(optimizer, FrequencyAwareAdamW):
            self._frequency_optimizer = optimizer
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        metadata = self.architecture_metadata(
            getattr(self, "transplant_source", None),
            getattr(self, "transplant_mapping_version", None))
        checkpoint["architecture"] = metadata
        checkpoint["nnue_architecture"] = metadata
        if self.simple_qat_hysteresis != "off":
            # Training-only settings, separate from exported architecture/hash.
            checkpoint["quant_hysteresis_settings"] = self.hysteresis_settings()

    def hysteresis_settings(self):
        return {"mode": self.simple_qat_hysteresis,
                "cooldown": self.simple_qat_hysteresis_cooldown,
                "band": self.simple_qat_hysteresis_band,
                "restore_margin": self.simple_qat_hysteresis_restore_margin,
                "version": HYSTERESIS_VERSION, "coverage": "touched_only"}

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        saved_hysteresis = checkpoint.get("quant_hysteresis_settings", {"mode": "off"})
        expected_hysteresis = self.hysteresis_settings()
        mismatch = (saved_hysteresis.get("mode") != self.simple_qat_hysteresis or
                    (self.simple_qat_hysteresis != "off" and saved_hysteresis != expected_hysteresis))
        if mismatch and self.enforce_hysteresis_resume_match:
            raise ValueError("Simple QAT hysteresis training-state resume mismatch")
        metadata = checkpoint.get("architecture") or checkpoint.get("nnue_architecture")
        if not metadata or metadata.get("architecture_type") != ARCHITECTURE_TYPE:
            raise ValueError("checkpoint is not a HalfKA_hm2 simple checkpoint")
        if int(metadata.get("simple_schema_version", -1)) != SIMPLE_SCHEMA_VERSION:
            raise ValueError("HalfKA_hm2 simple schema mismatch")
        # Loading restores tensors, not a scratch initialization. Keep the
        # source provenance even when a caller supplied another init option.
        self.simple_ft_init = metadata.get("simple_ft_init", "legacy")
        self.input.initialization = self.simple_ft_init
        self.input.initialization_width = (1.0 / FT_QUANT_SCALE
            if self.simple_ft_init == "qat_safe" else math.sqrt(1.0 / FT_INPUTS))
        self.hparams["simple_ft_init"] = self.simple_ft_init
        saved_boundary = metadata.get("simple_quant_boundary_reg", "off")
        boundary_settings_mismatch = any(
            metadata.get(key) != expected for key, expected in (
                ("simple_quant_boundary_band", self.simple_quant_boundary_band),
                ("simple_quant_boundary_ft_weight", self.simple_quant_boundary_ft_weight),
                ("simple_quant_boundary_dense_weight", self.simple_quant_boundary_dense_weight),
                ("simple_quant_boundary_layer_weights", self.simple_quant_boundary_layer_weights),
                ("simple_quant_boundary_formula", QUANT_BOUNDARY_VERSION)))
        boundary_settings_mismatch |= (
            metadata.get("simple_quant_boundary_ft_interval", 1)
            != self.simple_quant_boundary_ft_interval)
        if (saved_boundary != self.simple_quant_boundary_reg or
                (saved_boundary != "off" and boundary_settings_mismatch)):
            if getattr(self, "enforce_quant_boundary_resume_match", False):
                raise ValueError("Simple quantization boundary resume mismatch")
            print("Weight-only Simple quantization boundary setting change: "
                  f"checkpoint={saved_boundary}, requested="
                  f"{self.simple_quant_boundary_reg}; optimizer will start fresh")
        saved_side_input = bool(metadata.get("use_side_input", False))
        saved_side_type = metadata.get("simple_side_input_type", "none")
        saved_shared_psqt = bool(metadata.get("use_shared_psqt", False))
        saved_factorization = metadata.get(
            "simple_ft_virtual_factorization", "off")
        saved_mapping_version = metadata.get(
            "simple_ft_virtual_mapping_version")
        saved_local_pair = metadata.get("simple_local_pair_feature", "off")
        if saved_local_pair not in SIMPLE_LOCAL_PAIR_FEATURES:
            raise ValueError(
                f"unknown checkpoint local pair feature: {saved_local_pair!r}")
        current_local_pair = self.simple_local_pair_feature
        if saved_local_pair != "off" and current_local_pair == "off":
            raise ValueError(
                "PP3Wide checkpoint cannot load into PP-OFF architecture")
        if saved_local_pair == current_local_pair and saved_local_pair != "off":
            expected_mapping = (
                LOCALPAIR_MAPPING[saved_local_pair]
                if saved_local_pair in LOCALPAIR_TYPES
                else PP3WIDE_MAPPING_VERSION)
            if metadata.get("simple_pp3wide_mapping_version") \
                    != expected_mapping:
                raise ValueError("Simple PP3Wide mapping version mismatch")
            expected_schema = (
                LOCALPAIR_SCHEMA[saved_local_pair]
                if saved_local_pair in LOCALPAIR_TYPES
                else PP3WIDE64_SCHEMA_VERSION
                if saved_local_pair == PP3WIDE64_TYPE
                else PP3WIDE_SCHEMA_VERSION)
            if int(metadata.get("simple_pp3wide_schema_version", -1)) \
                    != expected_schema:
                raise ValueError("Simple PP3Wide schema version mismatch")
        if saved_local_pair != current_local_pair and self.pp3wide is not None:
            state = checkpoint.setdefault("state_dict", {})
            state["pp3wide.weight"] = self.pp3wide.weight.detach().clone()
            if isinstance(self.pp3wide, SimplePp3Wide64):
                state["pp3wide.projection.weight"] = (
                    self.pp3wide.projection.weight.detach().clone())
            else:
                state.pop("pp3wide.projection.weight", None)
            print(
                "Weight-only Simple PP migration: "
                f"{saved_local_pair} -> "
                f"{self.simple_local_pair_feature}; init="
                f"{self.simple_pp3wide_init}")
        if (saved_factorization == "shared"
                and saved_mapping_version != FT_VIRTUAL_MAPPING_VERSION):
            raise ValueError(
                "Simple FT virtual mapping mismatch: checkpoint="
                f"{saved_mapping_version!r}, expected="
                f"{FT_VIRTUAL_MAPPING_VERSION!r}")
        current_factorization = self.simple_ft_virtual_factorization
        if saved_factorization not in FT_VIRTUAL_FACTORIZATION_MODES:
            raise ValueError(
                "unknown checkpoint Simple FT factorization: "
                f"{saved_factorization!r}")
        if saved_factorization != current_factorization:
            state = checkpoint.get("state_dict", {})
            source_weight = state.get("input.weight")
            if source_weight is None:
                raise ValueError("checkpoint is missing input.weight")
            if saved_factorization == "shared":
                source_virtual = state.get("input.virtual_weight")
                if source_virtual is None:
                    raise ValueError(
                        "factorized checkpoint is missing input.virtual_weight")
                effective = source_weight + source_virtual.repeat(
                    HALFKA_HM2_KING_BUCKETS, 1)
            else:
                effective = source_weight
            if current_factorization == "shared":
                # Do the same function-preserving mean decomposition as the
                # .pt migration path, but write tensors into Lightning's
                # pending state_dict before load_state_dict runs.
                virtual = effective.view(
                    HALFKA_HM2_KING_BUCKETS,
                    HALFKA_HM2_PLANES, -1).mean(dim=0)
                specific = effective - virtual.repeat(
                    HALFKA_HM2_KING_BUCKETS, 1)
                for _ in range(2):
                    reconstructed = specific + virtual.repeat(
                        HALFKA_HM2_KING_BUCKETS, 1)
                    specific.add_(effective - reconstructed)
                expanded_virtual = virtual.repeat(
                    HALFKA_HM2_KING_BUCKETS, 1)
                for _ in range(4):
                    reconstructed = specific + expanded_virtual
                    mismatch = reconstructed != effective
                    if not mismatch.any():
                        break
                    direction = torch.where(
                        reconstructed < effective,
                        torch.full_like(specific, float("inf")),
                        torch.full_like(specific, float("-inf")))
                    specific = torch.where(
                        mismatch, torch.nextafter(specific, direction),
                        specific)
                desired_q = torch.round(effective * FT_QUANT_SCALE)
                for _ in range(4):
                    reconstructed = specific + expanded_virtual
                    actual_q = torch.round(
                        reconstructed * FT_QUANT_SCALE)
                    mismatch = actual_q != desired_q
                    if not mismatch.any():
                        break
                    direction = torch.where(
                        actual_q < desired_q,
                        torch.full_like(specific, float("inf")),
                        torch.full_like(specific, float("-inf")))
                    specific = torch.where(
                        mismatch, torch.nextafter(specific, direction),
                        specific)
                state["input.weight"] = specific
                state["input.virtual_weight"] = virtual
            else:
                state["input.weight"] = effective
                state.pop("input.virtual_weight", None)
            print(
                "Simple FT checkpoint parameterization conversion: "
                f"{saved_factorization} -> {current_factorization}; "
                "weight-only resume requires a fresh optimizer")
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
