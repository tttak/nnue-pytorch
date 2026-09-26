"""Shared evaluation contract and metrics for the Simple NNUE.

Production comparisons always use :data:`FIXED_FORWARD_CONTRACT`, regardless
of how a checkpoint was trained.  Float evaluation is an explicitly named
diagnostic and must never be selected from checkpoint QAT metadata.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch


FIXED_FORWARD_CONTRACT = "fixed"
FLOAT_DIAGNOSTIC_CONTRACT = "float_diagnostic"
SUPPORTED_FORWARD_CONTRACTS = (
    FIXED_FORWARD_CONTRACT, FLOAT_DIAGNOSTIC_CONTRACT)
PAIR_WINDOWS = (1, 2, 3, 4, 5, 8, 12, 16, 24, 30,
                50, 100, 200, 500, 1000, 2000)


def clear_simple_qat_cache(model) -> None:
    for module in model.modules():
        if hasattr(module, "_qat_eval_cache"):
            module._qat_eval_cache = None


def validate_comparison_contract(contract: str, model_names: Iterable[str]) -> str:
    """Validate the one contract shared by every model in a comparison.

    Deliberately accepting one scalar contract (rather than a per-model map)
    makes an accidental ``baseline=float, candidate=fixed`` comparison
    impossible through the public comparison API.
    """
    names = tuple(model_names)
    if not names:
        raise ValueError("at least one model is required")
    if not isinstance(contract, str):
        raise TypeError("evaluation contract must be one shared string")
    if contract not in SUPPORTED_FORWARD_CONTRACTS:
        raise ValueError(
            f"unsupported Simple evaluation contract {contract!r}; "
            f"expected one of {SUPPORTED_FORWARD_CONTRACTS}")
    return contract


@contextmanager
def simple_forward_contract(model, contract: str):
    """Temporarily apply a deployment or diagnostic forward contract."""
    validate_comparison_contract(contract, ("model",))
    previous = model.simple_qat_mode
    # ``full`` is the Python integer/fake-fixed reference for the deployed
    # Simple C++ network.  It is correct for both QAT-off and QAT-full models.
    model.simple_qat_mode = (
        "full" if contract == FIXED_FORWARD_CONTRACT else "off")
    clear_simple_qat_cache(model)
    try:
        yield
    finally:
        model.simple_qat_mode = previous
        clear_simple_qat_cache(model)


@torch.no_grad()
def evaluate_simple_cp(model, batch, contract: str = FIXED_FORWARD_CONTRACT):
    """Evaluate one SparseBatch tuple in cp under an explicit contract."""
    with simple_forward_contract(model, contract):
        return (model(*batch[:6], batch[8], ply=batch[11], material=batch[9])
                .view(-1) * model.nnue2score)


def probability(cp, offset1: float, offset2: float, scale: float):
    if torch.is_tensor(cp):
        return 0.5 * (1.0 + torch.sigmoid((cp - offset1) / scale)
                      - torch.sigmoid((-cp - offset2) / scale))
    cp = np.asarray(cp, dtype=np.float64)
    return 0.5 * (1.0 + 1.0 / (1.0 + np.exp(-(cp - offset1) / scale))
                  - 1.0 / (1.0 + np.exp(-(-cp - offset2) / scale)))


def _stable_multikey_order(raw_probability, material, ply):
    order = np.argsort(raw_probability, kind="stable")
    material_bin = np.rint(material / 200.0)
    order = order[np.argsort(material_bin[order], kind="stable")]
    ply_bin = np.floor_divide(ply.astype(np.int64), 10)
    return order[np.argsort(ply_bin[order], kind="stable")]


@dataclass
class PairMetricAccumulator:
    raw_correct: int = 0
    raw_count: int = 0
    consensus_correct: int = 0
    consensus_count: int = 0
    agreement_count: int = 0
    agreement_raw_correct: int = 0
    agreement_consensus_correct: int = 0
    disagreement_count: int = 0
    disagreement_raw_correct: int = 0
    disagreement_consensus_correct: int = 0

    def update(self, *, source, ply, material, raw_probability,
               prediction_probability, consensus_probability=None,
               consensus_available=None):
        """Accumulate production-style pair metrics for one batch.

        The current mixed stream has ordered/ranking semantics only for source
        3, so both raw and consensus pairs are formed there.  Raw direction
        uses the original teacher score.  Consensus direction is scored only
        when both samples have a real sidecar target; it is never synthesized
        from the raw score.
        """
        source = np.asarray(source).reshape(-1)
        selected = np.flatnonzero(source == 3)
        if selected.size <= 1:
            return
        ply = np.asarray(ply).reshape(-1)
        material = np.asarray(material).reshape(-1)
        raw = np.asarray(raw_probability).reshape(-1)
        pred = np.asarray(prediction_probability).reshape(-1)
        consensus = (None if consensus_probability is None
                     else np.asarray(consensus_probability).reshape(-1))
        available = (np.zeros(source.size, dtype=bool)
                     if consensus_available is None
                     else np.asarray(consensus_available, dtype=bool).reshape(-1))

        order = _stable_multikey_order(
            raw[selected], material[selected], ply[selected])
        indices = selected[order]
        n = indices.size
        for window in PAIR_WINDOWS:
            if window >= n:
                continue
            left = indices[:-window]
            right = indices[window:]
            raw_gap = np.abs(raw[left] - raw[right])
            valid = ((raw_gap > 0.003) & (raw_gap <= 0.10)
                     & (np.abs(material[left] - material[right]) <= 50)
                     & (np.abs(ply[left] - ply[right]) <= 10))
            left, right = left[valid], right[valid]
            if not left.size:
                continue
            raw_dir = np.sign(raw[left] - raw[right])
            pred_positive = (pred[left] - pred[right]) > 0
            # Match the production pair metric exactly: direction is a
            # greater-than comparison, so a quantized prediction tie belongs
            # to the non-positive side rather than becoming a third class.
            raw_valid = raw_dir != 0
            raw_ok = pred_positive == (raw_dir > 0)
            self.raw_correct += int(raw_ok[raw_valid].sum())
            self.raw_count += int(raw_valid.sum())

            if consensus is None:
                continue
            consensus_dir = np.sign(consensus[left] - consensus[right])
            # A real sidecar pair remains part of the consensus metric even
            # when its quantized target is tied, matching Experiment 115's
            # established regression oracle and pairwise greater-than rule.
            consensus_valid = available[left] & available[right]
            consensus_ok = pred_positive == (consensus_dir > 0)
            self.consensus_correct += int(consensus_ok[consensus_valid].sum())
            self.consensus_count += int(consensus_valid.sum())

            # Agreement/disagreement is meaningful only when both targets
            # have a direction. Consensus ties stay in the overall metric
            # above but are excluded from this diagnostic split.
            comparable = consensus_valid & raw_valid & (consensus_dir != 0)
            agreement = comparable & (raw_dir == consensus_dir)
            disagreement = comparable & (raw_dir != consensus_dir)
            self.agreement_count += int(agreement.sum())
            self.agreement_raw_correct += int(raw_ok[agreement].sum())
            self.agreement_consensus_correct += int(consensus_ok[agreement].sum())
            self.disagreement_count += int(disagreement.sum())
            self.disagreement_raw_correct += int(raw_ok[disagreement].sum())
            self.disagreement_consensus_correct += int(
                consensus_ok[disagreement].sum())

    @staticmethod
    def _ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else None

    def as_dict(self):
        return {
            "raw_pair_accuracy": self._ratio(
                self.raw_correct, self.raw_count),
            "raw_pair_count": self.raw_count,
            "consensus_pair_accuracy": self._ratio(
                self.consensus_correct, self.consensus_count),
            "consensus_pair_count": self.consensus_count,
            "agreement_pair_count": self.agreement_count,
            "agreement_raw_accuracy": self._ratio(
                self.agreement_raw_correct, self.agreement_count),
            "agreement_consensus_accuracy": self._ratio(
                self.agreement_consensus_correct, self.agreement_count),
            "disagreement_pair_count": self.disagreement_count,
            "disagreement_raw_accuracy": self._ratio(
                self.disagreement_raw_correct, self.disagreement_count),
            "disagreement_consensus_accuracy": self._ratio(
                self.disagreement_consensus_correct, self.disagreement_count),
        }
