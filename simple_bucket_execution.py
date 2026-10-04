"""Process-local Simple head execution; no learned/serialized state."""
from dataclasses import dataclass
import numpy as np
import torch

MODES = ("mask", "index_reuse", "grouped_reuse")
DEFAULT_MODE = "grouped_reuse"
METADATA_ATTRIBUTE = "_simple_bucket_metadata"


def resolve_mode(mode, architecture):
    """Use the production grouped dispatch only for Simple, never Complex."""
    return validate_mode(mode if mode is not None else
                         DEFAULT_MODE if architecture == "halfka_hm2_simple" else "mask")


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError(f"simple_bucket_execution must be one of {MODES}, got {mode!r}")
    return mode


@dataclass(frozen=True)
class BucketMetadata:
    counts: tuple
    permutation: torch.Tensor

    def to(self, device):
        return BucketMetadata(self.counts, self.permutation.to(device, non_blocking=True))


def make_metadata(cpu_ids, device):
    """Only call on native/CPU ids before H2D, never on a CUDA tensor."""
    ids = np.asarray(cpu_ids).reshape(-1)
    if ids.size and (ids.min() < 0 or ids.max() >= 9):
        raise ValueError("Simple bucket must be in [0,8]")
    counts = tuple(int(n) for n in np.bincount(ids, minlength=9))
    perm = torch.from_numpy(np.concatenate([np.flatnonzero(ids == b) for b in range(9)]))
    device = torch.device(device)
    if device.type == "cuda":
        perm = perm.pin_memory()
    return BucketMetadata(counts, perm.to(device, non_blocking=True))


def execute_heads(net, x, side, residual, metadata, collect, out, parts,
                  psqt_deep, psqt_shortcut):
    """E161 B1: same within-bucket order, head/QAT order, no CUDA counts."""
    if net.simple_bucket_execution == "grouped_reuse":
        return execute_grouped_heads(net, x, side, residual, metadata, collect,
                                     out, parts, psqt_deep, psqt_shortcut)
    p = 0
    for bucket, stack in enumerate(net.layer_stacks):
        count = metadata.counts[bucket]
        if not count:
            continue
        ids = metadata.permutation[p:p + count]
        result = stack(
            x.index_select(0, ids),
            None if side is None else side.index_select(0, ids),
            None if residual is None else residual.index_select(0, ids),
            collect_diagnostics=collect, qat_mode=net.simple_qat_mode)
        if collect:
            result, diagnostics = result
            for name, value in diagnostics.items():
                parts.setdefault(name, []).append(value)
            if net.use_shared_psqt:
                psqt_deep.index_copy_(0, ids, diagnostics["deep"].view(-1))
                psqt_shortcut.index_copy_(0, ids, diagnostics["shortcut"].view(-1))
        out.index_copy_(0, ids, result)
        p += count
    return out


class _HeadGradientBuffer(torch.autograd.Function):
    """Match independent index_copy backward buffers for exact bias reduction."""
    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, gradient):
        return gradient.clone(memory_format=torch.contiguous_format)


def execute_grouped_heads(net, x, side, residual, metadata, collect, out, parts,
                          psqt_deep, psqt_shortcut):
    """Stable one-gather dispatch; metadata and all learned head operations unchanged."""
    counts, permutation = metadata.counts, metadata.permutation
    if not x.shape[0]:
        return out
    inputs = x.index_select(0, permutation).split(counts)
    sides = (side.index_select(0, permutation).split(counts)
             if side is not None else (None,) * len(counts))
    residuals = (residual.index_select(0, permutation).split(counts)
                 if residual is not None else (None,) * len(counts))
    outputs = []
    offset = 0
    for bucket, stack in enumerate(net.layer_stacks):
        count = counts[bucket]
        if not count:
            continue
        result = stack(inputs[bucket], sides[bucket], residuals[bucket],
                       collect_diagnostics=collect, qat_mode=net.simple_qat_mode)
        if collect:
            result, diagnostics = result
            for name, value in diagnostics.items():
                parts.setdefault(name, []).append(value)
            if net.use_shared_psqt:
                ids = permutation[offset:offset + count]
                psqt_deep.index_copy_(0, ids, diagnostics["deep"].view(-1))
                psqt_shortcut.index_copy_(0, ids, diagnostics["shortcut"].view(-1))
        if torch.is_grad_enabled():
            result = _HeadGradientBuffer.apply(result)
        outputs.append(result)
        offset += count
    out.index_copy_(0, permutation, torch.cat(outputs))
    return out
