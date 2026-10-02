"""Training-only FM Diff value sharing; runtime affine remains 128 -> 64."""
import torch
from torch import nn
from torch.nn import functional as F


def normalize_fm_diff_shared(mode):
    mode = str(mode)
    if mode not in ("off", "value"):
        raise ValueError(f"complex_fm_diff_shared must be off or value, got {mode}")
    return mode


def make_value_factor(reference):
    # Linear's discarded random initialization must not advance any RNG stream.
    with torch.random.fork_rng(devices=[]):
        factor = nn.Linear(128, 32)
    factor = factor.to(device=reference.device, dtype=reference.dtype)
    nn.init.zeros_(factor.weight)
    nn.init.zeros_(factor.bias)
    return factor


def effective_diff_parameters(layer, factor, count):
    if factor is None:
        return layer.weight, layer.bias
    weight = layer.weight.reshape(count, 64, 128)
    bias = layer.bias.reshape(count, 64)
    # Rows 0:32 are gate; rows 32:64 are value in each bucket.
    weight = torch.cat((weight[:, :32], weight[:, 32:] + factor.weight), dim=1)
    bias = torch.cat((bias[:, :32], bias[:, 32:] + factor.bias), dim=1)
    return weight.reshape(count * 64, 128), bias.reshape(count * 64)


def diff_affine(layer, factor, count, x):
    if factor is None:
        return layer(x)
    weight, bias = effective_diff_parameters(layer, factor, count)
    return F.linear(x, weight, bias)


@torch.no_grad()
def clip_effective_diff(layer, factor, count, limit, bias_scale=8128.0):
    weight = layer.weight.view(count, 64, 128)
    weight[:, :32].clamp_(-limit, limit)
    value = weight[:, 32:]
    value.copy_(torch.maximum(torch.minimum(value, limit - factor.weight),
                              -limit - factor.weight))
    # Existing biases have no small activation-range clipping. Keep the same
    # int32 export range, including the shared value bias in that range.
    low = -(2**31) / bias_scale
    high = (2**31 - 256) / bias_scale  # float32-safe below int32 maximum
    bias = layer.bias.view(count, 64)
    bias[:, :32].clamp_(low, high)
    value_bias = bias[:, 32:]
    value_bias.copy_(torch.maximum(torch.minimum(value_bias, high - factor.bias),
                                   low - factor.bias))
