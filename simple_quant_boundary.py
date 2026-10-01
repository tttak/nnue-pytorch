"""Training-only margin penalty for Simple NNUE float master weights.

The forward quantizer is deliberately not implemented here.  This module only
adds a small gradient away from rounding boundaries; nn.bin remains unchanged.
"""

import torch
import torch.nn.functional as F


MODES = ("off", "weak", "mild", "medium")
FORMULA_VERSION = "margin_v1"
# Calibrated on Experiment 115 D_full_5M, one production-like mixed batch.
# Each layer's regularizer gradient is 0.3% (mild) or 1.0% (medium) of its
# corresponding base-loss gradient at the source checkpoint.
CALIBRATED_PRESETS = {
    # Experiment 145: 16 production-stream batches, band .025. FT runs at
    # steps 0,8,...; the mean ratio including skipped steps is .075%.
    "weak": {"ft": 2.2410190688274967, "fc0": 0.01808449019252933,
             "fc1": 0.0007204283837605178, "fc2": 5.5517056949341887e-05},
    "mild": {"ft": 0.3793299844860965, "fc0": 0.02836063276960786,
             "fc1": 0.0012036765245303585, "fc2": 8.186217535416252e-05},
    "medium": {"ft": 1.2644332816203219, "fc0": 0.09453544256535953,
               "fc1": 0.004012255081767862, "fc2": 0.00027287391784720837},
}


def ft_regularization_due(global_step, interval):
    """Use restored optimizer global_step, so resume keeps the same cadence."""
    if int(interval) != interval or interval < 1:
        raise ValueError("FT regularization interval must be a positive integer")
    return int(global_step) % int(interval) == 0


def boundary_penalty(weight, scale, low, high, band):
    """Mean squared boundary incursion over representable interior weights.

    ``round`` is detached so the derivative points toward the current bin's
    center on either side of a boundary.  Saturated weights are excluded.
    """
    x = weight * scale
    center = torch.round(x).detach()
    distance = 0.5 - (x - center).abs()
    interior = (x > low + 0.5) & (x < high - 0.5)
    penalty = F.relu(band - distance).square()
    return torch.where(interior, penalty, torch.zeros_like(penalty)).mean()


def touched_ft_rows(white_indices, white_values, black_indices, black_values):
    white = white_indices[(white_indices >= 0) & (white_values != 0)]
    black = black_indices[(black_indices >= 0) & (black_values != 0)]
    return torch.unique(torch.cat((white, black)).long())
