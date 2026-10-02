"""Training-only post-AdamW shallow-reversal filter (Experiment 146).

No forward/serializer quantizer changes. FT coverage is explicitly touched-only;
zero-gradient AdamW momentum updates on other rows are not intercepted.
"""
import torch
from simple_frequency_aware_optimizer import FrequencyAwareAdamW

VERSION = "post_adamw_shallow_reverse_v1"
MODES = ("off", "anti_flip")
DEFAULT_COOLDOWN = 32
DEFAULT_BAND = 2.6212064783688287e-5
DEFAULT_MARGIN = 1e-5


def quantized_bin(weight, scale, low, high):
    """Same round-to-even / scale / clamp as Simple _fake_quantize."""
    return torch.round(weight * scale).clamp(low, high).to(torch.int16)


@torch.no_grad()
def filter_update(weight, before, history, scale, low, high, cooldown, band, margin):
    """One observed update consumes one touch; suppression never renews cooldown.

    history: uint8, direction in upper 2 bits, cooldown in lower 6 bits.
    Returns weight/history and GPU integer counters, avoiding host synchronization.
    """
    after = quantized_bin(weight, scale, low, high)
    delta = after.to(torch.int32) - before.to(torch.int32)
    crossed = delta != 0
    direction = torch.where(delta > 0, 1, 2).to(torch.uint8)
    old_direction = history >> 6
    remaining = (history & 63).to(torch.int16)
    reverse = crossed & (old_direction != 0) & (old_direction != direction)
    boundary = before.float() + torch.where(delta > 0, .5, -.5)
    penetration = torch.where(delta > 0, 1., -1.) * (weight * scale - boundary)
    eligible = reverse & (remaining > 0)
    suppressed = eligible & (delta.abs() == 1) & (penetration >= 0) & (penetration < band)
    # Restore to old side, not bin center. nextafter protects tie-to-even and
    # cases where margin is below the master's ULP at large magnitudes.
    restore = (boundary + torch.where(delta > 0, -margin, margin)) / scale
    toward_old = torch.where(delta > 0, -torch.inf, torch.inf)
    restore = torch.nextafter(restore, toward_old)
    fixed = torch.where(suppressed, restore, weight)
    accepted = crossed & ~suppressed
    new_remaining = (remaining - 1).clamp_min(0).to(torch.uint8)
    new_remaining = torch.where(accepted, cooldown, new_remaining)
    new_direction = torch.where(accepted, direction, old_direction)
    packed = (new_direction << 6) | new_remaining
    stats = torch.stack([crossed.sum(), (crossed & (old_direction == 0)).sum(),
        (crossed & (old_direction == direction)).sum(), reverse.sum(), eligible.sum(),
        suppressed.sum(), (eligible & ~suppressed).sum(), (reverse & (remaining == 0)).sum(),
        accepted.sum(), (remaining > 0).sum(), weight.new_tensor(weight.numel(), dtype=torch.int64)])
    return fixed, packed, stats


COUNTERS = ("proposed_crossings", "first_crossings", "same_direction_crossings", "reverse_crossings",
            "reverse_within_cooldown", "suppressed_shallow", "allowed_strong_reverse", "expired_reverse",
            "accepted_crossings", "active_cooldown_observations", "element_observations")


class HysteresisAdamW(torch.optim.AdamW):
    """AdamW moments remain untouched; only float master parameter is corrected."""
    def __init__(self, params, *, targets, row_provider, cooldown=DEFAULT_COOLDOWN,
                 band=DEFAULT_BAND, restore_margin=DEFAULT_MARGIN, chunk_rows=1024, **kwargs):
        super().__init__(params, **kwargs)
        if not 1 <= int(cooldown) <= 63 or int(cooldown) != cooldown:
            raise ValueError("hysteresis cooldown must be integer 1..63")
        if not 0 < band < .5 or not 0 < restore_margin < .5:
            raise ValueError("hysteresis band/margin must be in (0,.5)")
        self.targets, self.row_provider = list(targets), row_provider
        self.settings = dict(cooldown=int(cooldown), band=float(band), restore_margin=float(restore_margin),
                             version=VERSION, coverage="touched_only", gap_policy="invalidate_history")
        self.chunk_rows = int(chunk_rows)
        self.history = {name: torch.zeros_like(p, dtype=torch.uint8) for name, p, *_ in self.targets}
        self.counters = {name: torch.zeros(len(COUNTERS), dtype=torch.int64, device=p.device)
                         for name, p, *_ in self.targets}
        self.observation_step = 0
        self.row_last_seen = {name: torch.full((p.shape[0],), -1, dtype=torch.int64, device=p.device)
                              for name, p, *_ in self.targets if name == "FT"}

    @torch.no_grad()
    def capture(self):
        rows = self.row_provider()
        if rows is None:
            raise RuntimeError("anti_flip requires current batch touched FT rows")
        snapshots = []
        for name, p, scale, lo, hi in self.targets:
            if name == "FT":
                # No float full-table clone. q_before only selected rows, int16.
                before = torch.empty((len(rows), p.shape[1]), dtype=torch.int16, device=p.device)
                for start in range(0, len(rows), self.chunk_rows):
                    ix = rows[start:start + self.chunk_rows]
                    before[start:start + len(ix)] = quantized_bin(p.index_select(0, ix), scale, lo, hi)
                snapshots.append((rows, before))
            else:
                snapshots.append((None, quantized_bin(p, scale, lo, hi)))
        return snapshots

    @torch.no_grad()
    def apply(self, snapshots):
        for (name, p, scale, lo, hi), (rows, before) in zip(self.targets, snapshots):
            if rows is None:
                fixed, history, stats = filter_update(p, before, self.history[name], scale, lo, hi,
                    self.settings["cooldown"], self.settings["band"], self.settings["restore_margin"])
                p.copy_(fixed)
                self.history[name].copy_(history)
                self.counters[name].add_(stats)
            else:
                # Rounding scan is unavoidable for intercepted rows, but only
                # sparse crossings need float penetration/restore calculations.
                after = torch.empty_like(before)
                for start in range(0, len(rows), self.chunk_rows):
                    ix = rows[start:start + self.chunk_rows]
                    after[start:start + len(ix)] = quantized_bin(p.index_select(0, ix), scale, lo, hi)
                changed = torch.nonzero(after != before, as_tuple=False)
                local_r, col = changed[:, 0], changed[:, 1]
                row = rows[local_r]
                packed_cross = self.history[name][row, col]
                valid_cross = self.row_last_seen[name][row] == self.observation_step - 1
                packed_cross = torch.where(valid_cross, packed_cross, 0)
                fixed, cross_history, stats = filter_update(p[row, col], before[local_r, col],
                    packed_cross, scale, lo, hi, self.settings['cooldown'], self.settings['band'],
                    self.settings['restore_margin'])
                # Advance cooldown for every intercepted touch, not just events.
                for start in range(0, len(rows), self.chunk_rows):
                    ix = rows[start:start + self.chunk_rows]
                    prior = self.row_last_seen[name].index_select(0, ix)
                    packed = self.history[name].index_select(0, ix)
                    # Momentum can move an unobserved row. Do not mistake an
                    # intervening same-direction/first crossing for a reverse.
                    packed = torch.where((prior == self.observation_step - 1)[:, None], packed, 0)
                    self.history[name].index_copy_(0, ix, packed - ((packed & 63) > 0).to(torch.uint8))
                p[row, col] = fixed
                self.history[name][row, col] = cross_history
                # Active counter is for crossing candidates, not all weights.
                stats[-1] = before.numel()
                self.counters[name].add_(stats)
                self.row_last_seen[name].index_fill_(0, rows, self.observation_step)
        self.observation_step += 1

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        snapshots = self.capture()
        result = super().step()
        self.apply(snapshots)
        return loss if closure is not None else result

    def state_dict(self):
        result = super().state_dict()
        result["quant_hysteresis"] = {"settings": self.settings, "history": self.history,
                                      "counters": self.counters, "row_last_seen": self.row_last_seen,
                                      "observation_step": self.observation_step}
        return result

    def load_state_dict(self, state):
        saved = state.get("quant_hysteresis")
        if saved is None or saved["settings"] != self.settings:
            raise ValueError("quantization hysteresis optimizer resume mismatch")
        if saved["history"].keys() != self.history.keys():
            raise ValueError("quantization hysteresis history keys mismatch")
        for name, target in self.history.items():
            value = saved["history"][name]
            if value.dtype != torch.uint8 or value.shape != target.shape:
                raise ValueError("quantization hysteresis history shape/dtype mismatch")
        super().load_state_dict({k: v for k, v in state.items() if k != "quant_hysteresis"})
        for name, target in self.history.items():
            target.copy_(saved["history"][name].to(target.device))
            self.counters[name].copy_(saved["counters"][name].to(target.device))
        for name, target in self.row_last_seen.items():
            target.copy_(saved['row_last_seen'][name].to(target.device))
        self.observation_step = int(saved['observation_step'])


class FrequencyHysteresisAdamW(HysteresisAdamW, FrequencyAwareAdamW):
    """Capture bins -> row-scaled AdamW -> anti-flip -> accepted history.

    Cooperative MRO deliberately wraps the existing FrequencyAwareAdamW.step
    inside HysteresisAdamW.step. Neither standalone optimizer changes.
    Moments are never corrected. FT coverage remains touched-only.
    """
    def state_dict(self):
        result = super().state_dict()
        result['frequency_hysteresis'] = {
            'version': 'frequency_then_anti_flip_v1',
            'row_scale': self.row_scale.detach().cpu().clone(),
        }
        return result

    def load_state_dict(self, state):
        saved = state.get('frequency_hysteresis')
        if (saved is None or saved.get('version') != 'frequency_then_anti_flip_v1'
                or not torch.equal(saved['row_scale'].cpu(), self.row_scale.detach().cpu())):
            raise ValueError('frequency + anti_flip resume requires identical row scales/update order')
        super().load_state_dict({k: v for k, v in state.items() if k != 'frequency_hysteresis'})
