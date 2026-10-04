"""Train-only full-size Simple head parameterization; inference tensors stay dense."""
import torch
from torch import nn
from torch.nn.utils import parametrize


class AddShared(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        # The parent model owns it once; avoid nine duplicate state/optimizer entries.
        object.__setattr__(self, "shared", parameter)

    def forward(self, specific):
        return specific + self.shared


def original(layer, name):
    if parametrize.is_parametrized(layer, name):
        return getattr(layer.parametrizations, name).original
    return getattr(layer, name)


def set_mode(model, name, enabled):
    attribute = name + "_shared_factor"
    shared = getattr(model, attribute, None)
    if not hasattr(model, attribute):
        setattr(model, attribute, None)
    if enabled == (shared is not None):
        return
    if enabled:
        sample = getattr(model.layer_stacks[0], name)
        # Zero creation consumes no RNG, unlike nn.Linear initialization.
        shared = nn.Module()
        shared.register_parameter("weight", nn.Parameter(torch.zeros_like(sample.weight)))
        shared.register_parameter("bias", nn.Parameter(torch.zeros_like(sample.bias)))
        setattr(model, attribute, shared)
        for stack in model.layer_stacks:
            layer = getattr(stack, name)
            for key in ("weight", "bias"):
                parametrize.register_parametrization(layer, key, AddShared(getattr(shared, key)))
    else:
        for stack in model.layer_stacks:
            layer = getattr(stack, name)
            for key in ("weight", "bias"):
                parametrize.remove_parametrizations(layer, key, leave_parametrized=True)
        with torch.no_grad():
            for p in shared.parameters():
                p.zero_()
                p.requires_grad_(False)
        # No active module => historical OFF state_dict and optimizer layout.
        setattr(model, attribute, None)


def migrate_state(model, state):
    for name in ("fc1", "fc2"):
        shared = getattr(model, name + "_shared_factor", None)
        for key in ("weight", "bias"):
            sk = name + "_shared_factor." + key
            saved_shared = state.get(sk)
            for i in range(len(model.layer_stacks)):
                plain = f"layer_stacks.{i}.{name}.{key}"
                factored = f"layer_stacks.{i}.{name}.parametrizations.{key}.original"
                value = state.pop(factored, state.pop(plain, None))
                if value is None:
                    raise ValueError("missing Simple head tensor: " + plain)
                if shared is None and saved_shared is not None:
                    value = value + saved_shared
                state[factored if shared is not None else plain] = value
            if shared is not None:
                if saved_shared is None:
                    state[sk] = torch.zeros_like(getattr(shared, key))
            else:
                state.pop(sk, None)


@torch.no_grad()
def clip_effective(layer, low, high):
    weight = layer.weight
    specific = original(layer, "weight")
    if parametrize.is_parametrized(layer, "weight"):
        specific.add_(weight.clamp(low, high) - weight)
    else:
        specific.clamp_(low, high)
