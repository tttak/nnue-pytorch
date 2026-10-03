"""Complex training-only FC0 ablation; exported independent FC0 is unchanged."""
import torch

TRAINING_KEY = "fc0_shared_factor_training"


def is_disabled(stacks):
    return bool(getattr(stacks, "disable_fc0_shared_factor", False))


def assert_zero_shared(stacks):
    if torch.count_nonzero(stacks.l1_fact.weight).item() or torch.count_nonzero(stacks.l1_fact.bias).item():
        raise ValueError("Disabled FC0 shared factor must have zero weight/bias")


def set_disabled(stacks, disabled):
    """Fold BEFORE zero/freeze. Idempotent for an already disabled checkpoint."""
    disabled = bool(disabled)
    was_disabled = is_disabled(stacks)
    if disabled:
        if was_disabled:
            assert_zero_shared(stacks)
        else:
            with torch.no_grad():
                stacks.l1.weight.view(stacks.count, 32, -1).add_(stacks.l1_fact.weight)
                stacks.l1.bias.view(stacks.count, 32).add_(stacks.l1_fact.bias)
                stacks.l1_fact.weight.zero_()
                stacks.l1_fact.bias.zero_()
        for p in stacks.l1_fact.parameters():
            p.grad = None
            p.requires_grad_(False)
    elif was_disabled:
        for p in stacks.l1_fact.parameters():
            p.requires_grad_(True)
    stacks.disable_fc0_shared_factor = disabled


def load_checkpoint_policy(checkpoint, disabled, enforce_optimizer_match, count):
    """Called before tensor load. Read saved metadata, not CLI-merged hparams."""
    saved = bool(checkpoint.get(TRAINING_KEY, {}).get("disabled", False))
    if enforce_optimizer_match and saved != bool(disabled):
        raise ValueError("FC0 shared factor mismatch for training-state resume; use --resume-from-model")
    state = checkpoint["state_dict"]
    sw = state["layer_stacks.l1_fact.weight"]
    sb = state["layer_stacks.l1_fact.bias"]
    if saved and (torch.count_nonzero(sw).item() or torch.count_nonzero(sb).item()):
        raise ValueError("Disabled FC0 checkpoint contains nonzero shared weight/bias")
    if disabled and not saved:
        # No source tensor mutation: safe even with mmap or shared source state.
        state["layer_stacks.l1.weight"] = (
            state["layer_stacks.l1.weight"].view(count, 32, -1) + sw).reshape(count*32, -1)
        state["layer_stacks.l1.bias"] = (
            state["layer_stacks.l1.bias"].view(count, 32) + sb).reshape(count*32)
        state["layer_stacks.l1_fact.weight"] = torch.zeros_like(sw)
        state["layer_stacks.l1_fact.bias"] = torch.zeros_like(sb)
