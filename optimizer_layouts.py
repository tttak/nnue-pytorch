"""Declarative parameter-subgroup and optimizer-layout definitions.

Layouts contain only ``subgroup -> optimizer preset`` assignments.  Optimizer
classes and hyperparameters remain in :mod:`optimizer_presets`.
"""

from __future__ import annotations

from collections import OrderedDict


OPTIMIZER_LAYOUT_SCHEMA_VERSION = 3

# ``*_control`` leaves are deliberately separate.  This preserves the exact
# Experiment 48 assignments (matrix/representation tensors versus biases,
# gates and blends) while retaining the requested logical parent names.
PARAMETER_SUBGROUPS = (
    "ft", "router",
    "pair_weights", "main_gates", "main_gates_control",
    "l2_fc1_output", "l2_fc1_output_control",
    "fm", "fm_control",
    "lca_cross", "lca_cross_control",
    "qkv", "qkv_control",
    "phase", "phase_control", "aux", "aux_control",
)

LOGICAL_SUBGROUP_PARENTS = {
    "pair_weights": "main_gates",
    "main_gates_control": "main_gates",
    "l2_fc1_output_control": "l2_fc1_output",
    "fm_control": "fm",
    "lca_cross_control": "lca_cross",
    "qkv_control": "qkv",
    "phase_control": "phase",
    "aux_control": "aux",
}


def expand_subgroup_selection(groups):
    """Expand a logical group name to its control/pair-weight leaves."""
    requested = tuple(dict.fromkeys(groups))
    unknown = set(requested) - set(PARAMETER_SUBGROUPS)
    if unknown:
        raise ValueError(f"Unknown optimizer subgroups: {sorted(unknown)}")
    selected = []
    for subgroup in PARAMETER_SUBGROUPS:
        parent = LOGICAL_SUBGROUP_PARENTS.get(subgroup)
        if subgroup in requested or parent in requested:
            selected.append(subgroup)
    return tuple(selected)


def parameter_subgroup(name: str) -> str:
    """Return the one and only optimizer subgroup for a named parameter."""
    if name in ("input.weight", "input.bias", "input.v"):
        return "ft"
    if name.startswith("layer_stacks.router."):
        return "router"
    if name.startswith(("main_aux_head.", "fm_aux_head.")):
        return "aux_control" if name.endswith(".bias") else "aux"
    if name == "pair_weights":
        return "pair_weights"
    if name == "layer_stacks.blend":
        return "main_gates_control"
    if name.startswith(("layer_stacks.l1.", "layer_stacks.l1_fact.")):
        return "main_gates_control" if name.endswith(".bias") else "main_gates"
    if name.startswith(("layer_stacks.l2.", "layer_stacks.output.")):
        return ("l2_fc1_output_control" if name.endswith(".bias")
                else "l2_fc1_output")
    if name.startswith(("layer_stacks.side_input_encode.",
                        "layer_stacks.side_input_l2_residual.")):
        return ("l2_fc1_output_control" if name.endswith(".bias")
                else "l2_fc1_output")
    if name.startswith(("layer_stacks.pair_relation_embedding.",
                        "layer_stacks.pair_relation_ln.",
                        "layer_stacks.pair_relation_proj.")):
        return ("l2_fc1_output_control" if name.endswith(".bias")
                else "l2_fc1_output")
    if name == "layer_stacks.pair_relation_gate":
        return "l2_fc1_output_control"
    if name.startswith(("layer_stacks.fm_diff.", "layer_stacks.fm_abs.")):
        return "fm_control" if name.endswith(".bias") else "fm"
    if name.startswith("layer_stacks.cross_proj."):
        return "lca_cross_control" if name.endswith(".bias") else "lca_cross"
    if name == "layer_stacks.lca_temp":
        return "lca_cross_control"
    if name.startswith(("layer_stacks.q_proj.", "layer_stacks.k_proj.",
                        "layer_stacks.v_proj.")):
        return "qkv_control" if name.endswith(".bias") else "qkv"
    if name.startswith("layer_stacks.phase_proj."):
        return "phase_control" if name.endswith(".bias") else "phase"
    raise KeyError(f"Unclassified optimizer parameter: {name}")


def _layout(default: str, **overrides: str) -> dict[str, str]:
    result = OrderedDict((group, default) for group in PARAMETER_SUBGROUPS)
    unknown = set(overrides) - set(PARAMETER_SUBGROUPS)
    if unknown:
        raise KeyError(f"Unknown optimizer subgroups in layout: {sorted(unknown)}")
    result.update(overrides)
    return dict(result)


OPTIMIZER_LAYOUTS = {
    "production_baseline": _layout("adamw8bit"),
    "exp47_adamw": _layout("adamw", ft="frozen", router="frozen"),
    "exp47_radam": _layout("radam", ft="frozen", router="frozen"),
    "exp47_adabelief": _layout(
        "adabeliefw", ft="frozen", router="frozen"),
    # Experiment 48 H1: only selected deep 2-D tensors used RAdam.
    "exp48_h1": _layout(
        "adabeliefw", ft="frozen", router="frozen",
        l2_fc1_output="radam", lca_cross="radam", qkv="radam"),
    # Experiment 48 H2: every 2-D matrix used RAdam. pair_weights is 3-D.
    "exp48_h2": _layout(
        "adabeliefw", ft="frozen", router="frozen",
        main_gates="radam", l2_fc1_output="radam", fm="radam",
        lca_cross="radam", qkv="radam", phase="radam", aux="radam"),
    # Experiment 48 H3: representation tensors use RAdam; controls use Ada.
    "exp48_h3": _layout(
        "radam", ft="frozen", router="frozen",
        main_gates_control="adabeliefw",
        l2_fc1_output_control="adabeliefw",
        fm_control="adabeliefw",
        lca_cross_control="adabeliefw",
        qkv_control="adabeliefw",
        phase="adabeliefw", phase_control="adabeliefw",
        aux="adabeliefw", aux_control="adabeliefw"),
    # Ready-to-run low-complexity candidates: keep the large FT on the current
    # AdamW8bit baseline and change only the dense Other parameters.
    "other_lamb": _layout("lamb", ft="adamw8bit"),
    "other_novograd": _layout("novograd", ft="adamw8bit"),
    "other_ranger": _layout("ranger", ft="adamw8bit"),
    "other_sm3": _layout("sm3", ft="adamw8bit"),
    "other_stableadamw": _layout("stableadamw", ft="adamw8bit"),

    "mixed_optimizer_sample": _layout(
        "adamw",

        ft="adamw8bit",

        router="radam",

        pair_weights="lion_other",

        main_gates="lamb",
        main_gates_control="adabeliefw",

        l2_fc1_output="stableadamw",
        l2_fc1_output_control="adabeliefw",

        fm="novograd",
        fm_control="adamw",

        lca_cross="ranger",
        lca_cross_control="adabeliefw",

        qkv="radam",
        qkv_control="adamw",

        phase="sm3",
        phase_control="adamw",

        aux="lamb",
        aux_control="adamw",
    ),

}

# Reserved extension point.  Keeping LR multipliers separate from layouts
# preserves the simple subgroup->preset schema while allowing a future
# experiment to tune one subgroup without creating extra optimizer instances.
OPTIMIZER_LAYOUT_LR_OVERRIDES = {
    name: {} for name in OPTIMIZER_LAYOUTS
}


def optimizer_layout_lr_scale(layout_name: str, subgroup: str) -> float:
    if layout_name is None:
        return 1.0
    return float(OPTIMIZER_LAYOUT_LR_OVERRIDES.get(layout_name, {}).get(
        subgroup, 1.0))


def available_optimizer_layouts():
    return tuple(OPTIMIZER_LAYOUTS)


def resolve_optimizer_layout(name: str) -> dict[str, str]:
    try:
        layout = dict(OPTIMIZER_LAYOUTS[name])
    except KeyError as exc:
        raise ValueError(f"Unknown optimizer layout: {name}") from exc
    missing = set(PARAMETER_SUBGROUPS) - set(layout)
    extra = set(layout) - set(PARAMETER_SUBGROUPS)
    if missing or extra:
        raise ValueError(
            f"Invalid optimizer layout {name}: missing={sorted(missing)}, "
            f"extra={sorted(extra)}")
    return layout


def legacy_optimizer_layout(ft: str, other: str, freeze_ft_router: bool):
    """Translate the old two-optimizer CLI into the declarative model."""
    result = _layout(other)
    result["ft"] = "frozen" if freeze_ft_router else ft
    result["router"] = "frozen" if freeze_ft_router else other
    return result
