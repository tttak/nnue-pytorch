"""Optimizer presets and a Lightning-compatible composite optimizer.

The command line selects only preset names.  Hyperparameters deliberately live
here so optimizer experiments remain reproducible and do not grow a large set
of loosely coupled CLI flags.
"""

from collections import defaultdict
from copy import deepcopy

import bitsandbytes as bnb
import torch

from adabelief import AdaBelief

try:
    from pytorch_optimizer import Lamb, NovoGrad, Ranger, SM3, SOAP, StableAdamW
except ImportError:  # Optional experiment dependency; default path stays usable.
    Lamb = NovoGrad = Ranger = SM3 = SOAP = StableAdamW = None


OPTIMIZER_PRESETS = {
    # This is the pre-preset production configuration.  Do not change it
    # without treating that as a training-behaviour change.
    "adamw8bit": {
        "class": bnb.optim.AdamW8bit,
        "lr_scale": 1.0,
        "kwargs": {
            "betas": (0.9, 0.995),
            "eps": 1e-7,
            "weight_decay": 1e-6,
            "min_8bit_size": 1_000_000,
        },
    },
    "adamw": {
        "class": torch.optim.AdamW,
        "lr_scale": 1.0,
        "kwargs": {
            "betas": (0.9, 0.995),
            "eps": 1e-7,
            "weight_decay": 1e-6,
        },
    },
    "lion8bit": {
        "class": bnb.optim.Lion8bit,
        # Lion's sign-based update is materially larger than AdamW's at the
        # same nominal LR for the sparse FT rows.  Fixed-batch sanity at 1e-5
        # produced 4-6x larger cumulative FT updates, so use the conservative
        # first-candidate ratio recommended for Lion-style comparisons.
        "lr_scale": 0.2,
        "kwargs": {
            "betas": (0.9, 0.99),
            "weight_decay": 1e-6,
            "min_8bit_size": 1_000_000,
        },
    },
    "lion_other": {
        # Experiment 47: the Other tensors are all below min_8bit_size, so
        # bitsandbytes keeps their Lion state in FP32.  Keep a separate preset
        # from the sparse-FT experiment because its LR is tuned independently.
        "class": bnb.optim.Lion8bit,
        # 0.2 (the sparse-FT preset) recovered only 27-46% of AdamW's
        # 110-step subgroup update.  0.7 matches the dense Other trajectory
        # much more closely without producing a loss spike.
        "lr_scale": 0.7,
        "kwargs": {
            "betas": (0.9, 0.99),
            "weight_decay": 1e-6,
            "min_8bit_size": 1_000_000,
        },
    },
    "adafactor": {
        "class": torch.optim.Adafactor,
        "lr_scale": 1.0,
        "kwargs": {
            "beta2_decay": -0.8,
            "eps": (None, 1e-3),
            "d": 1.0,
            "weight_decay": 0.0,
        },
    },
    "radam": {
        # Reliable PyTorch FP32 fallback for the Other-group experiment.
        # decoupled_weight_decay=True gives AdamW-style decay semantics.
        "class": torch.optim.RAdam,
        # Fixed-batch sanity at the baseline LR produced about one third of
        # AdamW's cumulative Other update during RAdam's rectification phase.
        # Use 3x so the first-candidate update scale is comparable.
        "lr_scale": 3.0,
        "kwargs": {
            "betas": (0.9, 0.995),
            "eps": 1e-7,
            "weight_decay": 1e-6,
            "decoupled_weight_decay": True,
        },
    },
    "nadamw": {
        # Experiment 47 NAdamW extension.  PyTorch's NAdam implements
        # AdamW-style decoupled decay when this flag is enabled.
        "class": torch.optim.NAdam,
        "lr_scale": 1.0,
        "kwargs": {
            "betas": (0.9, 0.995),
            "eps": 1e-7,
            "weight_decay": 1e-6,
            "momentum_decay": 4e-3,
            "decoupled_weight_decay": True,
        },
    },
    "adabeliefw": {
        # Experiment 47 extension: local minimal AdaBelief implementation
        # with AdamW-style decoupled weight decay.
        "class": AdaBelief,
        # Fixed 110-batch sanity: 1.0x yielded roughly 2.4-3.3x AdamW's
        # cumulative subgroup update.  0.3x is the closest conservative
        # aggregate match across Main/L2/FM/QKV/Phase/aux groups.
        "lr_scale": 0.3,
        "kwargs": {
            "betas": (0.9, 0.995),
            "eps": 1e-7,
            "weight_decay": 1e-6,
            "decoupled_weight_decay": True,
        },
    },
    "lamb": {
        # Low-friction pytorch-optimizer implementation.  This is a
        # conservative first preset, not a claim that the LR is optimal for
        # sparse FT rows; run a short update-scale sanity before long jobs.
        "class": Lamb,
        "lr_scale": 1.0,
        "kwargs": {
            "betas": (0.9, 0.995),
            "eps": 1e-6,
            "weight_decay": 1e-6,
            "weight_decouple": True,
            "fixed_decay": False,
            "grad_averaging": True,
            "max_grad_norm": 1.0,
            "adam": False,
        },
    },
    "novograd": {
        "class": NovoGrad,
        "lr_scale": 1.0,
        "kwargs": {
            "betas": (0.95, 0.98),
            "eps": 1e-8,
            "weight_decay": 1e-6,
            "weight_decouple": True,
            "fixed_decay": False,
            "grad_averaging": False,
        },
    },
    "ranger": {
        # RAdam + Lookahead from pytorch-optimizer.  The preset is intended
        # first for the relatively small dense Other group; applying its FP32
        # state to the 267M-parameter FT requires a separate memory study.
        "class": Ranger,
        "lr_scale": 1.0,
        "kwargs": {
            "betas": (0.95, 0.999),
            "eps": 1e-5,
            "weight_decay": 1e-6,
            "weight_decouple": True,
            "fixed_decay": False,
            "alpha": 0.5,
            "k": 6,
            "use_gc": True,
            "gc_conv_only": False,
        },
    },
    "sm3": {
        # Kept deliberately conservative because SM3's commonly used LR is
        # much larger and strongly workload-dependent.
        "class": SM3,
        "lr_scale": 1.0,
        "kwargs": {
            "momentum": 0.0,
            "beta": 0.0,
            "eps": 1e-30,
        },
    },
    "stableadamw": {
        "class": StableAdamW,
        "lr_scale": 1.0,
        "kwargs": {
            "betas": (0.9, 0.995),
            "eps": 1e-8,
            "weight_decay": 1e-6,
            "weight_decouple": True,
            # Current NNUE parameters are FP32; Kahan summation is mainly
            # useful for low-precision parameter storage.
            "kahan_sum": False,
        },
    },
    "soap": {
        "class": SOAP,
        "lr_scale": 2.0,
        "kwargs": {
            "betas": (0.95, 0.95),
            "shampoo_beta": 0.95,
            "weight_decay": 1e-6,
            "precondition_frequency": 10,
            "max_precondition_dim": 512,
            "merge_dims": False,
            "precondition_1d": False,
            "correct_bias": True,
            "normalize_gradient": False,
            "eps": 1e-8,
        },
    },
    "soap2d_adamw": {
        "class": "soap_hybrid",
        "fallback": "adamw",
        "lr_scale": 2.0,
        "fallback_lr_scale": 1.0,
        "kwargs": {
            "betas": (0.95, 0.95), "shampoo_beta": 0.95,
            "weight_decay": 1e-6, "precondition_frequency": 10,
            "max_precondition_dim": 512, "merge_dims": False,
            "precondition_1d": False, "correct_bias": True,
            "normalize_gradient": False, "eps": 1e-8,
        },
    },
    "soap2d_radam": {
        "class": "soap_hybrid",
        "fallback": "radam",
        "lr_scale": 2.0,
        "fallback_lr_scale": 3.0,
        "kwargs": {
            "betas": (0.95, 0.95), "shampoo_beta": 0.95,
            "weight_decay": 1e-6, "precondition_frequency": 10,
            "max_precondition_dim": 512, "merge_dims": False,
            "precondition_1d": False, "correct_bias": True,
            "normalize_gradient": False, "eps": 1e-8,
        },
    },
    "hybrid_h1_deep_radam": {
        "class": "radam_adabelief_hybrid",
        "assignment": "h1_deep_radam",
        "lr_scale": 1.0,
        "kwargs": {},
    },
    "hybrid_h2_matrix_radam": {
        "class": "radam_adabelief_hybrid",
        "assignment": "h2_matrix_radam",
        "lr_scale": 1.0,
        "kwargs": {},
    },
    "hybrid_h3_control_adabelief": {
        "class": "radam_adabelief_hybrid",
        "assignment": "h3_control_adabelief",
        "lr_scale": 1.0,
        "kwargs": {},
    },
}


def radam_adabelief_assignment(candidate, name, parameter):
    """Return the child optimizer for an Experiment 48 Other parameter."""
    deep_prefixes = (
        "layer_stacks.l2", "layer_stacks.output",
        "layer_stacks.cross_proj", "layer_stacks.q_proj",
        "layer_stacks.k_proj", "layer_stacks.v_proj",
    )
    if candidate == "h1_deep_radam":
        return "radam" if name.startswith(deep_prefixes) and parameter.ndim == 2 else "adabeliefw"
    if candidate == "h2_matrix_radam":
        return "radam" if parameter.ndim == 2 else "adabeliefw"
    if candidate == "h3_control_adabelief":
        control = (
            name.endswith(".bias") or name == "layer_stacks.blend"
            or name == "layer_stacks.lca_temp"
            or name.startswith("layer_stacks.phase_proj")
            or name.startswith("main_aux_head")
            or name.startswith("fm_aux_head")
        )
        return "adabeliefw" if control else "radam"
    raise ValueError(f"Unknown RAdam/AdaBelief hybrid assignment: {candidate}")


def available_optimizer_presets():
    return tuple(OPTIMIZER_PRESETS)


def optimizer_preset_summary(name, base_lr):
    preset = OPTIMIZER_PRESETS[name]
    class_value = preset["class"]
    class_name = (class_value if isinstance(class_value, str)
                  else class_value.__name__ if class_value is not None
                  else "unavailable")
    result = {
        "name": name,
        "class": class_name,
        "lr": float(base_lr) * preset["lr_scale"],
        **deepcopy(preset["kwargs"]),
    }
    if class_value == "soap_hybrid":
        result["fallback"] = preset["fallback"]
        result["fallback_lr"] = float(base_lr) * preset["fallback_lr_scale"]
    elif class_value == "radam_adabelief_hybrid":
        result["assignment"] = preset["assignment"]
        result["radam_lr"] = float(base_lr) * OPTIMIZER_PRESETS["radam"]["lr_scale"]
        result["adabeliefw_lr"] = float(base_lr) * OPTIMIZER_PRESETS["adabeliefw"]["lr_scale"]
    return result


def build_optimizer(name, param_groups, base_lr, parameter_names=None):
    preset = OPTIMIZER_PRESETS[name]
    lr_scale = float(preset["lr_scale"])
    # Model groups carry explicit relative LRs (for example input.v=1.5x).
    # Optimizer constructor defaults do not override those values, so apply
    # the preset scale to every explicit group LR as well.
    scaled_param_groups = []
    for group in param_groups:
        scaled_group = dict(group)
        if "lr" in scaled_group:
            scaled_group["lr"] = float(scaled_group["lr"]) * lr_scale
        scaled_param_groups.append(scaled_group)
    if preset["class"] == "soap_hybrid":
        if SOAP is None:
            raise ImportError("SOAP presets require pytorch-optimizer")
        matrix_groups, fallback_groups = [], []
        for original in param_groups:
            matrix = [p for p in original["params"] if p.ndim == 2]
            fallback = [p for p in original["params"] if p.ndim != 2]
            if matrix:
                group = dict(original); group["params"] = matrix
                if "lr" in group:
                    group["lr"] = float(group["lr"]) * lr_scale
                matrix_groups.append(group)
            if fallback:
                group = dict(original); group["params"] = fallback
                if "lr" in group:
                    group["lr"] = float(group["lr"]) * preset["fallback_lr_scale"]
                fallback_groups.append(group)
        soap = SOAP(matrix_groups, lr=float(base_lr) * lr_scale,
                    **deepcopy(preset["kwargs"]))
        fallback_name = preset["fallback"]
        fallback_preset = OPTIMIZER_PRESETS[fallback_name]
        fallback = fallback_preset["class"](
            fallback_groups,
            lr=float(base_lr) * preset["fallback_lr_scale"],
            **deepcopy(fallback_preset["kwargs"]),
        )
        return CompositeOptimizer([
            ("soap_2d", soap), (f"fallback_{fallback_name}", fallback)])
    if preset["class"] == "radam_adabelief_hybrid":
        if parameter_names is None:
            raise ValueError(f"Optimizer preset {name} requires parameter_names")
        child_groups = {"radam": [], "adabeliefw": []}
        for original in param_groups:
            split = {"radam": [], "adabeliefw": []}
            for parameter in original["params"]:
                parameter_name = parameter_names.get(id(parameter))
                if parameter_name is None:
                    raise KeyError("Hybrid optimizer received an unnamed parameter")
                child = radam_adabelief_assignment(
                    preset["assignment"], parameter_name, parameter)
                split[child].append(parameter)
            for child, parameters in split.items():
                if not parameters:
                    continue
                group = dict(original)
                group["params"] = parameters
                child_preset = OPTIMIZER_PRESETS[child]
                if "lr" in group:
                    group["lr"] = float(group["lr"]) * child_preset["lr_scale"]
                child_groups[child].append(group)
        children = []
        for child in ("radam", "adabeliefw"):
            child_preset = OPTIMIZER_PRESETS[child]
            optimizer = child_preset["class"](
                child_groups[child],
                lr=float(base_lr) * child_preset["lr_scale"],
                **deepcopy(child_preset["kwargs"]),
            )
            children.append((child, optimizer))
        return CompositeOptimizer(children)
    if preset["class"] is None:
        raise ImportError(f"Optimizer preset {name} requires pytorch-optimizer")
    return preset["class"](
        scaled_param_groups,
        lr=float(base_lr) * lr_scale,
        **deepcopy(preset["kwargs"]),
    )


class CompositeOptimizer(torch.optim.Optimizer):
    """Expose heterogeneous child optimizers as one Lightning optimizer.

    Lightning 2.6 automatic optimization supports one optimizer.  This wrapper
    keeps that path (including one closure/backward and one scheduler) while
    delegating the parameter updates to disjoint FT and Other optimizers.
    Child count is arbitrary.  Its checkpoint is explicitly child-structured
    and validates optimizer names before loading, preventing partial or silent
    state restoration.
    """

    def __init__(self, named_optimizers):
        if not named_optimizers:
            raise ValueError("CompositeOptimizer needs at least one child")
        self.optimizer_names = tuple(name for name, _ in named_optimizers)
        self.optimizers = tuple(optimizer for _, optimizer in named_optimizers)
        all_params = [
            parameter
            for optimizer in self.optimizers
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        if len({id(parameter) for parameter in all_params}) != len(all_params):
            raise ValueError("Composite optimizer children contain duplicate parameters")

        # Initialize Optimizer hooks/attributes, then expose the actual child
        # dictionaries so schedulers update the same LR values used by children.
        super().__init__(all_params, defaults={})
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]
        self._refresh_state_view()

    def _refresh_state_view(self):
        state = defaultdict(dict)
        for optimizer in self.optimizers:
            state.update(optimizer.state)
        self.state = state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        for index, optimizer in enumerate(self.optimizers):
            child_loss = optimizer.step(closure if index == 0 else None)
            if index == 0:
                loss = child_loss
        self._refresh_state_view()
        return loss

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "format_version": 1,
            "optimizer_names": list(self.optimizer_names),
            "children": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state_dict):
        names = tuple(state_dict.get("optimizer_names", ()))
        if names != self.optimizer_names:
            raise ValueError(
                "Composite optimizer checkpoint mismatch: "
                f"saved={names}, requested={self.optimizer_names}"
            )
        children = state_dict.get("children", ())
        if len(children) != len(self.optimizers):
            raise ValueError("Composite optimizer checkpoint child count mismatch")
        for optimizer, child_state in zip(self.optimizers, children):
            optimizer.load_state_dict(child_state)
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]
        self._refresh_state_view()
