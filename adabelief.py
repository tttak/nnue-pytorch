"""Minimal PyTorch AdaBelief optimizer used by optimizer experiments.

This deliberately implements only the configuration needed by this project:
dense/sparse-free gradients, decoupled weight decay, and no AMSGrad.  Keeping
the implementation small makes its update rule and checkpoint state explicit.
"""
from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


class AdaBelief(Optimizer):
    """AdaBelief with AdamW-style decoupled weight decay.

    The second moment tracks ``(gradient - first_moment) ** 2``.  Epsilon is
    added outside the square root, matching the standard Adam-style numerical
    stabilization used by the rest of this repository.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, decoupled_weight_decay=True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        super().__init__(params, dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            decoupled = group["decoupled_weight_decay"]
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("AdaBelief does not support sparse gradients")
                if not decoupled and weight_decay:
                    grad = grad.add(parameter, alpha=weight_decay)
                elif weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format)
                    state["exp_avg_var"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format)
                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_var = state["exp_avg_var"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(
                    residual, residual, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1
                denom = exp_avg_var.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                parameter.addcdiv_(exp_avg, denom, value=-step_size)
        return loss
