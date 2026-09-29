"""Training-only row-scaled AdamW for Simple HalfKA_HM2 FT weights.

The FT update is computed by the same AdamW equations as torch.optim.AdamW,
then the *parameter delta* (not the gradient) is multiplied by a per-row
scale.  Adam moments are deliberately computed from the unscaled gradient.
All non-FT parameters remain on the ordinary torch AdamW implementation.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


_CUDA_SOURCE = r'''
extern "C" __global__ void row_scaled_adamw(
    float* p, const float* g, float* m, float* v,
    const float* row_scale, const unsigned char* touched,
    const long long elements, const int width,
    const float lr, const float beta1, const float beta2,
    const float eps, const float weight_decay,
    const float bias_correction1, const float sqrt_bias_correction2) {
  const long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= elements) return;
  const float grad = g[i];
  const float mi = beta1 * m[i] + (1.0f - beta1) * grad;
  const float vi = beta2 * v[i] + (1.0f - beta2) * grad * grad;
  m[i] = mi;
  v[i] = vi;
  const float old_value = p[i];
  const float decayed = old_value * (1.0f - lr * weight_decay);
  const float denom = sqrtf(vi) / sqrt_bias_correction2 + eps;
  const float base_new = decayed - (lr / bias_correction1) * mi / denom;
  const int row = (int)(i / width);
  const float scale = touched[row] ? row_scale[row] : 1.0f;
  p[i] = old_value + scale * (base_new - old_value);
}
'''


class FrequencyAwareAdamW(torch.optim.AdamW):
    """AdamW with exact row scaling for one 2-D float32 FT parameter."""

    def __init__(self, params, ft_weight: torch.nn.Parameter,
                 row_scale: torch.Tensor, **kwargs):
        super().__init__(params, **kwargs)
        if ft_weight.ndim != 2 or ft_weight.dtype != torch.float32:
            raise ValueError("frequency-aware FT weight must be 2-D float32")
        if tuple(row_scale.shape) != (ft_weight.shape[0],):
            raise ValueError("frequency scale table shape mismatch")
        self.ft_weight = ft_weight
        self.row_scale = row_scale.detach()
        self._touched_rows: Optional[torch.Tensor] = None
        self._cuda_kernel = None

    def set_touched_rows(self, rows: torch.Tensor) -> None:
        self._touched_rows = rows.detach()

    def _ft_group(self):
        for group in self.param_groups:
            if any(parameter is self.ft_weight for parameter in group["params"]):
                return group
        raise RuntimeError("FT weight is absent from optimizer groups")

    @torch.no_grad()
    def _step_ft_cpu(self, group, grad, state, touched):
        beta1, beta2 = group["betas"]
        state["step"] += 1
        step = int(state["step"].item())
        m, v = state["exp_avg"], state["exp_avg_sq"]
        m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        old = self.ft_weight.detach().clone()  # micro-test/CPU fallback only
        decayed = old * (1.0 - group["lr"] * group["weight_decay"])
        denom = v.sqrt() / math.sqrt(1.0 - beta2 ** step) + group["eps"]
        base_new = decayed - (group["lr"] / (1.0 - beta1 ** step)) * m / denom
        scale = torch.where(touched, self.row_scale.to(old),
                            torch.ones_like(self.row_scale, device=old.device))
        self.ft_weight.copy_(old + scale[:, None] * (base_new - old))

    @torch.no_grad()
    def _step_ft_cuda(self, group, grad, state, touched):
        import cupy as cp
        if self._cuda_kernel is None:
            self._cuda_kernel = cp.RawKernel(_CUDA_SOURCE, "row_scaled_adamw")
        beta1, beta2 = group["betas"]
        state["step"] += 1
        step = int(state["step"].item())
        scale = self.row_scale.to(device=self.ft_weight.device,
                                  dtype=torch.float32)
        touched_u8 = touched.to(device=self.ft_weight.device, dtype=torch.uint8)
        tensors = [self.ft_weight, grad, state["exp_avg"], state["exp_avg_sq"],
                   scale, touched_u8]
        arrays = [cp.from_dlpack(value.detach()) for value in tensors]
        n = self.ft_weight.numel()
        args = (*arrays, np.int64(n), np.int32(self.ft_weight.shape[1]),
                np.float32(group["lr"]), np.float32(beta1), np.float32(beta2),
                np.float32(group["eps"]), np.float32(group["weight_decay"]),
                np.float32(1.0 - beta1 ** step),
                np.float32(math.sqrt(1.0 - beta2 ** step)))
        stream = cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
        with stream:
            self._cuda_kernel(((n + 255) // 256,), (256,), args)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            # Lightning passes the forward/backward work as the optimizer
            # closure.  Run it first so the current batch can publish its
            # touched FT rows before we choose the custom update path.
            with torch.enable_grad():
                loss = closure()
        grad = self.ft_weight.grad
        if grad is None:
            super().step()
            return loss
        if grad.is_sparse:
            raise RuntimeError("frequency-aware FT expects the current dense gradient")
        touched_rows = self._touched_rows
        if touched_rows is None:
            raise RuntimeError("frequency-aware optimizer has no touched-row list")
        touched = torch.zeros(self.ft_weight.shape[0], dtype=torch.bool,
                              device=self.ft_weight.device)
        touched[touched_rows.long()] = True
        self.ft_weight.grad = None
        super().step()
        self.ft_weight.grad = grad
        state = self.state[self.ft_weight]
        if not state:
            state["step"] = torch.zeros((), dtype=torch.float32,
                                        device=self.ft_weight.device)
            state["exp_avg"] = torch.zeros_like(self.ft_weight)
            state["exp_avg_sq"] = torch.zeros_like(self.ft_weight)
        group = self._ft_group()
        if self.ft_weight.is_cuda:
            self._step_ft_cuda(group, grad, state, touched)
        else:
            self._step_ft_cpu(group, grad, state, touched)
        self._touched_rows = None
        return loss
