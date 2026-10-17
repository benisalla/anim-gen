"""Exponential moving average (EMA) of model weights.

    ema = decay * ema + (1 - decay) * weights      after every optimizer step

The averaged model is smoother than the raw one; for both GANs and diffusion
it gives clearly better samples and FID, so sampling always uses it.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMA(nn.Module):
    """Holds a frozen copy of `model` and moves it towards the live weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999, start: int = 0) -> None:
        super().__init__()
        self.decay = decay
        self.start = start  # before this step the copy simply follows the model
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        decay = self.decay if step >= self.start else 0.0
        for ema_p, p in zip(self.model.parameters(), model.parameters(), strict=True):
            ema_p.lerp_(p.detach(), 1.0 - decay)
        # Buffers (BatchNorm running statistics, spectral-norm vectors) are copied.
        for ema_b, b in zip(self.model.buffers(), model.buffers(), strict=True):
            ema_b.copy_(b)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


__all__ = ["EMA"]
