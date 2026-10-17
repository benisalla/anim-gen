"""Gaussian diffusion: how images are noised (training) and denoised (sampling).

Forward process, fixed:  x_t = sqrt(a_t) * x_0 + sqrt(1 - a_t) * noise,
where a_t ("alpha bar") falls from ~1 (t = 0, clean) to ~0 (t = T-1, pure noise).

Training: pick a random t, noise the image, ask the U-Net to predict
- "eps": the noise that was added (DDPM), or
- "v":   sqrt(a_t) * noise - sqrt(1 - a_t) * x_0 (Salimans & Ho 2022; better at high res).
The loss is a plain mean squared error.

Sampling: start from pure noise and step backwards with DDIM (Song et al. 2021).
eta = 0 is deterministic and works with few steps (50-250); eta = 1 with all
T steps is the original DDPM sampler. Classifier-free guidance (Ho & Salimans
2022) mixes the conditional and unconditional predictions:
    pred = uncond + guidance_scale * (cond - uncond)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def make_alpha_bar(timesteps: int, schedule: str = "cosine") -> torch.Tensor:
    """a_t for t = 0 .. T-1 (float64)."""
    if schedule == "linear":
        scale = 1000 / timesteps  # keeps the DDPM endpoints for any T
        betas = torch.linspace(scale * 1e-4, scale * 0.02, timesteps, dtype=torch.float64)
    elif schedule == "cosine":  # Nichol & Dhariwal: no wasted steps near pure noise
        s = 0.008
        steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
    else:
        raise ValueError(f"unknown noise schedule: {schedule!r}")
    return torch.cumprod(1 - betas, dim=0)


def _expand(values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Pick values[t] per sample and shape it [B, 1, 1, 1] to broadcast over images."""
    return values[t].float()[:, None, None, None]


class GaussianDiffusion(nn.Module):
    """Holds the noise schedule (as buffers, so .to(device) moves it)."""

    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = "cosine",
        prediction: str = "eps",
        null_label: int | None = None,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.prediction = prediction
        self.null_label = null_label  # label id meaning "no class" (None = unconditional)
        self.register_buffer("alpha_bar", make_alpha_bar(timesteps, schedule), persistent=False)

    # --- training ---------------------------------------------------------

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Noise clean images x0 to level t."""
        a = _expand(self.alpha_bar, t)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def target(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """What the network should output for (x0, t, noise)."""
        if self.prediction == "eps":
            return noise
        a = _expand(self.alpha_bar, t)
        return a.sqrt() * noise - (1 - a).sqrt() * x0

    def training_loss(
        self, model: nn.Module, x0: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """MSE between the network output and its target at a random noise level."""
        t = torch.randint(0, self.timesteps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        prediction = model(self.q_sample(x0, t, noise), t, y)
        return F.mse_loss(prediction.float(), self.target(x0, t, noise))

    # --- sampling ---------------------------------------------------------

    def predict_x0_eps(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None,
        guidance_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Network output -> (clean image estimate clipped to [-1, 1], matching noise)."""
        if y is not None and guidance_scale != 1.0 and self.null_label is not None:
            # One forward on [cond; uncond] instead of two.
            null = torch.full_like(y, self.null_label)
            out = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null])).float()
            cond, uncond = out.chunk(2)
            out = uncond + guidance_scale * (cond - uncond)
        else:
            out = model(x, t, y).float()
        a = _expand(self.alpha_bar, t)
        if self.prediction == "eps":
            x0 = (x - (1 - a).sqrt() * out) / a.sqrt()
        else:  # v
            x0 = a.sqrt() * x - (1 - a).sqrt() * out
        x0 = x0.clamp(-1, 1)
        eps = (x - a.sqrt() * x0) / (1 - a).sqrt()  # consistent with the clipped x0
        return x0, eps

    def sample_timesteps(self, steps: int) -> list[int]:
        """`steps` timesteps from T-1 down to 0, evenly spaced."""
        steps = max(1, min(steps, self.timesteps))
        ts = torch.linspace(self.timesteps - 1, 0, steps).round().long().tolist()
        return list(dict.fromkeys(ts))  # drop duplicates, keep order

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        noise: torch.Tensor,
        y: torch.Tensor | None = None,
        steps: int = 100,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Denoise `noise` [B, C, H, W] into images in [-1, 1] with DDIM."""
        x = noise
        ts = self.sample_timesteps(steps)
        for i, t_now in enumerate(ts):
            t = torch.full((x.shape[0],), t_now, device=x.device, dtype=torch.long)
            x0, eps = self.predict_x0_eps(model, x, t, y, guidance_scale)
            a = self.alpha_bar[t_now].float()
            a_prev = self.alpha_bar[ts[i + 1]].float() if i + 1 < len(ts) else x.new_tensor(1.0)
            sigma = eta * ((1 - a_prev) / (1 - a) * (1 - a / a_prev)).clamp(min=0).sqrt()
            x = a_prev.sqrt() * x0 + (1 - a_prev - sigma**2).clamp(min=0).sqrt() * eps
            if eta > 0 and i + 1 < len(ts):
                z = torch.randn(x.shape, device=x.device, generator=generator)
                x = x + sigma * z
        return x.clamp(-1, 1)


__all__ = ["GaussianDiffusion", "make_alpha_bar"]
