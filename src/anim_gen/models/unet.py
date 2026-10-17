"""U-Net denoiser for diffusion (DDPM, Ho et al. 2020; improvements from
Nichol & Dhariwal 2021 and Dhariwal & Nichol 2021).

Input: a noisy image x_t, the timestep t and (optionally) a class label.
Output: a tensor shaped like the image: the predicted noise (or velocity).

    x_t -> conv3 -> [ResBlock x n (+attn) -> down] per level -> middle
        -> [ResBlock x (n+1) with skip concat (+attn) -> up] per level -> conv3

Every ResBlock receives the time (+ class) embedding through a FiLM-style
scale and shift, so the same network can denoise at every noise level.
The label index `num_classes` is the "no label" token used for
classifier-free guidance.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from anim_gen.config import ModelConfig


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Sinusoidal embedding of (possibly fractional) timesteps, [B] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


def group_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with up to 32 groups (works for small test widths too)."""
    groups = math.gcd(32, channels)
    return nn.GroupNorm(groups, channels)


def zero_module(module: nn.Module) -> nn.Module:
    """Zero-init a layer so the residual branch starts as the identity."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * out_ch)  # -> (scale, shift)
        self.norm2 = group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = zero_module(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention over all pixels, with a residual connection."""

    def __init__(self, channels: int, head_dim: int = 64) -> None:
        super().__init__()
        self.heads = max(1, channels // head_dim)
        self.norm = group_norm(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.out = zero_module(nn.Conv2d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        # [B, heads, HW, head_dim] for scaled_dot_product_attention
        out = F.scaled_dot_product_attention(
            q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)
        )
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.out(out)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        ch = config.unet_channels
        emb_dim = 4 * ch
        self.time_mlp = nn.Sequential(nn.Linear(ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        # One extra row: index num_classes = "no label" (classifier-free guidance).
        self.class_embed = (
            nn.Embedding(config.num_classes + 1, emb_dim) if config.num_classes else None
        )
        self.conv_in = nn.Conv2d(config.channels, ch, 3, padding=1)

        # --- encoder: remember the width of every skip connection ---
        self.down = nn.ModuleList()
        skip_widths = [ch]
        width, res = ch, config.image_size
        for level, mult in enumerate(config.channel_mult):
            for _ in range(config.num_res_blocks):
                out = ch * mult
                layers = [ResBlock(width, out, emb_dim, config.dropout)]
                if res in config.attn_resolutions:
                    layers.append(AttentionBlock(out))
                self.down.append(nn.ModuleList(layers))
                width = out
                skip_widths.append(width)
            if level != len(config.channel_mult) - 1:
                self.down.append(nn.ModuleList([Downsample(width)]))
                skip_widths.append(width)
                res //= 2

        self.mid = nn.ModuleList(
            [
                ResBlock(width, width, emb_dim, config.dropout),
                AttentionBlock(width),
                ResBlock(width, width, emb_dim, config.dropout),
            ]
        )

        # --- decoder: one more block per level, each eats one skip ---
        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(config.channel_mult))):
            for i in range(config.num_res_blocks + 1):
                out = ch * mult
                layers = [ResBlock(width + skip_widths.pop(), out, emb_dim, config.dropout)]
                if res in config.attn_resolutions:
                    layers.append(AttentionBlock(out))
                if level != 0 and i == config.num_res_blocks:
                    layers.append(Upsample(out))
                    res *= 2
                self.up.append(nn.ModuleList(layers))
                width = out
        assert not skip_widths

        self.out = nn.Sequential(
            group_norm(width), nn.SiLU(), zero_module(nn.Conv2d(width, config.channels, 3, padding=1))
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """x [B, C, H, W], t [B] timesteps, y [B] labels (num_classes = none) -> [B, C, H, W]."""
        emb = self.time_mlp(timestep_embedding(t, self.config.unet_channels))
        if self.class_embed is not None:
            if y is None:  # unconditional request on a conditional model
                y = torch.full((x.shape[0],), self.config.num_classes, device=x.device)
            emb = emb + self.class_embed(y)

        h = self.conv_in(x)
        skips = [h]
        for layers in self.down:
            for layer in layers:
                h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
            skips.append(h)
        for layer in self.mid:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        for layers in self.up:
            h = torch.cat([h, skips.pop()], dim=1)
            for layer in layers:
                h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        return self.out(h)


__all__ = ["UNet", "timestep_embedding"]
