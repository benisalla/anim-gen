"""ResNet GAN in the style of SNGAN / SAGAN / BigGAN.

The strongest GAN recipe that still fits in one file:
- residual blocks in both networks (gradients flow through the skip path),
- spectral norm on every discriminator layer,
- class-conditional BatchNorm in the generator,
- projection discriminator for class conditioning,
- optional self-attention at one resolution (attn_resolution),
- trained with the hinge loss (losses.py), Adam(0, 0.9), and an EMA generator.

Generator   z -> Linear -> 4x4 -> GBlock(up) x n -> BN, ReLU, conv3 -> tanh
Discriminator   x -> DBlock(down) x n -> ReLU -> global sum pool -> Linear (+ projection)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from imagegen.config import ModelConfig
from imagegen.models.layers import (
    ConditionalBatchNorm2d,
    SelfAttention,
    gan_widths,
    raw_weight,
    sn,
)


def _orthogonal_init(module: nn.Module) -> None:
    """Orthogonal init (BigGAN): keeps signal scale stable through deep stacks."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(raw_weight(module))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class GBlock(nn.Module):
    """BN -> ReLU -> upsample -> conv3 -> BN -> ReLU -> conv3, plus an upsampled skip."""

    def __init__(self, in_ch: int, out_ch: int, num_classes: int) -> None:
        super().__init__()
        self.bn1 = ConditionalBatchNorm2d(in_ch, num_classes)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn2 = ConditionalBatchNorm2d(out_ch, num_classes)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        h = F.relu(self.bn1(x, y))
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = self.conv2(F.relu(self.bn2(h, y)))
        return h + self.skip(F.interpolate(x, scale_factor=2, mode="nearest"))


class DBlock(nn.Module):
    """(ReLU) -> conv3 -> ReLU -> conv3 -> avgpool, plus a pooled 1x1 skip.

    The first block skips the leading ReLU: its input is the raw image.
    """

    def __init__(self, in_ch: int, out_ch: int, use_sn: bool, first: bool = False) -> None:
        super().__init__()
        self.first = first
        self.conv1 = sn(nn.Conv2d(in_ch, out_ch, 3, padding=1), use_sn)
        self.conv2 = sn(nn.Conv2d(out_ch, out_ch, 3, padding=1), use_sn)
        self.skip = sn(nn.Conv2d(in_ch, out_ch, 1), use_sn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x if self.first else F.relu(x)
        h = self.conv2(F.relu(self.conv1(h)))
        h = F.avg_pool2d(h, 2)
        if self.first:  # pool first: the 1x1 conv then runs on 4x fewer pixels
            return h + self.skip(F.avg_pool2d(x, 2))
        return h + F.avg_pool2d(self.skip(x), 2)


class ResNetGenerator(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        widths = gan_widths(config.image_size, config.g_channels, config.max_channels)
        self.proj = nn.Linear(config.z_dim, widths[4] * 4 * 4)
        self.blocks = nn.ModuleList()
        attn = {}
        r = 4
        while r < config.image_size:
            self.blocks.append(GBlock(widths[r], widths[2 * r], config.num_classes))
            r *= 2
            if r == config.attn_resolution:
                attn[str(len(self.blocks) - 1)] = SelfAttention(widths[r])
        self.attn = nn.ModuleDict(attn)  # keyed by the index of the block it follows
        self.out_bn = nn.BatchNorm2d(widths[config.image_size])
        self.out_conv = nn.Conv2d(widths[config.image_size], config.channels, 3, padding=1)
        self.apply(_orthogonal_init)

    def forward(self, z: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """z [B, z_dim], y [B] class ids (or None) -> images [B, C, H, W] in [-1, 1]."""
        h = self.proj(z).view(z.shape[0], -1, 4, 4)
        for i, block in enumerate(self.blocks):
            h = block(h, y)
            if str(i) in self.attn:
                h = self.attn[str(i)](h)
        return torch.tanh(self.out_conv(F.relu(self.out_bn(h))))


class ResNetDiscriminator(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        use_sn = config.spectral_norm
        widths = gan_widths(config.image_size // 2, config.d_channels, config.max_channels)
        blocks: list[nn.Module] = []
        in_ch, r = config.channels, config.image_size
        while r > 4:
            r //= 2
            blocks.append(DBlock(in_ch, widths[r], use_sn, first=not blocks))
            in_ch = widths[r]
            if r == config.attn_resolution:
                blocks.append(SelfAttention(in_ch, use_sn=use_sn))
        self.blocks = nn.Sequential(*blocks)
        self.head = sn(nn.Linear(in_ch, 1), use_sn)
        self.embed = (
            sn(nn.Embedding(config.num_classes, in_ch), use_sn) if config.num_classes else None
        )
        self.apply(_orthogonal_init)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """images [B, C, H, W], y [B] (or None) -> realness logits [B]."""
        h = F.relu(self.blocks(x)).sum(dim=(2, 3))  # global sum pooling -> [B, C]
        out = self.head(h).squeeze(1)
        if self.embed is not None:  # projection discriminator
            out = out + (self.embed(y) * h).sum(dim=1)
        return out


__all__ = ["ResNetDiscriminator", "ResNetGenerator"]
