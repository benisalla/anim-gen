"""DCGAN (Radford et al., 2015), for any power-of-two resolution.

Generator: noise z -> 4x4 feature map -> (transposed conv, BN, ReLU) x n -> tanh image.
Discriminator: image -> (strided conv, LeakyReLU) x n -> one realness logit.

The classic recipe, with two modern touches:
- the discriminator returns a raw logit (no sigmoid); losses.py applies the
  numerically stable form (softplus / BCE-with-logits) for each loss;
- optional class conditioning: label embedding concatenated to z (generator)
  and a projection term (discriminator, Miyato & Koyama 2018).
"""

from __future__ import annotations

import torch
from torch import nn

from imagegen.config import ModelConfig
from imagegen.models.layers import gan_widths, raw_weight, sn


def _dcgan_init(module: nn.Module) -> None:
    """N(0, 0.02) weights, as in the DCGAN paper."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Embedding)):
        nn.init.normal_(raw_weight(module), 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) and module.affine:
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


class DCGANGenerator(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        widths = gan_widths(config.image_size // 2, config.g_channels, config.max_channels)
        self.embed = nn.Embedding(config.num_classes, config.z_dim) if config.num_classes else None
        in_dim = config.z_dim * (2 if self.embed is not None else 1)

        layers: list[nn.Module] = [
            # [B, in_dim, 1, 1] -> [B, C4, 4, 4]
            nn.ConvTranspose2d(in_dim, widths[4], 4, 1, 0, bias=False),
            nn.BatchNorm2d(widths[4]),
            nn.ReLU(inplace=True),
        ]
        r = 4
        while r < config.image_size // 2:  # each block doubles the resolution
            layers += [
                nn.ConvTranspose2d(widths[r], widths[2 * r], 4, 2, 1, bias=False),
                nn.BatchNorm2d(widths[2 * r]),
                nn.ReLU(inplace=True),
            ]
            r *= 2
        layers += [nn.ConvTranspose2d(widths[r], config.channels, 4, 2, 1), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.apply(_dcgan_init)

    def forward(self, z: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """z [B, z_dim], y [B] class ids (or None) -> images [B, C, H, W] in [-1, 1]."""
        if self.embed is not None:
            z = torch.cat([z, self.embed(y)], dim=1)
        return self.net(z[:, :, None, None])


class DCGANDiscriminator(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        use_sn = config.spectral_norm
        widths = gan_widths(config.image_size // 2, config.d_channels, config.max_channels)
        r = config.image_size // 2
        layers: list[nn.Module] = [
            sn(nn.Conv2d(config.channels, widths[r], 4, 2, 1), use_sn),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        while r > 4:  # each block halves the resolution
            layers.append(sn(nn.Conv2d(widths[r], widths[r // 2], 4, 2, 1, bias=use_sn), use_sn))
            if not use_sn:  # DCGAN uses BN; with spectral norm we use no normalisation
                layers.append(nn.BatchNorm2d(widths[r // 2]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            r //= 2
        self.features = nn.Sequential(*layers)
        feature_dim = widths[4] * 4 * 4
        self.head = sn(nn.Linear(feature_dim, 1), use_sn)
        self.embed = (
            sn(nn.Embedding(config.num_classes, feature_dim), use_sn)
            if config.num_classes
            else None
        )
        self.apply(_dcgan_init)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """images [B, C, H, W], y [B] (or None) -> realness logits [B]."""
        h = self.features(x).flatten(1)
        out = self.head(h).squeeze(1)
        if self.embed is not None:  # projection: high when h matches the class embedding
            out = out + (self.embed(y) * h).sum(dim=1)
        return out


__all__ = ["DCGANDiscriminator", "DCGANGenerator"]
