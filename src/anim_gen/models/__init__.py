"""Network architectures and a small registry to build them from a ModelConfig.

arch     family      networks
dcgan    GAN         DCGANGenerator / DCGANDiscriminator
resnet   GAN         ResNetGenerator / ResNetDiscriminator (SNGAN / BigGAN style)
unet     diffusion   UNet denoiser (DDPM / improved DDPM)
"""

from __future__ import annotations

from torch import nn

from anim_gen.config import ModelConfig
from anim_gen.models.dcgan import DCGANDiscriminator, DCGANGenerator
from anim_gen.models.resnet_gan import ResNetDiscriminator, ResNetGenerator
from anim_gen.models.unet import UNet

GENERATORS = {"dcgan": DCGANGenerator, "resnet": ResNetGenerator}
DISCRIMINATORS = {"dcgan": DCGANDiscriminator, "resnet": ResNetDiscriminator}


def build_generator(config: ModelConfig) -> nn.Module:
    return GENERATORS[config.arch](config)


def build_discriminator(config: ModelConfig) -> nn.Module:
    return DISCRIMINATORS[config.arch](config)


def build_unet(config: ModelConfig) -> nn.Module:
    return UNet(config)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


__all__ = [
    "DCGANDiscriminator",
    "DCGANGenerator",
    "ResNetDiscriminator",
    "ResNetGenerator",
    "UNet",
    "build_discriminator",
    "build_generator",
    "build_unet",
    "count_parameters",
]
