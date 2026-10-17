"""Every architecture produces the right shapes, at several resolutions."""

import pytest
import torch
from conftest import tiny_model_config
from torch.nn.utils import parametrize

from anim_gen.models import build_discriminator, build_generator, build_unet
from anim_gen.models.layers import ConditionalBatchNorm2d, SelfAttention, gan_widths


def test_gan_widths_double_and_cap():
    assert gan_widths(32, 64, 512) == {32: 64, 16: 128, 8: 256, 4: 512}
    assert gan_widths(32, 64, 128) == {32: 64, 16: 128, 8: 128, 4: 128}


@pytest.mark.parametrize("arch", ["dcgan", "resnet"])
@pytest.mark.parametrize("size", [8, 16, 32, 64])
@pytest.mark.parametrize("num_classes", [0, 3])
def test_gan_shapes(arch, size, num_classes):
    config = tiny_model_config(
        arch, num_classes, image_size=size, channels=1 if size == 8 else 3,
        attn_resolution=size // 2,
    )  # fmt: skip
    G, D = build_generator(config), build_discriminator(config)
    y = torch.tensor([0, 2]) if num_classes else None
    images = G(torch.randn(2, config.z_dim), y)
    assert images.shape == (2, config.channels, size, size)
    assert images.abs().max() <= 1  # tanh output
    assert D(images, y).shape == (2,)


@pytest.mark.parametrize("arch", ["dcgan", "resnet"])
def test_spectral_norm_is_applied_to_the_discriminator(arch):
    D = build_discriminator(tiny_model_config(arch, spectral_norm=True))
    convs = [m for m in D.modules() if isinstance(m, torch.nn.Conv2d)]
    assert convs and all(parametrize.is_parametrized(m, "weight") for m in convs)
    D = build_discriminator(tiny_model_config(arch, spectral_norm=False))
    assert not any(parametrize.is_parametrized(m) for m in D.modules())


def test_class_conditioning_changes_the_output():
    config = tiny_model_config("resnet", num_classes=3)
    G = build_generator(config).eval()
    torch.nn.init.normal_(G.blocks[0].bn1.embed.weight)  # make the classes differ
    z = torch.randn(1, config.z_dim)
    assert not torch.allclose(G(z, torch.tensor([0])), G(z, torch.tensor([1])))


def test_conditional_batchnorm_starts_as_plain_batchnorm():
    x = torch.randn(4, 6, 5, 5)
    cbn = ConditionalBatchNorm2d(6, num_classes=3)
    plain = torch.nn.BatchNorm2d(6, affine=False)
    assert torch.allclose(cbn(x, torch.tensor([0, 1, 2, 0])), plain(x), atol=1e-6)


def test_self_attention_starts_as_identity():
    x = torch.randn(2, 16, 4, 4)
    assert torch.equal(SelfAttention(16)(x), x)  # gamma = 0


@pytest.mark.parametrize("num_classes", [0, 3])
@pytest.mark.parametrize("size,mult", [(16, (1, 2)), (32, (1, 2, 2))])
def test_unet_shapes(num_classes, size, mult):
    config = tiny_model_config("unet", num_classes, image_size=size, channel_mult=mult)
    unet = build_unet(config)
    x = torch.randn(2, 3, size, size)
    t = torch.tensor([0, 49])
    y = torch.tensor([1, num_classes]) if num_classes else None  # num_classes = "no label"
    assert unet(x, t, y).shape == x.shape


def test_unet_output_starts_at_zero():
    """The last conv is zero-initialised: an untrained model predicts 0 noise."""
    unet = build_unet(tiny_model_config("unet"))
    out = unet(torch.randn(2, 3, 16, 16), torch.tensor([3, 7]))
    assert torch.equal(out, torch.zeros_like(out))


def test_unet_depends_on_time():
    unet = build_unet(tiny_model_config("unet"))
    for p in unet.parameters():  # undo every zero init (the time enters via residual branches)
        if not p.any():
            torch.nn.init.normal_(p, std=0.1)
    x = torch.randn(1, 3, 16, 16)
    assert not torch.allclose(unet(x, torch.tensor([1])), unet(x, torch.tensor([40])))
