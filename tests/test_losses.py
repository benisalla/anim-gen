"""GAN losses have the textbook values; penalties behave on simple critics."""

import pytest
import torch
import torch.nn.functional as F

from imagegen.diffaugment import diff_augment, parse_policy
from imagegen.ema import EMA
from imagegen.losses import d_loss, g_loss, gradient_penalty, r1_penalty

REAL = torch.tensor([2.0, -1.0, 0.5])
FAKE = torch.tensor([-3.0, 0.0, 1.5])


def test_nonsaturating_is_binary_cross_entropy():
    expected = F.binary_cross_entropy_with_logits(REAL, torch.ones(3)) + (
        F.binary_cross_entropy_with_logits(FAKE, torch.zeros(3))
    )
    assert torch.allclose(d_loss("nonsaturating", REAL, FAKE), expected)
    assert torch.allclose(
        g_loss("nonsaturating", FAKE), F.binary_cross_entropy_with_logits(FAKE, torch.ones(3))
    )


def test_hinge_wgan_lsgan_values():
    assert torch.isclose(
        d_loss("hinge", REAL, FAKE), torch.tensor((0 + 2 + 0.5) / 3 + (0 + 1 + 2.5) / 3)
    )
    assert torch.isclose(g_loss("hinge", FAKE), -FAKE.mean())
    assert torch.isclose(d_loss("wgan-gp", REAL, FAKE), FAKE.mean() - REAL.mean())
    lsgan = 0.5 * ((REAL - 1) ** 2).mean() + 0.5 * (FAKE**2).mean()
    assert torch.isclose(d_loss("lsgan", REAL, FAKE), lsgan)


def test_unknown_loss():
    with pytest.raises(ValueError):
        d_loss("nope", REAL, FAKE)


def test_gradient_penalty_is_zero_for_a_unit_slope_critic():
    w = torch.randn(3 * 4 * 4)
    w = w / w.norm()  # ||grad|| = ||w|| = 1 everywhere

    def critic(x):
        return x.flatten(1) @ w

    real, fake = torch.randn(5, 3, 4, 4), torch.randn(5, 3, 4, 4)
    assert gradient_penalty(critic, real, fake).item() == pytest.approx(0.0, abs=1e-10)
    assert gradient_penalty(lambda x: 3 * critic(x), real, fake).item() == pytest.approx(4.0)


def test_r1_of_a_linear_critic_is_its_squared_norm():
    w = torch.randn(12)
    real = torch.randn(4, 3, 2, 2, requires_grad=True)
    logits = real.flatten(1) @ w
    assert r1_penalty(logits, real).item() == pytest.approx(w.pow(2).sum().item(), rel=1e-5)


# --- DiffAugment --------------------------------------------------------------


def test_diffaugment_keeps_shape_and_is_differentiable():
    x = torch.randn(4, 3, 16, 16, requires_grad=True)
    out = diff_augment(x, "color,translation,cutout")
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_empty_policy_is_identity_and_bad_names_fail():
    x = torch.randn(2, 3, 8, 8)
    assert torch.equal(diff_augment(x, ""), x)
    with pytest.raises(ValueError):
        parse_policy("color,rotate")


def test_cutout_zeroes_a_quarter_at_most():
    x = torch.ones(64, 1, 16, 16)
    out = diff_augment(x, "cutout")
    zero_fraction = (out == 0).float().flatten(1).mean(1)
    assert zero_fraction.max() <= 0.25 + 1e-6 and zero_fraction.mean() > 0.05


# --- EMA ------------------------------------------------------------------


def test_ema_update_rule():
    model = torch.nn.Linear(2, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    ema = EMA(model, decay=0.5, start=2)
    torch.nn.init.ones_(model.weight)
    ema.update(model, step=0)  # before `start`: copy
    assert torch.equal(ema.model.weight, model.weight)
    torch.nn.init.constant_(model.weight, 3.0)
    ema.update(model, step=2)  # 0.5 * 1 + 0.5 * 3
    assert torch.allclose(ema.model.weight, torch.full((1, 2), 2.0))
    assert not ema.model.weight.requires_grad
