"""GAN losses and regularisers. Discriminators output raw logits (no sigmoid).

    loss            discriminator                          generator
    nonsaturating   softplus(-D(x)) + softplus(D(G(z)))    softplus(-D(G(z)))
    hinge           relu(1 - D(x)) + relu(1 + D(G(z)))     -D(G(z))
    wgan-gp         D(G(z)) - D(x)  (+ gradient penalty)   -D(G(z))
    lsgan           (D(x) - 1)^2 + D(G(z))^2  (/2)         (D(G(z)) - 1)^2 / 2

softplus(-l) is exactly BCE(sigmoid(l), 1), computed without overflow.
"Non-saturating": the generator maximises log D(G(z)) instead of minimising
log(1 - D(G(z))), which has vanishing gradients when D wins early on.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F


def d_loss(kind: str, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Discriminator loss (to minimise), averaged over the batch."""
    if kind == "nonsaturating":
        return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
    if kind == "hinge":
        return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
    if kind == "wgan-gp":
        return fake_logits.mean() - real_logits.mean()
    if kind == "lsgan":
        return 0.5 * ((real_logits - 1) ** 2).mean() + 0.5 * (fake_logits**2).mean()
    raise ValueError(f"unknown GAN loss: {kind!r}")


def g_loss(kind: str, fake_logits: torch.Tensor) -> torch.Tensor:
    """Generator loss (to minimise), averaged over the batch."""
    if kind == "nonsaturating":
        return F.softplus(-fake_logits).mean()
    if kind in ("hinge", "wgan-gp"):
        return -fake_logits.mean()
    if kind == "lsgan":
        return 0.5 * ((fake_logits - 1) ** 2).mean()
    raise ValueError(f"unknown GAN loss: {kind!r}")


def gradient_penalty(
    critic: Callable[[torch.Tensor], torch.Tensor], real: torch.Tensor, fake: torch.Tensor
) -> torch.Tensor:
    """WGAN-GP (Gulrajani et al. 2017): E[(||grad D(x_hat)|| - 1)^2].

    x_hat is a random point on the line between a real and a fake image; the
    penalty keeps the critic 1-Lipschitz there.
    """
    alpha = torch.rand(real.shape[0], 1, 1, 1, device=real.device)
    mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    (grad,) = torch.autograd.grad(critic(mixed).sum(), mixed, create_graph=True)
    return ((grad.flatten(1).norm(dim=1) - 1) ** 2).mean()


def r1_penalty(real_logits: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """R1 (Mescheder et al. 2018): E[||grad D(x)||^2] on real images only.

    `real` must have requires_grad=True before D was called on it.
    Use as  loss + gamma / 2 * r1_penalty(...)  (StyleGAN convention).
    """
    (grad,) = torch.autograd.grad(real_logits.sum(), real, create_graph=True)
    return grad.flatten(1).pow(2).sum(dim=1).mean()


__all__ = ["d_loss", "g_loss", "gradient_penalty", "r1_penalty"]
