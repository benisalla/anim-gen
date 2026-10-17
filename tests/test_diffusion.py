"""The noise schedule and the DDIM sampler are mathematically consistent."""

import pytest
import torch

from anim_gen.diffusion import GaussianDiffusion, make_alpha_bar


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_alpha_bar_goes_from_clean_to_noise(schedule):
    a = make_alpha_bar(1000, schedule)
    assert a.shape == (1000,)
    assert (a[1:] < a[:-1]).all()  # strictly more noise at every step
    assert a[0] > 0.99 and a[-1] < 0.01


def test_q_sample_mixes_image_and_noise():
    d = GaussianDiffusion(100)
    x0, noise = torch.ones(2, 1, 2, 2), torch.zeros(2, 1, 2, 2)
    t = torch.tensor([0, 99])
    out = d.q_sample(x0, t, noise)
    assert torch.allclose(out[0], d.alpha_bar[0].sqrt().float() * x0[0])
    assert out[1].abs().max() < 0.1  # almost all signal is gone at the end


class Oracle(torch.nn.Module):
    """A 'perfect' denoiser for a dataset of one image x0."""

    def __init__(self, diffusion: GaussianDiffusion, x0: torch.Tensor) -> None:
        super().__init__()
        self.d, self.x0 = diffusion, x0
        self.seen_labels = []

    def forward(self, x, t, y=None):
        self.seen_labels.append(y)
        a = self.d.alpha_bar[t].float()[:, None, None, None]
        eps = (x - a.sqrt() * self.x0) / (1 - a).sqrt()
        return self.d.target(self.x0.expand_as(x), t, eps)


@pytest.mark.parametrize("prediction", ["eps", "v"])
@pytest.mark.parametrize("eta,steps", [(0.0, 10), (0.0, 100), (1.0, 100)])
def test_sampler_recovers_the_image_with_a_perfect_model(prediction, eta, steps):
    d = GaussianDiffusion(100, prediction=prediction)
    x0 = torch.rand(1, 3, 4, 4) * 1.6 - 0.8
    out = d.sample(Oracle(d, x0), torch.randn(3, 3, 4, 4), steps=steps, eta=eta)
    assert torch.allclose(out, x0.expand(3, -1, -1, -1), atol=1e-4)


def test_target_matches_definition():
    d = GaussianDiffusion(100, prediction="v")
    x0, noise, t = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4), torch.tensor([10, 80])
    a = d.alpha_bar[t].float()[:, None, None, None]
    assert torch.allclose(d.target(x0, t, noise), a.sqrt() * noise - (1 - a).sqrt() * x0)


def test_guidance_runs_conditional_and_unconditional_in_one_batch():
    d = GaussianDiffusion(100, null_label=5)
    model = Oracle(d, torch.zeros(1, 1, 2, 2))
    d.sample(model, torch.randn(2, 1, 2, 2), y=torch.tensor([0, 3]), steps=2, guidance_scale=3.0)
    assert model.seen_labels[0].tolist() == [0, 3, 5, 5]  # [cond; uncond]


def test_sample_timesteps():
    d = GaussianDiffusion(1000)
    ts = d.sample_timesteps(10)
    assert ts[0] == 999 and ts[-1] == 0 and len(ts) == 10
    assert d.sample_timesteps(5000) == list(range(999, -1, -1))  # capped at T


def test_training_loss_is_mse_to_the_target():
    d = GaussianDiffusion(100)
    zero_model = lambda x, t, y=None: torch.zeros_like(x)
    torch.manual_seed(0)
    loss = d.training_loss(zero_model, torch.randn(256, 1, 8, 8))
    assert loss.item() == pytest.approx(1.0, abs=0.05)  # E[noise^2] = 1
