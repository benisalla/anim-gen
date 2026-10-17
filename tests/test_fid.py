"""FID maths, checked against the textbook formula and scipy."""

import numpy as np
import pytest
import torch
from conftest import tiny_model_config

from anim_gen.fid import FeatureStats, frechet_distance, generated_stats
from anim_gen.tasks import build_task


def test_feature_stats_match_numpy():
    f = torch.randn(500, 6, dtype=torch.float64) * torch.arange(1, 7) + 3
    stats = FeatureStats(6)
    for chunk in f.split(64):
        stats.add(chunk)
    mu, cov = stats.mean_cov()
    assert torch.allclose(mu, f.mean(0))
    assert torch.allclose(cov, torch.from_numpy(np.cov(f.numpy(), rowvar=False)))


def test_frechet_distance_against_scipy():
    linalg = pytest.importorskip("scipy.linalg")
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(8, 8, generator=g), torch.randn(8, 8, generator=g)
    cov1, cov2 = (a @ a.T).double(), (b @ b.T).double()
    mu1, mu2 = torch.randn(8, generator=g).double(), torch.randn(8, generator=g).double()
    covmean = linalg.sqrtm(cov1.numpy() @ cov2.numpy()).real
    expected = ((mu1 - mu2) ** 2).sum() + cov1.trace() + cov2.trace() - 2 * np.trace(covmean)
    assert frechet_distance(mu1, cov1, mu2, cov2) == pytest.approx(float(expected), rel=1e-6)


def test_identical_statistics_give_zero():
    x = torch.randn(10, 4, dtype=torch.float64)
    cov = x.T @ x
    assert frechet_distance(torch.zeros(4), cov, torch.zeros(4), cov) == pytest.approx(0, abs=1e-6)


def test_generated_stats_uses_the_task():
    task = build_task(tiny_model_config("dcgan", num_classes=3), None, torch.device("cpu"))

    class MeanColour(torch.nn.Module):  # stand-in for Inception: 2048-d features
        def forward(self, images):
            return images.mean(dim=(2, 3)).repeat(1, 683)[:, :2048]

    mu, cov = generated_stats(task, MeanColour(), num_samples=10, batch_size=4)
    assert mu.shape == (2048,) and cov.shape == (2048, 2048)
