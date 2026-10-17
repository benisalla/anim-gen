# Testing guide

Three levels of checks, from seconds to hours:

1. **Unit tests** (`pytest`): every module, CPU only, no network.
2. **Notebook check**: the Kaggle/Colab notebook executed end to end in smoke mode.
3. **Sanity runs on real data**: does each model family actually learn MNIST?

## 1. Unit tests

```bash
pytest                       # everything (~2 min on a laptop CPU)
pytest tests/test_diffusion.py
pytest -k resume -v
ruff check src tests && ruff format --check src tests
```

Tests use the procedural `shapes` dataset and tiny models: no GPU, no download.

| File | What it proves |
|------|----------------|
| `test_config.py` | invalid settings are rejected early; every preset parses into valid configs; flags override presets; configs survive a round trip through a checkpoint |
| `test_data.py` | images are `[C, S, S]` in [-1, 1]; white → +1, black → −1; folders map sub-folders to classes (nested files included, non-images ignored); `--limit` keeps class names |
| `test_models.py` | every GAN at 8–64 px, conditional or not, gives the right shapes and a tanh-bounded output; spectral norm is on every D conv (and off when disabled); conditioning changes the output; conditional BN and self-attention start as identity; the U-Net gives the right shapes, starts at exactly 0 and depends on t |
| `test_losses.py` | the four GAN losses equal their textbook formulas (non-saturating = BCE with logits); WGAN-GP is 0 for a unit-slope critic and 4 for slope 3; R1 of a linear critic is ‖w‖²; DiffAugment keeps shapes, passes gradients, cutout removes ≤ 1/4; the EMA update rule |
| `test_diffusion.py` | schedules go from clean to noise; with a **perfect denoiser** the DDIM/DDPM sampler returns the exact image, for ε and v prediction, 10 or 100 steps, η = 0 or 1; guidance batches [cond; uncond]; an all-zero model has loss ≈ 1 |
| `test_train.py` | all five training set-ups (DCGAN / ResNet with hinge + DiffAugment + 2 D steps / WGAN-GP / LSGAN + R1 / U-Net) write checkpoints, grids and metrics; **resume** continues the step count and the metrics file; weights and optimizer states reload exactly; FID hook writes `best.pt` at the best score; the diffusion loss halves on shapes in 120 steps; bf16 autocast; the time budget stops and saves; the CLI runs a preset |
| `test_sample.py` | a reloaded checkpoint generates **the same images** as the model that was trained; batching, interpolation and the CLI (grid, folder, class by name, unknown class error); slerp endpoints |
| `test_fid.py` | streaming mean/covariance equal numpy's; the Fréchet distance equals the scipy `sqrtm` formula; identical statistics give 0 |
| `test_prepare.py` | folders and zips become square, resized, class-organised PNG folders at several sizes |
| `test_distributed.py` | **2 processes** (gloo on CPU) that start from *different* random weights end with **identical** weights, for a ResNet GAN with WGAN-GP (double backward across processes) and for the U-Net; only rank 0 writes files |

## 2. Notebook check

The notebook is built from `src/` and was executed end to end on CPU in
smoke mode (`SMOKE_TEST = True`) with `shapes-dcgan` and with a conditional
`shapes-diffusion` (classifier-free guidance, per-class rows, latent walks):
every cell ran. To repeat it:

```bash
python notebooks/build_notebook.py . notebooks/train_colab_kaggle.ipynb
# then open it, set PRESET = "shapes-dcgan" and SMOKE_TEST = True, Run all
```

## 3. Sanity runs on real data

Measured on a shared, heavily loaded 8-core CPU (no GPU), `OMP_NUM_THREADS=2`:

| run | command | result |
|---|---|---|
| conditional DCGAN | `anim-train --preset mnist-dcgan --conditional --steps 4000` | recognisable digits per row from step 500; d_loss ≈ 0.3, g_loss ≈ 3 (stable, no collapse) |
| conditional diffusion | `anim-train --preset mnist-diffusion --unet-channels 16 --steps 2500 --batch-size 32 --lr 1e-3 --warmup-steps 100 --ema-decay 0.995 --guidance-scale 2` | denoising loss 1.0 → 0.036 |

Grids: `docs/images/mnist-dcgan-4k.png`, `docs/images/mnist-diffusion-2.5k.png`.
Not yet done: CIFAR-10 and AFHQ runs (GPU), FID numbers.
