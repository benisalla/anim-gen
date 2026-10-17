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

Every architecture and every GAN loss trained on MNIST (conditional: row r of a grid
should be digit r). CPU only, on a shared and often saturated 8-core machine,
`OMP_NUM_THREADS=2`, so models and step counts are small. All grids are the EMA weights.

| # | algorithm | command (all with `--conditional` / conditional presets) | result |
|---|---|---|---|
| 1 | DCGAN, non-saturating | `--preset mnist-dcgan --steps 4000` | **works**: correct digit per row from step 500; d_loss ≈ 0.3, g_loss ≈ 3, stable ([grid](images/mnist-dcgan-4k.png)) |
| 2 | DCGAN, LSGAN + R1 (γ = 1) | `--preset mnist-dcgan --loss lsgan --r1-gamma 1 --steps 3000` | **works**: correct digit per row, clean strokes; d_loss ≈ 0.19, r1 ≈ 0.04 ([grid](images/mnist-dcgan-lsgan-3k.png)) |
| 3 | DCGAN, WGAN-GP, BatchNorm critic | `--loss wgan-gp --d-steps 5 --beta1 0 --beta2 0.9 --lr-g 1e-4 --lr-d 1e-4` | **diverges**: critic gap D(real) − D(fake) grew to 223 by step 600 while the penalty stayed ≈ 2.7. BatchNorm mixes samples, the penalty is per sample; the code now warns about this combination |
| 4 | DCGAN, WGAN-GP, spectral-norm critic | same + `--spectral-norm --steps 1500` | **works**: the Wasserstein estimate falls steadily (35 → 3.3), penalty ≈ 0.02; correct digit per row, rougher strokes after only 1.5k G steps ([grid](images/mnist-dcgan-wgangp-sn-1.5k.png)) |
| 5 | ResNet GAN (hinge, SN, self-attention 16, projection D), 16 ch | `--preset cifar10-resnet --dataset mnist --no-hflip --g-channels 16 --d-channels 16 --max-channels 64 --d-steps 1 --batch-size 32 --steps 1500` | **learns digits, weak class control**: digit shapes form, but rows only partly follow the label. The label does change the output (mean pixel change 0.15 vs 0.25 for DCGAN), so conditioning is wired; the tiny D (0.08M) is too weak to enforce it in 1.5k steps ([grid](images/mnist-resnet16-1.5k.png)) |
| 6 | ResNet GAN, 32 ch, 2 D steps | same with `--g-channels 32 --d-channels 32 --max-channels 128 --d-steps 2 --batch-size 64` | **works, best GAN result**: smooth, varied digits, correct digit per row; rows already follow the classes at step 500. With enough D capacity the projection discriminator enforces the label ([grid](images/mnist-resnet32-3k.png)) |
| 7 | diffusion U-Net, ε-prediction | `--preset mnist-diffusion --prediction eps --unet-channels 16 --steps 2500 --batch-size 32 --lr 1e-3 --warmup-steps 100 --ema-decay 0.995 --guidance-scale 2` | **partly**: correct digit per row, but many samples have grey / speckled backgrounds (corner brightness −0.55 instead of −1) with every sampler (DDIM 50, guidance 1 or 2, DDPM-style η = 1 with 250 steps). The denoiser itself is accurate on real noisy images up to t = 900; x₀ = (x − √(1−ᾱ)ε̂)/√ᾱ amplifies errors ~70× at the noisiest steps ([grid](images/mnist-diffusion-2.5k.png)) |
| 8 | diffusion U-Net, v-prediction | same with `--prediction v` | **works**: clean black backgrounds, correct digit per row; loss 1.0 → 0.095 (v-targets have a different scale than ε) ([grid](images/mnist-diffusion-v-2.5k.png)). The MNIST and CIFAR-10 diffusion presets now use v-prediction |

Not done here (needs a GPU and/or the dataset): CIFAR-10 runs (the download was
interrupted on this machine), AFHQ training (data prepared: 16,130 images, 5,653 cat /
5,239 dog / 5,238 wild; a 64 px ResNet GAN run started but ran at < 0.1 it/s), FID numbers.
