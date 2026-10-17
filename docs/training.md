# Training guide

Everything goes through one command, `imagegen-train`, configured by a **preset**
plus any flag you want to override:

```bash
imagegen-train --list-presets
imagegen-train --preset cifar10-resnet                        # as is
imagegen-train --preset cifar10-resnet --batch-size 128 --steps 100000 --output-dir checkpoints/c10
imagegen-train --help                                         # every flag, with its default
```

Every flag is a field of `DataConfig`, `ModelConfig` or `TrainConfig`
(`src/imagegen/config.py`), spelled with dashes (`--lr-g`, `--attn-resolution`).
Booleans have a `--no-` form (`--no-hflip`, `--no-conditional`).

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'               # + '.[hf]' to prepare AFHQ, '.[tensorboard]' for TensorBoard
```

CPU-only machine: install the CPU wheel of PyTorch first
(`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`).

## 2. Presets

| preset | model | data | default steps | notes |
|---|---|---|---:|---|
| `mnist-dcgan` | DCGAN 0.6M + 0.2M | MNIST 32×32 | 10k | add `--conditional` for one digit per row |
| `mnist-diffusion` | U-Net 1.7M, conditional | MNIST 32×32 | 20k | |
| `cifar10-dcgan` | DCGAN | CIFAR-10 | 50k | unconditional baseline |
| `cifar10-resnet` | ResNet GAN, conditional, SN, attention 16 | CIFAR-10 | 60k | `d_steps 2`, hinge |
| `cifar10-diffusion` | U-Net 38M (DDPM CIFAR size), conditional | CIFAR-10 | 200k | the quality option |
| `afhq64-resnet` | ResNet GAN, conditional, DiffAugment | AFHQ 64×64 | 100k | |
| `afhq128-resnet` | ResNet GAN, conditional, DiffAugment | AFHQ 128×128 | 150k | batch 32 |
| `afhq64-diffusion` | U-Net 92M, v-prediction | AFHQ 64×64 | 300k | |
| `afhq128-diffusion` | U-Net 95M, v-prediction | AFHQ 128×128 | 400k | batch 32 |
| `shapes-dcgan`, `shapes-diffusion` | tiny | procedural shapes 16×16 | 500 | no download; pipeline checks |

AFHQ presets read `data/afhq64` / `data/afhq128`: build them first, see
[datasets.md](datasets.md).

## 3. What a run writes

```
checkpoints/run/
  last.pt                  every --ckpt-every steps, and at the end (resume / sample from it)
  best.pt                  lowest FID so far (only with --fid-every > 0)
  real.png                 a grid of training images, for comparison
  samples/step_0001000.png preview grid every --sample-every steps (EMA weights, fixed noise)
  metrics.jsonl            one JSON line per log step / sample / FID
  tensorboard/             with --tensorboard
```

Preview grids always use the **same noise**, so you can watch the same
samples improve. For a conditional model, row r shows class r mod K.

Read the logs:

- **GAN**: `d_loss`, `g_loss`, and `d_real` / `d_fake` (mean D logits on real
  and fake images). A healthy run keeps them roughly level; `d_loss` → 0 and
  `g_loss` climbing for thousands of steps means D won (try `--diffaug`,
  a lower `--lr-d`, or `--r1-gamma`). Judge by the grids and FID, not the losses.
- **diffusion**: `loss` is the denoising MSE: ≈ 1 at the start, then falling
  slowly for a long time; `grad_norm` before clipping.

## 4. GPU, multi-GPU, multi-node

```bash
imagegen-train --preset cifar10-resnet                         # 1 GPU (used automatically)
torchrun --nproc_per_node=4 -m imagegen.train --preset cifar10-diffusion    # 4 GPUs
# 2 machines x 8 GPUs (run on each machine, same --rdzv-endpoint):
torchrun --nnodes=2 --nproc_per_node=8 --rdzv-backend=c10d --rdzv-endpoint=HOST:29500 \
         -m imagegen.train --preset afhq128-diffusion
```

`--batch-size` is **per GPU**. Each process computes its own gradients and
one all-reduce averages them before every optimizer step; only rank 0 writes
files. On SLURM, wrap the same `torchrun` line in `srun`.

**Precision**: `--precision auto` (default) uses bf16 autocast on Ampere and
newer GPUs (A100, L4, RTX 30xx/40xx, H100) and fp32 elsewhere (T4, P100).

## 5. Kaggle / Colab

[`notebooks/train_colab_kaggle.ipynb`](../notebooks/train_colab_kaggle.ipynb)
is self-contained: upload it, pick a `PRESET`, Run all. It shows a live
dashboard (loss curves + the newest sample grid), stops cleanly before the
12 h limit (`TIME_BUDGET_HOURS`) and resumes from `last.pt`. The same budget
is available on the CLI: `--time-budget-hours 11.5`.

## 6. Resume

```bash
imagegen-train --preset cifar10-resnet --output-dir checkpoints/c10 --resume checkpoints/c10/last.pt
```

Weights, EMA weights, optimizer and LR-scheduler state, RNG state and the
step counter are restored; `metrics.jsonl` keeps the events up to that step.
Use the **same preset and flags** as the original run (`--steps` may be raised).

## 7. FID during training

```bash
imagegen-train --preset cifar10-resnet --fid-every 5000 --fid-samples 10000
```

Downloads Inception-v3 weights (~100 MB) once. Real-image statistics are
computed once and cached in the output folder. `best.pt` follows the lowest
FID. On several GPUs the work is split between them. Afterwards:
`imagegen-fid --checkpoint checkpoints/run/best.pt --num-samples 10000`.

## 8. Tuning notes

- **GAN on a small dataset** (AFHQ, your own photos): keep `--diffaug
  color,translation,cutout`; consider `--r1-gamma 0.1`–`1`.
- **WGAN-GP**: use `--loss wgan-gp --d-steps 5 --beta1 0 --beta2 0.9`; set
  `--no-spectral-norm` for the classic version.
- **Diffusion**: quality keeps improving for a long time; the EMA (`--ema-decay
  0.9995`) matters a lot. Sample with `--steps 100`–`250`, and for a conditional
  model try `--guidance-scale 1.5`–`3`.
- **Memory**: lower `--batch-size`; for the U-Net also `--unet-channels` or fewer
  `--attn-resolutions` (attention at 32×32+ is expensive).
- **CPU-only**: shrink the model, e.g. MNIST diffusion with `--unet-channels 16
  --batch-size 32 --lr 1e-3` (see [testing.md](testing.md) for measured runs).
