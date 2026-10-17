# Generating images

A checkpoint (`last.pt` or `best.pt`) holds the configs it was trained
with, so it rebuilds its own model: no architecture flags needed. Sampling
always uses the **EMA** weights.

## Command line

```bash
# a grid of 64 images
anim-sample --checkpoint checkpoints/run/last.pt --num 64 --out samples.png

# conditional model: one class (name or id)
anim-sample --checkpoint checkpoints/afhq64/last.pt --class cat --num 16 --out cats.png

# one PNG per image, e.g. to compute FID with another tool
anim-sample --checkpoint ... --num 10000 --folder generated/ --batch-size 250

# latent walks: 4 rows of 8 images between two random noise inputs
anim-sample --checkpoint ... --interpolate 4 --out walk.png
```

Diffusion-only flags:

| flag | default | meaning |
|---|---|---|
| `--steps` | 100 | DDIM steps; more = slower and usually a little better (50–250) |
| `--eta` | 0.0 | 0 = deterministic DDIM; 1 = DDPM-like fresh noise at each step |
| `--guidance-scale` | 1.0 | classifier-free guidance for conditional models; 1.5–3 = sharper, less diverse |

Common flags: `--seed` (same seed = same images), `--device`,
`--precision bf16` (faster on recent GPUs).

## Python

```python
import torch
from anim_gen.sample import generate, interpolate, load_task
from anim_gen.images import save_grid

task, data_config, class_names = load_task("checkpoints/run/last.pt", device="cuda")
g = torch.Generator().manual_seed(0)

labels = None
if task.conditional:  # e.g. 8 images of each class
    labels = torch.arange(len(class_names)).repeat_interleave(8).to(task.device)
images = generate(task, len(labels) if labels is not None else 64, labels, generator=g,
                  steps=100, guidance_scale=2.0)   # [N, C, H, W] in [-1, 1], on CPU
save_grid(images, "grid.png", nrow=8)
```

`task.generate(noise, labels)` is the low-level call: `task.noise(n)` gives
the right kind of noise (a z vector for GANs, an image-shaped tensor for
diffusion).

## FID

```bash
anim-fid --checkpoint checkpoints/run/last.pt --num-samples 10000
```

Compares 10k generated images with the dataset's test split (or the folder),
using Inception-v3 features. See [architecture.md](architecture.md#5-fid-fidpy)
for what the number means.
