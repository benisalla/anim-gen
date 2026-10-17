"""Build notebooks/train_colab_kaggle.ipynb: a self-contained training notebook.

Every module of src/anim_gen/ is written into the notebook as a `%%writefile`
cell, so the notebook runs the exact package code and needs no GitHub access.
Regenerate after changing src/anim_gen (keeps the notebook identical to the code):
    python notebooks/build_notebook.py . notebooks/train_colab_kaggle.ipynb
"""

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
PACKAGE = ROOT / "src" / "anim_gen"

# Written in dependency order; each one gets its own cell.
MODULES = [
    ("config.py", "every hyperparameter + the named presets"),
    ("data.py", "datasets: MNIST, CIFAR-10, image folders, procedural shapes"),
    ("images.py", "tensor -> PNG helpers"),
    ("runtime.py", "device, precision, multi-GPU helpers"),
    ("models/layers.py", "spectral norm, conditional BatchNorm, self-attention"),
    ("models/dcgan.py", "DCGAN generator / discriminator"),
    ("models/resnet_gan.py", "ResNet GAN (SNGAN / BigGAN style)"),
    ("models/unet.py", "diffusion U-Net"),
    ("models/__init__.py", "architecture registry"),
    ("losses.py", "GAN losses, gradient penalty, R1"),
    ("diffaugment.py", "DiffAugment"),
    ("ema.py", "EMA of the weights"),
    ("diffusion.py", "noise schedule, DDIM / DDPM sampler, guidance"),
    ("tasks.py", "GANTask / DiffusionTask"),
    ("fid.py", "FID"),
    ("train.py", "the training loop"),
    ("sample.py", "sampling helpers"),
    ("prepare.py", "dataset preparation (AFHQ)"),
    ("__init__.py", "package"),
]

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


# ---------------------------------------------------------------------------
md("""
# anim-gen — training notebook (Kaggle & Colab)

Train an image generator on a **free Kaggle or Colab GPU**: a GAN (DCGAN or
ResNet-SNGAN) or a diffusion model (DDPM/DDIM U-Net), on MNIST, CIFAR-10 or
**AFHQ animal faces** (cat / dog / wild).

The notebook is **self-contained**: section 2 writes the `anim_gen` package
(identical to the repo) to disk, then everything uses it. Pick a preset in
section 3 and press **Run all**.

| | Kaggle | Colab |
|---|---|---|
| GPU | *Settings → Accelerator → GPU T4 x2* (or P100) | *Runtime → Change runtime type → T4 GPU* |
| Internet | *Settings → Internet → On* (to download data) | on by default |
| Session limit | 12 h per session, 30 h GPU / week | ~12 h, can disconnect when idle |
| Checkpoints kept in | `/kaggle/working` (saved with "Save Version") | Google Drive if `USE_GOOGLE_DRIVE = True` |

**Sections:** 1. Install · 2. The package · 3. Settings · 4. Data · 5. Model ·
6. Live dashboard · 7. Training · 8. Results · 9. Download
""")

md("""
## 1. Install

Kaggle and Colab already have PyTorch, torchvision and matplotlib. `datasets`
is only needed to download AFHQ from the Hugging Face Hub.
""")
code("""
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets>=3.6"])
print("ready")
""")

md("""
## 2. The `anim_gen` package

Each cell below writes one file of the repo's `src/anim_gen/` package
(generated from the repo by `notebooks/build_notebook.py`). You do not need to
read them to train; they are here so the notebook runs anywhere, offline.
""")
code("""
import os

os.makedirs("anim_gen/models", exist_ok=True)
""")
for name, what in MODULES:
    source = (PACKAGE / name).read_text().rstrip() + "\n"
    cells.append(nbf.v4.new_markdown_cell(f"`anim_gen/{name}`: {what}"))
    cells.append(nbf.v4.new_code_cell(f"%%writefile anim_gen/{name}\n{source}"))
code("""
import importlib
import sys

sys.path.insert(0, os.getcwd())
importlib.invalidate_caches()
import anim_gen

print("anim_gen", anim_gen.__version__, "ready")
""")

# ---------------------------------------------------------------------------
md("""
## 3. Settings

Pick a preset (or change any field below it). Good first choices:

| preset | what | T4 time (rough) |
|---|---|---|
| `mnist-dcgan` | classic DCGAN on digits | minutes |
| `cifar10-resnet` | class-conditional ResNet GAN, hinge loss, spectral norm, self-attention | a few hours |
| `cifar10-diffusion` | class-conditional DDPM U-Net, 38M params | many hours (resume across sessions) |
| `afhq64-resnet` | animal faces 64×64, ResNet GAN + DiffAugment | several hours |
| `afhq64-diffusion` | animal faces 64×64, U-Net, v-prediction | long (resume across sessions) |

`SMOKE_TEST = True` runs a tiny version of the chosen preset in a few minutes,
to check the whole notebook. Keep it `False` for real training.
""")
code("""
import dataclasses
import logging
from pathlib import Path

import torch

from anim_gen.config import PRESETS, DataConfig, ModelConfig, TrainConfig
from anim_gen.train import configs_from_args, parse_args

PRESET = "cifar10-resnet"   # see the table above, or anim_gen.config.PRESETS
SMOKE_TEST = False          # True = tiny quick check of the whole notebook
USE_GOOGLE_DRIVE = False    # Colab only: keep checkpoints on Drive (survive disconnects)
RESUME = True               # continue from last.pt in OUTPUT_DIR if it exists
TIME_BUDGET_HOURS = 11.0    # stop cleanly (and save) before the 12 h session limit
FID_EVERY = 0               # e.g. 5000: FID on 5k samples (downloads Inception, ~100 MB)
OVERRIDES = {}              # any config field, e.g. {"steps": 20000, "batch_size": 128}

# --- where things go -------------------------------------------------------------
ON_KAGGLE = Path("/kaggle/working").exists()
ON_COLAB = "COLAB_RELEASE_TAG" in os.environ
if ON_COLAB and USE_GOOGLE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    WORK = Path("/content/drive/MyDrive/anim-gen")
elif ON_KAGGLE:
    WORK = Path("/kaggle/working")
elif ON_COLAB:
    WORK = Path("/content")
else:
    WORK = Path(".")
DATA_DIR = (Path("/kaggle/temp") if ON_KAGGLE else WORK) / "data"  # Kaggle: temp = not saved
OUTPUT_DIR = WORK / "checkpoints" / (PRESET + ("-smoke" if SMOKE_TEST else ""))

# --- configs: preset -> overrides -> environment ---------------------------------------
args = parse_args(["--preset", PRESET])
data_config, model_config, train_config = configs_from_args(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = DATA_DIR / data_config.data_dir.name if data_config.dataset == "folder" else DATA_DIR
env = {
    "data_dir": data_dir, "output_dir": OUTPUT_DIR, "device": device.type,
    "num_workers": min(4, os.cpu_count() or 1), "fid_every": FID_EVERY,
    "time_budget_hours": TIME_BUDGET_HOURS, "log_every": 100,
}
if SMOKE_TEST:
    env.update(steps=200, log_every=20, sample_every=100, ckpt_every=100, limit=512,
               batch_size=32, fid_every=0, sample_steps=20)
settings = {**env, **OVERRIDES}


def apply(config, values):
    names = {f.name for f in dataclasses.fields(config)}
    return dataclasses.replace(config, **{k: v for k, v in values.items() if k in names})


data_config = apply(data_config, settings)
model_config = apply(model_config, settings)
train_config = apply(train_config, settings)
if RESUME and (OUTPUT_DIR / "last.pt").exists():
    train_config = dataclasses.replace(train_config, resume=OUTPUT_DIR / "last.pt")

if device.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} ({props.total_memory / 1e9:.0f} GB), compute {props.major}.{props.minor}"
          f" -> precision {'bf16' if props.major >= 8 else 'fp32'}")
else:
    print("No GPU: enable one in the notebook settings (fine only for SMOKE_TEST).")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                    force=True)
for name, cfg in [("data", data_config), ("model", model_config), ("train", train_config)]:
    print(f"\\n{name}:", ", ".join(f"{k}={v}" for k, v in dataclasses.asdict(cfg).items()))
""")

# ---------------------------------------------------------------------------
md("""
## 4. Data

MNIST and CIFAR-10 download automatically. AFHQ (Animal Faces HQ, 512×512,
~15k images of cats, dogs and wild animals) is streamed from the Hugging Face
Hub and resized once to the preset's resolution (~10 minutes for 64×64).
If the Hub rate-limits you, add a token: *Add-ons → Secrets* `HF_TOKEN` on
Kaggle, or `os.environ["HF_TOKEN"] = "..."`.
""")
code("""
from anim_gen.data import build_dataset, dataset_classes
from anim_gen.prepare import iter_hf, prepare

if data_config.dataset == "folder" and not any(data_config.data_dir.rglob("*.png")):
    print(f"preparing AFHQ at {data_config.image_size}px in {data_config.data_dir} ...")
    limit = data_config.limit if SMOKE_TEST else None
    counts = prepare(iter_hf("huggan/AFHQ", "train", "image", "label"),
                     str(data_config.data_dir), [data_config.image_size], limit=limit)
    print(counts)

dataset = build_dataset(data_config)
print(f"{len(dataset):,} images, classes: {dataset_classes(dataset)}")
""")
code("""
import matplotlib.pyplot as plt

from anim_gen.images import image_grid


def show(images, title="", nrow=None, size=6):
    grid = image_grid(images, nrow=nrow)
    plt.figure(figsize=(size, size))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.title(title)
    plt.axis("off")
    plt.show()


real = torch.stack([dataset[i][0] for i in torch.randperm(len(dataset))[:64].tolist()])
show(real, "real training images")
""")

# ---------------------------------------------------------------------------
md("""
## 5. Model

`build_and_train` (next sections) builds the networks from the configs; here
we build them once on their own to show their sizes.
""")
code("""
from anim_gen.data import num_classes
from anim_gen.models import count_parameters
from anim_gen.tasks import build_task

shaped = dataclasses.replace(model_config, num_classes=num_classes(data_config, dataset))
preview_task = build_task(shaped, train_config, torch.device("cpu"))
for name, net in preview_task.networks.items():
    if name not in ("G_ema", "ema"):
        print(f"{name:5} {count_parameters(net) / 1e6:6.2f}M parameters")
print("family:", shaped.family, "| conditional:", shaped.num_classes > 0)
del preview_task
""")

# ---------------------------------------------------------------------------
md("""
## 6. Live dashboard

Redrawn at every log step: the losses, and the newest preview grid (same
fixed noise every time, so you can watch the samples sharpen; for a
conditional model each row is one class).

- **GAN**: `d_loss` / `g_loss` should *not* go to 0; a stable run keeps them
  roughly level. `d_real` / `d_fake` are the discriminator's mean logits.
- **diffusion**: `loss` is the denoising MSE (starts ≈ 1, falls fast, then slowly).
""")
code("""
import json

from IPython.display import clear_output
from PIL import Image

history = {"train": [], "fid": [], "sample": None}
if train_config.resume is not None and (OUTPUT_DIR / "metrics.jsonl").exists():
    for line in (OUTPUT_DIR / "metrics.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["kind"] == "sample":
            history["sample"] = event["path"]
        elif event["kind"] in history:
            history[event["kind"]].append(event)


def dashboard(event):
    kind = event["kind"]
    if kind == "sample":
        history["sample"] = event["path"]
        return
    history[kind].append(event)
    if kind != "train":
        return
    rows = history["train"]
    skip = {"kind", "step", "epoch", "total_steps", "it_per_sec"}
    keys = [k for k in rows[-1] if k not in skip]
    clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 2]})
    steps = [r["step"] for r in rows]
    for key in keys:
        ax1.plot(steps, [r.get(key, float("nan")) for r in rows], label=key)
    for r in history["fid"]:
        ax1.annotate(f"FID {r['fid']:.1f}", (r["step"], 0), fontsize=8)
    ax1.set_xlabel("step")
    ax1.grid(alpha=0.3)
    ax1.legend()
    last = rows[-1]
    eta = (last["total_steps"] - last["step"]) / max(last["it_per_sec"], 1e-9) / 60
    ax1.set_title(f"step {last['step']:,}/{last['total_steps']:,} | "
                  f"{last['it_per_sec']:.1f} it/s | ETA {eta:.0f} min")
    if history["sample"]:
        ax2.imshow(Image.open(history["sample"]))
        ax2.set_title(Path(history["sample"]).stem)
    ax2.axis("off")
    fig.tight_layout()
    plt.show()
""")

# ---------------------------------------------------------------------------
md("""
## 7. Training

Same code as the repo's `anim-train`. It writes to `OUTPUT_DIR`:
`last.pt` (every `ckpt_every` steps), `best.pt` (lowest FID, if `FID_EVERY > 0`),
`samples/step_*.png`, `real.png` and `metrics.jsonl`. The checkpoints work with
the repo's `anim-sample`, `anim-fid` and `anim-train --resume`.

**Interrupted, or hit the time budget?** Run the notebook again with
`RESUME = True`: it continues from `last.pt`. On Kaggle, to continue in a *new*
session, add the previous version's output as input data and copy its
`checkpoints/` folder to `/kaggle/working` first.
""")
code("""
from anim_gen.train import build_and_train

task, state = build_and_train(data_config, model_config, train_config, callback=dashboard)
print(f"stopped at step {state.step:,}; outputs in {OUTPUT_DIR}")
""")

# ---------------------------------------------------------------------------
md("""
## 8. Results

Samples from the EMA weights. For a conditional model: one row per class, then
latent walks (spherical interpolation between two noise inputs).
""")
code("""
from anim_gen.sample import generate, interpolate

g = torch.Generator().manual_seed(0)
kwargs = {"steps": 100, "guidance_scale": 2.0 if task.conditional else 1.0}
if task.conditional:
    classes = dataset_classes(dataset)
    per_class = 8
    labels = torch.arange(len(classes)).repeat_interleave(per_class).to(task.device)
    images = generate(task, len(labels), labels, generator=g, **kwargs)
    show(images, "one row per class: " + ", ".join(classes), nrow=per_class, size=8)
else:
    show(generate(task, 64, generator=g, **kwargs), "samples", size=8)
walk = interpolate(task, rows=4, cols=8, labels=task.random_labels(4, g), generator=g, **kwargs)
show(walk, "latent walks", nrow=8, size=8)
""")

# ---------------------------------------------------------------------------
md("""
## 9. Download the trained model

`anim-gen-run.zip` contains `last.pt`, `best.pt` (if any), the sample grids
and the metrics. Locally: `anim-sample --checkpoint last.pt --num 64`.
""")
code("""
import shutil

archive = shutil.make_archive(str(WORK / "anim-gen-run"), "zip", OUTPUT_DIR)
print("saved", archive)
if ON_COLAB:
    from google.colab import files

    files.download(archive)
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
}
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
