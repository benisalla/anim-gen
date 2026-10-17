# Datasets

| `--dataset` | images | size | classes | how it is obtained |
|---|---:|---|---:|---|
| `mnist` | 60k train / 10k test | 28×28 gray → resized to 32 | 10 | torchvision, downloaded to `--data-dir` |
| `fashion-mnist` | 60k / 10k | 28×28 gray → 32 | 10 | torchvision |
| `cifar10` | 50k / 10k | 32×32 RGB | 10 | torchvision |
| `folder` | any | any → `--image-size` | one per sub-folder | your files, or `anim-prepare` |
| `shapes` | 2048 / 512 | `--image-size` | 3 | generated on the fly (no download) |

Images become tensors in [-1, 1]: resize the short side, centre crop,
optional random horizontal flip (`--hflip`, off for the digit presets).
`--conditional` uses the labels (class-conditional generation);
`--limit N` keeps only the first N images.

## AFHQ: high-quality animal faces

[AFHQ](https://github.com/clovaai/stargan-v2#animal-faces-hq-dataset-afhq)
(Choi et al., StarGAN v2) has ~15k 512×512 photos of animal faces in three
classes: **cat, dog, wild** (lions, tigers, foxes, wolves, ...). It is a
standard benchmark for image generation and is exactly what this project is
about. It is mirrored on the Hugging Face Hub as
[`huggan/AFHQ`](https://huggingface.co/datasets/huggan/AFHQ) (3.6 GB).

Decoding and resizing 512×512 images at every step is slow, so prepare a
resized copy once:

```bash
pip install -e '.[hf]'
anim-prepare --source hf:huggan/AFHQ --sizes 64 128 --out 'data/afhq{size}'
```

This streams the dataset (nothing is cached), centre-crops, resizes with a
Lanczos filter and writes `data/afhq64/{cat,dog,wild}/*.png` and the same
for 128. Then:

```bash
anim-train --preset afhq64-resnet          # reads data/afhq64
anim-train --preset afhq64-diffusion
```

If the Hub rate-limits you (HTTP 429), log in once with `hf auth login` or
set `HF_TOKEN`. `--limit 500` prepares a small subset for a quick try.

## Your own images

Any folder works; one sub-folder per class (or a flat folder for one class):

```bash
anim-prepare --source ~/Pictures/my_animals --sizes 64 --out data/mine64
anim-prepare --source anim-gen-sample.zip --sizes 64 --out data/sample64   # zips too
anim-train --preset afhq64-resnet --data-dir data/mine64 --no-conditional
```

`anim-prepare` is optional: `--dataset folder --data-dir DIR` also reads
full-size images directly, just more slowly. With a few hundred to a few
thousand images, use a GAN preset with DiffAugment (the AFHQ ones).

## Other Hugging Face image datasets

`anim-prepare --source hf:<dataset id>` works with any dataset that has an
image column: set `--image-column` / `--label-column` (`--label-column ''`
for none) and `--split`.
