"""Datasets: every item is (image, label), image a float tensor in [-1, 1].

[-1, 1] matches the generator's tanh output and the diffusion model's
Gaussian noise scale. Supported sources:

- "mnist", "fashion-mnist", "cifar10": torchvision, downloaded on first use
- "folder": any directory of images. Sub-directories become classes
  (data/afhq64/cat/*.png, data/afhq64/dog/*.png, ...); a flat folder is one class.
  Use `imagegen-prepare` to build one from a Hugging Face dataset such as AFHQ.
- "shapes": procedural circles / squares / triangles, generated on the fly.
  No download: used by the tests and for quick checks of the whole pipeline.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from imagegen.config import DataConfig

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def build_transform(config: DataConfig, train: bool = True) -> transforms.Compose:
    """PIL image -> [C, H, W] float tensor in [-1, 1]."""
    steps: list = [
        # Resize the short side, then crop the centre: keeps the aspect ratio.
        transforms.Resize(config.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(config.image_size),
    ]
    if train and config.hflip:
        steps.append(transforms.RandomHorizontalFlip())
    steps += [
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize([0.5] * config.channels, [0.5] * config.channels),  # [-1, 1]
    ]
    return transforms.Compose(steps)


class ImageFolderDataset(Dataset):
    """All images under `root`, recursively. First-level sub-folders are classes."""

    def __init__(self, root: Path, transform=None) -> None:
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"image folder not found: {root} (see docs/datasets.md)")
        subdirs = sorted(p.name for p in root.iterdir() if p.is_dir())
        self.classes = subdirs or [root.name]
        self.samples: list[tuple[Path, int]] = []
        for label, name in enumerate(self.classes):
            folder = root / name if subdirs else root
            self.samples += [
                (path, label)
                for path in sorted(folder.rglob("*"))
                if path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        if not self.samples:
            raise FileNotFoundError(f"no images ({', '.join(IMAGE_EXTENSIONS)}) under {root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class ShapesDataset(Dataset):
    """Procedural images: one coloured shape on a dark background.

    Class 0 = circle, 1 = square, 2 = triangle. Item i is always the same image
    (generated from seed i), so the dataset is deterministic and needs no disk.
    """

    def __init__(self, size: int = 2048, image_size: int = 32, hflip: bool = False) -> None:
        self.classes = ["circle", "square", "triangle"]
        self.size = size
        self.image_size = image_size
        self.hflip = hflip
        coords = (torch.arange(image_size) + 0.5) / image_size  # pixel centres in (0, 1)
        self.yy, self.xx = torch.meshgrid(coords, coords, indexing="ij")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        g = torch.Generator().manual_seed(index)
        label = index % 3
        cx, cy = (0.3 + 0.4 * torch.rand(2, generator=g)).tolist()
        r = 0.15 + 0.15 * torch.rand(1, generator=g).item()
        dx, dy = self.xx - cx, self.yy - cy
        if label == 0:
            mask = dx**2 + dy**2 <= r**2
        elif label == 1:
            mask = (dx.abs() <= r) & (dy.abs() <= r)
        else:  # upward triangle: below the two slanted edges, above the base
            mask = (dy <= r) & (dy >= -r + 2 * dx.abs())
        colour = 0.4 + 0.6 * torch.rand(3, 1, 1, generator=g)
        background = 0.15 * torch.rand(3, 1, 1, generator=g)
        image = torch.where(mask, colour, background)  # [3, H, W] in [0, 1]
        if self.hflip and torch.rand(()) < 0.5:
            image = image.flip(-1)
        return image * 2 - 1, label


class _Limited(Subset):
    """Subset that keeps the parent's `classes`."""

    def __init__(self, dataset: Dataset, n: int) -> None:
        super().__init__(dataset, range(min(n, len(dataset))))
        self.classes = dataset_classes(dataset)


def dataset_classes(dataset: Dataset) -> list[str]:
    """Class names of a dataset (one unnamed class when it has none)."""
    return list(getattr(dataset, "classes", None) or ["all"])


def build_dataset(config: DataConfig, train: bool = True) -> Dataset:
    """The dataset described by `config`, split "train" or "test"."""
    transform = build_transform(config, train=train)
    root = Path(config.data_dir)
    if config.dataset == "mnist":
        dataset = datasets.MNIST(root, train=train, transform=transform, download=config.download)
    elif config.dataset == "fashion-mnist":
        dataset = datasets.FashionMNIST(
            root, train=train, transform=transform, download=config.download
        )
    elif config.dataset == "cifar10":
        dataset = datasets.CIFAR10(root, train=train, transform=transform, download=config.download)
    elif config.dataset == "folder":
        dataset = ImageFolderDataset(root, transform=transform)  # no split: generative models
    elif config.dataset == "shapes":
        dataset = ShapesDataset(2048 if train else 512, config.image_size, config.hflip and train)
    else:  # pragma: no cover - DataConfig already validates the name
        raise ValueError(config.dataset)
    if config.limit is not None:
        dataset = _Limited(dataset, config.limit)
    return dataset


def num_classes(config: DataConfig, dataset: Dataset) -> int:
    """Number of classes the model is conditioned on (0 = unconditional)."""
    return len(dataset_classes(dataset)) if config.conditional else 0


__all__ = [
    "ImageFolderDataset",
    "ShapesDataset",
    "build_dataset",
    "build_transform",
    "dataset_classes",
    "num_classes",
]
