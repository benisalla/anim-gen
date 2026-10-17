"""Turn model outputs (tensors in [-1, 1]) into image files."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    """[-1, 1] float -> [0, 255] uint8, same shape."""
    return ((images.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def image_grid(images: torch.Tensor, nrow: int | None = None, padding: int = 2) -> torch.Tensor:
    """[N, C, H, W] in [-1, 1] -> one [3, H', W'] uint8 grid, as square as possible."""
    nrow = nrow or max(1, int(math.sqrt(images.shape[0])))
    images = images.float().clamp(-1, 1)
    if images.shape[1] == 1:
        images = images.expand(-1, 3, -1, -1)  # grayscale -> RGB so every grid saves the same
    grid = make_grid(images, nrow=nrow, padding=padding, pad_value=-1.0)
    return to_uint8(grid)


def save_grid(images: torch.Tensor, path: Path, nrow: int | None = None) -> Path:
    """Save a batch as one PNG grid. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = image_grid(images.detach().cpu(), nrow=nrow)
    Image.fromarray(grid.permute(1, 2, 0).numpy()).save(path)
    return path


def save_images(images: torch.Tensor, folder: Path, start: int = 0) -> list[Path]:
    """Save every image of a batch as its own PNG: folder/000000.png, ..."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, image in enumerate(to_uint8(images.detach().cpu())):
        array = image.permute(1, 2, 0).numpy()
        if array.shape[2] == 1:
            array = array[:, :, 0]  # PIL wants [H, W] for grayscale
        path = folder / f"{start + i:06d}.png"
        Image.fromarray(array).save(path)
        paths.append(path)
    return paths


__all__ = ["image_grid", "save_grid", "save_images", "to_uint8"]
