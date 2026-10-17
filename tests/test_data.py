"""Datasets return (image in [-1, 1], label); folders map sub-folders to classes."""

import torch
from PIL import Image

from anim_gen.config import DataConfig
from anim_gen.data import (
    ImageFolderDataset,
    ShapesDataset,
    build_dataset,
    build_transform,
    dataset_classes,
    num_classes,
)


def write_png(path, size=(40, 30), colour=(255, 0, 0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)


def test_shapes_dataset_is_deterministic_and_in_range():
    ds = ShapesDataset(size=9, image_size=16)
    image, label = ds[4]
    assert image.shape == (3, 16, 16)
    assert image.min() >= -1 and image.max() <= 1
    assert label == 4 % 3
    assert torch.equal(ds[4][0], image)  # same index -> same image
    assert not torch.equal(ds[5][0], image)
    assert dataset_classes(ds) == ["circle", "square", "triangle"]


def test_transform_resizes_crops_and_normalises():
    config = DataConfig(dataset="folder", image_size=16, hflip=False)
    out = build_transform(config)(Image.new("RGB", (40, 30), (255, 255, 255)))
    assert out.shape == (3, 16, 16)
    assert torch.allclose(out, torch.ones_like(out))  # white -> +1
    out = build_transform(config)(Image.new("RGB", (40, 30), (0, 0, 0)))
    assert torch.allclose(out, -torch.ones_like(out))  # black -> -1


def test_folder_with_class_subfolders(tmp_path):
    for i in range(3):
        write_png(tmp_path / "cat" / f"{i}.png")
    write_png(tmp_path / "dog" / "nested" / "a.jpg")  # nested files belong to their top class
    (tmp_path / "dog" / "notes.txt").write_text("not an image")
    ds = build_dataset(DataConfig(dataset="folder", data_dir=tmp_path, image_size=8))
    assert isinstance(ds, ImageFolderDataset)
    assert ds.classes == ["cat", "dog"]
    assert len(ds) == 4
    image, label = ds[3]
    assert image.shape == (3, 8, 8) and label == 1


def test_flat_folder_is_one_class(tmp_path):
    for i in range(2):
        write_png(tmp_path / f"{i}.png")
    ds = ImageFolderDataset(tmp_path)
    assert len(ds) == 2 and len(ds.classes) == 1


def test_limit_keeps_class_names():
    config = DataConfig(dataset="shapes", image_size=8, limit=5, conditional=True)
    ds = build_dataset(config)
    assert len(ds) == 5
    assert num_classes(config, ds) == 3
    assert num_classes(DataConfig(dataset="shapes", image_size=8), ds) == 0  # unconditional
