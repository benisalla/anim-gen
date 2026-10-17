"""A checkpoint reloads into the same sampler; the CLI writes the files it promises."""

import pytest
import torch
from conftest import tiny_model_config, tiny_train_config
from PIL import Image

from anim_gen.config import DataConfig
from anim_gen.data import build_dataset
from anim_gen.images import image_grid, to_uint8
from anim_gen.sample import generate, interpolate, load_task, main, slerp
from anim_gen.tasks import build_task
from anim_gen.train import LAST_CHECKPOINT, build_loader, train

CLASSES = ["circle", "square", "triangle"]


@pytest.fixture(params=["resnet", "unet"])
def checkpoint(request, tmp_path):
    model_config = tiny_model_config(request.param, num_classes=3)
    train_config = tiny_train_config(tmp_path, steps=2)
    data = DataConfig(dataset="shapes", image_size=16, conditional=True)
    task = build_task(model_config, train_config, torch.device("cpu"))
    train(task, build_loader(build_dataset(data), train_config, torch.device("cpu")),
          data, model_config, train_config, CLASSES)  # fmt: skip
    return task, train_config.output_dir / LAST_CHECKPOINT


def test_reloaded_checkpoint_generates_the_same_images(checkpoint):
    task, path = checkpoint
    loaded, data_config, class_names = load_task(path)
    assert class_names == CLASSES and data_config.dataset == "shapes"
    assert loaded.optimizers == {}  # inference only
    g = torch.Generator().manual_seed(0)
    noise, labels = task.noise(4, g), torch.tensor([0, 1, 2, 0])
    expected = task.generate(noise, labels, steps=4)
    assert torch.allclose(loaded.generate(noise, labels, steps=4), expected, atol=1e-5)


def test_generate_in_batches_and_interpolate(checkpoint):
    task, _ = checkpoint
    images = generate(task, 5, task.random_labels(5), batch_size=2, steps=3)
    assert images.shape == (5, 3, 16, 16)
    walk = interpolate(task, rows=2, cols=4, labels=torch.tensor([0, 1]), steps=3)
    assert walk.shape == (8, 3, 16, 16)


def test_cli_writes_grid_and_folder(checkpoint, tmp_path):
    _, path = checkpoint
    main([
        "--checkpoint", str(path), "--num", "6", "--out", str(tmp_path / "grid.png"),
        "--folder", str(tmp_path / "imgs"), "--class", "square", "--steps", "3", "--device", "cpu",
    ])  # fmt: skip
    assert Image.open(tmp_path / "grid.png").mode == "RGB"
    assert len(list((tmp_path / "imgs").glob("*.png"))) == 6
    with pytest.raises(ValueError, match="unknown class"):
        main(["--checkpoint", str(path), "--class", "unicorn", "--device", "cpu"])


def test_slerp_endpoints_and_norm():
    a, b = torch.randn(3, 64), torch.randn(3, 64)
    assert torch.allclose(slerp(a, b, 0.0), a, atol=1e-5)
    assert torch.allclose(slerp(a, b, 1.0), b, atol=1e-5)
    mid = slerp(a, b, 0.5).norm(dim=1)
    assert (mid > 0.8 * torch.minimum(a.norm(dim=1), b.norm(dim=1))).all()


def test_image_helpers():
    x = torch.tensor([-1.0, 0.0, 1.0, 5.0])
    assert to_uint8(x).tolist() == [0, 128, 255, 255]
    grid = image_grid(torch.zeros(4, 1, 8, 8), nrow=2, padding=0)
    assert grid.shape == (3, 16, 16) and grid.dtype == torch.uint8
