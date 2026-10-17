"""Build a training image folder from a Hugging Face dataset, a folder or a zip.

High-resolution datasets are slow to decode and resize at every step, so we
do it once: centre-crop to a square, resize with a high-quality filter, and
write one PNG per image, one sub-folder per class:

    data/afhq64/cat/000000.png  data/afhq64/dog/000001.png  ...

Examples:
    # AFHQ (Animal Faces HQ: cat / dog / wild, 512x512), streamed from the Hub
    imagegen-prepare --source hf:huggan/AFHQ --sizes 64 128 --out 'data/afhq{size}'
    # your own photos: a folder (sub-folders = classes) or a zip of one
    imagegen-prepare --source ~/Pictures/animals.zip --sizes 64 --out 'data/animals{size}'

Then train with:  imagegen-train --preset afhq64-resnet --data-dir data/afhq64
"""

from __future__ import annotations

import argparse
import io
import logging
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath

from PIL import Image

from imagegen.data import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


def center_square(image: Image.Image) -> Image.Image:
    """Crop the largest centred square."""
    w, h = image.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return image.crop((left, top, left + side, top + side))


def resize(image: Image.Image, size: int) -> Image.Image:
    """Square-crop and resize with Lanczos (sharp, little aliasing)."""
    return center_square(image.convert("RGB")).resize((size, size), Image.Resampling.LANCZOS)


def iter_folder(root: Path) -> Iterator[tuple[Image.Image, str]]:
    """(image, class name) for every image under root; sub-folders are classes."""
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    groups = [(d, d.name) for d in subdirs] or [(root, "all")]
    for folder, label in groups:
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                with Image.open(path) as img:
                    yield img.convert("RGB"), label


def iter_zip(path: Path) -> Iterator[tuple[Image.Image, str]]:
    """Same as iter_folder, reading straight from a zip archive (no extraction)."""
    with zipfile.ZipFile(path) as archive:
        names = sorted(
            n for n in archive.namelist()
            if PurePosixPath(n).suffix.lower() in IMAGE_EXTENSIONS and "__MACOSX" not in n
        )  # fmt: skip
        parents = {PurePosixPath(n).parent for n in names}
        # One common parent = flat archive (one class); otherwise the parent folder is the class.
        flat = len(parents) == 1
        for name in names:
            label = "all" if flat else PurePosixPath(name).parent.name
            with Image.open(io.BytesIO(archive.read(name))) as img:
                yield img.convert("RGB"), label


def iter_hf(
    dataset_id: str, split: str, image_column: str, label_column: str | None
) -> Iterator[tuple[Image.Image, str]]:
    """Stream a Hugging Face image dataset (nothing is cached on disk)."""
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise SystemExit("imagegen-prepare hf:... needs `pip install -e '.[hf]'`") from e
    dataset = load_dataset(dataset_id, split=split, streaming=True)
    names = None
    if label_column is not None:
        feature = dataset.features.get(label_column) if dataset.features else None
        names = getattr(feature, "names", None)
    for row in dataset:
        label = "all"
        if label_column is not None and label_column in row:
            value = row[label_column]
            label = names[value] if names is not None else str(value)
        yield row[image_column].convert("RGB"), label


def prepare(
    images: Iterator[tuple[Image.Image, str]],
    out_pattern: str,
    sizes: Sequence[int],
    limit: int | None = None,
) -> dict[int, int]:
    """Write every image at every size. Returns {size: images written}."""
    counts = dict.fromkeys(sizes, 0)
    for index, (image, label) in enumerate(images):
        if limit is not None and index >= limit:
            break
        for size in sizes:
            folder = Path(out_pattern.format(size=size)) / label
            folder.mkdir(parents=True, exist_ok=True)
            resize(image, size).save(folder / f"{index:06d}.png")
            counts[size] += 1
        if (index + 1) % 1000 == 0:
            logger.info("%d images done", index + 1)
    return counts


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Resize a dataset into a training image folder.")
    p.add_argument("--source", required=True, help="hf:<dataset id>, a folder, or a .zip")
    p.add_argument("--out", required=True, help="output folder; '{size}' is replaced per size")
    p.add_argument("--sizes", type=int, nargs="+", default=[64])
    p.add_argument("--split", default="train", help="Hugging Face split")
    p.add_argument("--image-column", default="image")
    p.add_argument("--label-column", default="label", help="'' for no classes")
    p.add_argument("--limit", type=int, default=None, help="stop after N images")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # hide Hugging Face HTTP logs

    if len(args.sizes) > 1 and "{size}" not in args.out:
        p.error("with several --sizes, --out must contain '{size}'")
    if args.source.startswith("hf:"):
        images = iter_hf(args.source[3:], args.split, args.image_column, args.label_column or None)
    else:
        source = Path(args.source).expanduser()
        images = iter_zip(source) if source.suffix == ".zip" else iter_folder(source)
    counts = prepare(images, args.out, args.sizes, args.limit)
    for size, n in counts.items():
        logger.info("%s: %d images", args.out.format(size=size), n)


if __name__ == "__main__":
    main()


__all__ = ["center_square", "iter_folder", "iter_hf", "iter_zip", "prepare", "resize"]
