"""anim-prepare turns folders and zips into square, class-organised PNG folders."""

import zipfile

from PIL import Image

from anim_gen.prepare import center_square, iter_folder, iter_zip, main, resize


def test_center_square_and_resize():
    image = Image.new("RGB", (60, 40))
    assert center_square(image).size == (40, 40)
    assert resize(image, 16).size == (16, 16)


def test_folder_source_to_several_sizes(tmp_path):
    for cls in ("cat", "dog"):
        (tmp_path / "src" / cls).mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (50, 30), (i * 100, 0, 0)).save(tmp_path / "src" / cls / f"{i}.jpg")
    assert [label for _, label in iter_folder(tmp_path / "src")] == ["cat", "cat", "dog", "dog"]
    main(
        [
            "--source",
            str(tmp_path / "src"),
            "--sizes",
            "8",
            "16",
            "--out",
            str(tmp_path / "out{size}"),
        ]
    )
    for size in (8, 16):
        files = sorted((tmp_path / f"out{size}").rglob("*.png"))
        assert [f.parent.name for f in files] == ["cat", "cat", "dog", "dog"]
        assert Image.open(files[0]).size == (size, size)


def test_zip_source(tmp_path):
    archive = tmp_path / "animals.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for i in range(3):
            path = tmp_path / f"{i}.png"
            Image.new("RGB", (20, 20)).save(path)
            z.write(path, f"anim-gen-sample/{i}.png")  # flat archive, like the original data
    assert [label for _, label in iter_zip(archive)] == ["all", "all", "all"]
