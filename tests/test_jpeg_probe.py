"""The native JPEG probe accepts only files whose Pillow check result it reproduces."""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from ultrafast_yolo_dataset import _native, _scan, scan

LIMIT = 64 * 1024**2
LABELS = [
    "0 .5 .5 .2 .2\n1 .3 .3 .1 .1",
    "0 .5 .5 .2 .2\n0 .5 .5 .2 .2",  # duplicate row
    "",  # empty
    None,  # missing
    "0 .1 .1 .8 .1 .8 .8\n1 .2 .2 .4 .2 .4 .4",  # polygons
    "5 .5 .5 .2 .2",  # class out of range: native defers, reference rejects
    "0 .5 .5 .2 .2 é",  # non-ASCII: native defers
    "0 -0 .1 .8 .1 .8 .8",  # negative-zero polygon coordinate: native defers, reference accepts
]


def reference(path):
    try:
        message, shape = _scan._check_image(str(path), "reject")
        return ("ok", shape, message)
    except Exception as error:  # noqa: BLE001 - classify like the reference
        return ("error", type(error).__name__, str(error))


def probe(paths):
    statuses, heights, widths = _native.probe_jpegs([str(p) for p in paths], 4, Image.MAX_IMAGE_PIXELS or 0, LIMIT)
    return list(zip(statuses, heights, widths))


def segment(marker, payload):
    return bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


def insert_after_soi(data, extra):
    return data[:2] + extra + data[2:]


def variants(root):
    rng = np.random.default_rng(0)
    rgb = Image.fromarray(rng.integers(0, 255, (37, 53, 3), dtype=np.uint8))
    files, expect_pillow = {}, set()

    def save(name, image=rgb, **options):
        path = root / f"{name}.jpg"
        image.save(path, format=options.pop("format", "JPEG"), **options)
        files[name] = path
        return path

    save("baseline")
    save("gray", rgb.convert("L"))
    save("cmyk", rgb.convert("CMYK"))
    save("progressive", progressive=True)
    save("optimized", optimize=True, quality=35, subsampling=2)
    save("icc", icc_profile=bytes(range(256)) * 3)
    save("comment", comment=b"hello")
    for orientation in range(1, 9):
        exif = Image.Exif()
        exif[274] = orientation
        exif[271] = "maker"
        save(f"exif{orientation}", exif=exif.tobytes())
    exif = Image.Exif()
    exif[271] = "no orientation"
    save("exif_without_orientation", exif=exif.tobytes())
    xmp = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:Description tiff:Orientation="6"/></x:xmpmeta>'
    save("xmp_orientation", xmp=xmp)
    expect_pillow.add("xmp_orientation")
    save("mpo", format="MPO", save_all=True, append_images=[rgb.rotate(90, expand=True)])
    expect_pillow.add("mpo")
    save("tiny", Image.new("RGB", (9, 40)))
    expect_pillow.add("tiny")
    data = files["baseline"].read_bytes()
    raw = {
        "truncated": data[: len(data) // 2],
        "missing_eoi": data[:-2],
        "empty": b"",
        "garbage": b"not an image at all",
        "fill_bytes": insert_after_soi(data, b"\xff"),
        "photoshop": insert_after_soi(
            data,
            segment(
                0xED, b"Photoshop 3.0\x008BIM\x04\x04\x00\x00\x00\x00\x00\x048BIM\x03\xed\x00\x00\x00\x00\x00\x02ab"
            ),
        ),
        "photoshop_truncated": insert_after_soi(data, segment(0xED, b"Photoshop 3.0\x008BIM\x04\x04")),
        "two_exif": insert_after_soi(
            files["exif6"].read_bytes(), segment(0xE1, b"Exif\x00\x00II*\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00")
        ),
        "short_jfif": insert_after_soi(data, segment(0xE0, b"JFIF")),
        "bad_marker": insert_after_soi(data, b"\xff\x01"),
        "junk": insert_after_soi(data, b"\x00\x11"),
        "bad_dqt": insert_after_soi(data, segment(0xDB, b"\x00" + bytes(10))),
    }
    for name, content in raw.items():
        path = root / f"{name}.jpg"
        path.write_bytes(content)
        files[name] = path
        if name not in ("fill_bytes", "photoshop"):  # valid JPEG structure
            expect_pillow.add(name)
    png = root / "png_named.jpg"
    rgb.save(png, format="PNG")
    files["png_named"] = png
    expect_pillow.add("png_named")
    return files, expect_pillow


def test_probe_reproduces_pillow_or_defers(tmp_path):
    files, expect_pillow = variants(tmp_path)
    names = sorted(files)
    results = dict(zip(names, probe([files[n] for n in names])))
    for name in names:
        status, height, width = results[name]
        outcome = reference(files[name])
        if name in expect_pillow:
            assert status == 1, name
        if status == 0:
            assert outcome == ("ok", (height, width), ""), (name, outcome, (height, width))
    accepted = {n for n in names if results[n][0] == 0}
    assert {"baseline", "gray", "cmyk", "progressive", "icc", "comment", "fill_bytes", "photoshop"} <= accepted
    assert {f"exif{k}" for k in range(1, 9)} <= accepted
    assert results["exif6"][1:] == (53, 37) and results["exif1"][1:] == (37, 53)


def test_decompression_limit_and_byte_limit_defer(tmp_path):
    path = tmp_path / "big.jpg"
    Image.new("RGB", (200, 100)).save(path)
    assert list(_native.probe_jpegs([str(path)], 1, 100 * 200, LIMIT)[0]) == [0]
    assert list(_native.probe_jpegs([str(path)], 1, 100 * 200 - 1, LIMIT)[0]) == [1]
    assert list(_native.probe_jpegs([str(path)], 1, 0, os.path.getsize(path) - 1)[0]) == [1]


def pairs(root):
    images, labels = root / "images", root / "labels"
    images.mkdir()
    labels.mkdir()
    (root / "variants").mkdir()
    files, _ = variants(root / "variants")
    paths = []
    for i, (name, source) in enumerate(sorted(files.items())):
        target = images / f"{i:03d}_{name}.jpg"
        target.write_bytes(Path(source).read_bytes())
        text = LABELS[i % len(LABELS)]
        if text is not None:
            (labels / f"{i:03d}_{name}.txt").write_text(text)
        paths.append(str(target))
    return paths, [p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt" for p in paths]


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_scan_with_probe_matches_pillow_only_scan(tmp_path, monkeypatch, task):
    images, labels = pairs(tmp_path)
    fast = scan(images, labels, num_classes=2, task=task, workers=3, max_in_flight=7)
    monkeypatch.setattr(_scan, "NATIVE_JPEG_PROBE", False)
    slow = scan(images, labels, num_classes=2, task=task, workers=3, max_in_flight=7)
    for name in ("source_indices", "image_shapes", "object_offsets", "_labels", "segment_points", "segment_offsets"):
        a, b = getattr(fast, name), getattr(slow, name)
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), name
    assert dict(fast.summary) == dict(slow.summary)
    assert fast.diagnostics == slow.diagnostics


@pytest.mark.skipif(
    not os.environ.get("UYD_REAL_JPEG_DIR"), reason="set UYD_REAL_JPEG_DIR to a directory of real JPEGs"
)
def test_real_jpegs_match_pillow():
    paths = sorted(str(p) for p in Path(os.environ["UYD_REAL_JPEG_DIR"]).rglob("*.jpg"))
    results = probe(paths)
    accepted = 0
    for path, (status, height, width) in zip(paths, results):
        if status == 0:
            accepted += 1
            assert reference(path) == ("ok", (height, width), ""), path
    assert accepted > 0.9 * len(paths)
