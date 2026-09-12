import hashlib
from pathlib import Path

import pytest
from PIL import Image
from ultrafast_yolo_dataset import scan
from ultrafast_yolo_dataset._scan import ResourceLimitError
from ultralytics.data.utils import verify_image_label


def equal(got, expected):
    assert got["im_file"] == expected[0]
    assert got["shape"] == expected[2]
    assert got["cls"].tobytes() == expected[1][:, :1].tobytes()
    assert got["bboxes"].tobytes() == expected[1][:, 1:].tobytes()
    assert len(got["segments"]) == len(expected[3])
    for a, b in zip(got["segments"], expected[3]):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    assert got["keypoints"] is None and got["normalized"] and got["bbox_format"] == "xywh"


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("chunk", [1, 3, 256])
def test_hybrid_scan_full_reference(tmp_path, workers, chunk):
    texts = [
        "0 .5 .5 .2 .2",
        "",
        None,
        "invalid",
        "0 .5 .5 .2 .2\n0 .5 .5 .2 .2",
        "0 0 0 1 0 1 1",
        "0 0 0 1 0 1 1\n0 .5 .5 .2 .2",
        "nan .5 .5 .2 .2",
        "0\u00a0.5 .5 .2 .2",
        "1_0 .5 .5 .2 .2",
        "0 .5 .5 .2 .2",
    ]
    images, labels = [], []
    for i, text in enumerate(texts):
        image = tmp_path / f"图像 {i}.png"
        label = tmp_path / f"标注 {i}.txt"
        Image.new("RGB", (32, 24)).save(image)
        if text is not None:
            label.write_text(text)
        images.append(str(image))
        labels.append(str(label))
    Path(images[-1]).write_bytes(b"corrupt PNG")
    expected = [verify_image_label((im, lb, "val: ", False, 80, 0, 0, False)) for im, lb in zip(images, labels)]
    result = scan(images, labels, num_classes=80, workers=workers, max_in_flight=chunk, prefix="val: ")
    valid = [r for r in expected if r[0] is not None]
    assert result.num_valid == len(valid)
    for got, ref in zip(result.to_ultralytics_labels(), valid):
        equal(got, ref)
    assert dict(result.summary) == dict(
        total=len(images),
        **{
            name: sum(r[i] for r in expected)
            for name, i in [("missing", 5), ("found", 6), ("empty", 7), ("corrupt", 8)]
        },
    )
    assert [d.message for d in result.diagnostics] == [r[-1] for r in expected if r[-1]]
    assert result.fallback_count == 5
    with pytest.raises(ValueError):
        result.boxes.flags.writeable = True
    first = result.to_ultralytics_labels()
    first[0]["bboxes"][:] = 999
    equal(result.to_ultralytics_labels()[0], valid[0])


@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_exif_shape(tmp_path, orientation):
    im, lb = tmp_path / "a.jpg", tmp_path / "a.txt"
    exif = Image.Exif()
    exif[274] = orientation
    Image.new("RGB", (40, 20)).save(im, exif=exif)
    lb.write_text("0 .5 .5 .1 .1")
    expected = verify_image_label((str(im), str(lb), "", False, 80, 0, 0, False))
    equal(scan([im], [lb], num_classes=80).to_ultralytics_labels()[0], expected)


def test_jpeg_repair_policy_and_byte_identity(tmp_path):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (32, 24), (123, 45, 67)).save(source)
    # A fully decodable JPEG with a non-EOI tail exercises the repair branch;
    # deleting encoded bytes can instead make both Pillow decoders reject it.
    damaged = source.read_bytes() + b"trailing data"
    ref, candidate = tmp_path / "reference.jpg", tmp_path / "candidate.jpg"
    ref.write_bytes(damaged)
    candidate.write_bytes(damaged)
    label = tmp_path / "a.txt"
    label.write_text("0 .5 .5 .2 .2")
    rejected = scan([candidate], [label], num_classes=1)
    assert rejected.num_valid == 0 and rejected.diagnostics[0].code == "repair_required"
    assert candidate.read_bytes() == damaged
    expected = verify_image_label((str(ref), str(label), "", False, 1, 0, 0, False))
    actual = scan([candidate], [label], num_classes=1, repair_jpeg="reference")
    assert actual.num_valid == 1 and expected[0] is not None
    assert hashlib.sha256(candidate.read_bytes()).digest() == hashlib.sha256(ref.read_bytes()).digest()
    assert actual.diagnostics[0].message.replace(str(candidate), str(ref)) == expected[-1]


def test_invalid_image_limits_and_empty_scan(tmp_path):
    im, lb = tmp_path / "tiny.png", tmp_path / "large.txt"
    Image.new("RGB", (5, 5)).save(im)
    lb.write_text("0 .5 .5 .2 .2")
    r = scan([im], [lb], num_classes=1, max_file_bytes=1)
    assert r.summary["corrupt"] == 1 and r.summary["found"] == 0
    Image.new("RGB", (32, 32)).save(im)
    with pytest.raises(ResourceLimitError):
        scan([im], [lb], num_classes=1, max_file_bytes=1)
    empty = scan([], [], num_classes=1)
    assert empty.to_ultralytics_labels() == [] and empty.boxes.shape == (0, 4)


def test_signed_zero_segment_reduction_fallback(tmp_path):
    import numpy as np

    image = tmp_path / "image.png"
    Image.new("RGB", (32, 32)).save(image)
    rng = np.random.default_rng(912)
    labels = []
    for i in range(64):
        points = rng.choice(np.array([0.0, -0.0]), (int(rng.integers(3, 129)), 2))
        points[0, 0] = -0.0
        label = tmp_path / f"{i}.txt"
        label.write_text("0 " + " ".join(str(float(v)) for v in points.ravel()))
        labels.append(label)
    result = scan([image] * len(labels), labels, num_classes=1)
    assert result.fallback_count == 64
    for got, label in zip(result.to_ultralytics_labels(), labels):
        expected = verify_image_label((str(image), str(label), "", False, 1, 0, 0, False))
        equal(got, expected)


def test_repair_provenance_survives_overwritten_label_diagnostics(tmp_path):
    images, labels = [], []
    for index, contents in enumerate(["0 .5 .5 .2 .2\n0 .5 .5 .2 .2", "invalid label"]):
        image, label = tmp_path / f"{index}.jpg", tmp_path / f"{index}.txt"
        Image.new("RGB", (32, 24)).save(image)
        image.write_bytes(image.read_bytes() + b"trailing data")
        label.write_text(contents)
        images.append(image)
        labels.append(label)
    result = scan(images, labels, num_classes=1, repair_jpeg="reference", prefix="train: ")
    assert result.num_valid == 1 and result.summary["corrupt"] == 1
    assert result.repaired_source_indices.tolist() == [0, 1]
    assert result.label_paths == tuple(map(str, labels))
    assert result.config["prefix"] == "train: "
    assert not any("restored and saved" in d.message for d in result.diagnostics)
    assert all(image.read_bytes().endswith(b"\xff\xd9") for image in images)
    with pytest.raises(ValueError):
        result.repaired_source_indices.flags.writeable = True
