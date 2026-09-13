"""Fast index/cache parity against the unmodified pinned Ultralytics dataset."""

import copy
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from ultrafast_yolo_dataset import _fast, _native
from ultrafast_yolo_dataset.ultralytics import FastYOLODataset
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data import dataset
from ultralytics.data.utils import IMG_FORMATS

DETECT = "0 .5 .5 .4 .4\n1 .3 .3 .2 .2"
SEGMENT = "0 .1 .1 .8 .1 .8 .8\n1 .2 .2 .4 .2 .4 .4"


def corpus(root, task="detect", nested=True):
    images, labels = root / "images", root / "labels"
    text = DETECT if task == "detect" else SEGMENT
    for i in range(12):
        sub = f"s{i % 3}" if nested else ""
        (images / sub).mkdir(parents=True, exist_ok=True)
        (labels / sub).mkdir(parents=True, exist_ok=True)
        ext = ("png", "jpg", "JPG")[i % 3]
        Image.new("RGB", (48 + i, 32 + i), (i * 20, 31, 74)).save(
            images / sub / f"{i}.{ext}", format="PNG" if ext == "png" else "JPEG"
        )
        body = text + "\n" + text.splitlines()[0] if i == 1 else "" if i == 5 else text  # duplicate, empty
        if i != 6:  # missing label
            (labels / sub / f"{i}.txt").write_text(body)
    first = images / ("s0" if nested else "")
    (first / "bad.jpg").write_bytes(b"not an image")
    (labels / ("s0" if nested else "") / "bad.txt").write_text(text)
    # Entries the reference glob and extension filter ignore.
    (images / ".hidden.jpg").write_bytes(b"x")
    (images / ".hidden_dir").mkdir()
    Image.new("RGB", (40, 40)).save(images / ".hidden_dir" / "x.jpg")
    (images / "notes.txt").write_text("not an image")
    (images / "README").write_text("no extension")
    return images


def kwargs(images, task="detect", **changes):
    values = {
        "img_path": str(images) if not isinstance(images, list) else images,
        "imgsz": 64,
        "batch_size": 4,
        "augment": False,
        "hyp": copy.deepcopy(DEFAULT_CFG),
        "data": {"names": {0: "a", 1: "b"}},
        "task": task,
    }
    values.update(changes)
    return values


def equal(a, b):
    assert type(a) is type(b), (type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    elif isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and torch.equal(a, b)
    else:
        assert a == b


def reference(args):
    # Never let a stale reference .cache stand in for the current inputs.
    root = args["img_path"] if isinstance(args["img_path"], str) else args["img_path"][0]
    for dirpath, _, files in os.walk(os.path.dirname(os.path.abspath(root))):
        for name in files:
            if name.endswith(".cache"):
                os.unlink(os.path.join(dirpath, name))
    return dataset.YOLODataset(**args)


def batch(ds, workers=0):
    loader = DataLoader(ds, batch_size=4, num_workers=workers, collate_fn=dataset.YOLODataset.collate_fn)
    return list(loader)


def same(ref, fast):
    assert isinstance(fast.labels, (_fast.LazyLabels, list))
    equal(list(fast.labels), ref.labels)
    equal([fast.labels[i] for i in range(len(fast.labels))], ref.labels)
    equal(fast.labels[1:3], ref.labels[1:3])
    equal(fast.labels[-1], ref.labels[-1])
    assert fast.im_files == ref.im_files and fast.label_files == ref.label_files
    assert fast.ni == ref.ni and len(fast) == len(ref)
    assert [fast.npy_files[i] for i in range(len(fast.npy_files))] == ref.npy_files
    assert fast.npy_files[1:3] == ref.npy_files[1:3]
    equal(batch(fast), batch(ref))


@pytest.mark.parametrize("task", ["detect", "segment"])
@pytest.mark.parametrize("nested", [True, False])
def test_fast_matches_reference_on_miss_and_hit(tmp_path, task, nested):
    args = kwargs(corpus(tmp_path, task, nested), task)
    first = FastYOLODataset(**args, annotation_cache="fast")
    assert not first.annotation_cache_hit and first.annotation_cache_write_ok, first.annotation_cache_write_error
    second = FastYOLODataset(**args, annotation_cache="fast")
    assert second.annotation_cache_hit, second.annotation_cache_miss_reason
    assert isinstance(second.labels, _fast.LazyLabels)
    ref = reference(args)
    same(ref, first)
    same(ref, second)
    # Every access is an independent copy; augmentation cannot alter the cache.
    second.labels[0]["bboxes"][:] = 99
    equal(second.labels[0], ref.labels[0])


def rewrite(path, text, mtime_shift=0.0):
    info = path.stat()
    path.write_text(text)
    if mtime_shift:
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + int(mtime_shift * 1e9)))


@pytest.mark.parametrize("change", ["resize_label", "same_size_label", "delete_label", "add_image", "touch_image"])
def test_fast_cache_invalidation(tmp_path, change):
    images = corpus(tmp_path)
    args = kwargs(images)
    FastYOLODataset(**args, annotation_cache="fast")
    label = tmp_path / "labels" / "s0" / "0.txt"
    if change == "resize_label":
        rewrite(label, "1 .5 .5 .2 .2")
    elif change == "same_size_label":
        # Equal size: the reference's summed-size hash cannot detect this edit.
        rewrite(label, DETECT.replace("0 .5", "1 .5"), mtime_shift=5)
    elif change == "delete_label":
        label.unlink()
    elif change == "add_image":
        Image.new("RGB", (50, 50)).save(images / "s1" / "new.png")
    else:
        image = images / "s0" / "0.png"
        info = image.stat()
        os.utime(image, ns=(info.st_atime_ns, info.st_mtime_ns + 5_000_000_000))
    changed = FastYOLODataset(**args, annotation_cache="fast")
    assert not changed.annotation_cache_hit and "changed" in changed.annotation_cache_miss_reason
    same(reference(args), changed)
    assert FastYOLODataset(**args, annotation_cache="fast").annotation_cache_hit


@pytest.mark.parametrize("damage", ["magic", "table", "metadata", "array", "truncate", "empty"])
def test_damaged_fast_cache_is_a_miss(tmp_path, damage):
    args = kwargs(corpus(tmp_path, "segment"), "segment")
    first = FastYOLODataset(**args, annotation_cache="fast")
    path = first.annotation_cache_path
    data = bytearray(Path(path).read_bytes())
    if damage == "magic":
        data[0] ^= 1
    elif damage == "table":
        data[20] ^= 1
    elif damage == "metadata":
        data[_fast._aligned(16 + 48 * 9) + 3] ^= 1
    elif damage == "array":
        data[-5] ^= 1
    elif damage == "truncate":
        data = data[: len(data) // 2]
    else:
        data = bytearray()
    Path(path).write_bytes(bytes(data))
    rebuilt = FastYOLODataset(**args, annotation_cache="fast")
    assert not rebuilt.annotation_cache_hit and rebuilt.annotation_cache_write_ok
    same(reference(args), rebuilt)


@pytest.mark.parametrize(
    "changes",
    [
        {"fraction": 0.5},
        {"fraction": 3},
        {"single_cls": True},
        {"classes": [1]},
        {"rect": True},
        {"cache": "ram"},
    ],
    ids=["fraction", "count", "single_cls", "classes", "rect", "ram"],
)
def test_fast_options_match_reference(tmp_path, changes):
    args = kwargs(corpus(tmp_path, "segment"), "segment", **changes)
    for attempt in range(2):
        fast = FastYOLODataset(**args, annotation_cache="fast")
        assert fast.annotation_cache_hit == bool(attempt)
        ref = reference(args)
        same(ref, fast)
        if changes.get("rect"):
            assert np.array_equal(fast.batch_shapes, ref.batch_shapes) and np.array_equal(fast.batch, ref.batch)


def test_fast_reference_discovery_forms(tmp_path, monkeypatch):
    images = corpus(tmp_path)
    listing = tmp_path / "list.txt"
    files = sorted(str(p) for p in images.rglob("*") if p.suffix[1:].lower() in IMG_FORMATS and "/." not in str(p))
    listing.write_text("\n".join(files))
    for img_path in ([str(images / "s0"), str(images / "s1")], str(listing)):
        args = kwargs(img_path)
        FastYOLODataset(**args, annotation_cache="fast", cache_dir=tmp_path / "caches")
        fast = FastYOLODataset(**args, annotation_cache="fast", cache_dir=tmp_path / "caches")
        assert fast.annotation_cache_hit
        same(dataset.YOLODataset(**args), fast)
    monkeypatch.chdir(tmp_path)
    args = kwargs("images")
    FastYOLODataset(**args, annotation_cache="fast")
    fast = FastYOLODataset(**args, annotation_cache="fast")
    assert fast.annotation_cache_hit and fast.im_files[0].startswith("images/")
    same(reference(args), fast)


def test_symlinked_directory_uses_reference_discovery(tmp_path):
    images = corpus(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    Image.new("RGB", (44, 44)).save(other / "linked.png")
    (images / "linked").symlink_to(other, target_is_directory=True)
    assert _native.discover_dataset(str(images), sorted(IMG_FORMATS), 4) is None
    args = kwargs(images)
    FastYOLODataset(**args, annotation_cache="fast")
    fast = FastYOLODataset(**args, annotation_cache="fast")
    assert fast.annotation_cache_hit
    same(reference(args), fast)


def test_mixed_boxes_and_segments_drop_segments_like_reference(tmp_path):
    images = corpus(tmp_path, "detect")
    (tmp_path / "labels" / "s0" / "0.txt").write_text(SEGMENT)
    args = kwargs(images)
    for _ in range(2):
        fast = FastYOLODataset(**args, annotation_cache="fast")
        same(reference(args), fast)
        assert all(label["segments"] == [] for label in fast.labels)


def test_no_labels_errors_match_reference(tmp_path):
    images = corpus(tmp_path)
    for path in (tmp_path / "labels").rglob("*.txt"):
        path.unlink()
    args = kwargs(images, augment=True)
    with pytest.raises(ValueError, match="No labels found") as expected:
        dataset.YOLODataset(**args)
    with pytest.raises(ValueError, match="No labels found") as actual:
        FastYOLODataset(**args, annotation_cache="fast")
    assert str(actual.value) == str(expected.value)


def test_lazy_labels_pickle_and_loader_workers(tmp_path):
    args = kwargs(corpus(tmp_path, "segment"), "segment")
    FastYOLODataset(**args, annotation_cache="fast")
    fast = FastYOLODataset(**args, annotation_cache="fast")
    ref = reference(args)
    restored = pickle.loads(pickle.dumps(fast.labels))
    equal(list(restored), ref.labels)
    equal(batch(fast, workers=2), batch(ref))


def test_discovery_matches_reference_glob(tmp_path):
    images = corpus(tmp_path)
    (images / "dir.jpg").mkdir()  # the reference glob reports matching directories
    index = _native.discover_dataset(str(images), sorted(IMG_FORMATS), 4)
    probe = object.__new__(dataset.YOLODataset)
    probe.fraction, probe.prefix = 1.0, ""
    ref = dataset.YOLODataset.get_img_files(probe, str(images))
    assert index.image_paths() == ref
    assert _native.discover_dataset(str(images) + "/", sorted(IMG_FORMATS), 4) is None
    assert _native.discover_dataset(str(tmp_path / "im[a]ges"), sorted(IMG_FORMATS), 4) is None


def test_index_metadata_matches_os_stat(tmp_path):
    images = corpus(tmp_path)
    index = _native.discover_dataset(str(images), sorted(IMG_FORMATS), 4)
    image_meta, label_meta = index.metadata()
    for path, meta in zip(index.image_paths(), image_meta):
        info = os.stat(path)
        assert meta == (
            (0 if os.path.isfile(path) else 2),
            info.st_size if os.path.isfile(path) else 0,
            info.st_mtime_ns,
        )
    for path, meta in zip(index.label_paths(), label_meta):
        expected = (0, os.stat(path).st_size, os.stat(path).st_mtime_ns) if os.path.exists(path) else (1, 0, 0)
        assert meta == expected
    before = index.digest()
    time.sleep(0.01)
    assert _native.discover_dataset(str(images), sorted(IMG_FORMATS), 1).digest() == before


def test_chunked_sha256_is_order_and_length_sensitive():
    data = np.arange(3 << 20, dtype=np.uint8)
    one = _native.chunked_sha256(memoryview(data), 1)
    assert one == _native.chunked_sha256(memoryview(data), 8)
    assert one != _native.chunked_sha256(memoryview(data[:-1]), 8)
    small = _native.chunked_sha256(b"abc", 1)
    assert len(small) == 32 and small != _native.chunked_sha256(b"abd", 1)
