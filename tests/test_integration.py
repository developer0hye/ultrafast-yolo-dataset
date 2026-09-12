import copy
import gc
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from ultrafast_yolo_dataset.ultralytics import FastYOLODataset, UnsupportedProfile, build_yolo_dataset
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data import build, dataset, utils


def corpus(root, task="detect"):
    images, labels = root / "images", root / "labels"
    images.mkdir()
    labels.mkdir()
    for i in range(8):
        Image.new("RGB", (48 + i, 32 + i), (i * 25, 31, 74)).save(images / f"{i}.png")
        text = "0 .5 .5 .4 .4\n1 .3 .3 .2 .2" if task == "detect" else "0 .1 .1 .8 .1 .8 .8\n1 .2 .2 .4 .2 .4 .4"
        # Include duplicates, missing and empty labels in actual framework data.
        if i == 1:
            text += "\n" + text.splitlines()[0]
        if i != 6:
            (labels / f"{i}.txt").write_text("" if i == 7 else text)
    return images


def kwargs(images, task="detect", **changes):
    return dict(
        img_path=str(images),
        imgsz=64,
        batch_size=2,
        augment=False,
        hyp=copy.deepcopy(DEFAULT_CFG),
        data={"names": {0: "a", 1: "b"}},
        task=task,
        **changes,
    )


def equal(a, b):
    assert type(a) is type(b)
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


@pytest.mark.parametrize("task", ["detect", "segment"])
@pytest.mark.parametrize("workers", [0, 2])
def test_actual_dataset_and_collated_batches(tmp_path, task, workers):
    images = corpus(tmp_path, task)
    candidate = FastYOLODataset(**kwargs(images, task))
    assert not tmp_path.joinpath("labels.cache").exists()
    reference = dataset.YOLODataset(**kwargs(images, task))
    equal(candidate.labels, reference.labels)
    assert candidate.scan_evidence["summary"] == {"found": 7, "missing": 1, "empty": 1, "corrupt": 0, "total": 8}
    assert candidate.scan_evidence["fallback_count"] == 0
    assert candidate.im_files == reference.im_files
    opts = {"batch_size": 2, "num_workers": workers, "collate_fn": dataset.YOLODataset.collate_fn}
    if workers:
        opts["multiprocessing_context"] = "spawn"
    for a, b in zip(DataLoader(candidate, **opts), DataLoader(reference, **opts)):
        equal(a, b)


def test_native_legacy_reference_and_reload(tmp_path):
    images = corpus(tmp_path, "segment")
    args = kwargs(images, "segment")
    first = FastYOLODataset(**args, annotation_cache="ultralytics", trust_legacy_cache=True)
    assert first.annotation_cache_write_ok is True and not first.annotation_cache_hit
    payload = utils.load_dataset_cache_file(tmp_path / "labels.cache")
    assert payload["results"] == (7, 1, 1, 0, 8)
    assert payload["ultrafast_profile"]["task"] == "segment"
    second = FastYOLODataset(**args, annotation_cache="ultralytics", trust_legacy_cache=True)
    assert second.annotation_cache_hit and second.scan_evidence is None
    reference = dataset.YOLODataset(**args)
    equal(first.labels, second.labels)
    equal(first.labels, reference.labels)
    # Per-dataset materialization remains independent after legacy cache hits.
    second.labels[0]["bboxes"][:] = 99
    third = FastYOLODataset(**args, annotation_cache="ultralytics", trust_legacy_cache=True)
    equal(first.labels, third.labels)
    assert not list(tmp_path.glob(".labels.cache.*"))


@pytest.mark.parametrize("damage", [b"", b"not a numpy cache", b"\x93NUMPY\x01\x00"])
def test_truncated_legacy_rebuild_and_gc_state(tmp_path, damage):
    images = corpus(tmp_path)
    (tmp_path / "labels.cache").write_bytes(damage)
    was_enabled = gc.isenabled()
    candidate = FastYOLODataset(**kwargs(images), annotation_cache="ultralytics", trust_legacy_cache=True)
    assert not candidate.annotation_cache_hit and candidate.annotation_cache_write_ok
    assert candidate.annotation_cache_miss_reason
    assert gc.isenabled() == was_enabled


def test_legacy_configuration_change_and_unknown_profile_rebuild(tmp_path):
    images = corpus(tmp_path)
    args = kwargs(images)
    # Reference caches do not identify the scan configuration. Do not guess it.
    dataset.YOLODataset(**args)
    first = FastYOLODataset(**args, annotation_cache="ultralytics", trust_legacy_cache=True)
    assert not first.annotation_cache_hit and "configuration" in first.annotation_cache_miss_reason
    changed = FastYOLODataset(**args, single_cls=True, annotation_cache="ultralytics", trust_legacy_cache=True)
    assert not changed.annotation_cache_hit and "configuration" in changed.annotation_cache_miss_reason
    assert all((x["cls"] == 0).all() for x in changed.labels)


def test_legacy_malformed_logical_schema_rebuilds(tmp_path):
    images = corpus(tmp_path)
    args = kwargs(images)
    cache = tmp_path / "labels.cache"
    FastYOLODataset(**args, annotation_cache="ultralytics", trust_legacy_cache=True)
    valid = utils.load_dataset_cache_file(cache)
    for case in ("counter_type", "shape", "array", "order", "count"):
        altered = copy.deepcopy(valid)
        if case == "counter_type":
            altered["results"] = (7, 1, 1, 0, "8")
        elif case == "shape":
            altered["labels"][0].pop("shape")
        elif case == "array":
            altered["labels"][0]["bboxes"] = np.zeros((1, 4), dtype=np.float64)
        elif case == "order":
            altered["labels"] = altered["labels"][::-1]
        else:
            altered["results"] = (7, 1, 1, 1, 8)
        with cache.open("wb") as stream:
            np.save(stream, altered)
        rebuilt = FastYOLODataset(**args, annotation_cache="ultralytics", trust_legacy_cache=True)
        assert not rebuilt.annotation_cache_hit and rebuilt.annotation_cache_write_ok, case


@pytest.mark.parametrize("enabled", [True, False])
def test_concurrent_legacy_reads_preserve_process_gc(tmp_path, monkeypatch, enabled):
    import ultrafast_yolo_dataset.ultralytics as integration

    images = corpus(tmp_path)
    candidate = FastYOLODataset(**kwargs(images), annotation_cache="ultralytics", trust_legacy_cache=True)
    original = utils.load_dataset_cache_file
    active, maximum = 0, 0
    counter_lock = threading.Lock()

    def observed(path):
        nonlocal active, maximum
        with counter_lock:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(0.01)  # allow competing callers to overlap if unserialized
            return original(path)
        finally:
            with counter_lock:
                active -= 1

    monkeypatch.setattr(utils, "load_dataset_cache_file", observed)
    prior = gc.isenabled()
    gc.enable() if enabled else gc.disable()
    try:

        def load(_):
            return integration._read_trusted_legacy(
                tmp_path / "labels.cache", candidate.get_cache_hash(), candidate._cache_profile(), candidate.im_files
            )

        with ThreadPoolExecutor(4) as pool:
            values = list(pool.map(load, range(8)))
        assert maximum == 1 and len(values) == 8
        assert gc.isenabled() is enabled
    finally:
        gc.enable() if prior else gc.disable()


def test_legacy_trust_is_explicit_and_default_does_not_unpickle(tmp_path, monkeypatch):
    images = corpus(tmp_path)
    cache = tmp_path / "labels.cache"
    cache.write_bytes(b"untrusted")
    with pytest.raises(ValueError, match="trust_legacy_cache"):
        FastYOLODataset(**kwargs(images), annotation_cache="ultralytics")
    # No legacy file reads or writes occur in the default scan-only mode.
    candidate = FastYOLODataset(**kwargs(images))
    assert len(candidate) == 8 and cache.read_bytes() == b"untrusted"


def test_cache_write_failure_preserves_dataset_and_existing_file(tmp_path, monkeypatch):
    import ultrafast_yolo_dataset.ultralytics as integration

    images = corpus(tmp_path)
    cache = tmp_path / "labels.cache"
    cache.write_bytes(b"old invalid cache")

    def failed_replace(*args):
        raise PermissionError("injected replacement denial")

    monkeypatch.setattr(integration.os, "replace", failed_replace)
    candidate = FastYOLODataset(**kwargs(images), annotation_cache="ultralytics", trust_legacy_cache=True)
    assert len(candidate) == 8 and candidate.annotation_cache_write_ok is False
    assert cache.read_bytes() == b"old invalid cache"
    assert not list(tmp_path.glob(".labels.cache.*"))


def test_unknown_subclass_task_and_runtime_hook_rejected(tmp_path, monkeypatch):
    class Custom(FastYOLODataset):
        def verify_args(self):
            raise AssertionError("must not be silently ignored")

    with pytest.raises(UnsupportedProfile, match="custom hooks"):
        Custom(img_path="does-not-exist")
    with pytest.raises(UnsupportedProfile, match="custom hooks"):
        FastYOLODataset(task="pose", img_path="does-not-exist")
    monkeypatch.setattr(dataset.YOLODataset, "verify_args", lambda self: None)
    with pytest.raises(UnsupportedProfile, match="verify_args"):
        FastYOLODataset(img_path="does-not-exist")


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_explicit_trainer_factory(tmp_path, task):
    images = corpus(tmp_path, task)
    cfg = copy.deepcopy(DEFAULT_CFG)
    cfg.task, cfg.imgsz = task, 64
    data = {"names": {0: "a", 1: "b"}}
    reference = build.build_yolo_dataset(cfg, str(images), 2, data, mode="val", fraction=0.75)
    candidate = build_yolo_dataset(cfg, str(images), 2, data, mode="val", fraction=0.75)
    equal(reference.labels, candidate.labels)
    assert reference.im_files == candidate.im_files
    equal(reference[0], candidate[0])
    # The adapter does not replace the module's original builder/class.
    assert build.YOLODataset is dataset.YOLODataset
    pickle.loads(pickle.dumps(candidate))


@pytest.mark.parametrize("task", ["detect", "segment"])
@pytest.mark.parametrize("workers", [0, 2])
def test_native_cache_actual_dataset_cold_warm_and_first_batches(tmp_path, monkeypatch, task, workers):
    images = corpus(tmp_path, task)
    legacy = tmp_path / "labels.cache"
    legacy.write_bytes(b"untrusted legacy file must not be opened")
    native_options = {"annotation_cache": "native", "cache_dir": tmp_path / "separate-cache"}

    def forbidden_pickle(*args, **kwargs):
        raise AssertionError("native cache path must not call np.load")

    with monkeypatch.context() as patch:
        patch.setattr(np, "load", forbidden_pickle)
        first = FastYOLODataset(**kwargs(images, task), **native_options)
        cached = FastYOLODataset(**kwargs(images, task), **native_options)
    assert not first.annotation_cache_hit and first.annotation_cache_write_ok
    assert cached.annotation_cache_hit and cached.scan_evidence is None
    assert cached.annotation_cache_path.endswith(".uydcache")
    assert legacy.read_bytes() == b"untrusted legacy file must not be opened"
    assert len(list((tmp_path / "separate-cache").glob("*.uydcache"))) == 1
    legacy.unlink()
    reference = dataset.YOLODataset(**kwargs(images, task))
    equal(first.labels, reference.labels)
    equal(cached.labels, reference.labels)
    options = {"batch_size": 2, "num_workers": workers, "collate_fn": dataset.YOLODataset.collate_fn}
    if workers:
        options["multiprocessing_context"] = "spawn"
    for a, b in zip(DataLoader(cached, **options), DataLoader(reference, **options)):
        equal(a, b)
