import functools
import gc
import hashlib
import json
import multiprocessing as mp
import os
import struct
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_scan import equal
from ultrafast_yolo_dataset import _native, load_cache, scan, scan_cached
from ultrafast_yolo_dataset._cache import CacheMiss
from ultrafast_yolo_dataset._snapshots import InputChangedError
from ultralytics.data.utils import verify_image_label


def corpus(root, texts=None):
    texts = (
        texts if texts is not None else ["0 .5 .5 .2 .2", "0 0 0 1 0 1 1", "0\u00a0.5 .5 .2 .2", None, "", "invalid"]
    )
    images, labels = [], []
    for index, text in enumerate(texts):
        image, label = root / f"图像 {index}.png", root / f"标注 {index}.txt"
        Image.new("RGB", (32 + index, 24 + index)).save(image)
        if text is not None:
            label.write_text(text)
        images.append(str(image))
        labels.append(str(label))
    return images, labels


def assert_equal_results(a, b):
    assert a.image_paths == b.image_paths and a.label_paths == b.label_paths
    assert a.summary == b.summary and a.diagnostics == b.diagnostics and a.fallback_count == b.fallback_count
    for name in (
        "source_indices",
        "image_shapes",
        "object_offsets",
        "_labels",
        "segment_points",
        "segment_offsets",
        "segment_object_indices",
        "repaired_source_indices",
    ):
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), name


@pytest.mark.parametrize("policy", ["content", "metadata"])
def test_snapshot_parity_roundtrip_owned_storage(tmp_path, policy):
    images, labels = corpus(tmp_path)
    expected = scan(images, labels, num_classes=2, prefix="val: ")
    captured = scan(images, labels, num_classes=2, prefix="val: ", fingerprint=policy)
    assert_equal_results(captured, expected)
    assert captured._provenance is not None
    path = tmp_path / "data.uydcache"
    assert captured.save_cache(path).saved
    assert captured.save_cache(path).reused
    loaded = load_cache(path, images, labels, num_classes=2, prefix="val: ", fingerprint=policy)
    assert not isinstance(loaded, CacheMiss)
    assert_equal_results(loaded, expected)
    assert_equal_results(scan_cached(path, images, labels, num_classes=2, prefix="val: ", fingerprint=policy), expected)
    with pytest.raises(ValueError):
        loaded.boxes.flags.writeable = True
    mutable = loaded.to_ultralytics_labels()
    original = loaded.boxes.copy()
    mutable[0]["bboxes"][:] = 999
    np.testing.assert_array_equal(loaded.boxes, original)
    retained = loaded.boxes
    del loaded
    gc.collect()
    np.testing.assert_array_equal(retained, original)


def test_empty_cache(tmp_path):
    result = scan_cached(tmp_path / "empty.uydcache", [], [], num_classes=1)
    assert result.num_valid == 0 and result.summary["total"] == 0
    assert scan_cached(tmp_path / "empty.uydcache", [], [], num_classes=1).cache_hit


def test_relative_request_cannot_rebind_while_waiting_for_lock(tmp_path, monkeypatch):
    import ultrafast_yolo_dataset._cache as cache_module

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    images, labels = corpus(first, ["0 .5 .5 .2 .2"])
    corpus(second, ["0 .5 .5 .3 .3"])
    monkeypatch.chdir(first)

    @contextmanager
    def change_directory(*_):
        monkeypatch.chdir(second)
        yield

    monkeypatch.setattr(cache_module, "_locked", change_directory)
    with pytest.raises(InputChangedError, match="working directory changed"):
        scan_cached(first / "relative.uydcache", [Path(images[0]).name], [Path(labels[0]).name], num_classes=1)
    assert not (first / "relative.uydcache").exists()


@pytest.mark.parametrize("damage", ["short", "kind", "nonfile_digest", "missing_metadata", "encoding", "schema"])
def test_compact_proof_corruption_with_valid_section_checksums(tmp_path, damage):
    images, labels = corpus(tmp_path, [None])
    path = tmp_path / "compact.uydcache"
    scan_cached(path, images, labels, num_classes=1)
    sections = _native.read_cache_sections(str(path), 1024**2)
    assert len(sections[-1]) == len(sections[-2]) == 57
    metadata = json.loads(sections[0])
    assert "proof" not in metadata  # No per-file Python dictionaries on a hit.
    if damage in ("encoding", "schema"):
        metadata["proof_encoding" if damage == "encoding" else "schema"] = "unsupported"
        sections[0] = json.dumps(metadata).encode()
    else:
        data = bytearray(sections[-1])
        if damage == "short":
            data.pop()
        else:
            data[{"kind": 0, "nonfile_digest": 25, "missing_metadata": 1}[damage]] = 7
        sections[-1] = bytes(data)
    path.unlink()
    _native.write_cache_sections(str(path), sections, 1024**2)
    assert isinstance(load_cache(path, images, labels, num_classes=1), CacheMiss)


@pytest.mark.parametrize("damage", ["missing", "extra", "bool_size", "negative_size", "digest", "missing_metadata"])
def test_pack_fingerprint_rejects_invalid_schema(damage):
    entry = {"kind": "missing", "size": 0, "modified_ns": 0, "sha256": None}
    if damage == "missing":
        del entry["sha256"]
    elif damage == "extra":
        entry["extra"] = 1
    elif damage == "bool_size":
        entry["size"] = True
    elif damage == "negative_size":
        entry["size"] = -1
    elif damage == "digest":
        entry.update(kind="file", sha256="g" * 64)
    else:
        entry["modified_ns"] = 1
    with pytest.raises(ValueError):
        _native.pack_fingerprint_table(json.dumps([entry]), 1)


def test_loaded_compact_proof_can_be_saved_and_reused(tmp_path):
    images, labels = corpus(tmp_path)
    path = tmp_path / "first.uydcache"
    scan_cached(path, images, labels, num_classes=2)
    loaded = load_cache(path, images, labels, num_classes=2)
    assert all(isinstance(value, bytes) and len(value) == len(images) * 57 for value in loaded._provenance.values())
    assert loaded.save_cache(path).reused
    copied = tmp_path / "second.uydcache"
    assert loaded.save_cache(copied).saved
    assert_equal_results(load_cache(copied, images, labels, num_classes=2), loaded)


def test_corrupt_and_missing_images_use_reference_diagnostics(tmp_path):
    images, labels = corpus(tmp_path, ["0 .5 .5 .2 .2"] * 4)
    Path(images[0]).write_bytes(b"broken image")
    Path(images[1]).unlink()
    Path(images[2]).unlink()
    Path(images[2]).mkdir()
    expected = [verify_image_label((i, l, "val: ", False, 1, 0, 0, False)) for i, l in zip(images, labels)]
    result = scan_cached(tmp_path / "errors.uydcache", images, labels, num_classes=1, prefix="val: ")
    assert [d.message for d in result.diagnostics] == [r[-1] for r in expected if r[-1]]
    equal(result.to_ultralytics_labels()[0], expected[-1])
    assert scan_cached(tmp_path / "errors.uydcache", images, labels, num_classes=1, prefix="val: ").cache_hit


@pytest.mark.parametrize("fallback", [False, True])
def test_exact_label_bytes_used_after_path_changes(tmp_path, monkeypatch, fallback):
    import ultrafast_yolo_dataset._scan as scan_module

    text = "0\u00a0.5 .5 .2 .2" if fallback else "0 .5 .5 .2 .2"
    images, labels = corpus(tmp_path, [text])
    original = _native.snapshot_labels
    reference = verify_image_label((images[0], labels[0], "", False, 1, 0, 0, False))

    def replace_after_capture(*args):
        result = original(*args)
        before = os.stat(labels[0])
        Path(labels[0]).write_text(text.replace(".2", ".3"))
        os.utime(labels[0], ns=(before.st_atime_ns, before.st_mtime_ns))
        return result

    monkeypatch.setattr(scan_module._native, "snapshot_labels", replace_after_capture)
    result = scan(images, labels, num_classes=1, fingerprint="content")
    equal(result.to_ultralytics_labels()[0], reference)
    assert result._provenance["labels"][0]["sha256"] == hashlib.sha256(text.encode()).hexdigest()
    with pytest.raises(InputChangedError):
        result.save_cache(tmp_path / "stale.uydcache")
    assert not list(tmp_path.glob("*.uydcache")) and not list(tmp_path.glob(".*.tmp"))


def test_pillow_uses_captured_image_not_replaced_path(tmp_path, monkeypatch):
    import ultrafast_yolo_dataset._scan as scan_module

    images, labels = corpus(tmp_path, ["0 .5 .5 .2 .2"])
    original = scan_module._check_image

    @functools.wraps(original)
    def replace_before_pillow(path, *args, **kwargs):
        Image.new("RGB", (80, 60)).save(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(scan_module, "_check_image", replace_before_pillow)
    result = scan(images, labels, num_classes=1, fingerprint="content")
    assert result.image_shapes.tolist() == [[24, 32]]
    with pytest.raises(InputChangedError):
        result.save_cache(tmp_path / "stale.uydcache")


def test_post_serialization_revalidation_preserves_old_cache(tmp_path, monkeypatch):
    images, labels = corpus(tmp_path, ["0 .5 .5 .2 .2"])
    path = tmp_path / "dataset.uydcache"
    scan_cached(path, images, labels, num_classes=1)
    old = path.read_bytes()
    Path(labels[0]).write_text("0 .5 .5 .3 .3")
    result = scan(images, labels, num_classes=1, fingerprint="content")
    original = _native.write_cache_sections

    def change_during_write(*args):
        original(*args)
        Path(labels[0]).write_text("0 .5 .5 .4 .4")

    monkeypatch.setattr(_native, "write_cache_sections", change_during_write)
    with pytest.raises(InputChangedError):
        result.save_cache(path)
    assert path.read_bytes() == old and not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("policy", ["content", "metadata"])
def test_same_size_same_mtime_content_change_contract(tmp_path, policy):
    images, labels = corpus(tmp_path, ["0 .5 .5 .2 .2"])
    path = tmp_path / "data.uydcache"
    original = scan_cached(path, images, labels, num_classes=1, fingerprint=policy)
    before = os.stat(labels[0])
    Path(labels[0]).write_text("0 .5 .5 .3 .3")
    os.utime(labels[0], ns=(before.st_atime_ns, before.st_mtime_ns))
    loaded = load_cache(path, images, labels, num_classes=1, fingerprint=policy)
    if policy == "content":
        assert isinstance(loaded, CacheMiss)
    else:
        assert_equal_results(loaded, original)  # documented weaker opt-in policy
    # Saving a captured result always requires exact bytes, even in metadata mode.
    with pytest.raises(InputChangedError):
        original.save_cache(path)


@pytest.mark.parametrize("change", ["task", "single_cls", "classes", "prefix", "repair", "paths", "policy"])
def test_configuration_and_path_order_invalidation(tmp_path, change):
    images, labels = corpus(tmp_path)
    path = tmp_path / "data.uydcache"
    scan_cached(path, images, labels, num_classes=2)
    options = {"num_classes": 2}
    if change == "task":
        options["task"] = "segment"
    elif change == "single_cls":
        options["single_cls"] = True
    elif change == "classes":
        options["num_classes"] = 3
    elif change == "prefix":
        options["prefix"] = "val: "
    elif change == "repair":
        options["repair_jpeg"] = "reference"
    elif change == "policy":
        options["fingerprint"] = "metadata"
    else:
        images, labels = images[::-1], labels[::-1]
    assert isinstance(load_cache(path, images, labels, **options), CacheMiss)


@pytest.mark.parametrize(
    "damage", ["empty", "magic", "version", "header", "last", "trailing", "checksum", "huge_length"]
)
def test_container_corruption_is_cache_miss(tmp_path, damage):
    images, labels = corpus(tmp_path)
    path = tmp_path / "data.uydcache"
    scan_cached(path, images, labels, num_classes=2)
    data = bytearray(path.read_bytes())
    if damage == "empty":
        data.clear()
    elif damage == "magic":
        data[0] ^= 1
    elif damage == "version":
        data[8] ^= 127
    elif damage == "header":
        data = data[:25]
    elif damage == "last":
        data = data[:-1]
    elif damage == "trailing":
        data += b"x"
    elif damage == "checksum":
        data[-1] ^= 1
    else:
        data[16:24] = struct.pack("<Q", 2**64 - 1)
    path.write_bytes(data)
    assert isinstance(load_cache(path, images, labels, num_classes=2), CacheMiss)


@pytest.mark.parametrize("damage", ["dtype", "shape", "profile", "source", "offset", "segment", "counter"])
def test_checksummed_but_invalid_logical_schema_is_rejected(tmp_path, damage):
    images, labels = corpus(tmp_path)
    good, bad = tmp_path / "good.uydcache", tmp_path / "bad.uydcache"
    scan_cached(good, images, labels, num_classes=2)
    sections = _native.read_cache_sections(str(good), 1024**2)
    metadata = json.loads(sections[0])
    if damage == "dtype":
        metadata["arrays"]["_labels"]["dtype"] = "object"
    elif damage == "shape":
        metadata["arrays"]["_labels"]["shape"] = [2**63 - 1, 5]
    elif damage == "profile":
        metadata["profile"] = "unknown"
    elif damage == "counter":
        metadata["summary"]["corrupt"] = 0
    else:
        index = {"source": 1, "offset": 3, "segment": 7}[damage]
        array = np.frombuffer(sections[index], dtype="<i8").copy()
        array[0] = -1
        sections[index] = array.tobytes()
    sections[0] = json.dumps(metadata).encode()
    _native.write_cache_sections(str(bad), sections, 1024**2)
    assert isinstance(load_cache(bad, images, labels, num_classes=2), CacheMiss)


def test_explicit_jpeg_repair_fingerprints_encoder_output(tmp_path):
    images, labels = corpus(tmp_path, ["0 .5 .5 .2 .2\n0 .5 .5 .2 .2", "invalid"])
    for i in range(2):
        image = tmp_path / f"{i}.jpg"
        Image.new("RGB", (32, 24), (123, 45, 67)).save(image)
        image.write_bytes(image.read_bytes() + b"tail")
        images[i] = str(image)
    references = []
    for i, path in enumerate(images):
        reference = tmp_path / f"ref-{i}.jpg"
        reference.write_bytes(Path(path).read_bytes())
        verify_image_label((str(reference), labels[i], "", False, 1, 0, 0, False))
        references.append(reference.read_bytes())
    path = tmp_path / "repaired.uydcache"
    result = scan_cached(path, images, labels, num_classes=1, repair_jpeg="reference")
    assert result.repaired_source_indices.tolist() == [0, 1]
    for i, image in enumerate(images):
        assert Path(image).read_bytes() == references[i]
        assert result._provenance["images"][i]["sha256"] == hashlib.sha256(references[i]).hexdigest()
    loaded = scan_cached(path, images, labels, num_classes=1, repair_jpeg="reference")
    assert loaded.cache_hit
    assert_equal_results(loaded, result)


def test_write_failure_and_lock_timeout_keep_valid_scan(tmp_path, monkeypatch):
    import ultrafast_yolo_dataset._cache as cache_module

    images, labels = corpus(tmp_path)
    path = tmp_path / "data.uydcache"
    lock = _native.CacheLock(str(path) + ".lock")
    assert lock.try_acquire()
    result = scan_cached(path, images, labels, num_classes=2, lock_timeout=0.01)
    assert result.num_valid == 5 and "timeout" in result.cache_write_error
    lock.release()

    def denied(*args):
        raise PermissionError("injected publish denial")

    monkeypatch.setattr(cache_module.os, "replace", denied)
    result = scan_cached(path, images, labels, num_classes=2)
    assert result.num_valid == 5 and "denial" in result.cache_write_error
    assert not path.exists() and not list(tmp_path.glob(".*.tmp"))


def test_ordinary_scan_cannot_claim_snapshot_provenance(tmp_path):
    images, labels = corpus(tmp_path)
    result = scan(images, labels, num_classes=2)
    assert not result.save_cache(tmp_path / "data.uydcache").saved


def test_cache_destination_cannot_overwrite_an_input(tmp_path):
    image, label = tmp_path / "image.uydcache", tmp_path / "label.txt"
    Image.new("RGB", (32, 24)).save(image, "PNG")
    label.write_text("0 .5 .5 .2 .2")
    original = image.read_bytes()
    with pytest.raises(ValueError, match="overlap"):
        scan_cached(image, [image], [label], num_classes=1)
    assert image.read_bytes() == original


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX special-file check")
def test_nonregular_cache_rejected_without_reading_fifo(tmp_path):
    path = tmp_path / "pipe.uydcache"
    os.mkfifo(path)
    result = load_cache(path, [], [], num_classes=1)
    assert isinstance(result, CacheMiss) and "regular file" in result.reason


def test_relative_path_context_is_preserved(tmp_path, monkeypatch):
    corpus(tmp_path, ["0 .5 .5 .2 .2"])
    monkeypatch.chdir(tmp_path)
    images, labels = ["图像 0.png"], ["标注 0.txt"]
    result = scan(images, labels, num_classes=1, fingerprint="content")
    cache = tmp_path / "relative.uydcache"
    assert result.save_cache(cache).saved
    assert not isinstance(load_cache(cache, images, labels, num_classes=1), CacheMiss)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    with pytest.raises(InputChangedError, match="context"):
        result.save_cache(cache)
    assert isinstance(load_cache(cache, images, labels, num_classes=1), CacheMiss)


def test_capture_limit_does_not_lose_successful_jpeg_repair(tmp_path):
    image, label = tmp_path / "a.jpg", tmp_path / "a.txt"
    Image.new("RGB", (32, 24)).save(image)
    image.write_bytes(image.read_bytes() + b"tail")
    label.write_text("0 .5 .5 .2 .2")
    result = scan_cached(
        tmp_path / "small.uydcache", [image], [label], num_classes=1, repair_jpeg="reference", max_image_bytes=1
    )
    assert result.num_valid == 1 and result.repaired_source_indices.tolist() == [0]
    assert "restored and saved" in result.diagnostics[0].message
    assert result.cache_write_error and not (tmp_path / "small.uydcache").exists()


def test_fallback_snapshot_budget_preserves_reference_result(tmp_path):
    images, labels = corpus(tmp_path, ["0\u00a0.5 .5 .2 .2"] * 16)
    with pytest.raises(_native.SnapshotLimitError):
        _native.snapshot_labels(labels, 1, False, 4, 1024, 32)
    result = scan_cached(tmp_path / "small.uydcache", images, labels, num_classes=1, max_fallback_bytes=32)
    assert result.num_valid == 16 and result.fallback_count == 16 and result.cache_write_error
    assert_equal_results(result, scan(images, labels, num_classes=1))


def test_cancelled_write_cleans_temp_and_releases_lock(tmp_path, monkeypatch):
    images, labels = corpus(tmp_path)
    path = tmp_path / "cancel.uydcache"
    original = _native.write_cache_sections

    def cancelled(*args):
        original(*args)
        raise KeyboardInterrupt()

    with monkeypatch.context() as patch:
        patch.setattr(_native, "write_cache_sections", cancelled)
        with pytest.raises(KeyboardInterrupt):
            scan_cached(path, images, labels, num_classes=2)
    assert not path.exists() and not list(tmp_path.glob(".*.tmp"))
    assert scan_cached(path, images, labels, num_classes=2, lock_timeout=0.1).cache_write_error is None


def _inherited_lock_worker(lock, queue):
    messages = []
    for method in (lock.try_acquire, lock.release):
        try:
            method()
        except RuntimeError as error:
            messages.append(str(error))
    queue.put(messages)


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="POSIX fork ownership check")
def test_inherited_lock_cannot_unlock_parent(tmp_path):
    lock = _native.CacheLock(str(tmp_path / "owner.lock"))
    assert lock.try_acquire()
    context = mp.get_context("fork")
    queue = context.Queue()
    process = context.Process(target=_inherited_lock_worker, args=(lock, queue))
    process.start()
    messages = queue.get(timeout=10)
    process.join(10)
    assert process.exitcode == 0 and len(messages) == 2 and all("after fork" in m for m in messages)
    contender = _native.CacheLock(str(tmp_path / "owner.lock"))
    assert not contender.try_acquire()
    lock.release()
    assert contender.try_acquire()
    contender.release()


def _concurrent_worker(root, start, queue):
    import ultrafast_yolo_dataset._cache as cache_module

    root = Path(root)
    original = cache_module.scan

    @functools.wraps(original)
    def measured(*args, **kwargs):
        with (root / "scans.txt").open("a") as stream:
            stream.write(f"{os.getpid()}\n")
        time.sleep(0.05)
        return original(*args, **kwargs)

    cache_module.scan = measured
    start.wait(20)
    result = scan_cached(root / "shared.uydcache", [str(root / "image.png")], [str(root / "label.txt")], num_classes=1)
    queue.put((result.cache_hit, result.boxes.tobytes(), result.cache_write_error))


def test_concurrent_processes_scan_once_and_reuse_after_lock(tmp_path):
    Image.new("RGB", (32, 24)).save(tmp_path / "image.png")
    (tmp_path / "label.txt").write_text("0 .5 .5 .2 .2")
    context = mp.get_context("spawn")
    start, queue = context.Event(), context.Queue()
    processes = [context.Process(target=_concurrent_worker, args=(str(tmp_path), start, queue)) for _ in range(3)]
    for process in processes:
        process.start()
    start.set()
    try:
        results = [queue.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(10)
            assert process.exitcode == 0
        assert sorted(r[0] for r in results) == [False, True, True]
        assert len({r[1] for r in results}) == 1 and all(r[2] is None for r in results)
        assert len((tmp_path / "scans.txt").read_text().splitlines()) == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
