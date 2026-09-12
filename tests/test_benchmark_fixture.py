"""Small synthetic filesystem cases for benchmark preflight, not performance data."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "fixture_files", Path(__file__).resolve().parents[1] / "bench/fixture_files.py"
)
fixture_files = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture_files)


def original_fingerprint(root):
    digest, total = hashlib.sha256(), 0
    paths = sorted((root / "images").rglob("*.jpg")) + sorted((root / "labels").rglob("*.txt"))
    for path in paths:
        data = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + data)
        total += len(data)
    return {"files": len(paths), "bytes": total, "sha256": digest.hexdigest()}


def make_fixture(root, count):
    label_digest, label_bytes = hashlib.sha256(), 0
    for index in range(count):
        stem = f"{index // 1000:04d}/{index:08d}"
        image, label = root / "images" / (stem + ".jpg"), root / "labels" / (stem + ".txt")
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        # This verifier checks files/bytes, not JPEG decoding. Actual benchmark
        # image decoding stays in the existing Pillow pipeline.
        image.write_bytes(f"synthetic image bytes {index}".encode())
        data = f"{index % 3} .5 .5 .2 .2\n".encode()
        label.write_bytes(data)
        label_digest.update((stem + ".txt").encode() + b"\0" + data)
        label_bytes += len(data)
    labels = {
        "count": count,
        "task": "detect",
        "bytes": label_bytes,
        "sha256_names_and_contents": label_digest.hexdigest(),
    }
    (root / "labels/manifest.json").write_text(json.dumps(labels))
    manifest = {
        "kind": "ultrafast-yolo-startup-synthetic-v1",
        "count": count,
        "task": "detect",
        "labels": labels,
        "fingerprint": original_fingerprint(root),
    }
    (root / "startup.json").write_text(json.dumps(manifest))
    return manifest


def test_streaming_hash_preserves_original_global_path_order(tmp_path):
    for folder, suffix in (("images", ".jpg"), ("labels", ".txt")):
        for name in ("a/2", "a/1", "a-", "a.z/z", "00", ".hidden", ""):
            path = tmp_path / folder / (name + suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((folder + ":" + name).encode())
        (tmp_path / folder / "ignored.cache").write_bytes(b"not an input")
    assert fixture_files.fingerprint(tmp_path) == original_fingerprint(tmp_path)


def test_shard_boundary_and_benchmark_cache_files(tmp_path):
    manifest = make_fixture(tmp_path, 1003)
    (tmp_path / "labels/0000.cache").write_bytes(b"not a label")
    result = fixture_files.audit_synthetic(tmp_path, 1003)
    assert result["counts"] == {"images": 1003, "labels": 1003}
    assert result["fingerprint"] == manifest["fingerprint"]
    assert result["all_regular_single_link_files"]


@pytest.mark.parametrize(
    "fault,message",
    [
        ("content", "whole-fixture fingerprint mismatch"),
        ("missing", "unexpected/missing files"),
        ("extra", "unexpected/missing files"),
        ("count", "wrong synthetic fixture kind or pair count"),
    ],
)
def test_fixture_drift_is_rejected(tmp_path, fault, message):
    make_fixture(tmp_path, 3)
    path = tmp_path / "images/0000/00000000.jpg"
    if fault == "content":
        path.write_bytes(b"changed bytes")
    elif fault == "missing":
        path.unlink()
    elif fault == "extra":
        path.with_name("extra.jpg").write_bytes(b"extra")
    with pytest.raises(ValueError, match=message):
        fixture_files.audit_synthetic(tmp_path, 4 if fault == "count" else 3)


@pytest.mark.parametrize("kind", ["hardlink", "symlink"])
def test_repeated_paths_do_not_pass_as_distinct_files(tmp_path, kind):
    make_fixture(tmp_path, 3)
    target = tmp_path / "images/0000/00000000.jpg"
    alias = target.with_name("00000001.jpg")
    alias.unlink()
    try:
        (os.link if kind == "hardlink" else os.symlink)(target, alias)
    except OSError as error:
        pytest.skip(f"filesystem does not permit {kind}: {error}")
    with pytest.raises(ValueError, match="unaliased regular file"):
        fixture_files.audit_synthetic(tmp_path, 3)


def test_directory_symlink_does_not_change_original_hash_traversal(tmp_path):
    make_fixture(tmp_path, 3)
    target = tmp_path / "images/0000"
    try:
        (tmp_path / "images/alias").symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"filesystem does not permit directory symlinks: {error}")
    assert fixture_files.fingerprint(tmp_path) == original_fingerprint(tmp_path)
    with pytest.raises(ValueError, match="unexpected shard directories"):
        fixture_files.audit_synthetic(tmp_path, 3)
