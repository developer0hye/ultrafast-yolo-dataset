"""Verify the public JSON and fixed-width proof against independent Python bytes."""

import hashlib
import json
import os
import struct

import pytest
from ultrafast_yolo_dataset import _native


@pytest.mark.parametrize("workers", [1, 4])
def test_json_and_table_bytes_for_mixed_file_kinds(tmp_path, workers):
    files = [(tmp_path / "empty", b""), (tmp_path / "abc", b"abc"), (tmp_path / "binary", bytes(range(256)))]
    for path, data in files:
        path.write_bytes(data)
    directory = tmp_path / "directory"
    directory.mkdir()
    paths = [path for path, _ in files] + [directory, tmp_path / "missing"]
    kinds = ["file"] * 3 + ["directory", "missing"]
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "fifo"
        os.mkfifo(fifo)
        paths.append(fifo)
        kinds.append("other")
    expected, proof = [], bytearray()
    payloads = dict(files)
    for path, kind in zip(paths, kinds):
        metadata = None if kind == "missing" else path.stat()
        value = {
            "kind": kind,
            "size": 0 if metadata is None else metadata.st_size,
            "modified_ns": 0 if metadata is None else metadata.st_mtime_ns,
            "sha256": hashlib.sha256(payloads[path]).hexdigest() if kind == "file" else None,
        }
        expected.append(value)
        snapshot = _native.read_snapshot(str(path), 4096)
        assert snapshot.fingerprint_json() == json.dumps(value, separators=(",", ":"))
        assert snapshot.data == payloads.get(path)
        proof.extend(struct.pack("<BQ", {"file": 0, "missing": 1, "directory": 2, "other": 3}[kind], value["size"]))
        proof.extend(value["modified_ns"].to_bytes(16, "little", signed=True))
        proof.extend(bytes.fromhex(value["sha256"]) if value["sha256"] is not None else bytes(32))
    strings = list(map(str, paths))
    serialized = _native.fingerprint_paths(strings, 4096, workers, True)
    assert serialized == json.dumps(expected, separators=(",", ":"))
    table = _native.pack_fingerprint_table(serialized, len(paths))
    assert table == bytes(proof)
    _native.verify_fingerprint_table(table, strings, 4096, workers, True)
    _native.verify_fingerprint_table(table, strings, 4096, workers, False)


def test_content_validation_rejects_same_size_rewrite_with_restored_mtime(tmp_path):
    path = tmp_path / "same-size.bin"
    path.write_bytes(b"abc")
    paths = [str(path)]
    snapshot = _native.read_snapshot(str(path), 4096)
    original_json = snapshot.fingerprint_json()
    proof = _native.pack_fingerprint_table("[" + original_json + "]", 1)
    metadata = path.stat()
    path.write_bytes(b"xyz")
    os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    assert path.stat().st_mtime_ns == metadata.st_mtime_ns
    _native.verify_fingerprint_table(proof, paths, 4096, 1, False)
    with pytest.raises(_native.InputChangedError, match="input differs from captured bytes"):
        _native.verify_fingerprint_table(proof, paths, 4096, 1, True)
    assert snapshot.data == b"abc" and snapshot.fingerprint_json() == original_json
    fresh = json.loads(_native.fingerprint_paths(paths, 4096, 1, True))[0]
    assert fresh["sha256"] == hashlib.sha256(b"xyz").hexdigest()
