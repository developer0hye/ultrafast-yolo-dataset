"""Cross-check accelerated SHA256 at padding/read boundaries against hashlib."""

import hashlib
import json

import pytest
from ultrafast_yolo_dataset import _native


@pytest.mark.parametrize("size", [0, 1, 55, 56, 63, 64, 65, 119, 120, 127, 128, 129, 65535, 65536, 65537, 1000003])
def test_snapshot_and_compact_proof_sha256_against_hashlib(tmp_path, size):
    data = (bytes(range(256)) * ((size + 255) // 256))[:size]
    path = tmp_path / "snapshot.bin"
    path.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    snapshot = _native.read_snapshot(str(path), 2 * 1024**2)
    assert snapshot.data == data
    proof = json.loads(snapshot.fingerprint_json())
    assert proof["sha256"] == expected
    paths = [str(path)] * 3
    assert [entry["sha256"] for entry in json.loads(_native.fingerprint_paths(paths, 2 * 1024**2, 3, True))] == [
        expected
    ] * 3
    table = _native.pack_fingerprint_table(json.dumps([proof] * 3), 3)
    assert table[25:57].hex() == expected
    _native.verify_fingerprint_table(table, paths, 2 * 1024**2, 3, True)
    cache = tmp_path / "sections.uydcache"
    _native.write_cache_sections(str(cache), [b"{}", data], 2 * 1024**2)
    assert cache.read_bytes()[64:96].hex() == expected  # Second descriptor's SHA256.
    assert _native.read_cache_sections(str(cache), 2 * 1024**2) == [b"{}", data]
