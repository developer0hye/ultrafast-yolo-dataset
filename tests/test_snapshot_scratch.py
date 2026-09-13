"""Mixed reads catch stale scratch tails and cross-call ownership/worker mistakes."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from ultrafast_yolo_dataset import _native


def make_files(root):
    sizes = (131077, 0, 1, 65535, 3, 65536, 65537, 7, 0)
    pairs = []
    for index, size in enumerate(sizes):
        data = bytes((i + index * 37) % 256 for i in range(size))
        path = root / f"read-{index}.bin"
        path.write_bytes(data)
        pairs.append((path, data))
    return pairs


def test_retained_snapshots_survive_mixed_reads_and_file_removal(tmp_path):
    pairs = make_files(tmp_path)
    snapshots = [_native.read_snapshot(str(path), 200000) for path, _ in pairs]
    for path, _ in pairs:
        path.unlink()
    for result, (_, expected) in zip(snapshots, pairs):
        assert result.data == expected
        assert json.loads(result.fingerprint_json())["sha256"] == hashlib.sha256(expected).hexdigest()


@pytest.mark.parametrize("workers", [1, 3, 7])
def test_mixed_lengths_across_chunks_match_hashlib_and_compact_proofs(tmp_path, workers):
    pairs = make_files(tmp_path)
    ordered = [pairs[(i * 7) % len(pairs)] for i in range(777)]
    paths = [str(path) for path, _ in ordered]
    values = json.loads(_native.fingerprint_paths(paths, 200000, workers, True))
    assert [v["sha256"] for v in values] == [hashlib.sha256(data).hexdigest() for _, data in ordered]
    proof = _native.pack_fingerprint_table(json.dumps(values), len(paths))
    _native.verify_fingerprint_table(proof, paths, 200000, workers, True)
    metadata = json.loads(_native.fingerprint_paths(paths, 200000, workers, False))
    assert [v["sha256"] for v in metadata] == [None] * len(paths)
    assert [v["size"] for v in metadata] == [len(data) for _, data in ordered]


def test_python_threads_preserve_distinct_snapshot_outputs(tmp_path):
    pairs = make_files(tmp_path)

    def read(index):
        path, expected = pairs[index % len(pairs)]
        result = _native.read_snapshot(str(path), 200000)
        assert result.data == expected
        assert json.loads(result.fingerprint_json())["sha256"] == hashlib.sha256(expected).hexdigest()
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(read, range(90)))
    for index, result in enumerate(results):
        assert result.data == pairs[index % len(pairs)][1]


def test_limit_error_then_small_and_missing_inputs_do_not_reuse_payload(tmp_path):
    pairs = make_files(tmp_path)
    large, expected = pairs[0]
    prior = _native.read_snapshot(str(large), 200000)
    with pytest.raises(_native.SnapshotLimitError):
        _native.read_snapshot(str(large), 1)
    small, data = pairs[2]
    result = _native.read_snapshot(str(small), 200000)
    assert result.data == data
    assert prior.data == expected
    missing = _native.read_snapshot(str(tmp_path / "missing"), 200000)
    assert missing.kind == "missing" and missing.data is None
