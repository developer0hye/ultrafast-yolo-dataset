"""Small independent format-oracle checks for the future reader benchmark."""

import hashlib
import importlib.util
import struct
import sys
from pathlib import Path

import pytest

if sys.platform not in {"darwin", "linux"}:
    pytest.skip("reader RSS benchmark targets macOS and Linux", allow_module_level=True)

spec = importlib.util.spec_from_file_location(
    "cache_read_comparison", Path(__file__).parents[1] / "bench/cache_read_comparison.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def make_container(sections):
    return (
        b"UYDCACHE"
        + struct.pack("<II", 1, len(sections))
        + b"".join(struct.pack("<Q", len(x)) + hashlib.sha256(x).digest() for x in sections)
        + b"".join(sections)
    )


def test_streaming_oracle_preserves_empty_and_binary_sections(tmp_path):
    payloads = [b"{}", b"", b"\x00\xff\x01", b"tail"]
    value = make_container(payloads)
    path = tmp_path / "fixture.uydcache"
    path.write_bytes(value)
    result = bench.inspect_container(path)
    assert result["bytes"] == len(value)
    assert result["sha256"] == hashlib.sha256(value).hexdigest()
    assert result["sections"] == [{"bytes": len(x), "sha256": hashlib.sha256(x).hexdigest()} for x in payloads]


@pytest.mark.parametrize(
    "damage", ["magic", "version", "zero_sections", "descriptor", "truncated_payload", "checksum", "trailing"]
)
def test_streaming_oracle_rejects_damaged_containers(tmp_path, damage):
    data = bytearray(make_container([b"{}", b"payload"]))
    if damage == "magic":
        data[0] ^= 1
    elif damage == "version":
        data[8:12] = struct.pack("<I", 2)
    elif damage == "zero_sections":
        data[12:16] = struct.pack("<I", 0)
    elif damage == "descriptor":
        del data[20:]
    elif damage == "truncated_payload":
        del data[-1:]
    elif damage == "checksum":
        data[-1] ^= 1
    elif damage == "trailing":
        data += b"X"
    path = tmp_path / "bad.uydcache"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        bench.inspect_container(path)


def test_five_proportional_pairs_have_exact_constant_interval():
    pairs = [(2, 1), (10, 5), (6, 3), (4, 2), (8, 4)]
    result = bench.summary(pairs)
    assert result["baseline_median"] == 6
    assert result["candidate_median"] == 3
    assert result["baseline_over_candidate"] == 2
    assert result["paired_bootstrap_ci95"] == [2, 2]
    assert result["raw_pairs"] == pairs
