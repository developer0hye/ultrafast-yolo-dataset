"""Container read semantics at chunk boundaries and owned-buffer lifetimes."""

import gc
import hashlib
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from ultrafast_yolo_dataset import _native

CHUNK = 4 * 1024 * 1024


def container(sections):
    header = b"UYDCACHE" + struct.pack("<II", 1, len(sections))
    descriptors = b"".join(struct.pack("<Q", len(s)) + hashlib.sha256(s).digest() for s in sections)
    return header + descriptors + b"".join(sections)


def payload(length):
    block = bytes(range(256))
    return (block * ((length + 255) // 256))[:length]


@pytest.mark.parametrize("length", [0, 1, CHUNK - 1, CHUNK, CHUNK + 1, CHUNK * 2 + 17])
def test_exact_sections_across_read_chunks(tmp_path, length):
    expected = [b"{}", payload(length), b"", b"trailer"]
    encoded = container(expected)
    path = tmp_path / "sections.cache"
    path.write_bytes(encoded)
    result = _native.read_cache_sections(str(path), len(encoded))
    assert result == expected
    assert type(result) is list and all(type(section) is bytes for section in result)
    assert all(memoryview(section).readonly for section in result)


def test_numpy_owner_survives_list_and_file_removal(tmp_path):
    expected = payload(CHUNK + 31)
    encoded = container([b"{}", expected])
    path = tmp_path / "sections.cache"
    path.write_bytes(encoded)
    sections = _native.read_cache_sections(str(path), len(encoded))
    array = np.frombuffer(sections[1], dtype=np.uint8)
    assert not array.flags.writeable
    del sections
    path.unlink()
    gc.collect()
    assert array.tobytes() == expected
    with pytest.raises(ValueError):
        array.setflags(write=True)


def test_late_checksum_failure_preserves_previous_result(tmp_path):
    expected = [b"{}", payload(CHUNK + 3), payload(CHUNK + 19)]
    encoded = container(expected)
    path = tmp_path / "sections.cache"
    path.write_bytes(encoded)
    previous = _native.read_cache_sections(str(path), len(encoded))
    broken = bytearray(encoded)
    broken[-1] ^= 1
    path.write_bytes(broken)
    with pytest.raises(_native.CacheFormatError, match="checksum"):
        _native.read_cache_sections(str(path), len(encoded))
    assert previous == expected
    path.write_bytes(encoded)
    assert _native.read_cache_sections(str(path), len(encoded)) == expected


def test_concurrent_readers_return_independent_immutable_bytes(tmp_path):
    expected = [b"{}", payload(CHUNK + 7)]
    encoded = container(expected)
    path = tmp_path / "sections.cache"
    path.write_bytes(encoded)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: _native.read_cache_sections(str(path), len(encoded)), range(8)))
    assert all(result == expected for result in results)
    assert len({id(result[1]) for result in results}) == len(results)
    path.unlink()
    assert all(result[1] == expected[1] for result in results)


@pytest.mark.parametrize("damage", ["truncated", "oversize", "zero_limit"])
def test_container_size_limits(tmp_path, damage):
    encoded = container([b"{}", payload(CHUNK + 1)])
    path = tmp_path / "sections.cache"
    path.write_bytes(encoded[:-1] if damage == "truncated" else encoded)
    limit = 0 if damage == "zero_limit" else len(encoded) - int(damage == "oversize")
    error = ValueError if damage == "zero_limit" else _native.CacheFormatError
    with pytest.raises(error):
        _native.read_cache_sections(str(path), limit)
