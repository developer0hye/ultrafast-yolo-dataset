import gc
import signal

import numpy as np
import pytest
from ultrafast_yolo_dataset import _native


def fixture(seed, count):
    rng = np.random.default_rng(seed)
    sources = np.arange(count, dtype=np.int64) * 2
    paths = tuple(f"图像 {i}.png" for i in range(count * 2))
    shapes = np.column_stack((np.arange(count) + 24, np.arange(count) + 32)).astype(np.int64)
    sizes = rng.integers(1, 18, count)
    if count:
        sizes[-1] = 0
    offsets = np.r_[np.int64(0), np.cumsum(sizes, dtype=np.int64)]
    # Materialization must copy any float32 payload bit-for-bit; this does not
    # change the parser's narrower accepted domain (finite annotations).
    labels = rng.integers(0, 2**32, (int(offsets[-1]), 5), dtype=np.uint32).view(np.float32)
    objects = np.array([j for i in range(count) if i % 2 for j in range(offsets[i], offsets[i + 1])], dtype=np.int64)
    polygon_sizes = rng.integers(3, 13, len(objects))
    segment_offsets = np.r_[np.int64(0), np.cumsum(polygon_sizes, dtype=np.int64)]
    points = rng.integers(0, 2**32, (int(segment_offsets[-1]), 2), dtype=np.uint32).view(np.float32)
    return [paths, sources, shapes, offsets, labels, points, segment_offsets, objects]


def python_export(args):
    paths, sources, shapes, offsets, labels, points, segment_offsets, objects = args
    result = []
    for i, source in enumerate(sources):
        start, end = offsets[i : i + 2]
        segments = [
            points[segment_offsets[j] : segment_offsets[j + 1]].copy()
            for j, obj in enumerate(objects)
            if start <= obj < end
        ]
        result.append(
            {
                "im_file": paths[source],
                "shape": tuple(map(int, shapes[i])),
                "cls": labels[start:end, :1].copy(),
                "bboxes": labels[start:end, 1:].copy(),
                "segments": segments,
                "keypoints": None,
                "normalized": True,
                "bbox_format": "xywh",
            }
        )
    return result


def assert_equal(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert a.keys() == b.keys()
        for key in ("im_file", "shape", "keypoints", "normalized", "bbox_format"):
            assert a[key] == b[key] and type(a[key]) is type(b[key])
        assert len(a["segments"]) == len(b["segments"])
        for x, y in [(a["cls"], b["cls"]), (a["bboxes"], b["bboxes"]), *zip(a["segments"], b["segments"])]:
            assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
            assert x.flags.writeable and x.flags.c_contiguous
            assert x.flags.owndata and x.base is None


def test_seeded_materialization_parity():
    for seed in range(200):
        values = fixture(seed, seed % 40)
        assert_equal(_native.materialize_labels(*values), python_export(values))


def test_exported_arrays_are_independent_and_keep_ownership():
    values = fixture(72, 8)
    expected = python_export(values)
    first, second = _native.materialize_labels(*values), _native.materialize_labels(*values)
    arrays = [a for row in first for a in [row["cls"], row["bboxes"], *row["segments"]]]
    for i, a in enumerate(arrays):
        assert not any(np.shares_memory(a, b) for b in arrays[i + 1 :])
        assert not np.shares_memory(a, values[4]) and not np.shares_memory(a, values[5])
        a[...] = -9
    assert_equal(second, expected)
    assert_equal(_native.materialize_labels(*values), expected)
    del first, second, values
    gc.collect()
    assert all(np.all(a == -9) for a in arrays)


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="requires Unix interval timer")
def test_large_materialization_honors_interrupts():
    count = 100000
    args = [
        ("image.png",) * count,
        np.arange(count, dtype=np.int64),
        np.full((count, 2), 32, dtype=np.int64),
        np.arange(count + 1, dtype=np.int64),
        np.zeros((count, 5), dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
        np.zeros(1, dtype=np.int64),
        np.empty(0, dtype=np.int64),
    ]

    def interrupt(*_):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGALRM, interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            signal.setitimer(signal.ITIMER_REAL, 0.005)
            _native.materialize_labels(*args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert _native.materialize_labels(*fixture(0, 0)) == []


@pytest.mark.parametrize("index", range(1, 8))
def test_misaligned_packed_array_rejected_before_typed_reads(index):
    values = fixture(2, 8)
    original = values[index]
    unaligned = np.ndarray(original.shape, dtype=original.dtype, buffer=bytearray(original.nbytes + 1), offset=1)
    unaligned[:] = original
    assert unaligned.flags.c_contiguous and not unaligned.flags.aligned
    values[index] = unaligned
    with pytest.raises(ValueError, match="aligned and C-contiguous"):
        _native.materialize_labels(*values)


@pytest.mark.parametrize("index", [2, 4])
def test_column_major_packed_matrices_rejected(index):
    values = fixture(2, 8)
    values[index] = np.asfortranarray(values[index])
    with pytest.raises(ValueError, match="C-contiguous"):
        _native.materialize_labels(*values)


@pytest.mark.parametrize("index", [1, 4])
def test_non_native_endian_arrays_rejected(index):
    values = fixture(2, 8)
    values[index] = values[index].astype(values[index].dtype.newbyteorder("S"))
    with pytest.raises(TypeError):
        _native.materialize_labels(*values)


@pytest.mark.parametrize(
    "damage",
    [
        "source",
        "source_order",
        "shape",
        "label_columns",
        "point_columns",
        "offset_start",
        "offset_end",
        "offset_order",
        "segment_end",
        "object_bounds",
        "object_order",
        "partial",
        "path",
        "stride",
    ],
)
def test_malformed_materialization_inputs_are_rejected(damage):
    values = fixture(2, 8)
    if damage == "source":
        values[1][0] = 999
    elif damage == "source_order":
        values[1][1] = values[1][0]
    elif damage == "shape":
        values[2] = values[2][:-1].copy()
    elif damage == "label_columns":
        values[4] = values[4][:, :4].copy()
    elif damage == "point_columns":
        values[5] = values[5][:, :1].copy()
    elif damage == "offset_start":
        values[3][0] = -1
    elif damage == "offset_end":
        values[3][-1] += 1
    elif damage == "offset_order":
        values[3][1] = -1
    elif damage == "segment_end":
        values[6][-1] += 1
    elif damage == "object_bounds":
        values[7][0] = -1
    elif damage == "object_order":
        values[7][1] = values[7][0]
    elif damage == "partial":
        values[7][0] = 0  # One segment attached to the first Detection-only image.
    elif damage == "path":
        values[0] = (None, *values[0][1:])
    else:
        values[4] = values[4][::-1]
    with pytest.raises((ValueError, TypeError)):
        _native.materialize_labels(*values)
