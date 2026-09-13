"""Metadata-validated packed annotation cache with on-demand label dictionaries.

The reference validates a cache by one Python ``os.stat`` per file, unpickles one
dictionary and several small arrays per image, and builds a ``Path`` per image.
Each of those is a per-object interpreter cost repeated hundreds of thousands of
times. Here:

* Rust lists every directory once with its entries' metadata and digests the
  ordered paths, kinds, sizes and modification times without the GIL;
* annotations stay in a few contiguous arrays inside a memory-mapped file,
  checksummed in parallel natively, then viewed by NumPy without copying;
* a per-image dictionary is created only when that sample is requested.

Validation is per file (kind, size, modification time) plus the ordered path
list, which is stronger than the reference's total-size-plus-paths hash.
"""

import json
import mmap
import operator
import os
import struct
import sys
import traceback
import uuid
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import _native

MAGIC = b"UYDFAST1"
SCHEMA = 3
ALIGN = 64
_HEADER = struct.Struct("<8sII")
_SECTION = struct.Struct("<QQ32s")
_ARRAYS = {
    "sources": ("<i8", 1),  # ordered input index of each valid image
    "shapes": ("<i8", 2),  # (h, w) per valid image
    "object_offsets": ("<i8", 1),  # valid images + 1
    "rows": ("<f4", 2),  # (objects, 5): class, x, y, w, h
    "segment_objects": ("<i8", 1),  # object index of each segment
    "image_segments": ("<i8", 1),  # first segment of each valid image, + 1
    "segment_offsets": ("<i8", 1),  # segments + 1
    "points": ("<f4", 2),  # (points, 2)
}


class CacheMiss(Exception):
    """The fast cache is absent, stale, damaged, or from another profile."""


@lru_cache(maxsize=1)
def profile():
    """Everything that could change a scan result for identical input bytes."""
    import hashlib

    from ._cache import _profile

    digest = hashlib.sha256(_profile().encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def _checksum(buffer, workers):
    return _native.chunked_sha256(buffer, workers)


def _aligned(offset):
    return (offset + ALIGN - 1) // ALIGN * ALIGN


def arrays_from_scan(result):
    """Convert a ScanResult's packed arrays into the fast cache layout."""
    objects = np.asarray(result.segment_object_indices, dtype="<i8")
    offsets = np.asarray(result.object_offsets, dtype="<i8")
    return {
        "sources": np.asarray(result.source_indices, dtype="<i8"),
        "shapes": np.asarray(result.image_shapes, dtype="<i8").reshape(-1, 2),
        "object_offsets": offsets,
        "rows": np.asarray(result._labels, dtype="<f4").reshape(-1, 5),
        "segment_objects": objects,
        "image_segments": np.searchsorted(objects, offsets).astype("<i8"),
        "segment_offsets": np.asarray(result.segment_offsets, dtype="<i8"),
        "points": np.asarray(result.segment_points, dtype="<f4").reshape(-1, 2),
    }


def save(path, *, digest, config, summary, diagnostics, fallback_count, arrays, workers):
    """Atomically publish a cache. Returns None on success or a failure reason."""
    path = Path(path)
    metadata = {
        "schema": SCHEMA,
        "profile": profile(),
        "byteorder": sys.byteorder,
        "config": config,
        "digest": digest.hex(),
        "summary": summary,
        "diagnostics": diagnostics,
        "fallback_count": fallback_count,
        "arrays": {k: {"dtype": _ARRAYS[k][0], "shape": list(arrays[k].shape)} for k in _ARRAYS},
    }
    payloads = [json.dumps(metadata, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()]
    # Flat byte views (valid for empty arrays too); no section is copied.
    payloads += [np.ascontiguousarray(arrays[k], dtype=_ARRAYS[k][0]).reshape(-1).view(np.uint8) for k in _ARRAYS]
    offset = _aligned(_HEADER.size + _SECTION.size * len(payloads))
    table = []
    for payload in payloads:
        view = memoryview(payload)
        table.append((offset, view.nbytes, _checksum(view, workers)))
        offset = _aligned(offset + view.nbytes)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(temporary, "xb") as stream:
            stream.write(_HEADER.pack(MAGIC, SCHEMA, len(payloads)))
            stream.writelines(_SECTION.pack(*entry) for entry in table)
            for (start, _, _), payload in zip(table, payloads):
                stream.write(b"\0" * (start - stream.tell()))
                stream.write(memoryview(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return None
    except OSError as error:
        return f"{type(error).__name__}: {error}"
    finally:
        temporary.unlink(missing_ok=True)


def _require(condition, message):
    if not condition:
        raise CacheMiss(message)


def load(path, *, digest, config, workers):
    """Return (metadata, arrays) for a valid cache, else raise CacheMiss.

    A rejected cache is unmapped before the error propagates: the caller then
    rewrites it, and Windows cannot replace a file that is still mapped.
    """
    if sys.byteorder != "little":
        raise CacheMiss("fast cache supports little-endian hosts only")
    try:
        with open(path, "rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            _require(size >= _HEADER.size, "truncated header")
            # The mapping outlives the descriptor; arrays keep it alive.
            mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    except (OSError, ValueError) as error:
        raise CacheMiss(f"{type(error).__name__}: {error}") from None
    try:
        return _parse(mapped, size, digest, config, workers)
    except BaseException as error:
        # Finished frames of the traceback still hold the rejected arrays,
        # which export the mapping; drop them so it can close.
        traceback.clear_frames(error.__traceback__)
        mapped.close()
        raise


def _parse(mapped, size, digest, config, workers):
    magic, schema, count = _HEADER.unpack_from(mapped, 0)
    _require(magic == MAGIC and schema == SCHEMA, "unknown cache format")
    _require(count == len(_ARRAYS) + 1, "unexpected section count")
    _require(_HEADER.size + _SECTION.size * count <= size, "truncated section table")
    table = [_SECTION.unpack_from(mapped, _HEADER.size + i * _SECTION.size) for i in range(count)]
    for offset, length, _ in table:
        _require(offset % ALIGN == 0 and offset + length <= size, "section out of bounds")
    view = memoryview(mapped)
    try:
        for offset, length, expected in table:
            _require(_checksum(view[offset : offset + length], workers) == expected, "section checksum mismatch")
        offset, length, _ = table[0]
        try:
            metadata = json.loads(bytes(view[offset : offset + length]))
        except ValueError:
            raise CacheMiss("invalid metadata") from None
    finally:
        view.release()
    _require(isinstance(metadata, dict) and metadata.get("schema") == SCHEMA, "invalid metadata")
    _require(metadata.get("profile") == profile(), "scanner/reference profile changed")
    _require(metadata.get("config") == config, "scan configuration changed")
    _require(metadata.get("digest") == digest.hex(), "input paths or file metadata changed")
    arrays = {}
    descriptors = metadata.get("arrays")
    _require(isinstance(descriptors, dict) and set(descriptors) == set(_ARRAYS), "invalid array manifest")
    for (name, (dtype, rank)), (offset, length, _) in zip(_ARRAYS.items(), table[1:]):
        shape = descriptors[name].get("shape")
        _require(descriptors[name].get("dtype") == dtype, "invalid array dtype")
        _require(
            isinstance(shape, list) and len(shape) == rank and all(type(n) is int and n >= 0 for n in shape),
            "invalid array shape",
        )
        items = int(np.prod(shape, dtype=np.int64))
        _require(items * np.dtype(dtype).itemsize == length, "array length disagrees with shape")
        arrays[name] = np.frombuffer(mapped, dtype=dtype, count=items, offset=offset).reshape(shape)
    _validate(arrays, metadata)
    return metadata, arrays


def _validate(arrays, metadata):
    """Cheap structural checks: every later index/slice stays in bounds."""
    sources, shapes, offsets, rows = (arrays[k] for k in ("sources", "shapes", "object_offsets", "rows"))
    objects, image_segments, segment_offsets, points = (
        arrays[k] for k in ("segment_objects", "image_segments", "segment_offsets", "points")
    )
    summary = metadata.get("summary")
    _require(isinstance(summary, dict) and type(summary.get("total")) is int, "invalid summary")
    total, valid = summary["total"], len(sources)
    _require(shapes.shape == (valid, 2) and rows.shape[1:] == (5,) and points.shape[1:] == (2,), "invalid shapes")
    _require(
        valid == 0 or (sources[0] >= 0 and sources[-1] < total and bool(np.all(np.diff(sources) > 0))),
        "invalid sources",
    )
    for values, length, end in (
        (offsets, valid + 1, len(rows)),
        (segment_offsets, len(objects) + 1, len(points)),
        (image_segments, valid + 1, len(objects)),
    ):
        _require(len(values) == length and values[0] == 0 and values[-1] == end, "invalid offset endpoints")
        _require(bool(np.all(np.diff(values) >= 0)), "invalid offset order")
    _require(len(objects) == 0 or bool(np.all(np.diff(objects) > 0)), "invalid segment objects")
    per_image = np.diff(image_segments)
    _require(bool(np.all((per_image == 0) | (per_image == np.diff(offsets)))), "partial segment alignment")
    _require(bool(np.all(np.diff(segment_offsets) > 0)), "empty segment")
    _require(np.array_equal(np.searchsorted(objects, offsets), image_segments), "inconsistent segment index")


class LazyLabels(Sequence):
    """The reference's list of per-image label dicts, created on access.

    Each access returns a new dict with new writable arrays, exactly as a
    fresh reference label would be, so in-place augmentation cannot alter the
    packed storage or another sample. Mutating a returned dict does not persist;
    the dataset adapter materializes a real list before any in-place update.
    """

    __slots__ = ("_arrays", "_drop_segments", "_im_files", "_n", "_paths")

    def __init__(self, im_files, arrays, drop_segments=False):
        self._im_files = im_files
        self._arrays = arrays
        self._n = len(arrays["sources"])
        self._drop_segments = drop_segments
        self._paths = None

    def __len__(self):
        return self._n

    def _item(self, i):
        a = self._arrays
        begin, end = int(a["object_offsets"][i]), int(a["object_offsets"][i + 1])
        rows = a["rows"][begin:end]
        if self._drop_segments:
            segments = []
        else:
            first, last = int(a["image_segments"][i]), int(a["image_segments"][i + 1])
            bounds = a["segment_offsets"][first : last + 1].tolist()
            points = a["points"]
            segments = [points[bounds[k] : bounds[k + 1]].copy() for k in range(last - first)]
        h, w = a["shapes"][i].tolist()
        return {
            "im_file": self._im_files[int(a["sources"][i])],
            "shape": (h, w),
            "cls": rows[:, :1].copy(),
            "bboxes": rows[:, 1:].copy(),
            "segments": segments,
            "keypoints": None,
            "normalized": True,
            "bbox_format": "xywh",
        }

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._item(i) for i in range(*index.indices(self._n))]
        index = operator.index(index)
        if index < 0:
            index += self._n
        if not 0 <= index < self._n:
            raise IndexError("label index out of range")
        return self._item(index)

    def __iter__(self):
        # Whole-dataset passes (label plots, class counts) use one native call
        # per chunk instead of one interpreter round trip per image.
        a = self._arrays
        if self._paths is None:
            self._paths = tuple(self._im_files)
        step = 16384
        for start in range(0, self._n, step):
            stop = min(start + step, self._n)
            begin, end = int(a["object_offsets"][start]), int(a["object_offsets"][stop])
            first, last = int(a["image_segments"][start]), int(a["image_segments"][stop])
            if self._drop_segments:
                first = last
            yield from _native.materialize_labels(
                self._paths,
                np.ascontiguousarray(a["sources"][start:stop]),
                np.ascontiguousarray(a["shapes"][start:stop]),
                np.ascontiguousarray(a["object_offsets"][start : stop + 1] - begin),
                np.ascontiguousarray(a["rows"][begin:end]),
                np.ascontiguousarray(a["points"][a["segment_offsets"][first] : a["segment_offsets"][last]]),
                np.ascontiguousarray(a["segment_offsets"][first : last + 1] - a["segment_offsets"][first]),
                np.ascontiguousarray(a["segment_objects"][first:last] - begin),
            )

    def materialize(self):
        """A real list of independent dicts, for in-place reference updates."""
        return list(self)

    def counts(self):
        """(boxes, segments) totals without creating per-image objects."""
        segments = 0 if self._drop_segments else len(self._arrays["segment_objects"])
        return len(self._arrays["rows"]), segments

    def without_segments(self):
        return LazyLabels(self._im_files, self._arrays, drop_segments=True)

    def with_zero_classes(self):
        """The reference's single_cls ``cls[:] = 0``, on a private copy of the rows."""
        arrays = dict(self._arrays)
        rows = np.array(arrays["rows"])
        rows[:, 0] = 0
        arrays["rows"] = rows
        return LazyLabels(self._im_files, arrays, drop_segments=self._drop_segments)

    def __reduce__(self):
        # Worker processes receive compact arrays, not per-image objects.
        arrays = {k: np.array(v) for k, v in self._arrays.items()}
        return LazyLabels, (self._im_files, arrays, self._drop_segments)
