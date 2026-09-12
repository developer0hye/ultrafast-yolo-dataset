"""Versioned native cache: owned arrays, captured-byte provenance, atomic writes."""

import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import MappingProxyType

import numpy as np
import PIL
from PIL import Image

from . import _native
from ._scan import Diagnostic, ScanResult, scan
from ._snapshots import InputChangedError

SCHEMA = 2
DEFAULT_MAX_CACHE_BYTES = 2 * 1024**3
_ARRAYS = {
    "source_indices": ("<i8", 1),
    "image_shapes": ("<i8", 2),
    "object_offsets": ("<i8", 1),
    "_labels": ("<f4", 2),
    "segment_points": ("<f4", 2),
    "segment_offsets": ("<i8", 1),
    "segment_object_indices": ("<i8", 1),
    "repaired_source_indices": ("<i8", 1),
}
_CONFIG_KEYS = (
    "num_classes",
    "task",
    "single_cls",
    "image_validation",
    "repair_jpeg",
    "max_file_bytes",
    "prefix",
    "max_image_bytes",
    "max_fallback_bytes",
)
_SUMMARY_KEYS = {"found", "missing", "empty", "corrupt", "total"}


@dataclass(frozen=True)
class CacheMiss:
    reason: str


@dataclass(frozen=True)
class SaveResult:
    saved: bool
    reused: bool = False
    reason: str | None = None


def _profile():
    digest = hashlib.sha256(_native.native_cache_profile().encode())
    for name in ("_cache.py", "_scan.py", "_snapshots.py", "_reference.py", "__init__.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    digest.update(f"{np.__version__}|{PIL.__version__}|{sys.version_info[:2]}|{sys.byteorder}".encode())
    digest.update(f"{sys.platform}|{platform.machine()}".encode())
    cpu = getattr(np._core._multiarray_umath, "__cpu_features__", {})
    digest.update(json.dumps(cpu, sort_keys=True).encode())
    # Pillow's optional plugins and Ultralytics' lazy HEIF opener affect both
    # valid formats and failure diagnostics. Never reuse a cache across an
    # opener/plugin-state change merely because Pillow's version is unchanged.
    Image.init()
    for name, (factory, _) in sorted(Image.OPEN.items()):
        digest.update(f"{name}:{factory.__module__}:{factory.__qualname__}".encode())
    try:
        digest.update(inspect.getsource(Image.open).encode())
    except (OSError, TypeError):
        digest.update(repr(Image.open).encode())
    try:
        digest.update(importlib.metadata.version("pi-heif").encode())
    except importlib.metadata.PackageNotFoundError:
        digest.update(b"pi-heif-unavailable")
    return digest.hexdigest()


def _request(images, labels, policy, options):
    if sys.byteorder != "little":
        raise ValueError("native cache currently supports little-endian hosts only")
    images = [os.fsdecode(os.fspath(p)) for p in images]
    labels = [os.fsdecode(os.fspath(p)) for p in labels]
    if len(images) != len(labels) or len(images) > 1_000_000:
        raise ValueError("equal image/label counts, at most 1,000,000, are required")
    if policy not in ("content", "metadata"):
        raise ValueError("cache fingerprint must be content or metadata")
    bound = inspect.signature(scan).bind(images, labels, fingerprint=policy, **options)
    bound.apply_defaults()
    values = bound.arguments
    if values["task"] not in ("detect", "segment") or values["image_validation"] != "pillow":
        raise ValueError("cache supports Detect/Segment with Pillow verification")
    if values["repair_jpeg"] not in ("reject", "reference") or values["num_classes"] < 1:
        raise ValueError("invalid scan configuration")
    if not 1 <= values["workers"] <= 256 or not 1 <= values["max_in_flight"] <= 65536:
        raise ValueError("invalid scan worker/queue limits")
    if min(values[k] for k in ("max_file_bytes", "max_image_bytes", "max_fallback_bytes")) < 1:
        raise ValueError("byte limits must be positive")
    config = {k: values[k] for k in _CONFIG_KEYS}
    cwd = os.getcwd() if any(not os.path.isabs(p) for p in chain(images, labels)) else None
    return images, labels, config, cwd, values["workers"]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _packed_proof(proof, count):
    _require(isinstance(proof, dict) and set(proof) == {"images", "labels"}, "invalid provenance manifest")
    packed = {}
    for kind in ("images", "labels"):
        entries = proof[kind]
        if isinstance(entries, bytes):
            _native.check_fingerprint_table(entries, count)
            packed[kind] = entries
        else:
            _require(isinstance(entries, list) and len(entries) == count, "invalid fingerprint count")
            packed[kind] = _native.pack_fingerprint_table(
                json.dumps(entries, ensure_ascii=False, allow_nan=False, separators=(",", ":")), count
            )
    return packed


def _validate_arrays(arrays, metadata):
    count = len(metadata["images"])
    sources, shapes, offsets, labels = (
        arrays[k] for k in ("source_indices", "image_shapes", "object_offsets", "_labels")
    )
    points, segment_offsets, objects, repaired = (
        arrays[k] for k in ("segment_points", "segment_offsets", "segment_object_indices", "repaired_source_indices")
    )
    valid, rows, segments = len(sources), len(labels), len(objects)
    _require(
        shapes.shape == (valid, 2) and labels.shape == (rows, 5) and points.shape[1:] == (2,), "invalid matrix shapes"
    )
    for indices in (sources, repaired):
        _require(
            np.all(indices >= 0) and np.all(indices < count) and np.all(np.diff(indices) > 0),
            "invalid source order/bounds",
        )
    _require(np.all(shapes >= 10), "invalid image dimensions")
    for values, expected_len, end in ((offsets, valid + 1, rows), (segment_offsets, segments + 1, len(points))):
        _require(len(values) == expected_len and values[0] == 0 and values[-1] == end, "invalid offset endpoints")
        _require(
            np.all(values >= 0) and np.all(values <= end) and np.all(np.diff(values) >= 0),
            "invalid offset order/bounds",
        )
    _require(
        np.all(objects >= 0) and np.all(objects < rows) and np.all(np.diff(objects) > 0),
        "invalid segment object indices",
    )
    per_image = np.diff(np.searchsorted(objects, offsets))
    _require(np.all((per_image == 0) | (per_image == np.diff(offsets))), "partial image segment alignment")
    _require(np.all(np.diff(segment_offsets) > 0), "empty segment in cache")
    for array in (labels, points):
        flat = array.reshape(-1)
        for start in range(0, len(flat), 262144):
            _require(np.isfinite(flat[start : start + 262144]).all(), "nonfinite cache annotation")
    summary = metadata["summary"]
    _require(isinstance(summary, dict) and set(summary) == _SUMMARY_KEYS, "invalid summary schema")
    _require(all(type(n) is int and 0 <= n <= count for n in summary.values()), "invalid counters")
    _require(summary["total"] == count and valid == count - summary["corrupt"], "invalid valid/corrupt count")
    _require(
        summary["empty"] <= summary["found"] and summary["found"] + summary["missing"] <= count, "inconsistent counters"
    )
    _require(
        type(metadata["fallback_count"]) is int and 0 <= metadata["fallback_count"] <= count, "invalid fallback count"
    )
    diagnostics = metadata["diagnostics"]
    _require(isinstance(diagnostics, list) and len(diagnostics) <= count, "invalid diagnostics")
    previous = -1
    for item in diagnostics:
        _require(
            isinstance(item, dict) and set(item) == {"source_index", "code", "message"}, "invalid diagnostic schema"
        )
        index = item["source_index"]
        _require(type(index) is int and previous < index < count, "invalid diagnostic source order")
        _require(isinstance(item["code"], str) and isinstance(item["message"], str), "invalid diagnostic text")
        previous = index


def _decode(sections, images, labels, config, cwd, policy):
    _require(len(sections) == len(_ARRAYS) + 3, "unexpected section count")
    metadata = json.loads(sections[0])
    _require(isinstance(metadata, dict), "invalid metadata object")
    _require(
        type(metadata["schema"]) is int and metadata["schema"] == SCHEMA and metadata["profile"] == _profile(),
        "unknown schema/parser/reference profile",
    )
    _require(metadata["byteorder"] == "little", "unsupported byte order")
    _require(
        metadata["images"] == images and metadata["labels"] == labels and metadata["cwd"] == cwd,
        "ordered path manifest changed",
    )
    _require(
        metadata["config"] == config and metadata["fingerprint"] == policy, "scan configuration/fingerprint changed"
    )
    _require(metadata["proof_encoding"] == "fixed57-le-v1", "unsupported fingerprint encoding")
    proof = _packed_proof(dict(zip(("images", "labels"), sections[-2:])), len(images))
    descriptors = metadata["arrays"]
    _require(isinstance(descriptors, dict) and set(descriptors) == set(_ARRAYS), "invalid array manifest")
    arrays = {}
    for (name, (dtype, rank)), contents in zip(_ARRAYS.items(), sections[1:]):
        descriptor = descriptors[name]
        _require(isinstance(descriptor, dict) and descriptor.get("dtype") == dtype, "invalid array dtype")
        shape = descriptor["shape"]
        _require(
            isinstance(shape, list) and len(shape) == rank and all(type(n) is int and n >= 0 for n in shape),
            "invalid array shape",
        )
        size = math.prod(shape)
        _require(size <= np.iinfo(np.intp).max // np.dtype(dtype).itemsize, "array shape overflow")
        _require(size * np.dtype(dtype).itemsize == len(contents), "array size disagrees with section")
        arrays[name] = np.frombuffer(contents, dtype=dtype).reshape(shape)
    _validate_arrays(arrays, metadata)
    result = object.__new__(ScanResult)
    result.image_paths, result.label_paths = tuple(images), tuple(labels)
    result.num_inputs, result.num_valid = len(images), len(arrays["source_indices"])
    for name, array in arrays.items():
        setattr(result, name, array)
    result.classes, result.boxes = result._labels[:, :1], result._labels[:, 1:]
    result.summary, result.config = MappingProxyType(metadata["summary"]), MappingProxyType(config)
    result.diagnostics = tuple(Diagnostic(**item) for item in metadata["diagnostics"])
    result.fallback_count, result.fingerprint_policy = metadata["fallback_count"], policy
    result._provenance = proof
    result._source_cwd = metadata["cwd"]
    result.cache_unavailable_reason = None
    return result


def _validate_inputs(result, workers, *, content):
    if result._source_cwd is not None and os.getcwd() != result._source_cwd:
        raise InputChangedError("working directory differs from captured relative-path context")
    proof = _packed_proof(result._provenance, result.num_inputs)
    for kind, paths, limit in (
        ("images", result.image_paths, result.config["max_image_bytes"]),
        ("labels", result.label_paths, result.config["max_file_bytes"]),
    ):
        _native.verify_fingerprint_table(proof[kind], paths, limit, workers, content)
    if result._source_cwd is not None and os.getcwd() != result._source_cwd:
        raise InputChangedError("working directory changed during input validation")


def load_cache(
    path, image_paths, label_paths, *, fingerprint="content", max_cache_bytes=DEFAULT_MAX_CACHE_BYTES, **scan_options
):
    """Return a validated ScanResult or CacheMiss, without unpickling any data."""
    request = _request(image_paths, label_paths, fingerprint, scan_options)
    return _load_requested(path, request, fingerprint, max_cache_bytes)


def _load_requested(path, request, fingerprint, max_cache_bytes):
    images, labels, config, cwd, workers = request
    if type(max_cache_bytes) is not int or max_cache_bytes <= 0:
        raise ValueError("max_cache_bytes must be a positive integer")
    try:
        sections = _native.read_cache_sections(os.fsdecode(os.fspath(path)), max_cache_bytes)
        result = _decode(sections, images, labels, config, cwd, fingerprint)
        _validate_inputs(result, workers, content=fingerprint == "content")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        OverflowError,
        RecursionError,
        InputChangedError,
    ) as error:
        return CacheMiss(f"{type(error).__name__}: {error}")
    result.cache_hit = True
    result.cache_path = str(path)
    result.cache_write_error = None
    return result


@contextmanager
def _locked(path, timeout):
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("lock_timeout must be finite and nonnegative")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _native.CacheLock(str(path) + ".lock")
    deadline = time.monotonic() + timeout
    try:
        while not lock.try_acquire():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"cache lock timeout: {path}")
            time.sleep(min(0.01, max(0, deadline - time.monotonic())))
        yield
    finally:
        lock.release()


def _encode(result):
    images, labels, config, cwd, _ = _request(
        result.image_paths, result.label_paths, result.fingerprint_policy, dict(result.config)
    )
    if cwd != result._source_cwd:
        raise InputChangedError("relative-path context changed before cache publication")
    proof = _packed_proof(result._provenance, len(images))
    arrays = {name: np.asarray(getattr(result, name), dtype=dtype) for name, (dtype, _) in _ARRAYS.items()}
    metadata = {
        "schema": SCHEMA,
        "profile": _profile(),
        "byteorder": "little",
        "reference_sha": "795a556942a12fe0124cf767888194a1d0b83e2e",
        "parser_version": "0.1.0a1-native-cache-2",
        "images": images,
        "labels": labels,
        "cwd": cwd,
        "config": config,
        "fingerprint": result.fingerprint_policy,
        "proof_encoding": "fixed57-le-v1",
        "summary": dict(result.summary),
        "diagnostics": [vars(item) for item in result.diagnostics],
        "fallback_count": result.fallback_count,
        "arrays": {name: {"dtype": _ARRAYS[name][0], "shape": list(array.shape)} for name, array in arrays.items()},
    }
    _validate_arrays(arrays, metadata)
    sections = [json.dumps(metadata, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()]
    sections.extend(array.tobytes(order="C") for array in arrays.values())
    sections.extend(proof.values())
    return sections


def _save_locked(result, path, workers, max_cache_bytes):
    if result._provenance is None:
        return SaveResult(False, reason=result.cache_unavailable_reason or "scan did not capture input snapshots")
    for source in (*result.image_paths, *result.label_paths):
        if os.path.abspath(source) == str(path):
            raise ValueError("cache destination overlaps a dataset input")
        if Path(source).name.casefold() == path.name.casefold():
            try:
                if os.path.samefile(source, path):
                    raise ValueError("cache destination aliases a dataset input")
            except FileNotFoundError:
                pass
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        sections = _encode(result)
        _native.write_cache_sections(str(temporary), sections, max_cache_bytes)
        # Validate after serialization, immediately before atomic publication.
        # Even metadata-mode saves require proof of the exact captured bytes.
        _validate_inputs(result, workers, content=True)
        os.replace(temporary, path)
        _native.sync_directory(str(path.parent))
        return SaveResult(True)
    except InputChangedError:
        raise
    except (OSError, _native.CacheFormatError, _native.SnapshotLimitError) as error:
        return SaveResult(False, reason=f"{type(error).__name__}: {error}")
    finally:
        temporary.unlink(missing_ok=True)


def save_cache(result, path, *, workers=4, lock_timeout=30.0, max_cache_bytes=DEFAULT_MAX_CACHE_BYTES):
    """Save captured results, refusing changed inputs before atomic publication."""
    path = Path(path).absolute()
    if path.suffix != ".uydcache":
        raise ValueError("native caches must use .uydcache; legacy .cache files are separate")
    if result._provenance is None:
        return SaveResult(False, reason=result.cache_unavailable_reason or "scan did not capture input snapshots")
    if result._source_cwd is not None and os.getcwd() != result._source_cwd:
        raise InputChangedError("relative-path context changed before cache publication")
    try:
        with _locked(path, lock_timeout):
            existing = load_cache(
                path,
                result.image_paths,
                result.label_paths,
                fingerprint=result.fingerprint_policy,
                workers=workers,
                max_cache_bytes=max_cache_bytes,
                **dict(result.config),
            )
            if isinstance(existing, ScanResult) and existing._provenance == _packed_proof(
                result._provenance, result.num_inputs
            ):
                if result.fingerprint_policy == "metadata":
                    _validate_inputs(result, workers, content=True)
                return SaveResult(True, reused=True)
            return _save_locked(result, path, workers, max_cache_bytes)
    except (OSError, TimeoutError) as error:
        return SaveResult(False, reason=f"{type(error).__name__}: {error}")


def scan_cached(
    path,
    image_paths,
    label_paths,
    *,
    fingerprint="content",
    lock_timeout=30.0,
    max_cache_bytes=DEFAULT_MAX_CACHE_BYTES,
    **scan_options,
):
    """Load or scan under a process lock, rechecking after another writer finishes."""
    path = Path(path).absolute()
    if path.suffix != ".uydcache":
        raise ValueError("native caches must use .uydcache")
    request = _request(image_paths, label_paths, fingerprint, scan_options)
    images, labels, _, cwd, workers = request
    existing = _load_requested(path, request, fingerprint, max_cache_bytes)
    if isinstance(existing, ScanResult):
        return existing
    result = None
    try:
        with _locked(path, lock_timeout):
            existing = _load_requested(path, request, fingerprint, max_cache_bytes)
            if isinstance(existing, ScanResult):
                return existing
            if cwd is not None and os.getcwd() != cwd:
                raise InputChangedError("working directory changed since cache request")
            result = scan(images, labels, fingerprint=fingerprint, **scan_options)
            saved = _save_locked(result, path, workers, max_cache_bytes)
    except (OSError, TimeoutError) as error:
        # A cache-directory/lock failure must not prevent dataset initialization.
        if result is None:
            if cwd is not None and os.getcwd() != cwd:
                raise InputChangedError("working directory changed since cache request")
            result = scan(images, labels, **scan_options)
        saved = SaveResult(False, reason=f"{type(error).__name__}: {error}")
    result.cache_hit, result.cache_path, result.cache_write_error = False, str(path), saved.reason
    return result
