"""Bounded hybrid scan: Pillow verification followed by the Rust label engine."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from PIL import Image, ImageOps

from . import _native, parse_labels
from ._reference import exif_size, verify_image_label

IMG_FORMATS = {"avif", "bmp", "dng", "heic", "heif", "jp2", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp"}
NATIVE_JPEG_PROBE = True  # tests disable it to compare against Pillow-only scans
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}
FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"


class RepairRequired(ValueError):
    """A JPEG would require the explicitly enabled reference repair side effect."""


class ResourceLimitError(ValueError):
    """A configured resource limit prevented scanning; not a corrupt-label status."""


@dataclass(frozen=True)
class Diagnostic:
    source_index: int
    code: str
    message: str


def _check_image(path, repair, *, opener=None, tail=None, save_repaired=None):
    opener = Image.open if opener is None else opener
    with opener(path) as im:
        im.verify()
        w, h = exif_size(im)
        shape = (h, w)
        assert (h > 9) & (w > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"Invalid image format {im.format}. {FORMATS_HELP_MSG}"
        jpeg = im.format.lower() in {"jpg", "jpeg"}
    message = ""
    if jpeg:
        if tail is None:
            with open(path, "rb") as f:
                f.seek(-2, 2)
                damaged = f.read() != b"\xff\xd9"
        else:
            damaged = tail != b"\xff\xd9"
        if damaged:
            if repair == "reject":
                raise RepairRequired(
                    "corrupt JPEG requires repair; explicitly set repair_jpeg='reference' to modify the file"
                )
            with opener(path) as im:
                corrected = ImageOps.exif_transpose(im)
                try:
                    if save_repaired is None:
                        corrected.save(path, "JPEG", subsampling=0, quality=100)
                    else:
                        save_repaired(corrected)
                finally:
                    corrected.close()
            message = f"{path}: corrupt JPEG restored and saved"
    return message, shape


def _verify_image(args):
    path, repair = args
    try:
        message, shape = _check_image(path, repair)
        return shape, message, ""
    except RepairRequired as e:
        return None, str(e), "repair_required"
    except MemoryError:
        raise
    except Exception as e:  # noqa: BLE001 - retain the reference's Pillow failure classification
        return None, str(e), "corrupt_image"


def _immutable(a, dtype, shape=None):
    a = np.asarray(a, dtype=dtype)
    if shape is not None:
        a = a.reshape(shape)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


class ScanResult:
    """Owned immutable packed annotations and deterministic scan diagnostics."""

    def __init__(
        self,
        image_paths,
        source_indices,
        shapes,
        rows,
        segments,
        summary,
        diagnostics,
        fallbacks,
        config,
        *,
        label_paths=(),
        repaired_sources=(),
        provenance=None,
        fingerprint=None,
        cache_unavailable_reason=None,
        source_cwd=None,
    ):
        self.image_paths = tuple(image_paths)
        self.label_paths = tuple(label_paths)
        # Keep successful image repairs independently of diagnostic messages:
        # duplicate/corrupt labels can overwrite the reference's JPEG message.
        # These indices are bookkeeping, NOT proof of a stable content snapshot.
        self.repaired_source_indices = _immutable(repaired_sources, np.int64)
        self._provenance = provenance
        self.fingerprint_policy = fingerprint
        self.cache_unavailable_reason = cache_unavailable_reason
        self._source_cwd = source_cwd
        self.num_inputs = len(image_paths)
        self.num_valid = len(source_indices)
        self.source_indices = _immutable(source_indices, np.int64)
        self.image_shapes = _immutable(shapes, np.int64, (-1, 2))
        offsets = np.r_[np.int64(0), np.cumsum([len(r) for r in rows], dtype=np.int64)]
        self.object_offsets = _immutable(offsets, np.int64)
        labels = np.concatenate(rows) if rows else np.empty((0, 5), np.float32)
        self._labels = _immutable(labels, np.float32, (-1, 5))
        self.classes, self.boxes = self._labels[:, :1], self._labels[:, 1:]
        points, objects = [], []
        for index, per_image in enumerate(segments):
            points.extend(per_image)
            objects.extend(range(int(offsets[index]), int(offsets[index]) + len(per_image)))
        self.segment_points = _immutable(np.concatenate(points) if points else np.empty((0, 2)), np.float32, (-1, 2))
        self.segment_offsets = _immutable(
            np.r_[np.int64(0), np.cumsum([len(p) for p in points], dtype=np.int64)], np.int64
        )
        self.segment_object_indices = _immutable(objects, np.int64)
        self.summary = MappingProxyType(dict(summary))
        self.diagnostics = tuple(diagnostics)
        self.fallback_count = fallbacks
        self.config = MappingProxyType(dict(config))

    @classmethod
    def _from_arrays(cls, image_paths, label_paths, *, arrays, summary, diagnostics, fallbacks, config, source_cwd):
        """Construct from already packed arrays (the probed scan); same fields as __init__."""
        self = object.__new__(cls)
        self.image_paths, self.label_paths = tuple(image_paths), tuple(label_paths)
        self.repaired_source_indices = _immutable([], np.int64)
        self._provenance, self.fingerprint_policy, self.cache_unavailable_reason = None, None, None
        self._source_cwd = source_cwd
        self.num_inputs, self.num_valid = len(image_paths), len(arrays["source_indices"])
        self.source_indices = _immutable(arrays["source_indices"], np.int64)
        self.image_shapes = _immutable(arrays["image_shapes"], np.int64, (-1, 2))
        self.object_offsets = _immutable(arrays["object_offsets"], np.int64)
        self._labels = _immutable(arrays["labels"], np.float32, (-1, 5))
        self.classes, self.boxes = self._labels[:, :1], self._labels[:, 1:]
        self.segment_points = _immutable(arrays["segment_points"], np.float32, (-1, 2))
        self.segment_offsets = _immutable(arrays["segment_offsets"], np.int64)
        self.segment_object_indices = _immutable(arrays["segment_object_indices"], np.int64)
        self.summary = MappingProxyType(dict(summary))
        self.diagnostics = tuple(diagnostics)
        self.fallback_count = fallbacks
        self.config = MappingProxyType(dict(config))
        return self

    def save_cache(self, path, **options):
        """Publish a snapshot-backed native cache; ordinary scans cannot be saved."""
        from ._cache import save_cache

        return save_cache(self, path, **options)

    def to_ultralytics_labels(self):
        return _native.materialize_labels(
            self.image_paths,
            self.source_indices,
            self.image_shapes,
            self.object_offsets,
            self._labels,
            self.segment_points,
            self.segment_offsets,
            self.segment_object_indices,
        )


def _splice(arrays, fallback, results):
    """Packed label arrays with reference results spliced in at fallback positions.

    Contiguous runs of natively parsed files are copied as whole blocks, so a
    few fallback files cost a few slices rather than per-file Python work.
    """
    offsets, rows = arrays["object_offsets"], arrays["labels"]
    objects, bounds, points = arrays["segment_object_indices"], arrays["segment_offsets"], arrays["segment_points"]
    first_segment = np.searchsorted(objects, offsets)
    keep, counts, blocks, lengths, point_blocks, object_blocks = [], [], [], [], [], []
    base = 0

    def take(a, b):
        nonlocal base
        if a >= b:
            return
        r0, r1, s0, s1 = int(offsets[a]), int(offsets[b]), int(first_segment[a]), int(first_segment[b])
        keep.append(np.arange(a, b))
        counts.append(np.diff(offsets[a : b + 1]))
        blocks.append(rows[r0:r1])
        lengths.append(np.diff(bounds[s0 : s1 + 1]))
        point_blocks.append(points[bounds[s0] : bounds[s1]])
        object_blocks.append(objects[s0:s1] - r0 + base)
        base += r1 - r0

    previous = 0
    for position in fallback:
        take(previous, position)
        result = results[position]
        if result is not None:
            row, segments = result[1], result[3]
            keep.append(np.array([position]))
            counts.append(np.array([len(row)]))
            blocks.append(row)
            if segments:
                lengths.append(np.array([len(s) for s in segments]))
                point_blocks.append(np.concatenate(segments))
                object_blocks.append(base + np.arange(len(segments)))
            base += len(row)
        previous = position + 1
    take(previous, len(offsets) - 1)

    def cat(parts, dtype, width=None):
        shape = (-1, width) if width else (-1,)
        parts = [np.asarray(p, dtype=dtype).reshape(shape) for p in parts]
        return np.concatenate(parts) if parts else np.empty((0, width) if width else (0,), dtype=dtype)

    return {
        "keep": cat(keep, np.int64),
        "object_offsets": np.r_[np.int64(0), np.cumsum(cat(counts, np.int64), dtype=np.int64)],
        "labels": cat(blocks, np.float32, 5),
        "segment_offsets": np.r_[np.int64(0), np.cumsum(cat(lengths, np.int64), dtype=np.int64)],
        "segment_points": cat(point_blocks, np.float32, 2),
        "segment_object_indices": cat(object_blocks, np.int64),
    }


def _scan_probed(images, labels, probe, *, num_classes, task, single_cls, workers, max_file_bytes, prefix, config, cwd):
    """Scan with probed images and one label parse; per-file Python only for exceptions.

    Used with repair_jpeg='reject', where Pillow has no side effects, so
    verifying deferred images before parsing labels cannot change any file.
    """
    statuses, heights, widths = probe
    count = len(images)
    shapes = np.empty((count, 2), dtype=np.int64)
    shapes[:, 0], shapes[:, 1] = heights, widths
    deferred = [i for i, status in enumerate(statuses) if status]
    broken = {}
    if deferred:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            checked = pool.map(_verify_image, [(images[i], "reject") for i in deferred])
            for i, (shape, message, error) in zip(deferred, checked):
                if error:
                    broken[i] = (error, message)
                else:
                    shapes[i] = shape
    usable = np.ones(count, dtype=bool)
    usable[list(broken)] = False
    valid = np.flatnonzero(usable)
    packed = parse_labels(
        [labels[i] for i in valid.tolist()],
        num_classes=num_classes,
        task=task,
        single_cls=single_cls,
        workers=workers,
        max_file_bytes=max_file_bytes,
    )
    arrays = packed.to_arrays()
    del packed
    status = arrays["statuses"]
    limited = np.flatnonzero(status == 4)
    if len(limited):
        raise ResourceLimitError(f"label exceeds max_file_bytes={max_file_bytes}: {labels[int(valid[limited[0]])]}")
    diagnostics = [
        Diagnostic(i, error, f"{prefix}{images[i]}: ignoring corrupt image/label: {message}")
        for i, (error, message) in broken.items()
    ]
    regular = status != 3
    summary = {
        "found": int(np.count_nonzero(regular & (status != 1))),
        "missing": int(np.count_nonzero(status == 1)),
        "empty": int(np.count_nonzero(status == 2)),
        "corrupt": len(broken),
        "total": count,
    }
    duplicates = arrays["duplicates"]
    for position in np.flatnonzero(regular & (duplicates > 0)).tolist():
        source = int(valid[position])
        message = f"{prefix}{images[source]}: {int(duplicates[position])} duplicate labels removed"
        diagnostics.append(Diagnostic(source, "duplicate", message))
    fallback = np.flatnonzero(~regular).tolist()
    results = {}
    for position in fallback:
        source = int(valid[position])
        shape = (int(shapes[source, 0]), int(shapes[source, 1]))
        result = verify_image_label(
            (images[source], labels[source], prefix, False, num_classes, 0, 0, single_cls),
            check_image=lambda _, verified_image=("", shape): verified_image,
        )
        for name, value in zip(("missing", "found", "empty", "corrupt"), result[5:9]):
            summary[name] += value
        if result[-1]:
            diagnostics.append(Diagnostic(source, "reference_label", result[-1]))
        results[position] = None if result[0] is None else result
    if fallback:
        spliced = _splice(arrays, fallback, results)
        valid = valid[spliced.pop("keep")]
        arrays = spliced
    arrays["source_indices"] = valid
    arrays["image_shapes"] = shapes[valid]
    diagnostics.sort(key=lambda d: d.source_index)
    return ScanResult._from_arrays(
        images,
        labels,
        arrays=arrays,
        summary=summary,
        diagnostics=diagnostics,
        fallbacks=len(fallback),
        config=config,
        source_cwd=cwd,
    )


def scan(
    image_paths,
    label_paths,
    *,
    num_classes,
    task="detect",
    single_cls=False,
    image_validation="pillow",
    repair_jpeg="reject",
    workers=4,
    max_in_flight=256,
    max_file_bytes=16 * 1024**2,
    prefix="",
    fingerprint=None,
    max_image_bytes=64 * 1024**2,
    max_fallback_bytes=64 * 1024**2,
):
    """Verify images and parse labels without changing discovery or path order.

    Default JPEG policy rejects files requiring repair. 'reference' explicitly
    enables the original save-to-input behavior. Cache operations are separate.
    """
    images = [os.fsdecode(os.fspath(p)) for p in image_paths]
    labels = [os.fsdecode(os.fspath(p)) for p in label_paths]
    source_cwd = os.getcwd() if any(not os.path.isabs(p) for p in images + labels) else None
    if len(images) != len(labels):
        raise ValueError("image_paths and label_paths must have equal lengths")
    if task not in ("detect", "segment") or image_validation != "pillow":
        raise ValueError("only detect/segment tasks with Pillow verification are supported")
    if repair_jpeg not in ("reject", "reference"):
        raise ValueError("repair_jpeg must be reject or reference")
    if not 1 <= workers <= 256 or not 1 <= max_in_flight <= 65536 or num_classes < 1:
        raise ValueError("invalid workers, max_in_flight, or num_classes")
    if min(max_file_bytes, max_image_bytes, max_fallback_bytes) < 1:
        raise ValueError("file/image/fallback byte limits must be positive")
    if fingerprint not in (None, "content", "metadata"):
        raise ValueError("fingerprint must be None, content, or metadata")
    if fingerprint:
        from . import _snapshots
    image_proofs, label_proofs, unavailable = [], [], []
    summary = {"found": 0, "missing": 0, "empty": 0, "corrupt": 0, "total": len(images)}
    diagnostics, sources, shapes, rows, all_segments, repaired_sources = [], [], [], [], [], []
    fallbacks = 0
    probe = None
    limit = Image.MAX_IMAGE_PIXELS
    if not fingerprint and NATIVE_JPEG_PROBE and (limit is None or limit > 0):
        # Pillow's JPEG marker loop is pure Python under the GIL. One native pass
        # probes every header; Pillow runs only where the probe cannot guarantee
        # Pillow's own result (see src/jpeg.rs).
        probe = _native.probe_jpegs(images, workers, limit or 0, max_image_bytes)
    if probe is not None and repair_jpeg == "reject":
        config = {
            "num_classes": num_classes,
            "task": task,
            "single_cls": single_cls,
            "image_validation": image_validation,
            "repair_jpeg": repair_jpeg,
            "max_file_bytes": max_file_bytes,
            "prefix": prefix,
            "max_image_bytes": max_image_bytes,
            "max_fallback_bytes": max_fallback_bytes,
        }
        return _scan_probed(
            images,
            labels,
            probe,
            num_classes=num_classes,
            task=task,
            single_cls=single_cls,
            workers=workers,
            max_file_bytes=max_file_bytes,
            prefix=prefix,
            config=config,
            cwd=source_cwd,
        )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(images), max_in_flight):
            stop = min(start + max_in_flight, len(images))
            # Pillow and Rayon stages run sequentially under the same CPU budget.
            if fingerprint:
                captured_images = list(
                    pool.map(_snapshots.verify_image, [(p, repair_jpeg, max_image_bytes) for p in images[start:stop]])
                )
                verified = [v[:3] for v in captured_images]
                image_proofs.extend(v[3] for v in captured_images)
                unavailable.extend(v[4] for v in captured_images if v[4])
            elif probe is not None:
                statuses, heights, widths = probe
                todo = [i for i in range(start, stop) if statuses[i]]
                pillow = dict(zip(todo, pool.map(_verify_image, [(images[i], repair_jpeg) for i in todo])))
                verified = [pillow[i] if statuses[i] else ((heights[i], widths[i]), "", "") for i in range(start, stop)]
            else:
                verified = list(pool.map(_verify_image, [(p, repair_jpeg) for p in images[start:stop]]))
            valid = [start + i for i, result in enumerate(verified) if result[0] is not None]
            captured_labels = None
            if fingerprint:
                try:
                    captured_labels = _native.snapshot_labels(
                        [labels[i] for i in valid],
                        num_classes,
                        single_cls,
                        workers,
                        max_file_bytes,
                        max_fallback_bytes,
                    )
                except (OSError, _native.SnapshotLimitError) as error:
                    unavailable.append(str(error))
            if captured_labels is not None:
                packed = captured_labels
                proofs = json.loads(captured_labels.fingerprints_json())
                indexed = dict(zip(valid, proofs))
                invalid = [start + i for i, v in enumerate(verified) if v[0] is None]
                try:
                    indexed.update(
                        zip(invalid, _snapshots.fingerprints([labels[i] for i in invalid], max_file_bytes, workers))
                    )
                except (OSError, _native.SnapshotLimitError) as error:
                    unavailable.append(str(error))
                label_proofs.extend(indexed.get(i) for i in range(start, stop))
            else:
                packed = parse_labels(
                    [labels[i] for i in valid],
                    num_classes=num_classes,
                    task=task,
                    single_cls=single_cls,
                    workers=workers,
                    max_file_bytes=max_file_bytes,
                )
                if fingerprint:
                    label_proofs.extend([None] * (stop - start))
            arrays = packed.to_arrays()
            del packed
            position = 0
            segment_index = 0
            for local, (shape, image_message, error) in enumerate(verified):
                source = start + local
                path = images[source]
                if error:
                    summary["corrupt"] += 1
                    diagnostics.append(
                        Diagnostic(source, error, f"{prefix}{path}: ignoring corrupt image/label: {image_message}")
                    )
                    continue
                if image_message:
                    repaired_sources.append(source)
                status = int(arrays["statuses"][position])
                begin, end = map(int, arrays["object_offsets"][position : position + 2])
                native_segments = []
                while (
                    segment_index < len(arrays["segment_object_indices"])
                    and arrays["segment_object_indices"][segment_index] < end
                ):
                    a, b = arrays["segment_offsets"][segment_index : segment_index + 2]
                    native_segments.append(arrays["segment_points"][a:b])
                    segment_index += 1
                duplicate_count = int(arrays["duplicates"][position])
                position += 1
                if status == 4:
                    raise ResourceLimitError(f"label exceeds max_file_bytes={max_file_bytes}: {labels[source]}")
                if status == 3:
                    fallbacks += 1
                    extra = {}
                    if captured_labels is not None:
                        # Use the exact bytes already read and hashed by Rust.
                        # Never reopen a potentially changed path for fallback.
                        data = captured_labels.fallback_bytes(position - 1)
                        extra = {"label_isfile": lambda _: True, "open_label": _snapshots.fallback_open(data)}
                    result = verify_image_label(
                        (path, labels[source], prefix, False, num_classes, 0, 0, single_cls),
                        check_image=lambda _, verified_image=(image_message, shape): verified_image,
                        **extra,
                    )
                    for name, value in zip(("missing", "found", "empty", "corrupt"), result[5:9]):
                        summary[name] += value
                    if result[-1]:
                        diagnostics.append(Diagnostic(source, "reference_label", result[-1]))
                    if result[0] is None:
                        continue
                    row, native_segments = result[1], result[3]
                else:
                    row = arrays["labels"][begin:end]
                    summary["missing" if status == 1 else "found"] += 1
                    summary["empty"] += status == 2
                    message = f"{prefix}{image_message}" if image_message else ""
                    if duplicate_count:
                        message = f"{prefix}{path}: {duplicate_count} duplicate labels removed"
                    if message:
                        diagnostics.append(
                            Diagnostic(source, "duplicate" if duplicate_count else "jpeg_repaired", message)
                        )
                sources.append(source)
                shapes.append(shape)
                rows.append(row)
                all_segments.append(native_segments)
    config = {
        "num_classes": num_classes,
        "task": task,
        "single_cls": single_cls,
        "image_validation": image_validation,
        "repair_jpeg": repair_jpeg,
        "max_file_bytes": max_file_bytes,
        "prefix": prefix,
        "max_image_bytes": max_image_bytes,
        "max_fallback_bytes": max_fallback_bytes,
    }
    if fingerprint and source_cwd is not None and os.getcwd() != source_cwd:
        raise _native.InputChangedError("working directory changed during relative-path scan")
    return ScanResult(
        images,
        sources,
        shapes,
        rows,
        all_segments,
        summary,
        diagnostics,
        fallbacks,
        config,
        label_paths=labels,
        repaired_sources=repaired_sources,
        provenance={"images": image_proofs, "labels": label_proofs} if fingerprint and not unavailable else None,
        fingerprint=fingerprint,
        cache_unavailable_reason="; ".join(dict.fromkeys(unavailable)) or None,
        source_cwd=source_cwd,
    )
