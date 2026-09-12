"""Bounded hybrid scan: Pillow verification followed by the Rust label engine."""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from PIL import Image, ImageOps

from . import parse_labels
from ._reference import exif_size, verify_image_label

IMG_FORMATS = {"avif", "bmp", "dng", "heic", "heif", "jp2", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp"}
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


def _check_image(path, repair):
    with Image.open(path) as im:
        im.verify()
        w, h = exif_size(im)
        shape = (h, w)
        assert (h > 9) & (w > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"Invalid image format {im.format}. {FORMATS_HELP_MSG}"
        jpeg = im.format.lower() in {"jpg", "jpeg"}
    message = ""
    if jpeg:
        with open(path, "rb") as f:
            f.seek(-2, 2)
            damaged = f.read() != b"\xff\xd9"
        if damaged:
            if repair == "reject":
                raise RepairRequired(
                    "corrupt JPEG requires repair; explicitly set repair_jpeg='reference' to modify the file"
                )
            with Image.open(path) as im:
                corrected = ImageOps.exif_transpose(im)
                try:
                    corrected.save(path, "JPEG", subsampling=0, quality=100)
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
    ):
        self.image_paths = tuple(image_paths)
        self.label_paths = tuple(label_paths)
        # Keep successful image repairs independently of diagnostic messages:
        # duplicate/corrupt labels can overwrite the reference's JPEG message.
        # These indices are bookkeeping, NOT proof of a stable content snapshot.
        self.repaired_source_indices = _immutable(repaired_sources, np.int64)
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

    def to_ultralytics_labels(self):
        labels = []
        segment_index = 0
        for index, source in enumerate(self.source_indices):
            begin, end = map(int, self.object_offsets[index : index + 2])
            segments = []
            while segment_index < len(self.segment_object_indices) and self.segment_object_indices[segment_index] < end:
                a, b = self.segment_offsets[segment_index : segment_index + 2]
                segments.append(self.segment_points[a:b].copy())
                segment_index += 1
            labels.append(
                {
                    "im_file": self.image_paths[int(source)],
                    "shape": tuple(map(int, self.image_shapes[index])),
                    "cls": self.classes[begin:end].copy(),
                    "bboxes": self.boxes[begin:end].copy(),
                    "segments": segments,
                    "keypoints": None,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        return labels


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
):
    """Verify images and parse labels without changing discovery or path order.

    Default JPEG policy rejects files requiring repair. 'reference' explicitly
    enables the original save-to-input behavior. Cache operations are separate.
    """
    images = [os.fsdecode(os.fspath(p)) for p in image_paths]
    labels = [os.fsdecode(os.fspath(p)) for p in label_paths]
    if len(images) != len(labels):
        raise ValueError("image_paths and label_paths must have equal lengths")
    if task not in ("detect", "segment") or image_validation != "pillow":
        raise ValueError("only detect/segment tasks with Pillow verification are supported")
    if repair_jpeg not in ("reject", "reference"):
        raise ValueError("repair_jpeg must be reject or reference")
    if not 1 <= workers <= 256 or not 1 <= max_in_flight <= 65536 or num_classes < 1:
        raise ValueError("invalid workers, max_in_flight, or num_classes")
    if max_file_bytes < 1:
        raise ValueError("max_file_bytes must be positive")
    summary = {"found": 0, "missing": 0, "empty": 0, "corrupt": 0, "total": len(images)}
    diagnostics, sources, shapes, rows, all_segments, repaired_sources = [], [], [], [], [], []
    fallbacks = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(images), max_in_flight):
            stop = min(start + max_in_flight, len(images))
            # Pillow and Rayon stages run sequentially under the same CPU budget.
            verified = list(pool.map(_verify_image, [(p, repair_jpeg) for p in images[start:stop]]))
            valid = [start + i for i, result in enumerate(verified) if result[0] is not None]
            packed = parse_labels(
                [labels[i] for i in valid],
                num_classes=num_classes,
                task=task,
                single_cls=single_cls,
                workers=workers,
                max_file_bytes=max_file_bytes,
            )
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
                    result = verify_image_label(
                        (path, labels[source], prefix, False, num_classes, 0, 0, single_cls),
                        check_image=lambda _, verified_image=(image_message, shape): verified_image,
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
    }
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
    )
