"""Explicit integration for the pinned base Detection/Segmentation dataset.

The native backend validates captured content; the optional legacy backend
retains Ultralytics' weaker path/size invalidation and trusted pickle bridge.
The fast backend validates every image and label file's kind, size and
modification time plus the ordered path list, keeps annotations packed in a
memory-mapped cache, and creates per-image label dictionaries on access.
"""

import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
import pickle
import stat
import tempfile
import threading
from collections.abc import Sequence
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import PIL
from PIL import Image
from ultralytics.data import base, build, dataset, utils
from ultralytics.data.base import BaseDataset
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr, ops, patches

from . import _fast, _native, scan

_PROFILE = json.loads(Path(__file__).with_name("_ultralytics_profile.json").read_text())
_BASE = dataset.YOLODataset
_LEGACY_GC_LOCK = threading.RLock()


def _reset_legacy_gc_lock():
    global _LEGACY_GC_LOCK
    _LEGACY_GC_LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_legacy_gc_lock)


class UnsupportedProfile(RuntimeError):
    """The caller must retain its original dataset implementation."""


def check_profile():
    """Reject source drift and replaced runtime hooks before discovery or writes."""
    if np.__version__ != _PROFILE["numpy"] or PIL.__version__ != _PROFILE["pillow"]:
        raise UnsupportedProfile("adapter requires NumPy 2.4.4 and Pillow 12.1.1")
    try:
        heif_version = importlib.metadata.version("pi-heif")
    except importlib.metadata.PackageNotFoundError:
        heif_version = None
    if heif_version != _PROFILE["pi_heif"]:
        raise UnsupportedProfile("full image/error profile requires pi-heif 1.4.0")
    if Image.open is not patches.image_open:
        raise UnsupportedProfile("overridden Pillow image opener")
    objects = {"BaseDataset": BaseDataset, "image_open": patches.image_open, "pillow_open": patches._image_open}
    for name in _PROFILE["sources"]:
        if name.startswith("YOLODataset."):
            objects[name] = getattr(dataset.YOLODataset, name.split(".")[1], None)
        elif name not in objects:
            objects[name] = next((getattr(m, name) for m in (dataset, utils, build, ops) if hasattr(m, name)), None)
    for name, expected in _PROFILE["sources"].items():
        try:
            actual = hashlib.sha256(inspect.getsource(objects[name]).encode()).hexdigest()
        except (OSError, TypeError):
            actual = None
        if actual != expected:
            raise UnsupportedProfile(f"unsupported Ultralytics source: {name}")
    # A class's source text alone cannot detect assigning a different function
    # to a runtime module alias. Check the aliases used by inherited methods too.
    for name in (
        "get_hash",
        "img2label_paths",
        "verify_image_label",
        "load_dataset_cache_file",
        "save_dataset_cache_file",
    ):
        if getattr(dataset, name) is not getattr(utils, name):
            raise UnsupportedProfile(f"overridden Ultralytics runtime alias: {name}")
    if dataset.YOLODataset is not _BASE or build.YOLODataset is not _BASE:
        raise UnsupportedProfile("overridden YOLODataset class")


def _read_trusted_legacy(path, expected_hash, expected_profile, expected_images):
    """Only reached after an explicit trust opt-in; pickle is not sandboxed."""
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("legacy cache is not a regular file")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise ValueError("legacy cache must be owned by the current user")
    # The helper toggles process-global GC. Serialize our callers so a second
    # load cannot snapshot a temporary disabled state and restore it last.
    # Reset this process-local lock after fork rather than inheriting a lock
    # potentially held by a different parent thread.
    with _LEGACY_GC_LOCK:
        enabled = gc.isenabled()
        try:
            value = utils.load_dataset_cache_file(path)
        finally:
            gc.enable() if enabled else gc.disable()
    if not isinstance(value, dict) or value.get("version") != dataset.DATASET_CACHE_VERSION:
        raise ValueError("unsupported legacy cache version")
    if value.get("hash") != expected_hash:
        raise ValueError("legacy cache hash mismatch")
    if value.get("ultrafast_profile") != expected_profile:
        raise ValueError("legacy cache scan configuration is unknown or different; rebuilding")
    if not isinstance(value.get("labels"), list) or not isinstance(value.get("msgs"), list):
        raise ValueError("invalid legacy cache schema")  # noqa: TRY004 - malformed cache is a cache miss
    counters = value.get("results")
    if not isinstance(counters, (tuple, list)) or len(counters) != 5:
        raise ValueError("invalid legacy cache counters")
    if any(type(n) is not int or n < 0 for n in counters) or any(n > counters[-1] for n in counters):
        raise ValueError("invalid legacy cache counter values")
    if len(value["labels"]) != counters[-1] - counters[3]:
        raise ValueError("legacy cache valid/corrupt count mismatch")
    if counters[-1] != len(expected_images) or any(not isinstance(message, str) for message in value["msgs"]):
        raise ValueError("invalid legacy cache inputs/messages")
    ordered_images = iter(expected_images)
    for label in value["labels"]:
        if not isinstance(label, dict) or not isinstance(label.get("im_file"), str):
            raise ValueError("invalid legacy label record")  # noqa: TRY004 - malformed cache is a cache miss
        for name, width in (("cls", 1), ("bboxes", 4)):
            array = label.get(name)
            if (
                not isinstance(array, np.ndarray)
                or array.dtype != np.float32
                or array.ndim != 2
                or array.shape[1] != width
            ):
                raise ValueError("invalid legacy label array")
        if len(label["cls"]) != len(label["bboxes"]):
            raise ValueError("invalid legacy label alignment")
        if not any(image == label["im_file"] for image in ordered_images):
            raise ValueError("legacy cache path order mismatch")
        shape = label.get("shape")
        if (
            not isinstance(shape, (tuple, list))
            or len(shape) != 2
            or any(not isinstance(n, int) or n < 10 for n in shape)
        ):
            raise ValueError("invalid legacy image shape")
        if (
            label.get("keypoints") is not None
            or label.get("normalized") is not True
            or label.get("bbox_format") != "xywh"
        ):
            raise ValueError("unsupported legacy label schema")
        segments = label.get("segments")
        if not isinstance(segments, list) or (segments and len(segments) != len(label["cls"])):
            raise ValueError("invalid legacy segment alignment")
        if any(
            not isinstance(s, np.ndarray) or s.dtype != np.float32 or s.ndim != 2 or s.shape[1] != 2 for s in segments
        ):
            raise ValueError("invalid legacy segment array")
    return value


def _write_legacy(path, value, prefix):
    """Use the pinned helper on a sibling temporary, then atomically publish."""
    temporary = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(descriptor)
        temporary = Path(name)
        utils.save_dataset_cache_file(prefix, temporary, value, dataset.DATASET_CACHE_VERSION)
        if not temporary.exists() or temporary.stat().st_size == 0:
            return False
        # Windows FlushFileBuffers (via os.fsync) requires write access.
        # Reopen without truncating the complete cache written by the helper.
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return True
    except OSError as error:
        LOGGER.warning(f"{prefix}Cache write failed; using scanned labels: {error}")
        return False
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class _NpyFiles(Sequence):
    """``[Path(f).with_suffix(".npy") for f in im_files]``, built per access.

    The reference builds all of these Path objects in its constructor although
    only disk-cached samples (and one existence check per loaded image) use them.
    """

    __slots__ = ("_files",)

    def __init__(self, files):
        self._files = files

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [Path(f).with_suffix(".npy") for f in self._files[index]]
        return Path(self._files[index]).with_suffix(".npy")


class FastYOLODataset(_BASE):
    """Base YOLODataset with an explicit hybrid scan and optional legacy bridge.

    annotation_cache='fast' indexes files natively, validates a packed cache by
    per-file metadata and returns labels as an on-demand sequence.
    Default annotation_cache='none' scans without reading/writing label caches.
    'ultralytics' requires trust_legacy_cache=True because upstream .cache uses
    pickle. Image RAM/disk caching remains controlled by the original `cache` arg.
    JPEG repair requires explicit repair_jpeg='reference'; default is 'reject'.
    """

    def __init__(
        self,
        *args,
        task="detect",
        scan_workers=4,
        max_in_flight=256,
        max_file_bytes=16 * 1024**2,
        repair_jpeg="reject",
        annotation_cache="none",
        trust_legacy_cache=False,
        cache_dir=None,
        cache_fingerprint="content",
        cache_lock_timeout=30.0,
        max_cache_bytes=2 * 1024**3,
        max_image_bytes=64 * 1024**2,
        max_fallback_bytes=64 * 1024**2,
        index_workers=16,
        **kwargs,
    ):
        check_profile()
        if type(self) is not FastYOLODataset or task not in ("detect", "segment"):
            raise UnsupportedProfile("only the base Detection/Segmentation dataset is supported; retain custom hooks")
        if annotation_cache not in ("none", "ultralytics", "native", "fast"):
            raise ValueError("annotation_cache must be none, fast, native, or ultralytics")
        if not 1 <= index_workers <= 256:
            raise ValueError("index_workers must be in 1..256")
        if annotation_cache == "ultralytics" and not trust_legacy_cache:
            raise ValueError("legacy pickle caches require explicit trust_legacy_cache=True")
        if not 1 <= scan_workers <= 256 or not 1 <= max_in_flight <= 65536 or max_file_bytes < 1:
            raise ValueError("invalid scan worker, queue, or file size limit")
        if repair_jpeg not in ("reject", "reference"):
            raise ValueError("repair_jpeg must be reject or reference")
        if cache_fingerprint not in ("content", "metadata"):
            raise ValueError("cache_fingerprint must be content or metadata")
        self._scan_options = {
            "workers": scan_workers,
            "max_in_flight": max_in_flight,
            "max_file_bytes": max_file_bytes,
            "repair_jpeg": repair_jpeg,
            "max_image_bytes": max_image_bytes,
            "max_fallback_bytes": max_fallback_bytes,
        }
        self._native_cache_options = {
            "fingerprint": cache_fingerprint,
            "lock_timeout": cache_lock_timeout,
            "max_cache_bytes": max_cache_bytes,
        }
        self._native_cache_dir = None if cache_dir is None else Path(cache_dir).absolute()
        self.annotation_cache = annotation_cache
        self.scan_evidence = None
        self.annotation_cache_hit = False
        self.annotation_cache_write_ok = None
        self.annotation_cache_miss_reason = None
        self.annotation_cache_write_error = None
        self.annotation_cache_path = None
        # Metadata lookups are latency-bound (kernel inode fetches), not
        # CPU-bound, so this pool is sized independently of scan_workers.
        self._index_workers = index_workers
        self._fast_index = None
        self.annotation_cache_digest = None
        if annotation_cache == "fast":
            self._yolo_init(*args, task=task, **kwargs)
        else:
            super().__init__(*args, task=task, **kwargs)

    def _yolo_init(self, *args, data=None, task="detect", **kwargs):
        """Pinned YOLODataset.__init__; check_profile guards its source hash."""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        self._base_init(*args, channels=self.data.get("channels", 3), **kwargs)

    def _base_init(
        self,
        img_path,
        imgsz=640,
        cache=False,
        augment=True,
        hyp=DEFAULT_CFG,
        prefix="",
        rect=False,
        batch_size=16,
        stride=32,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
        channels=3,
    ):
        """Pinned BaseDataset.__init__ with on-demand .npy paths.

        check_profile guards the BaseDataset source hash. The only change is
        npy_files: the reference eagerly builds one Path per image here.
        """
        super(BaseDataset, self).__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = base.get_split_fraction(fraction, "train")
        self.channels = channels
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache images (options are cache = True, False, None, "ram", "disk")
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = _NpyFiles(self.im_files)
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if self.cache == "ram" and self.check_cache_ram():
            if hyp.deterministic:
                LOGGER.warning(
                    "cache='ram' may produce non-deterministic training results. "
                    "Consider cache='disk' as a deterministic alternative if your disk space allows."
                )
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def _path_rules(self):
        return f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"

    def get_img_files(self, img_path):
        if self.annotation_cache != "fast":
            return super().get_img_files(img_path)
        index = None
        if os.sep == "/" and isinstance(img_path, (str, os.PathLike)) and os.path.isdir(img_path):
            # One native pass lists every directory together with metadata.
            index = _native.discover_dataset(str(Path(img_path)), sorted(base.IMG_FORMATS), self._index_workers)
        if index is None or not len(index):
            # Reference discovery (file lists, several roots, errors), then index its list.
            im_files = super().get_img_files(img_path)
            self._fast_index = _native.index_paths(im_files, self._index_workers, *self._path_rules())
            return im_files
        im_files = index.image_paths()
        count = self.fraction if isinstance(self.fraction, int) else max(1, round(len(im_files) * self.fraction))
        if count < len(im_files):
            im_files, index = im_files[:count], index.truncated(count)
        base.check_file_speeds(im_files, prefix=self.prefix)  # check image read speeds
        self._fast_index = index
        return im_files

    def _fast_cache_path(self, legacy_path):
        path = Path(legacy_path).absolute()
        if self._native_cache_dir is None:
            return path.with_suffix(".uydfast")
        key = hashlib.sha256(str(path).encode()).hexdigest()[:20]
        return self._native_cache_dir / f"{path.stem}-{key}.uydfast"

    def _fast_config(self):
        return {
            **self._cache_profile(),
            "max_image_bytes": self._scan_options["max_image_bytes"],
            "max_fallback_bytes": self._scan_options["max_fallback_bytes"],
            "validation": "ordered-paths+per-file-kind-size-mtime-v1",
            "path_rules": list(self._path_rules()),
        }

    def _scan_kwargs(self):
        return dict(
            num_classes=len(self.data["names"]),
            task="segment" if self.use_segments else "detect",
            single_cls=self.single_cls,
            prefix=self.prefix,
            **self._scan_options,
        )

    def _fast_labels(self):
        index = self._fast_index
        self.label_files = index.label_paths()
        legacy = Path(self.label_files[0]).parent.with_suffix(".cache")
        path = self._fast_cache_path(legacy)
        self.annotation_cache_path = str(path)
        digest, config = index.digest(), self._fast_config()
        self.annotation_cache_digest = digest
        try:
            metadata, arrays = _fast.load(path, digest=digest, config=config, workers=self._index_workers)
        except _fast.CacheMiss as miss:
            self.annotation_cache_miss_reason = str(miss)
            summary, messages, arrays = self._fast_scan(legacy, path, digest, config)
        else:
            self.annotation_cache_hit = True
            summary = metadata["summary"]
            messages = [item["message"] for item in metadata["diagnostics"]]
            if dataset.LOCAL_RANK in {-1, 0}:
                nf, nm, ne, nc, total = (summary[k] for k in ("found", "missing", "empty", "corrupt", "total"))
                message = f"Scanning {path}... {self.scan_summary(nf, nm, ne, nc)}"
                dataset.TQDM(None, desc=self.prefix + message, total=total, initial=total)
                if messages:
                    LOGGER.info("\n".join(messages))
        labels = _fast.LazyLabels(list(self.im_files), arrays)
        if not len(labels):
            issues = "\n  ".join(sorted(set(messages))) or "no error details"
            raise RuntimeError(f"No valid images found in {legacy}.\n  {issues}\n{dataset.HELP_URL}")
        sources = arrays["sources"]
        if len(sources) != len(self.im_files):
            self.im_files = [self.im_files[i] for i in sources.tolist()]  # update im_files
        self._fast_index = None  # only needed during construction; not picklable for loader workers
        return self._verify_packed(labels, legacy)

    def _fast_scan(self, legacy, path, digest, config):
        result = scan(self.im_files, self.label_files, **self._scan_kwargs())
        diagnostics = [vars(d) for d in result.diagnostics]
        summary = dict(result.summary)
        self.scan_evidence = {"summary": summary, "fallback_count": result.fallback_count, "diagnostics": diagnostics}
        messages = [d["message"] for d in diagnostics]
        if messages:
            LOGGER.info("\n".join(messages))
        if summary["found"] == 0:
            message = f"{self.prefix}No labels found in {legacy}. {dataset.HELP_URL}"
            if self.augment:
                raise ValueError(message)
            LOGGER.warning(message)
        fallback_count = result.fallback_count
        arrays = _fast.arrays_from_scan(result)
        del result
        if len(arrays["sources"]):  # the reference saves no cache without labels
            # Publish only if no input changed while it was being scanned.
            after = _native.index_paths(self.im_files, self._index_workers, *self._path_rules())
            if after.digest() != digest:
                self.annotation_cache_write_error = "inputs changed during the scan"
            else:
                self.annotation_cache_write_error = _fast.save(
                    path,
                    digest=digest,
                    config=config,
                    summary=summary,
                    diagnostics=diagnostics,
                    fallback_count=fallback_count,
                    arrays=arrays,
                    workers=self._index_workers,
                )
            self.annotation_cache_write_ok = self.annotation_cache_write_error is None
            if self.annotation_cache_write_error:
                LOGGER.warning(
                    f"{self.prefix}Fast cache not written; using scanned labels: {self.annotation_cache_write_error}"
                )
        return summary, messages, arrays

    def _verify_packed(self, labels, cache_path):
        """Pinned YOLODataset.verify_labels computed from packed totals."""
        len_boxes, len_segments = labels.counts()
        len_cls = len_boxes
        if self.use_segments and len_boxes != len_segments:
            raise ValueError(
                f"Segment dataset requires equal numbers of boxes and segments, but got len(segments) = "
                f"{len_segments}, len(boxes) = {len_boxes}. Please supply a segment dataset, not a detect dataset."
            )
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            labels = labels.without_segments()
        if len_cls == 0:
            LOGGER.warning(
                f"Labels are missing or empty in {cache_path}, training may not work correctly. {dataset.HELP_URL}"
            )
        return labels

    def update_labels(self, include_class):
        if isinstance(self.labels, _fast.LazyLabels):
            if include_class is None:
                # Without a class filter the reference loop only zeroes single_cls
                # classes; do that with one vectorized update of the packed rows.
                if self.single_cls:
                    self.labels = self.labels.with_zero_classes()
                return
            self.labels = self.labels.materialize()  # in-place reference update needs real dicts
        super().update_labels(include_class)

    def set_rectangle(self):
        if isinstance(self.labels, _fast.LazyLabels):
            self.labels = self.labels.materialize()  # the reference pops "shape" in place
        super().set_rectangle()

    def _load_or_scan_cache(self, cache_path, cache_hash):
        if self.annotation_cache == "native":
            value = self.cache_labels(cache_path)
            return value, self.annotation_cache_hit
        if self.annotation_cache == "ultralytics":
            try:
                value = _read_trusted_legacy(cache_path, cache_hash, self._cache_profile(), self.im_files)
            except (OSError, ValueError, EOFError, AttributeError, ImportError, pickle.UnpicklingError) as error:
                self.annotation_cache_miss_reason = f"{type(error).__name__}: {error}"
            else:
                self.annotation_cache_hit = True
                return value, True
        return self.cache_labels(cache_path), False

    def get_labels(self):
        if self.annotation_cache == "fast":
            return self._fast_labels()
        if self.annotation_cache != "native":
            return super().get_labels()
        # Retain the pinned get_labels tail, while avoiding its redundant
        # legacy path/size hashes. The native cache performs full validation.
        files = self.get_label_files()
        path = Path(files[0]).parent.with_suffix(".cache")
        value, exists = self._load_or_scan_cache(path, None)
        nf, nm, ne, nc, total = value["results"]
        if exists and dataset.LOCAL_RANK in {-1, 0}:
            message = f"Scanning {self.annotation_cache_path}... {self.scan_summary(nf, nm, ne, nc)}"
            dataset.TQDM(None, desc=self.prefix + message, total=total, initial=total)
            if value["msgs"]:
                LOGGER.info("\n".join(value["msgs"]))
        labels = value["labels"]
        if not labels:
            issues = "\n  ".join(sorted(set(value["msgs"]))) or "no error details"
            raise RuntimeError(f"No valid images found in {path}.\n  {issues}\n{dataset.HELP_URL}")
        self.im_files = [label["im_file"] for label in labels]
        self.verify_labels(labels, path)
        return labels

    def _native_cache_path(self, legacy_path):
        path = Path(legacy_path).absolute()
        if self._native_cache_dir is None:
            return path.with_suffix(".uydcache")
        key = hashlib.sha256(str(path).encode()).hexdigest()[:20]
        return self._native_cache_dir / f"{path.stem}-{key}.uydcache"

    def _cache_profile(self):
        return {
            "reference_sha": _PROFILE["reference_sha"],
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "pi_heif": _PROFILE["pi_heif"],
            "parser": "0.1.0a1-hybrid-2",
            "num_classes": len(self.data["names"]),
            "task": "segment" if self.use_segments else "detect",
            "single_cls": self.single_cls,
            "prefix": self.prefix,
            "repair_jpeg": self._scan_options["repair_jpeg"],
            "max_file_bytes": self._scan_options["max_file_bytes"],
        }

    def cache_labels(self, path=Path("./labels.cache")):
        options = dict(
            num_classes=len(self.data["names"]),
            task="segment" if self.use_segments else "detect",
            single_cls=self.single_cls,
            prefix=self.prefix,
            **self._scan_options,
        )
        if self.annotation_cache == "native":
            from . import scan_cached

            result = scan_cached(
                self._native_cache_path(path), self.im_files, self.label_files, **options, **self._native_cache_options
            )
            self.annotation_cache_hit = result.cache_hit
            self.annotation_cache_path = result.cache_path
            self.annotation_cache_write_error = result.cache_write_error
            self.annotation_cache_write_ok = None if result.cache_hit else result.cache_write_error is None
            if result.cache_write_error:
                LOGGER.warning(
                    f"{self.prefix}Native cache unavailable; using scanned labels: {result.cache_write_error}"
                )
        else:
            result = scan(
                self.im_files,
                self.label_files,
                **options,
            )
        # Retain only small evidence, not a second packed copy for every worker.
        self.scan_evidence = (
            None
            if self.annotation_cache_hit
            else {
                "summary": dict(result.summary),
                "fallback_count": result.fallback_count,
                "diagnostics": [vars(d) for d in result.diagnostics],
            }
        )
        messages = [d.message for d in result.diagnostics]
        if messages and not self.annotation_cache_hit:
            LOGGER.info("\n".join(messages))
        if result.summary["found"] == 0:
            message = f"{self.prefix}No labels found in {path}. {dataset.HELP_URL}"
            if self.augment:
                raise ValueError(message)
            LOGGER.warning(message)
        value = {
            "labels": result.to_ultralytics_labels(),
            "hash": None if self.annotation_cache == "native" else self.get_cache_hash(),
            "results": tuple(result.summary[k] for k in ("found", "missing", "empty", "corrupt", "total")),
            "msgs": messages,
            "version": dataset.DATASET_CACHE_VERSION,
            "ultrafast_profile": self._cache_profile(),
        }
        if self.annotation_cache == "ultralytics" and value["labels"]:
            self.annotation_cache_write_ok = _write_legacy(Path(path), value, self.prefix)
        return value


def build_yolo_dataset(
    cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False, fraction=None, **native_options
):
    """Explicit trainer factory with the pinned base builder's argument semantics."""
    check_profile()
    if multi_modal or cfg.task not in ("detect", "segment"):
        raise UnsupportedProfile("retain the original builder for multimodal or unsupported tasks")
    if data.get("complete"):
        fraction = 1.0
    elif fraction is None:
        fraction = build.get_split_fraction(cfg.fraction, mode)
    return FastYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=copy(cfg),
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=fraction,
        **native_options,
    )
