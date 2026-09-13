"""Explicit integration for the pinned base Detection/Segmentation dataset.

The native backend validates captured content; the optional legacy backend
retains Ultralytics' weaker path/size invalidation and trusted pickle bridge.
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
from copy import copy
from pathlib import Path

import numpy as np
import PIL
from PIL import Image
from ultralytics.data import build, dataset, utils
from ultralytics.data.base import BaseDataset
from ultralytics.utils import LOGGER, colorstr, ops, patches

from . import scan

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


class FastYOLODataset(_BASE):
    """Base YOLODataset with an explicit hybrid scan and optional legacy bridge.

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
        **kwargs,
    ):
        check_profile()
        if type(self) is not FastYOLODataset or task not in ("detect", "segment"):
            raise UnsupportedProfile("only the base Detection/Segmentation dataset is supported; retain custom hooks")
        if annotation_cache not in ("none", "ultralytics", "native"):
            raise ValueError("annotation_cache must be none, native, or ultralytics")
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
        super().__init__(*args, task=task, **kwargs)

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
