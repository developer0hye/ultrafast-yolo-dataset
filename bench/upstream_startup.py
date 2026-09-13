"""Dataset startup: unmodified Ultralytics YOLODataset versus annotation_cache='fast'.

The baseline is the pinned reference exactly as users run it: its own `.cache`
pickle, `get_hash` path/size validation and eager label objects. The candidate
must reproduce the complete label list, image/label paths, scan counters and
messages, and the first collated batch; any mismatch aborts the campaign.

Each measurement is a fresh process. Cache priming is a separate process and is
never timed. Pairs alternate their execution order. Imports occur before timing.
"""

import argparse
import copy
import functools
import hashlib
import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import PIL
import psutil
import torch
from startup import digest_value, peak_rss
from torch.utils.data import DataLoader
from ultrafast_yolo_dataset import _fast, _native
from ultrafast_yolo_dataset.ultralytics import FastYOLODataset
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data import base, dataset
from ultralytics.data import utils as data_utils

BACKENDS = ("upstream", "fast")


def manifest(root):
    path = root / "coco.json" if (root / "coco.json").exists() else root / "startup.json"
    return json.loads(path.read_text())


def fast_cache_dir(root):
    return root / "benchmark-caches" / "fast"


def settings(root, backend, args):
    values = {
        "img_path": str(root / manifest(root).get("image_directory", "images")),
        "imgsz": 640,
        "batch_size": 8,
        "augment": False,
        "hyp": copy.deepcopy(DEFAULT_CFG),
        "task": manifest(root)["task"],
        "data": {"names": dict(enumerate(map(str, range(80))))},
    }
    if backend == "fast":
        values.update(
            annotation_cache="fast",
            cache_dir=fast_cache_dir(root),
            index_workers=args.index_workers,
            scan_workers=args.scan_workers,
        )
    return values


def clear(root, backend, label_files):
    if backend == "fast":
        for path in fast_cache_dir(root).glob("*.uydfast"):
            path.unlink()
    else:
        Path(label_files[0]).parent.with_suffix(".cache").unlink(missing_ok=True)


def phase_timers(backend):
    values = {}

    def wrap(owner, name, label):
        original = getattr(owner, name)

        @functools.wraps(original)
        def measured(*a, **k):
            started = time.perf_counter()
            try:
                return original(*a, **k)
            finally:
                values[label] = values.get(label, 0.0) + time.perf_counter() - started

        setattr(owner, name, measured)

    if backend == "upstream":
        wrap(base.BaseDataset, "get_img_files", "discover")
        wrap(dataset.YOLODataset, "get_cache_hash", "validate_inputs")
        wrap(dataset, "load_dataset_cache_file", "load_cache")
        wrap(dataset.YOLODataset, "cache_labels", "scan_and_save")
        wrap(dataset.YOLODataset, "verify_labels", "verify_labels")
    else:
        wrap(FastYOLODataset, "get_img_files", "discover_and_stat")
        wrap(FastYOLODataset, "_fast_labels", "labels_total")
        wrap(_fast, "load", "load_cache")
        wrap(FastYOLODataset, "_fast_scan", "scan_and_save")
        wrap(FastYOLODataset, "_verify_packed", "verify_labels")
    wrap(base.BaseDataset, "build_transforms", "build_transforms")
    return values


def worker(root, backend, mode, prime, args):
    random.seed(912)
    np.random.seed(912)
    torch.manual_seed(912)
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    options = settings(root, backend, args)
    # Priming reuses a still-valid reference .cache (other evidence depends on
    # it); only this harness's own fast caches are cleared before priming.
    if (prime and backend == "fast") or mode == "miss":
        probe = FastYOLODataset.__new__(FastYOLODataset)
        probe.fraction = 1.0
        probe.prefix = ""
        images = base.BaseDataset.get_img_files(probe, options["img_path"])
        clear(root, backend, data_utils.img2label_paths(images))
        del images, probe
    cls = FastYOLODataset if backend == "fast" else dataset.YOLODataset
    timings = phase_timers(backend)
    before = resource.getrusage(resource.RUSAGE_SELF)
    rss_before, high_water_before = psutil.Process().memory_info().rss, peak_rss()
    started = time.perf_counter()
    result = cls(**options)
    constructed = time.perf_counter()
    rss_constructor = peak_rss()
    if prime:
        return {"primed": backend}
    loader = DataLoader(
        result, batch_size=8, num_workers=args.loader_workers, collate_fn=dataset.YOLODataset.collate_fn
    )
    first = next(iter(loader))
    ended = time.perf_counter()
    after, rss_batch = resource.getrusage(resource.RUSAGE_SELF), peak_rss()
    del loader
    # Verification is outside every timer and memory sample.
    if backend == "fast":
        assert result.annotation_cache_hit == (mode == "hit"), result.annotation_cache_miss_reason
        if mode == "miss":
            assert result.annotation_cache_write_ok, result.annotation_cache_write_error
        metadata, _ = _fast.load(
            result.annotation_cache_path,
            digest=result.annotation_cache_digest,
            config=result._fast_config(),
            workers=args.index_workers,
        )
        summary = metadata["summary"]
        messages = [d["message"] for d in metadata["diagnostics"]]
        cache_path = Path(result.annotation_cache_path)
    else:
        cache_path = Path(result.label_files[0]).parent.with_suffix(".cache")
        cached = np.load(cache_path, allow_pickle=True).item()  # the reference's own freshly written cache
        summary = dict(zip(("found", "missing", "empty", "corrupt", "total"), cached["results"]))
        messages = cached["msgs"]
    hashes = {}
    for name, value in (
        ("labels", list(result.labels)),
        ("im_files", result.im_files),
        ("label_files", result.label_files),
        ("first_batch", first),
    ):
        digest = hashlib.sha256()
        digest_value(value, digest)
        hashes[name] = digest.hexdigest()
    hashes["scan_summary_and_messages"] = hashlib.sha256(
        json.dumps({"summary": summary, "messages": messages}, sort_keys=True).encode()
    ).hexdigest()
    return {
        "backend": backend,
        "mode": mode,
        "constructor_s": constructed - started,
        "first_batch_total_s": ended - started,
        "phases_s": timings,
        "rss_before_constructor_bytes": rss_before,
        "peak_rss_before_constructor_bytes": high_water_before,
        "peak_rss_constructor_bytes": rss_constructor,
        "peak_rss_first_batch_bytes": rss_batch,
        "cache_bytes": cache_path.stat().st_size,
        "output_sha256": hashes,
        "scan_summary": summary,
        "diagnostic_count": len(messages),
        "user_s": after.ru_utime - before.ru_utime,
        "system_s": after.ru_stime - before.ru_stime,
        "loadavg": os.getloadavg(),
        "available_ram_bytes": psutil.virtual_memory().available,
    }


def paired_interval(reference, candidate, rounds, seed=912):
    indices = np.random.default_rng(seed).integers(0, rounds, (10000, rounds))
    ratios = np.median(reference[indices], axis=1) / np.median(candidate[indices], axis=1)
    return np.quantile(ratios, [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--mode", choices=["hit", "miss"], default="hit")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--index-workers", type=int, default=16)
    parser.add_argument("--scan-workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--worker", choices=BACKENDS)
    parser.add_argument("--prime", action="store_true")
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    root = args.corpus.resolve()
    if args.worker:
        value = worker(root, args.worker, args.mode, args.prime, args)
        with args.result.open("x") as stream:
            json.dump(value, stream)
        return
    if args.out is None:
        parser.error("--out required")
    run_dir = args.out.with_suffix(".runs")
    run_dir.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    sources = {
        str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
        for pattern in (
            "src/*.rs",
            "python/ultrafast_yolo_dataset/*.py",
            "bench/upstream_startup.py",
            "bench/startup.py",
        )
        for p in sorted(project.glob(pattern))
    }
    report = {
        "scope": "unmodified pinned Ultralytics YOLODataset (own .cache) vs FastYOLODataset annotation_cache='fast'; "
        "constructor and first batch in fresh processes",
        "validation": {
            "upstream": "get_hash: summed os.stat sizes of all image+label files plus the joined path string",
            "fast": "ordered image/label paths plus per-file kind, size and modification time (ns)",
        },
        "priming": "separate untimed processes, once per backend",
        "imports": "all modules imported before timing in every process",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "cv2": cv2.__version__,
        "torch": torch.__version__,
        "ram_bytes": psutil.virtual_memory().total,
        "logical_cpus": psutil.cpu_count(),
        "native_extension_sha256": hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
        "source_sha256": sources,
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "corpus": manifest(root),
        "results": [],
    }

    def run(backend, prefix, prime=False):
        output = run_dir / f"{prefix}-{backend}.json"
        command = [sys.executable, __file__, "--corpus", str(root), "--worker", backend, "--mode", args.mode]
        command += ["--result", str(output), "--index-workers", str(args.index_workers)]
        command += ["--scan-workers", str(args.scan_workers), "--loader-workers", str(args.loader_workers)]
        if prime:
            command.append("--prime")
        with (run_dir / f"{prefix}-{backend}.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        return json.loads(output.read_text())

    if args.mode == "hit":
        for backend in BACKENDS:
            run(backend, "prime", True)
            print("primed", backend, flush=True)
    expected = None
    for round_ in range(args.rounds):
        for backend in BACKENDS if round_ % 2 == 0 else BACKENDS[::-1]:
            value = run(backend, str(round_))
            expected = expected or value["output_sha256"]
            assert value["output_sha256"] == expected, f"parity mismatch: {backend} round {round_}"
            value.update(round=round_, parity=True)
            report["results"].append(value)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(round_, backend, f"{value['constructor_s']:.3f}s", flush=True)
    report["summary"] = {}
    for metric in ("constructor_s", "first_batch_total_s", "peak_rss_constructor_bytes", "peak_rss_first_batch_bytes"):
        values = {b: np.array([r[metric] for r in report["results"] if r["backend"] == b]) for b in BACKENDS}
        report["summary"][metric] = {
            "upstream_median": float(np.median(values["upstream"])),
            "fast_median": float(np.median(values["fast"])),
            "upstream_over_fast": float(np.median(values["upstream"]) / np.median(values["fast"])),
            "upstream_over_fast_ci95": paired_interval(values["upstream"], values["fast"], args.rounds),
        }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
