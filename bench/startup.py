"""Actual YOLODataset construction + first batch, with equivalent legacy caching.

This is a synthetic JPEG/TXT workload. Legacy path/size validation is weaker
than the planned native content cache; these warm-cache numbers cannot prove P4.
"""

import argparse
import copy
import hashlib
import io
import json
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np
import PIL
import psutil
import torch
from label_engine import prepare as prepare_labels
from PIL import Image
from torch.utils.data import DataLoader
from ultrafast_yolo_dataset.ultralytics import FastYOLODataset
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data import dataset
from ultralytics.utils import NUM_THREADS


def fingerprint(root):
    digest = hashlib.sha256()
    files = sorted((root / "images").rglob("*.jpg")) + sorted((root / "labels").rglob("*.txt"))
    size = 0
    for path in files:
        data = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + data)
        size += len(data)
    return {"files": len(files), "bytes": size, "sha256": digest.hexdigest()}


def prepare(root, count, task):
    root.mkdir(parents=True, exist_ok=False)
    prepare_labels(root / "labels", task, count)
    templates = []
    for i in range(16):
        buffer = io.BytesIO()
        Image.new("RGB", [(320, 240), (640, 480), (1280, 720)][i % 3], (i * 15, 45, 71)).save(
            buffer, "JPEG", quality=90
        )
        templates.append(buffer.getvalue())
    for i in range(count):
        path = root / "images" / f"{i // 1000:04d}" / f"{i:08d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(templates[i % 16])
    manifest = {
        "kind": "ultrafast-yolo-startup-synthetic-v1",
        "task": task,
        "count": count,
        "labels": json.loads((root / "labels/manifest.json").read_text()),
        "images": "16 solid-color JPEG templates, distinct physical files; dimensions cycle 320x240/640x480/1280x720",
        "fingerprint": fingerprint(root),
    }
    (root / "startup.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def digest_value(value, digest):
    if isinstance(value, dict):
        for key in sorted(value):
            digest.update(key.encode() + b"\0")
            digest_value(value[key], digest)
    elif isinstance(value, (list, tuple)):
        digest.update(f"{type(value).__name__}:{len(value)}".encode())
        for part in value:
            digest_value(part, digest)
    elif isinstance(value, (np.ndarray, torch.Tensor)):
        array = value.numpy() if isinstance(value, torch.Tensor) else value
        digest.update(f"{array.dtype}:{array.shape}".encode() + array.tobytes())
    else:
        digest.update(f"{type(value).__name__}:{value!r}".encode() + b"\0")


def peak_rss():
    if sys.platform.startswith("linux"):
        return int(Path("/proc/self/status").read_text().split("VmHWM:")[1].split()[0]) * 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def worker(root, backend, mode, workers):
    manifest = json.loads((root / "startup.json").read_text())
    assert manifest["kind"] == "ultrafast-yolo-startup-synthetic-v1"
    assert fingerprint(root) == manifest["fingerprint"]
    # This cache belongs to this generated fixture, not to a user's dataset.
    cache_path = root / "labels/0000.cache"
    cache_path.unlink(missing_ok=True)
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    random.seed(912)
    np.random.seed(912)
    torch.manual_seed(912)
    # Use the original module's worker hook in both reference cache scans and
    # our explicit scan budget; all other upstream hooks remain untouched.
    assert dataset.NUM_THREADS == workers, (dataset.NUM_THREADS, workers)
    options = {
        "img_path": str(root / "images"),
        "imgsz": 640,
        "batch_size": 8,
        "augment": False,
        "hyp": copy.deepcopy(DEFAULT_CFG),
        "task": manifest["task"],
        "data": {"names": dict(enumerate(map(str, range(80))))},
    }
    if backend == "native":
        options.update(scan_workers=workers, annotation_cache="ultralytics", trust_legacy_cache=True)
    cls = FastYOLODataset if backend == "native" else dataset.YOLODataset
    if mode == "hit":
        import gc

        primed = cls(**options)
        del primed
        gc.collect()
        # High-water RSS includes priming: do not use hit RSS as load-only peak.
    before = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter()
    result = cls(**options)
    constructed = time.perf_counter()
    rss_after_constructor = peak_rss()
    loader = DataLoader(result, batch_size=8, num_workers=0, collate_fn=dataset.YOLODataset.collate_fn)
    batch = next(iter(loader))
    first_batch = time.perf_counter()
    after = resource.getrusage(resource.RUSAGE_SELF)
    rss_after_batch = peak_rss()
    hashes = {}
    for name, value in [("labels", result.labels), ("first_batch", batch)]:
        digest = hashlib.sha256()
        digest_value(value, digest)
        hashes[name] = digest.hexdigest()
    assert fingerprint(root) == manifest["fingerprint"]
    if backend == "native":
        assert result.annotation_cache_hit == (mode == "hit")
        if mode == "miss":
            assert result.annotation_cache_write_ok
    return {
        "backend": backend,
        "cache": mode,
        "workers": workers,
        "loader_workers": 0,
        "constructor_s": constructed - start,
        "first_batch_total_s": first_batch - start,
        "first_batch_after_constructor_s": first_batch - constructed,
        "peak_rss_constructor_bytes": rss_after_constructor,
        "peak_rss_first_batch_bytes": rss_after_batch,
        "rss_scope": "process peak since launch, including untimed cache priming on hit runs",
        "cache_bytes": cache_path.stat().st_size,
        "output_sha256": hashes,
        "scan_evidence": getattr(result, "scan_evidence", None),
        "user_s": after.ru_utime - before.ru_utime,
        "system_s": after.ru_stime - before.ru_stime,
        "minor_faults": after.ru_minflt - before.ru_minflt,
        "major_faults": after.ru_majflt - before.ru_majflt,
        "loadavg": os.getloadavg(),
        "available_ram_bytes": psutil.virtual_memory().available,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--task", choices=["detect", "segment"], default="detect")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", choices=["native", "reference"])
    parser.add_argument("--result", type=Path)
    parser.add_argument("--mode", choices=["miss", "hit"], default="miss")
    parser.add_argument("--workers", type=int, default=NUM_THREADS)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare.resolve(), args.count, args.task)
        return
    if not args.corpus or (not args.worker and not args.out):
        parser.error("--corpus and --out required")
    if args.worker:
        with redirect_stdout(sys.stderr):
            value = worker(args.corpus.resolve(), args.worker, args.mode, args.workers)
        if args.result:
            with args.result.open("x") as stream:
                json.dump(value, stream)
        else:
            print(json.dumps(value))
        return
    run_dir = args.out.with_suffix(".runs")
    run_dir.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    report = {
        "scope": "actual YOLODataset discovery, hash, Pillow verification, labels, materialization, legacy-cache write/load, then first batch",
        "cache_guarantee": "both use legacy path/size hash; NOT native content-cache P4 evidence",
        "os_cache": "inputs pre-read for full fingerprints; no cold-cache claim",
        "imports": "excluded from timing; both processes import both backends",
        "reference_sha": "795a556942a12fe0124cf767888194a1d0b83e2e",
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "cv2": cv2.__version__,
        "torch": torch.__version__,
        "ram_bytes": psutil.virtual_memory().total,
        "physical_cpus": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(),
        "source_sha256": {
            str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in list((project / "src").glob("*.rs"))
            + list((project / "python").rglob("*.py"))
            + [Path(__file__).resolve()]
        },
        "corpus": json.loads((args.corpus / "startup.json").read_text()),
        "results": [],
    }
    expected = None
    for round_ in range(args.rounds):
        for backend in ["reference", "native"] if round_ % 2 == 0 else ["native", "reference"]:
            result_path = run_dir / f"{round_}-{backend}.json"
            with (run_dir / f"{round_}-{backend}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--corpus",
                        str(args.corpus),
                        "--worker",
                        backend,
                        "--mode",
                        args.mode,
                        "--workers",
                        str(args.workers),
                        "--result",
                        str(result_path),
                    ],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            value = json.loads(result_path.read_text())
            expected = expected or value["output_sha256"]
            assert value["output_sha256"] == expected, "full label/first batch parity failure"
            value.update(round=round_, parity=True)
            report["results"].append(value)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(round_, backend, value["constructor_s"], flush=True)
    for metric in ("constructor_s", "first_batch_total_s"):
        medians = {
            backend: statistics.median(v[metric] for v in report["results"] if v["backend"] == backend)
            for backend in ("reference", "native")
        }
        ordered = {
            backend: np.array([v[metric] for v in report["results"] if v["backend"] == backend])
            for backend in ("reference", "native")
        }
        indices = np.random.default_rng(912).integers(0, args.rounds, (10000, args.rounds))
        bootstrap = np.median(ordered["reference"][indices], axis=1) / np.median(ordered["native"][indices], axis=1)
        report.setdefault("summary", {})[metric] = {
            "median": medians,
            "p95": {backend: float(np.quantile(values, 0.95)) for backend, values in ordered.items()},
            "speedup": medians["reference"] / medians["native"],
            "paired_bootstrap_speedup_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        }
        print(metric, report["summary"][metric])
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
