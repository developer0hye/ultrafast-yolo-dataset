"""P1 only: distinct files -> read/parse/validate -> equivalent packed NumPy arrays.

Image verification, discovery and dataset/cache initialization are NOT measured.
python bench/label_engine.py --prepare /path/to/corpus --task segment --count 100000
python bench/label_engine.py --corpus /path/to/corpus --out bench/out/segment.json
"""
import argparse
import hashlib
import json
from multiprocessing.pool import ThreadPool
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tests"))
from reference import verify_image_label
from ultrafast_yolo_dataset import parse_labels


def prepare(root, task, count):
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20260912)
    templates = []
    for i in range(1000):
        n = 1 if i % 100 < 80 else (10 if i % 100 < 99 else 100)
        rows = []
        for j in range(n):
            if task == "detect":
                values = rng.uniform(.01, .99, 4)
            else:
                vertices = (3, 16, 100)[i % 3]
                values = rng.uniform(.01, .99, vertices * 2)
            rows.append(str(j % 80) + " " + " ".join(f"{v:.8f}" for v in values))
        templates.append(("\n".join(rows) + "\n").encode())
    digest = hashlib.sha256()
    total = 0
    for i in range(count):
        name = f"{i//1000:04d}/{i:08d}.txt"
        path = root/name
        path.parent.mkdir(exist_ok=True)
        contents = templates[i % len(templates)]
        path.write_bytes(contents)
        digest.update(name.encode() + b"\0" + contents)
        total += len(contents)
    manifest = {"count": count, "task": task, "seed": 20260912, "bytes": total,
                "sha256_names_and_contents": digest.hexdigest(), "distinct_files": count,
                "templates": 1000, "objects_distribution_percent": {"1": 80, "10": 19, "100": 1},
                "segment_vertices_cycle": [3, 16, 100], "synthetic": True,
                "os_cache": "uncontrolled; generation/previous runs may warm file cache"}
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest))


def reference(paths, workers):
    args = [("image.jpg", str(p), "", False, 80, 0, 0, False) for p in paths]
    with ThreadPool(workers) as pool:
        records = list(pool.imap(verify_image_label, args))
    labels, segments, objects, offsets, statuses, duplicates = [], [], [], [0], [], []
    for result in records:
        assert result[0] is not None, result[-1]
        labels.append(result[1])
        segments.extend(result[3])
        objects.extend(range(offsets[-1], offsets[-1]+len(result[3])))
        offsets.append(offsets[-1]+len(result[1]))
        statuses.append(2 if result[7] else 1 if result[5] else 0)
        duplicates.append(int(result[-1].split(": ")[-1].split()[0]) if "duplicate labels removed" in result[-1] else 0)
    return {"labels": np.concatenate(labels), "object_offsets": np.asarray(offsets, np.int64),
            "segment_points": np.concatenate(segments) if segments else np.empty((0, 2), np.float32),
            "segment_offsets": np.r_[np.int64(0), np.cumsum([len(s) for s in segments], dtype=np.int64)],
            "segment_object_indices": np.asarray(objects, np.int64), "statuses": np.asarray(statuses, np.uint8),
            "duplicates": np.asarray(duplicates, np.int64)}


def worker(root, backend, workers):
    manifest = json.loads((root/"manifest.json").read_text())
    paths = [root/f"{i//1000:04d}/{i:08d}.txt" for i in range(manifest["count"])]
    def verify_inputs():
        checksum = hashlib.sha256()
        for path in paths:
            checksum.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
        assert checksum.hexdigest() == manifest["sha256_names_and_contents"]
    verify_inputs()
    start = time.perf_counter()
    if backend == "reference":
        result = reference(paths, workers)
    else:
        packed = parse_labels(paths, num_classes=80, task=manifest["task"], workers=workers)
        result = packed.to_arrays()
        del packed
    wall = time.perf_counter() - start
    usage = resource.getrusage(resource.RUSAGE_SELF)
    assert not np.any(result["statuses"] >= 3), "unhandled input in timed workload"
    digest = hashlib.sha256()
    for name in sorted(result):
        array = result[name]
        digest.update(f"{name}:{array.shape}:{array.dtype}".encode() + array.tobytes())
    verify_inputs()
    return {"backend": backend, "workers": workers, "wall_s": wall, "labels_per_s": len(paths)/wall,
            "peak_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            "output_sha256": digest.hexdigest(), "reference_required": 0,
            "user_s": usage.ru_utime, "system_s": usage.ru_stime,
            "loadavg": os.getloadavg(), "available_ram_bytes": psutil.virtual_memory().available}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", type=Path)
    p.add_argument("--count", type=int, default=100000)
    p.add_argument("--task", choices=["detect", "segment"], default="detect")
    p.add_argument("--corpus", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--worker", choices=["reference", "native"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rounds", type=int, default=5)
    args = p.parse_args()
    if args.prepare:
        prepare(args.prepare, args.task, args.count)
        return
    if not args.corpus:
        p.error("--corpus required")
    if args.worker:
        print(json.dumps(worker(args.corpus, args.worker, args.workers)))
        return
    if not args.out:
        p.error("--out required")
    report = {"corpus": json.loads((args.corpus/"manifest.json").read_text()),
              "platform": platform.platform(), "python": sys.version, "numpy": np.__version__,
              "ram_bytes": psutil.virtual_memory().total, "logical_cpus": psutil.cpu_count(),
              "reference_sha": "795a556942a12fe0124cf767888194a1d0b83e2e",
              "scope": "P1 label-only file engine including export to equivalent packed arrays; image verification excluded",
              "input_checks": "full name/content SHA256 before and after each measurement, excluded from timing",
              "os_cache": "pre-read by content verification; no cold-cache claim",
              "results": []}
    expected = None
    for round_ in range(args.rounds):
        for backend in (["reference", "native"] if round_ % 2 == 0 else ["native", "reference"]):
            result = json.loads(subprocess.check_output([sys.executable, __file__, "--corpus", str(args.corpus),
                "--worker", backend, "--workers", str(args.workers)], text=True))
            expected = expected or result["output_sha256"]
            assert result["output_sha256"] == expected
            result.update(round=round_, parity=True)
            report["results"].append(result)
            print(round_+1, backend, result["wall_s"], flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
