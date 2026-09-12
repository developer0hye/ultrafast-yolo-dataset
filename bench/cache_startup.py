"""Native cache startup vs a content-checked, checksummed reference-cache baseline.

Priming uses separate processes. Timed hit workers never prime their own cache.
The auxiliary reference baseline retains the original NumPy/pickle representation,
uses the same Rust input fingerprint engine, avoids redundant weak hashing, and
verifies its cache payload before trusted deserialization. This is a benchmark
adapter, not a general reference cache builder with captured-byte provenance.
"""

import argparse
import copy
import functools
import hashlib
import io
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
from startup import digest_value, fingerprint, peak_rss
from torch.utils.data import DataLoader
from ultrafast_yolo_dataset import _native
from ultrafast_yolo_dataset.ultralytics import FastYOLODataset, check_profile
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data import dataset
from ultralytics.utils import NUM_THREADS

BACKENDS = ("reference", "reference-content", "native-content", "native-metadata")


def content_key(images, labels, workers):
    digest = hashlib.sha256()
    for name, paths, limit in (("images", images, 64 * 1024**2), ("labels", labels, 16 * 1024**2)):
        digest.update(name.encode())
        for path in paths:
            encoded = path.encode()
            digest.update(len(encoded).to_bytes(8, "little") + encoded)
        # Hash the engine's canonical JSON directly; the baseline need not build
        # Python per-file fingerprint dictionaries just to compare one root key.
        digest.update(_native.fingerprint_paths(paths, limit, workers, True).encode())
    return digest.hexdigest()


class ContentCheckedReference(dataset.YOLODataset):
    """Warm-cache baseline with equally strong input and payload validation."""

    def get_cache_hash(self):
        started = time.perf_counter()
        key = content_key(self.im_files, self.label_files, NUM_THREADS)
        self.fingerprint_s = time.perf_counter() - started
        return key

    def _load_or_scan_cache(self, cache_path, cache_hash):
        started = time.perf_counter()
        manifest = json.loads(cache_path.with_suffix(".content.json").read_text())
        assert manifest["input_content_key"] == cache_hash, "reference input content changed"
        payload = cache_path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == manifest["cache_sha256"], "reference cache payload changed"
        # Only the benchmark's own freshly generated, digest-checked pickle is read.
        import gc

        enabled = gc.isenabled()
        try:
            gc.disable()
            value = np.load(io.BytesIO(payload), allow_pickle=True).item()
        finally:
            gc.enable() if enabled else gc.disable()
        assert value["version"] == dataset.DATASET_CACHE_VERSION
        self.cache_decode_s = time.perf_counter() - started
        return value, True


def fixture_manifest(root):
    path = root / "coco.json" if (root / "coco.json").exists() else root / "startup.json"
    manifest = json.loads(path.read_text())
    assert manifest["kind"] in ("ultrafast-yolo-startup-synthetic-v1", "ultrafast-yolo-coco-v1")
    assert manifest["count"] > 0
    return manifest


def legacy_path(root):
    relative = Path(fixture_manifest(root).get("legacy_cache", "labels/0000.cache"))
    assert not relative.is_absolute() and ".." not in relative.parts and relative.parts[0] == "labels"
    assert relative.suffix == ".cache"
    return root / relative


def options(root, task, backend):
    values = {
        "img_path": str(root / fixture_manifest(root).get("image_directory", "images")),
        "imgsz": 640,
        "batch_size": 8,
        "augment": False,
        "hyp": copy.deepcopy(DEFAULT_CFG),
        "task": task,
        "data": {"names": dict(enumerate(map(str, range(80))))},
    }
    if backend.startswith("native-"):
        values.update(
            annotation_cache="native",
            cache_fingerprint=backend.split("-")[1],
            cache_dir=root / "benchmark-caches" / backend,
            scan_workers=NUM_THREADS,
        )
    return values


def clear_cache(root, backend):
    if backend.startswith("native-"):
        for path in (root / "benchmark-caches" / backend).glob("*.uydcache"):
            path.unlink()
    else:
        legacy_path(root).unlink(missing_ok=True)
        legacy_path(root).with_suffix(".content.json").unlink(missing_ok=True)


def stage_timers():
    import ultrafast_yolo_dataset._cache as cache_module
    from ultrafast_yolo_dataset._scan import ScanResult

    values = {}
    # Observe unmodified implementations. These durations are exclusive phases;
    # overall constructor/first-batch timings still include all other work.
    for owner, name in (
        (cache_module, "_validate_inputs"),
        (cache_module, "_decode"),
        (cache_module, "_encode"),
        (ScanResult, "to_ultralytics_labels"),
    ):
        original = getattr(owner, name)

        def wrap(function, label):
            @functools.wraps(function)
            def measured(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return function(*args, **kwargs)
                finally:
                    values[label] = values.get(label, 0) + time.perf_counter() - started

            return measured

        setattr(owner, name, wrap(original, name))
    return values


def worker(root, backend, mode, prime):
    manifest = fixture_manifest(root)
    assert fingerprint(root) == manifest["fingerprint"]
    check_profile()
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    random.seed(912)
    np.random.seed(912)
    torch.manual_seed(912)
    if prime or mode == "miss":
        clear_cache(root, backend)
    cls = FastYOLODataset if backend.startswith("native-") else dataset.YOLODataset
    if backend == "reference-content" and not prime:
        assert mode == "hit", "the optimized reference-content baseline is defined for hits"
        cls = ContentCheckedReference
    timings = stage_timers()
    before = resource.getrusage(resource.RUSAGE_SELF)
    rss_before, high_water_before = psutil.Process().memory_info().rss, peak_rss()
    start = time.perf_counter()
    result = cls(**options(root, manifest["task"], backend))
    constructed = time.perf_counter()
    rss_constructor = peak_rss()
    if prime:
        if backend in ("reference", "reference-content"):
            cache_path = legacy_path(root)
            # Priming occurred on a content-verified frozen fixture. The sidecar
            # binds that reference payload to the input content for hit checks.
            content = content_key(result.im_files, result.label_files, NUM_THREADS)
            cache_path.with_suffix(".content.json").write_text(
                json.dumps(
                    {
                        "input_content_key": content,
                        "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                    }
                )
            )
        assert fingerprint(root) == manifest["fingerprint"]
        return {"primed": backend}
    batch = next(iter(DataLoader(result, batch_size=8, num_workers=0, collate_fn=dataset.YOLODataset.collate_fn)))
    ended = time.perf_counter()
    after, rss_batch = resource.getrusage(resource.RUSAGE_SELF), peak_rss()
    cache_path = Path(result.annotation_cache_path) if backend.startswith("native-") else legacy_path(root)
    # Verification is outside all timers and memory samples. Hash counters and
    # ordered diagnostics as well as labels/batches on real error-bearing data.
    if backend.startswith("native-"):
        metadata = json.loads(_native.read_cache_sections(str(cache_path), 2 * 1024**3)[0])
        semantics = {"summary": metadata["summary"], "messages": [d["message"] for d in metadata["diagnostics"]]}
        fallback_count = metadata["fallback_count"]
    else:
        verified_cache = np.load(cache_path, allow_pickle=True).item()  # Own freshly generated benchmark cache.
        semantics = {
            "summary": dict(zip(("found", "missing", "empty", "corrupt", "total"), verified_cache["results"])),
            "messages": verified_cache["msgs"],
        }
        fallback_count = None
    hashes = {}
    for name, value in (("labels", result.labels), ("first_batch", batch)):
        digest = hashlib.sha256()
        digest_value(value, digest)
        hashes[name] = digest.hexdigest()
    hashes["scan_summary_and_messages"] = hashlib.sha256(json.dumps(semantics, sort_keys=True).encode()).hexdigest()
    assert fingerprint(root) == manifest["fingerprint"]
    if backend.startswith("native-"):
        assert result.annotation_cache_hit == (mode == "hit")
        assert result.annotation_cache_write_error is None
    elif backend == "reference-content":
        timings.update(input_fingerprints=result.fingerprint_s, reference_cache_decode=result.cache_decode_s)
    return {
        "backend": backend,
        "mode": mode,
        "workers": NUM_THREADS,
        "loader_workers": 0,
        "constructor_s": constructed - start,
        "first_batch_total_s": ended - start,
        "stages_s": timings,
        "rss_before_constructor_bytes": rss_before,
        "peak_rss_before_constructor_bytes": high_water_before,
        "peak_rss_constructor_bytes": rss_constructor,
        "peak_rss_first_batch_bytes": rss_batch,
        "cache_bytes": cache_path.stat().st_size,
        "output_sha256": hashes,
        "scan_summary": semantics["summary"],
        "diagnostic_count": len(semantics["messages"]),
        "native_fallback_count": fallback_count,
        "user_s": after.ru_utime - before.ru_utime,
        "system_s": after.ru_stime - before.ru_stime,
        "minor_faults": after.ru_minflt - before.ru_minflt,
        "major_faults": after.ru_majflt - before.ru_majflt,
        "loadavg": os.getloadavg(),
        "available_ram_bytes": psutil.virtual_memory().available,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--mode", choices=["hit", "miss"], default="hit")
    parser.add_argument("--backends", nargs="+", choices=BACKENDS)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--worker", choices=BACKENDS)
    parser.add_argument("--prime", action="store_true")
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    root = args.corpus.resolve()
    if args.worker:
        value = worker(root, args.worker, args.mode, args.prime)
        with args.result.open("x") as stream:
            json.dump(value, stream)
        return
    if args.out is None:
        parser.error("--out required")
    backends = args.backends or (
        ["reference-content", "native-content"] if args.mode == "hit" else ["reference", "native-content"]
    )
    run_dir = args.out.with_suffix(".runs")
    run_dir.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    report = {
        "scope": "actual YOLODataset constructor and first batch; content-hit baseline avoids redundant weak hashing and checks cache payload",
        "priming": "separate untimed processes, once per backend; never included in measured-worker RSS",
        "input_checks": "full fixture digest before/after each worker outside timing; pre-reads OS cache",
        "reference_content_baseline": "benchmark-only trusted NumPy/pickle cache; same Rust content-fingerprint engine, root input key, cache SHA256 before unpickling; primed on verified immutable fixture, not a general race-safe cache builder",
        "imports": "both backends imported in every process, outside timing",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "cv2": cv2.__version__,
        "torch": torch.__version__,
        "ram_bytes": psutil.virtual_memory().total,
        "logical_cpus": psutil.cpu_count(),
        "native_extension_sha256": hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in list((project / "src").glob("*.rs"))
            + list((project / "python").rglob("*.py"))
            + [Path(__file__).resolve()]
        },
        "corpus": fixture_manifest(root),
        "results": [],
    }

    def run(backend, prefix, prime=False):
        output = run_dir / f"{prefix}-{backend}.json"
        command = [
            sys.executable,
            __file__,
            "--corpus",
            str(root),
            "--worker",
            backend,
            "--mode",
            args.mode,
            "--result",
            str(output),
        ]
        if prime:
            command.append("--prime")
        with (run_dir / f"{prefix}-{backend}.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        return json.loads(output.read_text())

    if args.mode == "hit":
        primers = [b for b in backends if b.startswith("native-")]
        if any(b.startswith("reference") for b in backends):
            primers.insert(0, "reference-content")
        for backend in primers:
            run(backend, "prime", True)
            print("primed", backend, flush=True)
    expected = None
    for round_ in range(args.rounds):
        order = backends if round_ % 2 == 0 else backends[::-1]
        for backend in order:
            value = run(backend, str(round_))
            expected = expected or value["output_sha256"]
            assert value["output_sha256"] == expected, "full labels or first-batch parity mismatch"
            value.update(round=round_, parity=True)
            report["results"].append(value)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(round_, backend, value["constructor_s"], flush=True)
    report["summary"] = {}
    for metric in ("constructor_s", "first_batch_total_s", "peak_rss_constructor_bytes"):
        values = {b: np.array([r[metric] for r in report["results"] if r["backend"] == b]) for b in backends}
        summary = {b: {"median": float(np.median(v)), "p95": float(np.quantile(v, 0.95))} for b, v in values.items()}
        indices = np.random.default_rng(912).integers(0, args.rounds, (10000, args.rounds))
        for backend in backends[1:]:
            ratios = np.median(values[backends[0]][indices], axis=1) / np.median(values[backend][indices], axis=1)
            summary[backend]["reference_over_candidate_ci95"] = np.quantile(ratios, [0.025, 0.975]).tolist()
        report["summary"][metric] = summary
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
