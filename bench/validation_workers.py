"""Balanced fresh-process worker comparison for unchanged content validation.

Three unmeasured primers precede all six permutations of three worker counts.
Every measured trial verifies both images and labels from the same frozen cache.
This measures the input-validation phase, not full startup or a new binary.
"""

import argparse
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import threading
import time
import traceback

import psutil


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def save(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def trial(args, state):
    from ultrafast_yolo_dataset import _native

    extension = Path(_native.__file__).resolve()
    identity = json.loads(args.identity.read_text())
    assert extension.is_relative_to(Path(sys.prefix).resolve())
    assert sha(extension) == identity["extension_sha256"]
    assert sha(args.cache) == args.expected_cache_sha256
    sections = _native.read_cache_sections(str(args.cache), 2 * 1024**3)
    metadata = json.loads(sections[0])
    total = len(metadata["images"])
    assert total == len(metadata["labels"]) == 500000
    assert 1 <= args.count <= total
    indices = [i * total // args.count for i in range(args.count)]
    selected = {}
    for kind, proof in zip(("images", "labels"), sections[-2:], strict=True):
        assert len(proof) == total * 57
        paths = [metadata[kind][i] for i in indices]
        table = proof if args.count == total else b"".join(proof[i * 57 : (i + 1) * 57] for i in indices)
        selected[kind] = paths, table
    limits = {"images": metadata["config"]["max_image_bytes"], "labels": metadata["config"]["max_file_bytes"]}
    del sections, metadata, proof, paths, table, indices
    gc.collect()
    state.update(
        extension_sha256=identity["extension_sha256"],
        identity_sha256=sha(args.identity),
        cache_sha256=args.expected_cache_sha256,
        workers=args.trial_workers,
        count=args.count,
        content=True,
        python=sys.version,
        platform=platform.platform(),
        cpu_logical=os.cpu_count(),
        cpu_physical=psutil.cpu_count(logical=False),
        memory_total=psutil.virtual_memory().total,
        selected={
            kind: dict(
                count=len(paths),
                table_sha256=hashlib.sha256(table).hexdigest(),
                paths_sha256=hashlib.sha256(json.dumps(paths, ensure_ascii=False).encode()).hexdigest(),
                max_bytes=limits[kind],
            )
            for kind, (paths, table) in selected.items()
        },
        records=[],
        samples=[],
    )
    process = psutil.Process()
    stop = threading.Event()
    phase = "ready"

    def sample():
        memory = psutil.virtual_memory()
        state["samples"].append(
            dict(
                at_ns=time.time_ns(),
                phase=phase,
                rss_bytes=process.memory_info().rss,
                threads=process.num_threads(),
                cpu_times=psutil.cpu_times()._asdict(),
                loadavg=os.getloadavg(),
                available_memory=memory.available,
                swap_used=psutil.swap_memory().used,
                disk_io=psutil.disk_io_counters()._asdict(),
            )
        )

    def sampler():
        try:
            while not stop.wait(0.5):
                sample()
        except BaseException:
            state["sampling_error"] = traceback.format_exc()

    sample()
    observer = threading.Thread(target=sampler, daemon=True)
    observer.start()
    try:
        save(args.out, state)
        for kind in ("images", "labels"):
            phase = kind
            paths, table = selected[kind]
            rec = dict(kind=kind, started_at_ns=time.time_ns())
            before = resource.getrusage(resource.RUSAGE_SELF)
            started = time.perf_counter()
            _native.verify_fingerprint_table(table, paths, limits[kind], args.trial_workers, True)
            elapsed = time.perf_counter() - started
            after = resource.getrusage(resource.RUSAGE_SELF)
            rec.update(
                complete=True,
                verified_paths=len(paths),
                elapsed_s=elapsed,
                user_s=after.ru_utime - before.ru_utime,
                system_s=after.ru_stime - before.ru_stime,
                minor_faults=after.ru_minflt - before.ru_minflt,
                major_faults=after.ru_majflt - before.ru_majflt,
                finished_at_ns=time.time_ns(),
            )
            state["records"].append(rec)
            print(kind, len(paths), elapsed, flush=True)
            save(args.out, state)
    finally:
        stop.set()
        observer.join(timeout=10)
        assert not observer.is_alive()
    phase = "finished"
    sample()
    assert "sampling_error" not in state
    assert sha(args.cache) == args.expected_cache_sha256 and sha(extension) == identity["extension_sha256"]
    state.update(
        complete=True,
        passed=True,
        validation_elapsed_s=sum(rec["elapsed_s"] for rec in state["records"]),
        sampled_peak_rss=max(s["rss_bytes"] for s in state["samples"]),
        scope="Fresh process, unchanged native content validation; timing excludes cache decode/hashing, metadata extraction, sampler start/stop and report writes. RSS samples include validation-ready/final observations; sampled phase RSS, not whole-startup peak. Host counters include unrelated activity. No cold-cache or full-startup speedup claim.",
    )


def series(args, state):
    assert args.count == 500000 and len(args.workers) == 3 and len(set(args.workers)) == 3
    assert all(1 <= w <= 256 for w in args.workers)
    runs = args.out.with_suffix(".runs")
    runs.mkdir(exist_ok=False)
    permutations = list(itertools.permutations(args.workers))
    plan = [dict(primer=True, block=-1, position=i, workers=w) for i, w in enumerate(args.workers)]
    plan += [
        dict(primer=False, block=block, position=position, workers=w)
        for block, order in enumerate(permutations)
        for position, w in enumerate(order)
    ]
    state.update(plan=plan, records=[], planned_trials=21, measured_trials=18, pairs=500000)
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    overrides = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", YOLO_OFFLINE="true")
    env.update(overrides)
    state["environment_overrides"] = overrides
    frozen_script = sha(Path(__file__))
    binding = None
    for index, spec in enumerate(plan):
        assert sha(Path(__file__)) == frozen_script
        out = runs / f"{index:02d}.json"
        log = out.with_suffix(".log")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--cache",
            str(args.cache),
            "--identity",
            str(args.identity),
            "--expected-cache-sha256",
            args.expected_cache_sha256,
            "--out",
            str(out),
            "--count",
            str(args.count),
            "--trial-workers",
            str(spec["workers"]),
        ]
        rec = dict(spec=spec, command=command, path=str(out), started_at_ns=time.time_ns())
        state["records"].append(rec)
        save(args.out, state)
        print("start", index, spec, flush=True)
        with log.open("x") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            rec["pid"] = child.pid
            save(args.out, state)
            try:
                code = child.wait(timeout=600)
            except subprocess.TimeoutExpired:
                rec["timed_out"] = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    code = child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    code = child.wait(timeout=10)
        rec.update(returncode=code, finished_at_ns=time.time_ns(), log_sha256=sha(log), raw_sha256=sha(out))
        save(args.out, state)
        assert code == 0 and not rec.get("timed_out", False)
        raw = json.loads(out.read_text())
        assert raw["complete"] and raw["passed"] and raw["count"] == args.count
        assert raw["workers"] == spec["workers"] and raw["content"] is True
        assert [r["kind"] for r in raw["records"]] == ["images", "labels"]
        assert all(r["complete"] and r["verified_paths"] == args.count for r in raw["records"])
        current = {
            k: raw[k] for k in ("script_sha256", "extension_sha256", "identity_sha256", "cache_sha256", "selected")
        }
        if binding is None:
            binding = current
        assert current == binding
        assert current["script_sha256"] == frozen_script
        rec.update(
            validated=True, validation_elapsed_s=raw["validation_elapsed_s"], sampled_peak_rss=raw["sampled_peak_rss"]
        )
        save(args.out, state)
        print("done", index, rec["validation_elapsed_s"], flush=True)
    state.update(
        complete=True,
        passed=True,
        binding=binding,
        scope=f"Three unmeasured primers and 18 measured fresh processes across all six permutations of {args.workers} workers. Same binary and content checks on all 500,000 image/label pairs. Worker tuning for this input-validation phase; no full-startup, new-kernel or cold-cache claim.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[4, 7, 16])
    parser.add_argument("--count", type=int, default=500000)
    parser.add_argument("--trial-workers", type=int)
    args = parser.parse_args()
    state = dict(
        complete=False, passed=False, pid=os.getpid(), script_sha256=sha(Path(__file__)), started_at_ns=time.time_ns()
    )
    with args.out.open("x") as stream:
        json.dump(state, stream)
    try:
        (trial if args.trial_workers is not None else series)(args, state)
    except BaseException:
        state.update(complete=True, passed=False, error=traceback.format_exc())
        raise
    finally:
        state["finished_at_ns"] = time.time_ns()
        save(args.out, state)


if __name__ == "__main__":
    main()
