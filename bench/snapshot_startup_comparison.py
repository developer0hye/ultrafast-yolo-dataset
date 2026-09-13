"""Actual native YOLODataset P3/P4 comparison against the preceding tested wheel.

The existing startup worker supplies full input hashes, label/batch parity and
phase timers. This wrapper isolates generated cache directories per wheel and
records each installed descriptor and process identity outside measurement.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1048576):
            digest.update(block)
    return digest.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def encoded(value):
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def worker(args):
    reader = module("reader_descriptor", args.harness_root / "bench/cache_read_comparison.py")
    before = reader.descriptor(args.source, args.wheel, args.identity)
    sys.path.insert(0, str(args.harness_root / "bench"))
    startup = module("startup_worker", args.harness_root / "bench/cache_startup.py")
    # Change only the generated native cache location. Dataset class, verification,
    # parsing, counters, transforms and worker/first-batch settings stay unchanged.
    original_options = startup.options

    def options(root, task, backend):
        require(backend == "native-content", "native content comparison only")
        values = original_options(root, task, backend)
        values["cache_dir"] = args.cache_root
        return values

    def clear_cache(root, backend):
        require(backend == "native-content", "native content comparison only")
        require(not args.cache_root.is_relative_to(root), "generated caches must be outside immutable inputs")
        for path in args.cache_root.glob("*.uydcache"):
            require(path.is_file() and not path.is_symlink(), "unexpected cache entry")
            path.unlink()

    startup.options, startup.clear_cache = options, clear_cache
    start_ns = time.time_ns()
    result = startup.worker(args.corpus, "native-content", args.mode, args.prime)
    end_ns = time.time_ns()
    (cache,) = args.cache_root.glob("*.uydcache")
    require(before == reader.descriptor(args.source, args.wheel, args.identity), "worker identity drift")
    return {
        "version": args.version,
        "mode": args.mode,
        "prime": args.prime,
        "pid": os.getpid(),
        "wall_start_ns": start_ns,
        "wall_end_ns": end_ns,
        "wrapper_sha256": sha(Path(__file__)),
        "descriptor": before,
        "cache": {"path": str(cache), "sha256": sha(cache), "bytes": cache.stat().st_size},
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--harness-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--rounds", type=int, choices=(1, 5), default=5)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--version", choices=("baseline", "candidate"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--mode", choices=("hit", "miss"))
    parser.add_argument("--prime", action="store_true")
    for version in ("baseline", "candidate"):
        for field in ("python", "source", "wheel", "identity"):
            parser.add_argument(f"--{version}-{field}", type=Path)
    args = parser.parse_args()
    require(not args.out.exists(), "retain earlier output")
    require(args.corpus.is_absolute() and args.harness_root.is_absolute(), "use explicit absolute inputs")
    manifest = json.loads((args.corpus / "startup.json").read_text())
    require(manifest["count"] == args.pairs and manifest["task"] in ("detect", "segment"), "wrong complete fixture")
    if args.worker:
        args.out.write_text(encoded(worker(args)))
        return
    reader = module("reader_statistics", args.harness_root / "bench/cache_read_comparison.py")
    runs = args.out.with_suffix(".runs")
    runs.mkdir(exist_ok=False)
    cache_root = args.out.with_suffix(".caches")
    cache_root.mkdir(exist_ok=False)
    anchors = {v: json.loads(getattr(args, v + "_identity").read_text()) for v in ("baseline", "candidate")}
    a, b = anchors.values()
    require(a["toolchain"] == b["toolchain"], "different build settings")
    require(a["library_sources"].keys() == b["library_sources"].keys(), "different source inventory")
    require(
        {n for n in a["library_sources"] if a["library_sources"][n] != b["library_sources"][n]} == {"src/snapshot.rs"},
        "this experiment must isolate snapshot changes from the tested reader candidate",
    )
    report = {
        "complete": False,
        "rounds": args.rounds,
        "pairs": args.pairs,
        "corpus": manifest,
        "scope": "Previous tested native wheel versus snapshot-scratch candidate; actual constructor/cache generation and content-hit first batches. Not upstream-Python comparison.",
        "input_checks": "Existing startup worker checks all fixture names/content before and after each worker, outside timers.",
        "cache_policy": "Separate generated native cache directories per version; unchanged immutable image/label paths.",
        "memory_scope": "Process high-water RSS includes imports, descriptor checks and preflight; not working allocation.",
        "wrapper_sha256": sha(Path(__file__)),
        "harness_sources": {
            str(p.relative_to(args.harness_root)): sha(p) for p in sorted((args.harness_root / "bench").glob("*.py"))
        },
        "anchors": anchors,
        "anchor_sha256": {v: sha(getattr(args, v + "_identity")) for v in anchors},
        "records": [],
        "primers": [],
    }
    descriptors, expected, pids = {}, None, set()

    def save():
        args.out.write_text(encoded(report))

    def run(version, mode, repeat, prime=False):
        nonlocal expected
        name = f"{mode}-{'prime' if prime else repeat}-{version}"
        output = runs / (name + ".json")
        command = [
            getattr(args, version + "_python"),
            Path(__file__).resolve(),
            "--worker",
            "--version",
            version,
            "--corpus",
            args.corpus,
            "--pairs",
            args.pairs,
            "--harness-root",
            args.harness_root,
            "--source",
            getattr(args, version + "_source"),
            "--wheel",
            getattr(args, version + "_wheel"),
            "--identity",
            getattr(args, version + "_identity"),
            "--mode",
            mode,
            "--cache-root",
            cache_root / version,
            "--out",
            output,
        ]
        if prime:
            command += ["--prime"]
        command = list(map(str, command))
        log = runs / (name + ".log")
        with log.open("x") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        value = json.loads(output.read_text())
        require(value["pid"] not in pids, "worker PID reused within this campaign")
        pids.add(value["pid"])
        require(value["wall_end_ns"] > value["wall_start_ns"], "invalid wall clock order")
        require(value["version"] == version and value["mode"] == mode and value["prime"] is prime, "wrong worker")
        descriptor = value["descriptor"]
        require(value["wrapper_sha256"] == report["wrapper_sha256"], "wrapper changed")
        require(descriptor["build_identity_sha256"] == report["anchor_sha256"][version], "wrong build identity")
        if version in descriptors:
            require(descriptor == descriptors[version], "installed descriptor changed")
        descriptors[version] = descriptor
        if len(descriptors) == 2:
            left, right = (descriptors[v] for v in ("baseline", "candidate"))
            for key in (
                "python",
                "platform",
                "machine",
                "cpu",
                "physical_cpus",
                "logical_cpus",
                "ram_bytes",
                "packages",
                "toolchain",
                "harness_sha256",
            ):
                require(left[key] == right[key], "different runtime: " + key)
        result = value["result"]
        if prime:
            require(result == {"primed": "native-content"}, "wrong primer")
        else:
            require(result["backend"] == "native-content" and result["mode"] == mode, "wrong native mode")
            require(
                result["scan_summary"]
                == {"found": args.pairs, "missing": 0, "empty": 0, "corrupt": 0, "total": args.pairs},
                "unexpected fixture scan counts",
            )
            require(result["native_fallback_count"] == 0 and result["diagnostic_count"] == 0, "fallback or diagnostics")
            require(result["workers"] == 7 and result["loader_workers"] == 0, "worker settings changed")
            expected = expected or result["output_sha256"]
            require(result["output_sha256"] == expected, "full labels, first batch or diagnostic parity failure")
        entry = {
            "version": version,
            "mode": mode,
            "round": repeat,
            "prime": prime,
            "path": str(output),
            "sha256": sha(output),
            "log_sha256": sha(log),
            "command": command,
            "report": value,
        }
        report["primers" if prime else "records"].append(entry)
        save()
        print(name, "complete" if prime else result["constructor_s"], flush=True)

    save()
    try:
        for mode in ("miss", "hit"):
            if mode == "hit":
                for version in anchors:
                    run(version, mode, -1, True)
                require(
                    report["primers"][0]["report"]["cache"]["sha256"]
                    == report["primers"][1]["report"]["cache"]["sha256"],
                    "primed native cache bytes differ",
                )
            for repeat in range(args.rounds):
                for version in ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline"):
                    run(version, mode, repeat)
        require(len(report["records"]) == args.rounds * 4 and len(report["primers"]) == 2, "incomplete phases")
        report["output_sha256"] = expected
        if args.rounds == 5:
            rows = {(v["mode"], v["round"], v["version"]): v["report"]["result"] for v in report["records"]}
            report["summary"] = {
                mode: {
                    metric: reader.summary(
                        [[rows[mode, r, "baseline"][metric], rows[mode, r, "candidate"][metric]] for r in range(5)]
                    )
                    for metric in ("constructor_s", "first_batch_total_s", "peak_rss_constructor_bytes")
                }
                for mode in ("miss", "hit")
            }
        report["complete"] = True
        save()
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
