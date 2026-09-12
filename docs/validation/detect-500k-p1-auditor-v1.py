"""Audit frozen P1/P3/P4 reports without importing the measured ML libraries.

The caller declares the task, pair count, repetitions and worker count. Partial
reports require explicit opt-in and produce no aggregate. Identity and preflight
files are trusted comparison anchors, not signed attestations or input re-reads.
"""

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def number(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_summary(pairs):
    count = len(pairs)
    # At the planned five pairs, enumerate all 5**5 resamples. This avoids
    # depending on NumPy's RNG or accepting the parent harness's random interval.
    exact = count <= 5
    if exact:
        selections = itertools.product(range(count), repeat=count)
    else:
        rng = random.Random(912)
        selections = ([rng.randrange(count) for _ in pairs] for _ in range(10000))
    estimates = []
    for indices in selections:
        selected = [pairs[i] for i in indices]
        estimates.append(statistics.median(p[0] for p in selected) / statistics.median(p[1] for p in selected))
    reference = statistics.median(p[0] for p in pairs)
    candidate = statistics.median(p[1] for p in pairs)
    return {
        "pairs": count,
        "reference_median": reference,
        "candidate_median": candidate,
        "reference_over_candidate": reference / candidate,
        "paired_bootstrap_ci95": [quantile(estimates, 0.025), quantile(estimates, 0.975)],
        "bootstrap": "all ordered resamples" if exact else "10000 random.Random(912) ordered resamples",
        "resamples": len(estimates),
    }


def audit(
    report_path,
    identity_path,
    preflight_path,
    fixture_path,
    phase,
    task,
    pairs,
    repeats,
    workers,
    runs_dir=None,
    partial=False,
):
    require(phase in ("p1", "p3", "p4") and task in ("detect", "segment"), "unsupported protocol")
    require(all(type(v) is int and v > 0 for v in (pairs, repeats, workers)), "invalid protocol counts")
    report_bytes = report_path.read_bytes()
    identity_bytes, preflight_bytes = identity_path.read_bytes(), preflight_path.read_bytes()
    report, identity, preflight = map(json.loads, (report_bytes, identity_bytes, preflight_bytes))
    fixture_bytes = fixture_path.read_bytes()
    fixture = json.loads(fixture_bytes)
    require(report["source_sha256"] == identity["source_sha256"], "source identity mismatch")
    require(report["source_sha256"] and all(digest(v) for v in report["source_sha256"].values()), "bad source hashes")
    require(
        digest(report["native_extension_sha256"])
        and report["native_extension_sha256"] == identity["native_extension_sha256"],
        "native identity mismatch",
    )
    require(preflight["pairs"] == pairs, "wrong preflight pair count")
    require(preflight["counts"] == {"images": pairs, "labels": pairs}, "wrong preflight file counts")
    require(preflight["all_regular_single_link_files"] is True, "unverified distinct files")
    require(preflight["fingerprint"]["files"] == 2 * pairs, "wrong preflight total files")
    require(digest(preflight["fingerprint"]["sha256"]), "bad fixture digest")
    require(fixture["kind"] == "ultrafast-yolo-startup-synthetic-v1", "wrong fixture kind")
    require(fixture["task"] == task and fixture["count"] == pairs, "wrong fixture manifest task/count")
    require(fixture["fingerprint"] == preflight["fingerprint"], "fixture/preflight mismatch")
    corpus = report["corpus"]
    require(corpus["count"] == pairs and corpus["task"] == task, "wrong corpus task/count")
    if phase == "p1":
        require(corpus == fixture["labels"], "wrong P1 label manifest")
        require(corpus["distinct_files"] == pairs and corpus["synthetic"] is True, "wrong P1 fixture")
        # P1 serializes the nested label manifest, not the full startup manifest.
        require(digest(corpus["sha256_names_and_contents"]), "bad label fixture digest")
    else:
        require(corpus == fixture, "wrong startup manifest")
        require(corpus["kind"] == "ultrafast-yolo-startup-synthetic-v1", "not the declared synthetic suite")
        require(corpus["fingerprint"] == preflight["fingerprint"], "fixture fingerprint mismatch")
        require(corpus["labels"]["count"] == pairs and corpus["labels"]["task"] == task, "nested fixture mismatch")
    backends = {
        "p1": ("reference", "native"),
        "p3": ("reference", "native-content"),
        "p4": ("reference-content", "native-content"),
    }[phase]
    plan = [(r, backend) for r in range(repeats) for backend in (backends if r % 2 == 0 else backends[::-1])]
    records = report["results"]
    actual = [(r["round"], r["backend"]) for r in records]
    require(records and actual == plan[: len(actual)], "missing, duplicate, extra or reordered samples")
    complete = len(actual) == len(plan)
    require(partial or complete, "incomplete requested sample count")
    require(partial or not complete or phase == "p1" or "summary" in report, "parent aggregation did not finish")
    expected_output, validated, raw_files = None, {}, {}
    raw_dir = runs_dir or report_path.with_suffix(".runs")
    if phase == "p4":
        for backend in backends:
            name = f"prime-{backend}.json"
            data = (raw_dir / name).read_bytes()
            require(json.loads(data) == {"primed": backend}, "missing or wrong separate cache primer")
            raw_files[name] = sha(data)
    for row in records:
        key = (row["round"], row["backend"])
        require(row["parity"] is True and row["workers"] == workers, "worker configuration or parity mismatch")
        require(number(row["user_s"]) and number(row["system_s"]), "invalid CPU time")
        require(number(row["available_ram_bytes"], 1), "invalid RAM sample")
        require(len(row["loadavg"]) == 3 and all(number(v) for v in row["loadavg"]), "invalid load sample")
        if phase == "p1":
            require(number(row["wall_s"], 1e-12), "invalid P1 time")
            require(
                math.isclose(row["labels_per_s"], pairs / row["wall_s"], rel_tol=1e-12), "wrong P1 throughput/count"
            )
            require(row["reference_required"] == 0, "P1 fallback present")
            require(digest(row["output_sha256"]), "bad P1 output digest")
            require(number(row["peak_rss_bytes"], 1), "invalid P1 RSS")
            require(
                row["rss_method"] in ("/proc/self/status VmHWM", "getrusage ru_maxrss bytes"), "unknown P1 RSS method"
            )
        else:
            name = f"{key[0]}-{key[1]}.json"
            data = (raw_dir / name).read_bytes()
            require(
                canonical(json.loads(data))
                == canonical({k: v for k, v in row.items() if k not in ("round", "parity")}),
                "raw/parent disagreement",
            )
            raw_files[name] = sha(data)
            require(
                row["mode"] == ("hit" if phase == "p4" else "miss") and row["loader_workers"] == 0,
                "wrong cache/loader mode",
            )
            require(
                number(row["constructor_s"], 1e-12) and number(row["first_batch_total_s"], row["constructor_s"]),
                "invalid startup time",
            )
            require(all(number(v) for v in row["stages_s"].values()), "invalid stage time")
            expected_stages = set()
            if key[1] == "native-content":
                expected_stages = {
                    "_validate_inputs",
                    "to_ultralytics_labels",
                    "_decode" if phase == "p4" else "_encode",
                }
            elif phase == "p4":
                expected_stages = {"input_fingerprints", "reference_cache_decode"}
            require(set(row["stages_s"]) == expected_stages, "cache path or materialization not observed")
            require(all(number(v, 1e-12) for v in row["stages_s"].values()), "empty observed stage")
            require(
                row["scan_summary"] == {"found": pairs, "missing": 0, "empty": 0, "corrupt": 0, "total": pairs},
                "incomplete or unexpected scan",
            )
            require(row["diagnostic_count"] == 0, "unexpected synthetic diagnostics")
            require(
                row["native_fallback_count"] == (0 if key[1] == "native-content" else None), "unexpected fallback count"
            )
            require(
                set(row["output_sha256"]) == {"labels", "first_batch", "scan_summary_and_messages"},
                "incomplete output hashes",
            )
            require(all(digest(v) for v in row["output_sha256"].values()), "bad output hash")
            peaks = [
                row[k]
                for k in (
                    "peak_rss_before_constructor_bytes",
                    "peak_rss_constructor_bytes",
                    "peak_rss_first_batch_bytes",
                )
            ]
            require(all(number(v, 1) for v in peaks) and peaks == sorted(peaks), "invalid RSS high-water sequence")
            require(
                number(row["rss_before_constructor_bytes"], 1) and number(row["cache_bytes"], 1),
                "invalid initial RSS/cache size",
            )
        expected_output = expected_output or row["output_sha256"]
        require(row["output_sha256"] == expected_output, "full output parity mismatch")
        validated[key] = row
    result = {
        "complete_requested_samples": complete,
        "phase": phase,
        "task": task,
        "pairs": pairs,
        "repeats": repeats,
        "workers": workers,
        "completed_workers": len(records),
        "report_sha256": sha(report_bytes),
        "identity_sha256": sha(identity_bytes),
        "preflight_sha256": sha(preflight_bytes),
        "fixture_manifest_sha256": sha(fixture_bytes),
        "auditor_sha256": sha(Path(__file__).read_bytes()),
        "raw_sha256": raw_files,
        "output_sha256": expected_output,
        "limits": [
            "Compares supplied immutable artifacts; does not rerun or re-hash the full input corpus.",
            "P1 worker records are embedded in the parent; no independent per-worker JSON was retained by that harness.",
            "Peak RSS includes imports, preflight and allocator history; it is not working allocation or process-family RSS.",
            "Parent median/p95 values are checked; its random bootstrap endpoints are replaced by an independent calculation.",
        ],
    }
    if complete and not partial:
        metrics = (
            ("wall_s", "peak_rss_bytes")
            if phase == "p1"
            else ("constructor_s", "first_batch_total_s", "peak_rss_constructor_bytes")
        )
        result["summary"] = {}
        for metric in metrics:
            paired = [
                (validated[(r, backends[0])][metric], validated[(r, backends[1])][metric]) for r in range(repeats)
            ]
            if phase != "p1":
                for column, backend in enumerate(backends):
                    values = [v[column] for v in paired]
                    summary = report["summary"][metric][backend]
                    require(
                        math.isclose(summary["median"], statistics.median(values), rel_tol=1e-12),
                        "parent median mismatch",
                    )
                    require(math.isclose(summary["p95"], quantile(values, 0.95), rel_tol=1e-12), "parent p95 mismatch")
            result["summary"][metric] = paired_summary(paired)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=("p1", "p3", "p4"), required=True)
    parser.add_argument("--task", choices=("detect", "segment"), required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "retain earlier audit reports")
    result = audit(
        args.report,
        args.identity,
        args.preflight,
        args.fixture_manifest,
        args.phase,
        args.task,
        args.pairs,
        args.repeats,
        args.workers,
        args.runs_dir,
        args.allow_partial,
    )
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "complete_requested_samples": result["complete_requested_samples"],
                "completed_workers": result["completed_workers"],
            }
        )
    )


if __name__ == "__main__":
    main()
