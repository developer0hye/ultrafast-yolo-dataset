"""Audit the complete 500k-pair, 7/16-worker startup confirmation offline.

Run only after the campaign terminates: retained cache inspection reads hundreds
of MB. Imports only the standard library and an existing independent auditor,
never the benchmark producer, installed extension, or a recorded command.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import traceback
from zipfile import ZipFile

HELPER = Path(__file__).resolve().parents[1] / "docs/validation/audit-snapshot-startup-v1.py"
SPEC = importlib.util.spec_from_file_location("independent_startup_checks", HELPER)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)
require, sha, canonical = CHECK.require, CHECK.sha, CHECK.canonical
HARNESS = (
    "bench/startup_worker_trial.py",
    "bench/cache_startup.py",
    "bench/cache_read_comparison.py",
    "bench/startup.py",
    "bench/fixture_files.py",
    "bench/label_engine.py",
    "tests/reference.py",
)
METRICS = (
    "constructor_s",
    "first_batch_total_s",
    "peak_rss_constructor_bytes",
    "peak_rss_first_batch_bytes",
    "user_s",
    "system_s",
)


def decode(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)

    result = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    canonical(result)  # Also reject numeric overflow such as 1e999.
    return result


def read(path):
    return decode(path.read_text())


def number(value, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum, "invalid number")
    return value


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, "invalid integer")
    return value


def plan():
    rows = [dict(primer=True, pair=-1, position=i, workers=w) for i, w in enumerate((7, 16))]
    for pair in range(5):
        rows.extend(
            dict(primer=False, pair=pair, position=i, workers=w)
            for i, w in enumerate((7, 16) if pair % 2 == 0 else (16, 7))
        )
    return rows


def telemetry(path, rec, ram):
    observation = rec["host_observation"]
    require(observation["errors"] == [], "observer failed")
    require(Path(observation["path"]).name == path.name and sha(path) == observation["sha256"], "telemetry identity")
    samples = [decode(line) for line in path.read_text().splitlines()]
    require(integer(observation["samples"], 1) == len(samples), "telemetry count")
    last = rec["started_at_ns"]
    previous_cpu = None
    for sample in samples:
        at = integer(sample["at_ns"], 1)
        require(last <= at <= rec["finished_at_ns"], "telemetry chronology")
        last = at
        integer(sample["child_rss_bytes"], 1)
        integer(sample["child_threads"], 1)
        require(number(sample["available_memory"]) <= ram, "available RAM exceeds physical RAM")
        integer(sample["swap_used"])
        require(len(sample["loadavg"]) == 3, "load average shape")
        for value in sample["loadavg"]:
            number(value)
        for name in ("child_cpu_times", "host_cpu_times", "disk_io"):
            require(isinstance(sample[name], dict) and sample[name], "missing counter group")
            for value in sample[name].values():
                number(value)
        cpu = [sample["child_cpu_times"][k] for k in ("user", "system")]
        require(previous_cpu is None or all(a <= b for a, b in zip(previous_cpu, cpu)), "CPU counter reversal")
        previous_cpu = cpu
    gaps = [(b["at_ns"] - a["at_ns"]) / 1e9 for a, b in zip(samples, samples[1:])]
    return dict(
        samples=len(samples),
        sampled_whole_child_peak_rss_bytes=max(s["child_rss_bytes"] for s in samples),
        min_available_memory_bytes=min(s["available_memory"] for s in samples),
        max_swap_used_bytes=max(s["swap_used"] for s in samples),
        max_loadavg_1m=max(s["loadavg"][0] for s in samples),
        max_sample_gap_s=max(gaps, default=0),
        first_sample_delay_s=(samples[0]["at_ns"] - rec["started_at_ns"]) / 1e9,
        last_sample_to_exit_s=(rec["finished_at_ns"] - samples[-1]["at_ns"]) / 1e9,
    )


def audit(args):
    parent, launch, identity, reference, fixture = map(
        read,
        (
            args.series,
            args.launch,
            args.identity,
            args.reference_audit,
            args.fixture_manifest,
        ),
    )
    require(
        parent["complete"] is True and parent["passed"] is True and "error" not in parent, "unfinished/failed campaign"
    )
    require(
        (parent["planned_trials"], parent["planned_measured_trials"], parent["pairs"], parent["mode"])
        == (12, 10, 500000, "hit"),
        "campaign scope",
    )
    require(canonical(parent["plan"]) == canonical(plan()) and len(parent["records"]) == 12, "campaign plan/count")
    require(parent["script_sha256"] == launch["controller_sha256"] == sha(args.controller), "controller identity")
    require(parent["pid"] == launch["controller_pid"], "controller PID")
    require(
        parent["source_files"] == launch["source_files"] and set(parent["source_files"]) == set(HARNESS),
        "harness inventory",
    )
    harness = {name: sha(args.source / name) for name in HARNESS}
    require(harness == parent["source_files"], "frozen harness bytes")
    for field, path in (
        ("phase_audit_sha256", args.phase_audit),
        ("pilot_qualification_sha256", args.pilot_qualification),
    ):
        require(parent[field] == launch[field] == sha(path), "qualification anchor")
        require(read(path)["passed"] is True, "qualification failed")
    phase, pilot, runtime_reference = map(read, (args.phase_audit, args.pilot_qualification, args.runtime_reference))
    require((phase["trials"], phase["measured_trials"]) == (21, 18), "phase qualification scope")
    require(
        pilot["complete"] is True
        and len(pilot["records"]) == 6
        and all(
            r["validated"] is True and type(r["returncode"]) is int and r["returncode"] == 0 for r in pilot["records"]
        ),
        "pilot qualification scope",
    )
    require(
        sum(r["raw_sha256"] == sha(args.runtime_reference) for r in pilot["records"]) == 1,
        "runtime reference is not a qualified pilot",
    )
    require(
        runtime_reference["complete"] is True
        and runtime_reference["passed"] is True
        and "error" not in runtime_reference,
        "failed runtime reference",
    )
    require(canonical(parent["descriptor"]) == canonical(runtime_reference["descriptor"]), "qualified runtime drift")
    require(parent["original_output_audit_sha256"] == sha(args.reference_audit), "reference anchor")
    require(
        reference["complete_requested_samples"] is True
        and (reference["phase"], reference["task"], reference["pairs"]) == ("p3", "detect", 500000),
        "reference scope",
    )
    require(
        reference["fixture_manifest_sha256"] == sha(args.fixture_manifest) and parent["fixture"] == fixture,
        "fixture anchor",
    )
    require(
        fixture["count"] == 500000 and fixture["task"] == "detect" and fixture["fingerprint"]["files"] == 1000000,
        "fixture size/task",
    )
    expected = read(args.expected_outputs)
    require(expected == reference["output_sha256"] == parent["expected_output_sha256"], "reference outputs")
    require(
        set(expected) == {"labels", "first_batch", "scan_summary_and_messages"}
        and all(CHECK.digest(v) for v in expected.values()),
        "output hash shape",
    )
    require(
        identity["wheel_sha256"] == sha(args.wheel) and identity["toolchain"]["profile"] == "release", "qualified wheel"
    )
    paths = [args.library_source / n for n in ("Cargo.toml", "Cargo.lock", "pyproject.toml")]
    paths += sorted((args.library_source / "src").rglob("*.rs"))
    package = args.library_source / "python/ultrafast_yolo_dataset"
    paths += sorted(package.glob("*.py")) + sorted(package.glob("*.json"))
    sources = {str(p.relative_to(args.library_source)): sha(p) for p in paths}
    require(sources == identity["library_sources"], "qualified source inventory/bytes")
    with ZipFile(args.wheel) as wheel:
        natives = [
            n
            for n in wheel.namelist()
            if n.startswith("ultrafast_yolo_dataset/_native.") and n.endswith((".so", ".pyd"))
        ]
        require(
            len(natives) == 1 and hashlib.sha256(wheel.read(natives[0])).hexdigest() == identity["extension_sha256"],
            "wheel native bytes",
        )
        for name, digest in sources.items():
            if name.startswith("python/"):
                require(hashlib.sha256(wheel.read(name[7:])).hexdigest() == digest, "wheel Python bytes")
    cache_sha = sha(args.original_cache)
    require(cache_sha == launch["original_cache_sha256"] == parent["original_cache_sha256"], "original cache bytes")
    cache = CHECK.inspect_cache(args.original_cache)
    profile = hashlib.sha256()
    for name in CHECK.EMBEDDED:
        profile.update((args.library_source / name).read_bytes())
    require(cache["profile"] == profile.hexdigest(), "cache native profile")
    caches = {7: args.cache_w7, 16: args.cache_w16}
    for path in caches.values():
        require(path.resolve() != args.original_cache.resolve() and not path.is_symlink(), "private cache required")
        require(sha(path) == cache_sha, "retained private cache bytes")
    require(args.cache_w7.resolve() != args.cache_w16.resolve(), "distinct private caches")
    require(
        parent["environment_overrides"]
        == dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", YOLO_OFFLINE="true"),
        "thread environment",
    )
    rows, artifacts = [], {}
    previous_finish = integer(parent["started_at_ns"], 1)
    common_command = None
    for index, (spec, rec) in enumerate(zip(plan(), parent["records"], strict=True)):
        require(canonical(rec["spec"]) == canonical(spec), "record order")
        require(
            rec["validated"] is True
            and type(rec["returncode"]) is int
            and rec["returncode"] == 0
            and not rec.get("timed_out", False),
            "failed child",
        )
        start, finish = integer(rec["started_at_ns"], 1), integer(rec["finished_at_ns"], 1)
        require(previous_finish <= start < finish, "overlapping/reordered child")
        previous_finish = finish
        path = args.runs / f"{index:02d}-w{spec['workers']}.json"
        require(Path(rec["path"]).name == path.name and Path(rec["path"]).is_absolute(), "raw path")
        for file, digest in ((path, rec["raw_sha256"]), (path.with_suffix(".log"), rec["log_sha256"])):
            require(sha(file) == digest, "raw/log bytes")
            artifacts[file.name] = digest
        command = rec["command"]
        flags = (
            "corpus",
            "harness-root",
            "library-source",
            "wheel",
            "identity",
            "cache-root",
            "expected-outputs",
            "out",
            "workers",
            "mode",
            "expected-cache-sha256",
        )
        require(len(command) == 24 and command[2::2] == ["--" + f for f in flags], "command shape")
        opts = dict(zip(flags, command[3::2], strict=True))
        require(all(Path(p).is_absolute() for p in command[:2]), "absolute executable paths")
        require(
            command[1] == str(Path(parent["source"]) / HARNESS[0]) and opts["harness-root"] == parent["source"],
            "worker invocation",
        )
        require(
            opts["out"] == rec["path"]
            and opts["workers"] == str(spec["workers"])
            and opts["mode"] == "hit"
            and opts["expected-cache-sha256"] == cache_sha,
            "trial arguments",
        )
        require(Path(opts["cache-root"]).name == caches[spec["workers"]].parent.name, "cache assignment")
        require(not Path(opts["cache-root"]).is_relative_to(Path(opts["corpus"])), "cache inside corpus")
        for key, selected in (
            ("corpus", args.fixture_manifest.parent),
            ("library-source", args.library_source),
            ("wheel", args.wheel),
            ("identity", args.identity),
            ("expected-outputs", args.expected_outputs),
        ):
            require(Path(opts[key]).name == selected.name, "selected artifact/command mismatch: " + key)
        common = command[:2] + [opts[k] for k in flags if k not in ("cache-root", "out", "workers")]
        common_command = common if common_command is None else common_command
        require(common == common_command, "command drift")
        if index == 0:
            require(command == launch["record"]["command"] and rec["pid"] == launch["record"]["pid"], "launch binding")
        raw = read(path)
        require(raw["complete"] is True and raw["passed"] is True and "error" not in raw, "raw failed")
        require(
            integer(raw["pid"], 1) == rec["pid"]
            and raw["mode"] == "hit"
            and integer(raw["workers"], 1) == spec["workers"],
            "raw identity/settings",
        )
        rs, rf = integer(raw["started_at_ns"], 1), integer(raw["finished_at_ns"], 1)
        require(start <= rs < rf <= finish, "raw timestamps")
        require(
            raw["script_sha256"] == harness[HARNESS[0]]
            and raw["harness_sources"] == {k: v for k, v in harness.items() if k != HARNESS[0]},
            "raw sources",
        )
        require(raw["expected_outputs_sha256"] == sha(args.expected_outputs), "raw output anchor")
        descriptor = raw["descriptor"]
        require(canonical(descriptor) == canonical(parent["descriptor"]), "runtime drift")
        for key in ("wheel_sha256", "extension_sha256", "library_sources", "toolchain"):
            require(descriptor[key] == identity[key], "runtime build binding")
        require(
            descriptor["build_identity_sha256"] == sha(args.identity)
            and descriptor["harness_sha256"] == harness["bench/cache_read_comparison.py"],
            "runtime anchor",
        )
        require(
            raw["cache"]
            == {k: cache[k] for k in ("bytes", "sha256", "sections")} | {"file": caches[spec["workers"]].name},
            "cache container receipt",
        )
        result = raw["result"]
        require(canonical(result) == canonical(rec["result"]), "parent/result mismatch")
        integer(result["workers"], 1)
        integer(result["loader_workers"])
        for value in result["scan_summary"].values():
            integer(value)
        for name in ("diagnostic_count", "native_fallback_count", "minor_faults", "major_faults", "cache_bytes"):
            integer(result[name])
        CHECK.check_result(result, "hit", 500000, spec["workers"], expected, (rf - rs) / 1e9, cache)
        observed = telemetry(path.with_suffix(".host.jsonl"), rec, integer(descriptor["ram_bytes"], 1))
        artifacts[path.with_suffix(".host.jsonl").name] = rec["host_observation"]["sha256"]
        rows.append(
            dict(
                index=index, **spec, **{k: result[k] for k in METRICS}, stages_s=result["stages_s"], telemetry=observed
            )
        )
    require(integer(parent["finished_at_ns"], 1) >= previous_finish, "parent finish")
    measured = [r for r in rows if not r["primer"]]
    blocks = [{r["workers"]: r for r in measured if r["pair"] == i} for i in range(5)]
    summary = {metric: CHECK.summary([[b[7][metric], b[16][metric]] for b in blocks]) for metric in METRICS[:4]}
    return dict(
        passed=True,
        trials=12,
        measured_trials=10,
        pairs=500000,
        mode="hit",
        rows=rows,
        summary=summary,
        medians={
            str(w): {m: statistics.median(r[m] for r in measured if r["workers"] == w) for m in METRICS}
            for w in (7, 16)
        },
        descriptor=parent["descriptor"],
        artifacts=artifacts,
        series_sha256=sha(args.series),
        auditor_sha256=sha(Path(__file__)),
        helper_sha256=sha(HELPER),
        anchors={
            k: sha(getattr(args, k))
            for k in (
                "launch",
                "controller",
                "identity",
                "reference_audit",
                "fixture_manifest",
                "expected_outputs",
                "phase_audit",
                "pilot_qualification",
                "runtime_reference",
                "wheel",
            )
        },
        scope="Ratio of medians over five matched pairs with all 5**5 paired bootstrap resamples; exploratory shared-host result. Primers excluded. Process high-water RSS includes imports/preflight; sampled whole-child telemetry does not isolate startup or bound unsampled peaks. Artifact audit trusts frozen worker checks for full input fingerprints and installed-runtime execution; it does not rerun the corpus or prove other hosts/tasks/cache-miss performance.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "series",
        "launch",
        "controller",
        "identity",
        "reference-audit",
        "fixture-manifest",
        "expected-outputs",
        "phase-audit",
        "pilot-qualification",
        "runtime-reference",
        "wheel",
        "source",
        "library-source",
        "original-cache",
        "cache-w7",
        "cache-w16",
        "runs",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    # Reserve outside the failure handler: failed/incomplete receipts survive retries.
    with args.out.open("x") as stream:
        json.dump(dict(passed=False, complete=False), stream)
    try:
        result = audit(args)
    except BaseException:
        args.out.write_text(
            json.dumps(dict(passed=False, complete=True, error=traceback.format_exc()), indent=2) + "\n"
        )
        raise
    args.out.write_text(json.dumps(dict(complete=True, **result), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
