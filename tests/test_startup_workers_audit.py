"""Artificial artifact graphs test the auditor, never benchmark performance."""

from collections import Counter
import copy
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

SPEC = importlib.util.spec_from_file_location(
    "startup_workers_audit", Path(__file__).resolve().parents[1] / "bench/audit_startup_workers.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def fixture(root):
    args = SimpleNamespace(
        **{
            n: root / (n + ".json")
            for n in (
                "series",
                "launch",
                "identity",
                "reference_audit",
                "expected_outputs",
                "phase_audit",
                "pilot_qualification",
                "runtime_reference",
                "out",
            )
        }
    )
    args.source, args.library_source, args.runs = (root / n for n in ("source", "library", "runs"))
    args.controller = root / "controller.py"
    args.controller.write_text("# synthetic controller, never executed\n")
    args.fixture_manifest = root / "corpus/startup.json"
    args.wheel = root / "synthetic.whl"
    for name in AUDIT.HARNESS:
        p = args.source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# synthetic harness " + name + "\n")
    harness = {n: AUDIT.sha(args.source / n) for n in AUDIT.HARNESS}
    sources = {}
    for name in (*AUDIT.CHECK.EMBEDDED, "pyproject.toml", "python/ultrafast_yolo_dataset/__init__.py"):
        p = args.library_source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# synthetic source " + name + "\n")
        sources[name] = AUDIT.sha(p)
    with ZipFile(args.wheel, "w") as wheel:
        wheel.writestr("ultrafast_yolo_dataset/_native.synthetic.so", b"synthetic, never imported")
        wheel.write(
            args.library_source / "python/ultrafast_yolo_dataset/__init__.py", "ultrafast_yolo_dataset/__init__.py"
        )
    identity = dict(
        wheel_sha256=AUDIT.sha(args.wheel),
        extension_sha256=hashlib.sha256(b"synthetic, never imported").hexdigest(),
        library_sources=sources,
        toolchain={"profile": "release"},
    )
    write(args.identity, identity)
    descriptor = dict(
        **identity,
        build_identity_sha256=AUDIT.sha(args.identity),
        harness_sha256=harness["bench/cache_read_comparison.py"],
        ram_bytes=16 * 1024**3,
        cpu="synthetic",
        python="synthetic",
        packages=[],
    )
    profile = hashlib.sha256(b"".join((args.library_source / n).read_bytes() for n in AUDIT.CHECK.EMBEDDED)).hexdigest()
    sections = [json.dumps({"profile": profile}).encode()] + [b""] * 10
    cache_bytes = b"UYDCACHE" + struct.pack("<II", 1, 11)
    cache_bytes += b"".join(struct.pack("<Q", len(s)) + hashlib.sha256(s).digest() for s in sections)
    cache_bytes += b"".join(sections)
    for name in ("original_cache", "cache_w7", "cache_w16"):
        p = root / name / "fixture.uydcache"
        p.parent.mkdir()
        p.write_bytes(cache_bytes)
        setattr(args, name, p)
    cache = dict(
        file="fixture.uydcache",
        bytes=len(cache_bytes),
        sha256=AUDIT.sha(args.original_cache),
        sections=[dict(bytes=len(s), sha256=hashlib.sha256(s).hexdigest()) for s in sections],
    )
    expected = {key: str(i) * 64 for i, key in enumerate(("labels", "first_batch", "scan_summary_and_messages"), 1)}
    write(args.expected_outputs, expected)
    fixture_manifest = dict(count=500000, task="detect", fingerprint=dict(files=1000000, bytes=123, sha256="4" * 64))
    write(args.fixture_manifest, fixture_manifest)
    write(
        args.reference_audit,
        dict(
            complete_requested_samples=True,
            phase="p3",
            task="detect",
            pairs=500000,
            fixture_manifest_sha256=AUDIT.sha(args.fixture_manifest),
            output_sha256=expected,
        ),
    )
    write(args.phase_audit, dict(passed=True, trials=21, measured_trials=18))
    write(args.runtime_reference, dict(complete=True, passed=True, descriptor=descriptor))
    write(
        args.pilot_qualification,
        dict(
            complete=True,
            passed=True,
            records=[
                dict(
                    validated=True,
                    returncode=0,
                    raw_sha256=AUDIT.sha(args.runtime_reference) if i == 0 else str(i) * 64,
                )
                for i in range(6)
            ],
        ),
    )
    declared = [dict(primer=True, pair=-1, position=i, workers=w) for i, w in enumerate([7, 16])]
    for i, order in enumerate(([7, 16], [16, 7], [7, 16], [16, 7], [7, 16])):
        declared.extend(dict(primer=False, pair=i, position=j, workers=w) for j, w in enumerate(order))
    parent = dict(
        complete=True,
        passed=True,
        pid=99,
        started_at_ns=10**9,
        finished_at_ns=300 * 10**9,
        planned_trials=12,
        planned_measured_trials=10,
        pairs=500000,
        mode="hit",
        plan=declared,
        script_sha256=AUDIT.sha(args.controller),
        source=str(args.source),
        source_files=harness,
        phase_audit_sha256=AUDIT.sha(args.phase_audit),
        pilot_qualification_sha256=AUDIT.sha(args.pilot_qualification),
        original_output_audit_sha256=AUDIT.sha(args.reference_audit),
        fixture=fixture_manifest,
        expected_output_sha256=expected,
        original_cache_sha256=cache["sha256"],
        descriptor=descriptor,
        environment_overrides=dict(
            OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", YOLO_OFFLINE="true"
        ),
        records=[],
    )
    for i, spec in enumerate(declared):
        start, scale = (20 * i + 2) * 10**9, 1 if spec["workers"] == 7 else 2
        result = dict(
            backend="native-content",
            mode="hit",
            workers=spec["workers"],
            loader_workers=0,
            constructor_s=10 / scale,
            first_batch_total_s=12 / scale,
            stages_s={"_validate_inputs": 6 / scale, "_decode": 1, "to_ultralytics_labels": 0.5},
            rss_before_constructor_bytes=80,
            peak_rss_before_constructor_bytes=100,
            peak_rss_constructor_bytes=200,
            peak_rss_first_batch_bytes=250,
            scan_summary=dict(found=500000, missing=0, empty=0, corrupt=0, total=500000),
            diagnostic_count=0,
            native_fallback_count=0,
            output_sha256=expected,
            cache_bytes=cache["bytes"],
            user_s=1,
            system_s=2,
            minor_faults=4,
            major_faults=0,
            available_ram_bytes=1024,
            loadavg=[1, 2, 3],
        )
        path = args.runs / f"{i:02d}-w{spec['workers']}.json"
        raw = dict(
            complete=True,
            passed=True,
            pid=i + 100,
            started_at_ns=start + 100000000,
            finished_at_ns=start + 14000000000,
            workers=spec["workers"],
            mode="hit",
            script_sha256=harness[AUDIT.HARNESS[0]],
            descriptor=descriptor,
            harness_sources={k: v for k, v in harness.items() if k != AUDIT.HARNESS[0]},
            expected_outputs_sha256=AUDIT.sha(args.expected_outputs),
            cache=cache,
            result=result,
        )
        write(path, raw)
        log, host = path.with_suffix(".log"), path.with_suffix(".host.jsonl")
        log.write_text("synthetic log\n")
        samples = [
            dict(
                at_ns=start + (j + 1) * 10**9,
                child_rss_bytes=100 + j,
                child_threads=spec["workers"],
                available_memory=1024,
                swap_used=0,
                loadavg=[1, 2, 3],
                child_cpu_times=dict(user=j, system=j),
                host_cpu_times=dict(user=j, system=j, idle=j),
                disk_io=dict(read_bytes=j, write_bytes=j),
            )
            for j in range(3)
        ]
        host.write_text("".join(json.dumps(s) + "\n" for s in samples))
        opts = dict(
            corpus=args.fixture_manifest.parent,
            **{
                "harness-root": args.source,
                "library-source": args.library_source,
                "wheel": args.wheel,
                "identity": args.identity,
                "cache-root": getattr(args, "cache_w" + str(spec["workers"])).parent,
                "expected-outputs": args.expected_outputs,
                "out": path,
                "workers": spec["workers"],
                "mode": "hit",
                "expected-cache-sha256": cache["sha256"],
            },
        )
        command = [str(root / "python"), str(args.source / AUDIT.HARNESS[0])]
        for key, value in opts.items():
            command.extend(["--" + key, str(value)])
        parent["records"].append(
            dict(
                spec=spec,
                validated=True,
                returncode=0,
                pid=raw["pid"],
                path=str(path),
                command=command,
                started_at_ns=start,
                finished_at_ns=start + 15 * 10**9,
                raw_sha256=AUDIT.sha(path),
                log_sha256=AUDIT.sha(log),
                result=result,
                host_observation=dict(path=str(host), sha256=AUDIT.sha(host), samples=3, errors=[]),
            )
        )
    write(args.series, parent)
    write(
        args.launch,
        dict(
            controller_sha256=parent["script_sha256"],
            controller_pid=99,
            source_files=harness,
            phase_audit_sha256=parent["phase_audit_sha256"],
            pilot_qualification_sha256=parent["pilot_qualification_sha256"],
            original_cache_sha256=cache["sha256"],
            record=parent["records"][0],
        ),
    )
    return args


def test_complete_graph(tmp_path):
    report = AUDIT.audit(fixture(tmp_path))
    assert report["passed"] is True and report["trials"] == 12 and report["measured_trials"] == 10
    assert report["summary"]["constructor_s"]["baseline_over_candidate"] == 2
    assert report["summary"]["constructor_s"]["paired_bootstrap_ci95"] == [2, 2]
    assert report["medians"]["7"]["peak_rss_constructor_bytes"] == 200
    assert len(report["artifacts"]) == 36


@pytest.mark.parametrize(
    "damage",
    [
        "incomplete",
        "missing_trial",
        "reordered",
        "failed_child",
        "timeout",
        "wrong_workers",
        "wrong_labels",
        "cache_miss",
        "fallback",
        "bool_counter",
        "impossible_time",
        "stage_overflow",
        "rss_reversal",
        "parent_mismatch",
        "raw_timestamp",
        "overlap",
        "command_drift",
        "changed_source",
        "changed_cache",
        "changed_wheel",
        "changed_log",
        "runtime_drift",
        "samples_missing",
        "samples_count",
        "sample_time",
        "cpu_reversal",
        "ram_overflow",
        "observer_error",
        "invalid_pilot",
        "bad_cache_section",
    ],
)
def test_damaged_graph_rejected(tmp_path, damage):
    args = fixture(tmp_path)
    parent = AUDIT.read(args.series)
    rec = parent["records"][4]
    path = Path(rec["path"])
    raw = AUDIT.read(path)
    result = raw["result"]
    host = path.with_suffix(".host.jsonl")
    samples = [json.loads(line) for line in host.read_text().splitlines()]
    if damage == "incomplete":
        parent["complete"] = False
    elif damage == "missing_trial":
        parent["records"].pop()
    elif damage == "reordered":
        parent["records"][3], parent["records"][4] = parent["records"][4], parent["records"][3]
    elif damage == "failed_child":
        rec["returncode"] = -6
    elif damage == "timeout":
        rec["timed_out"] = True
    elif damage == "wrong_workers":
        result["workers"] = 4
    elif damage == "wrong_labels":
        result["output_sha256"]["labels"] = "a" * 64
    elif damage == "cache_miss":
        result["mode"] = "miss"
    elif damage == "fallback":
        result["native_fallback_count"] = 1
    elif damage == "bool_counter":
        result["scan_summary"]["missing"] = False
    elif damage == "impossible_time":
        result["first_batch_total_s"] = 500
    elif damage == "stage_overflow":
        result["stages_s"]["_validate_inputs"] = 500
    elif damage == "rss_reversal":
        result["peak_rss_constructor_bytes"] = 50
    elif damage == "raw_timestamp":
        raw["finished_at_ns"] = rec["finished_at_ns"] + 1
    elif damage == "overlap":
        rec["started_at_ns"] = parent["records"][3]["finished_at_ns"] - 1
    elif damage == "command_drift":
        rec["command"][0] = str(tmp_path / "other-python")
    elif damage == "changed_source":
        (args.source / "bench/startup.py").write_text("changed")
    elif damage == "changed_cache":
        args.cache_w16.write_bytes(b"changed")
    elif damage == "changed_wheel":
        args.wheel.write_bytes(b"changed")
    elif damage == "changed_log":
        path.with_suffix(".log").write_text("changed")
    elif damage == "runtime_drift":
        raw["descriptor"]["cpu"] = "different"
    elif damage == "samples_missing":
        samples = []
    elif damage == "samples_count":
        rec["host_observation"]["samples"] = 8
    elif damage == "sample_time":
        samples[-1]["at_ns"] = rec["finished_at_ns"] + 1
    elif damage == "cpu_reversal":
        samples[-1]["child_cpu_times"]["user"] = 0
    elif damage == "ram_overflow":
        samples[0]["available_memory"] = 2**62
    elif damage == "observer_error":
        rec["host_observation"]["errors"] = ["synthetic observer failure"]
    elif damage == "invalid_pilot":
        args.runtime_reference.write_text("{}")
    elif damage == "bad_cache_section":
        raw["cache"]["sections"][0]["sha256"] = "0" * 64
    # Update file receipts so rejection must inspect content, not just checksum.
    rec["result"] = copy.deepcopy(result)
    if damage == "parent_mismatch":
        rec["result"]["constructor_s"] += 1
    write(path, raw)
    rec["raw_sha256"] = AUDIT.sha(path)
    host.write_text("".join(json.dumps(s) + "\n" for s in samples))
    rec["host_observation"]["sha256"] = AUDIT.sha(host)
    write(args.series, parent)
    with pytest.raises((ValueError, KeyError, OSError)):
        AUDIT.audit(args)


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'])
def test_ambiguous_json_rejected(text):
    with pytest.raises(ValueError):
        AUDIT.decode(text)


def test_independent_bootstrap_oracle():
    pairs = [[12, 9], [18, 11], [10, 9], [15, 14], [21, 13]]
    oracle = []
    for selected in itertools.combinations_with_replacement(range(5), 5):
        multiplicity = math.factorial(5)
        for count in Counter(selected).values():
            multiplicity //= math.factorial(count)
        ratio = statistics.median(pairs[i][0] for i in selected) / statistics.median(pairs[i][1] for i in selected)
        oracle.extend([ratio] * multiplicity)
    oracle.sort()
    assert len(oracle) == 3125
    bounds = []
    for q in (0.025, 0.975):
        i = q * (len(oracle) - 1)
        bounds.append(oracle[int(i)] * (1 - (i % 1)) + oracle[math.ceil(i)] * (i % 1))
    assert AUDIT.CHECK.summary(pairs)["paired_bootstrap_ci95"] == pytest.approx(bounds)


def test_failed_cli_receipt_cannot_be_overwritten(tmp_path):
    args = fixture(tmp_path)
    parent = AUDIT.read(args.series)
    parent["complete"] = False
    write(args.series, parent)
    command = [sys.executable, str(Path(AUDIT.__file__))]
    for key, value in vars(args).items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode != 0 and AUDIT.read(args.out)["passed"] is False
    assert "unfinished/failed campaign" in AUDIT.read(args.out)["error"]
    original = args.out.read_bytes()
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0 and args.out.read_bytes() == original
