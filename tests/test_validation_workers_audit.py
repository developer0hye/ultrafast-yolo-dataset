"""Synthetic artifact controls; these fixtures are not measured benchmarks."""

from collections import Counter, defaultdict
import importlib.util
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "worker_audit", Path(__file__).resolve().parents[1] / "bench/summarize_validation_workers.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def write(path, value):
    path.write_text(json.dumps(value) + "\n")


def fixture(root):
    source = root / "validation_workers.py"
    source.write_text("# Synthetic audit fixture only\n")
    cache, extension, identity = (root / name for name in ("cache", "extension", "identity.json"))
    cache.write_bytes(b"synthetic cache, not a native cache")
    extension.write_bytes(b"synthetic binary")
    write(identity, {"extension_sha256": AUDIT.sha(extension)})
    runs = root / "runs"
    runs.mkdir()
    args = SimpleNamespace(
        series=root / "series.json",
        runs=runs,
        source=source,
        identity=identity,
        extension=extension,
        cache=cache,
        expected_cache_sha256=AUDIT.sha(cache),
    )
    plan = [dict(primer=True, block=-1, position=i, workers=w) for i, w in enumerate((4, 7, 16))]
    for block, order in enumerate(itertools.permutations((4, 7, 16))):
        plan.extend(dict(primer=False, block=block, position=i, workers=w) for i, w in enumerate(order))
    selected = {
        kind: dict(count=500000, paths_sha256="1" * 64, table_sha256="2" * 64, max_bytes=1024)
        for kind in ("images", "labels")
    }
    binding = dict(
        script_sha256=AUDIT.sha(source),
        extension_sha256=AUDIT.sha(extension),
        cache_sha256=AUDIT.sha(cache),
        identity_sha256=AUDIT.sha(identity),
        selected=selected,
    )
    parent = dict(
        complete=True,
        passed=True,
        planned_trials=21,
        measured_trials=18,
        pairs=500000,
        plan=plan,
        records=[],
        script_sha256=AUDIT.sha(source),
        binding=binding,
        started_at_ns=1,
        finished_at_ns=1000 * 10**9,
    )
    for index, spec in enumerate(plan):
        raw_path = runs / f"{index:02d}.json"
        raw_path.with_suffix(".log").write_text("synthetic successful fixture\n")
        base = (index * 20 + 1) * 10**9
        scale = {4: 1, 7: 2, 16: 4}[spec["workers"]]
        stages = [
            dict(
                kind=kind,
                complete=True,
                verified_paths=500000,
                elapsed_s=seconds / scale,
                user_s=0.1,
                system_s=0.1,
                started_at_ns=base + offset * 10**9,
                finished_at_ns=base + (offset + 5) * 10**9,
            )
            for kind, seconds, offset in (("images", 4, 1), ("labels", 2, 7))
        ]
        peak = 400000000 // scale
        raw = dict(
            complete=True,
            passed=True,
            pid=1000 + index,
            content=True,
            count=500000,
            workers=spec["workers"],
            started_at_ns=base,
            finished_at_ns=base + 14 * 10**9,
            records=stages,
            validation_elapsed_s=6 / scale,
            sampled_peak_rss=peak,
            samples=[
                dict(at_ns=base, phase="ready", rss_bytes=peak),
                dict(at_ns=base + 13 * 10**9, phase="finished", rss_bytes=peak),
            ],
            **binding,
        )
        write(raw_path, raw)
        command = [
            "/synthetic/python",
            str(source),
            "--cache",
            str(cache),
            "--identity",
            str(identity),
            "--expected-cache-sha256",
            AUDIT.sha(cache),
            "--out",
            str(raw_path),
            "--count",
            "500000",
            "--trial-workers",
            str(spec["workers"]),
        ]
        parent["records"].append(
            dict(
                spec=spec,
                validated=True,
                returncode=0,
                pid=1000 + index,
                command=command,
                path=str(raw_path),
                started_at_ns=base - 1,
                finished_at_ns=base + 15 * 10**9,
                raw_sha256=AUDIT.sha(raw_path),
                log_sha256=AUDIT.sha(raw_path.with_suffix(".log")),
                validation_elapsed_s=6 / scale,
                sampled_peak_rss=peak,
            )
        )
    write(args.series, parent)
    return args


def test_complete_synthetic_control(tmp_path):
    result = AUDIT.audit(fixture(tmp_path))
    assert result["passed"] and result["measured_trials"] == 18
    assert result["medians"]["4"]["total_s"] == 6
    for worker, expected in ((7, 2), (16, 4)):
        summary = result["four_worker_relative_ratios"][str(worker)]["total_s"]
        assert summary["median"] == expected
        assert summary["bootstrap_percentile_95"] == [expected, expected]


@pytest.mark.parametrize(
    "damage",
    [
        "partial",
        "missing_record",
        "plan",
        "command",
        "child_exit",
        "raw_count",
        "raw_worker",
        "selected",
        "zero_time",
        "total",
        "rss",
        "sample_order",
        "binary",
        "log",
        "raw_passed",
    ],
)
def test_corrupted_evidence_rejected_even_with_updated_raw_hash(tmp_path, damage):
    args = fixture(tmp_path)
    parent = AUDIT.read(args.series)
    record = parent["records"][4]
    path = args.runs / "04.json"
    raw = AUDIT.read(path)
    if damage == "partial":
        parent["complete"] = False
    elif damage == "missing_record":
        parent["records"].pop()
    elif damage == "plan":
        parent["plan"][3]["workers"] = 16
    elif damage == "command":
        record["command"][-1] = "4"
    elif damage == "child_exit":
        record["returncode"] = -15
    elif damage == "raw_count":
        raw["records"][1]["verified_paths"] -= 1
    elif damage == "raw_worker":
        raw["workers"] = 4
    elif damage == "selected":
        raw["selected"]["images"]["paths_sha256"] = "9" * 64
    elif damage == "zero_time":
        raw["records"][0]["elapsed_s"] = 0
    elif damage == "total":
        raw["validation_elapsed_s"] = record["validation_elapsed_s"] = 999
    elif damage == "rss":
        raw["sampled_peak_rss"] = record["sampled_peak_rss"] = 1
    elif damage == "sample_order":
        raw["samples"][1]["at_ns"] = 0
    elif damage == "binary":
        args.extension.write_bytes(b"changed binary")
    elif damage == "log":
        path.with_suffix(".log").write_text("changed log")
    elif damage == "raw_passed":
        raw["passed"] = False
    write(path, raw)
    record["raw_sha256"] = AUDIT.sha(path)
    write(args.series, parent)
    with pytest.raises((AssertionError, ValueError, KeyError)):
        AUDIT.audit(args)


def test_resampling_matches_independent_multinomial_oracle():
    values = [0.5, 0.75, 1, 1.25, 1.5, 2]
    distribution = defaultdict(int)
    for indices in itertools.combinations_with_replacement(range(6), 6):
        count = math.factorial(6)
        for occurrences in Counter(indices).values():
            count //= math.factorial(occurrences)
        median = (values[indices[2]] + values[indices[3]]) / 2
        distribution[median] += count
    ordered = [value for value, count in sorted(distribution.items()) for _ in range(count)]
    assert len(ordered) == 46656
    expected = []
    for quantile in (0.025, 0.975):
        index = (len(ordered) - 1) * quantile
        left = math.floor(index)
        expected.append(ordered[left] * (1 - (index - left)) + ordered[left + 1] * (index - left))
    actual = AUDIT.paired_summary(values)
    assert actual["bootstrap_percentile_95"] == expected and actual["resamples"] == 46656


@pytest.mark.parametrize("payload", ['{"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}', '{"a": 1e999}'])
def test_invalid_json(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(payload)
    with pytest.raises((AssertionError, ValueError)):
        AUDIT.read(path)


def test_cli_failure_receipt_cannot_be_overwritten(tmp_path):
    args = fixture(tmp_path)
    parent = AUDIT.read(args.series)
    parent["complete"] = False
    write(args.series, parent)
    output = tmp_path / "audit.json"
    command = [sys.executable, str(Path(AUDIT.__file__)), "--out", str(output)]
    for name, value in vars(args).items():
        command.extend(["--" + name.replace("_", "-"), str(value)])
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode != 0 and AUDIT.read(output)["passed"] is False
    before = output.read_bytes()
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0 and output.read_bytes() == before
