"""Adversarial reports for the independent large-fixture result auditor."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("startup_audit", Path(__file__).parents[1] / "bench/audit_startup.py")
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def write(path, value):
    path.write_text(json.dumps(value))


def fixture(tmp_path, phase="p3"):
    pairs, repeats, workers = 10, 5, 4
    labels = {
        "count": pairs,
        "task": "detect",
        "distinct_files": pairs,
        "synthetic": True,
        "sha256_names_and_contents": "a" * 64,
    }
    fingerprint = {"files": pairs * 2, "bytes": 200, "sha256": "b" * 64}
    corpus = {
        "kind": "ultrafast-yolo-startup-synthetic-v1",
        "count": pairs,
        "task": "detect",
        "labels": labels,
        "fingerprint": fingerprint,
    }
    identity = {"source_sha256": {"bench/cache_startup.py": "c" * 64}, "native_extension_sha256": "d" * 64}
    preflight = {
        "pairs": pairs,
        "counts": {"images": pairs, "labels": pairs},
        "fingerprint": fingerprint,
        "all_regular_single_link_files": True,
    }
    report = dict(identity, corpus=labels if phase == "p1" else corpus, results=[])
    backends = {
        "p1": ("reference", "native"),
        "p3": ("reference", "native-content"),
        "p4": ("reference-content", "native-content"),
    }[phase]
    raw = tmp_path / "report.runs"
    raw.mkdir()
    if phase == "p4":
        for backend in backends:
            write(raw / f"prime-{backend}.json", {"primed": backend})
    for repeat in range(repeats):
        for backend in backends if repeat % 2 == 0 else backends[::-1]:
            scale = 2 if backend == backends[0] else 1
            time = (repeat + 1) * scale
            row = {
                "round": repeat,
                "backend": backend,
                "workers": workers,
                "parity": True,
                "user_s": 0.1,
                "system_s": 0.1,
                "available_ram_bytes": 100000,
                "loadavg": [1, 2, 3],
            }
            if phase == "p1":
                row.update(
                    wall_s=time,
                    labels_per_s=pairs / time,
                    reference_required=0,
                    output_sha256="e" * 64,
                    peak_rss_bytes=scale * 100,
                    rss_method="/proc/self/status VmHWM",
                )
            else:
                row.update(
                    mode="hit" if phase == "p4" else "miss",
                    loader_workers=0,
                    constructor_s=time,
                    first_batch_total_s=time * 1.5,
                    stages_s={},
                    scan_summary={"found": pairs, "missing": 0, "empty": 0, "corrupt": 0, "total": pairs},
                    diagnostic_count=0,
                    native_fallback_count=0 if backend == "native-content" else None,
                    output_sha256={key: "e" * 64 for key in ("labels", "first_batch", "scan_summary_and_messages")},
                    rss_before_constructor_bytes=45,
                    peak_rss_before_constructor_bytes=50,
                    peak_rss_constructor_bytes=scale * 100,
                    peak_rss_first_batch_bytes=scale * 110,
                    cache_bytes=100,
                )
                if backend == "native-content":
                    row["stages_s"] = {
                        "_validate_inputs": 0.01,
                        "to_ultralytics_labels": 0.01,
                        "_decode" if phase == "p4" else "_encode": 0.01,
                    }
                elif phase == "p4":
                    row["stages_s"] = {"input_fingerprints": 0.01, "reference_cache_decode": 0.01}
                write(raw / f"{repeat}-{backend}.json", {k: v for k, v in row.items() if k not in ("round", "parity")})
            report["results"].append(row)
    if phase != "p1":
        report["summary"] = {}
        for metric in ("constructor_s", "first_batch_total_s", "peak_rss_constructor_bytes"):
            report["summary"][metric] = {}
            for backend in backends:
                values = [r[metric] for r in report["results"] if r["backend"] == backend]
                report["summary"][metric][backend] = {
                    "median": auditor.quantile(values, 0.5),
                    "p95": auditor.quantile(values, 0.95),
                }
    paths = [tmp_path / name for name in ("report.json", "identity.json", "preflight.json", "fixture.json")]
    for path, value in zip(paths, (report, identity, preflight, corpus)):
        write(path, value)
    return paths, report, (phase, "detect", pairs, repeats, workers)


@pytest.mark.parametrize("phase", ["p1", "p3", "p4"])
def test_complete_known_ratio_and_exact_bootstrap(tmp_path, phase):
    paths, _, config = fixture(tmp_path, phase)
    result = auditor.audit(*paths, *config)
    assert result["complete_requested_samples"] and result["completed_workers"] == 10
    for summary in result["summary"].values():
        assert summary["reference_over_candidate"] == 2
        assert summary["paired_bootstrap_ci95"] == [2, 2]
        assert summary["resamples"] == 3125


def test_partial_never_gets_summary(tmp_path):
    paths, report, config = fixture(tmp_path)
    report["results"] = report["results"][:3]
    write(paths[0], report)
    with pytest.raises(ValueError, match="incomplete requested"):
        auditor.audit(*paths, *config)
    result = auditor.audit(*paths, *config, partial=True)
    assert not result["complete_requested_samples"] and "summary" not in result


def test_complete_rows_before_parent_summary_are_interim_only(tmp_path):
    paths, report, config = fixture(tmp_path)
    del report["summary"]
    write(paths[0], report)
    with pytest.raises(ValueError, match="aggregation"):
        auditor.audit(*paths, *config)
    assert "summary" not in auditor.audit(*paths, *config, partial=True)


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "reordered", "wrong_worker", "wrong_scan", "changed_output", "nonfinite_time", "rss_backwards"],
)
def test_bad_rows_rejected_even_if_raw_matches_parent(tmp_path, mutation):
    paths, report, config = fixture(tmp_path)
    row = report["results"][1]
    if mutation == "duplicate":
        report["results"][1] = report["results"][0]
    elif mutation == "reordered":
        report["results"][:2] = report["results"][:2][::-1]
    elif mutation == "wrong_worker":
        row["workers"] = 8
    elif mutation == "wrong_scan":
        row["scan_summary"]["total"] = 9
    elif mutation == "changed_output":
        row["output_sha256"]["labels"] = "f" * 64
    elif mutation == "nonfinite_time":
        row["constructor_s"] = float("nan")
    else:
        row["peak_rss_constructor_bytes"] = 49
    for row in report["results"]:
        write(
            paths[0].with_suffix(".runs") / f"{row['round']}-{row['backend']}.json",
            {k: v for k, v in row.items() if k not in ("round", "parity")},
        )
    write(paths[0], report)
    with pytest.raises(ValueError):
        auditor.audit(*paths, *config)


@pytest.mark.parametrize("mutation", ["raw", "source", "native", "preflight", "fixture", "median", "p95", "primer"])
def test_artifact_mismatch_rejected(tmp_path, mutation):
    paths, report, config = fixture(tmp_path, "p4")
    if mutation == "raw":
        raw = paths[0].with_suffix(".runs") / "0-reference-content.json"
        value = json.loads(raw.read_text())
        value["constructor_s"] += 1
        write(raw, value)
    elif mutation in ("source", "native"):
        report["source_sha256"]["bench/cache_startup.py"] = "f" * 64 if mutation == "source" else "c" * 64
        if mutation == "native":
            report["native_extension_sha256"] = "f" * 64
    elif mutation in ("preflight", "fixture"):
        path = paths[2 if mutation == "preflight" else 3]
        value = json.loads(path.read_text())
        value["fingerprint"]["sha256"] = "f" * 64
        write(path, value)
    elif mutation == "primer":
        write(paths[0].with_suffix(".runs") / "prime-native-content.json", {"primed": "reference-content"})
    else:
        report["summary"]["constructor_s"]["reference-content"][mutation] = 123
    write(paths[0], report)
    with pytest.raises(ValueError):
        auditor.audit(*paths, *config)


@pytest.mark.parametrize("mutation", ["label_manifest", "throughput"])
def test_p1_manifest_and_consumed_count(tmp_path, mutation):
    paths, report, config = fixture(tmp_path, "p1")
    if mutation == "label_manifest":
        report["corpus"]["sha256_names_and_contents"] = "f" * 64
    else:
        report["results"][0]["labels_per_s"] = 1
    write(paths[0], report)
    with pytest.raises(ValueError):
        auditor.audit(*paths, *config)


def test_quantile_interpolation():
    assert auditor.quantile([0, 10, 20, 30, 40], 0.95) == 38


def test_cache_hit_must_observe_decode_not_rebuild(tmp_path):
    paths, report, config = fixture(tmp_path, "p4")
    row = report["results"][1]
    row["stages_s"]["_encode"] = row["stages_s"].pop("_decode")
    write(paths[0], report)
    write(
        paths[0].with_suffix(".runs") / "0-native-content.json",
        {k: v for k, v in row.items() if k not in ("round", "parity")},
    )
    with pytest.raises(ValueError, match="cache path"):
        auditor.audit(*paths, *config)
