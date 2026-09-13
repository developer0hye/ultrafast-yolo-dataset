"""Qualify the artifact auditor on disposable copies of a completed pilot.

Run only when the host's benchmark reservation has ended. This never reruns
dataset operations. The five-round control is explicitly synthetic: it exercises
the aggregation branch and must never be reported as measured performance.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace


def encoded(value):
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arguments", type=Path, required=True, help="JSON object of explicit auditor arguments, excluding out"
    )
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    if options.out.exists():
        raise ValueError("retain earlier qualification")
    auditor_path = Path(__file__).with_name("audit-snapshot-startup-v1.py")
    spec = importlib.util.spec_from_file_location("snapshot_artifact_auditor", auditor_path)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    original_args = json.loads(options.arguments.read_text())
    path_keys = {k for k in original_args if k not in ("rounds", "pairs", "workers", "task")}
    original_args = {k: Path(v) if k in path_keys else v for k, v in original_args.items()}
    original = auditor.read(original_args["report"])
    if original_args["rounds"] != 1 or not original["complete"]:
        raise ValueError("requires the completed one-round pilot")
    controls, failures = [], []

    with tempfile.TemporaryDirectory(prefix="snapshot-audit-qualification-") as folder:
        root = Path(folder)
        runs, caches = root / "runs", root / "caches"
        shutil.copytree(original_args["runs_dir"], runs)
        shutil.copytree(original_args["caches_dir"], caches)
        args = SimpleNamespace(**dict(original_args, report=root / "report.json", runs_dir=runs, caches_dir=caches))

        def bind(report):
            """Rebind edited raw/parent records so semantic checks must reject."""
            for row in report["records"] + report["primers"]:
                path = runs / Path(row["path"]).name
                path.write_text(encoded(row["report"]))
                row["sha256"] = auditor.sha(path)
            args.report.write_text(encoded(report))

        def control(report, label):
            bind(report)
            result = auditor.audit(args)
            if not result["complete"]:
                raise AssertionError("valid control rejected")
            controls.append({"name": label, "measured_workers_in_fixture": result["measured_workers"]})

        def reject(label, change, report=None, raw_rebind=True):
            value = copy.deepcopy(original if report is None else report)
            bind(value)
            change(value)
            if raw_rebind:
                bind(value)
            else:
                args.report.write_text(encoded(value))
            try:
                auditor.audit(args)
            except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
                failures.append({"case": label, "rejected": True, "reason": str(error)})
            else:
                raise AssertionError("damaged evidence accepted: " + label)

        control(copy.deepcopy(original), "unmodified actual one-round pilot")
        reject("incomplete", lambda d: d.update(complete=False))
        reject("error retained despite complete", lambda d: d.update(error="interrupted"))
        reject("wrong pair count", lambda d: d.update(pairs=d["pairs"] - 1))
        reject("missing measured worker", lambda d: d["records"].pop())
        reject("reordered workers", lambda d: d["records"].reverse())
        reject("missing primer", lambda d: d["primers"].pop())
        reject("raw hash mismatch", lambda d: d["records"][0].update(sha256="0" * 64), raw_rebind=False)
        reject("log hash mismatch", lambda d: d["records"][0].update(log_sha256="0" * 64))
        reject("raw parent mismatch", lambda d: d["records"][0]["report"].update(pid=1), raw_rebind=False)
        reject(
            "wrong source digest",
            lambda d: d["records"][0]["report"]["descriptor"]["library_sources"].update({"src/snapshot.rs": "0" * 64}),
        )
        reject("wrong compiled profile", lambda d: d["records"][0]["report"].update(compiled_source_profile="0" * 64))
        reject("wrong wrapper digest", lambda d: d.update(wrapper_sha256="0" * 64))
        reject("reused PID", lambda d: d["records"][1]["report"].update(pid=d["records"][0]["report"]["pid"]))
        reject(
            "overlapping intervals",
            lambda d: d["records"][1]["report"].update(wall_start_ns=d["records"][0]["report"]["wall_start_ns"]),
        )
        reject("missing stage", lambda d: d["records"][0]["report"]["result"]["stages_s"].pop("_validate_inputs"))
        reject("fallback", lambda d: d["records"][0]["report"]["result"].update(native_fallback_count=1))
        reject("wrong scan count", lambda d: d["records"][0]["report"]["result"]["scan_summary"].update(found=1))
        reject(
            "wrong first batch",
            lambda d: d["records"][0]["report"]["result"]["output_sha256"].update(first_batch="0" * 64),
        )
        reject("negative stage", lambda d: d["records"][0]["report"]["result"]["stages_s"].update(_encode=-1))
        reject(
            "stages exceed constructor", lambda d: d["records"][0]["report"]["result"]["stages_s"].update(_encode=1000)
        )
        reject("wrong cache mode", lambda d: d["records"][0]["report"]["result"].update(mode="hit"))
        reject("wrong worker count", lambda d: d["records"][0]["report"]["result"].update(workers=1))
        reject(
            "RSS high water decreases",
            lambda d: d["records"][0]["report"]["result"].update(peak_rss_constructor_bytes=1),
        )
        reject(
            "RSS before exceeds high water",
            lambda d: d["records"][0]["report"]["result"].update(rss_before_constructor_bytes=10**15),
        )
        reject(
            "negative available memory", lambda d: d["records"][0]["report"]["result"].update(available_ram_bytes=-1)
        )
        reject("negative CPU time", lambda d: d["records"][0]["report"]["result"].update(user_s=-1))
        reject(
            "wrong cache payload", lambda d: d["records"][0]["report"]["cache"]["sections"][1].update(sha256="0" * 64)
        )
        reject("wrong cache profile", lambda d: d["records"][0]["report"]["cache"].update(profile="0" * 64))
        reject(
            "primer has measured result",
            lambda d: d["primers"][0].update(report=copy.deepcopy(d["records"][0]["report"])),
        )
        reject("command switches backend version", lambda d: d["records"][0]["command"].__setitem__(4, "candidate"))
        reject("pilot claims summary", lambda d: d.update(summary={}))

        # Explicit synthetic records check the five-round branch, never speed.
        expanded = copy.deepcopy(original)
        expanded.update(rounds=5, qualification_synthetic=True, records=[])
        for mode in ("miss", "hit"):
            for repeat in range(5):
                for version in ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline"):
                    row = copy.deepcopy(
                        next(r for r in original["records"] if r["mode"] == mode and r["version"] == version)
                    )
                    row["round"] = repeat
                    factor = ((1, 1.25, 0.8, 1.5, 1.1) if version == "baseline" else (1, 1.1, 1.3, 0.9, 1.2))[repeat]
                    for metric in ("constructor_s", "first_batch_total_s"):
                        row["report"]["result"][metric] *= factor
                    row["report"]["result"]["stages_s"] = {
                        key: value * factor for key, value in row["report"]["result"]["stages_s"].items()
                    }
                    name = f"{mode}-{repeat}-{version}"
                    row["path"] = str(Path(row["path"]).with_name(name + ".json"))
                    row["command"][row["command"].index("--out") + 1] = row["path"]
                    source_log = runs / f"{mode}-0-{version}.log"
                    if repeat:
                        shutil.copyfile(source_log, runs / (name + ".log"))
                    expanded["records"].append(row)
        ordered = expanded["records"][:10] + expanded["primers"] + expanded["records"][10:]
        for index, row in enumerate(ordered):
            row["report"].update(
                pid=10000 + index, wall_start_ns=10**15 + index * 10**12, wall_end_ns=10**15 + index * 10**12 + 10**11
            )
        expanded["summary"] = {
            mode: {
                metric: auditor.summary(
                    [
                        [
                            next(
                                row["report"]["result"][metric]
                                for row in expanded["records"]
                                if row["mode"] == mode and row["round"] == r and row["version"] == v
                            )
                            for v in auditor.VERSIONS
                        ]
                        for r in range(5)
                    ]
                )
                for metric in auditor.METRICS
            }
            for mode in ("miss", "hit")
        }
        args.rounds = 5
        control(copy.deepcopy(expanded), "synthetic five-round aggregation control; not a measurement")
        reject("wrong median", lambda d: d["summary"]["miss"]["constructor_s"].update(candidate_median=1), expanded)
        reject(
            "wrong ratio", lambda d: d["summary"]["hit"]["constructor_s"].update(baseline_over_candidate=123), expanded
        )
        reject(
            "wrong CI", lambda d: d["summary"]["miss"]["constructor_s"].update(paired_bootstrap_ci95=[0, 100]), expanded
        )
        reject(
            "wrong summary raw pairs",
            lambda d: d["summary"]["hit"]["constructor_s"]["raw_pairs"][0].__setitem__(0, 1),
            expanded,
        )
        control(copy.deepcopy(expanded), "synthetic five-round post-mutation control; not a measurement")

    receipt = {
        "complete": True,
        "controls": controls,
        "negative_cases": failures,
        "negative_case_count": len(failures),
        "auditor_sha256": auditor.sha(auditor_path),
        "qualifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arguments_sha256": auditor.sha(options.arguments),
        "pilot_report_sha256": auditor.sha(original_args["report"]),
        "scope": "Disposable artifact corruption checks; no dataset execution. Expanded five-round fixture is synthetic and supplies no performance evidence.",
    }
    with options.out.open("x") as stream:
        stream.write(encoded(receipt))
    print(f"Passed {len(controls)} controls and rejected {len(failures)} damaged artifacts.")


if __name__ == "__main__":
    main()
