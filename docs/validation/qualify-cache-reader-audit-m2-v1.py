"""Reject damaged real measurement artifacts, including consistent parent/raw edits."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path("/Volumes/T7/ultrafast-vision-build")
REPO = Path(__file__).parents[2]
AUDITOR = Path(__file__).with_name("audit-cache-reader-m2-v1.py")
spec = importlib.util.spec_from_file_location("audit_reader", AUDITOR)
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)
original = json.loads((ROOT / "cache-reader-paired-m2-v1.json").read_text())
raw = {p.name: p.read_bytes() for p in (ROOT / "cache-reader-paired-m2-v1.runs").glob("*.json")}
manifest = (ROOT / "cache-reader-seven-layouts-m2-v1/manifest.json").read_bytes()
identities = {
    "baseline": (REPO / "docs/validation/cache-reader-baseline-m2-identity-v1.json").read_bytes(),
    "candidate": (REPO / "docs/validation/cache-copy-m2-build-identity-v1.json").read_bytes(),
}
launch = json.loads((ROOT / "cache-reader-paired-m2-v1-launch.json").read_text())


def encoded(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def mutate_raw(report, files, key, value, descriptor=False):
    row = report["records"][0]
    name = Path(row["path"]).name
    worker = json.loads(files[name])
    if descriptor:
        worker["descriptor"][key] = value
    else:
        worker[key] = value
    files[name] = encoded(worker)
    row["sha256"] = hashlib.sha256(files[name]).hexdigest()
    row["report"] = {k: v for k, v in worker.items() if k != "descriptor"}


control = auditor.audit(encoded(original), raw, manifest, identities, launch)
assert control["complete"] and control["measured_workers"] == 70
damages = [
    "incomplete",
    "missing_worker",
    "order",
    "raw_bytes",
    "parent_value",
    "worker_median",
    "negative_time",
    "output_unverified",
    "environment_drift",
    "wrong_fixture",
    "wrong_command",
    "duplicate_command_flag",
    "extra_raw",
    "aggregate",
    "interval",
    "wrong_binary",
    "negative_rss",
]
results = []
for damage in damages:
    report, files = copy.deepcopy(original), dict(raw)
    if damage == "incomplete":
        report["complete"] = False
    elif damage == "missing_worker":
        report["records"].pop()
    elif damage == "order":
        report["records"][0], report["records"][1] = report["records"][1], report["records"][0]
    elif damage == "raw_bytes":
        name = Path(report["records"][0]["path"]).name
        files[name] += b" "
    elif damage == "parent_value":
        report["records"][0]["report"]["median_ns"] += 1
    elif damage == "worker_median":
        mutate_raw(report, files, "median_ns", 1)
    elif damage == "negative_time":
        mutate_raw(report, files, "samples_ns", [1, 1, -1, 1, 1])
    elif damage == "output_unverified":
        mutate_raw(report, files, "output_verified", False)
    elif damage == "environment_drift":
        mutate_raw(report, files, "cpu", "other CPU", descriptor=True)
    elif damage == "wrong_fixture":
        mutate_raw(report, files, "fixture", original["fixture_manifest"]["cases"][1])
    elif damage == "wrong_command":
        report["records"][0]["command"][0] = "other-python"
    elif damage == "duplicate_command_flag":
        report["records"][0]["command"] += ["--worker", "measure"]
    elif damage == "extra_raw":
        files["extra.json"] = b"{}"
    elif damage == "aggregate":
        report["summary"][0]["median_ns"]["baseline_median"] += 1
    elif damage == "interval":
        report["summary"][0]["median_ns"]["paired_bootstrap_ci95"][0] += 0.01
    elif damage == "wrong_binary":
        row = report["descriptors"]["candidate"]
        worker = copy.deepcopy(row["report"])
        worker["descriptor"]["extension_sha256"] = "0" * 64
        name = Path(row["path"]).name
        files[name] = encoded(worker)
        row["sha256"], row["report"] = hashlib.sha256(files[name]).hexdigest(), worker
    elif damage == "negative_rss":
        mutate_raw(report, files, "peak_rss_bytes", -1)
    try:
        auditor.audit(encoded(report), files, manifest, identities, launch)
    except ValueError as error:
        results.append({"damage": damage, "rejected": True, "reason": str(error)})
    else:
        raise AssertionError("accepted damaged evidence: " + damage)
out = ROOT / "cache-reader-paired-m2-v1-audit-negative.json"
assert not out.exists()
out.write_text(
    json.dumps(
        {
            "valid_control_passed": True,
            "rejected": len(results),
            "cases": results,
            "auditor_sha256": hashlib.sha256(AUDITOR.read_bytes()).hexdigest(),
            "test_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "17 mutations of the actual completed report; no native workloads repeated",
        },
        indent=2,
    )
    + "\n"
)
print("Valid control passed; all 17 damaged artifact cases rejected.")
