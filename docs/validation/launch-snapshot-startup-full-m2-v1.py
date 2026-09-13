"""Start the full comparison only after the corrected actual-startup pilot passes."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path("/Volumes/T7/ultrafast-vision-build")
WRAPPER = ROOT / "snapshot-startup-wrapper-m2-v2.py"
REPORT = ROOT / "snapshot-startup-full-m2-v1.json"
LAUNCH = ROOT / "snapshot-startup-full-m2-v1-launch.json"
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", YOLO_OFFLINE="true")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert not REPORT.exists() and not LAUNCH.exists()
assert sha(WRAPPER) == "c8074cabf59db8727b0576de9f7e9c3fe0b4e9a059e180219aaffb423397ee14"
pilot_path = ROOT / "snapshot-startup-pilot-m2-v2.json"
pilot = json.loads(pilot_path.read_text())
audit_path = ROOT / "snapshot-startup-pilot-m2-v2-audit.json"
audit = json.loads(audit_path.read_text())
assert audit["complete"] and audit["report_sha256"] == sha(pilot_path)
assert pilot["complete"] and pilot["rounds"] == 1 and pilot["pairs"] == 1003
assert len(pilot["records"]) == 4 and len(pilot["primers"]) == 2
for row in pilot["records"] + pilot["primers"]:
    path = Path(row["path"])
    assert sha(path) == row["sha256"] == audit["raw_sha256"][path.name]
    assert json.loads(path.read_text()) == row["report"]
corpus = ROOT / "startup-detect-500k-v1"
manifest = json.loads((corpus / "startup.json").read_text())
assert manifest["count"] == 500000 and manifest["task"] == "detect"
assert manifest["fingerprint"] == {
    "files": 1000000,
    "bytes": 3625093750,
    "sha256": "931bd8059aefe560c3601e49d9b5f08ddc87831440bb0a1b4118575edca83f5d",
}
build = json.loads((ROOT / "dataset-snapshot-scratch-m2-v1-build-receipt.json").read_text())
assert build["complete"] and build["tests"] == 193 and build["rust_tests"] == 5 and build["skipped"] == 0
command = [
    ROOT / "dataset-snapshot-scratch-m2-v1-clean/bin/python",
    WRAPPER,
    "--corpus",
    corpus,
    "--pairs",
    "500000",
    "--rounds",
    "5",
    "--harness-root",
    ROOT / "dataset-snapshot-scratch-m2-v1-source",
    "--out",
    REPORT,
]
for version in ("baseline", "candidate"):
    row = next(r for r in pilot["records"] if r["version"] == version)
    original = row["command"]
    command += [f"--{version}-python", original[0]]
    for field in ("source", "wheel", "identity"):
        command += [f"--{version}-{field}", original[original.index("--" + field) + 1]]
command = list(map(str, command))
launch = {
    "command": command,
    "wrapper_sha256": sha(WRAPPER),
    "controller_sha256": sha(Path(__file__)),
    "pilot_sha256": sha(pilot_path),
    "pilot_audit_sha256": sha(audit_path),
    "build_receipt_sha256": sha(ROOT / "dataset-snapshot-scratch-m2-v1-build-receipt.json"),
    "fixture_manifest_sha256": sha(corpus / "startup.json"),
    "environment_overrides": {
        k: ENV[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "YOLO_OFFLINE")
    },
    "expected_workers": 20,
    "expected_separate_primers": 2,
    "complete": False,
}
LAUNCH.write_text(json.dumps(launch, indent=2) + "\n")
with (ROOT / "snapshot-startup-full-m2-v1.log").open("x") as log:
    result = subprocess.run(command, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=False)
launch["returncode"] = result.returncode
LAUNCH.write_text(json.dumps(launch, indent=2) + "\n")
assert result.returncode == 0, "comparison failed; preserve raw output and do not restart"
report = json.loads(REPORT.read_text())
assert report["complete"] and len(report["records"]) == 20 and len(report["primers"]) == 2
reference = json.loads((ROOT / "dataset-500k-p3-complete-audit-v1.json").read_text())
assert report["output_sha256"] == reference["output_sha256"]
launch["complete"] = True
launch["report_sha256"] = sha(REPORT)
launch["outputs_match_original_full_reference"] = True
LAUNCH.write_text(json.dumps(launch, indent=2) + "\n")
print("Full P3/P4 comparison complete; independent raw/statistical audit still required.")
