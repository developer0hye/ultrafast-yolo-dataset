"""Qualify the two installed binaries, then run the frozen seven-condition matrix."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path("/Volumes/T7/ultrafast-vision-build")
REPO = Path("/Users/yhkwon/Documents/ultrafast-yolo-dataset-cache-read-copy")
SOURCE = ROOT / "dataset-cache-read-copy-m2-v1-source"
SCRIPT = SOURCE / "bench/cache_read_comparison.py"
FIXTURE = ROOT / "cache-reader-seven-layouts-m2-v1"
OUT = ROOT / "cache-reader-paired-m2-v1.json"
QUALIFY = ROOT / "cache-reader-qualification-m2-v1"
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", YOLO_OFFLINE="true")
VERSIONS = {
    "baseline": {
        "python": ROOT / "mask-resize-roi-clean-v2/bin/python",
        "root": Path("/Users/yhkwon/Documents/ultrafast-yolo-dataset"),
        "wheel": ROOT / "dataset-notices-dist-v1/ultrafast_yolo_dataset-0.1.0a1-cp312-cp312-macosx_11_0_arm64.whl",
        "identity": REPO / "docs/validation/cache-reader-baseline-m2-identity-v1.json",
    },
    "candidate": {
        "python": ROOT / "dataset-cache-read-copy-m2-v1-clean/bin/python",
        "root": SOURCE,
        "wheel": ROOT
        / "dataset-cache-read-copy-m2-v1-dist/ultrafast_yolo_dataset-0.1.0a1-cp312-cp312-macosx_11_0_arm64.whl",
        "identity": ROOT / "dataset-cache-read-copy-m2-v1-build-identity.json",
    },
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


receipt = json.loads((ROOT / "dataset-cache-read-copy-m2-v1-resume-receipt.json").read_text())
assert receipt["complete"] and receipt["tests"] == 187 and receipt["skipped"] == 0
assert sha(SCRIPT) == "f79fd09baf58e551a241fa00db2fd8b40c07d736f880fe92ed90c25f5e76c0b3"
manifest = json.loads((FIXTURE / "manifest.json").read_text())
assert len(manifest["cases"]) == 7 and manifest["cases"][6]["bytes"] == 195001501
assert not OUT.exists()
QUALIFY.mkdir(exist_ok=False)
state = {
    "complete": False,
    "scope": "Untimed campaign qualification; do not pool these samples into paired results",
    "commands": [],
    "script_sha256": sha(Path(__file__)),
    "harness_sha256": sha(SCRIPT),
    "fixture_manifest_sha256": sha(FIXTURE / "manifest.json"),
}


def save():
    (QUALIFY / "receipt.json").write_text(json.dumps(state, indent=2) + "\n")


def run(command, name):
    command = list(map(str, command))
    log = QUALIFY / (name + ".log")
    with log.open("x") as stream:
        result = subprocess.run(command, env=ENV, stdout=stream, stderr=subprocess.STDOUT, check=False)
    state["commands"].append({"command": command, "returncode": result.returncode, "log_sha256": sha(log)})
    save()
    assert result.returncode == 0, log


save()
try:
    descriptors = {}
    for version, args in VERSIONS.items():
        base = [args["python"], SCRIPT, "--fixture", FIXTURE]
        for key in ("root", "wheel", "identity"):
            base += ["--" + key, args[key]]
        out = QUALIFY / (version + "-describe.json")
        run([*base, "--worker", "describe", "--out", out], version + "-describe")
        descriptors[version] = json.loads(out.read_text())["descriptor"]
        for case in (0, 6):
            name = f"{version}-case{case}"
            out = QUALIFY / (name + ".json")
            run([*base, "--worker", "measure", "--case", case, "--out", out], name)
            result = json.loads(out.read_text())
            assert result["descriptor"] == descriptors[version]
            assert result["fixture"] == manifest["cases"][case] and result["output_verified"] is True
            assert len(result["samples_ns"]) == 5 and all(n > 0 for n in result["samples_ns"])
    a, b = descriptors.values()
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
        assert a[key] == b[key], key
    assert {n for n in a["library_sources"] if a["library_sources"][n] != b["library_sources"][n]} == {
        "src/cache.rs",
        "python/ultrafast_yolo_dataset/_cache.py",
    }
    state["complete"] = True
    state["raw_sha256"] = {p.name: sha(p) for p in QUALIFY.glob("*.json") if p.name != "receipt.json"}
    save()
except BaseException as error:
    state["error"] = f"{type(error).__name__}: {error}"
    save()
    raise

command = [VERSIONS["candidate"]["python"], SCRIPT, "--fixture", FIXTURE, "--out", OUT]
for version, args in VERSIONS.items():
    for key, value in args.items():
        command += [f"--{version}-{key}", value]
launch = {
    "command": list(map(str, command)),
    "environment_overrides": {
        k: ENV[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "YOLO_OFFLINE")
    },
    "qualification_receipt_sha256": sha(QUALIFY / "receipt.json"),
    "script_sha256": sha(Path(__file__)),
}
(ROOT / "cache-reader-paired-m2-v1-launch.json").write_text(json.dumps(launch, indent=2) + "\n")
with (ROOT / "cache-reader-paired-m2-v1.log").open("x") as stream:
    completed = subprocess.run(list(map(str, command)), env=ENV, stdout=stream, stderr=subprocess.STDOUT, check=False)
assert completed.returncode == 0, "paired matrix failed; retain all artifacts"
result = json.loads(OUT.read_text())
assert result["complete"] and len(result["records"]) == 70 and len(result["summary"]) == 7
print("Qualification and all 70 measured processes complete; independent final audit remains required.")
