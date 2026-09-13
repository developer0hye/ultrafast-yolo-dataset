"""Wait for the current M2 experiment, audit it, then build/test one frozen candidate.

No build, fixture copy, dependency installation or test runs before the exact
measurement parent exits and all ten P4 workers pass the independent audit.
"""

import hashlib
import importlib.util
import json
import os
import subprocess
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

ROOT = Path("/Volumes/T7/ultrafast-vision-build")
MAIN = Path("/Users/yhkwon/Documents/ultrafast-yolo-dataset")
CANDIDATE = Path("/Users/yhkwon/Documents/ultrafast-yolo-dataset-cache-read-copy")
REF = "f120cab2f907e36fdcfd93d0879fe56ec4324017"
BASEPY = ROOT / "venv/bin/python"
RUNTIME = ROOT / "mask-resize-roi-clean-v2/bin/python"
UV = Path("/Users/yhkwon/.local/bin/uv")
AUDITOR = Path("/Users/yhkwon/Documents/ultrafast-yolo-dataset-startup-audit/bench/audit_startup.py")
PID = 27081
PREFIX = "dataset-cache-read-copy-m2-v1"
STATE = ROOT / (PREFIX + "-pipeline.json")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ps(field):
    result = subprocess.run(["ps", "-p", str(PID), "-o", field + "="], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


stamp = ps("lstart")
require(stamp == "Sun Sep 13 04:35:46 2026", "wrong measurement parent start time")
require(ps("command") == "zsh /Volumes/T7/ultrafast-vision-build/dataset-detect-500k-v1.sh", "wrong watched command")
require(not STATE.exists(), "retain earlier pipeline state")
require(sha(AUDITOR) == "d745e43cc6c63dc77335d5756c315447f8a22a86f70b483b7e812456c497b80d", "startup auditor drift")
state = {
    "complete": False,
    "phase": "waiting-for-P4",
    "watched_pid": PID,
    "watched_start": stamp,
    "source_commit": REF,
    "script_sha256": sha(Path(__file__)),
    "commands": [],
}


def save():
    STATE.write_text(json.dumps(state, indent=2) + "\n")


ENV = dict(
    os.environ,
    RUSTUP_TOOLCHAIN="stable",
    CARGO_BUILD_JOBS="4",
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    YOLO_OFFLINE="true",
)
for key in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET"):
    ENV.pop(key, None)


def run(command, name, cwd=None):
    command = list(map(str, command))
    log = ROOT / (PREFIX + "-" + name + ".log")
    require(not log.exists(), "retain command logs")
    print(name, flush=True)
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=cwd, env=ENV, stdout=stream, stderr=subprocess.STDOUT, check=False)
    state["commands"].append(
        {
            "command": command,
            "cwd": str(cwd) if cwd else None,
            "log": str(log),
            "log_sha256": sha(log),
            "returncode": result.returncode,
        }
    )
    save()
    require(result.returncode == 0, f"{name} failed; see {log}")
    return log


save()
print(f"Waiting for exact P4 parent {PID}", flush=True)
try:
    while ps("lstart") == stamp:
        time.sleep(10)
    state["phase"] = "audit-complete-P4"
    save()
    report = ROOT / "startup-detect-500k-p4-v1.json"
    data = report.read_bytes()
    parsed = json.loads(data)
    require(len(parsed["results"]) == 10 and "summary" in parsed, "P4 incomplete; build not started")
    identity = MAIN / "docs/validation/detect-500k-launch-v1.json"
    frozen = json.loads(identity.read_text())
    require(all(sha(MAIN / name) == value for name, value in frozen["source_sha256"].items()), "P4 sources drifted")
    require(
        sha(AUDITOR) == "d745e43cc6c63dc77335d5756c315447f8a22a86f70b483b7e812456c497b80d",
        "auditor changed while waiting",
    )
    sealed = ROOT / "dataset-500k-p4-complete-v1.json"
    with sealed.open("xb") as stream:
        stream.write(data)
    audited = ROOT / "dataset-500k-p4-complete-audit-v1.json"
    run(
        [
            BASEPY,
            AUDITOR,
            "--report",
            sealed,
            "--identity",
            identity,
            "--preflight",
            ROOT / "startup-detect-500k-preflight-v1.json",
            "--fixture-manifest",
            ROOT / "startup-detect-500k-v1/startup.json",
            "--runs-dir",
            report.with_suffix(".runs"),
            "--phase",
            "p4",
            "--task",
            "detect",
            "--pairs",
            "500000",
            "--repeats",
            "5",
            "--workers",
            "7",
            "--out",
            audited,
        ],
        "p4-audit",
    )
    receipt = json.loads(audited.read_text())
    require(receipt["complete_requested_samples"] is True and receipt["completed_workers"] == 10, "P4 audit incomplete")
    p3 = json.loads((ROOT / "dataset-500k-p3-complete-audit-v1.json").read_text())
    require(receipt["output_sha256"] == p3["output_sha256"], "P3/P4 complete outputs disagree")
    state["p4_report_sha256"], state["p4_audit_sha256"] = sha(sealed), sha(audited)
    state["phase"] = "build-frozen-candidate"
    save()
    rustc = subprocess.check_output(["/Users/yhkwon/.cargo/bin/rustc", "+stable", "-Vv"], text=True)
    cargo = subprocess.check_output(["/Users/yhkwon/.cargo/bin/cargo", "+stable", "-V"], text=True)
    require(
        "release: 1.98.0\n" in rustc and "host: aarch64-apple-darwin\n" in rustc and "LLVM version: 22.1.8" in rustc,
        "wrong pinned Rust toolchain",
    )
    require(cargo.startswith("cargo 1.98.0 "), "wrong Cargo toolchain")
    target, source, dist, clean = [ROOT / (PREFIX + "-" + name) for name in ("target", "source", "dist", "clean")]
    require(not any(p.exists() for p in (target, source, dist, clean)), "use new isolated build paths")
    archive = ROOT / (PREFIX + "-source.tar")
    require(not archive.exists(), "retain source archive")
    run(["git", "archive", "--format=tar", "--output", archive, REF], "archive", cwd=CANDIDATE)
    source.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(source, filter="data")
    state["source_archive_sha256"] = sha(archive)
    ENV["CARGO_TARGET_DIR"] = str(target)
    ENV["MACOSX_DEPLOYMENT_TARGET"] = "11.0"
    run(
        [
            BASEPY,
            "-m",
            "maturin",
            "build",
            "--release",
            "--locked",
            "--offline",
            "--interpreter",
            RUNTIME,
            "--out",
            dist,
        ],
        "build",
        cwd=source,
    )
    run([BASEPY, "-m", "maturin", "sdist", "--out", dist], "sdist", cwd=source)
    (wheel,) = dist.glob("*.whl")
    run([UV, "venv", "--offline", "--seed", "--python", RUNTIME, clean], "venv")
    python = clean / "bin/python"
    run(
        [UV, "pip", "install", "--offline", "--no-deps", "--python", python, "numpy==2.4.4", "pillow==12.1.1", wheel],
        "standalone-install",
    )
    (source / "dist").symlink_to(dist, target_is_directory=True)
    run([python, source / "bench/ci_wheel_check.py"], "wheel-check", cwd=source)
    run(
        [python, source / "bench/wheel_smoke.py", "--out", ROOT / (PREFIX + "-standalone.json")],
        "standalone-smoke",
        cwd=source,
    )
    state["phase"] = "full-installed-tests"
    save()
    freeze = run([UV, "--quiet", "pip", "freeze", "--python", RUNTIME], "baseline-freeze")
    requirements = ROOT / (PREFIX + "-requirements.txt")
    rows = freeze.read_text().splitlines()
    require(any(row.startswith("ultrafast-yolo-dataset") for row in rows), "baseline package absent from freeze")
    requirements.write_text("\n".join(row for row in rows if not row.startswith("ultrafast-yolo-dataset")) + "\n")
    run([UV, "pip", "install", "--offline", "--no-deps", "--python", python, "-r", requirements], "runtime-install")
    run([UV, "pip", "check", "--python", python], "pip-check")
    run([UV, "--quiet", "pip", "freeze", "--python", python], "candidate-freeze")
    junit = ROOT / (PREFIX + "-tests.xml")
    run([python, "-m", "pytest", "-q", "tests", "--junitxml", junit], "tests", cwd=source)
    suites = ET.parse(junit).getroot().findall("testsuite")
    require(sum(int(s.attrib["tests"]) for s in suites) >= 178, "unexpectedly small suite")
    cases = [case for suite in suites for case in suite.findall("testcase")]
    for module_name, expected_count in (("test_cache_section_buffers", 12), ("test_cache_read_comparison", 9)):
        selected = [case for case in cases if case.attrib["classname"].endswith(module_name)]
        require(
            len(selected) == expected_count and all(case.find("skipped") is None for case in selected),
            "new targeted cases missing or skipped",
        )

    require(all(int(s.attrib[k]) == 0 for s in suites for k in ("failures", "errors")), "failing installed suite")
    require(
        subprocess.check_output(["/Users/yhkwon/.cargo/bin/rustc", "+stable", "-Vv"], text=True) == rustc,
        "compiler changed during build",
    )
    with ZipFile(wheel) as zipped:
        (native,) = [
            n for n in zipped.namelist() if n.startswith("ultrafast_yolo_dataset/_native.") and n.endswith(".so")
        ]
        binary = zipped.read(native)
    require(
        sha(target / "release/lib_native.dylib") == hashlib.sha256(binary).hexdigest(),
        "wheel/native build output mismatch",
    )
    (compiled,) = (target / "release/.fingerprint").glob("ultrafast-yolo-dataset-*/lib-_native.json")
    flags = json.loads(compiled.read_text())["rustflags"]
    spec = importlib.util.spec_from_file_location("cache_read_comparison", source / "bench/cache_read_comparison.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build_identity = {
        "wheel_sha256": sha(wheel),
        "extension_sha256": hashlib.sha256(binary).hexdigest(),
        "library_sources": module.library_sources(source),
        "toolchain": {
            "rustc": rustc,
            "cargo": cargo,
            "target": "aarch64-apple-darwin",
            "profile": "release",
            "rustflags": " ".join(flags),
        },
    }
    identity_out = ROOT / (PREFIX + "-build-identity.json")
    identity_out.write_text(json.dumps(build_identity, indent=2) + "\n")
    state.update(
        complete=True,
        phase="installed-tests-complete",
        wheel_sha256=sha(wheel),
        tests=sum(int(s.attrib["tests"]) for s in suites),
        skipped=sum(int(s.attrib["skipped"]) for s in suites),
        junit_sha256=sha(junit),
        build_identity_sha256=sha(identity_out),
        cargo_fingerprint_sha256=sha(compiled),
    )
    save()
    print("P4 audited; candidate build and installed tests completed", flush=True)
except BaseException as error:
    state["error"] = f"{type(error).__name__}: {error}"
    save()
    raise
