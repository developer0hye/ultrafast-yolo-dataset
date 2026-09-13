"""Build and test a selected, frozen source archive in a fresh M2 environment."""

import hashlib
import json
import os
import subprocess
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path("/Volumes/T7/ultrafast-vision-build")
REPO = Path(__file__).parents[2]
REF = "9e2ab29"
PREFIX = "dataset-snapshot-scratch-m2-v1"
BASEPY = ROOT / "venv/bin/python"
RUNTIME = ROOT / "dataset-cache-read-copy-m2-v1-clean/bin/python"
UV = Path("/Users/yhkwon/.local/bin/uv")
SOURCE, TARGET, DIST, CLEAN = [ROOT / (PREFIX + "-" + n) for n in ("source", "target", "dist", "clean")]
STATE = ROOT / (PREFIX + "-state.json")
assert not STATE.exists() and not any(p.exists() for p in (SOURCE, TARGET, DIST, CLEAN))
ENV = dict(
    os.environ,
    RUSTUP_TOOLCHAIN="stable",
    CARGO_TARGET_DIR=str(TARGET),
    CARGO_BUILD_JOBS="4",
    PYO3_PYTHON=str(RUNTIME),
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    YOLO_OFFLINE="true",
    MACOSX_DEPLOYMENT_TARGET="11.0",
)
for key in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET"):
    ENV.pop(key, None)
state = {
    "complete": False,
    "commands": [],
    "source_ref": REF,
    "environment_overrides": {
        k: ENV[k]
        for k in (
            "RUSTUP_TOOLCHAIN",
            "CARGO_TARGET_DIR",
            "CARGO_BUILD_JOBS",
            "PYO3_PYTHON",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "YOLO_OFFLINE",
            "MACOSX_DEPLOYMENT_TARGET",
        )
    },
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save():
    STATE.write_text(json.dumps(state, indent=2) + "\n")


def run(command, name, cwd=None):
    log = ROOT / (PREFIX + "-" + name + ".log")
    command = list(map(str, command))
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=cwd, env=ENV, stdout=stream, stderr=subprocess.STDOUT, check=False)
    state["commands"].append(
        {
            "command": command,
            "cwd": str(cwd) if cwd else None,
            "returncode": result.returncode,
            "log": str(log),
            "log_sha256": sha(log),
        }
    )
    save()
    assert result.returncode == 0, f"{name} failed: {log}"
    print(name + " passed", flush=True)
    return log


state["script_sha256"] = sha(Path(__file__))
save()
try:
    state["source_commit"] = subprocess.check_output(["git", "rev-parse", REF], cwd=REPO, text=True).strip()
    state["rustc"] = subprocess.check_output(["/Users/yhkwon/.cargo/bin/rustc", "+stable", "-Vv"], text=True)
    state["cargo"] = subprocess.check_output(["/Users/yhkwon/.cargo/bin/cargo", "+stable", "-V"], text=True)
    assert "release: 1.98.0\n" in state["rustc"] and "LLVM version: 22.1.8" in state["rustc"]
    assert state["cargo"].startswith("cargo 1.98.0 ")
    tracked = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", REF], cwd=REPO, text=True).splitlines()
    selected = [
        name
        for name in tracked
        if name.startswith(("src/", "python/", "tests/"))
        or (name.startswith("bench/") and name.endswith(".py"))
        or name
        in {"Cargo.toml", "Cargo.lock", "pyproject.toml", "README.md", "LICENSE", "NOTICE", "docs/SNAPSHOT_SCRATCH.md"}
    ]
    archive = ROOT / (PREFIX + "-source.tar")
    run(["git", "archive", "--format=tar", "--output", archive, REF, *selected], "archive", REPO)
    SOURCE.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(SOURCE, filter="data")
    state["source_selection"] = {name: sha(SOURCE / name) for name in selected}
    state["source_archive_sha256"] = sha(archive)
    state["source_scope"] = (
        "Selected complete library/build/test/benchmark sources; historical result archives and unrelated docs omitted"
    )
    save()
    cargo = Path("/Users/yhkwon/.cargo/bin/cargo")
    run([cargo, "+stable", "fmt", "--all", "--check"], "fmt", SOURCE)
    run([cargo, "+stable", "test", "--lib", "--locked", "--offline"], "rust-tests", SOURCE)
    run(
        [cargo, "+stable", "clippy", "--all-targets", "--locked", "--offline", "--", "-D", "warnings"], "clippy", SOURCE
    )
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
            DIST,
        ],
        "build",
        SOURCE,
    )
    run([BASEPY, "-m", "maturin", "sdist", "--out", DIST], "sdist", SOURCE)
    (wheel,) = DIST.glob("*.whl")
    state["wheel_sha256"] = sha(wheel)
    run([UV, "venv", "--offline", "--seed", "--python", RUNTIME, CLEAN], "venv")
    python = CLEAN / "bin/python"
    run(
        [UV, "pip", "install", "--offline", "--no-deps", "--python", python, "numpy==2.4.4", "pillow==12.1.1", wheel],
        "standalone-install",
    )
    (SOURCE / "dist").symlink_to(DIST, target_is_directory=True)
    run([python, "bench/ci_wheel_check.py"], "wheel-check", SOURCE)
    run([python, "bench/wheel_smoke.py", "--out", ROOT / (PREFIX + "-standalone.json")], "standalone-smoke", SOURCE)
    run(
        [
            UV,
            "pip",
            "install",
            "--offline",
            "--no-deps",
            "--python",
            python,
            "-r",
            ROOT / "dataset-cache-read-copy-m2-v1-requirements.txt",
        ],
        "runtime-install",
    )
    run([UV, "pip", "check", "--python", python], "pip-check")
    run([UV, "--quiet", "pip", "freeze", "--python", python], "freeze")
    xml = ROOT / (PREFIX + "-tests.xml")
    run([python, "-m", "pytest", "-q", "tests", "--junitxml", xml], "tests", SOURCE)
    suites = ET.parse(xml).getroot().findall("testsuite")
    assert sum(int(s.attrib["tests"]) for s in suites) == 193
    assert all(int(s.attrib[k]) == 0 for s in suites for k in ("failures", "errors", "skipped"))
    cases = [c for s in suites for c in s.findall("testcase")]
    assert sum(c.attrib["classname"].endswith("test_snapshot_scratch") for c in cases) == 6
    assert all(sha(SOURCE / name) == value for name, value in state["source_selection"].items())
    state.update(complete=True, tests=193, junit_sha256=sha(xml))
    save()
    print("Frozen candidate build and 193 installed tests passed; performance unmeasured.")
except BaseException as error:
    state["error"] = f"{type(error).__name__}: {error}"
    save()
    raise
