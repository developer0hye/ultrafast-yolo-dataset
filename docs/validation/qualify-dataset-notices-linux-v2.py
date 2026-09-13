"""Build a frozen dataset wheel and qualify a fresh combined Linux runtime."""

import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path("/home/yonghye/ultrafast-vision-build")
PREFIX = "dataset-notices-linux-v2"
SOURCE = ROOT / (PREFIX + "-source")
DIST = ROOT / (PREFIX + "-dist")
CLEAN = ROOT / (PREFIX + "-clean")
BUILD_PY = ROOT / "venv312-cuda/bin/python"
PYTHON = CLEAN / "bin/python"
UV = "/home/yonghye/.local/bin/uv"
CARGO = "/home/yonghye/.cargo/bin/cargo"
RUSTC = "/home/yonghye/.cargo/bin/rustc"
MASK_SOURCE = ROOT / "mask-shared-linux-v2-source"
ENV = os.environ.copy()
for key in ("PYTHONPATH", "PYTHONHOME", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"):
    ENV.pop(key, None)
ENV.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    YOLO_OFFLINE="true",
    CARGO_BUILD_JOBS="2",
    CARGO_TARGET_DIR=str(ROOT / (PREFIX + "-target")),
    PYO3_PYTHON=str(BUILD_PY),
    RUSTUP_TOOLCHAIN="1.98.0",
    PATH="/home/yonghye/.cargo/bin:" + ENV["PATH"],
)
STATE = {
    "complete": False,
    "passed": False,
    "controller_pid": os.getpid(),
    "started_at_ns": time.time_ns(),
    "commands": [],
}


def path(suffix):
    return ROOT / (PREFIX + "-" + suffix)


def sha(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def save():
    temporary = path("qualification.tmp")
    temporary.write_text(json.dumps(STATE, indent=2) + "\n")
    os.replace(temporary, path("qualification.json"))


def run(label, command, *, cwd=SOURCE, timeout=900):
    record = {
        "label": label,
        "command": list(map(str, command)),
        "cwd": str(cwd),
        "started_at_ns": time.time_ns(),
        "log": str(path(label + ".log")),
    }
    STATE["commands"].append(record)
    save()
    print(label, "started", flush=True)
    with Path(record["log"]).open("x") as output:
        child = subprocess.Popen(
            record["command"], cwd=cwd, env=ENV, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        )
        record["pid"] = child.pid
        save()
        try:
            record["returncode"] = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            record["timed_out"] = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            record["returncode"] = child.returncode
            raise
        finally:
            record.update(finished_at_ns=time.time_ns(), log_sha256=sha(record["log"]))
            save()
    print(label, "returncode", record["returncode"], flush=True)
    assert record["returncode"] == 0, Path(record["log"]).read_text()[-10000:]
    return Path(record["log"]).read_text()


def pins(lines):
    result = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        name = re.split(r"==| @ ", line, maxsplit=1)[0]
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized not in ("pip", "ultrafast-maskops", "ultrafast-yolo-dataset"):
            assert normalized not in result
            result[normalized] = line[len(name) :]
    return result


def suite(label, source):
    collected = run(label + "-collect", [PYTHON, "-m", "pytest", "--collect-only", "-q", "tests"], cwd=source)
    planned = [line for line in collected.splitlines() if line.startswith("tests/") and "::" in line]
    assert len(planned) > 100 and len(planned) == len(set(planned))
    run(label, [PYTHON, "-m", "pytest", "-q", "tests", "--junitxml=" + str(path(label + ".xml"))], cwd=source)
    cases = ET.parse(path(label + ".xml")).findall(".//testcase")
    failed = [c.attrib for c in cases if c.find("failure") is not None or c.find("error") is not None]
    skipped = [c.attrib for c in cases if c.find("skipped") is not None]
    result = {
        "collected": len(planned),
        "cases": len(cases),
        "failed": failed,
        "skipped": skipped,
        "xml_sha256": sha(path(label + ".xml")),
        "collection_log_sha256": sha(path(label + "-collect.log")),
    }
    STATE[label] = result
    save()
    assert len(cases) == len(planned) and not failed and not skipped


def main():
    assert not path("qualification.json").exists() and not DIST.exists() and not CLEAN.exists()
    assert not Path(ENV["CARGO_TARGET_DIR"]).exists()
    source = json.loads(path("source.json").read_text())
    assert source["source_commit"] == "16d4d6175e749c7213fc3bd24caed7815b59c053"
    assert sha(path("source.tar")) == source["archive_sha256"]
    for name, expected in source["files"].items():
        assert sha(SOURCE / name) == expected, name
    mask_build = json.loads((ROOT / "mask-shared-linux-v2-build.json").read_text())
    (mask_wheel,) = (ROOT / "mask-shared-linux-v2-dist").glob("*.whl")
    assert sha(mask_wheel) == "8b467c837c323812a0201b81210a6dcd0c180ef50ba0d16531f07ef2c741cd9a"
    assert mask_build["artifacts"][mask_wheel.name] == sha(mask_wheel)
    mask_source = json.loads((ROOT / "mask-shared-linux-v2-source.json").read_text())
    for name, expected in mask_source["files"].items():
        assert sha(MASK_SOURCE / name) == expected, name
    STATE.update(
        source_commit=source["source_commit"],
        source_archive_sha256=source["archive_sha256"],
        mask_wheel=str(mask_wheel),
        mask_wheel_sha256=sha(mask_wheel),
        controller_sha256=sha(__file__),
        environment_overrides={
            k: ENV[k]
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "YOLO_OFFLINE",
                "CARGO_BUILD_JOBS",
                "CARGO_TARGET_DIR",
                "PYO3_PYTHON",
                "RUSTUP_TOOLCHAIN",
            )
        },
    )
    save()
    rust = run("rustc", [RUSTC, "-Vv"])
    assert "release: 1.98.0\n" in rust and "host: x86_64-unknown-linux-gnu\n" in rust
    sysroot = Path(run("rust-sysroot", [RUSTC, "--print", "sysroot"]).strip())
    supplied = sysroot / "share/doc/rust/COPYRIGHT-library.html"
    bundled = SOURCE / "python/ultrafast_yolo_dataset/licenses/rust-stdlib/1.98.0/COPYRIGHT-library.html"
    assert supplied.read_bytes() == bundled.read_bytes()
    STATE["compiler_notice_sha256"] = sha(bundled)
    run("cargo-fmt", [CARGO, "fmt", "--check"])
    run(
        "build",
        [UV, "build", SOURCE, "--python", BUILD_PY, "--wheel", "--sdist", "--no-build-isolation", "--out-dir", DIST],
    )
    (wheel,) = DIST.glob("*.whl")
    (sdist,) = DIST.glob("*.tar.gz")
    STATE["artifacts"] = {p.name: sha(p) for p in (wheel, sdist)}
    STATE["build_passed"] = True
    save()
    run("cargo-tests", [CARGO, "test", "--lib", "--locked"])
    run("cargo-clippy", [CARGO, "clippy", "--all-targets", "--locked", "--", "-D", "warnings"])
    run("standalone-venv", [UV, "venv", "--seed", "--python", BUILD_PY, CLEAN])
    assert "include-system-site-packages = false" in (CLEAN / "pyvenv.cfg").read_text()
    run(
        "standalone-dependencies",
        [UV, "pip", "install", "--python", PYTHON, "--no-deps", "numpy==2.4.4", "pillow==12.1.1", "pip==26.2.1"],
    )
    run("standalone-wheel", [UV, "pip", "install", "--python", PYTHON, "--no-deps", wheel])
    run(
        "standalone-isolation",
        [
            PYTHON,
            "-c",
            'import importlib.util; assert all(importlib.util.find_spec(x) is None for x in ("torch", "ultralytics", "cv2", "ultrafast_maskops"))',
        ],
    )
    assert not (SOURCE / "dist").exists()
    (SOURCE / "dist").symlink_to(DIST, target_is_directory=True)
    run("standalone-audit", [PYTHON, "bench/ci_wheel_check.py"])
    audit = json.loads((DIST / "wheel-validation.json").read_text())
    assert audit["notice_files_checked"] == 137 and audit["clean"]
    run("standalone-cache", [PYTHON, "bench/wheel_smoke.py", "--out", path("standalone-cache.json")])
    smoke = json.loads(path("standalone-cache.json").read_text())
    assert len(smoke["cases"]) == 4 and all(c["parity"] for c in smoke["cases"])
    assert smoke["content_edit_detected"] and smoke["metadata_weaker_contract_confirmed"]
    STATE.update(
        standalone_passed=True, notice_files_checked=137, standalone_audit_sha256=sha(DIST / "wheel-validation.json")
    )
    save()
    baseline = ROOT / "mask-unit-scale-linux-freeze-v1.txt"
    lines = baseline.read_text().splitlines()
    excluded = [line for line in lines if line.startswith("ultrafast-maskops @ ")]
    assert len(excluded) == 1
    requirements = path("requirements.txt")
    requirements.write_text("\n".join(line for line in lines if line not in excluded) + "\n")
    STATE.update(
        baseline_freeze_sha256=sha(baseline), requirements_sha256=sha(requirements), excluded_old_mask_wheel=excluded
    )
    run("combined-dependencies", [UV, "pip", "install", "--python", PYTHON, "--no-deps", "-r", requirements])
    run("combined-mask-wheel", [UV, "pip", "install", "--python", PYTHON, "--no-deps", mask_wheel])
    run("pip-check", [PYTHON, "-m", "pip", "check"])
    freeze = run("freeze", [PYTHON, "-m", "pip", "freeze"])
    assert pins(lines) == pins(freeze.splitlines()) and len(pins(lines)) == 79
    STATE["normalized_dependency_pins_match"] = True
    installed = Path(
        run("installed-root", [PYTHON, "-c", 'import sysconfig; print(sysconfig.get_path("platlib"))']).strip()
    )
    for manifest, origin in ((source, SOURCE), (mask_source, MASK_SOURCE)):
        for name, expected in manifest["files"].items():
            assert sha(origin / name) == expected, name
            if name.startswith("python/"):
                assert sha(installed / name.removeprefix("python/")) == expected, name
    run(
        "combined-dataset-audit",
        [
            PYTHON,
            "bench/audit_wheel.py",
            "--wheel",
            wheel,
            "--installed-root",
            installed,
            "--out",
            path("combined-dataset-audit.json"),
        ],
    )
    run(
        "combined-mask-audit",
        [
            PYTHON,
            "bench/audit_wheel.py",
            "--wheel",
            mask_wheel,
            "--installed-root",
            installed,
            "--out",
            path("combined-mask-audit.json"),
        ],
        cwd=MASK_SOURCE,
    )
    suite("dataset-tests", SOURCE)
    suite("mask-tests", MASK_SOURCE)
    for manifest, origin in ((source, SOURCE), (mask_source, MASK_SOURCE)):
        for name, expected in manifest["files"].items():
            assert sha(origin / name) == expected, name
            if name.startswith("python/"):
                assert sha(installed / name.removeprefix("python/")) == expected, name
    assert run("rustc-after", [RUSTC, "-Vv"]) == rust
    STATE.update(
        complete=True,
        passed=True,
        finished_at_ns=time.time_ns(),
        installed_sources_unchanged=True,
        gpu_training_executed=False,
    )
    save()
    print(
        json.dumps(
            {k: STATE[k] for k in ("complete", "passed", "dataset-tests", "mask-tests", "gpu_training_executed")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    # Reserve before entering the failure handler so reruns cannot overwrite old evidence.
    assert not path("qualification.json").exists()
    try:
        main()
    except BaseException:
        STATE.update(complete=True, passed=False, error=traceback.format_exc(), finished_at_ns=time.time_ns())
        save()
        raise
