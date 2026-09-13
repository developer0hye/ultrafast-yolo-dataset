"""Wait for exact macOS startup processes, then qualify and run the auditor.

Uses staged script copies. Never restarts a measurement, imports native libraries,
or reads full-size caches before the measured processes have terminated.
"""

import ast
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path("/Volumes/T7/ultrafast-vision-build")
PREFIX = "snapshot-startup-full-m2-v1-qualification"
STAGE = ROOT / (PREFIX + "-stage")
OUTPUT = ROOT / (PREFIX + ".json")
DEPENDENCY = ROOT / (PREFIX + "-dependency.json")
FILES = ("audit-snapshot-startup-v1.py", "qualify-snapshot-startup-audit-v1.py", "pyproject.toml",
         "pilot-arguments.json", "full-arguments.json")
STATE = {"complete": False, "passed": False, "controller_pid": os.getpid(), "commands": []}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save():
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(STATE, indent=2) + "\n")
    os.replace(temporary, OUTPUT)


def process(pid):
    result = subprocess.run(["ps", "-ww", "-p", str(pid), "-o", "stat=", "-o", "lstart=", "-o", "command="],
                            text=True, capture_output=True)
    if result.returncode == 1 and not result.stdout.strip():
        return None
    result.check_returncode()
    status, identity = result.stdout.strip().split(maxsplit=1)
    return {"status": status, "identity": identity}


def boot_time():
    return subprocess.check_output(["sysctl", "-n", "kern.boottime"], text=True).strip()


def run(label, command):
    log = ROOT / (PREFIX + "-" + label + ".log")
    record = {"label": label, "command": command, "cwd": str(STAGE), "started_at_ns": time.time_ns()}
    STATE["commands"].append(record)
    save()
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", YOLO_OFFLINE="true")
    with log.open("x") as stream:
        child = subprocess.Popen(command, cwd=STAGE, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record["pid"] = child.pid
        save()
        code = child.wait()
    record.update(returncode=code, finished_at_ns=time.time_ns(), log_sha256=sha(log))
    save()
    print(label, "returncode", code, flush=True)
    if code:
        print(log.read_text()[-10000:], flush=True)
        raise RuntimeError(f"{label} failed; retain this first attempt")


def main():
    assert not OUTPUT.exists(), "preserve earlier qualification"
    dependency = json.loads(DEPENDENCY.read_text())
    assert boot_time() == dependency["boot_time"]
    for pid, identity in dependency["processes"].items():
        current = process(int(pid))
        assert current is not None and current["identity"] == identity and "Z" not in current["status"]
    before = {name: sha(STAGE / name) for name in FILES}
    STATE.update(phase="waiting-for-measurement", started_at_ns=time.time_ns(),
                 script_sha256=sha(Path(__file__)), dependency_sha256=sha(DEPENDENCY),
                 source_before_sha256=before, benchmark_restarted=False)
    save()
    print("Waiting for exact startup handles", list(dependency["processes"]), flush=True)
    while True:
        assert boot_time() == dependency["boot_time"], "host rebooted during wait"
        live = []
        for pid, identity in dependency["processes"].items():
            current = process(int(pid))
            if current is not None and current["identity"] == identity and "Z" not in current["status"]:
                live.append(pid)
        if not live:
            break
        time.sleep(20)
    launch_path = ROOT / "snapshot-startup-full-m2-v1-launch.json"
    launch = json.loads(launch_path.read_text())
    report_path = ROOT / "snapshot-startup-full-m2-v1.json"
    report = json.loads(report_path.read_text())
    assert launch["complete"] is True and launch["returncode"] == 0
    assert launch["outputs_match_original_full_reference"] is True
    assert report["complete"] is True and report["rounds"] == 5 and report["pairs"] == 500000
    assert len(report["records"]) == 20 and len(report["primers"]) == 2
    assert sha(report_path) == launch["report_sha256"]
    assert {name: sha(STAGE / name) for name in FILES} == before, "stage changed while waiting"
    STATE.update(phase="qualifying-auditor", launch_sha256=sha(launch_path), report_sha256=sha(report_path),
                 qualification_started_at_ns=time.time_ns())
    save()
    python_files = list(FILES[:2])
    trees = {name: ast.dump(ast.parse((STAGE / name).read_text()), include_attributes=False) for name in python_files}
    ruff = ["/Users/yhkwon/.local/bin/uvx", "ruff==0.15.6"]
    run("format", ruff + ["format", *python_files])
    assert all(ast.dump(ast.parse((STAGE / name).read_text()), include_attributes=False) == tree for name, tree in trees.items())
    STATE.update(formatting_preserved_ast=True, source_after_sha256={name: sha(STAGE / name) for name in FILES})
    save()
    run("lint", ruff + ["check", *python_files])
    run("format-check", ruff + ["format", "--check", *python_files])
    python = str(ROOT / "dataset-snapshot-scratch-m2-v1-clean/bin/python")
    negative_path = ROOT / (PREFIX + "-negative.json")
    run("negative", [python, "qualify-snapshot-startup-audit-v1.py", "--arguments", "pilot-arguments.json",
                     "--out", str(negative_path)])
    negative = json.loads(negative_path.read_text())
    assert negative["complete"] and len(negative["controls"]) == 3 and negative["negative_case_count"] == 35
    assert len(negative["negative_cases"]) == len({case["case"] for case in negative["negative_cases"]}) == 35
    assert all(case["rejected"] for case in negative["negative_cases"])
    assert negative["auditor_sha256"] == sha(STAGE / FILES[0])
    assert negative["qualifier_sha256"] == sha(STAGE / FILES[1])
    assert negative["arguments_sha256"] == sha(STAGE / "pilot-arguments.json")
    STATE.update(phase="auditing-full-series", negative_sha256=sha(negative_path))
    save()
    arguments = json.loads((STAGE / "full-arguments.json").read_text())
    audit_path = ROOT / (PREFIX + "-audit.json")
    command = [python, "audit-snapshot-startup-v1.py"]
    for name, value in arguments.items():
        command.extend(["--" + name.replace("_", "-"), str(value)])
    command.extend(["--out", str(audit_path)])
    run("audit", command)
    audit = json.loads(audit_path.read_text())
    assert audit["complete"] and audit["measured_workers"] == 20 and audit["separate_primers"] == 2
    assert audit["pairs"] == 500000 and audit["rounds"] == 5
    assert audit["report_sha256"] == STATE["report_sha256"] and audit["auditor_sha256"] == negative["auditor_sha256"]
    assert {name: sha(STAGE / name) for name in FILES} == STATE["source_after_sha256"]
    STATE.update(complete=True, passed=True, phase="audited-full-series", finished_at_ns=time.time_ns(),
                 audit_sha256=sha(audit_path),
                 scope="35 artifact rejection cases and three controls, including synthetic five-round controls; independent audit of 20 actual measured workers and two primers. Native-versus-native incremental startup, not upstream Python or training. Evidence archive/readback remains separate.")
    save()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        STATE.update(complete=True, passed=False, error=traceback.format_exc(), finished_at_ns=time.time_ns())
        save()
        raise
