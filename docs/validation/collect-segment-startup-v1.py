"""Preserve a terminal Linux Segmentation P3/P4 phase and recalculate its audit.

Run explicitly after the selected phase and its server audit have succeeded.
The archive embeds the previously sealed P1 evidence, including exact source,
wheel, fixture manifests and its documented incomplete Linux license bundle.
No measured libraries are imported and no corpus/cache bytes are read or changed.
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REMOTE = Path("/home/yonghye/ultrafast-vision-build")
PREFIX = "dataset-segment-500k-linux-v1"
P1_ARCHIVE = REPO / "bench/results/segment-500k-linux-p1-complete-v1.tar.gz"
P1_SHA256 = "4092e95c9d7c4e1cf982c293c2192848fa874c9f981e0913c78d530bcd210fdb"
AUDITOR = REPO / "bench/audit_startup.py"
AUDITOR_SHA256 = "d745e43cc6c63dc77335d5756c315447f8a22a86f70b483b7e812456c497b80d"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def collect(phase):
    stem = f"segment-500k-linux-{phase}"
    archive_path = REPO / f"bench/results/{stem}-complete-v1.tar.gz"
    receipt_path = REPO / f"docs/validation/{stem}-preservation-v1.json"
    report_target = REPO / f"docs/validation/{stem}-report-v1.json"
    audit_target = REPO / f"docs/validation/{stem}-audit-v1.json"
    require(not any(p.exists() for p in (archive_path, receipt_path, report_target, audit_target)),
            "preserve earlier artifacts; inspect them before any retry")
    p1_bytes = P1_ARCHIVE.read_bytes()
    require(sha(p1_bytes) == P1_SHA256, "previous P1 archive identity mismatch")
    require(sha(AUDITOR.read_bytes()) == AUDITOR_SHA256, "use the qualified frozen auditor")
    with tempfile.TemporaryDirectory(prefix=f"segment-{phase}-collection-") as temporary:
        root = Path(temporary)
        state_path = root / f"{PREFIX}-state.json"
        subprocess.run(["scp", f"yonghye-pc:{REMOTE / state_path.name}", str(state_path)], check=True)
        state = read(state_path)
        allowed = ("p4", "all-three-phases-audited") if phase == "p3" else ("all-three-phases-audited",)
        require(state["phase"] in allowed and "error" not in state, "selected phase has not succeeded")
        if phase == "p4":
            require(state["complete"] is True, "P4 controller completion is unproven")
        commands = {}
        for name in (phase, phase + "-audit"):
            matches = [c for c in state["commands"] if Path(c["log"]).name == f"{PREFIX}-{name}.log"]
            require(len(matches) == 1 and matches[0]["returncode"] == 0, f"{name} did not terminate successfully")
            commands[name] = matches[0]
        names = [f"{PREFIX}-{suffix}" for suffix in (
            f"{phase}.json", f"{phase}.log", f"{phase}-audit.json", f"{phase}-audit.log",
            "identity.json", "preflight.json",
        )]
        subprocess.run(["scp", *[f"yonghye-pc:{REMOTE / n}" for n in names],
                        f"yonghye-pc:{REMOTE / 'startup-segment-500k-linux-v1/startup.json'}", str(root)], check=True)
        runs = root / f"{PREFIX}-{phase}.runs"
        subprocess.run(["scp", "-r", f"yonghye-pc:{REMOTE / runs.name}", str(runs)], check=True)
        report = root / f"{PREFIX}-{phase}.json"
        audited = root / f"{PREFIX}-{phase}-audit.json"
        identity = root / f"{PREFIX}-identity.json"
        preflight = root / f"{PREFIX}-preflight.json"
        require(state["identity_sha256"] == sha(identity.read_bytes()), "controller identity mismatch")
        for name, command in commands.items():
            require(command["log_sha256"] == sha((root / f"{PREFIX}-{name}.log").read_bytes()),
                    f"{name} log identity mismatch")
        with tarfile.open(P1_ARCHIVE) as p1:
            for name in (identity.name, preflight.name, "startup.json"):
                require(p1.extractfile(name).read() == (root / name).read_bytes(), f"P1/{phase} anchor mismatch: {name}")
            capacity = json.loads(p1.extractfile(f"{PREFIX}-capacity-reference.json").read())
        spec = importlib.util.spec_from_file_location("independent_startup_audit", AUDITOR)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        observed = module.audit(report, identity, preflight, root / "startup.json",
                                phase, "segment", 500000, 5, 8, runs_dir=runs)
        require(observed == read(audited), "local audit disagrees with server audit")
        require(observed["complete_requested_samples"] and observed["completed_workers"] == 10,
                "incomplete requested samples")
        require(observed["output_sha256"] == capacity["output_sha256"] == state["capacity"]["output_sha256"],
                "outputs differ from full-capacity reference")
        expected_raw = {f"{i}-{backend}.json" for i in range(5) for backend in
                        (("reference", "native-content") if phase == "p3" else ("reference-content", "native-content"))}
        if phase == "p4":
            expected_raw.update({"prime-reference-content.json", "prime-native-content.json"})
        require(set(observed["raw_sha256"]) == expected_raw, "wrong worker/primer inventory")
        expected_runs = expected_raw | {str(Path(n).with_suffix(".log")) for n in expected_raw}
        require({p.name for p in runs.iterdir()} == expected_runs, "missing or unexpected worker files")
        require(all(p.is_file() and not p.is_symlink() for p in runs.iterdir()), "nonregular worker artifacts")
        (root / "p1-evidence.tar.gz").write_bytes(p1_bytes)
        (root / "audit_startup.py").write_bytes(AUDITOR.read_bytes())
        (root / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        files = {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
                 for p in sorted(root.rglob("*")) if p.is_file()}
        sealed = root / "sealed.tar.gz"
        with tarfile.open(sealed, "x:gz", compresslevel=1) as archive:
            for name in files:
                archive.add(root / name, arcname=name, recursive=False)
        with tarfile.open(sealed) as archive:
            members = archive.getmembers()
            require(len(members) == len(files) and all(m.isfile() for m in members), "archive member inventory mismatch")
            require({m.name: {"bytes": m.size, "sha256": sha(archive.extractfile(m).read())} for m in members} == files,
                    "archive readback mismatch")
        result = {
            "complete": True, "phase": phase, "archive": str(archive_path.relative_to(REPO)),
            "archive_sha256": sha(sealed.read_bytes()), "archive_bytes": sealed.stat().st_size,
            "files": files, "file_count": len(files), "embedded_p1_archive_sha256": P1_SHA256,
            "local_independent_recalculation_matches": True, "all_archive_members_read_back": True,
            "output_sha256": observed["output_sha256"], "summary": observed["summary"],
            "limits": observed["limits"] + [
                "The embedded P1 archive retains exact source/wheel bytes and the older Linux wheel's missing dependency-license bundle finding; redistribution qualification remains open.",
                "The million input files and generated caches remain external; their bytes are not re-read by this collector.",
                "Controller state is a checkpoint after the selected phase; no later phase is certified by this receipt.",
                "P3 compares ordinary reference generation with stronger native input revalidation; P4 uses content-validated caches on both sides.",
                "No runtime tests, ML imports or benchmark jobs are repeated during collection.",
            ],
        }
        for target, data in ((archive_path, sealed.read_bytes()), (report_target, report.read_bytes()),
                             (audit_target, audited.read_bytes()),
                             (receipt_path, (json.dumps(result, indent=2) + "\n").encode())):
            with target.open("xb") as stream:
                stream.write(data)
            require(target.read_bytes() == data, f"published artifact readback mismatch: {target}")
        print(json.dumps({k: result[k] for k in ("phase", "archive_sha256", "file_count", "summary")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("p3", "p4"))
    collect(parser.parse_args().phase)
