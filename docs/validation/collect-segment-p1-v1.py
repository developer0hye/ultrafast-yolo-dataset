"""Preserve the terminal Linux Segmentation P1 phase while P3/P4 continue.

Read-only SSH collection and a small local artifact audit, not another benchmark.
The full fixture is represented by its generator/source and verified manifests.
"""

import hashlib
import importlib.util
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path
from zipfile import ZipFile

REPO = Path(__file__).resolve().parents[2]
PREFIX = "dataset-segment-500k-linux-v1"
REMOTE = Path("/home/yonghye/ultrafast-vision-build")
ARCHIVE = REPO / "bench/results/segment-500k-linux-p1-complete-v1.tar.gz"
RECEIPT = REPO / "docs/validation/segment-500k-linux-p1-preservation-v1.json"
AUDITOR = REPO / "bench/audit_startup.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


assert not ARCHIVE.exists() and not RECEIPT.exists(), "preserve earlier evidence"
assert sha(AUDITOR) == "d745e43cc6c63dc77335d5756c315447f8a22a86f70b483b7e812456c497b80d"
with tempfile.TemporaryDirectory(prefix="segment-p1-collection-") as temporary:
    root = Path(temporary)
    remote_files = [
        REMOTE / f"{PREFIX}-{suffix}" for suffix in (
            "p1.json", "p1.log", "p1-audit.json", "p1-audit.log", "identity.json",
            "preflight.json", "state.json", "source.json", "source.tar",
            "capacity-reference.json", "capacity-native-content.json",
        )
    ] + [
        REMOTE / "run-segment-500k-linux-v1.py",
        REMOTE / "startup-segment-500k-linux-v1/startup.json",
        REMOTE / "startup-segment-500k-linux-v1/labels/manifest.json",
        REMOTE / "cuda-wheels-byte-access-v2/dataset/ultrafast_yolo_dataset-0.1.0a1-cp312-cp312-linux_x86_64.whl",
    ]
    subprocess.run(["scp", *[f"yonghye-pc:{path}" for path in remote_files], str(root)], check=True)
    report, identity, preflight, audited = (root / f"{PREFIX}-{suffix}.json" for suffix in ("p1", "identity", "preflight", "p1-audit"))
    assert sha(report) == "a421cfa3dcc29e366bd34b283cd5a00e95bd308131a8cc338b0c6022b1c26fb3"
    assert sha(identity) == "53b90b9d03b706e91f8c710928558009ef380aeeaa4c54c17e358784db7ad140"
    anchor, state, selection = read(identity), read(root / f"{PREFIX}-state.json"), read(root / f"{PREFIX}-source.json")
    assert state["phase"] in ("p3", "p4", "all-three-phases-audited")
    assert state["identity_sha256"] == sha(identity)
    for name in ("p1", "p1-audit"):
        (command,) = [c for c in state["commands"] if Path(c["log"]).name == f"{PREFIX}-{name}.log"]
        assert command["returncode"] == 0 and command["log_sha256"] == sha(root / f"{PREFIX}-{name}.log")
    assert anchor["script_sha256"] == sha(root / "run-segment-500k-linux-v1.py")
    assert sha(root / f"{PREFIX}-source.tar") == anchor["source_archive_sha256"] == selection["archive_sha256"]
    assert selection["commit"] == anchor["source_commit"] == "3c4610c437122af08a3db26840bbbe3cf7c12ea6"
    with tarfile.open(root / f"{PREFIX}-source.tar") as source:
        members = source.getmembers()
        regular = [m for m in members if m.isfile()]
        assert len(regular) == 176 and all(m.isfile() or m.isdir() for m in members)
        assert len({m.name for m in members}) == len(members)
        assert {m.name: hashlib.sha256(source.extractfile(m).read()).hexdigest() for m in regular} == selection["selected_files"]
        h = hashlib.sha256()
        for name in ("src/parser.rs", "src/snapshot.rs", "src/provenance.rs", "src/materialize.rs", "src/cache.rs", "src/lib.rs", "Cargo.toml", "Cargo.lock"):
            h.update(source.extractfile(name).read())
        assert h.hexdigest() == anchor["native_cache_profile"]
        (wheel,) = root.glob("*.whl")
        assert sha(wheel) == anchor["wheel_sha256"]
        wheel_sources = {"runtime_matches": [], "missing_license_files": [], "different_license_files": []}
        with ZipFile(wheel) as zipped:
            (extension,) = [name for name in zipped.namelist() if name.startswith("ultrafast_yolo_dataset/_native.") and name.endswith(".so")]
            assert hashlib.sha256(zipped.read(extension)).hexdigest() == anchor["native_extension_sha256"]
            for name in selection["selected_files"]:
                relative = name.removeprefix("python/")
                if name.startswith("python/ultrafast_yolo_dataset/licenses/"):
                    if relative not in zipped.namelist():
                        wheel_sources["missing_license_files"].append(relative)
                    elif zipped.read(relative) != source.extractfile(name).read():
                        wheel_sources["different_license_files"].append(relative)
                elif name.startswith("python/ultrafast_yolo_dataset/") and name.endswith((".py", ".json")):
                    assert zipped.read(relative) == source.extractfile(name).read()
                    wheel_sources["runtime_matches"].append(relative)
            wheel_sources["existing_license_entries"] = [n for n in zipped.namelist() if "/licenses/" in n]
        wheel_sources["license_bundle_complete"] = not (
            wheel_sources["missing_license_files"] or wheel_sources["different_license_files"]
        )
        wheel_sources["scope"] = "Benchmark runtime sources/native digest match. The older Linux wheel has project LICENSE/NOTICE but lacks the later 137-file dependency-license bundle; release redistribution qualification remains open."
        assert len(wheel_sources["missing_license_files"]) == 137 and not wheel_sources["different_license_files"]
    (root / "wheel-source-audit.json").write_text(json.dumps(wheel_sources, indent=2) + "\n")
    assert read(root / "manifest.json") == read(root / "startup.json")["labels"]
    spec = importlib.util.spec_from_file_location("independent_startup_audit", AUDITOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed = module.audit(report, identity, preflight, root / "startup.json", "p1", "segment", 500000, 5, 4)
    assert observed == read(audited), "independent local readback/recalculation disagrees"
    assert observed["complete_requested_samples"] and observed["completed_workers"] == 10
    storage = {}
    for name, command in (
        ("findmnt", ["findmnt", "-J", "-T", str(REMOTE), "-o", "SOURCE,FSTYPE,OPTIONS"]),
        ("lsblk", ["lsblk", "-J", "-o", "NAME,MODEL,TRAN,ROTA,FSTYPE,MOUNTPOINTS"]),
    ):
        value = subprocess.run(["ssh", "yonghye-pc", *command], capture_output=True, text=True, check=True)
        storage[name] = json.loads(value.stdout)
    storage["scope"] = "Read-only storage inventory after P1 completed; not device-throughput or cold-cache evidence."
    (root / "storage-after-p1.json").write_text(json.dumps(storage, indent=2) + "\n")
    (root / "audit_startup.py").write_bytes(AUDITOR.read_bytes())
    (root / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(root.iterdir())}
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "w:gz", compresslevel=1) as archive:
        for name in files:
            archive.add(root / name, arcname=name, recursive=False)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        members = archive.getmembers()
        assert len(members) == len(files) and all(m.isfile() for m in members)
        assert {m.name: {"bytes": m.size, "sha256": hashlib.sha256(archive.extractfile(m).read()).hexdigest()} for m in members} == files
    for name, origin in (("report", report), ("audit", audited)):
        target = REPO / f"docs/validation/segment-500k-linux-p1-{name}-v1.json"
        with target.open("xb") as stream:
            stream.write(origin.read_bytes())
        assert sha(target) == sha(origin)
    result = {
        "complete": True, "scope": "Completed five-pair 500k Segmentation P1 only; P3/P4 continue separately.",
        "archive": str(ARCHIVE.relative_to(REPO)), "archive_sha256": sha(ARCHIVE),
        "archive_bytes": ARCHIVE.stat().st_size, "files": files, "file_count": len(files),
        "collector_sha256": sha(Path(__file__)), "local_independent_recalculation_matches": True,
        "collection_note": "Initial collection checks exposed 69 directory entries in the source TAR and the older wheel's missing dependency-license bundle. All 176 regular sources are checked separately; missing license files are reported rather than treated as runtime mismatches. Benchmark/source bytes were unchanged and no measurement was rerun.",
        "wheel_source_audit": wheel_sources,
        "output_sha256": observed["output_sha256"], "summary": observed["summary"],
        "limits": observed["limits"] + [
            "Input files are not embedded; complete fixture/source manifests and the unchanged generator are retained.",
            "Source and wheel identities match launch evidence; this collector does not repeat runtime tests or measurements.",
            "Controller state is a post-P1 checkpoint, not completion evidence for P3/P4.",
        ],
    }
    with RECEIPT.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("archive_sha256", "archive_bytes", "file_count", "summary")}, indent=2))
