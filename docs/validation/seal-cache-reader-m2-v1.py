"""Preserve audited raw measurements, reproducible input containers and both wheels."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path("/Volumes/T7/ultrafast-vision-build")
REPO = Path(__file__).parents[2]
SOURCE = ROOT / "dataset-cache-read-copy-m2-v1-source"
FIXTURE = ROOT / "cache-reader-seven-layouts-m2-v1"
OUT = ROOT / "cache-reader-paired-m2-v1-evidence.tar.gz"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1048576):
            digest.update(block)
    return digest.hexdigest()


assert not OUT.exists()
audit = json.loads((ROOT / "cache-reader-paired-m2-v1-audit.json").read_text())
assert audit["complete"] and audit["measured_workers"] == 70 and audit["all_fixture_files_rehashed_after_campaign"]
assert sha(REPO / "docs/validation/audit-cache-reader-m2-v1.py") == audit["auditor_sha256"]
negative = json.loads((ROOT / "cache-reader-paired-m2-v1-audit-negative.json").read_text())
assert negative["valid_control_passed"] and negative["rejected"] == 17
assert negative["auditor_sha256"] == audit["auditor_sha256"]
files = {}
for suffix in (
    ".json",
    ".log",
    "-audit.json",
    "-audit.log",
    "-audit-negative.json",
    "-audit-negative.log",
    "-launch.json",
):
    path = ROOT / ("cache-reader-paired-m2-v1" + suffix)
    files["reports/" + path.name] = path
for directory, prefix in (
    (ROOT / "cache-reader-paired-m2-v1.runs", "runs"),
    (ROOT / "cache-reader-qualification-m2-v1", "qualification"),
    (FIXTURE, "fixture"),
):
    for path in directory.iterdir():
        assert path.is_file()
        files[prefix + "/" + path.name] = path
assert len([n for n in files if n.startswith("runs/")]) == 144
for name in (
    "run-cache-reader-m2-v1.py",
    "audit-cache-reader-m2-v1.py",
    "qualify-cache-reader-audit-m2-v1.py",
    "seal-cache-reader-m2-v1.py",
):
    files["tools/" + name] = REPO / "docs/validation" / name
files["tools/cache_read_comparison.py"] = SOURCE / "bench/cache_read_comparison.py"
files["reports/prepare.log"] = ROOT / "cache-reader-seven-layouts-m2-v1-prepare.log"
files["reports/controller.log"] = ROOT / "cache-reader-qualify-and-run-m2-v1.log"
for version, directory in (
    ("baseline", ROOT / "dataset-notices-dist-v1"),
    ("candidate", ROOT / "dataset-cache-read-copy-m2-v1-dist"),
):
    for path in directory.iterdir():
        if path.suffix == ".whl" or path.name.endswith(".tar.gz"):
            files[version + "/" + path.name] = path
for name in ("identity", "provenance", "cargo-fingerprint", "rustc-info"):
    path = REPO / "docs/validation" / ("cache-reader-baseline-m2-" + name + "-v1.json")
    files["baseline/" + path.name] = path
files["candidate/build-evidence.tar.gz"] = REPO / "bench/results/cache-copy-m2-build-evidence-v1.tar.gz"
files["candidate/build-identity.json"] = REPO / "docs/validation/cache-copy-m2-build-identity-v1.json"
files["candidate/resume-receipt.json"] = REPO / "docs/validation/cache-copy-m2-resume-v1.json"
baseline = json.loads((REPO / "docs/validation/cache-reader-baseline-m2-identity-v1.json").read_text())
for name, expected in baseline["library_sources"].items():
    path = Path("/Users/yhkwon/Documents/ultrafast-yolo-dataset") / name
    assert sha(path) == expected
    files["baseline/source/" + name] = path
manifest = {name: {"sha256": sha(path), "bytes": path.stat().st_size} for name, path in sorted(files.items())}
for name, expected in audit["raw_sha256"].items():
    assert manifest["runs/" + name]["sha256"] == expected
for case in json.loads((FIXTURE / "manifest.json").read_text())["cases"]:
    assert manifest["fixture/" + case["file"]] == {"sha256": case["sha256"], "bytes": case["bytes"]}
manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
with tarfile.open(OUT, "w:gz") as archive:
    for name, path in sorted(files.items()):
        entry = tarfile.TarInfo(name)
        entry.size, entry.mtime = manifest[name]["bytes"], 0
        with path.open("rb") as stream:
            archive.addfile(entry, stream)
    entry = tarfile.TarInfo("manifest.json")
    entry.size = len(manifest_bytes)
    archive.addfile(entry, io.BytesIO(manifest_bytes))
with tarfile.open(OUT) as archive:
    assert {m.name for m in archive.getmembers()} == set(files) | {"manifest.json"}
    for member in archive.getmembers():
        with archive.extractfile(member) as stream:
            if member.name == "manifest.json":
                assert stream.read() == manifest_bytes
                continue
            digest = hashlib.sha256()
            while block := stream.read(1048576):
                digest.update(block)
        assert digest.hexdigest() == manifest[member.name]["sha256"]
receipt = {
    "archive_sha256": sha(OUT),
    "archive_bytes": OUT.stat().st_size,
    "archive_files": len(files) + 1,
    "all_members_read_back": True,
    "manifest": manifest,
    "scope": "Complete seven-layout native-reader campaign: all inputs including actual captured 500k cache, 70 measured workers, two descriptors, qualification, audited statistics, both wheels/sdists and retained build provenance",
}
path = ROOT / "cache-reader-paired-m2-v1-preservation.json"
with path.open("x") as stream:
    stream.write(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k != "manifest"}))
