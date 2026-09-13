"""Seal the final P4 audit, raw workers and exact launch sources after completion.

No corpus bytes or native caches are rewritten. Run only once the independent
final audit exists; the build controller owns final report/audit production.
"""

import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path("/Volumes/T7/ultrafast-vision-build")
MAIN = Path("/Users/yhkwon/Documents/ultrafast-yolo-dataset")
AUDITOR = Path("/Users/yhkwon/Documents/ultrafast-yolo-dataset-startup-audit/bench/audit_startup.py")


def sha(data):
    return hashlib.sha256(data).hexdigest()


report = ROOT / "dataset-500k-p4-complete-v1.json"
audit = ROOT / "dataset-500k-p4-complete-audit-v1.json"
identity = MAIN / "docs/validation/detect-500k-launch-v1.json"
preflight = ROOT / "startup-detect-500k-preflight-v1.json"
fixture = ROOT / "startup-detect-500k-v1/startup.json"
runs = ROOT / "startup-detect-500k-p4-v1.runs"
receipt = json.loads(audit.read_text())
assert receipt["complete_requested_samples"] is True and receipt["completed_workers"] == 10
assert receipt["phase"] == "p4" and receipt["task"] == "detect" and receipt["pairs"] == 500000
assert sha(report.read_bytes()) == receipt["report_sha256"]
assert sha(identity.read_bytes()) == receipt["identity_sha256"]
assert sha(preflight.read_bytes()) == receipt["preflight_sha256"]
assert sha(fixture.read_bytes()) == receipt["fixture_manifest_sha256"]
assert sha(AUDITOR.read_bytes()) == receipt["auditor_sha256"]
files = {
    "report.json": report.read_bytes(),
    "audit.json": audit.read_bytes(),
    "identity.json": identity.read_bytes(),
    "preflight.json": preflight.read_bytes(),
    "fixture.json": fixture.read_bytes(),
    "auditor.py": AUDITOR.read_bytes(),
    "launch.sh": (ROOT / "dataset-detect-500k-v1.sh").read_bytes(),
    "run.log": (ROOT / "startup-detect-500k-p4-v1.log").read_bytes(),
    "host.json": (MAIN / "docs/validation/detect-500k-host-supplement-v1.json").read_bytes(),
}
for name, expected in receipt["raw_sha256"].items():
    raw = (runs / name).read_bytes()
    assert sha(raw) == expected, name
    files["runs/" + name] = raw
    log = (runs / name).with_suffix(".log")
    assert log.is_file(), log
    files["runs/" + log.name] = log.read_bytes()
assert len(receipt["raw_sha256"]) == 12
assert {p.name for p in runs.iterdir()} == {Path(n).name for n in files if n.startswith("runs/")}
for name, expected in json.loads(identity.read_text())["source_sha256"].items():
    data = (MAIN / name).read_bytes()
    assert sha(data) == expected, name
    files["sources/" + name] = data
assert (
    receipt["output_sha256"]
    == json.loads((ROOT / "dataset-500k-p3-complete-audit-v1.json").read_text())["output_sha256"]
)
files["manifest.json"] = (json.dumps({name: sha(data) for name, data in files.items()}, indent=2) + "\n").encode()
out = ROOT / "dataset-500k-p4-complete-evidence-v1.tar.gz"
assert not out.exists(), "retain earlier archives"
with tarfile.open(out, "w:gz") as archive:
    for name, data in sorted(files.items()):
        member = tarfile.TarInfo(name)
        member.size, member.mtime = len(data), 0
        archive.addfile(member, io.BytesIO(data))
with tarfile.open(out) as archive:
    actual = {m.name: archive.extractfile(m).read() for m in archive.getmembers()}
assert actual == files
preserved = {
    "scope": "Complete 500k Detection P4 artifact preservation; native cache and fixture remain external",
    "archive_sha256": sha(out.read_bytes()),
    "files": {name: sha(data) for name, data in files.items()},
    "bytes": out.stat().st_size,
    "complete_workers": 10,
    "primers": 2,
    "output_sha256": receipt["output_sha256"],
    "all_archive_members_read_back": True,
}
with (ROOT / "dataset-500k-p4-preservation-v1.json").open("x") as stream:
    stream.write(json.dumps(preserved, indent=2) + "\n")
print(json.dumps({"archive_sha256": preserved["archive_sha256"], "files": len(files), "bytes": preserved["bytes"]}))
