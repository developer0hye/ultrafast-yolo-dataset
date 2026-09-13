"""Seal the complete M2 native-versus-native startup evidence and read it back."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path("/Volumes/T7/ultrafast-vision-build")
REPO = Path("/Users/yhkwon/Documents/ultrafast-yolo-dataset-snapshot-scratch")
PREFIX = "snapshot-startup-full-m2-v1"
QUAL = "snapshot-startup-full-m2-v2-qualification"


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def main():
    archive_path = ROOT / (PREFIX + "-evidence.tar.gz")
    receipt_path = ROOT / (PREFIX + "-preservation.json")
    assert not archive_path.exists() and not receipt_path.exists()
    stage = ROOT / (QUAL + "-stage")
    qualification = json.loads((ROOT / (QUAL + ".json")).read_text())
    audit = json.loads((ROOT / (QUAL + "-audit.json")).read_text())
    negative = json.loads((ROOT / (QUAL + "-negative.json")).read_text())
    launch = json.loads((ROOT / (PREFIX + "-launch.json")).read_text())
    report = json.loads((ROOT / (PREFIX + ".json")).read_text())
    args = json.loads((stage / "full-arguments.json").read_text())
    assert qualification["complete"] and qualification["passed"] and qualification["formatting_preserved_ast"]
    assert audit["complete"] and audit["measured_workers"] == 20 and audit["separate_primers"] == 2
    assert audit["pairs"] == 500000 and audit["rounds"] == 5 and audit["workers"] == 7
    assert launch["complete"] and launch["returncode"] == 0 and launch["outputs_match_original_full_reference"]
    assert report["complete"] and len(report["records"]) == 20 and len(report["primers"]) == 2
    assert negative["complete"] and len(negative["controls"]) == 3 and negative["negative_case_count"] == 35
    assert len(negative["negative_cases"]) == len({c["case"] for c in negative["negative_cases"]}) == 35
    assert all(c["rejected"] for c in negative["negative_cases"])
    files = {}

    def add(name, path, expected=None):
        path = Path(path)
        assert name not in files and path.is_file() and not path.is_symlink()
        digest = sha(path)
        assert expected is None or digest == expected, name
        files[name] = {"path": path, "bytes": path.stat().st_size, "sha256": digest}

    assert qualification["report_sha256"] == launch["report_sha256"] == audit["report_sha256"]
    add("report.json", args["report"], audit["report_sha256"])
    add("launch.json", ROOT / (PREFIX + "-launch.json"), qualification["launch_sha256"])
    add("controller.log", ROOT / (PREFIX + ".log"))
    add("audit/qualification.json", ROOT / (QUAL + ".json"))
    add("audit/result.json", ROOT / (QUAL + "-audit.json"), qualification["audit_sha256"])
    add("audit/negative.json", ROOT / (QUAL + "-negative.json"), qualification["negative_sha256"])
    assert [c["label"] for c in qualification["commands"]] == ["format", "lint", "format-check", "negative", "audit"]
    for command in qualification["commands"]:
        assert command["returncode"] == 0
        add(
            "audit/logs/" + command["label"] + ".log",
            ROOT / (QUAL + "-" + command["label"] + ".log"),
            command["log_sha256"],
        )
    for name, digest in qualification["source_after_sha256"].items():
        add("audit/stage/" + name, stage / name, digest)
    assert (
        negative["auditor_sha256"]
        == audit["auditor_sha256"]
        == files["audit/stage/audit-snapshot-startup-v1.py"]["sha256"]
    )
    assert negative["qualifier_sha256"] == files["audit/stage/qualify-snapshot-startup-audit-v1.py"]["sha256"]
    assert negative["arguments_sha256"] == files["audit/stage/pilot-arguments.json"]["sha256"]
    add("audit/watcher.py", ROOT / "qualify-snapshot-after-startup-v2.py", qualification["script_sha256"])
    add("audit/dependency.json", ROOT / (QUAL + "-dependency.json"), qualification["dependency_sha256"])
    add("audit/launch-identity.json", ROOT / (QUAL + "-launch-identity.json"))
    for name in (
        "qualify-snapshot-after-startup-v1.py",
        "snapshot-startup-full-m2-v1-qualification.json",
        "snapshot-startup-full-m2-v1-qualification-cancelled.json",
    ):
        add("audit/prior-idle-watcher/" + name, ROOT / name)
    for name, digest in qualification["source_before_sha256"].items():
        add("audit/before-format/" + name, ROOT / "snapshot-startup-full-m2-v1-qualification-stage" / name, digest)
    runs = Path(args["runs_dir"])
    assert {p.name for p in runs.iterdir()} == set(audit["raw_sha256"])
    for name, digest in audit["raw_sha256"].items():
        add("runs/" + name, runs / name, digest)
    for name, digest in audit["anchors"].items():
        add("anchors/" + name + (".py" if name == "wrapper" else ".json"), args[name], digest)
    for name, digest in report["harness_sources"].items():
        add("harness/" + name, Path(args["harness_root"]) / name, digest)
    for version in ("baseline", "candidate"):
        anchor = report["anchors"][version]
        add(
            "wheels/" + version + "/" + Path(args[version + "_wheel"]).name,
            args[version + "_wheel"],
            anchor["wheel_sha256"],
        )
        for name, digest in anchor["library_sources"].items():
            add("sources/" + version + "/" + name, Path(args[version + "_source"]) / name, digest)
        caches = list((Path(args["caches_dir"]) / version).glob("*.uydcache"))
        assert len(caches) == 1
        add("caches/" + version + ".uydcache", caches[0], audit["cache_checks"][version]["sha256"])
    add(
        "measurement-controller.py",
        REPO / "docs/validation/launch-snapshot-startup-full-m2-v1.py",
        launch["controller_sha256"],
    )
    add(
        "candidate-build-receipt.json",
        ROOT / "dataset-snapshot-scratch-m2-v1-build-receipt.json",
        launch["build_receipt_sha256"],
    )
    add(
        "pilot/evidence.tar.gz",
        REPO / "bench/results/snapshot-startup-pilot-m2-v2.tar.gz",
        "6674d37983be1cd1fe6ae4a3c594726ffa2b4eeec49b811fe3b813f2e815f684",
    )
    add("pilot/preservation.json", REPO / "docs/validation/snapshot-startup-pilot-m2-v2-preservation.json")
    add("sealer.py", Path(__file__))
    manifest = {name: {k: value[k] for k in ("bytes", "sha256")} for name, value in files.items()}
    manifest_data = (json.dumps(manifest, indent=2) + "\n").encode()
    with archive_path.open("xb") as output, tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, value in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = value["bytes"]
            with value["path"].open("rb") as stream:
                archive.addfile(info, stream)
        info = tarfile.TarInfo("manifest.json")
        info.size = len(manifest_data)
        archive.addfile(info, io.BytesIO(manifest_data))
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        assert len(members) == len(files) + 1
        assert {member.name for member in members} == set(files) | {"manifest.json"}
        assert archive.extractfile("manifest.json").read() == manifest_data
        for name, expected in manifest.items():
            digest, size = hashlib.sha256(), 0
            with archive.extractfile(name) as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
                    size += len(block)
            assert size == expected["bytes"] and digest.hexdigest() == expected["sha256"], name
    receipt = {
        "preservation_complete": True,
        "all_members_read_back": True,
        "archive_sha256": sha(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "file_count": len(files) + 1,
        "manifest": manifest,
        "scope": "20 actual measured startup processes, two primers, full retained caches, exact source/wheels and qualified artifact audit. Native-versus-native increment; no production or training qualification.",
    }
    with receipt_path.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "manifest"}))


if __name__ == "__main__":
    main()
