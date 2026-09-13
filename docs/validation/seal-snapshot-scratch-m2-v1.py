"""Bind compiled snapshot candidate to its selected source archive and installed tests."""

import hashlib
import importlib.util
import io
import json
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import tomllib

ROOT = Path("/Volumes/T7/ultrafast-vision-build")
REPO = Path(__file__).parents[2]
PREFIX = "dataset-snapshot-scratch-m2-v1"


def path(name):
    return ROOT / (PREFIX + "-" + name)


def sha(data):
    return hashlib.sha256(data).hexdigest()


state = json.loads(path("state.json").read_text())
assert state["complete"] and state["tests"] == 193 and "error" not in state
assert all(c["returncode"] == 0 and sha(Path(c["log"]).read_bytes()) == c["log_sha256"] for c in state["commands"])
source, target, dist = (path(n) for n in ("source", "target", "dist"))
assert all(sha((source / n).read_bytes()) == h for n, h in state["source_selection"].items())
assert sha(path("source.tar").read_bytes()) == state["source_archive_sha256"]
with tarfile.open(path("source.tar")) as archive:
    archived = {m.name: sha(archive.extractfile(m).read()) for m in archive.getmembers() if m.isfile()}
assert archived == state["source_selection"]
spec = importlib.util.spec_from_file_location("reader", source / "bench/cache_read_comparison.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
sources = reader.library_sources(source)
(wheel,) = dist.glob("*.whl")
(sdist,) = dist.glob("*.tar.gz")
with tarfile.open(sdist) as archive:
    files = {
        str(Path(m.name).relative_to(Path(m.name).parts[0])): archive.extractfile(m).read()
        for m in archive.getmembers()
        if m.isfile()
    }
for name in sources:
    if name == "Cargo.toml":
        packaged, original = tomllib.loads(files[name].decode()), tomllib.loads((source / name).read_text())
        assert packaged["package"].pop("readme") == "README.md"
        assert packaged == original
    else:
        assert sha(files[name]) == sources[name]
with ZipFile(wheel) as archive:
    (native,) = [n for n in archive.namelist() if n.startswith("ultrafast_yolo_dataset/_native.") and n.endswith(".so")]
    extension = archive.read(native)
    for name in archive.namelist():
        if name.startswith("ultrafast_yolo_dataset/"):
            assert archive.read(name) == (path("clean/lib/python3.12/site-packages") / name).read_bytes()
assert sha(extension) == sha((target / "release/lib_native.dylib").read_bytes())
assert sha(extension) == sha((target / "release/deps/lib_native.dylib").read_bytes())
(fingerprint,) = (target / "release/.fingerprint").glob("ultrafast-yolo-dataset-*/lib-_native.json")
rustc_info = json.loads((target / ".rustc_info.json").read_text())
assert any(x.get("stdout") == state["rustc"] for x in rustc_info["outputs"].values())
identity = {
    "wheel_sha256": sha(wheel.read_bytes()),
    "extension_sha256": sha(extension),
    "library_sources": sources,
    "toolchain": {
        "rustc": state["rustc"],
        "cargo": state["cargo"],
        "target": "aarch64-apple-darwin",
        "profile": "release",
        "rustflags": " ".join(json.loads(fingerprint.read_text())["rustflags"]),
    },
}
baseline = json.loads((REPO / "docs/validation/cache-copy-m2-build-identity-v1.json").read_text())
assert identity["toolchain"] == baseline["toolchain"]
assert set(sources) == set(baseline["library_sources"])
assert {n for n in sources if sources[n] != baseline["library_sources"][n]} == {"src/snapshot.rs"}
assert "5 passed; 0 failed" in path("rust-tests.log").read_text()
suites = ET.parse(path("tests.xml")).getroot().findall("testsuite")
assert sum(int(s.attrib["tests"]) for s in suites) == 193
assert all(int(s.attrib[k]) == 0 for s in suites for k in ("errors", "failures", "skipped"))


def dependencies(p):
    return sorted(x for x in p.read_text().splitlines() if not x.startswith("ultrafast-yolo-dataset"))


assert dependencies(path("freeze.log")) == dependencies(ROOT / "dataset-cache-read-copy-m2-v1-candidate-freeze.txt")
smoke = json.loads(path("standalone.json").read_text())
assert smoke["extension_sha256"] == identity["extension_sha256"] and len(smoke["cases"]) == 4
assert all(c["parity"] and c["cache_hit"] for c in smoke["cases"])
assert smoke["content_edit_detected"] and smoke["metadata_weaker_contract_confirmed"]
audit = json.loads((dist / "wheel-validation.json").read_text())
assert audit["record_valid"] and not audit["errors"] and audit["notice_files_checked"] == 137
assert audit["wheel_sha256"] == identity["wheel_sha256"]
files = {p.name: p.read_bytes() for p in ROOT.glob(PREFIX + "-*.log")}
for p in (
    path("state.json"),
    path("source.tar"),
    path("tests.xml"),
    path("standalone.json"),
    wheel,
    sdist,
    dist / "wheel-validation.json",
    Path(__file__),
    REPO / "docs/validation/build-snapshot-scratch-m2-v1.py",
):
    files[p.name] = p.read_bytes()
files["cargo-fingerprint.json"] = fingerprint.read_bytes()
files["rustc-info.json"] = (target / ".rustc_info.json").read_bytes()
files["identity.json"] = (json.dumps(identity, indent=2) + "\n").encode()
manifest = {n: sha(d) for n, d in files.items()}
files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
out = path("build-evidence.tar.gz")
assert not out.exists() and not path("identity.json").exists()
with tarfile.open(out, "w:gz") as archive:
    for name, data in sorted(files.items()):
        entry = tarfile.TarInfo(name)
        entry.size, entry.mtime = len(data), 0
        archive.addfile(entry, io.BytesIO(data))
with tarfile.open(out) as archive:
    assert {m.name: archive.extractfile(m).read() for m in archive.getmembers()} == files
receipt = {
    "complete": True,
    "tests": 193,
    "rust_tests": 5,
    "skipped": 0,
    "source_commit": state["source_commit"],
    "archive_sha256": sha(out.read_bytes()),
    "archive_members": len(files),
    "archive_bytes": out.stat().st_size,
    "all_members_read_back": True,
    "identity": identity,
    "manifest": manifest,
    "limits": [
        state["source_scope"],
        "Sdist Cargo.toml adds only package.readme=README.md.",
        "M2 correctness/build evidence, not a speedup, Linux or platform-matrix result.",
        "Native source provenance relies on retained build artifacts, not embedded Rust source hashes.",
    ],
}
path("identity.json").write_bytes(files["identity.json"])
path("build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(
    json.dumps({k: receipt[k] for k in ("tests", "rust_tests", "archive_sha256", "archive_members", "archive_bytes")})
)
