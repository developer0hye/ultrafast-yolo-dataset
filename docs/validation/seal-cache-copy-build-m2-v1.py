"""Validate and preserve the manually resumed installation, without rewriting failures."""

import hashlib
import importlib.util
import io
import json
import subprocess
import tarfile
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

ROOT = Path("/Volumes/T7/ultrafast-vision-build")
PREFIX = "dataset-cache-read-copy-m2-v1"
REPO = Path("/Users/yhkwon/Documents/ultrafast-yolo-dataset-cache-read-copy")


def path(suffix):
    return ROOT / (PREFIX + "-" + suffix)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(p):
    return digest(p.read_bytes())


source, target, dist = (path(s) for s in ("source", "target", "dist"))
original = json.loads(path("pipeline.json").read_text())
assert original["complete"] is False and "standalone-install failed" in original["error"]
assert original["source_commit"] == "f120cab2f907e36fdcfd93d0879fe56ec4324017"
assert original["source_archive_sha256"] == sha(path("source.tar"))
assert all(row["returncode"] == 0 for row in original["commands"][:-1])
assert all(sha(Path(row["log"])) == row["log_sha256"] for row in original["commands"])
assert "All installed packages are compatible" in path("pip-check.log").read_text()
suites = ET.parse(path("tests.xml")).getroot().findall("testsuite")
assert suites and all(int(s.attrib[k]) == 0 for s in suites for k in ("errors", "failures", "skipped"))
cases = [c for s in suites for c in s.findall("testcase")]
assert len(cases) == sum(int(s.attrib["tests"]) for s in suites) >= 178
for name, count in (("test_cache_section_buffers", 12), ("test_cache_read_comparison", 9)):
    selected = [c for c in cases if c.attrib["classname"].endswith(name)]
    assert len(selected) == count and all(c.find("skipped") is None for c in selected)

spec = importlib.util.spec_from_file_location("reader", source / "bench/cache_read_comparison.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sources = module.library_sources(source)
with tarfile.open(path("source.tar")) as archive:
    frozen = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
for name, data in frozen.items():
    assert (source / name).read_bytes() == data, name
assert all(digest(frozen[name]) == value for name, value in sources.items())

(wheel,) = dist.glob("*.whl")
(sdist,) = dist.glob("*.tar.gz")
with tarfile.open(sdist) as archive:
    packed = {
        str(Path(m.name).relative_to(Path(m.name).parts[0])): archive.extractfile(m).read()
        for m in archive.getmembers()
        if m.isfile()
    }
for name in sources:
    if name == "Cargo.toml":
        cargo_pack = tomllib.loads(packed[name].decode())
        cargo_original = tomllib.loads(frozen[name].decode())
        assert cargo_pack["package"].pop("readme") == "README.md"
        assert cargo_pack == cargo_original
    else:
        assert packed[name] == frozen[name], name
with ZipFile(wheel) as archive:
    (native,) = [n for n in archive.namelist() if n.startswith("ultrafast_yolo_dataset/_native.") and n.endswith(".so")]
    binary = archive.read(native)
    installed = path("clean/lib/python3.12/site-packages")
    for name in archive.namelist():
        if name.startswith("ultrafast_yolo_dataset/"):
            assert archive.read(name) == (installed / name).read_bytes(), name
assert digest(binary) == sha(target / "release/lib_native.dylib") == sha(target / "release/deps/lib_native.dylib")
(fingerprint,) = (target / "release/.fingerprint").glob("ultrafast-yolo-dataset-*/lib-_native.json")
flags = json.loads(fingerprint.read_text())["rustflags"]
rustc = subprocess.check_output(["/Users/yhkwon/.cargo/bin/rustc", "+stable", "-Vv"], text=True)
cargo = subprocess.check_output(["/Users/yhkwon/.cargo/bin/cargo", "+stable", "-V"], text=True)
rustc_info = json.loads((target / ".rustc_info.json").read_text())
assert any(v.get("stdout") == rustc for v in rustc_info["outputs"].values())
identity = {
    "wheel_sha256": sha(wheel),
    "extension_sha256": digest(binary),
    "library_sources": sources,
    "toolchain": {
        "rustc": rustc,
        "cargo": cargo,
        "target": "aarch64-apple-darwin",
        "profile": "release",
        "rustflags": " ".join(flags),
    },
}
baseline = json.loads((REPO / "docs/validation/cache-reader-baseline-m2-identity-v1.json").read_text())
assert baseline["toolchain"] == identity["toolchain"]
assert {n for n in sources if sources[n] != baseline["library_sources"][n]} == {
    "src/cache.rs",
    "python/ultrafast_yolo_dataset/_cache.py",
}


def deps(file):
    return sorted(s for s in file.read_text().splitlines() if not s.startswith("ultrafast-yolo-dataset"))


assert deps(path("baseline-freeze.txt")) == deps(path("candidate-freeze.txt"))
smoke = json.loads(path("standalone.json").read_text())
assert smoke["extension_sha256"] == identity["extension_sha256"]
assert len(smoke["cases"]) == 4 and all(c["cache_hit"] and c["parity"] for c in smoke["cases"])
assert smoke["content_edit_detected"] and smoke["metadata_weaker_contract_confirmed"]
wheel_check = json.loads((dist / "wheel-validation.json").read_text())
assert wheel_check["wheel_sha256"] == sha(wheel) and wheel_check["record_valid"] and not wheel_check["errors"]
assert wheel_check["notice_files_checked"] == 137
files = {p.name: p.read_bytes() for p in ROOT.glob(PREFIX + "-*.log")}
for p in (
    path("pipeline.json"),
    path("source.tar"),
    path("standalone.json"),
    path("tests.xml"),
    path("baseline-freeze.txt"),
    path("candidate-freeze.txt"),
    path("requirements.txt"),
    wheel,
    sdist,
    dist / "wheel-validation.json",
    Path(__file__),
):
    files[p.name] = p.read_bytes()
files["cargo-fingerprint.json"] = fingerprint.read_bytes()
files["rustc-info.json"] = (target / ".rustc_info.json").read_bytes()
files["build-identity.json"] = (json.dumps(identity, indent=2) + "\n").encode()
receipt = {
    "scope": "M2 installed candidate correctness/build evidence, not a performance or platform-matrix result",
    "complete": True,
    "tests": len(cases),
    "errors": 0,
    "failures": 0,
    "skipped": 0,
    "source_commit": original["source_commit"],
    "build_identity": identity,
    "recovery": "Original offline standalone installation failed on uncached Pillow; the subsequent offline runtime installation failed on uncached contourpy. Online installs retained pinned dependencies; the original failed controller state and all failure logs are preserved unchanged.",
    "limits": [
        "Cargo.toml sdist normalization adds only package.readme=README.md.",
        "Current Cargo version is recorded; retained Cargo fingerprint and cached rustc output bind build settings.",
        "macOS 11.0 wheel tag is not evidence of execution on macOS 11.0.",
        "Source/binary provenance relies on archived build artifacts, not embedded Rust source hashes.",
    ],
    "files": {n: digest(d) for n, d in sorted(files.items())},
}
files["resume-receipt.json"] = (json.dumps(receipt, indent=2) + "\n").encode()
out = path("build-evidence.tar.gz")
assert not out.exists() and not path("build-identity.json").exists()
with tarfile.open(out, "w:gz") as archive:
    for name, data in sorted(files.items()):
        member = tarfile.TarInfo(name)
        member.size, member.mtime = len(data), 0
        archive.addfile(member, io.BytesIO(data))
with tarfile.open(out) as archive:
    assert {m.name: archive.extractfile(m).read() for m in archive.getmembers()} == files
receipt["archive_sha256"] = sha(out)
receipt["archive_members"] = len(files)
receipt["all_members_read_back"] = True
path("build-identity.json").write_bytes(files["build-identity.json"])
path("resume-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"tests": len(cases), "wheel": sha(wheel), "archive": sha(out), "files": len(files)}))
