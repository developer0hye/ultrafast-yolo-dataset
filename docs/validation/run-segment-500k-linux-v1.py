"""Full distinct-file Segmentation preparation, capacity check and P1/P3/P4 series.

Run on the now-idle Linux benchmark host. The tested baseline wheel is retained;
this is not a measurement of the M2-only reader or snapshot candidates.
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from zipfile import ZipFile

import psutil
import ultrafast_yolo_dataset as native
from ultralytics.utils import NUM_THREADS

ROOT = Path("/home/yonghye/ultrafast-vision-build")
SOURCE = Path("/home/yonghye/ultrafast-yolo-dataset-segment500k-v1")
CORPUS = ROOT / "startup-segment-500k-linux-v1"
PREFIX = "dataset-segment-500k-linux-v1"
STATE = ROOT / (PREFIX + "-state.json")
IDENTITY = ROOT / (PREFIX + "-identity.json")
PREFLIGHT = ROOT / (PREFIX + "-preflight.json")
WHEEL = ROOT / "cuda-wheels-byte-access-v2/dataset/ultrafast_yolo_dataset-0.1.0a1-cp312-cp312-linux_x86_64.whl"
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", YOLO_OFFLINE="true")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert not STATE.exists() and not CORPUS.exists() and not IDENTITY.exists()
assert not Path("/proc/1483383").exists(), "the mask follow-up must have terminated"
assert json.loads((ROOT / "mask-unit-scale-linux-loader-followup-v1.json").read_text())["complete"]
assert json.loads((ROOT / "mask-unit-scale-linux-loader-audit-negative-v1.json").read_text())["rejected"] == 44
selection = json.loads((ROOT / (PREFIX + "-source.json")).read_text())
assert all(sha(SOURCE / n) == h for n, h in selection["selected_files"].items())
assert sha(WHEEL) == "b40a103f4234c9853150af699a20cd6eb38bce1bdf8357d2bb79cd90119c7f0e"
extension = Path(native._native.__file__)
assert sha(extension) == "274dc3ae164435e9fa5f588aec3d184a4faffc50a38d6bf8b50273c86f78f7b1"
package = Path(native.__file__).parent
with ZipFile(WHEEL) as zipped:
    assert zipped.read("ultrafast_yolo_dataset/" + extension.name) == extension.read_bytes()
    for path in (SOURCE / "python/ultrafast_yolo_dataset").iterdir():
        if path.is_file() and path.suffix in (".py", ".json"):
            assert path.read_bytes() == (package / path.name).read_bytes() == zipped.read("ultrafast_yolo_dataset/" + path.name)
profile = hashlib.sha256()
for name in ("src/parser.rs", "src/snapshot.rs", "src/provenance.rs", "src/materialize.rs",
             "src/cache.rs", "src/lib.rs", "Cargo.toml", "Cargo.lock"):
    profile.update((SOURCE / name).read_bytes())
assert profile.hexdigest() == native._native.native_cache_profile() == "4d82cb5159b4d31eda4074a1a0cca679c7b92977c6ea2f9c685550e045ff2e7e"
assert psutil.virtual_memory().available > 12 * 1024**3 and shutil.disk_usage(ROOT).free > 25 * 1024**3
sys.path.insert(0, str(SOURCE / "bench"))
from fixture_files import source_hashes

identity = {
    "scope": "Full 500k Segmentation baseline experiment launch; not completion evidence",
    "recorded_unix_ns": time.time_ns(), "source_commit": selection["commit"],
    "source_archive_sha256": selection["archive_sha256"], "source_sha256": source_hashes(SOURCE),
    "native_extension_sha256": sha(extension), "wheel_sha256": sha(WHEEL), "native_cache_profile": profile.hexdigest(),
    "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
    "cpu": next(s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines() if s.startswith("model name")),
    "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
    "ram_bytes": psutil.virtual_memory().total, "available_ram_bytes": psutil.virtual_memory().available,
    "storage_free_bytes": shutil.disk_usage(ROOT).free, "loadavg": os.getloadavg(),
    "packages": sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()),
    "task": "segment", "pairs": 500000, "expected_regular_files": 1000000,
    "rounds": 5, "p1_workers": 4, "startup_workers": NUM_THREADS, "loader_workers": 0,
    "script_sha256": sha(Path(__file__)), "constraint": "One build/test/benchmark job per host; preserve source/runtime/inputs",
}
IDENTITY.write_text(json.dumps(identity, indent=2) + "\n")
state = {"complete": False, "phase": "preparation", "commands": [], "identity_sha256": sha(IDENTITY)}


def save():
    STATE.write_text(json.dumps(state, indent=2) + "\n")


def run(command, name):
    log = ROOT / (PREFIX + "-" + name + ".log")
    command = list(map(str, command))
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=SOURCE, env=ENV, stdout=stream, stderr=subprocess.STDOUT, check=False)
    state["commands"].append({"command": command, "cwd": str(SOURCE), "returncode": result.returncode,
                              "log": str(log), "log_sha256": sha(log)})
    save()
    assert result.returncode == 0, f"{name} failed; preserve artifacts and inspect before retry"
    print(name + " completed", flush=True)


save()
try:
    run([sys.executable, "bench/startup.py", "--prepare", CORPUS, "--count", "500000", "--task", "segment"], "prepare")
    fixture = CORPUS / "startup.json"
    manifest = json.loads(fixture.read_text())
    assert manifest["count"] == 500000 and manifest["labels"]["bytes"] == 1584173000
    run([sys.executable, "bench/fixture_files.py", "--corpus", CORPUS, "--pairs", "500000", "--out", PREFLIGHT], "preflight")
    preflight = json.loads(PREFLIGHT.read_text())
    assert preflight["all_regular_single_link_files"] and preflight["counts"] == {"images": 500000, "labels": 500000}
    state["phase"] = "full-capacity-qualification"
    save()
    capacity = []
    for backend in ("reference", "native-content"):
        output = ROOT / (PREFIX + "-capacity-" + backend + ".json")
        run([sys.executable, "bench/cache_startup.py", "--corpus", CORPUS, "--worker", backend,
             "--mode", "miss", "--result", output], "capacity-" + backend)
        capacity.append(json.loads(output.read_text()))
    assert capacity[0]["output_sha256"] == capacity[1]["output_sha256"]
    assert capacity[1]["native_fallback_count"] == 0
    assert all(v["scan_summary"]["found"] == 500000 and v["scan_summary"]["corrupt"] == 0 for v in capacity)
    (cache,) = (CORPUS / "benchmark-caches/native-content").glob("*.uydcache")
    assert cache.stat().st_size <= 2 * 1024**3
    with cache.open("rb") as stream:
        header = stream.read(16)
        count = struct.unpack("<I", header[12:16])[0]
        sections = [struct.unpack("<Q", row[:8])[0] for row in (stream.read(40) for _ in range(count))]
    assert count == 11 and sections[0] <= 256 * 1024**2
    state["capacity"] = {"output_sha256": capacity[0]["output_sha256"], "native_cache_bytes": cache.stat().st_size,
                         "section_bytes": sections, "reference_peak_rss": capacity[0]["peak_rss_constructor_bytes"],
                         "native_peak_rss": capacity[1]["peak_rss_constructor_bytes"]}
    save()
    for phase, mode, workers in (("p1", None, 4), ("p3", "miss", NUM_THREADS), ("p4", "hit", NUM_THREADS)):
        state["phase"] = phase
        save()
        report = ROOT / (PREFIX + "-" + phase + ".json")
        if phase == "p1":
            command = [sys.executable, "bench/label_engine.py", "--corpus", CORPUS / "labels", "--workers", 4,
                       "--rounds", 5, "--out", report]
        else:
            backends = ("reference", "native-content") if mode == "miss" else ("reference-content", "native-content")
            command = [sys.executable, "bench/cache_startup.py", "--corpus", CORPUS, "--mode", mode,
                       "--backends", *backends, "--rounds", 5, "--out", report]
        run(command, phase)
        audit = ROOT / (PREFIX + "-" + phase + "-audit.json")
        command = [sys.executable, "bench/audit_startup.py", "--report", report, "--identity", IDENTITY,
                   "--preflight", PREFLIGHT, "--fixture-manifest", fixture, "--phase", phase, "--task", "segment",
                   "--pairs", 500000, "--repeats", 5, "--workers", workers, "--out", audit]
        if phase != "p1":
            command += ["--runs-dir", report.with_suffix(".runs")]
        run(command, phase + "-audit")
        audited = json.loads(audit.read_text())
        assert audited["complete_requested_samples"] and audited["completed_workers"] == 10
        if phase != "p1":
            assert audited["output_sha256"] == state["capacity"]["output_sha256"]
        assert source_hashes(SOURCE) == identity["source_sha256"]
    state.update(complete=True, phase="all-three-phases-audited")
    save()
    print("Full 500k Segmentation capacity, P1/P3/P4 and independent audits completed.")
except BaseException as error:
    state["error"] = f"{type(error).__name__}: {error}"
    save()
    raise
