"""Separate paired container-reader experiment; not full cache-hit startup.

Prepare fixtures only on an idle host. Six frozen synthetic layouts plus a captured real native cache cover read
chunk boundaries, section layout and an actual completed startup workload.
Each measurement is a fresh process; fixture generation is never timed.
Build identities must come from the corresponding preserved build/test run.
"""

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import resource
import shutil
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path
from zipfile import ZipFile

MIB = 1024**2
LAYOUTS = [(0,), (4 * MIB - 1,), (4 * MIB + 1,), (64 * MIB,), (256 * MIB,), (64 * MIB,) * 4]
SAMPLES = 5
REPEATS = 5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(MIB):
            h.update(block)
    return h.hexdigest()


def encoded(value):
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def inspect_container(path):
    """Independent streaming format/digest check, without the native reader."""
    sections = []
    with path.open("rb") as stream:
        header = stream.read(16)
        require(len(header) == 16 and header[:8] == b"UYDCACHE", "wrong container magic/header")
        version, count = struct.unpack("<II", header[8:])
        require(version == 1 and 1 <= count <= 16, "wrong container version/count")
        descriptors = []
        for _ in range(count):
            row = stream.read(40)
            require(len(row) == 40, "truncated descriptor")
            length = struct.unpack("<Q", row[:8])[0]
            descriptors.append((length, row[8:].hex()))
        require(descriptors[0][0] <= 256 * MIB, "metadata too large")
        for length, expected in descriptors:
            left, h = length, hashlib.sha256()
            while left:
                block = stream.read(min(left, MIB))
                require(block, "truncated section")
                h.update(block)
                left -= len(block)
            require(h.hexdigest() == expected, "section checksum mismatch")
            sections.append({"bytes": length, "sha256": expected})
        require(stream.read(1) == b"", "trailing container bytes")
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": file_hash(path), "sections": sections}


def prepare(root, captured_cache):
    require(captured_cache is not None and captured_cache.is_file(), "capture the completed native startup cache")
    root.mkdir(parents=True, exist_ok=False)
    block = bytes(range(256)) * (MIB // 256)
    cases = []
    for index, layout in enumerate(LAYOUTS):
        path = root / f"case-{index}.uydcache"
        descriptions = [{"bytes": 2, "sha256": sha(b"{}")}]
        for length in layout:
            h = hashlib.sha256()
            for start in range(0, length, MIB):
                h.update(memoryview(block)[: min(MIB, length - start)])
            descriptions.append({"bytes": length, "sha256": h.hexdigest()})
        with path.open("xb") as stream:
            stream.write(b"UYDCACHE" + struct.pack("<II", 1, len(descriptions)))
            for section in descriptions:
                stream.write(struct.pack("<Q", section["bytes"]) + bytes.fromhex(section["sha256"]))
            stream.write(b"{}")
            for length in layout:
                for start in range(0, length, MIB):
                    stream.write(memoryview(block)[: min(MIB, length - start)])
        actual = inspect_container(path)
        require(actual["sections"] == descriptions, "prepared fixture differs from expected sections")
        cases.append(actual)
    # Capture only after the startup experiment terminates. Never point workers
    # at a cache that another process can replace during the paired matrix.
    cache_hash = file_hash(captured_cache)
    captured = root / "case-6.uydcache"
    with captured_cache.open("rb") as src, captured.open("xb") as dst:
        shutil.copyfileobj(src, dst, MIB)
    observed = inspect_container(captured)
    require(observed["sha256"] == cache_hash == file_hash(captured_cache), "source cache changed while capturing")
    cases.append(observed)
    manifest = {
        "kind": "cache-reader-seven-layouts-v1",
        "captured_cache": {"source_path": str(captured_cache.resolve()), "sha256": cache_hash},
        "cases": cases,
        "generator_sha256": file_hash(Path(__file__)),
        "scope": "Six synthetic containers and a copied completed startup cache; no JSON decoding or dataset input validation.",
    }
    (root / "manifest.json").write_text(encoded(manifest))
    return manifest


def library_sources(root):
    names = [root / name for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml")]
    names.extend(sorted((root / "src").rglob("*.rs")))
    package = root / "python/ultrafast_yolo_dataset"
    names.extend(sorted(package.glob("*.py")))
    names.extend(sorted(package.glob("*.json")))
    return {str(p.relative_to(root)): file_hash(p) for p in names}


def descriptor(root, wheel, identity):
    import psutil
    import ultrafast_yolo_dataset as native

    anchor = json.loads(identity.read_text())
    extension, package = Path(native._native.__file__), Path(native.__file__).parent
    require(package.resolve().is_relative_to(Path(sys.prefix).resolve()), "requires installed wheel")
    require(file_hash(wheel) == anchor["wheel_sha256"], "wrong wheel")
    require(file_hash(extension) == anchor["extension_sha256"], "wrong extension")
    require(library_sources(root) == anchor["library_sources"], "wrong declared build source snapshot")
    require(
        set(anchor["toolchain"]) == {"rustc", "cargo", "target", "profile", "rustflags"}
        and all(anchor["toolchain"][key] for key in ("rustc", "cargo", "target", "profile"))
        and isinstance(anchor["toolchain"]["rustflags"], str),
        "record exact build toolchain",
    )
    require(anchor["toolchain"]["profile"] == "release", "release wheel required")
    with ZipFile(wheel) as archive:
        require(
            archive.read("ultrafast_yolo_dataset/" + extension.name) == extension.read_bytes(),
            "installed extension differs from wheel",
        )
        for name in archive.namelist():
            if (
                name.startswith("ultrafast_yolo_dataset/")
                and name.endswith((".py", ".json"))
                and "/licenses/" not in name
            ):
                relative = Path(name).relative_to("ultrafast_yolo_dataset")
                data = archive.read(name)
                require(
                    data == (package / relative).read_bytes() == (root / "python" / name).read_bytes(),
                    "installed Python source mismatch",
                )
    packages = sorted(
        (d.metadata["Name"].lower().replace("_", "-"), d.version)
        for d in importlib.metadata.distributions()
        if d.metadata["Name"].lower().replace("_", "-") != "ultrafast-yolo-dataset"
    )
    return {
        "wheel_sha256": anchor["wheel_sha256"],
        "extension_sha256": anchor["extension_sha256"],
        "library_sources": anchor["library_sources"],
        "toolchain": anchor["toolchain"],
        "build_identity_sha256": file_hash(identity),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": (
            subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            if sys.platform == "darwin"
            else next(
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            )
        ),
        "physical_cpus": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "packages": packages,
        "harness_sha256": file_hash(Path(__file__)),
    }


def worker(args):
    import psutil
    from ultrafast_yolo_dataset import _native

    identity = descriptor(args.root, args.wheel, args.identity)
    if args.worker == "describe":
        return {"descriptor": identity}
    manifest = json.loads((args.fixture / "manifest.json").read_text())
    require(manifest["kind"] == "cache-reader-seven-layouts-v1" and len(manifest["cases"]) == 7, "wrong fixture")
    case = manifest["cases"][args.case]
    require(case["file"] == f"case-{args.case}.uydcache", "unexpected fixture path")
    if args.case < 6:
        require([s["bytes"] for s in case["sections"]] == [2, *LAYOUTS[args.case]], "wrong frozen layout")
    else:
        require(case["sha256"] == manifest["captured_cache"]["sha256"], "wrong captured startup cache")
    path = args.fixture / case["file"]
    require(inspect_container(path) == case, "fixture content mismatch before timing")

    def read():
        return _native.read_cache_sections(str(path), case["bytes"])

    def verify(value):
        require(type(value) is list and all(type(v) is bytes for v in value), "wrong return ownership/type")
        require([{"bytes": len(v), "sha256": sha(v)} for v in value] == case["sections"], "wrong section output")

    output = read()
    verify(output)
    del output
    for _ in range(2):
        output = read()
        del output
    rss_before = psutil.Process().memory_info().rss
    load_before, available_before = os.getloadavg(), psutil.virtual_memory().available
    times = []
    for _ in range(SAMPLES):
        start = time.perf_counter_ns()
        output = read()
        times.append(time.perf_counter_ns() - start)
        # Returning the owned result is timed; destruction is excluded.
        del output
    peak = (
        int(Path("/proc/self/status").read_text().split("VmHWM:")[1].split()[0]) * 1024
        if sys.platform.startswith("linux")
        else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    )
    output = read()
    verify(output)
    del output
    require(inspect_container(path) == case, "fixture changed during measurement")
    require(all(type(n) is int and n > 0 for n in times), "invalid clock sample")
    return {
        "descriptor": identity,
        "case": args.case,
        "fixture": case,
        "samples_ns": times,
        "median_ns": statistics.median(times),
        "peak_rss_bytes": peak,
        "rss_after_warmup_bytes": rss_before,
        "load_before": load_before,
        "load_after": os.getloadavg(),
        "available_ram_before_bytes": available_before,
        "available_ram_after_bytes": psutil.virtual_memory().available,
        "output_verified": True,
    }


def summary(pairs):
    estimates = [
        statistics.median(pairs[i][0] for i in indices) / statistics.median(pairs[i][1] for i in indices)
        for indices in itertools.product(range(5), repeat=5)
    ]
    estimates.sort()

    def percentile(p):
        x = p * (len(estimates) - 1)
        a, b = math.floor(x), math.ceil(x)
        return estimates[a] + (estimates[b] - estimates[a]) * (x - a)

    a, b = [statistics.median(pair[i] for pair in pairs) for i in (0, 1)]
    return {
        "baseline_median": a,
        "candidate_median": b,
        "baseline_over_candidate": a / b,
        "paired_bootstrap_ci95": [percentile(0.025), percentile(0.975)],
        "raw_pairs": pairs,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", type=Path)
    p.add_argument("--capture-cache", type=Path)
    p.add_argument("--fixture", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--worker", choices=("describe", "measure"))
    p.add_argument("--root", type=Path)
    p.add_argument("--wheel", type=Path)
    p.add_argument("--identity", type=Path)
    p.add_argument("--case", type=int, choices=range(7))
    for label in ("baseline", "candidate"):
        for field in ("python", "root", "wheel", "identity"):
            p.add_argument(f"--{label}-{field}", type=Path)
    args = p.parse_args()
    if args.prepare:
        print(encoded(prepare(args.prepare, args.capture_cache)))
        return
    require(args.out is not None and not args.out.exists(), "use a new output path")
    if args.worker:
        with args.out.open("x") as stream:
            stream.write(encoded(worker(args)))
        return
    require(args.fixture is not None, "prepared fixture required")
    paths = {
        label: [getattr(args, f"{label}_{field}") for field in ("python", "root", "wheel", "identity")]
        for label in ("baseline", "candidate")
    }
    require(
        all(path is not None and path.exists() for values in paths.values() for path in values),
        "both tested wheels and build identities required",
    )
    runs = args.out.with_suffix(".runs")
    runs.mkdir(exist_ok=False)
    report = {
        "complete": False,
        "samples": SAMPLES,
        "repeats": REPEATS,
        "harness_sha256": file_hash(Path(__file__)),
        "scope": "Native checksummed section reader returns owned bytes; excludes result destruction, Python cache decoding, input validation, materialization and startup.",
        "memory_scope": "Fresh-process high-water RSS including imports, wheel/source identity verification, small-buffer fixture hashing, same-backend warmups and reads; not isolated working allocation.",
        "os_cache": "Uncontrolled and pre-read during verification; no cold-cache claim.",
        "fixture_manifest": json.loads((args.fixture / "manifest.json").read_text()),
        "descriptors": {},
        "records": [],
    }

    def save():
        args.out.write_text(encoded(report))

    def run(label, name, extra):
        python, root, wheel, identity = paths[label]
        out = runs / (name + ".json")
        command = [
            str(python),
            str(Path(__file__).resolve()),
            "--root",
            str(root),
            "--wheel",
            str(wheel),
            "--identity",
            str(identity),
            "--fixture",
            str(args.fixture.resolve()),
            "--out",
            str(out),
            *extra,
        ]
        with (runs / (name + ".log")).open("x") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        raw = out.read_bytes()
        return {"path": str(out), "sha256": sha(raw), "command": command, "report": json.loads(raw)}

    save()
    try:
        for label in paths:
            report["descriptors"][label] = run(label, "describe-" + label, ["--worker", "describe"])
        a, b = [report["descriptors"][label]["report"]["descriptor"] for label in paths]
        for key in (
            "python",
            "platform",
            "machine",
            "cpu",
            "physical_cpus",
            "logical_cpus",
            "ram_bytes",
            "packages",
            "toolchain",
            "harness_sha256",
        ):
            require(a[key] == b[key], f"comparison environment differs: {key}")
        require(a["wheel_sha256"] != b["wheel_sha256"], "identical wheels")
        changed = {
            name
            for name in a["library_sources"].keys() | b["library_sources"].keys()
            if a["library_sources"].get(name) != b["library_sources"].get(name)
        }
        require(
            changed == {"src/cache.rs", "python/ultrafast_yolo_dataset/_cache.py"},
            "unexpected source changes for this candidate",
        )
        save()
        for r in range(REPEATS):
            for c in range(7):
                for label in paths if r % 2 == 0 else tuple(paths)[::-1]:
                    value = run(label, f"r{r}-c{c}-{label}", ["--worker", "measure", "--case", str(c)])
                    require(
                        value["report"]["descriptor"] == report["descriptors"][label]["report"]["descriptor"],
                        "worker identity drift",
                    )
                    require(
                        value["report"]["fixture"] == report["fixture_manifest"]["cases"][c], "worker fixture drift"
                    )
                    value["report"].pop("descriptor")
                    value.update(repeat=r, case=c, version=label)
                    report["records"].append(value)
                    save()
                print(f"paired repeat={r} case={c}", flush=True)
        rows = {(r["repeat"], r["case"], r["version"]): r["report"] for r in report["records"]}
        require(len(rows) == 70, "incomplete paired matrix")
        report["summary"] = [
            {
                "case": c,
                **{
                    metric: summary(
                        [(rows[r, c, "baseline"][metric], rows[r, c, "candidate"][metric]) for r in range(5)]
                    )
                    for metric in ("median_ns", "peak_rss_bytes")
                },
            }
            for c in range(7)
        ]
        report["complete"] = True
        save()
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
