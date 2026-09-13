"""Independent audit of the fixed M2 cache-reader comparison, including weighted bootstrap."""

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

HARNESS = "f79fd09baf58e551a241fa00db2fd8b40c07d736f880fe92ed90c25f5e76c0b3"
MANIFEST = "9e1278cc1987958cbd93e879e368ff3839c11c29e56ac42343d8fc6f3ccce00b"
WHEELS = {
    "baseline": "2ae058b33e702306d1ce89ea5cd43e7bc5fcdb87825b889f9556bfe5c864f595",
    "candidate": "25f0ce7ff2e40f106ec7af04a744c3883671c4ee502f3bd337e2414fe01b88d3",
}
EXTENSIONS = {
    "baseline": "5799c19d7fbe704451cb9ac3e7e72f1b6c8ecdd8219d7fca971dfcb24eff2b0b",
    "candidate": "ec52a219e5e159017d221422fa8292cbe24022157370548bd445e3fb082f32d6",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def arguments(command):
    require(isinstance(command, list) and len(command) >= 2, "missing command")
    flags = command[2:]
    require(len(flags) % 2 == 0, "malformed command")
    pairs = list(zip(flags[::2], flags[1::2]))
    require(len({k for k, _ in pairs}) == len(pairs), "duplicate command argument")
    return dict(pairs)


def stats(pairs):
    """Multiset weights independently reproduce the 3,125 ordered resamples."""
    weighted = []
    for indices in itertools.combinations_with_replacement(range(5), 5):
        multiplicity = math.factorial(5)
        for i in range(5):
            multiplicity //= math.factorial(indices.count(i))
        a = statistics.median(pairs[i][0] for i in indices)
        b = statistics.median(pairs[i][1] for i in indices)
        weighted.append((a / b, multiplicity))
    weighted.sort()
    require(sum(n for _, n in weighted) == 3125, "invalid bootstrap weights")

    def nth(index):
        for value, count in weighted:
            if index < count:
                return value
            index -= count
        raise ValueError("bootstrap index outside weighted distribution")

    def quantile(probability):
        index = probability * 3124
        low, high = math.floor(index), math.ceil(index)
        return nth(low) + (nth(high) - nth(low)) * (index - low)

    a, b = (statistics.median(p[i] for p in pairs) for i in (0, 1))
    return {
        "baseline_median": a,
        "candidate_median": b,
        "baseline_over_candidate": a / b,
        "paired_bootstrap_ci95": [quantile(0.025), quantile(0.975)],
        "raw_pairs": pairs,
    }


def audit(report_bytes, raw_files, manifest_bytes, identity_bytes, launch):
    report = json.loads(report_bytes)
    require(report.get("complete") is True and "error" not in report, "incomplete or failed matrix")
    require(report["samples"] == 5 and report["repeats"] == 5, "wrong repetition scope")
    require(report["harness_sha256"] == HARNESS, "wrong measured harness")
    require(sha(manifest_bytes) == MANIFEST, "fixture manifest changed")
    manifest = json.loads(manifest_bytes)
    require(
        report["fixture_manifest"] == manifest and manifest["generator_sha256"] == HARNESS, "wrong fixture identity"
    )
    require(len(manifest["cases"]) == 7 and manifest["cases"][6]["bytes"] == 195001501, "wrong full cache")
    require(manifest["cases"][6]["sha256"] == manifest["captured_cache"]["sha256"], "capture mismatch")
    layouts = [[2, 0], [2, 4194303], [2, 4194305], [2, 67108864], [2, 268435456], [2] + [67108864] * 4]
    require([[s["bytes"] for s in c["sections"]] for c in manifest["cases"][:6]] == layouts, "wrong layout")
    campaign = arguments(launch["command"])
    require(
        launch["environment_overrides"]
        == {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "YOLO_OFFLINE": "true"},
        "thread/environment settings differ",
    )
    identities = {k: json.loads(v) for k, v in identity_bytes.items()}
    descriptors, expected_names, checksums = {}, set(), {}

    def raw(row, name, version, worker, case=None):
        require(Path(row["path"]).name == name and name in raw_files, "wrong raw path")
        data = raw_files[name]
        require(sha(data) == row["sha256"], "raw bytes mismatch")
        checksums[name] = sha(data)
        expected_names.add(name)
        command = row["command"]
        flags = arguments(command)
        expected = {
            "--root": campaign[f"--{version}-root"],
            "--wheel": campaign[f"--{version}-wheel"],
            "--identity": campaign[f"--{version}-identity"],
            "--fixture": campaign["--fixture"],
            "--out": row["path"],
            "--worker": worker,
        }
        if case is not None:
            expected["--case"] = str(case)
        require(
            flags == expected and command[0] == campaign[f"--{version}-python"] and command[1] == launch["command"][1],
            "wrong measured worker command",
        )
        return json.loads(data)

    require(set(report["descriptors"]) == {"baseline", "candidate"}, "wrong versions")
    for version in ("baseline", "candidate"):
        row = report["descriptors"][version]
        value = raw(row, "describe-" + version + ".json", version, "describe")
        require(value == row["report"], "descriptor parent/raw disagreement")
        descriptor = value["descriptor"]
        anchor = identities[version]
        require(
            anchor["wheel_sha256"] == WHEELS[version] and anchor["extension_sha256"] == EXTENSIONS[version],
            "unrecognized binary pair",
        )
        require(descriptor["build_identity_sha256"] == sha(identity_bytes[version]), "wrong source anchor")
        require(all(descriptor[k] == anchor[k] for k in anchor), "descriptor/build identity disagreement")
        require(
            descriptor["harness_sha256"] == HARNESS and descriptor["ram_bytes"] == 17179869184, "wrong host or harness"
        )
        descriptors[version] = descriptor
    a, b = descriptors.values()
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
        require(a[key] == b[key], "different environments: " + key)
    require(a["library_sources"].keys() == b["library_sources"].keys(), "source inventory changed")
    require(
        {n for n in a["library_sources"] if a["library_sources"][n] != b["library_sources"][n]}
        == {"src/cache.rs", "python/ultrafast_yolo_dataset/_cache.py"},
        "unexpected library changes",
    )
    order = [
        (r, c, v)
        for r in range(5)
        for c in range(7)
        for v in (("baseline", "candidate") if r % 2 == 0 else ("candidate", "baseline"))
    ]
    require(len(report["records"]) == 70, "missing measured workers")
    rows = {}
    for row, key in zip(report["records"], order):
        repeat, case, version = key
        require((row["repeat"], row["case"], row["version"]) == key, "wrong counterbalanced order")
        name = f"r{repeat}-c{case}-{version}.json"
        value = raw(row, name, version, "measure", case)
        require(value.pop("descriptor") == descriptors[version], "worker environment drift")
        require(value == row["report"], "parent/raw measurement disagreement")
        require(value["case"] == case and value["fixture"] == manifest["cases"][case], "wrong measured fixture")
        require(value["output_verified"] is True, "unverified output")
        samples = value["samples_ns"]
        require(len(samples) == 5 and all(type(x) is int and x > 0 for x in samples), "invalid clock samples")
        require(value["median_ns"] == statistics.median(samples), "wrong worker median")
        for field in ("peak_rss_bytes", "rss_after_warmup_bytes"):
            require(type(value[field]) is int and 0 < value[field] <= a["ram_bytes"], "invalid RSS")
        for field in ("load_before", "load_after"):
            require(len(value[field]) == 3 and all(math.isfinite(x) and x >= 0 for x in value[field]), "invalid load")
        for field in ("available_ram_before_bytes", "available_ram_after_bytes"):
            require(type(value[field]) is int and 0 <= value[field] <= a["ram_bytes"], "invalid available RAM")
        rows[key] = value
    require(set(raw_files) == expected_names and len(checksums) == 72, "missing or extra raw reports")
    summary = []
    for case in range(7):
        result = {"case": case}
        for metric in ("median_ns", "peak_rss_bytes"):
            pairs = [[rows[r, case, "baseline"][metric], rows[r, case, "candidate"][metric]] for r in range(5)]
            result[metric] = stats(pairs)
        summary.append(result)
    require(report["summary"] == summary, "parent aggregate or interval disagreement")
    return {
        "complete": True,
        "measured_workers": 70,
        "descriptors": 2,
        "report_sha256": sha(report_bytes),
        "fixture_manifest_sha256": MANIFEST,
        "harness_sha256": HARNESS,
        "raw_sha256": checksums,
        "summary": summary,
        "bootstrap": "Independent multiset enumeration, weighted to all 3125 ordered resamples",
        "limits": [
            "Native reader only; not full cache loading, startup, save-path or training.",
            "Process peak RSS includes imports, verification and warmups; not isolated working allocation.",
            "Storage pre-read and OS cache uncontrolled; not cold-cache timing.",
            "Harness launches fresh processes serially but raw workers do not record individual PIDs or timestamps.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/Volumes/T7/ultrafast-vision-build"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    require(not args.out.exists(), "retain previous audit")
    repo = Path(__file__).parents[2]
    manifest = root / "cache-reader-seven-layouts-m2-v1/manifest.json"
    raw_files = {p.name: p.read_bytes() for p in (root / "cache-reader-paired-m2-v1.runs").glob("*.json")}
    identities = {
        "baseline": (repo / "docs/validation/cache-reader-baseline-m2-identity-v1.json").read_bytes(),
        "candidate": (repo / "docs/validation/cache-copy-m2-build-identity-v1.json").read_bytes(),
    }
    launch = json.loads((root / "cache-reader-paired-m2-v1-launch.json").read_text())
    result = audit(
        (root / "cache-reader-paired-m2-v1.json").read_bytes(), raw_files, manifest.read_bytes(), identities, launch
    )
    # Re-hash the immutable fixture after timing, without retaining large buffers.
    for case in json.loads(manifest.read_text())["cases"]:
        path = manifest.parent / case["file"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while data := stream.read(1048576):
                digest.update(data)
        require(path.stat().st_size == case["bytes"] and digest.hexdigest() == case["sha256"], "fixture file changed")
    result["all_fixture_files_rehashed_after_campaign"] = True
    result["auditor_sha256"] = sha(Path(__file__).read_bytes())
    result["logs_sha256"] = {}
    for name in raw_files:
        log = root / "cache-reader-paired-m2-v1.runs" / Path(name).with_suffix(".log")
        result["logs_sha256"][log.name] = sha(log.read_bytes())
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("Audited 70 measured workers, 2 descriptors and 7 complete fixture files.")


if __name__ == "__main__":
    main()
