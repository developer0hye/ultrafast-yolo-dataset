"""Independently audit completed native-versus-native startup evidence.

Uses only the standard library. Never executes recorded commands, imports the
measured implementation, reads the dataset, or changes the recorded environment.
Run full-size cache inspection after the timing campaign terminates. Small pilot
artifacts may be inspected separately. External build identities, preflight and
reference audit are caller-selected trust anchors, not signed attestations.
"""

import argparse
import hashlib
import itertools
import json
import math
import statistics
import struct
from pathlib import Path
from zipfile import ZipFile

VERSIONS = ("baseline", "candidate")
EMBEDDED = (
    "src/parser.rs", "src/snapshot.rs", "src/provenance.rs",
    "src/materialize.rs", "src/cache.rs", "src/lib.rs", "Cargo.toml", "Cargo.lock",
)
METRICS = ("constructor_s", "first_batch_total_s", "peak_rss_constructor_bytes")
RUNTIME = (
    "python", "platform", "machine", "cpu", "physical_cpus", "logical_cpus",
    "ram_bytes", "packages", "toolchain", "harness_sha256",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def number(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def summary(pairs):
    require(len(pairs) == 5, "summary requires five measured pairs")
    samples = []
    for indexes in itertools.product(range(5), repeat=5):
        left, right = ([pairs[i][column] for i in indexes] for column in (0, 1))
        samples.append(statistics.median(left) / statistics.median(right))
    samples.sort()

    def percentile(probability):
        position = probability * (len(samples) - 1)
        low, high = math.floor(position), math.ceil(position)
        return samples[low] + (samples[high] - samples[low]) * (position - low)

    left, right = (statistics.median(p[column] for p in pairs) for column in (0, 1))
    return {
        "baseline_median": left,
        "candidate_median": right,
        "baseline_over_candidate": left / right,
        "paired_bootstrap_ci95": [percentile(.025), percentile(.975)],
        "raw_pairs": pairs,
    }


def inspect_cache(path):
    """Stream all eleven sections and retain only JSON metadata in memory."""
    with path.open("rb") as stream:
        header = stream.read(16)
        require(header == b"UYDCACHE" + struct.pack("<II", 1, 11), "cache header/version/section count")
        sections = []
        for _ in range(11):
            descriptor = stream.read(40)
            require(len(descriptor) == 40, "truncated cache descriptor")
            sections.append({"bytes": struct.unpack("<Q", descriptor[:8])[0], "sha256": descriptor[8:].hex()})
        require(sections[0]["bytes"] <= 256 * 1024**2, "metadata limit")
        require(456 + sum(s["bytes"] for s in sections) == path.stat().st_size <= 2 * 1024**3, "cache byte limit/size")
        metadata = None
        for index, section in enumerate(sections):
            h, remaining = hashlib.sha256(), section["bytes"]
            chunks = []
            while remaining:
                block = stream.read(min(remaining, 1024 * 1024))
                require(block, "truncated cache payload")
                h.update(block)
                remaining -= len(block)
                if index == 0:
                    chunks.append(block)
            require(h.hexdigest() == section["sha256"], "cache section digest")
            if index == 0:
                metadata = json.loads(b"".join(chunks))
        require(stream.read(1) == b"", "trailing cache bytes")
    profile = metadata.pop("profile")
    require(digest(profile), "invalid cache profile")
    return {
        "sha256": sha(path), "bytes": path.stat().st_size, "sections": sections,
        "profile": profile,
        "metadata_except_profile_sha256": hashlib.sha256(canonical(metadata).encode()).hexdigest(),
    }


def check_result(result, mode, pairs, workers, expected, wall_seconds, cache):
    require(result["backend"] == "native-content" and result["mode"] == mode, "backend/mode")
    require(result["workers"] == workers and result["loader_workers"] == 0, "worker settings")
    require(result["scan_summary"] == {
        "found": pairs, "missing": 0, "empty": 0, "corrupt": 0, "total": pairs,
    }, "scan counts")
    require(result["diagnostic_count"] == 0 and result["native_fallback_count"] == 0, "diagnostics/fallback")
    require(result["output_sha256"] == expected, "output differs from independent reference")
    constructor, batch = result["constructor_s"], result["first_batch_total_s"]
    require(number(constructor, 1e-12) and number(batch, constructor) and batch <= wall_seconds, "startup timing")
    stages = result["stages_s"]
    require(set(stages) == {"_validate_inputs", "to_ultralytics_labels", "_encode" if mode == "miss" else "_decode"}, "observed cache stages")
    require(all(number(v, 1e-12) for v in stages.values()) and sum(stages.values()) <= constructor, "stage timing")
    peaks = [result[name] for name in (
        "peak_rss_before_constructor_bytes", "peak_rss_constructor_bytes", "peak_rss_first_batch_bytes",
    )]
    require(all(type(v) is int and v > 0 for v in peaks) and peaks == sorted(peaks), "RSS high-water order")
    require(number(result["rss_before_constructor_bytes"], 1) and result["rss_before_constructor_bytes"] <= peaks[0], "pre-constructor RSS")
    require(result["cache_bytes"] == cache["bytes"], "result cache size")
    require(all(number(result[name]) for name in ("user_s", "system_s", "minor_faults", "major_faults")), "CPU/fault counters")
    require(number(result["available_ram_bytes"], 1), "available memory")
    require(len(result["loadavg"]) == 3 and all(number(v) for v in result["loadavg"]), "host load")


def audit(args):
    report = read(args.report)
    require(report["complete"] is True and "error" not in report, "unfinished or failed campaign")
    require(args.rounds in (1, 5) and report["rounds"] == args.rounds and report["pairs"] == args.pairs, "declared campaign size")
    require(args.workers > 0 and args.pairs > 0, "invalid requested counts")
    fixture, preflight, reference = map(read, (args.fixture_manifest, args.preflight, args.reference_audit))
    require(report["corpus"] == fixture, "fixture manifest")
    require(fixture["kind"] == "ultrafast-yolo-startup-synthetic-v1" and fixture["task"] == args.task and fixture["count"] == args.pairs, "fixture scope")
    require(preflight["pairs"] == args.pairs and preflight["counts"] == {"images": args.pairs, "labels": args.pairs}, "preflight counts")
    require(preflight["all_regular_single_link_files"] is True, "physical file preflight")
    require(preflight["fingerprint"] == fixture["fingerprint"] and fixture["fingerprint"]["files"] == 2 * args.pairs, "input fingerprint")
    require(digest(fixture["fingerprint"]["sha256"]), "input digest")
    require(reference["complete_requested_samples"] is True, "reference audit incomplete")
    require(reference["phase"] in ("p3", "p4") and reference["task"] == args.task and reference["pairs"] == args.pairs, "reference audit scope")
    require(reference["fixture_manifest_sha256"] == sha(args.fixture_manifest) and reference["preflight_sha256"] == sha(args.preflight), "reference fixture/preflight binding")
    expected = reference["output_sha256"]
    require(set(expected) == {"labels", "first_batch", "scan_summary_and_messages"} and all(digest(v) for v in expected.values()), "reference output hashes")
    require(report["output_sha256"] == expected, "parent reference parity")
    wrapper_hash = sha(args.wrapper)
    require(report["wrapper_sha256"] == wrapper_hash, "wrapper identity")
    harness = {"bench/" + p.name: sha(p) for p in sorted((args.harness_root / "bench").glob("*.py"))}
    require(report["harness_sources"] == harness, "frozen harness source inventory")
    anchors, profiles, caches, sources = {}, {}, {}, {}
    for version in VERSIONS:
        identity_path = getattr(args, version + "_identity")
        source = getattr(args, version + "_source")
        wheel = getattr(args, version + "_wheel")
        anchor = read(identity_path)
        anchors[version] = anchor
        require(report["anchors"][version] == anchor and report["anchor_sha256"][version] == sha(identity_path), "external build identity")
        require(anchor["wheel_sha256"] == sha(wheel), "frozen wheel digest")
        require(anchor["toolchain"]["profile"] == "release", "release build required")
        paths = [source / name for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml")]
        paths += sorted((source / "src").rglob("*.rs"))
        package = source / "python/ultrafast_yolo_dataset"
        paths += sorted(package.glob("*.py")) + sorted(package.glob("*.json"))
        sources[version] = {str(p.relative_to(source)): sha(p) for p in paths}
        require(sources[version] == anchor["library_sources"], "library source inventory/digests")
        with ZipFile(wheel) as archive:
            extensions = [n for n in archive.namelist() if n.startswith("ultrafast_yolo_dataset/_native.") and n.endswith((".so", ".pyd"))]
            require(len(extensions) == 1 and hashlib.sha256(archive.read(extensions[0])).hexdigest() == anchor["extension_sha256"], "wheel extension")
            for name, expected_sha in sources[version].items():
                if name.startswith("python/"):
                    require(hashlib.sha256(archive.read(name.removeprefix("python/"))).hexdigest() == expected_sha, "wheel Python source")
        h = hashlib.sha256()
        for name in EMBEDDED:
            h.update((source / name).read_bytes())
        profiles[version] = h.hexdigest()
        entries = list((args.caches_dir / version).glob("*.uydcache"))
        require(len(entries) == 1, "expected one retained cache per version")
        caches[version] = inspect_cache(entries[0])
    require(anchors["baseline"]["toolchain"] == anchors["candidate"]["toolchain"], "build settings differ")
    require(sources["baseline"].keys() == sources["candidate"].keys(), "source inventory differs")
    require({k for k in sources["baseline"] if sources["baseline"][k] != sources["candidate"][k]} == {"src/snapshot.rs"}, "unrelated runtime changes")
    require(caches["baseline"]["sections"][1:] == caches["candidate"]["sections"][1:] and caches["baseline"]["metadata_except_profile_sha256"] == caches["candidate"]["metadata_except_profile_sha256"], "cache payload/metadata parity")

    plan = [(mode, r, v) for mode in ("miss", "hit") for r in range(args.rounds) for v in (VERSIONS if r % 2 == 0 else VERSIONS[::-1])]
    require([(r["mode"], r["round"], r["version"]) for r in report["records"]] == plan, "missing/duplicate/reordered measured workers")
    require([(r["mode"], r["round"], r["version"]) for r in report["primers"]] == [("hit", -1, v) for v in VERSIONS], "separate primers")
    ordered = report["records"][:2 * args.rounds] + report["primers"] + report["records"][2 * args.rounds:]
    descriptors, pids, raw_hashes, results, commands = {}, set(), {}, {}, {}
    end_ns = 0
    for row in ordered:
        mode, repeat, version = row["mode"], row["round"], row["version"]
        prime = repeat == -1
        require(row["prime"] is prime, "parent primer flag")
        name = f"{mode}-{'prime' if prime else repeat}-{version}"
        path, log = args.runs_dir / (name + ".json"), args.runs_dir / (name + ".log")
        value = read(path)
        require(Path(row["path"]).name == path.name and sha(path) == row["sha256"] and sha(log) == row["log_sha256"], "raw/log identity")
        require(canonical(value) == canonical(row["report"]), "raw/parent disagreement")
        raw_hashes[path.name], raw_hashes[log.name] = sha(path), sha(log)
        require(value["prime"] is prime and value["mode"] == mode and value["version"] == version, "raw worker identity")
        require(type(value["pid"]) is int and value["pid"] > 0 and value["pid"] not in pids, "fresh worker PID")
        pids.add(value["pid"])
        start, end = value["wall_start_ns"], value["wall_end_ns"]
        require(type(start) is int and type(end) is int and end > start >= end_ns, "overlapping/reordered worker intervals")
        end_ns = end
        descriptor = value["descriptor"]
        for key in ("wheel_sha256", "extension_sha256", "library_sources", "toolchain"):
            require(descriptor[key] == anchors[version][key], "descriptor build field: " + key)
        require(descriptor["build_identity_sha256"] == report["anchor_sha256"][version], "descriptor identity digest")
        require(descriptor["harness_sha256"] == harness["bench/cache_read_comparison.py"], "descriptor harness digest")
        require(version not in descriptors or descriptors[version] == descriptor, "runtime descriptor drift")
        descriptors[version] = descriptor
        require(value["wrapper_sha256"] == wrapper_hash and value["compiled_source_profile"] == profiles[version], "wrapper/compiled profile")
        require({k: v for k, v in value["cache"].items() if k != "path"} == caches[version], "retained cache differs from worker")
        command = row["command"]
        require(isinstance(command, list) and len(command) == (24 if prime else 23) and all(isinstance(v, str) for v in command), "command shape")
        require(Path(command[1]).name == args.wrapper.name and command[2] == "--worker", "wrapper command")
        flags = command[3:-1] if prime else command[3:]
        require(not prime or command[-1] == "--prime", "primer command")
        options = dict(zip(flags[::2], flags[1::2]))
        require(len(options) * 2 == len(flags), "duplicate command flags")
        require(set(options) == {"--version", "--corpus", "--pairs", "--harness-root", "--source", "--wheel", "--identity", "--mode", "--cache-root", "--out"}, "command options")
        require(options["--version"] == version and options["--mode"] == mode and options["--pairs"] == str(args.pairs), "command experiment settings")
        require(options["--out"] == row["path"] and Path(options["--corpus"]).name == args.fixture_manifest.parent.name, "command output/corpus")
        for flag, actual in (("source", getattr(args, version + "_source")), ("wheel", getattr(args, version + "_wheel")), ("identity", getattr(args, version + "_identity")), ("harness-root", args.harness_root)):
            require(Path(options["--" + flag]).name == actual.name, "command artifact: " + flag)
        require(Path(value["cache"]["path"]).parent == Path(options["--cache-root"]) and Path(options["--cache-root"]).name == version, "command cache isolation")
        stable_command = {k: v for k, v in options.items() if k not in ("--mode", "--out")}
        stable_command["python"] = command[0]
        require(version not in commands or commands[version] == stable_command, "command configuration drift")
        commands[version] = stable_command
        if prime:
            require(value["result"] == {"primed": "native-content"}, "primer result")
        else:
            check_result(value["result"], mode, args.pairs, args.workers, expected, (end - start) / 1e9, caches[version])
            results[mode, repeat, version] = value["result"]
    require(all(descriptors["baseline"][k] == descriptors["candidate"][k] for k in RUNTIME), "different runtime/host")
    recomputed = {}
    if args.rounds == 5:
        recomputed = {mode: {metric: summary([[results[mode, r, v][metric] for v in VERSIONS] for r in range(5)]) for metric in METRICS} for mode in ("miss", "hit")}
        require(canonical(report["summary"]) == canonical(recomputed), "parent statistics/raw pairs mismatch")
    else:
        require("summary" not in report, "pilot must not claim repeated summary")
    return {
        "complete": True, "report_sha256": sha(args.report), "auditor_sha256": sha(Path(__file__)),
        "pairs": args.pairs, "rounds": args.rounds, "workers": args.workers,
        "measured_workers": len(results), "separate_primers": 2,
        "anchors": {name: sha(getattr(args, name)) for name in ("fixture_manifest", "preflight", "reference_audit", "wrapper", "baseline_identity", "candidate_identity")},
        "raw_sha256": raw_hashes, "compiled_source_profiles": profiles,
        "cache_checks": caches, "output_sha256": expected, "summary": recomputed,
        "limitations": [
            "Native-versus-native incremental comparison; not upstream Python or training throughput.",
            "Cache profiles agree with retained cache bytes and the frozen wrapper's runtime assertions; the auditor does not re-execute runtime plugin/CPU profile computation.",
            "Preflight and per-worker frozen-harness checks bind inputs; this audit does not reread dataset files.",
            "RSS is process high-water at constructor/first batch, includes earlier imports/preflight, and is not isolated allocation.",
            "External artifacts are trust anchors, not cryptographic attestations of execution.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "runs-dir", "caches-dir", "wrapper", "harness-root", "fixture-manifest", "preflight", "reference-audit", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for version in VERSIONS:
        for name in ("identity", "source", "wheel"):
            parser.add_argument(f"--{version}-{name}", type=Path, required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--rounds", type=int, choices=(1, 5), required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--task", choices=("detect", "segment"), required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "retain earlier audit output")
    result = audit(args)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Verified {result['measured_workers']} measured workers and two separate primers.")


if __name__ == "__main__":
    main()
