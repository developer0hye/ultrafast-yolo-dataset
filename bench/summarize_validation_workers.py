"""Independently read all worker-comparison artifacts before reporting ratios."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import traceback


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            assert key not in result, f"duplicate JSON key: {key}"
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON: {value}")

    result = json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=invalid)
    json.dumps(result, allow_nan=False)  # Reject overflowed numeric literals too.
    return result


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def positive(value):
    assert type(value) in (int, float) and math.isfinite(value) and value > 0
    return value


def quantile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (index - lower)


def paired_summary(ratios):
    assert len(ratios) == 6 and all(positive(value) for value in ratios)
    # Enumerate all 6**6 block resamples; no random seed or Monte Carlo error.
    resamples = [statistics.median(sample) for sample in itertools.product(ratios, repeat=6)]
    return dict(
        block_ratios=ratios,
        median=statistics.median(ratios),
        bootstrap_percentile_95=[quantile(resamples, 0.025), quantile(resamples, 0.975)],
        resamples=len(resamples),
    )


def audit(args):
    parent = read(args.series)
    assert parent["complete"] is True and parent["passed"] is True and "error" not in parent
    assert (parent["planned_trials"], parent["measured_trials"], parent["pairs"]) == (21, 18, 500000)
    expected = [dict(primer=True, block=-1, position=i, workers=w) for i, w in enumerate((4, 7, 16))]
    expected += [
        dict(primer=False, block=b, position=i, workers=w)
        for b, order in enumerate(itertools.permutations((4, 7, 16)))
        for i, w in enumerate(order)
    ]
    assert canonical(parent["plan"]) == canonical(expected)
    assert len(parent["records"]) == 21
    source_sha = sha(args.source)
    assert parent["script_sha256"] == source_sha
    identity = read(args.identity)
    assert sha(args.extension) == identity["extension_sha256"]
    assert sha(args.cache) == args.expected_cache_sha256
    identity_sha = sha(args.identity)
    binding = None
    commands = None
    rows = []
    previous_finish = parent["started_at_ns"]
    for index, (spec, rec) in enumerate(zip(expected, parent["records"], strict=True)):
        assert canonical(rec["spec"]) == canonical(spec)
        assert rec["validated"] is True and type(rec["returncode"]) is int and rec["returncode"] == 0
        assert not rec.get("timed_out", False)
        assert previous_finish <= rec["started_at_ns"] < rec["finished_at_ns"]
        previous_finish = rec["finished_at_ns"]
        raw_path = args.runs / f"{index:02d}.json"
        log_path = raw_path.with_suffix(".log")
        assert sha(raw_path) == rec["raw_sha256"] and sha(log_path) == rec["log_sha256"]
        assert Path(rec["path"]).is_absolute() and Path(rec["path"]).name == raw_path.name
        command = rec["command"]
        assert len(command) == 14 and Path(command[0]).is_absolute() and Path(command[1]).name == args.source.name
        wanted = [
            command[0],
            command[1],
            "--cache",
            command[3],
            "--identity",
            command[5],
            "--expected-cache-sha256",
            args.expected_cache_sha256,
            "--out",
            rec["path"],
            "--count",
            "500000",
            "--trial-workers",
            str(spec["workers"]),
        ]
        assert command == wanted
        common = command[:8]
        commands = common if commands is None else commands
        assert common == commands
        raw = read(raw_path)
        assert raw["complete"] is True and raw["passed"] is True
        assert not (set(raw) & {"error", "sampling_error"})
        assert raw["pid"] == rec["pid"] and raw["content"] is True
        assert type(raw["count"]) is int and raw["count"] == 500000
        assert type(raw["workers"]) is int and raw["workers"] == spec["workers"]
        assert rec["started_at_ns"] <= raw["started_at_ns"] < raw["finished_at_ns"] <= rec["finished_at_ns"]
        current = {
            k: raw[k] for k in ("script_sha256", "extension_sha256", "identity_sha256", "cache_sha256", "selected")
        }
        assert current["script_sha256"] == source_sha
        assert current["extension_sha256"] == identity["extension_sha256"]
        assert current["identity_sha256"] == identity_sha and current["cache_sha256"] == args.expected_cache_sha256
        binding = current if binding is None else binding
        assert canonical(current) == canonical(binding)
        assert set(raw["selected"]) == {"images", "labels"}
        for item in raw["selected"].values():
            assert item["count"] == 500000 and type(item["count"]) is int
            for field in ("table_sha256", "paths_sha256"):
                assert len(item[field]) == 64 and all(c in "0123456789abcdef" for c in item[field])
            positive(item["max_bytes"])
        assert [r["kind"] for r in raw["records"]] == ["images", "labels"]
        finish = raw["started_at_ns"]
        for stage in raw["records"]:
            assert (
                stage["complete"] is True and type(stage["verified_paths"]) is int and stage["verified_paths"] == 500000
            )
            assert finish <= stage["started_at_ns"] < stage["finished_at_ns"] <= raw["finished_at_ns"]
            finish = stage["finished_at_ns"]
            positive(stage["elapsed_s"])
            assert stage["elapsed_s"] <= (stage["finished_at_ns"] - stage["started_at_ns"]) / 1e9 + 0.01
            assert stage["user_s"] >= 0 and stage["system_s"] >= 0
        total = sum(r["elapsed_s"] for r in raw["records"])
        assert total == raw["validation_elapsed_s"] == rec["validation_elapsed_s"]
        samples = raw["samples"]
        assert len(samples) >= 2 and samples[0]["phase"] == "ready" and samples[-1]["phase"] == "finished"
        assert all(s["phase"] in {"ready", "images", "labels", "finished"} for s in samples)
        assert all(a["at_ns"] <= b["at_ns"] for a, b in zip(samples, samples[1:]))
        assert raw["started_at_ns"] <= samples[0]["at_ns"] <= samples[-1]["at_ns"] <= raw["finished_at_ns"]
        peak = max(positive(s["rss_bytes"]) for s in samples)
        assert peak == raw["sampled_peak_rss"] == rec["sampled_peak_rss"]
        rows.append(
            dict(
                index=index,
                **spec,
                total_s=total,
                images_s=raw["records"][0]["elapsed_s"],
                labels_s=raw["records"][1]["elapsed_s"],
                sampled_peak_rss=peak,
                user_s=sum(r["user_s"] for r in raw["records"]),
                system_s=sum(r["system_s"] for r in raw["records"]),
            )
        )
    assert canonical(parent["binding"]) == canonical(binding)
    assert parent["finished_at_ns"] >= previous_finish
    measured = [row for row in rows if not row["primer"]]
    blocks = {block: {r["workers"]: r for r in measured if r["block"] == block} for block in range(6)}
    ratios = {
        str(w): {
            metric: paired_summary([blocks[b][4][metric] / blocks[b][w][metric] for b in range(6)])
            for metric in ("total_s", "images_s", "labels_s", "sampled_peak_rss")
        }
        for w in (7, 16)
    }
    medians = {
        str(w): {
            metric: statistics.median(r[metric] for r in measured if r["workers"] == w)
            for metric in ("total_s", "images_s", "labels_s", "sampled_peak_rss", "user_s", "system_s")
        }
        for w in (4, 7, 16)
    }
    return dict(
        passed=True,
        trials=21,
        measured_trials=18,
        verified_paths_per_trial=1000000,
        series_sha256=sha(args.series),
        auditor_sha256=sha(Path(__file__)),
        binding=binding,
        records=rows,
        medians=medians,
        four_worker_relative_ratios=ratios,
        scope="Independent raw/checksum/command/count/timing readback. Ratios above one favor the candidate. Exact percentile bootstrap of six matched block ratios is exploratory on this shared host and corpus, not a universal confidence guarantee. Three primers excluded by the predeclared plan. Sampled phase RSS is not whole-startup peak. Full startup and chosen-setting confirmation remain required.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("series", "runs", "source", "identity", "extension", "cache", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    args = parser.parse_args()
    with args.out.open("x") as stream:
        try:
            result = audit(args)
        except BaseException:
            result = dict(passed=False, error=traceback.format_exc())
            json.dump(result, stream, indent=2)
            stream.write("\n")
            raise
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result["medians"], indent=2))


if __name__ == "__main__":
    main()
