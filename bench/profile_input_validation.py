"""Read-only diagnostic of native input validation, not a startup speedup test.

Uses evenly spaced entries from an existing verified cache. Content verification
retains all native read/consistency checks. Metadata mode is an attribution
control, not an equivalent optimization. All cases share a process and OS cache.
"""

import argparse
import hashlib
import itertools
import json
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

from ultrafast_yolo_dataset import _native


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert args.count > 0 and args.rounds > 0
    state = {"complete": False, "passed": False, "pid": os.getpid(), "records": []}
    # Reserve outside the failure handler: preserve an earlier result on rerun.
    with args.out.open("x") as stream:
        stream.write(json.dumps(state) + "\n")

    def save():
        temporary = args.out.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        os.replace(temporary, args.out)

    try:
        extension = Path(_native.__file__).resolve()
        assert extension.is_relative_to(Path(sys.prefix).resolve())
        identity = json.loads(args.identity.read_text())
        extension_sha = sha(extension)
        assert extension_sha == identity["extension_sha256"]
        assert sha(args.cache) == args.expected_cache_sha256
        sections = _native.read_cache_sections(str(args.cache), 2 * 1024**3)
        metadata = json.loads(sections[0])
        total = len(metadata["images"])
        assert total == len(metadata["labels"]) >= args.count
        indices = [i * total // args.count for i in range(args.count)]
        assert len(set(indices)) == args.count
        selected = {}
        for kind, proof in zip(("images", "labels"), sections[-2:], strict=True):
            assert len(proof) == total * 57
            paths = [metadata[kind][i] for i in indices]
            table = b"".join(proof[i * 57 : (i + 1) * 57] for i in indices)
            selected[kind] = (paths, table)
        limits = {"images": metadata["config"]["max_image_bytes"], "labels": metadata["config"]["max_file_bytes"]}
        del sections, metadata, proof
        conditions = list(itertools.product(("images", "labels"), (1, 7), (True, False)))
        plan = [
            {"round": repeat, "kind": kind, "workers": workers, "content": content}
            for repeat in range(args.rounds)
            for kind, workers, content in (conditions if repeat % 2 == 0 else conditions[::-1])
        ]
        state.update(
            phase="profiling",
            args=vars(args) | {"cache": str(args.cache), "identity": str(args.identity), "out": str(args.out)},
            script_sha256=sha(Path(__file__)),
            identity_sha256=sha(args.identity),
            extension_sha256=extension_sha,
            python=sys.version,
            platform=platform.platform(),
            cache_sha256=args.expected_cache_sha256,
            original_pairs=total,
            selected_indices=indices,
            plan=plan,
            selected={
                kind: {
                    "count": len(paths),
                    "table_sha256": hashlib.sha256(table).hexdigest(),
                    "paths_sha256": hashlib.sha256(json.dumps(paths, ensure_ascii=False).encode()).hexdigest(),
                    "max_bytes": limits[kind],
                }
                for kind, (paths, table) in selected.items()
            },
            scope="Diagnostic within one process; no cache writes, weaker metadata control is not an equivalent optimization, no inference about full startup or training speedup.",
        )
        save()
        for condition in plan:
            paths, table = selected[condition["kind"]]
            record = dict(condition, started_at_ns=time.time_ns())
            state["records"].append(record)
            save()
            print("start", condition, flush=True)
            before = resource.getrusage(resource.RUSAGE_SELF)
            started = time.perf_counter()
            _native.verify_fingerprint_table(
                table, paths, limits[condition["kind"]], condition["workers"], condition["content"]
            )
            elapsed = time.perf_counter() - started
            after = resource.getrusage(resource.RUSAGE_SELF)
            record.update(
                complete=True,
                verified_paths=len(paths),
                elapsed_s=elapsed,
                user_s=after.ru_utime - before.ru_utime,
                system_s=after.ru_stime - before.ru_stime,
                minor_faults=after.ru_minflt - before.ru_minflt,
                major_faults=after.ru_majflt - before.ru_majflt,
                finished_at_ns=time.time_ns(),
            )
            save()
            print("done", condition, "elapsed_s", elapsed, flush=True)
        assert len(state["records"]) == len(plan) and all(r["complete"] for r in state["records"])
        assert sha(args.cache) == args.expected_cache_sha256 and sha(extension) == extension_sha
        state.update(complete=True, passed=True, phase="diagnostic-complete")
        save()
    except BaseException:
        state.update(complete=True, passed=False, error=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
