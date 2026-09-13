"""One actual constructor/first-batch trial with an explicit scan-worker count.

Use a private generated cache directory. Existing content/fixture/label/batch
checks remain active. The caller supplies the independently verified output
hashes and an isolated copy of the original validated cache for hit trials.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def execute(args, state):
    assert type(args.workers) is int and 1 <= args.workers <= 256
    assert args.cache_root.is_absolute() and args.cache_root.is_dir()
    assert not args.cache_root.is_relative_to(args.corpus)
    expected = json.loads(args.expected_outputs.read_text())
    assert set(expected) == {"labels", "first_batch", "scan_summary_and_messages"}
    harness_files = {
        name: sha(args.harness_root / name)
        for name in (
            "bench/cache_read_comparison.py",
            "bench/cache_startup.py",
            "bench/startup.py",
            "bench/fixture_files.py",
            "bench/label_engine.py",
            "tests/reference.py",
        )
    }
    reader = module("startup_worker_descriptor", args.harness_root / "bench/cache_read_comparison.py")
    before = reader.descriptor(args.library_source, args.wheel, args.identity)
    from ultrafast_yolo_dataset import _native

    embedded = hashlib.sha256()
    for name in (
        "src/parser.rs",
        "src/snapshot.rs",
        "src/provenance.rs",
        "src/materialize.rs",
        "src/cache.rs",
        "src/lib.rs",
        "Cargo.toml",
        "Cargo.lock",
    ):
        embedded.update((args.library_source / name).read_bytes())
    assert _native.native_cache_profile() == embedded.hexdigest()
    caches = list(args.cache_root.glob("*.uydcache"))
    if args.mode == "hit":
        assert len(caches) == 1 and not caches[0].is_symlink()
        assert sha(caches[0]) == args.expected_cache_sha256
    else:
        assert not caches, "miss trials require a fresh private cache directory"
    sys.path.insert(0, str(args.harness_root / "bench"))
    startup = module("explicit_startup_worker", args.harness_root / "bench/cache_startup.py")
    original_options = startup.options

    def options(root, task, backend):
        assert backend == "native-content"
        values = original_options(root, task, backend)
        values["cache_dir"] = args.cache_root
        return values

    def clear_cache(root, backend):
        assert backend == "native-content" and args.mode == "miss"
        # A miss starts empty. Preserve every original/input cache and reject
        # accidental reuse instead of clearing pre-existing artifacts.
        assert not list(args.cache_root.glob("*.uydcache"))

    startup.options, startup.clear_cache = options, clear_cache
    result = startup.worker(args.corpus, "native-content", args.mode, False, scan_workers=args.workers)
    assert result["workers"] == args.workers and result["loader_workers"] == 0
    assert result["output_sha256"] == expected
    (cache,) = args.cache_root.glob("*.uydcache")
    cache_after = reader.inspect_container(cache)
    if args.mode == "hit":
        assert cache_after["sha256"] == args.expected_cache_sha256
    assert before == reader.descriptor(args.library_source, args.wheel, args.identity)
    assert harness_files == {name: sha(args.harness_root / name) for name in harness_files}
    state.update(
        complete=True,
        passed=True,
        workers=args.workers,
        mode=args.mode,
        descriptor=before,
        harness_sources=harness_files,
        expected_outputs_sha256=sha(args.expected_outputs),
        result=result,
        cache=cache_after,
        scope="Actual native-content constructor/cache-hit or rebuild plus first batch, using explicit public scan_workers. Full fixture fingerprints before/after and all ordered label/batch/diagnostic hashes checked. Same library binary; configuration experiment only. Process high-water RSS includes imports and preflight. Not a repeated comparison by itself.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "corpus",
        "harness-root",
        "library-source",
        "wheel",
        "identity",
        "cache-root",
        "expected-outputs",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--mode", choices=("hit", "miss"), required=True)
    parser.add_argument("--expected-cache-sha256")
    args = parser.parse_args()
    state = dict(
        complete=False, passed=False, pid=os.getpid(), script_sha256=sha(Path(__file__)), started_at_ns=time.time_ns()
    )
    with args.out.open("x") as stream:
        json.dump(state, stream)
    try:
        execute(args, state)
    except BaseException:
        state.update(complete=True, passed=False, error=traceback.format_exc())
        raise
    finally:
        state["finished_at_ns"] = time.time_ns()
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(args.out)


if __name__ == "__main__":
    main()
