# 500k-pair startup experiment preparation

No 500k-pair measurement is completed yet. The scope is **500,000 image/label
pairs per task**, meaning 500,000 JPEG paths plus 500,000 TXT paths. Detection
and Segmentation are separate fixtures. These are synthetic scalability inputs;
the real COCO experiments remain separate evidence.

## Fixture and preflight

The existing generator uses 16 solid-color JPEG templates and 1,000 seeded label
templates, written as separate files into 1,000-file shards. Content is allowed
to repeat; a million references to a smaller set of files is not a million-file
experiment. `bench/fixture_files.py` verifies the complete expected path set,
regular single-link files, label manifest and full name/content hashes. It
rejects missing/extra files, symlinks and hard links. It does not claim unique
SSD blocks or validate JPEG decoding; the actual Pillow verifier remains in the
measured dataset pipeline.

The old fingerprint implementation retained sorted Path lists for the whole
fixture before measuring a constructor. The prepared replacement traverses in
the same component-wise sorted order, retaining directory-sized lists. Memory
still depends on the largest directory and recursion depth; the fixed sharding
is what bounds this preparation for the large synthetic suite. Full file reads
remain outside timing and warm the OS cache. Existing historical RSS results
include their original preflight behavior and must not be silently relabeled.

Nine filesystem cases are prepared in `tests/test_benchmark_fixture.py` for
hash/order compatibility, shard boundaries, ignored cache files, corruption,
missing/extra inputs and aliases. They have not run yet while both hosts execute
other experiments. Ruff and actionlint passed; that is not execution evidence.

Run preparation, preflight and every measurement sequentially on an otherwise
unoccupied benchmark host. Start with a small fixture to validate the revised
harness before preparing the complete workload. A small run is not fulfillment
of the 500k-pair gate. Use new paths and preserve failed/incomplete reports.

```sh
# Choose an unused directory on the measurement SSD; examples use Detection.
python bench/startup.py --prepare /measurement/new-detect-500k --count 500000 --task detect
python bench/fixture_files.py --corpus /measurement/new-detect-500k --pairs 500000 --out /measurement/new-detect-500k-preflight.json

# P1: label file read/parse/validate/export, no image verification.
python bench/label_engine.py --corpus /measurement/new-detect-500k/labels --workers 4 --rounds 5 --out /measurement/new-detect-500k-p1.json

# P3: actual constructor with cache generation, then first batch.
python bench/cache_startup.py --corpus /measurement/new-detect-500k --mode miss --backends reference native-content --rounds 5 --out /measurement/new-detect-500k-p3.json

# P4: content validation + cache payload validation + mutable Python objects.
python bench/cache_startup.py --corpus /measurement/new-detect-500k --mode hit --backends reference-content native-content --rounds 5 --out /measurement/new-detect-500k-p4.json
```

Repeat with a distinct `segment` fixture and `--task segment`. Do not run another
build, test or benchmark on that host during this series. The current M2 and
server jobs keep their existing inputs and installed binaries; none of these
large-fixture commands has been launched yet.

## Comparison and evidence requirements

- Keep five alternating fresh-process pairs for each task/phase. Cache-hit
  priming remains in separate untimed processes. Never pool priming with hits.
- Preserve every raw result and log. Require full packed-label parity for P1,
  and full mutable labels, first batch, counters and ordered diagnostics for
  actual startup. Confirm the complete requested sample count before reporting
  summaries; older harnesses do not all carry a `complete` flag.
- Record the actual installed extension hash, source inventory, dependencies,
  CPU/RAM, storage, available RAM and load. The revised P1/startup source inventory
  includes all benchmark helper files, Rust/build inputs and the frozen oracle.
  A source inventory is not by itself proof that a native binary was built from
  those files; retain the wheel/build provenance separately.
- Report constructor and first-batch time separately, with paired uncertainty.
  P1 throughput is not full startup or training throughput. P4 uses the existing
  optimized reference-content adapter with the same native fingerprint engine;
  weaker metadata-only hits cannot satisfy the content-mode target.
- Report process RSS and its pre-constructor high-water mark. The helper change
  reduces an avoidable path-list allocation, but does not remove interpreter,
  import, allocator or input-verification effects from process RSS. No memory
  reduction is claimed until fresh measurements establish it.
- Keep shared-host observations and failures. If Segmentation exceeds a host's
  memory headroom, retain that limitation and use the larger host for the full
  case; do not extrapolate a 500k result from a smaller successful subset.

The performance/memory targets and release gates in the original PRD remain
unchanged. This document and the revised preflight are preparation, not a pass.
