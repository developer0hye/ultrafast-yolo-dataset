# 500k-pair startup experiment

Detection P1 has completed all five paired measurements; P3 is running and P4
remains queued. Actual full-scale startup is not yet verified. Fixture preparation and its
complete distinct-file preflight passed on M2. The fixture contains 500,000 JPEGs
and 500,000 TXT files totaling 3,625,093,750 bytes, with full name/content SHA-256
`931bd8059aefe560c3601e49d9b5f08ddc87831440bb0a1b4118575edca83f5d`.
The generator and independent preflight fingerprints match; all entries are
regular single-link files. Original logs/manifests are retained in
`validation/detect-500k-{fixture,preflight,prepare}-v1.*`. The same frozen runtime
and inputs continue sequentially through five-pair P3/P4 measurements.
The active host is an Apple M2 with eight CPU cores and 16 GiB RAM, on macOS
26.6.2. Inputs reside on the `/Volumes/T7` USB SSD using APFS and 4 KiB device
blocks. A read-only mid-run hardware/package observation is preserved in
`validation/detect-500k-host-supplement-v1.json`; it matches the sealed P1
checkpoint's native extension, Python, NumPy and CPU/RAM identity. It does not
measure USB link speed, drive firmware or thermal stability. Treat results as
specific to this shared host/storage condition, not a general SSD throughput
claim or a cold-cache benchmark.
Segmentation remains pending; its resampled annotations need a separate memory
headroom assessment on the 16 GiB M2 host. The scope is **500,000 image/label
pairs per task**, meaning 500,000 JPEG paths plus 500,000 TXT paths. Detection
and Segmentation are separate fixtures. These are synthetic scalability inputs;
the real COCO experiments remain separate evidence.

## Completed Detection P1 on M2

All ten fresh processes completed on 500,000 distinct label files, using four
workers and alternating backend order. The full packed-output hashes match.
The independent artifact audit accepted the fixture/source/native identity,
sample order/count, zero fallback and all five pairs. Its exact source, original
parent report and log are preserved under `validation/detect-500k-p1-*`.

| Metric, median of five processes per backend | Reference | Native | Comparison |
| --- | ---: | ---: | ---: |
| File read, parse, validate and equivalent packed-array export | 50.8404 s | 32.0845 s | 1.5846x; 36.9% less time |
| Process high-water RSS sampled after that timed region | 748.53 MiB | 482.23 MiB | 35.6% lower |

The paired percentile bootstrap for the latency ratio is [1.5613, 1.6201],
enumerating all 3,125 ordered five-pair resamples. This interval describes these
repetitions on this host/fixture, not uncertainty across hardware or datasets.
The source reference is the label-only extraction at pinned upstream commit
`795a556942a12fe0124cf767888194a1d0b83e2e`, followed by equivalent packed export.
Discovery, image verification, dataset/cache construction and training are
excluded. Full label-content verification runs before and after each timed
region, warming file caches; this is not a cold-cache measurement.

RSS includes imports, path lists, preflight and allocator history. It is sampled
before output hashing and post-measurement input verification, so it is neither
isolated working allocation nor the final process-exit peak. The P1 harness
embeds worker records in its parent JSON rather than retaining independent
child files; Python/NumPy/native/CPU/RAM parent metadata matches the mid-run host
supplement, but there is no per-worker package inventory. The independent
auditor ran successfully on the actual artifacts; its 24 adversarial tests are
still pending while the hosts are occupied.

See [raw report](validation/detect-500k-p1-complete-v1.json),
[audit and exact statistics](validation/detect-500k-p1-complete-audit-v1.json),
and [byte-preservation receipt](validation/detect-500k-p1-preservation-v1.json).
These results do not complete the actual startup or Segmentation gates. P3/P4
and a separate full-scale Segmentation workload remain required.

## First Detection P3 pair, incomplete series

The first actual constructor/cache-generation pair completed on all 500,000
image/label pairs with seven scan workers and zero loader workers. Full mutable
labels, first-batch output and ordered scan diagnostics match. Both found all
500,000 labels with zero missing/empty/corrupt entries and zero diagnostics;
native fallback was zero. The independent partial artifact audit passed.

| Metric, single process per backend | Reference | Native content cache |
| --- | ---: | ---: |
| Constructor | 137.4544 s | 126.2336 s |
| Process high-water RSS sampled at constructor completion | 1,752.16 MiB | 1,901.19 MiB |
| Cache bytes | 128,361,630 | 195,001,501 |

This pair shows a modest latency improvement but **8.5% higher constructor RSS**.
Four paired repetitions remain; these are provisional individual measurements,
not a repeated-performance summary. The P1 memory reduction does not imply
lower memory for actual cache generation.

Native instrumentation records 48.4948 s in `_validate_inputs`, 1.5079 s in
`_encode` and 1.2059 s in mutable label export. These hooks cover only part of
the constructor. Source review locates input revalidation after serialization,
immediately before atomic cache publication: both image and label content are
checked against captured fingerprints. This is part of the native cache's
stronger consistency policy, not removable benchmark overhead. The reference
uses its ordinary legacy cache. Optimize implementation costs without dropping
the required content checks; use complete results before prioritizing changes.

The [first-pair archive](../bench/results/detect-500k-p3-first-pair-v1.tar.gz)
contains exact parent/child reports, the partial audit and its source; the
[preservation receipt](validation/detect-500k-p3-first-pair-v1.json) binds all
bytes. The same frozen experiment continues. P3 repeats, P4 and full-scale
Segmentation remain incomplete.

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
missing/extra inputs and aliases. All nine passed on M2; the combined 21-test
report (including 12 GPU-artifact checks) and helper source hashes are retained
in `validation/benchmark-audits-m2.*`. Ruff and actionlint also passed.

A fresh actual-startup pilot completed with 1,003 pairs per task, spanning two
shards. Detection and Segmentation each passed P3 cache generation and P4 content
hits: eight measured fresh processes and four separate primers. All 1,003 labels,
first-batch hashes and ordered scan diagnostics matched; native fallback counts
were zero. Original child reports match their parent entries and the native
extension matches the previously audited notice wheel. The 36 raw reports/logs
and preflight records are retained in `bench/results/fixture-pilot-1003-m2-v1.tar.gz`,
with source/archive hashes in `validation/fixture-pilot-1003-m2-v1.json`.
This one-pair pilot includes slower native timings and is harness qualification,
not evidence for a speedup or the 500k-pair gate.

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
build, test or benchmark on that host during this series. The launched M2
Detection script, source hashes, wheel identity and initial headroom are recorded
in `validation/detect-500k-launch-v1.json`. The server GPU job keeps its existing
inputs and installed binaries. Segmentation has not been launched.

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
