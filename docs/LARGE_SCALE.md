# 500k-pair startup experiment

Detection P1, P3 and P4 completed all five paired measurements and their artifact
audits. [Segmentation P1](SEGMENT_500K_P1.md) and
[P3 constructor/cache generation](SEGMENT_500K_STARTUP.md) also completed five
pairs on Linux. [P4 content hits](SEGMENT_500K_HITS.md) also completed all five pairs.
Fixture preparation and its
complete distinct-file preflight passed on M2. The fixture contains 500,000 JPEGs
and 500,000 TXT files totaling 3,625,093,750 bytes, with full name/content SHA-256
`931bd8059aefe560c3601e49d9b5f08ddc87831440bb0a1b4118575edca83f5d`.
The generator and independent preflight fingerprints match; all entries are
regular single-link files. Original logs/manifests are retained in
`validation/detect-500k-{fixture,preflight,prepare}-v1.*`. The same frozen runtime
and inputs were retained through the five-pair P4 measurement.
The active host is an Apple M2 with eight CPU cores and 16 GiB RAM, on macOS
26.6.2. Inputs reside on the `/Volumes/T7` USB SSD using APFS and 4 KiB device
blocks. A read-only mid-run hardware/package observation is preserved in
`validation/detect-500k-host-supplement-v1.json`; it matches the sealed P1
checkpoint's native extension, Python, NumPy and CPU/RAM identity. It does not
measure USB link speed, drive firmware or thermal stability. Treat results as
specific to this shared host/storage condition, not a general SSD throughput
claim or a cold-cache benchmark.
Repeated Segmentation P3 constructor results are now complete at 2.4053× with
13.86% lower constructor process RSS. Content hits completed at 1.0412× with
40.09% lower constructor process RSS; their 1.5× speed target remains unmet. Its P1
label-engine comparison is also complete. The pinned constructor retains raw polygons;
resampling happens per sample later. The [capacity plan](SEGMENT_500K_PLAN.md)
estimates 0.612 GiB of packed numeric payload before Python/intermediate overhead,
not peak RSS. Full capacity qualification has passed on the 32 GB server with
identical reference/native outputs; the separate one-pair observations and
original records are in that document. The repeated P4 results have their own
[full report and archive](SEGMENT_500K_HITS.md).
The scope is **500,000 image/label
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
auditor ran successfully on the actual artifacts; its 24 adversarial tests
passed on Linux with zero failures, errors or skips. The
[test receipt](validation/startup-auditor-linux-tested-v1.json) binds the exact
auditor/test sources and retained JUnit/log bytes. Source snapshots are also
preserved as `validation/startup-auditor-{audit_startup,test_startup_audit}.py`.

See [raw report](validation/detect-500k-p1-complete-v1.json),
[audit and exact statistics](validation/detect-500k-p1-complete-audit-v1.json),
and [byte-preservation receipt](validation/detect-500k-p1-preservation-v1.json).
P1 alone does not establish actual startup or Segmentation results. The completed
Detection P3 and P4 results are below; completed Segmentation results are linked above.

## Historical first Detection P3 pair

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
Four paired repetitions remained at this checkpoint; these are individual
measurements, not the subsequent complete-series summary below. The P1 memory reduction does not imply
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
bytes. Detection and Segmentation P1/P3/P4 subsequently completed; see the full reports.

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
in `validation/detect-500k-launch-v1.json`. At that launch, the server GPU job kept
its existing inputs and installed binaries and Segmentation had not started.
The later [Linux Segmentation campaign](SEGMENT_500K_PLAN.md) passed full capacity
qualification and all repeated P1/P3/P4 phases, including their independent audits.

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
unchanged. Completed phase measurements do not satisfy the remaining release gates.


## Completed 500k Detection cache generation (P3)

All five paired fresh-process repetitions completed and passed the independent
artifact audit. All ten runs have identical complete labels, first batches,
counters and ordered diagnostics, with 500,000 found pairs and no native fallback.
Every one of the 25 recorded launch source hashes still matches the source tree.

| Measurement | Reference median | Native median | Reference/native | Exact paired 95% interval |
|---|---:|---:|---:|---:|
| Constructor including cache generation | 137.4544 s | 126.2336 s | 1.0889 | 1.0825–1.1302 |
| Constructor through first batch | 137.5533 s | 126.2515 s | 1.0895 | 1.0825–1.1301 |
| Constructor high-water RSS | 1752.34375 MiB | 1901.18750 MiB | 0.9217 | 0.9200–0.9230 |

Constructor latency is about 8.16% lower, while peak RSS is about 8.49% higher.
This does not reproduce the label-engine-only P1 memory reduction. The native
cache uses captured-content revalidation before atomic publication, whereas the
ordinary reference cache has a weaker invalidation contract; this is an actual
cache-generation comparison, not proof of equivalent cache consistency costs.
The previously preserved first pair measured 48.49 s of native save-time content
revalidation. Buffer lifetime is a candidate explanation to investigate, not a
proven cause of the RSS regression or evidence that an untested fix works.

Intervals enumerate all 3,125 ordered paired resamples; parent medians/p95 values
are checked independently. Constructor timing excludes fixture checks, imports
and post-constructor output hashing. High-water RSS still includes imports,
preflight and allocator history. Full file checks can warm the storage cache;
these are shared-host observations, not cold-storage or working-allocation claims.
The fixture uses distinct files with repeated synthetic content, not 500,000
unique natural images. P4 content-mode cache hits completed separately and
Segmentation P1/P3/P4 have also completed; their reports are linked above.

The [complete archive](../bench/results/detect-500k-p3-complete-v1.tar.gz) preserves
the ten raw reports, parent, log, exact auditor, launch/preflight/fixture identity,
and all 25 recorded source files. The
[preservation manifest](validation/detect-500k-p3-complete-v1.json) binds every
archived file and retains the complete independent statistics. The auditor passed
on these real artifacts; its 24 adversarial tests also passed on Linux.

## Completed 500k Detection content-cache hits (P4)

All five alternating fresh-process pairs and two separate untimed primers
completed. Every measured run has identical complete labels, first batches and
ordered diagnostics, also matching P3, with 500,000 found labels and zero native
fallback. Seven scan workers and zero loader workers match the frozen protocol.

| Measurement | Reference content median | Native content median | Reference/native | Exact paired 95% interval |
|---|---:|---:|---:|---:|
| Constructor | 56.5013 s | 56.1217 s | 1.0068 | 0.9615–1.0922 |
| Constructor through first batch | 56.5208 s | 56.1419 s | 1.0067 | 0.9614–1.0921 |
| Constructor high-water RSS | 1478.43750 MiB | 1270.984375 MiB | 1.1632 | 1.0496–1.2213 |

There is no clear latency improvement; the 1.5x content-hit target is unmet.
Constructor high-water RSS is 14.03% lower, while the native disk cache is larger:
195,001,501 versus 128,361,630 bytes. Both backends use the same Rust input
fingerprint engine, so this does not establish Python-versus-Rust hashing gains.
The timed constructor includes content validation, cache loading and mutable
label objects. The experimental reader/save-buffer changes were not installed
in either P3 or P4; these observations describe their baseline only.

Intervals enumerate all 3,125 ordered paired resamples. RSS includes imports,
preflight and allocator history up to constructor completion, not isolated
working allocations. Input checks warm the OS cache; storage is uncontrolled
and shared-host interference remains possible. These timings are neither
cold-storage measurements nor training-throughput results.

The [59-file archive](../bench/results/detect-500k-p4-complete-v1.tar.gz) preserves
all ten raw workers, two primers and their logs, parent report, exact auditor,
launch/input/host identities and all 25 launch source files. Archive SHA-256 is
`2b58ad6b1ac6cca74161d7d4d26ab1d0dbb71451bdd1bbe3069beddd9e070af3`.
The [preservation receipt](validation/detect-500k-p4-preservation-v1.json) records
byte-for-byte readback of every archive member. The
[independent audit](validation/detect-500k-p4-complete-audit-v1.json) retains exact
statistics and full output hashes. Fixture and cache payload files remain
external and are not included in this small evidence archive.
