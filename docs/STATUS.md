# Implementation status — 2026-09-13

Active development; original PRD goals remain unchanged. Not release-ready.

This worktree evaluates input-validation worker settings using the previously
qualified cache-reader wheel (187 installed tests). The complete 21-process M2
experiment and independent audit passed: 16 workers reduced the content-check
phase from a 4-worker median of 58.28 s to 37.38 s, with higher CPU time and about
1.45 MiB more sampled RSS. The 7-worker median was 43.83 s. No default has changed;
full-size startup confirmation is pending. The auditor passed 22 tests, and six
actual 1,003-pair startup pilots at workers 4/7/16 passed cache rebuild/hit,
labels/batch/diagnostics and exact cache-byte checks. See
[VALIDATION_WORKERS_RESULTS.md](VALIDATION_WORKERS_RESULTS.md).

The source tree retains the unpromoted scratch-buffer candidate inherited from
the preceding branch, while this worker experiment uses the earlier reader
wheel without that candidate. Its previous full 500,000-pair startup comparison
did not establish a scratch-buffer speed or RSS gain. See
[SNAPSHOT_FULL_M2_RESULTS.md](SNAPSHOT_FULL_M2_RESULTS.md). Historical measurements
remain attached to their exact frozen binary/source identities.

The [Rust notice bundle](RUST_DEPENDENCIES.md) now contains 136 files plus its
manifest for 54 selected registry crates and the observed standard-library
versions. A fresh M2 wheel/sdist preserves all 137 files including the manifest;
installed-byte auditing, 65 core tests and four standalone cache cases passed.
The 12-job platform/Python [wheel CI workflow](WHEEL_CI.md) is prepared and
locally linted, with hosted execution awaiting the repository visibility choice.

- Rust Detection/Segmentation parser, ordered batched file reads and packed results implemented.
- Hybrid Pillow scan, explicit JPEG repair policy, reference diagnostic fallback, immutable packed arrays and mutable label export implemented.
- 34 parser/scan tests (10,000 seeded cases included) passed on macOS and Linux; 3 Rust unit tests passed on macOS. JPEG repair bytes and negative-zero segment reductions match the reference.
- Base Detect/Segment YOLODataset adapter and explicit trainer builder implemented, with source/profile/runtime-hook guards. Actual labels and collated first batches match at workers 0/2.
- Optional trusted legacy cache bridge exports caches readable by the original YOLODataset. Config/version/schema/corrupt-cache checks, atomic replacement and failed-write recovery are tested. This bridge uses the reference's weak path/size hash, NOT content validation.
- The full 50-test suite passed on macOS and Linux. Subsequent GC-state concurrency protection passed all 10 focused legacy tests on both hosts, including two new concurrent-read cases (52 unique tests across these scopes). Exact XML artifacts are committed.
- 100k distinct-file Detection/Segmentation P1 results are recorded in [BENCHMARKS.md](BENCHMARKS.md). They exclude image verification and are not startup/training speedups.
- Actual Detection startup: M2 1k pairs 0.228 → 0.168 s; server 100k pairs 28.135 → 15.006 s, five paired process runs. Process RSS is approximately equal. M2 1k legacy-cache hits regress 11.98 → 25.84 ms; both favorable and unfavorable results are retained.
- Successful JPEG repair indices are retained even when label diagnostics overwrite the repair message; ordered label paths and prefix are also preserved. These fields are not content-snapshot proof.
- macOS CPython 3.12 wheel built; packaged runtime/profile bytes and a fresh-process import/scan smoke were checked with existing dependencies. Clean installation and the supported wheel matrix remain open.
- Native cache implemented: captured image/label bytes (including reference fallback), repaired-byte proof, content/metadata policies, checksummed owned arrays, source/dependency/CPU/plugin profile guards, atomic publication and process-owned locks. Schema 2 uses fixed-width binary fingerprints validated directly in Rust. See [CACHE.md](CACHE.md).
- Cache checkpoint: 128 tests passed on macOS and Linux, including native Detect/Segment cold/warm batches at workers 0/2, corruption/race/fork/concurrent-writer cases, and hashlib comparisons at SHA padding and read boundaries. Four Rust unit tests, Clippy and Ruff passed on macOS.
- Synthetic 100k Detection native-cache generation: server 28.132 → 12.877 s (2.18×), RSS 576.4 → 559.0 MiB. Content-hit initialization still regresses: 2.386 → 2.678 s, despite a 5.4% RSS reduction. Initial slower implementations and exact source archives are retained.
- All 5,000 real COCO val2017 images were converted with the pinned upstream converter, with frozen archive/image/annotation/label hashes. Actual constructor, first batch, counters and diagnostics are compared on both hosts. ARM SHA acceleration fixes a real-data regression: M2 native-cache generation now improves Detection 0.908 → 0.644 s and Segmentation 1.380 → 0.864 s. Server COCO Detection still regresses, and content-hit targets remain unmet; see [BENCHMARKS.md](BENCHMARKS.md).
- Cache-checkpoint macOS CPython 3.12 wheel installed in a new standalone environment: 37 parser/hash tests plus Detect/Segment cold/warm cache smoke passed. Runtime/profile bytes match source.
- Native mutable label export now builds independent NumPy-owned arrays in Rust, with explicit layout/alignment/offset validation and signal handling. Cache requests normalize their ordered paths once and reject relative-path rebinding while waiting for a lock. See [MATERIALIZATION.md](MATERIALIZATION.md).
- Latest full suite: **157 tests passed on each host**. The current ARM wheel passed **65 parser/hash/materialization tests** and four Detect/Segment cache smoke cases in a new standalone environment; its extension is byte-identical to the M2 benchmark extension. Four Rust unit tests, Clippy and Ruff passed. The supported framework/platform wheel matrix remains open.
- Latest 100k-file content-hit constructor: reference 2.412 s versus native 2.385 s; peak process RSS 523.39 versus 494.83 MiB. This largely removes the previous startup regression but does not meet the 1.5× target. Full paired results, including slower real-COCO hits, are in [BENCHMARKS.md](BENCHMARKS.md).
- Linux CUDA-environment installed wheel passed all 157 tests. A later archive audit rejected six packaged bytecode files; explicit exclusions fixed the fresh v2 wheel, whose runtime source/native bytes match the tested installation. New standalone Linux installation passed all four Detect/Segment cache smoke cases and RECORD/installed-byte auditing; see [LINUX_WHEEL_VALIDATION.md](LINUX_WHEEL_VALIDATION.md).
- Combined FastYOLODataset + FastFormat completed one full real-data RTX 3070 YOLO11n-seg training pilot: 5,000 images, finite losses and changed weights. This is integration evidence, not a repeated throughput comparison or an accuracy result; detailed evidence is retained in the maskops repository.
- Remaining: cache-hit speed and broader process-memory improvements; 500k-file coverage; broader codecs/fuzz/platform tests; actual training throughput; dependency notice bundling and supported wheels/CI/distribution. Neither library is release-ready; the original goals remain active.
