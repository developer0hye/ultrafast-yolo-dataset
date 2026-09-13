# Implementation status — 2026-09-13

Active development; original PRD goals remain unchanged. Not release-ready.

The [500k-pair experiment](LARGE_SCALE.md) now has a prepared directory-streamed
preflight and distinct-file audit. Its nine filesystem tests and the actual
1,003-pair Detection/Segmentation P3/P4 pilot passed on M2. The fresh 500,000-pair
Detection fixture and complete one-million-file preflight passed. Five-pair P1
finished with all ten packed-output hashes matching: median file read/parse/
validate/export time 50.8404 → 32.0845 s (1.5846x), and process peak RSS sampled
after that timed region 748.53 → 482.23 MiB. The actual-artifact audit passed;
its 24 adversarial tests passed on Linux. These are label-engine results, excluding
image verification and actual initialization. Detection P3 and P4 completed and
passed their five-pair artifact audits; full-scale Segmentation remains incomplete. Historical results
retain the older fingerprint implementation and their original memory scope.

All five P3 constructor/cache-generation pairs passed full-label, first-batch
and diagnostic parity. Median constructor time was 137.4544 → 126.2336 s
(1.0889x; exact paired 95% interval 1.0825–1.1302), while constructor high-water
RSS increased from 1,752.34 to 1,901.19 MiB (8.49%). The full archive preserves
all ten raw reports and the independent audit; see LARGE_SCALE.md. This result
does not validate the experimental cache-buffer lifetime changes.

All five P4 content-cache-hit pairs also passed full output parity. Constructor
time was 56.5013 → 56.1217 s (1.0068x; paired 95% interval 0.9615–1.0922), with
no clear speed improvement. Constructor high-water RSS was 1,478.44 → 1,270.98 MiB,
14.03% lower. Both sides use the same native input fingerprint engine; this is
not a Python-versus-Rust hashing comparison. The native cache is larger on disk
(195,001,501 versus 128,361,630 bytes). The 1.5x cache-hit target remains unmet.
The [Segmentation capacity plan](SEGMENT_500K_PLAN.md) corrects the earlier
assumption that all polygons are resampled at constructor time; actual full-scale
memory use and timings still require measurement on the 32 GB server.

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
- The original installed Linux wheels completed the 45-job / 90-epoch RTX 3070 overlap comparison: all 15 groups have identical loss traces and final model hashes. Epoch median ratios are only 1.002–1.021x and process-family RSS does not consistently improve. Full raw/audit evidence and uncertainty are retained in the maskops repository, docs/GPU_RESULTS.md; this is not later-candidate or accuracy evidence.
- Remaining: cache-hit speed and broader process-memory improvements; 500k-file coverage; broader codecs/fuzz/platform tests; actual training throughput; dependency notice bundling and supported wheels/CI/distribution. Neither library is release-ready; the original goals remain active.
