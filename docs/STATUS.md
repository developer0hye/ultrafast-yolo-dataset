# Implementation status — 2026-09-12

Active development; original PRD goals remain unchanged. Not release-ready.

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
- Remaining: versioned content-validated native cache with actual-read provenance, repair tracking, corruption/concurrency/race tests; broader codecs/fuzz/platform tests; representative real/500k-file full startup and cache benchmarks; wheels/CI/distribution. No native content-cache or training-throughput gate is established.
