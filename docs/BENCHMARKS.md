# Prototype label-engine benchmarks — 2026-09-12

**P1 label-only measurements. Pillow verification, image I/O, discovery, cache handling and training are excluded. These numbers are not full dataset startup speedups.** The hybrid Pillow scan now has correctness tests but no P2/P3/P4 performance evidence yet.

Reference: extracted label logic from Ultralytics `795a556942a12fe0124cf767888194a1d0b83e2e`, NumPy 2.4.4. Both backends materialize the same packed arrays. Native export copies are timed. Five alternating independent process runs, four-worker budget. Each corpus has 100,000 distinct physical TXT files, with 1/10/100 objects at 80/19/1% frequencies; segment vertex counts cycle through 3/16/100. This is a synthetic file corpus, not replay of a short path list.

Host: Intel i5-10400 (6 cores/12 threads), 31.24 GiB visible RAM, local WD Blue SN550 NVMe SSD; RTX 3070 is unused in this CPU benchmark. Corpus hashes and all samples are in `bench/results`. Full input hashes are checked before/after the suite, pre-reading the corpus; no cold OS cache claim is made. Shared-host load is retained in raw samples.

| Corpus | Reference median | Native median | Speedup | Reference → native peak RSS |
|---|---:|---:|---:|---:|
| detect | 13.382 s | 0.339 s | 39.43× | 182.5 → 117.9 MiB |
| segment | 23.032 s | 1.047 s | 22.00× | 537.7 → 350.1 MiB |

Every result passed packed-output hash parity with zero label fallbacks. Peak RSS uses Linux VmHWM; hashing allocations occur after memory sampling. Single-worker historical samples are also retained as `*-100k-server-w1.json`; they are separate source/environment snapshots and are not averaged into the table.

34 tests passed on macOS and Linux: 21 parser tests including 10,000 seeded cases, plus 13 full Pillow/Ultralytics scan tests. These cover counters, diagnostics, missing/corrupt inputs, EXIF rotations, byte-identical JPEG repair and signed-zero segment fallback. See `validation-checkpoint.json` for result artifact hashes.

The P1 results above do not establish full startup speedups. The following separately measured adapter results include the full initialization path. Native content-cache, 500k-file, real-corpus and training gates remain open.

## Actual YOLODataset startup and first batch

The opt-in FastYOLODataset adapter and the original YOLODataset construct the same mutable label dictionaries, run the same discovery/image checks and create a legacy `.cache`. Each process starts without that annotation cache. Imports are outside timing; both implementations are imported in every process. Five alternating fresh-process runs per backend. The timer includes dataset discovery, path/size hashing, Pillow verification, label parsing and validation, Python materialization, cache serialization and transforms; a second timer extends through the first DataLoader batch (batch size 8, loader workers 0). Every full-label and first-batch hash matches.

The corpus contains distinct JPEG and TXT files. Sixteen solid-color JPEG templates cycle through 320×240, 640×480 and 1280×720, while labels use the P1 object-count distribution. This is an actual filesystem/framework workload but **a synthetic corpus, not a real training dataset**. Full file content checks before/after each run are outside timing and pre-read the OS cache. Repair was unnecessary in this corpus. The image verifier is the actual Pillow path, not the P1 stub.

| Host / files | Workers | Initialization reference → adapter | Speedup (95% paired bootstrap CI) | First batch total reference → adapter | Peak constructor process RSS reference → adapter |
|---|---:|---:|---:|---:|---:|
| M2 / 1,000 pairs | 7 | 0.22756 → 0.16841 s | 1.35× (1.34–1.43) | 0.23843 → 0.17925 s | 236.3 → 239.0 MiB |
| i5-10400 / 100,000 pairs | 8 | 28.13474 → 15.00623 s | 1.87× (1.87–1.90) | 28.15084 → 15.02005 s | 576.4 → 578.3 MiB |

The M2 has 16 GiB RAM and the generated corpus is on the external T7 SSD. The server corpus is on its local NVMe SSD. The worker count follows each host’s unmodified Ultralytics NUM_THREADS (7 on M2, 8 on the server); within each host the native scan uses the identical worker budget. P1 used 4 workers, so P1 and these startup ratios are not interchangeable.

**No process-memory improvement is demonstrated here.** The full pipeline retains the same Python label dictionaries; its process RSS is approximately equal, with a small increase for the adapter. The 35% P1 packed-engine RSS reduction must not be applied to these startup numbers. The 1.25× cache-miss feasibility target is exceeded on these Detection fixtures, but the representative Detect/Segment/real-data release gate is not proved.

### Legacy cache hits: regression retained

On M2 with 1,000 pairs, the original constructor takes a median **11.98 ms** and the adapter **25.84 ms** (0.46×). First batch total is **24.46 ms → 37.79 ms** (0.65×). Native integration verifies its source profile and checks loaded legacy records/configuration; the net added work is visible in these measurements. This result is retained in `startup-m2-detect-1k-hit.json`, rather than presented as an improvement. Cache priming occurs outside the timer but in the same process in this harness: hit-run RSS therefore includes priming and is **not load-only memory evidence**.

Both backends in this section use weak legacy path/size hashes. These are **not** results for the proposed content-validated native cache or its P4 target. The new cache must include provenance for the exact bytes parsed/verified and must be compared to a baseline with equivalent content validation. Same-size edits can evade legacy validation; stronger native-cache behavior is still required.

### Reproduce and remaining gates

```sh
python bench/startup.py --prepare /path/to/new/corpus --count 100000 --task detect
python bench/startup.py --corpus /path/to/new/corpus --out bench/out/unique-miss.json
python bench/startup.py --corpus /path/to/new/corpus --mode hit --out bench/out/unique-hit.json
```

Each output stem has a separate `.runs` directory with per-process logs and JSON; use a new output stem for a rerun. Raw report JSON is committed, while the generated corpus and complete process logs are retained on the measurement hosts. The original failed harness attempt mixed framework stdout with JSON, returned no usable comparison, and was rerun with separate result files. No failed sample was selected into these summaries.

Source hashes in each JSON identify the measured implementation. Subsequent scan repair-bookkeeping, legacy GC-state concurrency protection, and lint-only changes are not part of those snapshots. All final performance claims require exact release-candidate reruns. Still required: native content cache, its race/corruption/concurrency and equivalent-validation benchmarks; representative Segmentation/real/500k-file initialization; higher DataLoader workers and training; clean cross-platform wheels and CI.

## Checkpoint validation

The full 50-test suite passed on macOS and Linux after the repair-bookkeeping addition. A subsequent legacy GC-state concurrency fix passed the ten focused legacy tests on both hosts, including two new concurrent-read cases (52 unique tests across those scopes). The reader serializes its own calls around the upstream helper’s global GC toggles and resets that process-local lock after fork. Cross-process native-cache concurrency remains unimplemented. XML artifacts and their hashes are committed under `docs/validation`. Rust unit tests (3), Clippy with warnings denied, and Ruff passed.

A macOS arm64/CPython 3.12 wheel was built, its Python/profile files compared byte-for-byte with the source, and its unpacked module/extension imported and used for a hybrid scan in a fresh process. This reused the validated dependency environment; it is not a clean-install or cross-platform matrix result. See `docs/validation/wheel-smoke.json`.
