# Benchmarks and validation limits — 2026-09-13

**The first section contains P1 label-only measurements. Pillow verification, image I/O, discovery, cache handling and training are excluded from those numbers.** Actual dataset startup and cache results are reported separately below; parser-only ratios are not full startup speedups.

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

Both backends in this section use weak legacy path/size hashes. These are **not** results for the content-validated native cache or its P4 target. Same-size edits can evade legacy validation. The native-cache implementation now captures the exact bytes parsed/verified; its separate measurements use equivalent content validation on hits.

### Reproduce and remaining gates

```sh
python bench/startup.py --prepare /path/to/new/corpus --count 100000 --task detect
python bench/startup.py --corpus /path/to/new/corpus --out bench/out/unique-miss.json
python bench/startup.py --corpus /path/to/new/corpus --mode hit --out bench/out/unique-hit.json
```

Each output stem has a separate `.runs` directory with per-process logs and JSON; use a new output stem for a rerun. Raw report JSON is committed, while the generated corpus and complete process logs are retained on the measurement hosts. The original failed harness attempt mixed framework stdout with JSON, returned no usable comparison, and was rerun with separate result files. No failed sample was selected into these summaries.

Source hashes in each JSON identify the measured implementation. Subsequent scan repair-bookkeeping, legacy GC-state concurrency protection, and lint-only changes are not part of those snapshots. All final performance claims require exact release-candidate reruns. Still required: meeting native-cache performance gates; representative Segmentation/real/500k-file initialization; higher DataLoader workers and training; the supported wheel/platform matrix and CI.

## Checkpoint validation

The earlier full 50-test suite passed on macOS and Linux after the repair-bookkeeping addition. A subsequent legacy GC-state concurrency fix passed the ten focused legacy tests on both hosts, including two new concurrent-read cases (52 unique tests across those scopes). The reader serializes its own calls around the upstream helper’s global GC toggles and resets that process-local lock after fork. Native-cache validation is recorded separately below. Historical XML artifacts and their hashes are committed under `docs/validation`. At that checkpoint Rust unit tests (3), Clippy with warnings denied, and Ruff passed.

A macOS arm64/CPython 3.12 wheel was built, its Python/profile files compared byte-for-byte with the source, and its unpacked module/extension imported and used for a hybrid scan in a fresh process. This reused the validated dependency environment; it is not a clean-install or cross-platform matrix result. See `docs/validation/wheel-smoke.json`.

## Native content cache: initial regression and compact-fingerprint revision

`bench/cache_startup.py` measures the actual constructor and first DataLoader batch
with five alternating fresh processes per backend. Hit caches are primed in
**separate untimed processes**; priming cannot inflate the measured worker's RSS.
Whole-fixture content hashes run before/after every worker outside timing,
deliberately pre-reading the OS cache. Imports are outside timing and both
implementations are imported in every process. These are shared-host measurements;
load/available RAM and all raw samples are retained, without selecting favorable runs.

For hits, `reference-content` is a benchmark adapter retaining the original
NumPy/pickle cache and final Python objects. It uses the **same Rust input hashing
engine** as the candidate, compares an ordered content-root key, and checks SHA-256
of the cache payload before trusted deserialization. It avoids redundant legacy
weak hashing and need not materialize per-file fingerprint dictionaries. Its cache
is primed on the immutable, content-verified fixture. This is an optimized auxiliary
baseline for equivalent-strength reads, not a general reference cache producer with
captured-byte race guarantees. Comparing only against an artificially slow Python
hashing loop would overstate the native-cache benefit.

The initial native schema stored per-file proof dictionaries and hexadecimal
digests in JSON. Schema 2 uses fixed-width binary tables and Rust comparison,
retaining SHA-256 checks and the same mutable output arrays. Full-label and
first-batch hashes match in every reported run.

| Host / input pairs / schema | Constructor reference-content → native | Reference/native ratio (95% paired bootstrap CI) | Constructor peak RSS reference → native | Native cache bytes |
|---|---:|---:|---:|---:|
| M2 / 1k / initial | 27.50 → 76.98 ms | 0.36× (0.350–0.363) | 238.25 → 241.91 MiB | 541,929 |
| M2 / 1k / compact | 26.69 → 63.58 ms | 0.42× (0.414–0.425) | 238.27 → 241.33 MiB | 381,488 |
| i5-10400 / 100k / initial | 2.38789 → 4.24701 s | 0.56× (0.561–0.563) | 524.70 → 593.00 MiB | 54,853,669 |
| i5-10400 / 100k / compact | 2.38580 → 2.67762 s | 0.89× (0.885–0.894) | 524.05 → 495.92 MiB | 38,801,500 |

The compact revision reduces the 100k candidate's initialization time by 37% and
peak RSS by 16% relative to its own initial implementation. Against the optimized
reference-content baseline, however, it remains **12% slower**, with only a
**5.4% whole-process RSS reduction**. Its cache is still larger than the reference
cache (25,572,718 bytes), because it additionally stores complete input provenance.
First-batch totals are 2.40133 → 2.69215 s for compact 100k; 38.73 → 75.79 ms for
compact M2 1k. The ≥1.5× content-hit target is **not met**.

The first 100k samples isolate the bottleneck change: native decode 0.766 → 0.078 s,
input validation 1.552 → 0.762 s; mutable Python label materialization remains about
0.52 s. These are observed phase timings, not separate speedup claims or an
extrapolation to training. Improving Python materialization and the remaining
framework initialization work is still necessary.

Raw reports: `bench/results/cache-hit-{m2,server}-v{1,2}.json`. The source archives
`cache-v1-source.tar.gz` and `cache-v2-source.tar.gz` preserve the exact measured
runtime and harness snapshots, including the unfavorable implementation. Every
source hash in the initial two-host reports was checked against its archive. Later
COCO support and additional diagnostic hashing in the harness are separate from
those synthetic snapshots.

```sh
python bench/cache_startup.py --corpus /path/to/synthetic/corpus --mode hit --out bench/out/new-hit.json
python bench/cache_startup.py --corpus /path/to/synthetic/corpus --mode miss --out bench/out/new-miss.json
```

The schema-2 full suite passed **112 tests on each of macOS and Linux**. It covers
captured native/fallback/Pillow bytes, same-size restored-mtime edits, repaired JPEG
digests, corruption with valid section checksums, cancellation/write failure,
relative paths, lock timeout, fork ownership, and three competing processes
producing one scan and two hits. Actual Detect/Segment datasets match cold/warm
labels and batches at workers 0/2. Rust unit tests (4), Clippy with warnings denied,
and Ruff passed on macOS. See the XML artifacts in `docs/validation`.

The current macOS CPython 3.12 wheel was also installed in a **new standalone
virtual environment**, with NumPy/Pillow and no Ultralytics/Torch. All 21 parser
tests passed. `bench/wheel_smoke.py` verified Detect/Segment cold/warm cache parity
against the frozen label oracle and the content-vs-metadata edit contract. Wheel
Python/profile bytes were checked against source. This is a verified clean runtime
installation on one platform; framework wheel installation, other Python versions,
Linux distribution wheels, Windows and CI remain open.

### Native-cache generation on the synthetic fixtures

These misses compare the original default legacy-cache producer against the
snapshot-backed native content-cache producer. Image verification and final labels
are equivalent, but the candidate additionally captures/hashes all input bytes and
revalidates them before publishing. The reference uses its ordinary weak cache key.
Five alternating processes per backend include serialization and the first batch.

| Host / pairs | Constructor reference → native | Ratio (95% paired bootstrap CI) | First batch total reference → native | Peak constructor RSS reference → native |
|---|---:|---:|---:|---:|
| M2 / 1k | 0.19600 → 0.16075 s | 1.22× (0.91–1.48) | 0.20809 → 0.17255 s | 239.59 → 242.38 MiB |
| i5-10400 / 100k | 28.13212 → 12.87712 s | 2.18× (2.17–2.20) | 28.14810 → 12.89101 s | 576.42 → 558.98 MiB |

The small M2 result is noisy and its interval includes regression. The larger
fixture shows a 3.0% process-RSS reduction, not a large whole-pipeline memory gain.
These measurements precede the later ARM SHA feature and dependency-profile update;
the exact sources are retained in the v2 archive. Raw reports are
`cache-miss-{m2,server}-v2.json`.

## Real COCO val2017: two hosts, both tasks

All 5,000 images and 4,952 converted TXT files are physical files; 36,335 rows were
written by the pinned upstream converter. Both tasks have 48 missing label files,
zero corrupt images and zero native label fallbacks in these measured scans.
Full labels, first batches, counters and ordered diagnostic messages match in every
run. Archive, per-image, annotation, converter and converted-fixture hashes plus
reproduction commands are in [COCO_FIXTURE.md](COCO_FIXTURE.md).

The image payload is much larger than the synthetic small-JPEG corpus. The first
M2 experiment exposed a real regression: native generation took **1.355 s for
Detection and 1.560 s for Segmentation**, versus 0.908 and 1.442 s respectively.
Those four miss/hit reports (`coco-*-m2.json`) and their exact source archive
(`cache-coco-software-source.tar.gz`) are retained.

Inspection of the locked sha2 0.10.9 source found that its ARM SHA implementation
requires the `asm` feature; the default aarch64 path was software. The final
revision enables that feature for aarch64 and retains runtime CPU dispatch. Both
the candidate and the reference-content auxiliary baseline receive the same hashing
acceleration. The x86 implementation is unchanged. SHA digests are checked against
Python hashlib across padding, 64 KiB read, and large-input boundaries, including
the container's stored section checksums.

Latest cache-generation results (actual constructor, no annotation cache):

| Host / task | Reference → native | Ratio (95% paired bootstrap CI) | First batch total reference → native | Peak constructor RSS reference → native |
|---|---:|---:|---:|---:|
| M2 / Detection | 0.90847 → 0.64431 s | 1.41× (1.38–1.53) | 0.93100 → 0.66811 s | 250.58 → 257.55 MiB |
| M2 / Segmentation | 1.38032 → 0.86435 s | 1.60× (1.58–1.63) | 1.40888 → 0.89242 s | 295.53 → 291.56 MiB |
| i5-10400 / Detection | 1.47351 → 1.53402 s | 0.96× (0.95–0.97) | 1.50586 → 1.56589 s | 302.57 → 315.00 MiB |
| i5-10400 / Segmentation | 2.17738 → 1.81340 s | 1.20× (1.20–1.23) | 2.21868 → 1.85402 s | 340.99 → 341.99 MiB |

Latest content-validated cache hits (separately primed processes):

| Host / task | Reference-content → native | Ratio (95% paired bootstrap CI) | First batch total reference → native | Peak constructor RSS reference → native |
|---|---:|---:|---:|---:|
| M2 / Detection | 0.19145 → 0.22836 s | 0.84× (0.82–0.85) | 0.21406 → 0.24975 s | 249.23 → 250.97 MiB |
| M2 / Segmentation | 0.22119 → 0.26744 s | 0.83× (0.82–0.84) | 0.25008 → 0.29629 s | 292.30 → 277.13 MiB |
| i5-10400 / Detection | 0.65934 → 0.70162 s | 0.94× (0.94–1.00) | 0.69153 → 0.73477 s | 300.04 → 299.25 MiB |
| i5-10400 / Segmentation | 0.73740 → 0.80651 s | 0.91× (0.91–0.92) | 0.78062 → 0.84889 s | 340.39 → 316.79 MiB |

These results establish useful cache-generation gains for three of the four
real-data host/task combinations, but **not a universal startup gain**. Server
Detection regresses. Content hits remain slower in every median; their memory
benefit is limited, with the largest reduction here about 6.9% on server
Segmentation. The 1.5× hit target and broad memory goals remain open. Final mutable
Python label materialization and hashing/discovery costs still need optimization.
There is no steady-state or GPU training-throughput claim.

Final full validation passed **128 tests per host**. The ARM wheel was installed in
another new standalone environment: 37 parser/hash tests and the four Detect/Segment
cache smoke cases passed. Its Python/profile payload matches source. Environment,
compiler and extension hashes are recorded in `docs/validation/cache-environments.json`;
both measured extensions' compiled source profiles match the current Rust sources
and Cargo manifest/lockfile. Current raw reports are `coco-*-m2-arm.json` and
`coco-*-server.json`. Supported wheel/CI, 500k-file and actual training gates are
still pending.
