# ultrafast-yolo-dataset — Initial PRD

- Document version: 0.1 / Date: 2026-09-12
- Status: Proposal for starting implementation. Implementation, benchmarks, and package publishing have not been performed.
- Distribution name: `ultrafast-yolo-dataset` / Import name: `ultrafast_yolo_dataset`
- Proposed initial stack: Rust + PyO3 + maturin, bounded worker pool, NumPy output
- Primary users: Large YOLO dataset maintainers, training pipeline developers, and users running repeated experiments
- Reference source: Ultralytics commit `795a556942a12fe0124cf767888194a1d0b83e2e`.

## 1. Product definition

A Python extension that reads and validates large collections of YOLO annotation files to produce training-ready metadata and caches. It batches TXT parsing, validation, deduplication, and annotation packing in Rust while deferring Python object construction until required.

The primary performance targets are the initial dataset scan, rebuilding after a cache miss, and cache loading across repeated runs. It does not directly accelerate training-time image decoding or GPU computation. Adopt native loading, compact representations, explicit parity, and publicly reproducible benchmarks from `ultrafast-pycocotools`. [Project README](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/README.md)

## 2. Goals and non-goals

| ID | Goal | Proposed success criteria |
|---|---|---|
| G1 | Compatible Detection/Instance Segmentation annotations | Identical numeric bytes, shapes, ordering, statuses, and aggregates within supported profiles |
| G2 | Reduce Python parsing costs | ≥2× speedup for batch read+parse+validate on a 100k-file label suite |
| G3 | Improve actual initialization | ≥1.25× speedup for cache-miss preparation with equivalent image verification |
| G4 | Improve memory efficiency | ≥25% reduction in peak RSS on a 100k-file packed-result suite |
| G5 | Make warm caches useful | ≥1.5× speedup when producing the same final Python label objects |
| G6 | Reliable distribution | Clean installation of supported wheels and passing cache corruption/concurrency tests |

These numbers are unmeasured product targets. Freeze the environment and workloads in M0. If G3 is not met, do not present parser-only gains as complete startup gains. Record a PRD revision and measurement evidence when changing a gate.

### v0.1 scope

- YOLO TXT parsing/validation for Detection and Instance Segmentation.
- Batch `scan` accepting image/label path lists already discovered in Python.
- A hybrid pipeline combining reference image verification through Pillow with a Rust label engine.
- Compact native results, explicit Python materialization, and a verifiable versioned native cache.
- An opt-in adapter for the pinned Ultralytics base `YOLODataset`.
- Semantic interoperability with reference `.cache` dictionaries and a trusted legacy cache bridge.

### Non-goals

- Replacing PyTorch DataLoader, samplers, or distributed trainers; accelerating image decoding/augmentation.
- CUDA, model Inference, or COCO metric computation.
- Executing YAML download scripts, downloading remote datasets, or handling arbitrary URIs.
- Implementing all COCO/LVIS/Objects365/Grounding JSON parsers in v0.1.
- Native support for Pose, OBB, classification, semantic-mask, or depth datasets.
- Reimplementing every image codec in Rust or treating metadata-header parsing as complete verification.
- Automatic dataset modification, silent label clipping, or implicit class remapping.
- A mandatory dependency on `ultrafast-maskops`. Both projects must remain independently installable and verifiable.

## 3. Target Ultralytics hot paths

| Priority | Path | Intervention point |
|---|---|---|
| P0 | `data/utils.py::verify_image_label` | File read → tokenization → float32 → checks → dedup → arrays |
| P0 | `data/dataset.py::YOLODataset.cache_labels` | Replace per-file Python results with a batch native result |
| P0 | `YOLODataset.get_labels`, `_load_or_scan_cache` | Cache miss/hit, materialization, and lifecycle |
| P0 | `load_dataset_cache_file`, `save_dataset_cache_file` | Legacy bridge and separate native cache |
| P0 preservation | `check_image`, `exif_size` | Preserve Pillow verification, shapes, and JPEG handling |
| P1 | `data/base.py::BaseDataset.get_img_files` | Directory/list discovery and ordering |
| P1 | `img2label_paths`, `get_hash` | Path mapping and cache validation costs |
| P1/P2 | `utils/ops.py::segments2boxes` | Integrate float32 annotation geometry into the parser |

The current dataset exposes `verify_args`, `result_to_label`, `get_cache_hash`, and `scan_summary` hooks. The adapter must check whether it is handling the base YOLODataset and whether those hooks have been overridden. Do not replace the entire scan skeleton without understanding a subclass's semantics. [Pinned dataset source](https://github.com/ultralytics/ultralytics/blob/795a556942a12fe0124cf767888194a1d0b83e2e/ultralytics/data/dataset.py)

M0 profiling must separate discovery, filesystem stat, image verification, TXT reading, parsing/validation, Python materialization, hashing, and cache writing. If per-file open/stat costs are storage-bound, native parser acceleration will have limited impact on total time.

## 4. Compatibility and parity contract

### 4.1 Reference profile and modes

Record the Ultralytics SHA, Python, NumPy, Pillow and codec versions, OS/path semantics, and parser mode in `compatibility.json`. Freeze the first executable combination in M0. Exact numeric parity means equality of finite float32 arrays and metadata within the same profile.

| Surface | v0.1 contract |
|---|---|
| Native label parsing | Exact reference parity for supported Detection/Segmentation inputs |
| `scan(..., image_validation="pillow")` | Equivalent reference verification for healthy images; explicit JPEG repair policy as specified below |
| Ultralytics adapter | Enabled only for supported profile, base-class, and feature combinations |
| Native cache | Logical data parity with a separate format/version/invalidation contract |
| Legacy `.cache` | Python dictionary semantics; serialized file byte identity is not required |
| Unsupported tasks/subclasses | Fall back to the complete reference path or raise an explicit unsupported error |

Do not arbitrarily tighten permissive reference checks. A separate `strict` validator is a future feature. Do not add conditions absent from the source, such as requiring integer-valued classes, to the exact parser.

### 4.2 Annotation requirements

| ID | Behavior to preserve | Required fixtures |
|---|---|---|
| C01 | UTF-8 decoding, line/whitespace handling, and float32 conversion | LF/CRLF, tabs, blank lines, whitespace-only rows, scientific notation |
| C02 | Detection output contains class+xywh with shape `(n,5)` | Zero, one, and many rows; invalid column counts |
| C03 | Reference segment detection and conversion | Rows with more than six fields, mixed detection/segment rows, odd coordinate counts |
| C04 | Identical segment-to-bbox operations and rounding | Float32 extrema and xyxy→xywh conversion boundaries |
| C05 | Preserve the 1% tolerance in coordinate checks | -0.01, 0, 1, 1.01 and adjacent float32 values |
| C06 | Class upper-bound and `single_cls` check semantics | Maximum class, fractional classes, `single_cls=True` |
| C07 | Dedup results and conditional row ordering | Unsorted rows without duplicates and unsorted rows with duplicates |
| C08 | Segment and bbox/class index alignment | Different polygons with the same bbox |
| C09 | Missing/empty/corrupt distinctions and counter accumulation | Missing labels, empty labels, corrupt images, invalid labels |
| C10 | Output shape/dtype/None/list semantics | Empty cls `(0,1)`, boxes `(0,4)`, absent keypoints |
| C11 | Dataset-level mixed annotation handling | task=segment mismatches and mixed detect datasets |
| C12 | Input path and output record ordering | Relative paths, spaces/Unicode, duplicate paths, failed files |

Deduplication is not equivalent to a simple insertion-ordered hash set. When duplicates exist, the reference selects rows and segments in the index order returned by unique; without duplicates, it preserves the original order. Test canonical row keys, first occurrences, and final ordering separately in native deduplication.

Do not assume that the Segmentation dedup key contains all polygon vertices. Freeze cases where duplicates are determined from label rows after bbox conversion. Separate parser-level `single_cls` validation from subsequent dataset-level class remapping.

Match the source profile even in deciding whether checks apply to all original polygon coordinates or to converted bboxes. Do not clip coordinates outside the 1% tolerance to make labels valid. The references are the [verification function](https://github.com/ultralytics/ultralytics/blob/795a556942a12fe0124cf767888194a1d0b83e2e/ultralytics/data/utils.py#L306) and [geometry implementation](https://github.com/ultralytics/ultralytics/blob/795a556942a12fe0124cf767888194a1d0b83e2e/ultralytics/utils/ops.py#L451).

### 4.3 Errors and side effects

- Convert invalid labels into the same excluded image/label pairs and corrupt counters as the reference. Invalid call arguments raise `ValueError`.
- Native records store error codes, path indices, line/column information, and bounded payloads; the Python layer renders messages.
- Reference counters and record acceptance/rejection are exact gates. Compare duplicate and validation messages within the pinned profile as well.
- Use semantic parity for OS-originated errors by normalizing prefix/path/reason categories. Progress-bar refresh timing and ANSI formatting are outside the contract.
- Investigate NaN/Inf, malformed UTF-8, unusual numeric values, and syntax treated specially by Python/NumPy through the differential corpus. When equivalent native behavior cannot be demonstrated, fall back to the reference label path and report the count.
- Distinguish record-level input errors from native panics, OOM, and cache failures. Do not disguise native implementation failures as corrupt labels.

### 4.4 Image verification and JPEG repair

v0.1 retains pinned reference/Pillow verification. A Rust header parser must not replace it. Preserve EXIF-dependent `(height,width)`, accepted formats, small-image handling, and actual verification failures.

The reference can repair certain JPEGs and save changes to the original file. Standalone `scan` defaults to `repair_jpeg="reject"` and reports those files through structured repair-required results. This intentionally differs from the reference for damaged JPEGs and must not be described as fully drop-in behavior.

Adapters requiring the full reference side effects must require an explicit user configuration of `repair_jpeg="reference"`. In that mode, execute the same reference repair and compare output metadata, messages, and modified file hashes. Compute fingerprints of repaired files after repair. Benchmarks must use identical input copies so that the first backend does not change the other backend's input.

### 4.5 Discovery contract

The v0.1 adapter uses existing Ultralytics image discovery and fraction selection. Do not reorder, deduplicate, or resolve paths passed to `scan`. Mismatched lengths of the two input lists are an error.

When native discovery is introduced, match OS separators, extension case handling, recursive globbing, image-list `./` interpretation, sorted ordering, and integer/fraction slicing. Freeze symlink and permission-failure policies as well. [Pinned BaseDataset implementation](https://github.com/ultralytics/ultralytics/blob/795a556942a12fe0124cf767888194a1d0b83e2e/ultralytics/data/base.py#L168)

## 5. Proposed Python API

The following APIs are design proposals, not an already installable implementation.

```python
import ultrafast_yolo_dataset as uyd

result = uyd.scan(
    image_paths,
    label_paths,
    num_classes=80,
    task="segment",                 # v0.1: detect | segment
    single_cls=False,
    image_validation="pillow",      # v0.1 production mode
    repair_jpeg="reject",           # reject | reference
    workers=4,
    max_in_flight=256,
    prefix="train: ",
)

print(result.summary)
print(result.diagnostics)
labels = result.to_ultralytics_labels()  # mutable, owned arrays
result.save_cache("train.uydcache", fingerprint="content")

cached = uyd.load_cache(
    "train.uydcache",
    image_paths=image_paths,
    label_paths=label_paths,
    num_classes=80,
    task="segment",
    single_cls=False,
    image_validation="pillow",
    repair_jpeg="reject",
    fingerprint="content",
)
labels = cached.to_ultralytics_labels()
```

### 5.1 `ScanResult`

| Field/method | Contract |
|---|---|
| `num_inputs` / `num_valid` | Original pair count / valid record count |
| `source_indices` | Valid record → original input index, int64 |
| `image_shapes` | int64 `[num_valid,2]`, `(height,width)` |
| `object_offsets` | int64 `[num_valid+1]`, object ranges per record |
| `classes` / `boxes` | float32 `[M,1]` / `[M,4]`, normalized xywh |
| `segment_points` / `segment_offsets` | float32 `[P,2]` / int64 `[S+1]` |
| `segment_object_indices` | int64 `[S]`, segment-to-object mapping |
| `summary` | Found/missing/empty/corrupt/total; reference counter accumulation |
| `diagnostics` | Code/path/line/message in input order, plus fallback statistics |
| `to_ultralytics_labels()` | Materialization of a reference-format list[dict] |
| `save_cache(...)` | Atomic native cache write with fingerprint validation |

The field names above are a public schema draft. Native views are read-only; `ScanResult` or the ndarray base owner retains storage lifetime. By default, `to_ultralytics_labels()` returns writable copies so augmentation cannot contaminate the cache or other samples. Dictionary keys are `im_file`, `shape`, `cls`, `bboxes`, `segments`, `keypoints`, `normalized`, and `bbox_format`. In v0.1, keypoints are `None`.

The `summary` counters are not mutually exclusive categories. A label can be found and then fail verification, for example, so do not reconstruct counters merely by counting final statuses. The adapter must preserve reference training policies for all-empty or missing labels.

### 5.2 Label-only API and adapter

```python
# Independent annotation API; excludes image verification.
parsed = uyd.parse_labels(label_paths, num_classes=80, task="detect", workers=4)

from ultrafast_yolo_dataset.integrations.ultralytics import FastYOLODataset

dataset = FastYOLODataset(
    img_path="dataset/images/train",
    data={"names": {0: "object"}},
    task="detect",
    acceleration={
        "cache_backend": "native",  # native | ultralytics
        "repair_jpeg": "reject",
        "workers": 4,
    },
)
```

`parse_labels` does not guarantee image integrity or training readiness. Publish its benchmarks separately from full `scan` results. `FastYOLODataset` is a proposed explicit subclass; verify an example of inserting it into the actual trainer in M3. Do not present this as an existing official Ultralytics setting.

Represent native cache load failures as `CacheMiss(reason=...)` so the adapter can rebuild. Return explicit fallback reasons for unsupported profiles or tasks. Importing the package must not monkeypatch or automatically change the backend.

## 6. Native architecture

```text
Python path/config validation
    → bounded Pillow image-verification pool
    → indexed batches of valid image metadata / failures
    → Rust: read label bytes → parse → validate → dedup → pack
    → input-order merge of records, counters, diagnostics
    → owned ScanResult
    → optional native cache or Python list/dict materialization
```

### 6.1 Components

- `yolo-dataset-core`: Python-independent parser, numeric validation, geometry, deduplication, and packed records.
- `yolo-dataset-io`: Bounded read workers, cancellation, fingerprints, and atomic caching.
- `yolo-dataset-python`: PyO3 bindings, NumPy owners, errors, and configuration.
- `python/ultrafast_yolo_dataset`: Pillow stage, reference fallback, adapter, and diagnostic rendering.
- `tests/reference`: Pinned oracle runner and fixtures. Preserve licensing notices when copying reference code.

Select Rayon or a scoped worker pool in M0. Do not initialize a global pool during import. Use lazy, process-local pools to avoid thread creation before a DataLoader fork and unsafe reuse afterward. Test Linux fork and Windows/macOS spawn.

### 6.2 Parsing and numeric precision

Validate UTF-8 before tokenizing bytes. Prototype and benchmark the fast float parser, then verify rounding and accepted-syntax parity against NumPy string→float32 conversion. Do not assume `str -> f64 -> f32` always matches direct f32 conversion. Exact kernels must not use fast-math or reassociation optimizations.

Store objects in contiguous arrays and segments through ragged offsets. Use int64 offsets and checked conversion to usize. Deduplication must preserve reference numeric equality, signed zero, first occurrences, and sorting conditions. Do not retain original token strings or per-row Python objects in results.

### 6.3 GIL, concurrency, and memory budget

- Python path validation, Pillow, and object construction follow Python runtime rules. Release the GIL only in Rust read/parse/cache stages.
- Do not describe the entire hybrid v0.1 pipeline as one fully native operation. Measure batch round-trip costs between image and native stages.
- Start with a default worker budget of at most four available CPUs, shared by I/O and Pillow stages. Do not create nested T×T pools.
- Bound simultaneously retained image metadata, label bytes, and reorder buffers with `max_in_flight`.
- A slow file must not cause unbounded accumulation of out-of-order results. Close all file descriptors within their batch scope.
- Provide configurable file-byte, row, and point limits with explicit resource-limit errors. Record limits in the supported domain and benchmark manifest.
- Honor `KeyboardInterrupt` at chunk boundaries and never publish a partial cache.
- Measure RSS across repeated scans and after releasing results to distinguish leaks from allocator retention.

### 6.4 Cache design

Use the separate `.uydcache` extension for the v0.1 native cache. Do not overwrite legacy `.cache` files. Design the format around a small versioned metadata header and contiguous binary array sections. Finalize the codec in M0; do not use Python pickle or object dtypes.

Required metadata: magic, schema version, parser version, reference profile ID, endianness, task/configuration, original ordered path manifest, source indices, array dtypes/shapes/offsets, section checksums, fingerprint policy, counters, and diagnostics.

| Fingerprint policy | Validation method | Guarantees |
|---|---|---|
| `content` (default) | Hash ordered paths/configuration and all relevant image/label bytes | Detects changes that preserve size and mtime; includes file-reading costs |
| `metadata` (explicit opt-in) | Paths, file type, size, mtime_ns, and missing state | Potentially faster, but cannot detect content changes with identical metadata |
| Legacy compatibility | Separately compute the reference path/size-based hash | For the reference cache bridge; distinct from native content guarantees |

Content validation reads files even on warm loads, so datasets with substantial image I/O may see limited warm-cache gains. Do not use metadata-mode results to claim content-mode performance. Design an immutable-dataset-manifest path separately in the future.

- Include task, num_classes, single_cls, repair/verification policies, and parser/profile versions in the cache key.
- If a file changes during scanning, compare metadata before and after reading, then retry or return a changed-during-scan error. Content mode retains digests of the label bytes actually parsed and the image snapshots actually verified, and compares them with fingerprints before publishing. Metadata-only checks do not substitute for content validation. Do not cache results for which a consistent snapshot cannot be established.
- Publish through a unique temporary file in the same directory → flush/fsync → atomic replace. Use a multi-process lock and timeout.
- After waiting for a lock, validate and reuse an equivalent cache if another process has completed it. Network filesystem atomicity/locking is outside v0.1 guarantees.
- Treat truncation, checksum mismatch, invalid offsets/shapes, and unknown versions as cache misses. Validate sizes and section bounds before allocating.
- Support a separate writable `cache_dir` for read-only datasets. A cache write failure must not discard an otherwise valid scan result.
- Begin with loading into owned buffers in v0.1. Add mmap after verifying Windows lifetime/deletion behavior and writable augmentation policies.

The legacy bridge uses reference Python cache helpers and reads only trusted, user-owned caches. Compare deserialized arrays, shapes, fields, and counters rather than `.cache` file bytes. Do not reuse unknown schemas/profiles. [Reference cache helpers](https://github.com/ultralytics/ultralytics/blob/795a556942a12fe0124cf767888194a1d0b83e2e/ultralytics/data/utils.py)

## 7. Benchmark plan

### 7.1 Workload matrix

| Dimension | Cases |
|---|---|
| Scale | 1k, 10k, 100k, 500k image/label pairs; optional 1M stress case |
| Task | Detection and Instance Segmentation measured separately |
| Labels | 0,1,10,100 objects per image; mixed suites with realistic distributions |
| Polygons | 3,16,100,1000 vertices and heavy-tailed distributions |
| Errors | Fixed missing/empty/duplicate/corrupt proportions, separate from healthy corpora |
| Storage | Local SSD required; HDD/NFS reported separately |
| Cache state | No label cache with warm/cold OS cache; native/legacy cache hits |
| Parallelism | workers=1,2,4,8 and equal total CPU budgets |

Use a COCO2017 subset converted to YOLO as a real workload and publish conversion code, seeds, and annotation/image hashes. Large synthetic suites must create genuinely distinct files. Do not present repeated references to a small set of paths as 100k/1M-file filesystem performance. Separate large image-fixture generation costs from benchmark execution costs.

### 7.2 Timing boundaries

1. **P0 parser-only:** In-memory bytes → parsed records. Used for algorithm comparisons.
2. **P1 label engine:** File read + parse + validate + packed result. Explicitly excludes image verification.
3. **P2 scan:** Pillow verification + label engine + counters + final Python labels. Compare with a reference `verify_image_label` batch.
4. **P3 cache miss:** Discovery/hashing + P2 + cache serialization.
5. **P4 cache hit:** Invalidation checks + loading + materialization of the same mutable Python label objects.
6. **P5 application:** Dataset construction start → first batch. Measure subsequent epoch time separately.

The reference must perform equivalent work using its existing ThreadPool. For cache-hit comparisons, also provide an auxiliary baseline with equivalent fingerprint strength. Deleting a label cache does not clear the OS filesystem cache. Label runs `OS-cache state uncontrolled` when a cold OS cache cannot actually be controlled.

Use ≥30 timed samples for microbenchmarks and ≥5 independent process runs for full scan/cache cases, alternating execution order. Record median, p95, bootstrap 95% confidence intervals, labels/s, files/s, bytes/s, wall/user/system time, peak RSS, and cache file size. Collect page faults and open/stat counts where practical.

Save JSON/CSV output and a report containing the reference SHA, native commit, dependency lock, CPU/RAM/OS/storage, file distribution, input hashes, validation mode, repair policy, workers, fallback counts, timing boundaries, and per-run parity status.

### 7.3 Performance gates

- P1 median speedup ≥2× on each fixed 100k-file Detection/Segmentation suite.
- P3 geometric-mean speedup ≥1.25× with equivalent image verification and Python materialization.
- G4 compares the same logical data returned as packed results. Also publish peak RSS for legacy-compatible eager materialization.
- Proposed P4 content-mode target: ≥1.5×. If it cannot be met, revise cache scope/targets instead of substituting metadata-mode numbers.
- For small 1k-file datasets, P3 median regression must not exceed the larger of 10% of reference latency or 50ms.
- Do not meet targets by skipping corrupt/missing fixture checks or hiding fallback rates.
- Publish actual time to first batch. Improved steady-state epoch throughput is not a release requirement.

## 8. Testing strategy

| ID | Test | Release evidence |
|---|---|---|
| T1 | Pinned differential corpus | Zero array/record/counter/ordering mismatches |
| T2 | Parser property tests / fuzzing | ≥10,000 seeded cases; no crashes on arbitrary bytes, truncation, or extreme values |
| T3 | Float32 boundary tests | Tolerance boundaries, class extrema, rounding, signed zero, duplicate ordering |
| T4 | Filesystem/image fixtures | Unicode, relative paths, EXIF, tiny/corrupt images, permission/missing cases |
| T5 | Cache invalidation | Add/remove/rename, same-size edits, same-mtime edits, configuration/version changes |
| T6 | Cache faults/concurrency | Partial writes, invalid offsets, checksum failures, simultaneous writers, cancellation |
| T7 | Ultralytics integration | Identical dataset labels, ordering, first batches, and training/validation error policies |
| T8 | Wheels/lifetime | Fresh installation, result-owner release ordering, mutable copies, fork/spawn |

Required regressions include original ordering without duplicates, reordered output with duplicates, and different polygons with identical bboxes. Combine Rust unit tests, Python pytest/Hypothesis, and cargo-fuzz. Define applicable sanitizer/Miri coverage for unsafe buffer code and review it separately.

Image tests include EXIF orientations 1/3/6/8, healthy and repairable JPEGs, PNG verification failures, and unsupported codec paths. Compare JPEG repair parity on two isolated copies of the inputs; test the intentional differences of default reject mode separately.

Cache tests verify native→native round trips and native→legacy→reference reads. Whole cache payload byte identity is not required, but logical arrays must match exactly. Never return stale results after detecting a mismatch between native cache contents and live files.

Integration tests cover workers=0 and at least 2, training/validation, mixed segment/detect data, no valid images, missing labels, and fallback for overridden hooks. Verify that class filtering, rectangular sorting, and resampling still run after `get_labels`.

## 9. Milestones and implementation sequence

| Stage | Estimated development effort | Deliverables / exit criteria |
|---|---|---|
| M0: Oracle/profiling | 3–5 days | Profile lock, fixture manifest, stage timers, Python/NumPy float behavior, cache codec selection |
| M1: Detection engine | 1 week | parse_labels, counters, dedup/order, packed arrays, passing differential tests |
| M2: Segmentation + hybrid scan | 1–2 weeks | Segment geometry, Pillow stage, repair policy, full scan parity |
| M3: Cache + adapter | 1–2 weeks | Native cache, trusted legacy bridge, FastYOLODataset, invalidation/concurrency |
| M4: Benchmark + release | 1 week | P0–P5 report, wheels, CI, documentation, release candidate verification |

Estimates assume one developer. If image I/O/verification dominates in M0, record an Amdahl analysis of whether startup targets are achievable. Do not rush in a native image verifier that reduces verification coverage.

Initial issue sequence: reference fixture exporter → dedup/order specification → Detection parser → float32 parity fuzzing → segment boxes → hybrid scan → cache format/invalidation → adapter → full startup benchmark → wheel CI.

## 10. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Storage latency dominates | Parser gains have little effect on total time | Stage-level timing; distinguish local SSD and other storage |
| Pillow verification cost | Results differ from expectations of a fully native pipeline | Publish hybrid boundaries and compare equivalent verification |
| Float parser/unique semantics differ | Changed label selection or ordering | Exact oracle, boundary corpus, reference fallback |
| Python dict/list materialization | Lazy-result gains disappear in integration | Benchmark final mutable outputs |
| Cache invalidation strength | Stale results or reduced warm-cache gains | Content default, explicit metadata mode, separate reports |
| Files change during scanning | Mixed-snapshot cache | Before/after checks, bounded retries, blocked publication |
| Many workers / DDP | Duplicate scans and excessive file descriptors | Bounded shared budget, atomic cache locking, process-local pools |
| Upstream hook/version changes | Broken subclass semantics | Base-profile gates, reference fallback, compatibility CI |
| Codec/platform differences | Different image acceptance/rejection | Pinned Pillow/codec profiles and OS-specific fixtures |
| Licensing/distribution readiness | Delayed public release | Repository license/NOTICE, dependency inventory, fixture provenance |

The proposed initial repository license is AGPL-3.0-or-later; finalize it when creating the repository. Preserve attribution and licensing when porting reference implementations. Separate the Rust core from Python bindings in wheel builds and include a dependency license inventory. [Upstream license declaration](https://github.com/ultralytics/ultralytics/blob/795a556942a12fe0124cf767888194a1d0b83e2e/pyproject.toml)

## 11. Packaging and release criteria

Proposed v0.1 wheel matrix: CPython 3.10–3.13 × Linux x86-64 / macOS arm64 / Windows x86-64. Build with maturin and prioritize wheels that do not require users to install a Rust compiler. Freeze actual NumPy/PyO3/Python combinations after M0 CI verification. Python 3.14, Linux arm64, macOS x86-64, and free-threaded Python are follow-up targets.

- [ ] Detection/Segmentation C01–C12 and T1–T8 pass at the release commit.
- [ ] There are zero known numeric/metadata mismatches within the supported domain.
- [ ] Unsupported tasks, subclasses, and versions are correctly detected, with fallback recorded.
- [ ] Image verification and repair policies are explicit in the README and API.
- [ ] Content/metadata/legacy cache guarantees and costs are published separately.
- [ ] P1/P3/P4 and memory targets are met, or an evidence-backed PRD revision is completed before release.
- [ ] Public benchmarks include file reading, verification, materialization, and cache validation costs.
- [ ] Native cache corruption, atomicity, concurrent-writer, and stale-input tests pass.
- [ ] Supported wheels have been installed and verified in fresh environments without a source checkout.
- [ ] README, API/schema documentation, compatibility matrix, benchmark reproduction, CONTRIBUTING, and LICENSE/NOTICE are available.
- [ ] CI passes for the exact candidate commit, and integration smoke tests have been rerun using distribution artifacts.
- [ ] Adapter opt-in and rollback to the original YOLODataset are documented.

## 12. Future roadmap

- **v0.2 Pose:** Define a separate parity contract covering `kpt_shape`, 2D-to-visibility expansion, 3D keypoint fields, negative coordinates, and dedup keys.
- **v0.2 Discovery:** Move recursive enumeration, path mapping, and batched stat operations into native code while preserving upstream ordering, fraction selection, and symlink behavior.
- **v0.2 Cache:** Immutable dataset manifests, incremental per-file reuse, and read-only mmap results; verify invalidation and augmentation mutation together.
- **v0.3 OBB:** Support task-specific polygon formats while distinguishing downstream rotated-box conversion semantics.
- **v0.3 Annotation readers:** Start compact native loading with COCO JSON, then extend Objects365/LVIS/Grounding support by schema.
- **v0.3 Image metadata:** Explore a Rust image verifier as an experimental backend. Header-only fast modes require a product contract separate from full verification.
- **Later integration:** Distributed startup, network storage, and streaming/sharded datasets. Keep these outside required scope until local filesystem gates pass.
- **Independent interoperability:** Add a zero-copy contract for passing packed segments to `ultrafast-maskops` only after validating lifetime, precision, and parity on both sides.

## 13. Decisions to finalize in M0

1. The first passing Python/NumPy/Pillow/codec profile and numeric parser.
2. Startup time distribution by stage and achievable full-scan speedup.
3. Native cache codec, configuration schema, and byte/resource limits.
4. How users explicitly enable original-file JPEG repair in integration.
5. The first representative 100k-file corpus, measurement hardware, and storage/cache-state procedures.
6. Whether feasibility results support including both hybrid scan and native cache in v0.1.

This document defines product requirements and a validation plan. Its signatures, targets, and proposed platforms do not represent completed implementation or measured performance.
