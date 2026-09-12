# ultrafast-yolo-dataset

Experimental Rust/PyO3 YOLO label engine. The full dataset initialization and
cache library is under development. An explicit base YOLODataset adapter is available
for the pinned source profile; it is not a general replacement for custom datasets.
The original goals and release gates remain in [docs/PRD.md](docs/PRD.md).

```python
from ultrafast_yolo_dataset import parse_labels

result = parse_labels(paths, num_classes=80, task="segment", workers=4)
assert not result.reference_required
arrays = result.to_arrays()  # explicit, owned writable copies
```

The hybrid scan additionally retains Pillow verification and uses the pinned
Python label verifier for native inputs marked `reference_required`:

```python
from ultrafast_yolo_dataset import scan

result = scan(image_paths, label_paths, num_classes=80, repair_jpeg="reject")
labels = result.to_ultralytics_labels()  # writable, isolated arrays
```

`scan` defaults to rejecting JPEGs that would require repair, without modifying
them. Explicit `repair_jpeg="reference"` enables the original save-to-input behavior.
Image verification and native parsing run in bounded, sequential stages with a
shared worker budget. Packed scan arrays are backed by immutable owned bytes;
classes/boxes/segments are copied only when constructing mutable label dictionaries.
Found/missing/empty/corrupt counters retain the reference's overlapping semantics.

The Rust path batches file reads, decimal parsing, finite-value checks, segment
boxes, validation, conditional duplicate sorting and compact storage. At most
256 unpacked records are retained between merges. Each call owns a bounded Rayon
pool; no global pool survives fork. Signals are checked between chunks.

The initial numeric profile follows NumPy's string->float64->float32 rounding.
It preserves fractional classes, the 1% coordinate tolerance, original order
without duplicates, and lexicographic first occurrences when duplicates exist.
Segment duplicates are keyed by converted class/box rows, not polygon vertices.

`to_arrays()` exposes labels, object offsets, segment points/offsets/object indices,
statuses, and duplicate counts. Status codes are **0 valid, 1 missing, 2 empty,
3 reference required, 4 resource limit**. Invalid or currently unsupported labels
are not silently accepted. `parse_labels` callers handle codes 3/4 explicitly;
`scan` applies the reference label fallback for code 3 and raises a resource-limit
exception for code 4.

The file limit defaults to 16 MiB per label. Unicode numeric syntax, nonfinite
numbers, malformed labels, unusual whitespace, and segments containing negative
zero require the reference path. NumPy's SIMD extrema can select a different zero
sign by array length/CPU; the fallback preserves those bits instead of normalizing them.
Both supported tasks follow the reference's content-based segment detection.
This function does not verify images or make its output training-ready.

## Development and evidence

For the validated base Detect/Segment dataset, opt in explicitly:

```python
from ultrafast_yolo_dataset.ultralytics import FastYOLODataset

dataset = FastYOLODataset(
    img_path="data/images/train",
    data={"names": {0: "person"}},
    task="detect",
    scan_workers=4,
    annotation_cache="none",
)
```

The adapter inherits discovery, augmentation, label consistency checks and batch
collation. `build_yolo_dataset` in the same module is an explicit factory for a
custom trainer. Importing the library does not patch Ultralytics globals. Unknown
sources, overridden base hooks, further dataset subclasses and unsupported tasks
raise `UnsupportedProfile`; keep their original implementation.

`annotation_cache="ultralytics", trust_legacy_cache=True` enables the legacy
pickle bridge for trusted, user-owned caches. It retains **weak path/size hashing**:
same-size content changes can remain undetected, exactly as with the reference
cache policy. A stored scan-profile marker binds task/classes/repair policy;
unmarked reference caches are rescanned because their configuration is unknown.
Caches exported by this adapter can also be loaded by the original YOLODataset.
Truncated/incompatible caches rebuild, and write failure preserves valid scanned
labels.

`annotation_cache="native", cache_fingerprint="content"` enables the separate
versioned `.uydcache` format with captured-byte provenance, checksummed compact
arrays and process-locked atomic publication. Optional `cache_dir` supports
read-only dataset directories. Content mode verifies all image and label bytes
on every hit; metadata mode is an explicit weaker option. See
[the cache contract and format](docs/CACHE.md). Performance validation is ongoing.

Real COCO Detection/Segmentation fixtures and the full constructor/first-batch
benchmark are reproducible through [COCO_FIXTURE.md](docs/COCO_FIXTURE.md). Results
include regressions: faster cache generation does not imply faster cache reuse.
The [native mutable label exporter](docs/MATERIALIZATION.md) preserves independent
NumPy ownership while reducing Python object-construction work; its latest full
startup measurements and remaining regressions are also recorded.

```sh
uv venv --python 3.12
uv pip install -r requirements-test.txt
uv pip install -e .
cargo test --lib
pytest -q tests/test_parser.py
uv pip install -r requirements-integration.txt
pytest -q tests
```

The label-only oracle contains unmodified functions from Ultralytics commit
`795a556942a12fe0124cf767888194a1d0b83e2e`; only image checking is deliberately
excluded from that test layer. Original source/extracted hashes are recorded.
The suite includes 10,000 seeded Detection/Segmentation cases, numeric boundaries,
duplicate ordering/alignment, Unicode paths, missing/empty/error distinctions,
ownership and file-size limits. Additional arbitrary-byte fuzzing is pending.

```sh
python bench/label_engine.py --prepare /path/to/new/detect --task detect --count 100000
python bench/label_engine.py --corpus /path/to/new/detect --out bench/out/detect.json
```

The P1 benchmark uses 100,000 genuinely distinct files with a frozen synthetic
distribution, equal worker counts, five alternating fresh-process runs, complete
output hashes, and equivalent exported packed arrays. Full input checksums before
and after timing deliberately pre-read files: **no cold-cache claim**. The reference
runs Ultralytics label verification in its existing ThreadPool style; image checking
is excluded from both sides. Results are not full dataset-startup or training gains.

Still required: broader Pillow/codec/platform and diagnostic validation,
representative 100k/500k full-scan/cache/first-batch benchmarks,
fuzzing and cross-platform wheel/CI validation. See [docs/STATUS.md](docs/STATUS.md).

Prototype results, including unfavorable cases and measurement limitations, are
in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
