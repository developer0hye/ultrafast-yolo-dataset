# ultrafast-yolo-dataset

Experimental Rust/PyO3 YOLO label engine. The full dataset initialization and
cache library is under development; this is not yet a drop-in YOLODataset replacement.
The original goals and release gates remain in [docs/PRD.md](docs/PRD.md).

```python
from ultrafast_yolo_dataset import parse_labels
result = parse_labels(paths, num_classes=80, task="segment", workers=4)
assert not result.reference_required
arrays = result.to_arrays()  # explicit, owned writable copies
```

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
are not silently accepted. In this alpha, callers must handle codes 3/4 explicitly;
automatic reference fallback and exact diagnostics are still pending.

The file limit defaults to 16 MiB per label. Unicode numeric syntax, nonfinite
numbers, malformed labels and unusual whitespace can require the reference path.
Both supported tasks follow the reference's content-based segment detection.
This function does not verify images or make its output training-ready.

## Development and evidence

```sh
uv venv --python 3.12
uv pip install -r requirements-test.txt
uv pip install -e .
cargo test --lib
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

Still required: Pillow verification and explicit JPEG repair policies, exact error
fallback/counters, versioned content-validated atomic cache, legacy cache bridge,
actual YOLODataset integration, 100k/500k full-scan/cache/first-batch benchmarks,
fuzzing and cross-platform wheel/CI validation. See [docs/STATUS.md](docs/STATUS.md).
