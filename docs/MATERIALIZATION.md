# Native mutable label export

`ScanResult.to_ultralytics_labels()` now constructs framework records in Rust,
avoiding one Python loop with multiple NumPy slicing/copy calls per image and
polygon. The returned dictionary keys, array values/dtypes/shapes, source order,
segment association and Python scalar types are unchanged.

Every exported class, box and polygon array is C-contiguous, writable and directly
owned by NumPy (`OWNDATA=True`, `base=None`). Records do not share their numeric
allocations with one another, with another export, or with the immutable cache.
Deleting the `ScanResult` or exported dictionaries does not invalidate retained
arrays. This also preserves ordinary NumPy ownership behavior without adding a
Rust owner object for every small array.

The implementation allocates initialized NumPy buffers through its C API and
copies into them while holding the GIL. Allocation errors propagate through the
Python error return. No uninitialized Rust slices are exposed. Before reading
packed inputs, it checks C order, pointer alignment, matrix shapes, source bounds
and ordering, offset endpoints/monotonicity, segment object ordering and per-image
alignment. The NumPy binding's generic slice helper accepts Fortran-contiguous
buffers and does not independently check pointer alignment, so these checks must
precede typed reads. Wrong-endian arrays are rejected by typed extraction.

Signals are checked every 256 images during object creation. Tests cover 200 seeded
packed examples, including arbitrary float32 bit payloads, independent ownership,
retained array lifetime, malformed layouts and offsets, and interruption. Arbitrary
float bits are used only to verify copying; the parser still rejects nonfinite
annotations. The full suite additionally compares actual cold/warm Detection and
Segmentation dataset labels and first batches against the pinned Ultralytics source.

Cache requests are normalized once before the optimistic load and possible
post-lock recheck. Both calls use the same ordered paths/configuration and still
perform all checksum, schema and input-content validation. Relative paths cannot
silently change meaning if the working directory changes while waiting for a lock.

The first Rust prototype exported `Vec`-backed arrays with separate owner objects.
Its 100k-file measurements and exact source snapshot are retained as
`materialize-vec-server-100k.json` and `materialize-vec-source.tar.gz`. The first
NumPy-owned COCO measurements and their source snapshot are also retained. Final
measurements add explicit alignment/layout checks and are reported separately in
[BENCHMARKS.md](BENCHMARKS.md). Component improvements do not imply equal improvements
in complete dataset initialization, which still includes discovery and hashing.
