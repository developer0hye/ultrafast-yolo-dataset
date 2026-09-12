# Native annotation cache

The explicit `annotation_cache="native"` adapter stores `.uydcache` files. The
default adapter mode remains `"none"`. Native loading never uses pickle and does
not read or overwrite the separate legacy `.cache` file.

```python
from ultrafast_yolo_dataset.ultralytics import FastYOLODataset

dataset = FastYOLODataset(
    img_path="data/images/train",
    data={"names": {0: "person"}},
    task="segment",
    annotation_cache="native",
    cache_fingerprint="content",
    cache_dir="/writable/annotation-cache",
)
print(dataset.annotation_cache_hit, dataset.annotation_cache_write_error)
```

For explicit ordered pairs, use `scan_cached(path, images, labels, num_classes=...)`.
It returns a `ScanResult` with `cache_hit`, `cache_path`, and `cache_write_error`.
`load_cache(...)` returns a validated result or `CacheMiss(reason)`;
`scan(..., fingerprint="content")` captures inputs that can subsequently be saved
with `result.save_cache(path)`. Ordinary scans cannot claim captured-byte provenance.

## Input and compatibility contract

Content mode reads every input and checks SHA-256, file type, size and modification
time on every hit. This deliberately includes image bytes, so image I/O remains
part of a warm annotation-cache load. Ordered paths, relative-path working
directory, task/classes/single-class policy, verification/repair policy and byte
limits must also match. An equal-size edit with restored mtime invalidates content
mode. `metadata` is an explicit weaker policy that can miss such an edit; its
performance is not evidence for the content-mode target.

The profile binds compiled Rust sources and Cargo manifest/lockfile, Python scan/cache/reference sources,
NumPy/Pillow/Python versions, platform/CPU features, and Pillow opener/plugin state.
It intentionally invalidates across potentially different signed-zero behavior or
image verification. The integration profile pins Ultralytics
`795a556942a12fe0124cf767888194a1d0b83e2e`, including its patched `Image.open`, and
pi-heif 1.4.0. Ultralytics can activate HEIF support after an image-open failure;
changes to that registry conservatively invalidate an earlier cache. It is possible
to rescan unnecessarily across fresh processes with different plugin state.

Snapshots own the bytes used by Pillow and the native or reference label parser.
Handle/path identity and metadata are checked around reads. Explicit JPEG repair
records the encoded replacement's digest and checks the written bytes against it.
Saving revalidates the entire captured content after serialization and before
publication, even for metadata mode. A detected change raises `InputChangedError`;
it does not become a corrupt annotation or a successful cache write. These checks
do not freeze files indefinitely: subsequent training image reads still observe
the live dataset. Use stable inputs while training.

## Binary layout

The container header is `UYDCACHE` (8 bytes), little-endian container version 1
(`u32`), then section count (`u32`). Each descriptor is a little-endian length
(`u64`) and a 32-byte SHA-256 of that section. Section payloads follow contiguously.
The current logical schema is **2** and uses 11 sections:

| Index | Contents |
|---|---|
| 0 | UTF-8 JSON: profile, ordered paths, configuration, counters, diagnostics, array shapes/dtypes |
| 1–8 | Source indices, image shapes, object offsets, labels, segment points, segment offsets, segment object indices, repaired source indices |
| 9–10 | Image and label fingerprint tables, respectively |

Numeric arrays are little-endian `int64` indices/shapes or `float32` annotations.
Fingerprints use `fixed57-le-v1`: one byte of file kind (0 file, 1 missing,
2 directory, 3 other), 8 bytes of unsigned size, 16 bytes of signed modification
time in nanoseconds, and 32 digest bytes. Non-file digest bytes must be zero;
missing entries also have zero size/time. Fixed-width tables avoid hundreds of
thousands of Python dictionaries and hex-string objects during loading. Rust
validates and compares these tables directly against live inputs.

Unknown versions/profiles, truncated data, invalid checksums, out-of-bounds or
misaligned shapes/offsets, invalid record ordering, and inconsistent counters
become cache misses. Checksums detect damage; they do not authenticate a cache
against an attacker who can replace both content and checksums. Loaded arrays use
owned immutable byte buffers. Exported label dictionaries have independent writable
arrays so augmentation cannot alter a loaded cache or another sample.

## Publication, limits and failures

Writers use a stable sibling `.lock` file with an exclusive advisory OS lock. After
acquiring it they recheck for a cache produced by another writer. A unique sibling
temporary file is written and synced, inputs are revalidated, and `os.replace`
publishes it atomically. The directory is synced on Unix. Lock files are never
unlinked; a forked child cannot release its parent's lock. Interrupted or failed
prepublication writes clean up temporary files and preserve the old cache.

Directory/lock/write failures preserve a valid scan result and report a write
error. Input changes, `KeyboardInterrupt`, and memory allocation failures are not
silently suppressed. A directory-sync error occurs after replacement and can report
failure even when the complete new cache is already visible.

Defaults: cache 2 GiB, JSON metadata 256 MiB, image snapshot 64 MiB, label 16 MiB,
aggregate retained label-fallback snapshots 64 MiB, lock wait 30 seconds, and at
most one million image/label pairs. Failure to capture within snapshot budgets
can still return the ordinary valid scan, marked unavailable for caching. Parser
resource limits remain explicit. These bounds are not a claim of verified
one-million-file capacity or a whole-process memory bound.

The current validation covers macOS arm64 and Linux x86-64. Windows durability,
directory-error parity and wheel tests remain pending. Network filesystem locking
and atomicity are outside the current contract. Static FIFOs are rejected before
opening; adversarial regular-file-to-FIFO replacement during the open race has not
been validated. There is no mmap or persistent global thread pool.

Reproducible cache-startup measurements and their limitations are recorded in
[BENCHMARKS.md](BENCHMARKS.md). Correctness tests alone do not establish a speed or
process-memory benefit.
