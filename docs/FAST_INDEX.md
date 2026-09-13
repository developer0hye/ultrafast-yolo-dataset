# Fast index: dataset startup against the unmodified Ultralytics dataset

`FastYOLODataset(..., annotation_cache="fast")` replaces the three costs that
dominate the reference `YOLODataset` constructor on large datasets. Every result
below compares it with the **unmodified pinned Ultralytics `YOLODataset`**
(its own `.cache` pickle, `get_hash` validation and eager label objects) on the
same host and files, in fresh processes, with full output parity.

## Where the reference spends its time

Phase profile of the reference cache hit, 500,000 Detection pairs
(1,000,000 files), Apple M2 16 GiB, dataset on a USB APFS SSD:

| Phase | Seconds | Share | Cause |
|---|---:|---:|---|
| `get_cache_hash` (`os.stat` × 1,000,000) | 21.6 | 76% | one Python syscall and result object per file, serialized by the GIL |
| `get_img_files` (glob, filter, sort) | 2.3 | 8% | Python directory walk and string filtering |
| `npy_files` (`Path` × 500,000) | 2.3 | 8% | one `Path` object per image built in the constructor |
| `load_dataset_cache_file` (unpickle) | 0.7 | 3% | one dict and several small arrays per image |
| `verify_labels` | 0.7 | 2% | per-image Python iteration |

On a cache miss the reference also runs Pillow's JPEG marker loop (pure Python
under the GIL) for every image and calls `get_hash` twice.

## What changed

| Python cost | Replacement |
|---|---|
| per-file `os.stat`, glob, sort | one Rust pass lists each directory together with its entries' metadata (`getattrlistbulk` on macOS, parallel `stat` elsewhere), without the GIL |
| pickle of per-image dicts | a memory-mapped packed cache: a few contiguous arrays, section checksums verified in parallel, NumPy views without copying |
| eager label dicts | `LazyLabels`: each access builds the reference dict with new writable arrays; whole-dataset iteration materializes natively in chunks |
| eager `npy_files` | `Path` built per access |
| Pillow JPEG header parsing | a Rust probe replicating Pillow 12.1.1's marker walk, `exif_size` and the EOI check; any file it cannot guarantee goes to the unchanged Pillow path |
| per-file result assembly | one native label parse; the packed arrays are the result, with per-file Python only for exceptions |

Cache validation is **per file** — kind, size and modification time in
nanoseconds of every image and label — plus the ordered path list. The
reference hashes only the summed size of all files and the joined paths, so an
equal-size edit (or two edits whose size changes cancel) goes unnoticed there;
here any size or mtime change invalidates the cache. A cache is published only
if no input changed between the pre-scan index and a post-scan re-index.

## Results

### M2, 500,000 Detection pairs, cache hit

Five alternating fresh-process pairs; every run matched the reference's full
label list, image and label paths, counters, messages and first batch.

| Metric | Reference | Fast | Ratio |
|---|---:|---:|---:|
| Constructor + first batch (median) | 28.92 s | 2.89 s | **10.0×** (paired 95% CI 8.8–10.2) |
| Peak RSS after constructor (median) | 1,386 MB | 674 MB | 2.06× lower |

The fast constructor spends 2.69 s of 2.78 s (96.5%) in the native metadata
pass; all Python work together is under 0.1 s. That pass is bound by the host:
with 1,000,000 files and `kern.maxvnodes` = 250,902, every lookup recreates a
vnode, and the aggregate cost stays near 2.7 µs per file from 12 to 48 threads.

### Linux (i5-10400, 12 threads, NVMe), 500,000 Segmentation pairs

Commit `475f6c9`, same protocol and parity checks, five pairs each.

| Metric | Reference | Fast | Ratio |
|---|---:|---:|---:|
| Cache hit: constructor (median) | 12.25 s | 0.99 s | **12.4×** (CI 12.10–12.51) |
| Cache hit: peak RSS | 3.92 GB | 1.63 GB | 2.40× lower |
| Cache miss: constructor (median) | 191.73 s | 7.22 s | **26.6×** (CI 26.05–26.77) |
| Cache miss: peak RSS | 4.40 GB | 2.33 GB | 1.89× lower |

On Linux metadata calls and file opens are cheap (the reference's `get_hash`
takes 1.75 s). On a hit the reference's cost moves to unpickling polygons
(3.55 s) and per-image Python objects; the fast constructor spends 0.28 s
indexing 1,000,000 files and about 0.7 s validating and mapping the 1.5 GB
packed cache. On a miss the reference spends about 186 s in Pillow and Python
label processing under the GIL; the fast scan takes 6.8 s.

### M2, 500,000 Detection pairs, cache miss

| Commit | Reference | Fast | Ratio | Peak RSS |
|---|---:|---:|---:|---|
| `02df3ea` (native index only) | 134.5 s | 80.5 s | 1.67× (CI 1.66–1.96) | 1,725 → 974 MB |
| `475f6c9` (JPEG probe, packed assembly) | 132.2 s | 54.1 s | **2.45×** (CI 1.98–2.58) | 1,833 → 872 MB |

This host bounds the miss for both implementations: a breakdown of one fast
miss is 29.9 s opening and reading 500,000 JPEG headers and 20.9 s opening
500,000 label files, against 2.8 s for the directory index and 2.8 s for the
post-scan re-index. Opening files on this USB APFS volume with 1,000,000 files
(four times `kern.maxvnodes`) stays at about 25,000–27,000 files per second from
8 to 128 threads, so roughly 40 s of opens is a floor that the reference pays
too, and about 3× is the ceiling here. Linux, where opens are cheap, shows the
scan's own gain.

## Parity evidence

- `tests/test_fast_index.py`: miss and hit against the reference for Detection and
  Segmentation, nested and flat layouts; invalidation by resized, equal-size,
  deleted, added and touched files; six kinds of cache damage; `fraction`,
  `single_cls`, `classes`, `rect`, `cache="ram"`; file lists, several roots,
  relative paths and symlinked directories; mixed boxes and polygons; error
  messages; pickling and DataLoader workers.
- `tests/test_jpeg_probe.py`: accepted files reproduce Pillow's checked shape;
  truncated, EOI-less, MPO, XMP-orientation, partial-EXIF, Photoshop-error,
  junk, bad-marker, bad-DQT, tiny and non-JPEG files defer to Pillow. With
  `UYD_REAL_JPEG_DIR` pointing at COCO val2017, all 5,000 images are accepted
  with Pillow-identical shapes. Probed scans equal Pillow-only scans on a
  corpus with reference-parsed, duplicate, empty, missing, non-ASCII and
  negative-zero polygon labels.

## Reproduce

```sh
python bench/upstream_startup.py --corpus /path/to/startup-detect-500k --mode hit --rounds 5 --out hit.json
python bench/upstream_startup.py --corpus /path/to/startup-detect-500k --mode miss --rounds 5 --out miss.json
```

The corpus is generated by `bench/startup.py --prepare`. The miss campaign
deletes and rebuilds the reference `.cache`; back it up first if other evidence
depends on it. Raw per-run JSON is written next to the report in `*.runs/`.
