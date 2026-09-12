# Experimental cache buffer lifetime changes

This `perf/cache-read-copy` worktree contains an **unbuilt, untested candidate**.
The ongoing 500k-pair and GPU measurements use their original installed wheels
and source trees. No existing result measures this change. Rust formatting,
Ruff and workflow linting are source checks, not compilation or runtime evidence.

## Change

The current reader allocates a zeroed Rust `Vec<u8>` for each cache section,
reads and hashes the file into it, then allocates Python `bytes` and copies the
whole section again. During that copy, both buffers coexist. The candidate
allocates the final Python object with PyO3 0.25.1 `PyBytes::new_with` and reads
the same 4 MiB chunks directly into its initialization slice, calculating the
same SHA-256 as it reads. It publishes the section only after initialization
and checksum verification succeed.

For a section of L bytes, this removes the temporary vector's logical L-byte
payload and one L-byte copy. It still zero-initializes the final Python buffer.
It does not remove the cache payload itself, the JSON metadata, array validation,
input content hashing or the final mutable Ultralytics labels. Allocator overhead
and the highest-memory phase may dominate process RSS; no measured saving or
cache-hit speedup is claimed.

## Release serialized copies before save-time content revalidation

A later source review adds a separate save-path change: drop the local encoded
section list immediately after `write_cache_sections` returns. The Rust writer
already borrows immutable Python bytes and makes no second whole-cache copy.
It is synchronous, flushing and syncing the temporary file before returning;
its borrowed buffers and owning references have ended when Python resumes.
The encoded metadata/array copies are therefore no longer needed by the writer.
Previously Python retained the list across input content revalidation, atomic
replacement and directory sync. The candidate ends that lifetime before input
revalidation, while retaining the original ScanResult and provenance.

This was motivated by the first 500k-pair cache-generation result on the frozen
baseline: constructor RSS was 1,752.16 MiB for reference and 1,901.19 MiB for
native, with 48.4948 s in content revalidation. Those values measure neither
candidate and do not attribute the RSS difference to these copies. A shorter
object lifetime may not lower the process high-water mark or allocator RSS.
Shared provenance bytes remain owned by the ScanResult and are not freed merely
by dropping the section list. No cache format, content check, publication order,
locking, failure cleanup or reference-compatibility contract is changed.

The prior preparation receipt covers only the Rust reader prototype. This
save-path change is also **not runtime-tested or benchmarked**. After the hosts
are free, qualify it using existing save/load parity, mutation-before-publication,
write-failure and concurrent-writer tests, then profile the constructor phases
and run a separately identified cache-generation comparison. The direct-reader
and save-lifetime changes need separate performance attribution.

## Ownership, errors and GIL scope

The pinned PyO3 source was inspected: `new_with` allocates the bytes object,
zero-initializes it, passes a mutable slice to the initializer, and returns the
object only on success. On initializer error, the private object is dropped.
This candidate adds no unsafe block. The unexposed bytes object's owning handle
stays alive while file read/checksum chunks release the GIL; only the mutable
byte slice, file and hasher enter those closures. This reasoning still needs
compiler and runtime validation.

Container magic/version, section count/length bounds, checksum checks, file
modification checks and per-chunk Python signal checks remain in place. Empty
sections still verify SHA-256 of empty content. Lengths beyond Python's signed
byte capacity explicitly raise MemoryError before allocation. Returned sections
remain owned immutable `bytes`, including ownership through NumPy views.

The final Python buffer's zeroing occurs with the GIL held in `new_with`; the old
final vector-to-bytes copy also held the GIL, while its earlier vector zeroing
released it. Those are different blocking intervals. Measure concurrent-reader
behavior and interruption responsiveness rather than assuming they improve.
The source-review receipt identifies the inspected dependency bytes; it does not
prove memory safety, successful allocation or cache-format correctness.

## Required validation

Twelve new cases in `tests/test_cache_section_buffers.py` are prepared for zero,
one-byte and 4 MiB chunk boundaries, multi-chunk reads, exact built-in bytes
results, NumPy ownership after list/file removal, late checksum rejection,
independent concurrent readers and file/size limits. They have **not run**. The
existing corruption/logical-schema, content-validation, concurrency and full
framework cache suites must also pass from a freshly installed candidate wheel.

After a host is free, build the candidate with the same pinned toolchain and
dependency versions as the baseline. Run the focused cases and full existing
suite; separately compare large-section allocation/RSS and latency with checksum
and file reads included. Preserve failure artifacts and compare before/after
outputs. Then rerun actual P4 construction on fixed real and synthetic fixtures,
including full mutable labels and first-batch parity. This candidate must remain
separate until those checks establish both correctness and useful improvement.
