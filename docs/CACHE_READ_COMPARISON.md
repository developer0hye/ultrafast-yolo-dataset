# Prepared comparison of the native cache section reader

`bench/cache_read_comparison.py` is prepared but **has not run**. Both hosts are
still occupied by frozen measurements. The cache-buffer candidate remains
unbuilt and untested. Ruff formatting/linting passed; that does not validate
fixture generation, a descriptor, a native call, a paired run or its statistics.
Nine small helper tests are prepared in `tests/test_cache_read_comparison.py`
and have not run either.

This experiment isolates the Rust reader change in `CACHE_READ_COPY.md`.
The native API opens a file, checks headers and limits, reads/checksums sections,
checks final metadata and returns owned immutable Python bytes. The candidate
also changes Python save-buffer lifetime; that Python path is never called by
this experiment, so these results cannot attribute a save-path benefit.

## Frozen matrix before measurement

Use seven conditions, five counterbalanced baseline/candidate fresh-process
pairs each: 70 measured processes and two descriptor processes. Each measured
worker has two untimed warmups after output verification, then five timed reads.
The independent Python oracle hashes container sections using bounded 1 MiB
buffers. It runs outside timers, before and after the native measurements.
The generated fixture and captured cache remain immutable during the campaign.

| Case | Payload section layout after the 2-byte synthetic metadata section |
|---:|---|
| 0 | One empty section |
| 1 | One section of 4 MiB minus 1 byte |
| 2 | One section of 4 MiB plus 1 byte |
| 3 | One 64 MiB section |
| 4 | One 256 MiB section |
| 5 | Four 64 MiB sections, 256 MiB total |
| 6 | Byte-for-byte capture of the native cache from a completed startup run; original section layout retained |

The last case is required. Capture the actual 500k Detection native cache only
once its P4 experiment has terminated and been verified. Record its original
path and full checksum, copy it to the isolated fixture, and verify source and
copy hashes. Timed workers use only that copy. Do not replace an inconvenient
real-cache result with the synthetic single-large-section case.

For a single large section of L bytes, the old reader's logical payload briefly
includes both its L-byte Rust vector and its L-byte Python result. The candidate
initializes the final Python object directly. With several sections, the old
peak depends on each section's position and previously retained results:
`max_j(previous_payload_j + 2 * section_bytes_j)`. It is not generally the final
payload plus the largest section. For four equal 64 MiB sections, the largest
such logical payload is 320 MiB; the final returned bytes total 256 MiB. These
are source-level payload models, not measured RSS or allocator guarantees.
The actual cache can have a different highest-memory phase.

## Identity and measurement boundaries

Each worker verifies the installed extension against the measured wheel,
installed Python sources against the wheel and declared source snapshot, and
all declared Rust/library source hashes against an external preserved build
identity. The two identities must record matching Rust/Cargo versions, target,
release profile and RUSTFLAGS. Only `src/cache.rs` and
`python/ultrafast_yolo_dataset/_cache.py` may differ between this specific pair.
Python, installed dependency versions, platform, CPU and RAM must also match.
The native extension does not embed these Rust source hashes: source-to-binary
provenance relies on the corresponding retained build/test artifacts, not on
an independently generated identity file alone.

The identity JSON has these fields:

```json
{
  "wheel_sha256": "from preserved build artifact",
  "extension_sha256": "from the tested installed wheel",
  "library_sources": {"relative/path": "sha256 from that source snapshot"},
  "toolchain": {
    "rustc": "exact rustc -Vv output",
    "cargo": "exact cargo -V output",
    "target": "the build target",
    "profile": "release",
    "rustflags": "the actual flags, or an empty string when unset"
  }
}
```

The `library_sources(root)` helper enumerates Cargo manifests/lockfile,
pyproject, Rust files, top-level package Python and profile JSON files. Bind
that map during the identified build. Do not retrospectively label an unknown
binary as being compiled from the current working tree.

Latency ends when the owned result is returned; output destruction happens
outside the timer. Peak RSS is sampled after the timed reads and before the
post-run output/file verification. It includes imports, wheel/source checks,
streaming fixture checks and same-backend warmups. The small cases can be
dominated by that earlier allocation history; this is not isolated working
allocation. Cache pre-reading warms storage, and the OS cache is uncontrolled.
All raw samples, commands, logs, identities, output section hashes and failures
are retained. Final statistics compare medians of five process medians and
enumerate all 3,125 paired bootstrap resamples.

This is not the Python public `load_cache` path: JSON/array-schema decoding,
dataset input-content validation, mutable label materialization and first-batch
construction are excluded. Use full P4 measurements to establish startup gain.
Use separate cache-generation phase/RSS measurements for save-buffer lifetime.

## Run only after correctness and host-availability gates

1. Build and freshly install the candidate using the baseline's pinned
   toolchain/dependencies. Run the 12 new native buffer tests, these nine helper
   tests and the complete cache corruption/concurrency/failure/framework suite.
2. Prepare the six synthetic files and capture the verified real startup cache:
   `python bench/cache_read_comparison.py --prepare NEW_FIXTURE --capture-cache COMPLETED_NATIVE_CACHE`.
   Fixture creation/validation is separate from the paired timing processes.
3. Qualify descriptors, one native worker per binary, oracle agreement and
   aggregate behavior using retained evidence. If qualification changes the
   harness, refreeze it before any full measurement.
4. Run the whole seven-condition matrix with `--baseline-python`,
   `--baseline-root`, `--baseline-wheel`, `--baseline-identity` and corresponding
   `--candidate-*` arguments, plus `--fixture NEW_FIXTURE --out NEW_REPORT`.
5. Independently audit every raw record and recompute all statistics; preserve
   the fixture/real-cache identity, compiled artifacts and all slow conditions.
   Test rejection of corrupted evidence before publishing any result.

No new cache-reader performance or memory improvement is currently claimed.
