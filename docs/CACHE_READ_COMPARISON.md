# Comparison of the native cache section reader

The candidate Release wheel and all **187 installed tests passed on M2**,
including twelve native-buffer cases and nine reader helper cases. The seven-case
fixture is being prepared after completion of the original Detection P4 series.
Paired reader performance, full startup gains and Linux validation remain open.

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
    "rustflags": "effective Cargo fingerprint flags, including generated linker flags"
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

## Baseline provenance and queued M2 build

The [baseline identity](validation/cache-reader-baseline-m2-identity-v1.json) is
bound to wheel `2ae058b33e702306d1ce89ea5cd43e7bc5fcdb87825b889f9556bfe5c864f595`.
Its installed extension, wheel extension and retained Cargo release dylib are
byte-identical at `5799c19d7fbe704451cb9ac3e7e72f1b6c8ecdd8219d7fca971dfcb24eff2b0b`.
Fifteen of sixteen declared library source files match the paired sdist exactly.
For Cargo.toml, packaging adds only `package.readme = "README.md"`; all other
parsed TOML fields agree. This normalization was checked explicitly after an
initial exact-byte comparison rejected it. See the
[provenance receipt](validation/cache-reader-baseline-m2-provenance-v1.json),
[Cargo fingerprint](validation/cache-reader-baseline-m2-cargo-fingerprint-v1.json)
and [cached compiler information](validation/cache-reader-baseline-m2-rustc-info-v1.json).

The retained compiler information matches current Rust 1.98.0 / LLVM 22.1.8.
Effective release flags include the linker arguments generated by maturin for
macOS dynamic lookup; an unset shell RUSTFLAGS does not mean an empty Cargo
fingerprint. Cargo 1.98.0 was read in this session, as distinguished in the
receipt from retained compiler/build evidence.

## Completed M2 build and installed tests

[The follow-up controller](validation/build-cache-copy-after-p4-v1.py) waited for
the exact Detection parent, then audited all ten P4 workers and two primers and
confirmed matching complete P3/P4 outputs. It archived source commit
`f120cab2f907e36fdcfd93d0879fe56ec4324017`, built a Release wheel/sdist using Rust
1.98.0 / LLVM 22.1.8, and created a fresh CPython 3.12.13 environment.

The original controller stopped when the offline install could not find a
cached Pillow wheel. A subsequent offline dependency install also lacked
contourpy. Online installation recovered both using the same pinned requirements;
the failed controller state and logs are preserved unchanged. Dependency freezes
match the baseline after excluding the dataset wheel itself, and `uv pip check`
passed. This was installation recovery, not a source or test change.

The standalone wheel passed all four Detect/Segment content/metadata cold/warm
cache cases, restored-mtime content edit detection and immutable ownership
checks. Wheel RECORD, installed bytes, sdist sources and all 137 notice files
were checked. The full installed suite then passed **187 tests in 132.15 s**,
with zero failures, errors or skips, including all twelve native-buffer and nine
reader-helper cases. [JUnit](validation/cache-copy-m2-tests-v1.xml) and the
[test log](validation/cache-copy-m2-tests-v1.log) are retained.

Wheel SHA-256: `25f0ce7ff2e40f106ec7af04a744c3883671c4ee502f3bd337e2414fe01b88d3`.
Its extension, installed extension and both retained Cargo release dylibs are
identical at `ec52a219e5e159017d221422fa8292cbe24022157370548bd445e3fb082f32d6`.
The [build identity](validation/cache-copy-m2-build-identity-v1.json) matches the
baseline toolchain; only `src/cache.rs` and package `_cache.py` differ in the
sixteen-file library inventory. All extracted source files match the archived
commit. Sdist normalization adds only `package.readme = "README.md"` to Cargo.toml.
The macOS 11.0 wheel tag does not prove execution on macOS 11.0.

The [28-file build archive](../bench/results/cache-copy-m2-build-evidence-v1.tar.gz)
contains the source archive, wheel, sdist, logs, failed controller state,
dependency freezes, JUnit, standalone result, wheel audit, Cargo fingerprint,
cached compiler information, exact sealer and build identity. SHA-256:
`11074f87869f8313d7a71c8ebb2f36db2b4f7822d8f3709fd25d0182b7607234`.
The [resume receipt](validation/cache-copy-m2-resume-v1.json) records all member
hashes and successful byte-for-byte readback. Its sealer was checked with Ruff's
Python 3.12 target because it uses standard-library `tomllib`.

This proves the recorded M2 build and correctness scope. Reader latency/RSS,
full cache generation/content-hit startup and Linux/platform gates remain open.
