# Snapshot scratch candidate

This separate candidate starts from the tested cache-reader implementation at
`409c154`. Frozen library source `9e2ab29` now passes five Rust unit tests, Clippy,
standalone wheel/notice checks and all 193 installed Python tests on M2, with
zero skips or failures. Existing cache-reader results do not measure this
additional change. Performance and Linux validation remain open.

The completed 500k Detection content-hit baseline spent approximately 46 s of
its 56 s constructor in input-content validation. Source review found two
avoidable per-file costs in `src/snapshot.rs`: initializing a 64 KiB read buffer
and allocating four-element identity vectors for before/after file stamps.
Neither source observation proves the generated machine-code cost or a speedup.

This candidate lazily allocates one initialized 64 KiB buffer per reading thread
and reuses it across files. Only the bytes returned by each read are hashed or
copied. Owned snapshot payloads remain separate. Thread-local scratch is dropped
when its thread exits; a long-lived caller thread retains 64 KiB after its first
content read. Rayon pools remain bounded by the existing worker count. Metadata
only, missing and non-file cases return before scratch initialization.

File stamps use an inline four-element array with the same Unix identity fields
and unchanged equality checks. The non-Unix identity remains a constant ignored
component. No filesystem stat/open/read operation, pre/post stamp comparison,
SHA-256 check, byte limit, path-replacement guard or Python signal checkpoint is
removed. The change adds no unsafe code. It combines scratch reuse and stamp
storage changes; a combined benchmark cannot attribute their individual gains.

Validation requires the complete installed framework/cache suite, mixed large,
small and empty reads across 256-file chunk boundaries, concurrent callers,
retained output ownership, and recovery after a read-time failure. Compare the
actual content-validation path against the preceding tested wheel with equal
input bytes, worker count and runtime. Keep scratch-related memory increases and
slower cases. Full constructor/cache-generation measurements and Linux validation
remain necessary even if isolated input validation improves.

## Installed M2 validation and next comparison

The build uses the preceding candidate's Rust 1.98.0 / LLVM 22.1.8, release
settings and pinned Python dependencies. Wheel, installed extension and retained
Cargo dylibs are byte-identical. The selected source archive includes all library,
build, test and Python benchmark files from the commit; it omits historical
benchmark archives and unrelated documentation. Its contents match the frozen
source selection, and the sdist matches library sources with only the documented
Cargo README normalization.

See the [build receipt](validation/snapshot-scratch-m2-build-v1.json),
[installed JUnit](validation/snapshot-scratch-m2-tests-v1.xml),
[test output](validation/snapshot-scratch-m2-tests-v1.log) and
[build evidence archive](../bench/results/snapshot-scratch-m2-build-v1.tar.gz).
The full suite completed in 131.35 s, including six new mixed-size, chunk-boundary,
threaded, ownership and failure-recovery cases. A new Rust test also forces a
read-time limit failure and verifies the next snapshot succeeds.

`bench/snapshot_startup_comparison.py` compares the previous tested native wheel
with this candidate through the existing actual constructor/first-batch worker.
Only the generated native cache directory is redirected per version; dataset
class, scan/image verification, transforms and full input/output checks remain
unchanged. It records process identity and installed descriptors outside timing.
This is an incremental native-versus-native comparison, not an upstream-Python
comparison. A 1,003-pair pilot must pass both phases before the full 500k series.
The full series uses five alternating pairs per phase and two separate primers,
retaining raw reports/logs, complete label/batch/diagnostic hashes, stage timers
and peak RSS. Final results still need an independent full artifact audit.

## Pilot correction: source-bound cache profiles

The first pilot's P3 workers produced identical labels, first batches and ordered
diagnostics. Both P4 primers completed, but the wrapper then rejected their
whole-file SHA-256 difference. Inspection found that all ten array/fingerprint
sections match, and metadata differs only in `profile`. This is required behavior:
`native_cache_profile()` hashes six embedded Rust files and Cargo.toml/Cargo.lock,
so a Rust source change invalidates the old cache profile.

The corrected wrapper verifies each compiled profile against those eight source
files and each cache profile against its own runtime. It compares all metadata
except that one verified profile field and every array/fingerprint section hash.
The initial failed report/logs/caches are retained under
`/Volumes/T7/ultrafast-vision-build/snapshot-startup-pilot-m2-v1*`; the corrected
pilot uses new `v2` paths. No library code or test result changed for this fix.
The [runtime profile check](validation/native-source-profile-m2-v1.json) also
corrects earlier claims that the native extension had no embedded source digest.

The corrected pilot completed both P3 workers, two independent primers and both
P4 workers. Every complete label, first-batch and diagnostic hash matches the
original pinned-reference pilot. The two caches have identical annotation and
input-fingerprint sections and identical metadata after excluding their separately
verified implementation profiles. Single-run constructor times were
0.2785 → 0.3210 s for P3 and 0.05634 → 0.05783 s for P4; these slower candidate
observations are retained and are not five-pair performance conclusions.

The [40-file pilot archive](../bench/results/snapshot-startup-pilot-m2-v2.tar.gz)
preserves the initial failure, both exact wrapper versions, corrected raw runs,
generated caches and identity/reference anchors. SHA-256 is
`6674d37983be1cd1fe6ae4a3c594726ffa2b4eeec49b811fe3b813f2e815f684`.
The [audit](validation/snapshot-startup-pilot-m2-v2-audit.json) and
[preservation receipt](validation/snapshot-startup-pilot-m2-v2-preservation.json)
record full readback and output agreement.

After the corrected pilot passed, the full M2 Detection comparison started with
the unchanged one-million-file, 500,000-pair corpus. It runs five alternating
native-versus-native pairs for each phase, with separate untimed primers and
generated cache directories. The [launch controller](validation/launch-snapshot-startup-full-m2-v1.py)
requires 20 measured workers, two primers and output equality with the original
complete 500k reference experiment. Its artifacts use
`/Volumes/T7/ultrafast-vision-build/snapshot-startup-full-m2-v1*`.
The frozen wrapper SHA-256 is
`c8074cabf59db8727b0576de9f7e9c3fe0b4e9a059e180219aaffb423397ee14`.
This measurement is running; no full-scale scratch-candidate speedup is claimed.
