# Snapshot scratch candidate

This separate candidate starts from the tested cache-reader implementation at
`409c154`. It has not yet been compiled or measured. Existing cache-reader results
do not measure this additional change.

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
