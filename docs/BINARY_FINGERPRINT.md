# Binary fingerprint candidate

This unbuilt candidate branches from snapshot-scratch commit `44fdb53`.
Its native source and tests are prepared; compilation, installed-wheel tests and
performance qualification have **not** run. Both hosts are reserved by the
existing full-size startup campaigns, whose inputs and code remain unchanged.
No performance or compatibility result from those campaigns proves this change.

`snapshot::read` previously formatted every SHA-256 into an allocated lowercase
hexadecimal string. Native cache verification immediately decoded that string
back to 32 bytes when comparing its fixed-width provenance record. This candidate
retains `Option<[u8; 32]>` inside each fingerprint and copies the digest directly
into the verifier's stack record. No file open/read/stat or consistency check is
removed. Metadata-only verification continues to ignore the digest, and content
verification continues to compare every digest byte.

A custom serializer emits the existing lowercase hexadecimal string, or JSON
null for absent digests, only at the public JSON boundary. It uses a 64-byte
stack buffer. The strict JSON-to-proof parser, proof format, file kind codes,
signed modification times, field order and external Python signatures stay the
same. As with any source change, the compiled cache profile changes and correctly
invalidates a cache from the previous implementation.

This removes the digest string allocation from native read/verify paths. The
inline field may increase the fingerprint struct's size; total allocation/RSS
effects require measurement. JSON-producing APIs still allocate their output
strings. File-system costs may dominate full initialization, so neither the
source change nor a tiny-file microbenchmark establishes a startup speedup.

Before integration:

- Build in a new target directory and clean environment; preserve source,
  toolchain, wheel, installed extension and sdist evidence.
- Run Rust tests, Clippy and formatting, then the entire installed Python suite.
  New cases check exact JSON, independent 57-byte records, empty/files/directory/
  missing/FIFO states where supported, and same-size content changes with restored
  modification time. Existing mixed-size/concurrent/read-failure tests also apply.
- Compare against the tested scratch wheel with the same inputs and runtime.
  Qualify full constructors and content hits, including all output hashes and RSS.
  The current scratch experiment isolates only `src/snapshot.rs`; its frozen
  wrapper must not be reused unchanged for this two-file source experiment.
- Validate the exact candidate on Linux as well as M2 before considering a merge.

No build/test/measurement job for this candidate is queued automatically. Finish
each host's reserved campaign before starting its next qualification job.
