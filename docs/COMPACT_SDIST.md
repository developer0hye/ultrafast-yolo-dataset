# Compact source distribution and source-only rebuild

The original hosted Linux CPython 3.11 artifact at `e1c5861` contained a
**176,951,569-byte sdist**. Its source tree included **191,255,055 uncompressed
bytes of benchmark archives**. The 1,162,029-byte wheel was a small part of the
177,914,547-byte uploaded artifact. The original artifact digest and source
pyproject were checked on a GitHub runner, and its JUnit files independently
confirmed 86 core plus 92 integration tests, with no failures or skips. See the
[preserved investigation](validation/dataset-sdist-investigation-v1.json).

The candidate at `9a981e45066636b0d726559c3b9f846942635f50` excludes only
`bench/results/*.tar.gz*` and `bench/results/*.zip` from source distributions.
All historical archives remain tracked in Git. Lightweight reports and the
code, notices, tests and benchmark harnesses remain in the sdist.

The new sdist is **990,595 bytes: 99.44% smaller**. This measures distribution
size; runtime startup, allocation and training benchmarks retain their original
binary/source identities.

## Source integrity and rebuilding

The first candidate job stopped at a strict byte-identity check because the
default Maturin Cargo generator rewrote `Cargo.toml`. The native cache profile
embeds that manifest's bytes, so silently accepting a different manifest would
weaken source/binary provenance. The candidate uses Maturin's supported Git
source generator, which packages tracked files without the Cargo rewrite.
See the [Maturin configuration](https://www.maturin.rs/config.html) and
[version 1.15 assembly implementation](https://github.com/PyO3/maturin/blob/v1.15.0/src/source_distribution/mod.rs).
Git is needed to assemble this sdist from development history; the subsequent
wheel build uses an unpacked source tree outside the Git checkout.

The [second job](https://github.com/developer0hye/ultrafast-yolo-dataset/actions/runs/34745337881)
completed naturally on Linux CPython 3.12:

- All **202 required build/runtime/notice/test/harness files** match the checkout
  byte for byte in the generated source archive. No benchmark archive remains.
- The sdist was unpacked under the runner temporary directory, without `.git`,
  and the wheel was built from that source using a separate Cargo target.
- A new virtual environment installed that wheel. RECORD and installed payload
  checks passed, along with **137 bundled notice files**.
- **86 core tests passed with no failures/errors/skips**. Four standalone
  Detect/Segment content/metadata cache cases passed, including label parity,
  restored-mtime content-change detection and the explicitly weaker metadata
  policy's behavior.

The original artifact ZIP digest, wheel/native hashes, source manifest, JUnit,
standalone receipts and terminal job status were read back independently. The
[complete preserved bundle](validation/dataset-sdist-rebuild-v2-preservation.json)
includes both source archive copies, the rebuilt wheel and job logs/API metadata.

The first failed job's source revision/log are retained. Its newly generated
sdist was not uploaded because collection originally depended on successful
unpacking; the next revision also retains `original-dist` on inspection failure.
This limitation is recorded rather than reconstructing the lost artifact and
presenting it as the original.

The source-only rebuild above covers Linux CPython 3.12. The subsequent
[compact-package matrix](https://github.com/developer0hye/ultrafast-yolo-dataset/actions/runs/34745616205)
at `a9f3b2a6a7e5c319efb6e0f27de0b412d1e7aaab` completed all **12 wheel jobs**
for Ubuntu 24.04, Windows 2022 and macOS 15 ARM, each on Python 3.10–3.13.
All terminal job states and full raw logs were read back and preserved in the
[matrix receipt](validation/dataset-compact-matrix-v1-terminal.json).

Linux/macOS core tests passed 86 cases per job; Windows passed 85 with one
Unix signal test skipped. Python 3.11–3.13 framework suites passed 97 cases on
Linux/macOS and 95 with two Unix-only skips on Windows. Python 3.10 covers core
and standalone cache tests because the pinned Torch integration needs 3.11+.
Across repeated platform executions this is 1,895 passes and 10 skips, not a
count of distinct tests. macOS Python 3.12 also emitted one DeprecationWarning
in the inherited-lock ownership test for forking an already multithreaded
process. That warning is preserved in the original job log.

This matrix builds wheels and sdists from checkout; the separate source-only
job establishes unpacked-source rebuilding on Linux 3.12. Full readback and
permanent preservation of all 12 original wheel/JUnit/sdist artifact payloads
remain open. Portable release repair and broader release gates remain open.
