# Linux wheel and combined-runtime qualification

A fresh Linux CPython 3.12.14 wheel built from dataset commit
`16d4d6175e749c7213fc3bd24caed7815b59c053` passed standalone packaging/cache checks
and the complete dataset and mask test suites in one new runtime. This closes
the recorded Linux dependency-notice packaging gap for this local wheel. It is
not hosted platform-matrix, GPU-training or new performance evidence.

## Actual checks

- Rust 1.98.0, the version specified by the wheel workflow: formatting, four Rust
  unit tests and Clippy with warnings denied passed. The installed compiler's
  standard-library copyright HTML exactly matches the bundled 1.98.0 copy.
- The new wheel, sdist and installed package preserve all **137 notice files**.
  Wheel RECORD and installed runtime/native bytes passed the archive audit.
- A fresh environment containing only NumPy, Pillow, pip and the dataset wheel
  passed all four Detect/Segment content/metadata cold/warm cache cases, reference
  label/diagnostic parity and restored-mtime content-edit invalidation.
  Torch, OpenCV, Ultralytics and maskops were absent during these standalone checks.
- The same environment then received the pinned framework dependencies and the
  previously qualified mask packet wheel. All **79 dependency pins** outside pip
  and the two project packages match the frozen Linux baseline; `pip check` passed.
- **166 dataset tests** and **257 mask tests** passed in separate fresh Python
  processes in that combined environment, with no failures, errors or skips.
  Each executed count matches the separately recorded collection list. Both
  wheel payloads were audited after installation, and all tracked runtime source
  bytes were checked again after the suites.

The runtime is
`/home/yonghye/ultrafast-vision-build/dataset-notices-linux-v2-clean` on `yonghye-pc`.
Existing measured environments were not modified. The mask source/tests belong
to the frozen Linux packet qualification, not later synthetic coordinator tests.
This does not resolve the separately observed `file_system` transport failure;
the GPU launcher explicitly verifies Linux's `file_descriptor` setting.

## Identities and failed attempt

| Artifact | SHA-256 |
| --- | --- |
| Dataset wheel, `cp312-cp312-linux_x86_64` | `052885ef0c3dad89d62752fa4f4f6442e699c34cfdda52443dfdcde53210677c` |
| Dataset sdist | `b959abc0ee2abb7dcd0ce5ad05816dbdc93eaae83f17da7a1b9ee0ec464a4db8` |
| Frozen dataset source tar | `080b9250282c05fd04d4e7faf79372295d7106cfb6ea9d26dabe05f2babf7311` |
| Installed mask wheel | `8b467c837c323812a0201b81210a6dcd0c180ef50ba0d16531f07ef2c741cd9a` |
| Completed qualification receipt | `3cd9cf0bfc2f486c682be2df80f5140f1a9b3fb861b183fff20e71b5ebe2a3b9` |

The original attempt used the host's default Rust 1.97.1. Its build and four Rust
unit tests passed, but Clippy rejected the newer
`clippy::chunks_exact_to_as_chunks` lint name. That attempt is **failed**, with its
original wheel/sdist and logs retained. Rust 1.98.0 was installed alongside the
existing toolchain, and a new source directory, Cargo target directory, artifact
directory and environment were used. No lint suppression or runtime-source change
was introduced to obtain the successful second result.

The selected committed source archive includes build/runtime/test files and the
benchmark helpers, excluding historical result bundles. All 179 selected files
match the committed source manifest. The wheel's generic Linux tag is not a
manylinux ABI compatibility claim. The new compiler/wheel has no new startup or
throughput comparison; earlier benchmark numbers keep their original identities.

## Preserved evidence and next gate

The [55-member evidence bundle](../bench/results/dataset-notices-linux-v2-evidence.tar.gz)
is 4,880,832 bytes, SHA-256
`694a113d7410e5b7a0be7f8ea15c20ab8cb41602c8d190debd4c128183bbfb3c`.
It includes both attempts, exact source/wheel/sdist, all command logs and test
collections/JUnit, standalone results, installed-byte audits and the frozen mask
wheel's qualification bundle. Every member was read back and hashed; see the
[preservation manifest](validation/dataset-notices-linux-v2-preservation.json),
[successful receipt](validation/dataset-notices-linux-v2-qualification.json), and
[original rejected attempt](validation/dataset-notices-linux-v1-rejected.json).

The successful runtime is now selected for the actual 54-trial, 126-epoch
close-mosaic/resume GPU grid in the mask trainer-lifecycle worktree. Its initial
reference trial has started. Qualification here does not claim that grid passed;
full cohort parity, shutdown/resume behavior and independent final artifact
verification still require the actual training results. Hosted wheel/platform
execution and release/publication also remain open.
