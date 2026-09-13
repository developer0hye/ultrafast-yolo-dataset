# Rust dependency notice bundle

`validation/rust-dependency-notice-inventory.json` records the current lockfile's
58 resolved registry packages, including build-time dependencies and packages
for inactive platforms. Each cached `.crate` archive was verified against its
Cargo.lock checksum; every recorded top-level license/notice file was compared
byte-for-byte with that checksum-verified archive. This is a source inventory,
not the actual linked set for a particular wheel.

56 packages contain top-level notice files. The GNU support crates
`winapi-i686-pc-windows-gnu` and `winapi-x86_64-pc-windows-gnu` declare a license
expression but have no top-level notice file. They are absent from the selected
macOS arm64, Linux x86_64 and Windows MSVC normal/build dependency trees, so the
current bundle excludes them. GNU Windows is not a covered target. The inventory also retains combined and
exception-bearing expressions such as Unicode-3.0 and the LLVM exception, rather
than treating every dependency as interchangeable MIT text.

`python/ultrafast_yolo_dataset/licenses/manifest.json` records the conservative
source bundle collected by `bench/collect_rust_notices.py`: 54 selected registry
crates, 375 distinct source-comment notice blocks, and 136 files plus the
manifest. The three target trees are retained in `docs/validation/`. Four crates
are absent from all selected trees: the two GNU support crates, `portable-atomic`
and `portable-atomic-util`. Build dependencies are intentionally included; this
does not claim every selected crate is linked into the extension.

The bundle preserves verbatim archive notice files and source comment blocks
from checksum-verified crate archives. It also includes the self-contained Rust
standard-library copyright HTML and supplied license texts from Rust 1.98.0 on
macOS and Rust 1.97.1 on the Linux benchmark host. Each supplied file has a hash
in the manifest. Changing the lockfile, target set or compiler requires reviewing
and refreshing this collection. Maturin now uses locked dependency resolution.

A rebuilt M2 CPython 3.12 wheel and sdist now preserve all 137 bundle files,
including the manifest, exactly. A fresh NumPy/Pillow installation passed the
RECORD/installed-byte audit, all four standalone cache cases and 65 parser,
hashing and materialization tests. The wheel SHA-256 is
`2ae058b33e702306d1ce89ea5cd43e7bc5fcdb87825b889f9556bfe5c864f595`.
Reports, JUnit, archive hashes and logs are retained in
`docs/validation/dataset-notices-*`. Its macOS 11 arm64 tag is not proof of an
actual macOS 11 run. The full framework suite was not rerun for this packaging
change; the earlier 157-test results remain attached to their original wheel.

The prepared compatibility workflow uses the same source/wheel/sdist/installed
notice comparison and also checks the pinned Rust compiler's copyright HTML
against the supplied copy. Hosted CI has not run. Embedded notices,
platform-specific system dependencies and final redistribution coverage still
require review.

A subsequent [Linux CPython 3.12 rebuild](LINUX_WHEEL_QUALIFICATION.md) now also
preserves all 137 files in wheel, sdist and installation. It uses Rust 1.98.0
and verifies the compiler copyright bytes, with four standalone cache cases,
166 dataset tests and 257 mask tests passing in a fresh combined runtime. The
original Rust 1.97.1 Clippy failure remains preserved. This is local packaging
and integration evidence; broader platform and final redistribution review are
not implied by the notice-file comparison.
