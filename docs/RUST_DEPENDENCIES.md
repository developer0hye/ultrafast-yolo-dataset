# Rust dependency notice inventory

`validation/rust-dependency-notice-inventory.json` records the current lockfile's
58 resolved registry packages, including build-time dependencies and packages
for inactive platforms. Each cached `.crate` archive was verified against its
Cargo.lock checksum; every recorded top-level license/notice file was compared
byte-for-byte with that checksum-verified archive. This is a source inventory,
not the actual linked set for a particular wheel.

56 packages contain top-level notice files. The GNU support crates
`winapi-i686-pc-windows-gnu` and `winapi-x86_64-pc-windows-gnu` declare a license
expression but have no top-level notice file; this needs explicit resolution
before claiming complete coverage. The inventory also retains combined and
exception-bearing expressions such as Unicode-3.0 and the LLVM exception, rather
than treating every dependency as interchangeable MIT text.

Remaining work is to collect applicable embedded-source and Rust standard
library/toolchain notices, bundle the relevant texts into wheel/sdist artifacts,
and inspect the resulting files on supported platforms. Current installed-wheel
test success does not establish complete third-party notice packaging.
