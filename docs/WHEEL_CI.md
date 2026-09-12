# Wheel compatibility CI

The prepared workflow builds wheel and source archives on Ubuntu 24.04 x86_64,
Windows 2022 x86_64 and macOS 15 arm64 with CPython 3.10–3.13: 12 jobs. GitHub
Actions are pinned to full commits, Rust to 1.98.0 and Maturin to 1.15.0; Cargo
resolution is locked. This workflow has passed actionlint 1.7.12 locally. No
hosted run or platform result is claimed yet.

Each job installs the built wheel into a fresh NumPy/Pillow environment, checks
that imports come from that installation, and audits wheel RECORD and all
installed runtime/data bytes. Every source notice must match the wheel, sdist
and installed copy. Four standalone Detect/Segment content/metadata cache cases
exercise cold and warm reads, exact labels/diagnostics and content invalidation.
The parser, hashing and materialization tests then run against the installed
wheel. The prepared filesystem and startup-artifact checks run in the same core
job; the newly added 24 artifact checks still await their first local test run.
Python 3.11–3.13 also run scan, cache and integration suites with the pinned
Ultralytics commit and numerical dependencies. Python 3.10 uses NumPy 2.2.6 for
core-only coverage; it is not a validated framework-adapter profile. Artifacts
and reports are retained for 14 days.

The build checks the pinned Rust toolchain's standard-library copyright HTML
against the supplied copy. A mismatch requires collecting the relevant platform
notices, not removing the check. See [RUST_DEPENDENCIES.md](RUST_DEPENDENCIES.md)
for source collection scope and the distinction from final redistribution review.

This workflow does not establish portable release artifacts. Linux wheels still
need manylinux repair and audit; a macOS 11 deployment target is not execution
evidence on that OS; Windows requires downstream runtime dependency inspection.
Broader codec coverage, actual performance gates, final notice review and an
exact release-candidate validation remain open. Repository creation and hosted
execution await the user's public/private repository choice.
