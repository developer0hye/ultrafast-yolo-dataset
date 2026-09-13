# Private development snapshot — 2026-09-13

The owner approved uploading this project to `developer0hye/ultrafast-yolo-dataset`
as a private GitHub repository. This preserves development history and evidence;
it is not a release or a claim that all replacement gates have passed.

All seven existing branches are retained. The default `perf/validation-workers`
branch contains the latest M2 worker-scaling evidence and startup confirmation
protocol. It inherits the **unpromoted** scratch-buffer candidate; the running
startup campaign uses the earlier qualified `perf/cache-read-copy` runtime.
`feat/native-core` retains the base implementation; `build/linux-notice-wheel`
contains the Linux combined-runtime qualification. `bench/startup-audit`,
`perf/snapshot-scratch` and `perf/binary-fingerprint` retain their separate
investigations and limitations. Branches are not implicitly merged by this upload.

The 12-process, 500,000-pair startup confirmation is still running. Its complete
result and independent final audit are pending. The new
`bench/audit_startup_workers.py` was initially uploaded as a draft. On the
separate `bench/startup-worker-audit` branch it now passed 37 hosted synthetic
tests with preserved JUnit/source/artifact evidence. See
[STARTUP_AUDIT_QUALIFICATION.md](STARTUP_AUDIT_QUALIFICATION.md). It reuses the
earlier independent startup audit's cache, result and paired-bootstrap checks
and adds campaign/telemetry binding. The complete measured result remains pending.

Hosted wheel CI has not yet qualified this snapshot. Automatic Actions were
disabled for the initial multi-branch upload to avoid launching a separate
12-job wheel matrix for every historical development branch. They are now
enabled, and the first full matrix is running at the frozen `e1c5861` revision.

Tracked raw reports, manifests and evidence archives are included. Large local
datasets, build environments and live experiment output remain on their hosts;
completed final evidence will be added after the running campaigns terminate.
Git packaging/upload on the shared M2 host overlaps the startup campaign and
must be disclosed when interpreting that campaign's shared-host measurements.
