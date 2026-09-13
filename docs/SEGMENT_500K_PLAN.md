# Full 500k Segmentation startup: capacity and execution plan

The full 500,000-pair Segmentation fixture and both capacity runs have completed
on Linux. All output hashes match; five-pair P1 is complete and audited, while
P3/P4 remain in progress. See [the P1 results](SEGMENT_500K_P1.md).
Source review shows that its startup memory requirement must not be estimated by
resampling all instances to 1,000 points. That would be 14.8 GB (13.78 GiB) of
coordinates, but the pinned implementation does not retain that representation
for the entire dataset at construction time.

`BaseDataset.__init__` loads label dictionaries and builds transforms.
`YOLODataset.get_labels` returns the raw segment lists. Later,
`BaseDataset.get_image_and_label` deep-copies one label and invokes
`YOLODataset.update_labels_info`, which resamples that sample's polygons.
The native `materialize_labels` also copies each original segment's `b-a` points
into its independently owned array. It does not expand every annotation to
1,000 points. The source hashes were checked against the pinned Ultralytics
commit `795a556942a12fe0124cf767888194a1d0b83e2e`.

For the unchanged synthetic generator, 1,000 templates repeat 500 times. Objects
per file are 1/10/100 at 80%/19%/1%, and raw vertex counts cycle through 3/16/100.
Their correlation is included in the arithmetic; multiplying two independent
averages is not used.

| Quantity | Static count / payload |
|---|---:|
| Distinct image/label pairs | 500,000 |
| Objects before scan validation | 1,850,000 |
| Raw polygon points | 71,821,500 |
| Raw float32 coordinates | 574,572,000 bytes / 0.535 GiB |
| All eight packed numeric arrays | 657,172,016 bytes / 0.612 GiB |
| Two fixed-width fingerprint tables | 57,000,000 bytes |
| Label text | 1,584,173,000 bytes / 1.475 GiB |
| Label file payload rounded to 4 KiB blocks | 3,086,336,000 bytes |

The label-text calculation matches five times the retained 100k Segmentation
manifest. See [capacity receipt](validation/segment-500k-capacity-plan-v1.json)
for the per-array arithmetic and seven inspected source identities.

These numbers are payload estimates, not peak RSS estimates. They exclude
Python array/dictionary/list objects, paths, JSON metadata, parsed intermediate
values, serialized copies, input image buffers and allocator history. Reference
and native startup may retain several representations simultaneously. File
block rounding excludes JPEGs, caches and filesystem metadata. The native
container has a 2 GiB default total limit and a 256 MiB metadata-section limit;
record the actual generated section sizes rather than silently changing those
limits for the benchmark.

## Execution after the current host jobs finish

Use the 32 GB Linux server for the complete Segmentation fixture and benchmark.
The server DataLoader comparison and its fresh-reference audits have completed.
M2 is reserved for the separate snapshot-scratch comparison. Run the Linux
Segmentation controller alone on that server.

1. Select and record a tested installed dataset wheel, exact source snapshot,
   Python/dependency/toolchain identities, CPU/RAM/storage and current load.
   The cache-buffer candidate has built on M2 and passed all 187 installed tests;
   existing full Detection startup results apply to its frozen baseline. Linux
   validation and measured startup gains for that candidate remain open.
2. Generate a new full 500k Segmentation fixture with the existing templates;
   retain the manifest and verify all one million regular, single-link files,
   content hashes and physical-file distinction. Repeated template content
   remains explicit; this is not a natural-image diversity experiment.
3. Establish full-scale capacity with real reference and native construction,
   measuring constructor/first-batch RSS, actual cache size, scan counters,
   fallback counts and all output bytes. This qualification is separate from
   the paired timing series. Preserve failures; do not reduce the final dataset
   size or vertex counts to make a run pass.
4. Run P1 label read/parse/validate/export with five counterbalanced pairs. Then
   run P3 cache generation and P4 content-validated cache hits with five pairs
   each, equivalent image verification, separately primed hit processes and
   full mutable-label/first-batch parity. Use the audited Detection protocol;
   label parsing gains cannot stand in for constructor gains.
5. Independently audit all raw records, input/runtime identities, memory scope
   and full paired statistics. Report any regression and unsupported target.
   Compare cache read-buffer and save-lifetime changes only after their new
   installed wheel passes corruption, ownership, concurrency, failure and
   framework tests. Attribute the two changes separately where possible.

The arithmetic above corrects a capacity assumption. Actual capacity results are
recorded below separately from the ongoing repeated performance campaign.

## Linux launch after loader verification

The mask timing job, both fresh-reference passes and the full artifact audits
have terminated successfully; 44 damaged-evidence cases were rejected and the
complete loader archive was sealed before this launch. The server now runs the
[full Segmentation controller](validation/run-segment-500k-linux-v1.py), starting
with a new 500,000-pair fixture. The source snapshot is commit
`3c4610c437122af08a3db26840bbbe3cf7c12ea6`, with all 176 selected library/build/test/
Python benchmark files preserved and verified after transfer.

The existing tested Linux wheel remains the baseline: wheel SHA-256
`b40a103f4234c9853150af699a20cd6eb38bce1bdf8357d2bb79cd90119c7f0e`, extension
`274dc3ae164435e9fa5f588aec3d184a4faffc50a38d6bf8b50273c86f78f7b1`.
Installed runtime Python/profile bytes match both the wheel and source snapshot.
Its compiled aggregate source profile matches all six Rust sources and Cargo
manifests. This is not the M2-only reader/scratch candidate. CPU/RAM, package
versions, source inventory, initial free storage/load and worker counts are in
the [launch identity](validation/segment-500k-linux-launch-v1.json).

The controller preserves preparation and complete distinct-file preflight,
then performs separate full reference/native capacity runs without increasing
the 2 GiB container or 256 MiB metadata limits. If capacity and output parity pass,
it runs P1, P3 and P4 with five pairs each, independently auditing each phase.
Failures stop the sequence and retain logs; it does not reduce file/vertex counts.
Artifacts use `/home/yonghye/ultrafast-vision-build/dataset-segment-500k-linux-v1*`;
the new corpus is `startup-segment-500k-linux-v1`.

## Completed full-capacity qualification

The complete one-million-file preflight and both 500,000-pair constructors passed.
All labels, the first batch, and scan counters/diagnostics match, with 500,000
found pairs, zero corrupt/empty/missing pairs and zero native fallback. The
controller advanced to the phase campaigns. P1 subsequently completed and passed
both server and local artifact audits. These two capacity workers are
separate from, and will not be pooled into, the repeated phase measurements.

| Capacity worker | Constructor | Peak process RSS | Cache bytes |
|---|---:|---:|---:|
| Pinned Ultralytics reference | 193.5335 s | 4.1254 GiB | 769,130,128 |
| Original tested Linux native wheel | 79.2129 s | 3.5526 GiB | 808,173,521 |

This is one sequential reference/native qualification pair. It establishes that
the full fixture fits and produces identical outputs, not a repeated speedup or
memory-improvement conclusion. OS-cache warming and background load are not
controlled; native generation also performs stronger captured-input validation
than the ordinary reference's weak cache policy. RSS includes imports/preflight.

The native cache's 11 sections use 94,001,049 bytes of metadata, 657,172,016 bytes
of numeric arrays and 57,000,000 bytes of fingerprint tables, plus 456 bytes of
container overhead. The raw-coordinate section is exactly 574,572,000 bytes,
matching the static estimate. No 2 GiB container or 256 MiB metadata limit was
increased. The native disk cache is larger than the reference cache.

The [capacity checkpoint](validation/segment-500k-linux-capacity-checkpoint-v1.json)
binds the copied original worker JSON, complete file preflight, launch identity
and controller checkpoint by SHA-256. Those copies were read back after transfer.
It verifies output/count agreement and cache-size arithmetic; it does not rerun
the workload or independently re-read all inputs/cache sections. Final P1/P3/P4
conclusions require the complete phase records and their independent audits.

## Prepared P3/P4 artifact collection

`docs/validation/collect-segment-startup-v1.py --phase p3` (or `--phase p4`)
is prepared for an explicit Python invocation after that phase and its server
audit have terminated successfully. It has passed source syntax inspection only;
it has not yet collected either live phase and is not automatically queued.

The collector checks the controller's successful phase/audit commands, retains
every raw worker JSON and log (including both separate P4 primers), and reruns
the qualified standard-library artifact auditor locally. It requires agreement
with the server audit and the full-capacity output hashes. Each resulting archive
embeds the already sealed P1 evidence with the exact source/wheel and fixture
manifests; the documented Linux dependency-notice gap remains explicit. Every
archive member and copied report is read back before publishing the receipt.

Collection does not reread the million input files or generated cache bodies,
repeat ML tests, or establish performance for newer candidates. P3 collection
may run while P4 is active because it only copies the terminal phase's small
artifacts. Neither collector success nor a P3 checkpoint certifies P4 completion.
