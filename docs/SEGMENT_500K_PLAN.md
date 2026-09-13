# Full 500k Segmentation startup: capacity and execution plan

The 500,000-pair Segmentation workload remains required and has not run. Source
review shows that its startup memory requirement must not be estimated by
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
Keep the current server DataLoader comparison and its fresh-reference audit
isolated until they terminate. M2 Detection P4 has completed. Do not launch fixture creation or a second
benchmark concurrently on either host.

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

This plan corrects a capacity assumption; it is neither a generated corpus nor
a passing full-scale memory or performance result.
