# Completed 500k Segmentation cache generation

All five alternating reference/native pairs completed on the Linux server with
identical full labels, first batches and ordered diagnostics. P3 measures the
actual dataset constructor with cache generation, including Pillow image
verification, annotation validation and mutable label materialization. The
separate [P4 content-cache-hit campaign](SEGMENT_500K_HITS.md) also completed: a
1.0412× constructor ratio and 40.09% lower constructor process high-water RSS.

| Metric | Reference median | Native median | Reference/native | Paired 95% interval |
|---|---:|---:|---:|---:|
| Constructor | 191.6352 s | 79.6712 s | 2.4053× | 2.3994–2.4218 |
| Constructor plus first batch | 191.6532 s | 79.6890 s | 2.4050× | 2.3991–2.4215 |
| Constructor process high-water RSS | 4.1208 GiB | 3.5496 GiB | 1.1609× | 1.1549–1.1616 |

Median constructor process high-water RSS is **13.86% lower**. All five pairs
contribute to the ratio of medians and all 3,125 ordered paired bootstrap
resamples. No capacity run, primer or partial checkpoint is pooled into these
statistics. These intervals describe the observed paired trials on this host;
they do not establish effects across independent machines or natural datasets.

| Pair | Execution order | Reference seconds | Native seconds |
|---|---|---:|---:|
| 1 | reference → native | 192.9480 | 79.6712 |
| 2 | native → reference | 192.0090 | 79.4256 |
| 3 | reference → native | 191.2584 | 79.6891 |
| 4 | native → reference | 191.6352 | 79.4076 |
| 5 | reference → native | 191.3127 | 79.7326 |

## What this comparison establishes

Each fresh process constructs the actual reference YOLODataset or native
FastYOLODataset on 500,000 image/label pairs, using eight scan workers. It then
creates the first batch of eight samples with zero DataLoader workers. This is
startup and one-batch evidence, not epoch/training throughput or model accuracy.
The [P1 label-engine result](SEGMENT_500K_P1.md) measures a different workload;
its 16.81× ratio must not be substituted for this 2.41× constructor result.

Both sides perform image verification. Native cache generation also revalidates
captured input content; ordinary reference generation retains its weaker cache
policy. The compared operations therefore have different consistency costs,
which remain explicit rather than disabling native validation for the benchmark.
The native cache is **larger**: 808,173,521 versus 769,130,128 bytes. Neither the
2 GiB container limit nor the 256 MiB metadata limit was increased. The full
[capacity report](SEGMENT_500K_PLAN.md) gives the actual section sizes.

The fixture contains one million distinct regular single-link files, totaling
5,123,766,750 bytes. The 1,000 synthetic label templates repeat 500 times and
contain 1,850,000 objects and 71,821,500 raw polygon points before validation.
Repeated content is intentional; this is not 500,000 unique natural images.
Complete input-content checks run before and after each worker outside timing,
warming OS caches. The million files are not reread by the artifact auditor.

The host is Intel i5-10400 (6 physical/12 logical CPUs), 33,547,984,896 bytes RAM,
Linux x86_64/glibc 2.39 and CPython 3.12.14. The fixture is on the ext4 NVMe
partition recorded in the embedded P1 inventory. The RTX 3070 is not used in
this CPU experiment. No other agent build/test/benchmark was launched on that
host during the series; background activity is not experimentally eliminated.

RSS is the process high-water mark sampled immediately after construction.
It includes earlier imports, input preflight and allocator history. It is not
isolated allocation, process-family RSS, GPU memory, or the process-exit peak
after output/input verification. User/system CPU time and fault deltas span
construction and first-batch creation. Load averages and available RAM are
sampled after post-run full-file verification and cannot establish contention
during construction; see [measurement boundaries](BENCHMARKS.md).

## Output and build identity

All ten workers found all 500,000 pairs, with no empty/missing/corrupt entries,
zero diagnostics and zero native fallback. The full output hashes also match
the separately completed reference capacity worker:

- Labels: `b9f49377fff5a6d9417447d39e0566b40de91cdfcd548591168662a1927195f7`.
- First batch: `6de94f7bf381925ad9a4edcf665e46a91aee2db263e5db37fbba3abca7654209`.
- Scan summary and messages: `cc100000bc72bf22843f56481fbfa04c4d8ef8f95efb48b6a1f034b658f64616`.

This is the original tested Linux implementation, not the later reader,
snapshot-scratch or binary-digest candidates. Selected source commit is
`3c4610c437122af08a3db26840bbbe3cf7c12ea6`; exact source bytes, the selection
manifest and compiled source profile are retained in the embedded P1 archive.
Wheel SHA-256 is
`b40a103f4234c9853150af699a20cd6eb38bce1bdf8357d2bb79cd90119c7f0e`,
and installed extension SHA-256 is
`274dc3ae164435e9fa5f588aec3d184a4faffc50a38d6bf8b50273c86f78f7b1`.

## Independent audit and preservation

The server audit passed. Recalculating it locally using the qualified frozen
standard-library auditor produced exactly the same audit JSON. This checks raw
child/parent equality, complete order and counts, full output hashes, observed
cache-generation stages, memory/timing validity and parent medians/p95 values.
The reported paired confidence intervals use the independent exhaustive
resampling calculation, not the parent's random bootstrap endpoints.

See the [raw report](validation/segment-500k-linux-p3-report-v1.json),
[independent audit](validation/segment-500k-linux-p3-audit-v1.json) and
[preservation receipt](validation/segment-500k-linux-p3-preservation-v1.json).
The [31-file archive](../bench/results/segment-500k-linux-p3-complete-v1.tar.gz)
contains all ten child JSON files and logs, parent report/log, server audit/log,
controller checkpoint, anchors, auditor and collector. It also embeds the sealed
P1 archive, making the exact wheel, source, fixture manifests, capacity outputs
and storage inventory available without relying on live host paths. Archive
size is 1,356,768 bytes and SHA-256 is
`63bdd0d45193a0f6a0eeed58b346ac825c3fd15f085e16b51add9a6d86bf841b`.
Every outer archive member and published artifact was read back and compared.

Generated caches and the million input files remain external. The retained
generator/manifests support reproduction; this collection does not repeat the
runtime tests or reread cache bodies. The embedded source/wheel audit preserves
the older Linux wheel's missing 137-file dependency-license bundle finding.
Runtime identity is verified, but redistribution-qualified Linux packaging is
still required. P4 is separately documented; later candidates, broader platforms
and actual training gains retain their separate validation gates; neither library is release-ready.
