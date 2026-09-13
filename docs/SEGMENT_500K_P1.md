# Completed 500k Segmentation label-engine comparison

All five alternating reference/native pairs completed with identical packed
outputs. This P1 experiment measures **label file read, parse, validate and
equivalent packed export**. It excludes image verification, full dataset
construction, cache generation/hits, first batches and training.

| Metric | Reference median | Native median | Reference/native | Paired 95% interval |
|---|---:|---:|---:|---:|
| P1 elapsed time | 113.7618 s | 6.7693 s | 16.8055× | 16.0814–20.1145 |
| Process peak RSS | 2,518.14 MiB | 1,588.98 MiB | 1.5847× | 1.5798–1.5859 |

Median process peak RSS is **36.90% lower**. This is process high-water RSS
sampled after the timed region, including imports, input preflight and allocator
history. It is neither isolated temporary allocation nor process-family/GPU RAM.
The ratio is formed from the two medians; intervals enumerate all 3,125 ordered
paired bootstrap resamples of the five pairs.

| Pair | Execution order | Reference seconds | Native seconds |
|---|---|---:|---:|
| 1 | reference → native | 115.4270 | 6.6678 |
| 2 | native → reference | 113.2280 | 7.0086 |
| 3 | reference → native | 113.9921 | 7.0884 |
| 4 | native → reference | 113.7618 | 5.6557 |
| 5 | reference → native | 113.2893 | 6.7693 |

## Workload and scope

The fixture contains 500,000 distinct regular single-link TXT files, totaling
1,584,173,000 bytes. Its 1,000 synthetic templates repeat 500 times; file count
does not imply unique natural annotation content. Objects per file are 1/10/100
at 80%/19%/1%, with vertex counts cycling through 3/16/100. The fixture has
1,850,000 objects and 71,821,500 raw polygon points. The unchanged generator and
manifests are retained. Full label names/content are checked before and after
each worker outside timing, deliberately warming OS caches. No cold-storage or
general SSD-throughput claim is made.

Both backends use four workers and export equivalent packed numeric arrays.
The reference retains the pinned label verification semantics and ThreadPool
style, followed by Python-side packing. Native uses Rust parsing and packed
export. Therefore 16.81× describes the complete P1 workload, not parsing alone
or the speed of an ordinary Ultralytics constructor. All ten output hashes are
`c3e1f866a756e500b7148297d27a276d9d15d3067583a124b82173a34bfdf9cf`,
with zero unhandled/reference-required statuses.

The host is the Intel i5-10400 server (6 physical/12 logical CPUs, 33,547,984,896
bytes RAM), Linux x86_64/glibc 2.39, CPython 3.12.14. Its RTX 3070 is not used by
this CPU workload. Post-P1 storage inventory places the fixture on an ext4 NVMe
partition with `rw,relatime`; device inventory is retained. Background load and
available RAM were recorded per worker. Only this measurement campaign occupied
the host's build/test/benchmark slot.

## Build identity and preserved evidence

These results use the original tested Linux wheel, not the newer M2 reader,
snapshot-scratch or binary-digest candidates. Wheel SHA-256 is
`b40a103f4234c9853150af699a20cd6eb38bce1bdf8357d2bb79cd90119c7f0e`,
and extension SHA-256 is
`274dc3ae164435e9fa5f588aec3d184a4faffc50a38d6bf8b50273c86f78f7b1`.
The selected source is `3c4610c437122af08a3db26840bbbe3cf7c12ea6`;
all 176 regular source files, native compiled profile and wheel runtime Python
files were checked. The source TAR also has 69 directory entries.

The independent auditor passed on the server and produced exactly the same
result when rerun locally against the copied artifacts. See the
[raw report](validation/segment-500k-linux-p1-report-v1.json),
[audit](validation/segment-500k-linux-p1-audit-v1.json) and
[preservation receipt](validation/segment-500k-linux-p1-preservation-v1.json).
P1 embeds worker records in its parent JSON; separate child JSON was not retained
by that harness. This limitation is explicit in the audit.

The [19-file archive](../bench/results/segment-500k-linux-p1-complete-v1.tar.gz)
contains the raw report/log, audit/log, identities, full preflight and manifests,
source archive/selection, exact measured wheel, controller checkpoint, separate
capacity records, storage inventory and audit/collection scripts. It was reopened
and every member size/hash checked. Archive SHA-256 is
`4092e95c9d7c4e1cf982c293c2192848fa874c9f981e0913c78d530bcd210fdb`.
The million input files are not embedded; reproducibility uses the retained
generator, declared distribution and complete input fingerprints.

Collection also exposed a release gap: this older Linux wheel contains the
project LICENSE/NOTICE but lacks the later 137-file dependency-license bundle
present in the selected source. Runtime code identity and benchmark validity are
checked separately; wheel redistribution qualification remains open. Missing
paths are listed in the preservation receipt and archive's wheel-source audit.

The separate [capacity runs](SEGMENT_500K_PLAN.md) already produced identical full
constructor outputs. [Repeated P3 construction](SEGMENT_500K_STARTUP.md) has now
completed at 2.4053× with 13.86% lower constructor process RSS. P4 content-hit
measurements are still running. This P1 gain cannot substitute for either phase
or for training throughput.
