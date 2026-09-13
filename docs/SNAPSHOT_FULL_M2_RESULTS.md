# M2 500,000-pair snapshot scratch result: no established improvement

The full native-versus-native experiment completed all 20 fresh measured
processes and two separate cache primers. The candidate reuses per-thread
fingerprinting scratch space; the baseline is the previously tested native
cache-reader wheel. This is not a Python-versus-Rust or training comparison.

Both versions processed the same 500,000 Detection image/label pairs (one million
physical files) on the Apple M2, 16 GiB RAM host using the T7 volume. Each mode
used five alternating pairs, seven scan threads and zero DataLoader workers.
No sample was discarded. Cache miss rebuilds the native cache; content hit uses
the generated cache while retaining the content-validation contract.

| Mode / constructor metric | Previous native median | Scratch candidate median | Baseline / candidate | Paired bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Miss time | 121.324 s | 125.457 s | 0.9671 | 0.9669–1.5055 |
| Content-hit time | 53.787 s | 53.275 s | 1.0096 | 0.9984–1.0304 |
| Miss process high-water RSS | 1834.625 MiB | 1836.172 MiB | 0.9992 | 0.9602–1.0010 |
| Content-hit process high-water RSS | 1266.813 MiB | 1266.953 MiB | 0.9999 | 0.9996–1.0001 |

The candidate miss-time median is 3.41% higher; its hit-time median is 0.95%
lower. Every displayed interval includes 1. The experiment does not establish
a speed or process-memory improvement, and does not justify promoting this
candidate as a performance win. Both cache files are exactly 195,001,501 bytes;
their annotation sections and metadata except the implementation profile match.

The first-batch totals support the same interpretation: miss 121.344→125.478 s,
hit 53.860→53.305 s, with both speed-ratio intervals crossing 1. RSS includes
imports, preflight and allocator history; it is not isolated temporary allocation.
Summed process-family RSS was not measured by this startup experiment.

## What to optimize next

For content hits, the separately instrumented `_validate_inputs` median is
45.037 s for the baseline and 44.582 s for the candidate. Cache decode is only
0.286 s and 0.294 s, respectively; annotation materialization is 0.919 s and
0.930 s. These are medians of individual observed stages, not additive components
of a median trial. Input validation is the largest observed stage, so additional
cache-decoding or small scratch-allocation changes alone have limited room to
improve this workload. Further optimization must retain content checks, ordering,
error behavior and cache invalidation; weaker checking is not an equivalent win.

The two slower baseline miss samples (176.262 s and 188.882 s) remain in the
raw five-pair record and interval calculation. Host pressure snapshots were taken
after full untimed checks and cannot identify the cause of constructor timing
variation. Five pairs and a shared host limit the strength of inference.

## Validation and provenance

The [complete launch receipt](validation/snapshot-startup-full-m2-v1-launch.json)
records exit code zero and agreement with the original full-reference output.
The [qualified audit](validation/snapshot-startup-full-m2-v2-qualification-audit.json)
checks all raw/log hashes, alternating execution order, distinct worker PIDs,
non-overlapping worker intervals, source/wheel profiles, complete scan counts,
retained cache bytes, label/first-batch/diagnostic hashes and all aggregate metrics.
It enumerates all 3,125 ordered paired resamples to recompute each interval.

The [negative qualification](validation/snapshot-startup-full-m2-v2-qualification-negative.json)
passed three controls and rejected all 35 corrupted artifact cases. Two controls
are explicitly synthetic five-round expansions; they validate aggregation and
supply no measured performance evidence. The
[execution receipt](validation/snapshot-startup-full-m2-v2-qualification.json)
records all five commands returning zero and AST-preserving formatting.

Baseline wheel SHA-256:
`25f0ce7ff2e40f106ec7af04a744c3883671c4ee502f3bd337e2414fe01b88d3`.
Candidate wheel SHA-256:
`4d5dcc5ea64979f769d7c7df801bac3d217bbf30006becc00422608ee94fef7d`.
Full report SHA-256:
`0d293d4397166e283efc65101fbc78c0b92a81504b29894b682a412423e6e833`.

The [preservation receipt](validation/snapshot-startup-full-m2-v1-preservation.json)
binds all 129 archive members, including both complete generated caches, all
raw results/logs, frozen sources, wheels, reference anchors and audit execution.
The complete archive is 114,347,086 bytes, SHA-256
`306c32152ab97584eb1b51765cb383a30a6d4f0e65cb3eca996705d9e9dd6fa4`.
Every member passed readback at creation and again after copying locally.

The repository retains four parts no larger than 32 MiB, described by the
[bundle manifest](../bench/results/snapshot-startup-full-m2-v1-bundle.json).
Their ordered concatenation was checked against the original archive hash and
every reconstructed member's size and hash. Reconstruct from the repository root:

```sh
cat bench/results/snapshot-startup-full-m2-v1-evidence.tar.gz.part0[0-3] > /tmp/snapshot-startup-full-m2-v1-evidence.tar.gz
shasum -a 256 /tmp/snapshot-startup-full-m2-v1-evidence.tar.gz
```

Broader library replacement, Linux and Windows candidate qualification, GPU
training and release readiness are not established by this result.
