# Completed 500k Segmentation content-cache hits

All five alternating reference/native pairs and both separate cache primers
completed on Linux. All full-label, first-batch and ordered-diagnostic hashes
match the completed cache-generation and capacity reference results. Server and
local independent artifact audits agree exactly.

| Metric | Reference median | Native median | Reference/native | Paired 95% interval |
|---|---:|---:|---:|---:|
| Constructor | 18.9276 s | 18.1795 s | 1.0412× | 1.0349–1.0457 |
| Constructor plus first batch | 18.9615 s | 18.1980 s | 1.0420× | 1.0357–1.0465 |
| Constructor process high-water RSS | 4.4047 GiB | 2.6387 GiB | 1.6692× | 1.6688–1.6697 |

Median constructor process high-water RSS is **40.09% lower**. The approximately
0.75-second constructor saving is modest; the 1.5× cache-hit speed target remains
unmet. The confidence intervals enumerate all 3,125 ordered paired bootstrap
resamples of five pairs. Primers and capacity runs are excluded. These intervals
describe the observed trials on one shared host, not independent machines.

| Pair | Execution order | Reference seconds | Native seconds |
|---|---|---:|---:|
| 1 | reference → native | 18.9450 | 18.3064 |
| 2 | native → reference | 18.9944 | 18.1652 |
| 3 | reference → native | 18.9276 | 18.1197 |
| 4 | native → reference | 18.8773 | 18.1795 |
| 5 | reference → native | 18.8732 | 18.1822 |

## Scope and comparison policy

This is actual constructor and first-batch work on 500,000 synthetic image/label
pairs, one million distinct files, using eight scan workers and zero DataLoader
workers with batch size eight. The [cache-generation report](SEGMENT_500K_STARTUP.md)
records the same fixture, CPU/RAM/storage, source/wheel identities and output
hashes. The RTX 3070 is not used. These are startup results, not epoch throughput,
accuracy, or evidence for the later reader/scratch/binary-digest candidates.

Both backends validate input content using the same Rust fingerprint engine.
The reference comparator is a benchmark-only trusted NumPy/pickle cache, with
an input-root key and cache SHA-256 verified before unpickling. It is primed on
this verified immutable fixture; it is not the ordinary weak reference cache
policy or a general race-safe cache builder. Native cache hits validate their
payload and materialize mutable annotation arrays. This comparison does not
attribute the result to Python-versus-Rust hashing or omit native validation.
Neither side repeats Pillow image decoding on a successful content-cache hit.

The native cache remains **larger**: 808,173,521 versus 769,130,128 bytes.
The 2 GiB container and 256 MiB metadata limits remain unchanged. Repeated
synthetic content, full input checks before each timed worker and warm OS caches
limit generalization to natural datasets and cold storage. Background activity
was not experimentally eliminated; no other agent benchmark/build/test ran on
the server during this campaign.

RSS is the process high-water mark sampled immediately after construction:
4,729,466,880 versus 2,833,309,696 bytes. It includes imports, preflight and
allocator history. It is not isolated allocation, process-family RSS, GPU
memory, or the process-exit peak after post-run checks. Load/RAM snapshots are
sampled after full post-run file verification and do not measure concurrent
constructor pressure; see [measurement boundaries](BENCHMARKS.md).

## Audit and retained evidence

All ten measured workers found 500,000 labels with no empty/missing/corrupt
entries, zero diagnostics and no native fallback. The independent audit checks
raw child/parent equality, order/counts, separate primer proofs, output equality,
cache-hit stages, valid timing/memory, and parent medians/p95. The independently
enumerated intervals replace the parent's random bootstrap endpoints.

- [Raw parent report](validation/segment-500k-linux-p4-report-v1.json).
- [Independent audit and exact statistics](validation/segment-500k-linux-p4-audit-v1.json).
- [Preservation receipt](validation/segment-500k-linux-p4-preservation-v1.json).
- [35-file evidence archive](../bench/results/segment-500k-linux-p4-complete-v1.tar.gz).

The archive includes ten worker JSON/log pairs, two primer JSON/log pairs,
parent report/log, server audit/log, completed controller state, anchors, auditor
and collector. Its embedded sealed P1 archive retains the exact source/wheel,
fixture manifests, capacity records and storage inventory. Every outer member
and published artifact was read back and compared. Archive size is 1,256,174
bytes; SHA-256 is
`741936e4350276cc5401d7fa659b67364e49cdac8b5e35e2a1bafc5bed29998d`.

The million input files and generated cache bodies remain external; the
collector does not reread them or repeat runtime tests. The baseline Linux wheel
still lacks the later 137-file dependency-license bundle recorded in the P1
audit. A redistribution-qualified rebuild, later-candidate qualification,
platform coverage and actual training/lifecycle evidence remain required.
Neither library is release-ready.
