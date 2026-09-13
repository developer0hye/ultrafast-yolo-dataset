# M2 content validation: 16 workers reduced latency, with higher CPU cost

The complete predeclared experiment passed: **three primers and 18 measured fresh
processes**, each checking all 500,000 image paths and 500,000 label paths. The
independent artifact audit passed after the controller exited naturally. This
compares worker settings in the same previously qualified native cache-reader
wheel; no new Rust kernel was measured.

## Measured phase results

Medians of six measured processes per setting, excluding the three primers:

| Workers | Image checks (s) | Label checks (s) | Total checks (s) | Sampled process RSS (MiB) | Process CPU time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 30.36 | 27.92 | 58.28 | 574.57 | 55.38 |
| 7 | 22.43 | 21.39 | 43.83 | 574.94 | 75.93 |
| 16 | 19.05 | 18.39 | 37.38 | 576.02 | 112.98 |

CPU time sums user and system time within each trial before taking the median;
it can exceed wall time with multiple threads. Component medians need not sum
exactly to the total median. RSS is the maximum sampled process RSS between
validation-ready and final observations, not whole-startup high-water memory or
an allocation trace.

The median matched-block ratio is **1.5566x for 4 versus 16 workers**, with an
exact percentile bootstrap interval of **[1.5017, 1.5696]**. The 4-versus-7 ratio
is **1.3325x [1.2934, 1.3504]**. Each interval enumerates all 46,656 resamples of
the six matched blocks. This is exploratory evidence on one shared host and
corpus, not a universal confidence guarantee.

The separate time medians imply a **35.9% latency reduction** from 4 to 16 workers.
Relative to the 7-worker setting used by the preceding full-startup benchmark,
the medians fall from 43.83 to 37.38 seconds: **14.7% lower**. The larger 4-worker
comparison must not be presented as an incremental gain over that earlier
7-worker experiment. Four workers is the library's current default.

The lower latency costs more CPU time and about **1.45 MiB additional sampled
RSS**. There is no memory reduction from this parameter change. Host CPU/disk,
load, available-memory and swap observations are retained in every raw report;
they include unrelated activity. OS caches were not flushed. The six worker
orders balance within-block position, but do not establish cold-cache behavior.

## Qualification and limits

All original native content, size and before/open/after identity checks remain
active. Every trial binds the same extension, cache, ordered paths and stored
fingerprint tables. The complete raw/log hashes, child commands, natural zero
exits, full path counts, timing sums and sampled RSS maxima passed a separate
[auditor](../bench/summarize_validation_workers.py).

Its [22 tests](../tests/test_validation_workers_audit.py) passed with zero skips:
one complete synthetic graph, 15 damaged-evidence cases, an independent
multinomial bootstrap oracle, four invalid-JSON cases and CLI failure/output
preservation. These fixtures are synthetic checker tests, not performance data.
Ruff checks and formatting passed.

The follow-up actual startup worker now accepts an explicit `scan_workers`
argument and reports the effective setting. Six **1,003-pair functional pilots**
passed: 4/7/16 workers, each with cache rebuild and cache hit. Every case matched
the frozen pilot's complete labels, first batch and diagnostic hashes. All six
actual generated/reused cache files also matched the original cache bytes.
Source/wheel/installed identities and full fixture hashes were checked before
and after. This qualifies the pilot's functional scope, not a full-size startup
speedup.

The next gate is a repeated full-size constructor/first-batch comparison. The
[seven-versus-sixteen-worker cache-hit confirmation](STARTUP_WORKER_CONFIRMATION.md)
has launched with two separate primers and five alternating measured pairs.
Keep the library default unchanged until full startup and workload tradeoffs
are established. These results do not establish Windows/Linux worker tuning,
Segmentation startup, steady-state training gains or long-run storage lifetime.

## Preserved evidence

The [96-member archive](../bench/results/validation-workers-m2-v1-evidence.tar.gz)
contains all 21 raw reports/logs and system samples, final audit, checker source
and tests/JUnit, original pilot source, exact startup pilot sources/reports/cache
bytes, and the qualified baseline runtime bundle. The earlier full-cache bundle
and fixture manifests are included as anchors; the large cache is already
preserved there. Every archive member passed byte and SHA-256 readback.

Archive: **4,186,427 bytes**, SHA-256
`b883fb48fac1ee35375c61230a236c09309faa5cacb3ab79878750d4bfcd8c8e`.

See the [preservation manifest](validation/validation-workers-m2-v1-preservation.json),
[full phase audit](validation/validation-workers-m2-v1-audit.json),
[checker JUnit](validation/validation-workers-audit-m2-v1-tests.xml),
[startup pilot qualification](validation/startup-worker-tuning-m2-pilot-v1-qualification.json),
and [separate pilot readback](validation/startup-worker-tuning-m2-pilot-v1-readback.json).
