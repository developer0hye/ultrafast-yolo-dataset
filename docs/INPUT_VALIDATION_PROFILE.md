# M2 input validation: file access dominates the sampled window

After the complete startup campaign terminated, two read-only diagnostics ran
against the **previous native cache-reader wheel**, not a new optimization. The
[harness](../bench/profile_input_validation.py) selects evenly spaced image and
label entries from the verified 500,000-pair cache and calls the same native
fingerprint-table verifier used by `_validate_inputs`.

The 20,000-pair diagnostic completed 16 calls: images/labels × 1/7 workers ×
content/metadata modes × two rounds. The second round reverses condition order.
Every selected table verification succeeded, and the extension/cache hashes
matched before and after the run. The cache was not rewritten.

| Unchanged implementation, 20,000 paths, one worker | First occurrence | Later occurrence |
| --- | ---: | ---: |
| Image content verification | 9.743 s | 0.399 s |
| Label content verification | 9.053 s | 0.327 s |
| Image metadata verification | 0.0318 s | 0.0318 s |
| Label metadata verification | 0.0320 s | 0.0319 s |

These are raw within-process observations. No OS cache flush was performed;
the first occurrence is not a controlled cold-cache run and the later occurrence
is not a separately controlled warm-cache benchmark. Execution history and
working-set residency are major confounders. In particular, the 1-worker and
7-worker results cannot establish a concurrency speedup because their input-cache
state differs. Metadata verification omits content reads and is an attribution
control, not an equivalent faster replacement.

## Five-second stack observation

A separate 100,000-pair diagnostic completed eight validation calls. macOS
`sample` attached for five seconds during its first image/content/one-worker call.
The sampling command and its timestamps are retained; the entire attach interval
falls within that call. All timings from this second run are labeled instrumented.

The worker had 3,871 sampled stacks. The main thread had another 3,871 samples
waiting on the Rayon worker; these are excluded from the following denominator.

| Worker top frame | Samples | Share of worker samples |
| --- | ---: | ---: |
| `read` | 2,264 | 58.49% |
| `stat` | 1,233 | 31.85% |
| `__open` | 294 | 7.59% |
| SHA-256 `compress256` | 43 | 1.11% |

`read`, `stat` and `__open` account for 97.93% of worker observations in this
window. These include blocked system calls; they are **not CPU percentages**,
an exact full-run breakdown, or guaranteed removable time. The result supports
investigating file-access scheduling/concurrency before expecting large gains
from digest formatting or scratch-buffer allocation. It does not justify removing
the before/after identity, content, size or path-consistency checks.

The two diagnostics total 24 successful validation calls. They omit the complete
constructor, Pillow image verification and first-batch work. Subset selection
also changes locality and working-set size compared with a full one-million-file
scan. Any candidate must still pass the full startup protocol, race/error tests,
output parity and memory checks. No faster candidate has been implemented or
qualified by this profiling work.

## Preserved evidence

The [summary](validation/input-validation-profile-m2-summary-v1.json) contains all
24 raw timing/counter records and parsed stack counts. The
[30-member archive](../bench/results/input-validation-profile-m2-v1-evidence.tar.gz)
contains both complete reports and logs, sample output and attach receipt, exact
harness, baseline wheel/source/identity and the existing full-cache bundle anchor.
Every member passed byte-for-byte readback; see the
[preservation receipt](validation/input-validation-profile-m2-preservation-v1.json).

Archive SHA-256:
`7324b4d0b26faa2911ff016b6ba09f0a4d30aa1ac40fb6dd2294c7a78026a6a5`.
Archive size: 1,271,914 bytes. Runtime extension SHA-256:
`ec52a219e5e159017d221422fa8292cbe24022157370548bd445e3fb082f32d6`.
The previously sealed full-startup bundle retains the complete input cache;
the diagnostic archive references that bundle rather than duplicating it.
