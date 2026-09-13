# Input-validation worker tuning: full phase experiment completed

All 21 processes and the final independent audit passed. The 16-worker setting
reduced the measured validation phase relative to 4 and 7 workers, with higher
CPU time and slightly higher sampled RSS. See the complete
[results and limitations](VALIDATION_WORKERS_RESULTS.md). Full-size startup
tuning remains a separate gate.

The [file-access profile](INPUT_VALIDATION_PROFILE.md) motivates testing worker
concurrency before changing hash computation or consistency checks. This new
[harness](../bench/validation_workers.py) compares the **same installed native
binary**, using 4, 7 and 16 workers. It does not promote the scratch-buffer or
binary-fingerprint candidates.

## Declared experiment

- The frozen cache contains 500,000 image/label pairs. Every full trial verifies
  all image and label content against its stored fingerprints, including the
  existing native pre/open/post identity and size checks.
- Three unmeasured primer processes run first, one per worker count.
- All six permutations of the three worker counts follow, producing 18 measured
  fresh processes. Each worker count appears twice in each within-block position.
  Images precede labels inside every process, matching `_validate_inputs`.
- Each process reads the same cache and checks its SHA-256 before and after. The
  native extension, runtime identity, exact ordered paths and fingerprint tables
  are bound in every report. No cache files are rewritten or OS caches flushed.
- Native verification-call times are recorded separately for images and labels.
  Cache decoding/hashing, path extraction, telemetry start/stop and report writes
  are outside those timed intervals. This is a phase comparison, not full
  constructor/startup latency.
- A 0.5-second sampler records process RSS/thread counts plus system CPU, disk,
  load, available memory and swap counters. Sampled RSS includes ready/final
  observations; it does not establish unsampled or whole-startup peaks. Host
  counters include unrelated activity and cannot be attributed solely to this job.
- Any child failure or timeout stops the series. No failed process is retried,
  and no trial is selected or discarded based on its speed.

The runtime is the previously qualified cache-reader wheel, extension SHA-256
`ec52a219e5e159017d221422fa8292cbe24022157370548bd445e3fb082f32d6`.
The full cache has SHA-256
`2e94732eee42009919f1552b702c0ba158e1a0b1b1a49933f7e599eb9a2f1dc6`.
The existing full-startup evidence bundle preserves that cache; it will not be
duplicated for every worker setting.

## Pilot and remaining qualification

A separate 1,024-pair, four-worker child pilot passed full content checks,
telemetry and before/after cache/extension hashing, then exited zero. Its
measurements are not full-dataset results. The subsequent source edit changes
only the parent report's worker-label text, not the pilot's trial function.

The full phase comparison and independent readback have now passed. Any favorable
worker setting must still be checked in full startup and for memory/error
behavior before changing a default or claiming an end-to-end improvement.

## Preserved launch and initial checkpoint

The [launch record](validation/validation-workers-m2-v1-launch.json) binds source
commit `f7e65b6639765606ddcd8121af51b0c8fd5b31dd` and frozen harness SHA-256
`8865ee8eb92933f8ef959210f9ec4d261d52cd28cce31d101d257098c02a4e36`.
The original [pilot report](validation/validation-workers-m2-pilot-v1.json),
[log](validation/validation-workers-m2-pilot-v1.log) and exact
[pilot source](validation/validation-workers-m2-pilot-v1-source.py) are retained.
Its trial-function AST matches the launched source.

The [initial checkpoint](validation/validation-workers-m2-v1-initial-checkpoint.json)
contains nine completed trials: three primers and six measured processes, followed
by an active tenth trial. A [read-only observation](validation/validation-workers-m2-v1-initial-observation.json)
checked the completed raw/log hashes, zero child exit codes and all one million
path checks per trial, and confirmed the next process was live. This checkpoint
is not the completed series or its final audit. Raw files remain under
`/Volumes/T7/ultrafast-vision-build/validation-workers-m2-v1.runs`.

After the benchmark terminated naturally, the independent summarizer passed
22 synthetic tests and the full real-artifact audit. The original launch and
partial checkpoint above remain historical evidence; the complete results
and all raw files are retained in the final results bundle.
