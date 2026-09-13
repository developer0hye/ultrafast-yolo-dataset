# Input-validation worker tuning: full experiment pending

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

The full comparison and an independent readback of all results are still pending.
Any favorable worker setting must subsequently be checked in full startup and
for memory/error behavior before changing a library default or claiming an
end-to-end improvement. No speedup has been established by this new experiment.
