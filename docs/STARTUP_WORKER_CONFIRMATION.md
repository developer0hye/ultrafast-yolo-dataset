# Full startup confirmation: seven versus sixteen scan workers

The campaign has launched. Its [initial observation](validation/startup-worker-tuning-full-m2-v1-launch.json)
confirmed the first primer process was live, all frozen worker files matched
the tested pilot and whole-child telemetry was being written. Results and the
final independent audit remain pending.

The [phase experiment](VALIDATION_WORKERS_RESULTS.md) found a 14.7% reduction
in time medians from 7 to 16 workers. Seven is the setting used by the preceding
full-startup experiment, so this confirmation retains it as the baseline.
The separate 4-worker phase result is not substituted for that baseline.

The [controller](validation/run-startup-worker-tuning-full-m2-v1.py) uses the exact
worker and six imported source files from the successful 1,003-pair functional
pilots. It requires both the full phase audit and all six pilot results to pass.
The runtime is the same qualified native cache-reader wheel in both conditions.

## Predeclared protocol

- Use the complete 500,000-pair Detection fixture, one million JPEG/TXT files.
  Its known complete fingerprint and original reference output hashes are bound.
- Retain the original validated content cache. Copy it into a new private
  directory for each worker setting and verify identical bytes. Every trial must
  be a cache hit and leave those bytes unchanged.
- Run two separate unmeasured primers, first 7 then 16 workers.
- Measure five pairs in alternating order: 7/16, 16/7, 7/16, 16/7, 7/16.
  All twelve trials use fresh child processes and the same exact worker sources.
- Measure the actual constructor and constructor-through-first-batch times.
  Keep full input fingerprints before/after and all ordered label, first-batch
  and diagnostic hashes. The first-batch DataLoader uses zero workers in both
  conditions, as in the preceding startup benchmark.
- Preserve phase timers and the original process high-water memory observations.
  A one-second observer also records process/system CPU, RSS, load, available
  memory, swap and disk counters throughout each child. These samples include
  pre/post input checks; they do not isolate the timed constructor window.
- Any timeout, failed child, changed source/cache, output mismatch or observation
  failure stops the series. No failed run is retried or dropped. The parent
  cannot overwrite an earlier campaign file on accidental reinvocation.

This is a full-size **cache-hit startup** confirmation. Cache-rebuild performance,
Segmentation, other hosts/platforms and any library-default change remain separate
gates. An independent final artifact/statistics audit is required after all
twelve children and the controller exit; parent completion alone is insufficient.

The preceding full-size hit worker took about 572 seconds wall time, including
all untimed input checks, while its timed startup was about 52 seconds. This
series can therefore take around two hours; the 1,800-second per-child watchdog
is a failure bound, not a throughput target.

Authoritative output:
`/Volumes/T7/ultrafast-vision-build/startup-worker-tuning-full-m2-v1.json`.
Raw reports/logs and whole-child telemetry are in its sibling `.runs` directory.
No full-size startup improvement is established until the final audit passes.
