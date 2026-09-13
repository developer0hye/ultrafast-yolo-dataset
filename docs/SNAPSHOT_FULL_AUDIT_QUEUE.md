# Full startup audit completed after the actual macOS measurement

The [v2 controller](validation/qualify-snapshot-after-startup-v2.py) completed
after both measured processes terminated. Its
[execution receipt](validation/snapshot-startup-full-m2-v2-qualification.json)
records successful formatting, lint, 35 rejection cases, three controls and the
full-cache audit. All five commands returned zero. The
[launch identity](validation/snapshot-startup-full-m2-v2-qualification-launch-identity.json)
preserves the observed process, boot time, source hashes and measurement handles.
See the [actual full result](SNAPSHOT_FULL_M2_RESULTS.md); it does not establish a
speed or memory improvement for this candidate.

The controller checks both measurement PIDs together with their observed start
times and commands. It waits while either exact process remains live. A missing
or reused PID permits checking the terminal launch receipt; it does not trigger
a restart. The receipt must report successful completion, exit code zero, all
20 measured records plus two primers, and matching full-reference output.

Only then does it format staged copies of the two audit scripts, require unchanged
ASTs, and run Ruff checks. The rejection qualifier must pass three controls and
reject 35 damaged artifacts. Its expanded five-round controls are explicitly
synthetic and provide no speed evidence. After qualification, the independent
auditor checks the actual 500,000-pair, five-round miss/hit reports, original raw
records, retained cache sections, pinned source/wheel identities and reference
output hashes. The final audit and all command logs remain separate files.

The stage is `/Volumes/T7/ultrafast-vision-build/snapshot-startup-full-m2-v2-qualification-stage`.
The state and output prefix is `snapshot-startup-full-m2-v2-qualification` under
the same build root. The staged files remain preserved. The formatted scripts
were copied back only after checking the local originals against the recorded
pre-format hashes, the staged post-format hashes, and equal ASTs. The full
129-member evidence archive subsequently passed creation readback and local
reconstruction/readback; its bundle and receipt are linked from the full result.

The first watcher was intentionally stopped while idle, with no commands started.
The [cancellation receipt](validation/snapshot-startup-full-m2-v1-qualification-cancelled.json)
preserves that fact. Source review found that its failure handler could overwrite
an existing receipt on an accidental rerun. V2 reserves its output exclusively
outside that handler. The measured processes were not signaled, and this was not
a benchmark or audit test failure.
