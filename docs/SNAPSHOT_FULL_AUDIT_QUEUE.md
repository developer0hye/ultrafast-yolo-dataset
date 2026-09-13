# Full startup audit queued behind the actual macOS measurement

The [v2 controller](validation/qualify-snapshot-after-startup-v2.py) is running as
an idle dependent process. Its [launch identity](validation/snapshot-startup-full-m2-v2-qualification-launch-identity.json)
records the observed process, boot time, source hashes and two measurement
handles. No formatting, rejection tests or full-cache audit has run yet.

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
the same build root. Do not modify the staged files while it waits. When the
controller terminates, inspect actual outcomes and preserve any failures. Copy
formatted scripts back only after confirming the local originals still match
the recorded pre-format hashes. Evidence sealing and archive readback remain
required after a successful audit.

The first watcher was intentionally stopped while idle, with no commands started.
The [cancellation receipt](validation/snapshot-startup-full-m2-v1-qualification-cancelled.json)
preserves that fact. Source review found that its failure handler could overwrite
an existing receipt on an accidental rerun. V2 reserves its output exclusively
outside that handler. The measured processes were not signaled, and this was not
a benchmark or audit test failure.
