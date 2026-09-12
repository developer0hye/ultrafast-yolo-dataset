# Independent startup-result audit

`bench/audit_startup.py` audits frozen P1, P3 and P4 artifacts without importing
NumPy, Torch, Pillow or either native extension. It is being developed in a
separate worktree while the M2 500k-pair run remains bound to its original source
and installed wheel. No measured runtime or benchmark harness was changed.

The four real 1,003-pair pilot reports (Detection/Segmentation × P3/P4) pass this
auditor. Their exact output records are in
`validation/startup-auditor-pilot-*-v2.json`. These are artifact checks on existing
measurements, not benchmark reruns. Ruff and actionlint passed. The 24 synthetic
tests in `tests/test_startup_audit.py` are written but **not executed yet** because
both benchmark hosts have active workloads. The first real 500k-pair P1 reference
worker also passed a partial artifact audit, including fixture/source/native
identity and the declared count/throughput relationship. Its immutable report is
`bench/results/detect-500k-p1-checkpoint-1-v1.json`, with the audit under
`validation/detect-500k-p1-checkpoint-1-audit-v1.json`. There is no completed pair
or aggregate in that snapshot. Five-pair final-audit paths are not yet execution-
validated, and a full-scale audit is not complete.

The subsequent two-worker snapshot completes the first reference/native P1 pair
on all 500,000 labels. Its full packed-output hashes match and the partial audit
passes, still without an aggregate. See
`bench/results/detect-500k-p1-checkpoint-2-v1.json` and
`validation/detect-500k-p1-checkpoint-2-audit-v1.json`. Four more paired repetitions
remain. P1 RSS is the process high-water mark sampled immediately after the timed
parse/export, before output-digest and post-measurement input verification; it
is not the final process-exit RSS peak or isolated working allocation.

## Checked evidence

- The caller explicitly supplies the task, expected pair count, repetition count
  and worker count. The preflight's distinct-file counts and full fingerprint
  match the fixture manifest; P1's nested label manifest must match exactly.
- Source inventories and actual native-extension hashes match the supplied
  launch/identity receipt. These are trusted comparison anchors, not signed
  attestations. The current source tree is not used to reinterpret historical
  samples, and the corpus is not re-read during artifact auditing.
- Required samples follow the declared alternating backend order without
  missing, duplicate or additional trials. Partial reports require explicit
  opt-in and never produce a timing/memory summary. Cache reports also need the
  parent's completed aggregate before receiving a final audit.
- P3/P4 parent entries match independent child JSON exactly. P4 additionally
  requires separate primer artifacts for each backend. Observed native stage
  names must show encode for misses or decode for hits and mutable label export;
  the content-checked reference must show fingerprint and decode stages.
- Full label/first-batch/diagnostic hashes match across backends and repetitions.
  The synthetic fixture requires the complete found/total count, zero diagnostics
  and zero native fallback. This auditor is intentionally scoped to these
  well-formed synthetic fixtures; error-bearing real-data semantics have their
  separate tests and reports.
- Times and memory figures must be finite and valid. RSS high-water marks must
  be monotonic. P1 throughput must correspond to the full declared label count.

## Statistics and limits

The auditor recomputes medians and p95 values and compares them with the parent
cache reports. It independently calculates the paired ratio of medians and a
percentile bootstrap interval. At five pairs it enumerates all `5**5 = 3125`
ordered resamples, preserving reference/candidate pairing. For more than five
pairs it uses 10,000 explicitly seeded standard-library draws.

The parent's NumPy-random bootstrap endpoints are **not** claimed to match this
different calculation; use the independently computed interval with its stated
method. A one-pair pilot produces a degenerate interval and cannot establish
repeatability or a speedup. Do not combine process-level samples as if they were
independent machines or datasets.

P1's original harness embeds worker JSON in the parent report and does not retain
separate worker files. The auditor states this evidence limitation rather than
inventing independent raw provenance. All new audit outputs bind the exact input
bytes and raw-file hashes for later readback. RSS includes interpreter imports,
preflight and allocator history; it is not working allocation or family RSS.
Recorded runtime package fields still need comparison across the complete
experiment and with the environment/wheel provenance before a release claim.

## Final full-scale invocation

After the running Detection series finishes, execute each phase with the launch
receipt and the verified generated fixture. The current M2 cache harness uses
seven scan workers; P1 was explicitly launched with four. Example for P3:

```sh
python bench/audit_startup.py \
  --report /measurement/startup-detect-500k-p3-v1.json \
  --identity docs/validation/detect-500k-launch-v1.json \
  --preflight /measurement/startup-detect-500k-preflight-v1.json \
  --fixture-manifest /measurement/startup-detect-500k-v1/startup.json \
  --phase p3 --task detect --pairs 500000 --repeats 5 --workers 7 \
  --out /measurement/startup-detect-500k-p3-audit-v1.json
```

Audit P1 with `--phase p1 --workers 4` and its P1 report; P4 with `--phase p4
--workers 7` and its P4 report. P3 and P4 must also have identical final output
hashes for the same fixture. Require all three completed phases before calling
the Detection workload complete, and repeat the full procedure independently
for Segmentation. `--runs-dir` supports relocating P3/P4 raw files. Existing audit
outputs are never overwritten. `--allow-partial` only permits an interim audit.
