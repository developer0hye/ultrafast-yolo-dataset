# Startup confirmation auditor qualification

The draft auditor is developed separately on `bench/startup-worker-audit` while
the full startup measurement remains frozen on its predeclared sources. It does
not import the benchmark producer or installed library. The independent earlier
startup checker supplies container inspection, result checks and the exact
five-pair bootstrap; the new audit checks the full 12-process artifact graph,
qualified pilot runtime, command/order/exit status, source/wheel/cache identity,
reference outputs and whole-child telemetry. Caller-selected qualification and
launch files are trust anchors, not signed attestations.

The synthetic tests include a complete graph, 30 damaged graphs (including
updated raw-file digests), strict JSON rejection, a separately weighted
bootstrap oracle, and failed-receipt/exclusive-output behavior. They do not
represent measured speed or memory. Qualification is pending until the hosted
test job and its raw JUnit results are inspected.

Both local benchmark hosts are occupied. The existing wheel workflow accepts
an explicit `audit_only=true` manual dispatch on this branch, running only the
standalone standard-library auditor tests in a GitHub runner. This does not
build or qualify a wheel. The original full wheel matrices continue separately
at their frozen source commits; their result must not be attributed to this
new branch. The dispatch defaults to the full wheel matrix.

After qualification, run the full-size audit only once the measurement campaign
has terminated: cache inspection reads hundreds of MB. Include the qualified
pilot `w7-hit.json` as `--runtime-reference`, its six-trial qualification receipt,
the original output audit and exact launch/source/runtime/cache inputs. The
auditor rejects incomplete series and preserves failed output on reinvocation.

The telemetry spans imports and pre/post corpus checks as well as startup;
sampling gaps and endpoint delays are reported. Its peak is neither a timed
constructor-only peak nor an upper bound on unsampled RSS. A verified artifact
graph establishes the consistency of the retained evidence, not universal
replacement compatibility or an absence of interference on the shared host.
