# Windows cache compatibility follow-up

The first hosted matrix at source `e1c5861` exposed real failures in the
[Windows Python 3.11 job](https://github.com/developer0hye/ultrafast-yolo-dataset/actions/runs/34744593599/job/103690048444):
the integration/cache stage ended with **11 failures, 79 passes, 2 skips**.
The original run is retained; it is not retried or described as a platform pass.
The failed job log and API receipt are preserved alongside their source SHA.

## Changes under qualification

- Legacy cache publication reopened the completed sibling temporary as `rb`
  before `os.fsync`. Windows file-buffer flushing requires write access; the
  helper now uses `r+b`, preserving all bytes before atomic replacement. See
  [Microsoft's FlushFileBuffers contract](https://learn.microsoft.com/ko-kr/windows/win32/api/fileapi/nf-fileapi-flushfilebuffers).
  A sync-failure test confirms that the previous cache survives and temporary
  files are removed. Successful legacy export/reload remains covered by the
  original integration tests.
- A captured directory can produce `PermissionError` from Python `open` on
  Windows. The snapshot verifier now accepts that diagnostic for an already
  captured non-file only after revalidating the fingerprint. It retains the
  original error message. Regression tests check cache reuse with the diagnostic
  and rejection when the directory is replaced before that revalidation.
- Unicode whitespace fixtures used `Path.write_text` without an encoding,
  although the reference label reader explicitly uses UTF-8. The shared fixture
  and snapshot-mutation test now write UTF-8 explicitly. Both valid UTF-8 NBSP
  and invalid single-byte NBSP cases are compared with the pinned reference and
  round-tripped through the native cache. Runtime decoding policy is unchanged.

The full wheel matrix runs separately on this new source revision. Qualification
requires the actual installed-wheel tests and retained artifacts, not just these
code changes. The new tests are in the existing cache/integration files already
included in the matrix; no failed platform or test is removed. The active M2
startup and RTX 3070 lifecycle campaigns continue using their original frozen
runtime, so their results will not be attributed to this candidate.

The unusually large initial hosted artifacts (about 178 MB per dataset job)
remain a separate packaging investigation. Their contents and source-only
rebuild behavior must be inspected before changing the sdist manifest.
