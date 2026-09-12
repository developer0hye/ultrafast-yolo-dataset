# Measurement index

- `detect-vmhwm.json`, `segment-vmhwm.json`: P1 label-only, 100k distinct TXT files, four workers, Linux VmHWM.
- `*-100k-server-w1.json`: historical P1 single-worker runs; separate source/environment snapshots.
- `startup-m2-detect-1k-miss.json`, `startup-server-detect-100k-miss.json`: actual YOLODataset initialization, legacy-cache write and first batch.
- `startup-m2-detect-1k-hit.json`: legacy-cache hit regression, preserved. Its process RSS includes untimed cache priming and must not be called load-only memory.

No file in this directory proves native content-cache or steady-state training performance. See [the report](../../docs/BENCHMARKS.md) for scope, parity, hardware, statistical summaries and missing release gates.
