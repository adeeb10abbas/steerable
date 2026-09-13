# Development resource audit

This audit measures only quantities recoverable from the immutable, completed
development evidence. It does not run a model or simulator, issue a request,
execute an action, create an annotation, or release confirmation.

## Evidence and definitions

The formal job authenticates both passed development timing-sidecar receipts,
all eight development aggregate receipts, all 32 cell receipts, all 1,152
official request receipts, each recorder journal and completion, the four D1
server receipts, the N3/D1 topology receipts, and all 14 terminal queue-wrapper
snapshots. It then scans exactly the 12 retained behavioral attempt roots on the
PVC and hashes a stable, symlink-free, de-duplicated regular-file inventory.

The emitted measurements use these fixed definitions:

- episode wall time: last minus first recorder-journal monotonic timestamp;
- request round-trip time: model-response-received minus model-request-sent
  monotonic timestamp, summed over requests;
- N3 service time: server receipt completion minus start monotonic timestamp;
- D1 inference/decode time and allocator peaks: the producer's retained rank-0
  wrapper, per-rank forward proxy, offline-decode, and CUDA allocator records;
- raw storage: unique resolved regular-file paths in cell and request trees,
  with shared block overhead reported separately and storage-object
  de-duplication reported as a second scope.

These are not interchangeable clocks. In particular, summed D1 rank time is a
proxy rather than elapsed wall time, and PyTorch allocator peaks are not total
GPU-process or `nvidia-smi` peaks.

## Honest missingness and release boundary

The GM cluster is the selected execution site, and the execution assignment
already authorizes the bounded core schedule: at most two recording-pilot,
eight development, and 48 confirmation blocks (58 blocks and 232 behavioral
cells maximum; 96 confirmation cells per qualified model). This authorization
is present and is not a pending user-input dependency.

The completed runs did not retain N3 allocator peaks, simulator GPU peaks,
simultaneous all-process GPU peaks, completed two-rater annotation time,
adjudication time, or a measured safe execution-topology/concurrency envelope.
Every unavailable measured value is emitted with an explicit missing status and
`null`, never zero.

`ABLATION_SPEC.md` section 7 freezes budgets before confirmation and section 8
requires measured runtime, memory, storage, and annotation costs plus a selected
execution host and bounded run authorization. The latter two are satisfied;
the missing measured memory/annotation and safe-concurrency freeze are not. The
specification provides no declared-missingness exception for those resource
prerequisites. Consequently, this v1 audit uses new, non-legacy schemas and always emits
`safe_to_release_confirmation: false`. A successful audit is evidence recovery,
not completion of the confirmation resource gate.

## Descriptor-only release

After the implementation is committed and the results branch is fetched, an
operator can review a deterministic descriptor without editing the active
queue:

```bash
python3 workshops/corl2026_world_models/experiments/forecast_layout/development_resource_jobs.py \
  emit-formal \
  --study-commit "$(git rev-parse HEAD)" \
  --repository "$(pwd)" \
  --results-commit 260338bf3226bedc6103b44209aba73ba579de6e
```

The descriptor targets `wmf-forecast-0912-worker-09`. Its only publishable
success artifact is a compact signed queue-job receipt. The 32 signed cell
measurements, signed aggregate, signed file inventory, and signed compiler
receipt remain under the job's unique raw directory on the PVC. A terminal
failure is mutually exclusive with success, preserves zero science counts, and
does not reuse the immutable job directory.
