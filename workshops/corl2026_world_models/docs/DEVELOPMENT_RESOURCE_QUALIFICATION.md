# Development machine-resource qualification

This runtime closes only the GM machine-measurement gaps left explicit by
`DEVELOPMENT_RESOURCE_AUDIT.md`. It does not rerun a development cell, execute
a behavioral action, create a label, or release confirmation.

## Conservative qualification target

The qualified topology is deliberately serial:

| Model | Measured topology | Frozen concurrency |
| --- | --- | ---: |
| N3 | one two-B200 worker; exact fixed-input model runtime on logical GPU 0 and a settled RoboLab simulator on logical GPU 1 | one block |
| D1 | one two-B200 distributed model worker plus one distinct one-B200 simulator worker, synchronized through signed PVC receipts | one block |
| Across models | N3 and D1 are not overlapped | one global block |

This is enough to execute all retained confirmation blocks sequentially. It is
not evidence that two blocks can safely coexist. Higher concurrency requires a
new measurement and freeze; worker count is not scientific sample size.

Each model reruns its already qualified six-request P00 fixed-input protocol
solely for resource measurement. Thus a complete probe wave adds exactly 12
nonbehavioral generation requests. The simulators each perform one physical
reset and the already defined settling/stability holds, then remain resident
without a policy request or behavioral action. Settling holds are reported
separately and never counted as behavioral actions or robot episodes.

## Measurements and failure behavior

The sampler records every assigned B200 at 0.5-second intervals. Every row is
hash chained and includes device UUID, total/used memory, utilization, every
driver-reported compute process, and live process ancestry. A pass requires:

- the exact B200 count and pinned model/simulator topology;
- no pre-existing or unowned compute process;
- bounded sampling gaps and at least two samples in inference and overlap
  intervals;
- nonzero whole-process and whole-device peaks for every assigned runtime GPU;
- distinct N3 model/simulator GPUs and three distinct concurrent D1 GPUs;
- exact six-request qualification success for both models;
- exact reset/settle and zero-behavioral-action simulator receipts; and
- the observed whole-device peak multiplied by the frozen 1.10 headroom to fit
  within every sampled device's reported capacity.

`nvidia-smi memory.used` is the whole-device measurement. The sum of
`used_gpu_memory` over task-owned compute processes is reported separately.
N3 and D1 PyTorch allocator peaks are also retained, but allocator values are
never substituted for driver-visible memory.

Any missing sample, unknown process, peer failure, source/input mismatch,
undersampled interval, or insufficient headroom makes the attempt
technical-invalid. Raw logs and samples remain in that immutable job directory;
the job does not publish a partially passing machine freeze.

Process-start counts do not come from pre-created stdout/stderr files. Each
spawn has an immutable signed, fsynced attempt receipt immediately before the
spawn call and an immutable signed, fsynced outcome immediately after it
returns or raises. An authenticated successful outcome counts one process, an
authenticated failed outcome counts zero, and an attempt with no valid outcome
is reported as `null`. The richer launch receipt binds the successful outcome
to PID, process group, host, Linux process-start identity when observable,
command hash, and working directory.

The simulator child independently fsyncs a hash-chained lifecycle journal. It
writes before/after records around the physical reset and every settling or
stability hold. Consequently, failure after a returned reset reports exactly
one reset even if readiness was never published. A crash while a reset or hold
call is in flight reports `null` plus the tight completed/started lower and
upper bounds; it never guesses completion. Both the inner
`technical_failure.json` and the outer queue failure receipt reproduce this
accounting. These reset-settling holds remain nonbehavioral.

The paired D1 jobs use immutable signed `simulator_ready`, `model_started`, and
`model_done` records. A technical failure before either child launches also
writes the appropriate `simulator_failed` or technical-invalid
`model_started`/`model_done` marker. Thus each peer fails closed promptly rather
than waiting for the six-hour safety timeout; a failure marker can never satisfy
a passed-receipt or finalizer gate. During the six-request D1 probe, the model
wrapper polls the signed simulator-failure marker every 0.25 seconds. The probe
is a directly owned child in the queue wrapper process group; peer failure
terminates and reaps it, then the stopped server artifacts preserve exact
issued/completed request counts (or `null` if those artifacts do not validate).

## First descriptor wave

Wait for the immutable formal audit receipt from
`development-resource-compiler-formal-001` to appear on the results branch.
Use the fetched copy only to build descriptors; the detached runtime reopens
the exact original receipt on the PVC.

```bash
ROOT=workshops/corl2026_world_models
RESULTS_RECEIPT=results/jobs/development-resource-compiler-formal-001/publish/development_resource_job_receipt.json
AUDIT_SHA256=$(sha256sum "$RESULTS_RECEIPT" | awk '{print $1}')

python3 "$ROOT/experiments/forecast_layout/development_resource_qualification_jobs.py" \
  emit-probes \
  --study-commit "$(git rev-parse HEAD)" \
  --resource-audit-receipt "$RESULTS_RECEIPT" \
  --resource-audit-receipt-sha256 "$AUDIT_SHA256" \
  --output /tmp/development-resource-qualification-probes.json
```

The output must contain exactly these three descriptors:

1. `development-resource-qualification-n3-001` on role `n3`;
2. `development-resource-qualification-d1-model-001` on role `d1`; and
3. `development-resource-qualification-d1-simulator-001` on role
   `wmf-forecast-0912-worker-00`.

Append those exact descriptors once to the active queue only after the normal
operator-update check. Do not alter their order or start a second coordinator.
The D1 model job safely waits for its simulator peer, and every child is
bounded/reaped even when a peer fails.

## Receipt-gated compiler wave

After all three jobs publish passed receipts and the results branch is fetched,
build the final CPU-safe descriptor:

```bash
N3=results/jobs/development-resource-qualification-n3-001/publish/development_resource_qualification_job_receipt.json
D1_MODEL=results/jobs/development-resource-qualification-d1-model-001/publish/development_resource_qualification_job_receipt.json
D1_SIM=results/jobs/development-resource-qualification-d1-simulator-001/publish/development_resource_qualification_job_receipt.json

python3 "$ROOT/experiments/forecast_layout/development_resource_qualification_jobs.py" \
  emit-finalize \
  --study-commit "$(git rev-parse HEAD)" \
  --n3-receipt "$N3" \
  --n3-receipt-sha256 "$(sha256sum "$N3" | awk '{print $1}')" \
  --d1-model-receipt "$D1_MODEL" \
  --d1-model-receipt-sha256 "$(sha256sum "$D1_MODEL" | awk '{print $1}')" \
  --d1-simulator-receipt "$D1_SIM" \
  --d1-simulator-receipt-sha256 "$(sha256sum "$D1_SIM" | awk '{print $1}')" \
  --output /tmp/development-resource-qualification-finalize.json
```

The one descriptor is
`development-resource-qualification-finalize-001` on CPU-safe role
`wmf-forecast-0912-worker-09`. Its compact output is
`development_resource_machine_freeze.json`; raw sample journals, model output,
simulator evidence, and exact hashes stay on the GM PVC.

## Remaining human-derived field

A machine pass emits `machine_resource_gate_complete: true`, but always emits
`safe_to_release_confirmation: false`. Rater A time, rater B time, and any
adjudication time remain `null` until authenticated completed development
responses exist. Session duration, media-render time, queue wall time, and VLM
latency are not substitutes. The later development freeze must bind those real
human-derived values before confirmation can be released.
