# SGW-01 Kubernetes execution and recovery contract

This is an implementation handoff. **No cluster was contacted and no worker was deployed when this package was written.** `kubernetes/worker-job.yaml.in` is a template, not a runnable release. The receiving agents implement the worker below, verify the environment, render a concrete manifest, and then operate the finite released queue. A detached laptop process alone does not provide pod-replacement recovery.

## 1. Ownership and binding

Read the repository's `AGENTS.md` and `docs/WORK_LAPTOP_B200_HANDOFF.md`, then current V3 and V2 continuation documents/states/protocols. Historical handoffs describe completed cohorts and old resources; they are not a list of currently free GPUs. SGW-01 is a new study namespace and must not mutate their protocols or reuse their released queues.

Use the user's existing authorized work-Mac → Kubernetes route. Bind an explicit context, namespace, user-owned resource name, image digest, PVC name/mount, GPU count, and existing resource budget into `runtime_binding.json`. Query only the identified namespace/resources. Do not use all-namespaces discovery, inspect other users' workloads, delete an existing pod, stop an unrelated policy server, or reuse someone else's service. If ownership/budget cannot be established, preserve the spec and report that exact missing input.

The spec permits at most **two concurrent model workers** and at most **four allocated GPUs total**, subject to the user's existing quota/budget; start with one worker until reset, storage locking, and restart recovery pass. These are ceilings, not a request to reserve four GPUs. D1 may need multiple GPUs; measure the pinned runtime instead of altering model settings to fit. No new paid allocation is authorized by this file alone.

A full checkout is required. The authoring checkout is sparse; implementation files may exist as Git objects but be absent locally. Read them with `git show <pinned_commit>:<path>` or make an isolated full checkout on the execution host. Do not edit an unrelated dirty checkout.

Required runtime-binding fields (none may be inferred from this template):

```text
context, namespace, resource_owner, worker_image_digest,
pvc_name, pvc_mount_path, pvc_access_mode, lock_test_receipt,
model_gpu_counts, cpu_memory_limits, node_gpu_type,
cluster_version, source_commit, model_code_commits,
simulator_commit, renderer_receipt, persistent_write_receipt,
checkpoint_hashes, user_resource_budget, budget_source,
policy_ports, cache_reset_receipt, frame_time_mapping_hashes
```

Store references to existing Secrets; never copy credentials into the spec, logs, ConfigMaps, Git, or Overleaf. The worker needs no broad cluster permission. Runtime fields must come from current inspection or the user's existing allocation, not an old chat or this package.

## 2. Persistent directory contract

Use `<verified PVC mount>/sgw-01/<release-id>/`, with a new release ID for each scientific/runtime revision:

```text
release/                 immutable protocol, prompts, queue, hashes, fixtures, runtime binding
attempts/<cell-id>/<attempt-id>/
  intent.json            persisted before a model request
  request_index.jsonl    request IDs, timings, hashes, returned/executed horizons
  actions/ states/ observations/ predictions/ videos/
  events.jsonl           append-only, includes failures and termination
  result.json            metrics plus status; written atomically at completion
  manifest.json          every required artifact's size and SHA-256
cells/<cell-id>.complete.json    atomic pointer to the one validated completion
blocks/<block-id>.json    six condition completion pointers and integrity state
status/<worker-id>.json   atomic progress/heartbeat; no source of scientific truth
locks/<model-id>.lock     exclusive whole-worker filesystem lock
analysis/                regenerated summaries; never replaces raw recordings
handoff/STATUS.md         exact remaining cells, blockers, and restart command
```

Raw artifacts, checkpoints, package caches and environments live on the PVC. Git contains compact manifests, source, analysis and selected figures only. The release stores a full hash of the queue, protocol, prompts, fixture bundle and runtime. A resume refuses mismatches; it does not rewrite the expected hash.

## 3. Worker implementation to deliver

Create a new package `experiments/workshops/spatial_grounding_v1/`, separate from V2/V3:

| File | Responsibility |
| --- | --- |
| `contract.py` | Read immutable release, exact prompt/goal mapping, IDs and expected hashes; reject unqualified rows. |
| `fixtures.py` | Family fixtures, deterministic candidate generation, scripted reachability receipts and reset validation. |
| `adapters.py` | Narrow wrappers around pinned Nano and official DreamZero; full per-episode reset; no prompt alteration. |
| `recorder.py` | Append-only attempts; timestamps, raw predictions, actual images, actions/states; atomic completion manifest. |
| `scoring.py` | Physical relation, pickup/release/stability, final success, censoring and anchor disturbance. Simulator geometry is scoring-only. |
| `worker.py` | Persistent queue consumer, exclusive lock, bounded retries, signal handling, status and remaining-work report. |
| `compile.py` | Rebuild summaries from validated raw completion manifests, preserving denominators and missingness. |
| `release.py` | Validate fixtures/runtime/recording receipts and emit an immutable released queue and concrete Job manifests. |

Reuse interfaces and logging conventions from pinned `experiments/v3/cosmos_nano_phase_b/{live_client,live_support,robolab_bridge,runtime_adapter,serve_nano}.py`, `experiments/v3/dreamzero_phase_b/{client,contract,robolab_bridge,recover_future_trace}.py`, and `experiments/v3/phase_c_four_phrasings/`. Inspect their behavior before reuse. Do not copy goal-dependent termination or custom DreamZero s2 guidance into the new primary experiment.

The **new CLI to implement and test**, not an existing command, is:

```text
python -m experiments.workshops.spatial_grounding_v1.worker
  --release /data/sgw-01/<release-id>/release
  --model N3|D1 --family LAT|HEIGHT|DIST --stage P|D|C
  --resume --max-valid-episodes <stage branch size>
  --max-cell-attempts 3 --heartbeat-seconds 60
```

Per-model/family branch sizes: P=6, D=24, C=144. A Job consumes one model/family/stage queue partition and exits when its finite partition is complete. Run only one active partition for a given model. A coordinator advances P→D→C after receipts pass and selects the next qualified family. The global model lock also protects temporal-state isolation against accidental duplicate partition launches.

## 4. Completion, retries and exactly-one evidence record

Kubernetes can restart a workload and can sometimes start duplicate pods. Completion must be idempotent in the recorder; a Job's success alone is not scientific completion. See the [official Job documentation](https://kubernetes.io/docs/concepts/workloads/controllers/job/).

1. Acquire a kernel-managed exclusive lock whose cross-pod semantics were tested on this PVC. Never rely only on a heartbeat timeout to steal a lock. If the PVC cannot provide reliable exclusive locking, allow one verified worker only and require a coordinator to verify the old pod/process is gone before a replacement; do not deploy concurrent writers.
2. Check `cells/<cell-id>.complete.json`, verify its manifest and release identity, and skip it if valid, whether its outcome was success or failure.
3. Persist a unique attempt intent, reset the model and simulator completely, verify the six-cell block's initial-state tolerances, then issue the exact prompt.
4. On valid success or valid model failure, persist all data, fsync, validate, atomically publish the completion pointer with no-overwrite semantics, then advance. A duplicate writer loses publication and preserves its attempt without becoming another observation.
5. On process interruption, recorder loss, hardware failure or infrastructure error, preserve the partial attempt as technical invalidity. Resume the same cell from a fresh full reset, never from half an episode. Completed cells in the block remain untouched.
6. Maximum **three attempts total per cell**, counted durably across Job restarts. Before retrying, require a known technical cause; a bad grasp, wrong object, wrong direction, no motion, collision caused by the policy, or task timeout is not infrastructure failure. Do not rerun it.
7. After three technical-invalid attempts, mark that branch blocked, emit the exact cause, and stop its Job with a fatal exit. Other qualified independent branches may proceed. Never convert a blocked cell to success/failure or silently shrink the denominator.

Technical-invalidity decisions must be blind to whether a task would have succeeded. If a correction changes checkpoint, actions, fixtures, camera timing or scientific scoring, start a new release and disclose the cohort boundary. Byte corruption after a completed trial is data loss; do not silently replace the observation with a convenient rerun.

## 5. Continuous operation, interruption and resource limits

- A Kubernetes Job supervises the worker. It must start the model/simulator as owned child processes and propagate failure, not leave a forever-sleeping pod after the child dies. Use an init/reaper appropriate to the qualified image.
- Start-up verification covers driver/GPU memory, actual renderer scene creation, PVC capacity and write/fsync, and checkpoint availability. An import test alone is insufficient. Use existing Secrets for downloads if needed.
- Heartbeat every 60 seconds with release/model/partition/cell/attempt/request ID, progress timestamps, bytes written, GPU identity, completed/remaining/invalid counts, and last error. A heartbeat does not count as progress.
- Pilot sets a request deadline to max(300 seconds, 5×largest successful pilot request duration) and an episode deadline to max(900 seconds, 3×largest pilot episode duration). Freeze these before D/C. Start-up load limit is separately max(1,800 seconds, 3×measured load time). Retain deadline-triggered partial data. Repeated deadline failures exhaust the same three-attempt limit.
- Space check before each block: free persistent bytes must exceed the larger of 100 GiB and 1.5×pilot 95th-percentile episode size×all remaining episodes on that PVC. Include latent storage and all model partitions. A lack of storage stops intake before collecting an unrecordable cell.
- After pilot, estimate time/GPU-hours from measured durations and planned remaining counts, including model loading and retry allowance. Compare with the existing user allocation. Record the approved resource budget and refuse unbounded expansion. Do not invent an ETA or a completed resource reservation in the handoff.
- On SIGTERM: stop taking new cells; flush the current attempt; complete only if it actually finishes inside the 180-second grace period; otherwise mark interrupted. Close video writers and stop only this worker's children. Never fabricate a final state. A replacement resumes using completion pointers.
- After each complete block, atomically update status and regenerate compact counts. Operational monitoring may see progress and technical errors during confirmation; do not tune protocol or thresholds from live outcome charts.
- Publish a human-readable `STATUS.md` after every partition and on any blocker. It contains exact source/release hashes, Job/pod ownership, completed versus expected cells, unresolved technical attempts, PVC path, and a concrete verified restart command.
- Completion: validate 1,044 episode manifests if all branches qualify, compile results, check raw-media coverage, stop/release this study's resources, preserve PVC data. Do not keep consuming GPUs after the finite queue ends.

No Codex recurring automation is created by this package. Durable execution is the cluster worker's responsibility; agents may resume it in a later task from the recorded status.

## 6. Job template and rollout

`kubernetes/worker-job.yaml.in` defines a single partition (the Overleaf copy is supplied as `worker-job.yaml.in` at the project root). Its deliberate substitution fields must be resolved from `runtime_binding.json`; the release tool rejects any unresolved field, mutable image tag, missing module, missing receipt, or wrong partition size. Copy the verified image's existing resource/security/device settings rather than granting new privileges to make an untested renderer run.

The template uses `restartPolicy: Never`, bounded Job retry (`backoffLimit: 2`), and 180-second termination grace. Worker exit codes: 0=partition complete; 42=scientific/runtime release invalid; 43=cell attempts exhausted; 44=storage/budget blocked. If supported by the live cluster, use `podFailurePolicy` to fail immediately on 42–44. Otherwise the coordinator suspends that Job on those codes; durable cell attempt limits still prevent extra model requests. Do not install a new controller or upgrade the cluster for this study.

Do not attach an aggressive liveness probe to a long model request. The worker's measured deadlines handle request hangs. Kubernetes restart settings do not justify rerunning valid model failures.

## 7. Acceptance checks before unattended confirmation

Agents must demonstrate and retain receipts for these cases:

1. Parser and analytic coordinates correctly map all 18 prompts to physical goals; DIST inversion preserves its inequality.
2. Both goals are feasible in every fixture and reset/render identities are reproducible.
3. Identical model inputs expose actual nondeterminism; full reset prevents one episode's cached state entering another.
4. A success at action 120 does not terminate the action-450 episode; safety truncation is censored, not padded.
5. Recorder rejects a prediction aligned beyond the executed prefix or from a different request/camera/reset.
6. Killing a worker after two completed cells preserves both, retries only the incomplete cell, and completes the six-cell block once.
7. Two duplicate workers cannot issue concurrent model requests or publish two accepted observations for one cell.
8. A wrong-side or failed-grasp trial is marked complete and skipped on resume.
9. Missing/corrupt artifacts prevent publication; fake hashes or altered release identities fail closed.
10. Attempt limits survive pod replacement; a fourth attempt is refused. Disk-full and provenance drift produce useful status without a launch loop.
11. Both continuous metrics and all-trial binary denominators survive technical missingness and safety censoring correctly.
12. Regenerating analysis from raw manifests reproduces every table; no forecast label is inferred from simulator ground truth for an unobservable predicted image.

Use synthetic records and a lightweight fake adapter for recovery tests. Their successful checks are engineering evidence, not model results. After these checks and P/D validation, the receiving agents can continue the released queue autonomously within its fixed scope.
