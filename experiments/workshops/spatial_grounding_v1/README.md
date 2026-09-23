# SGW-01 implementation and launch boundary

The immutable handoff is in [`spec/`](spec/README.md). Its 18 prompts, 174
six-cell blocks and 1,044 planned episodes were reproduced byte-for-byte.
Neither the imported specification nor historical V2/V3 protocols are edited
by this implementation.

**No SGW-01 learned-policy request, behavioral episode, or physically qualified
fixture exists yet.** Local synthetic tests are engineering checks, not study
evidence. Runtime factories, server trace provenance, full resets, physical-time
maps and the live simulator remain subject to qualification before release.

Current restart state and cluster evidence:

- [`STATUS.md`](../../../artifacts/workshops/spatial_grounding_v1/STATUS.md)
- [`continuation_state.json`](../../../artifacts/workshops/spatial_grounding_v1/continuation_state.json)

## Local development

From the repository root:

```bash
uv sync --frozen --extra dev --extra sgw --python 3.12
.venv/bin/python -m pytest -q tests/test_sgw_*.py
.venv/bin/python tools/validate_vla_wam_v3_protocol.py --quiet
.venv/bin/python tools/validate_vla_wam_v2_protocol.py
```

The `sgw` extra supplies the real viewport encoder/decoder. Raw arrays are
retained losslessly; an encoded video does not replace timestamped source
frames. Production encoder identity must be recorded in the runtime binding.
The development extra also parses Kubernetes YAML for semantic regression
checks. Quote comma-separated environment values in flow mappings: attempt m
silently reduced the NVIDIA capabilities to `compute`, removing `nvidia-smi`.
Replacement r preserves the complete successful g Pod configuration, and its
parsed equality is checked before strict Kubernetes manifest validation.

## Implementation surfaces

`contract`, `release`, `recorder`, and `worker` implement release validation,
bounded persistent attempts, locking, completion pointers and finite partitions.
`adapters`, `runtime`, and `trace` implement the policy interface and provenance
checks. `scoring`, `compile`, and `prediction_annotations` keep physical outcomes,
technical missingness, censoring and unobserved predictions separate.

`nano_backend` calls the exact Cosmos `RobolabPolicyService`; `producer` and
`nano_wrapper_entrypoint` add an SGW-owned HTTP boundary, append-only request
trace, retained decoded futures and process attestation. The upstream RoboLab
server is WebSocket-based, not HTTP. Native camera composition is unchanged:
the service uses its 540 x 640 input defaults and separate `"480"` transform
resolution. The factory rejects tracked source changes and verifies every file
in the immutable checkpoint manifest before constructing the model. Published
`checkpoint.json` contains `{}`; it is not a revision manifest. Historical V2
checkpoint hashes are reused only for artifact identity, never behavioral data.
Local full-HTTP synthetic tests do not qualify real model execution or D1.
Any future launch must explicitly budget `SGW01_READINESS_TIMEOUT` for checkpoint
verification and model loading within the bounded Job deadline.

`runtime_preflight` is a zero-model renderer probe, not a fixture qualification.
`lat_workspace_capture` records the actual scene after that probe passes;
`lat_proposals`, `lat_candidate_generator`, `robolab_lat_qualification` and
`model_blind_qualification` are the model-blind LAT qualification path.
HEIGHT/DIST have contracts and selection checks, **not qualified physical
fixtures**. Do not substitute repeated LAT layouts for those branches.

The prospective generator includes geometric rejections within its 100-candidate
cap, records their reasons, and never refills rejected slots. Its table and
banana-clearance screen is not robot reachability or physical qualification.
Candidate qualification order and eventual layout selection use the frozen hash
order. Robot reachability and every reset must still be verified in the native
runtime before a family is released.

One selected proposal runs both goals three times in a fresh Isaac process:
`model_blind_qualification --proposal-file <file> --candidate-id <id>`, with the
required pinned runtime arguments. Every trial retains 450 issued commands,
451 observed states and raw RGB frames, plus an actually decoded viewport MP4.
Physically rejected trials retain the same evidence. Interrupted trials retain
issued-but-unobserved commands and any observed partial video; they are
infrastructure-invalid, not model failures. The shared physical scorer consumes
measured centers directly, without applying root-to-center offsets a second time.
Local synthetic integration is not a native fixture release.

`lat_workspace_capture --render-warmup-frames 120` is a separately bounded,
zero-action appearance diagnostic. It refreshes sensors after render-only updates,
rejects any physical-time advance, retains initial/intermediate/final RGB and a
viewport video, and records live USD asset-resolution paths. The diagnostic video
uses 30 FPS only for display; it does not represent advancing simulated time.
The default capture still performs no added warmup. Neither a nonblank image nor
successful texture decoding alone qualifies the rendered policy observation.
Actual A40 diagnostic z completed all 120 render-only updates at unchanged
simulation time and retained 121 decoded video frames. Its final three views
show the intended colored cube, red bowl, yellow banana and textured table;
the initial view still had dark surfaces. Every native LAT qualification reset
therefore performs the same recorded zero-physics warmup before publishing state
zero. This startup-readiness correction changes no scored action count. A future
production simulator binding must provide equivalent verified readiness before
any model observation; no production binding is released by this diagnostic.

First native candidate qualification ac completed six full recorded trials,
but all six failed pickup and violated the anchor-motion limit. Its 10,824 trial
file hashes and twelve complete trial/warmup videos were independently checked.
The pinned IK interface controls the Robotiq `base_link` flange, not a fingertip
or grasp center. The copied recipe lacks a measured flange-to-grasp transform;
do not scale that unqualified controller or infer intrinsic fixture infeasibility.
New capture/trace instrumentation records actual robot body-frame origins and
quaternions, explicitly not inferred contact centers. Measured tool geometry and
a prospectively bound controller correction remain prerequisites.

`lat_workspace_capture --render-warmup-frames 120 --gripper-calibration`
additionally measures actual finger geometry relative to named articulation
bodies, then holds the initial flange pose while closing the empty gripper for
30 actions and reopening for 30. This is **60 explicitly recorded calibration
actions**, not a zero-action render probe or a behavioral episode. It fits the
native five-second task, retains all 61 post-reset states/frames and a decoded
video, and rejects premature termination. Partial calibration evidence remains
on failure. Neither mesh bounds nor a source-code nominal fingertip height is
silently presented as an established grasp transform.

Native probe af completed this calibration. `grasp_calibration` derives the
prospective virtual TCP from the measured closed visual-pad midpoint, retaining
open/closed/reopened comparisons and binding the exact robot USD hash. This is
not a measured collision-contact center. SGW-ENG-003 preserves the original
rejected recipe and all candidate poses while adding corrected flange commands
and a release/retreat phase, still totaling exactly 450 actions.

Pass `--controller-calibration
artifacts/workshops/spatial_grounding_v1/controller_calibrations/lat-closed-pad-20260923.json`
to `model_blind_qualification` for this new recipe. The recorder saves the
controller identity before acting and includes it in the qualification receipt.
Omitting the flag retains the original, unqualified recipe for provenance;
neither recipe is a learned-policy action mapping.

## Execution order

1. Obtain a genuinely idle, authorized RTX/Vulkan-capable allocation. A
   Kubernetes GPU request or Ready pod is insufficient proof.
2. Run the immutable zero-model renderer preflight into a new persistent attempt
   directory. Preserve failures. Do not reuse or overwrite old receipts.
3. Capture actual assets/workspace, generate and qualify the family fixtures,
   and verify the real runtime, raw recorder, server trace and physical-time
   mappings. Keep simulator scoring state out of policy inputs.
4. Bind the direct fixed-input qualification and stage-specific receipts before
   releasing P, then D, then C. Advancement depends on technical correctness,
   never favorable task outcomes. Keep the supplied within-block order.
5. Run the finite released worker partition and regenerate analysis only from
   verified completion records. Do not promote synthetic records to a release.

The RTX PRO allocation was occupied, but a fresh A40 allocation passed the
idle guard. A bounded replacement renderer preflight passed with the
historically proven native-library order and actual three-camera scene
evidence. Zero-model workspace capture also succeeded, but it contains no
validated candidate slots. Prospective proposals and six-trial recording are
implemented, but native waypoint/reset validation remains outstanding.
Fixture/runtime qualification and concrete worker resource receipts
still block behavioral release. The existing B200 workload is not owned by
this task and must not be stopped. No policy server was started. The source manuscript remains a plan,
and Overleaf synchronization also requires an authenticated connection.

## SGW-01 execution receipts

The worker is deliberately fail-closed before production adapter construction.
`runtime_binding.json` must contain hash-bound paths for
`resource_budget_receipt`, `operational_authorization_receipt`, and
`external_allocation_receipt`; D/C also require
`storage_budget_receipt` and a nested hash-bound `measured_runtime_receipt`.

Resource-budget, storage, measured-runtime, and allocation receipts bind
`release_id`, `source_queue_sha256`, `source_queue_episode_count` (exactly
the immutable 1,044-source-episode study), `pvc_name`, `pvc_mount_path`, `study_root`, and
`runtime_identity_sha256`. The identity hash covers the immutable worker image,
source and simulator commits, model-code commits, checkpoint hashes, and GPU
type, excluding receipt references to avoid a self-reference hash cycle.

Storage receipts additionally contain a positive `pilot_p95_episode_bytes` and
`global_remaining_episode_count`. The worker requires at least 1,044 remaining
episodes, deliberately overestimating a P6 release unless a future frozen
whole-study completion ledger changes that bound. Required PVC free space is
`max(100 GiB, configured floor, ceil(1.5 * P95 * global remaining))`.

The resource-budget receipt is an approval/preflight estimate only; it is not a
worker-managed consumable ledger. It records a finite estimated remaining GPU
duration; numeric-capped mode additionally records approved GPU-hour coverage.
Uncapped existing-idle mode intentionally has no invented aggregate numeric
approval. For D/C, measured runtime evidence must match model, deadlines,
three-attempt allowance, and cover the estimate.

The [operational authorization](../../../artifacts/workshops/spatial_grounding_v1/operational_authorization.json)
is parent-owned (`schema_version:
sgw-01-operational-authorization-v1`) and must declare either
`numeric_global_gpu_hour_cap` or
`existing_idle_capacity_no_aggregate_hour_cap`. The latter is the explicit
user authorization to use only verified idle capacity already present in the
bound context and namespace; it never provisions cloud capacity and remains
limited to the frozen two workers/four GPUs, finite queue, retries, and Job
deadline. Its `scope` binds models, PVC/study root, context/namespace, protocol
and queue hashes, 1,044 registered episodes, and three attempts. Its
`constraints` forbid paid capacity, provisioning, and interference with
unowned workloads, while requiring a fresh idle allocation check and bounded
Job deadline. An uncapped mode does not make node availability an approval or
bypass the allocation proof.

The external allocation receipt is the coordinator-owned live enforcement
boundary. It requires `schema`, approved status, non-secret
`owner_approval_reference`, `reservation_id`, bound Kubernetes context and
namespace, `job_name`, `job_uid`, `pod_uid`, model, exact allocated GPU count,
`startTime`, `activeDeadlineSeconds`, `deadline_utc`, budget mode, and
`reservation_gpu_hours`. Numeric-capped mode additionally requires approved
and globally reserved GPU hours. Existing-idle uncapped mode instead requires
a separate hash-bound `gpu_idle_probe_receipt` from `gpu_idle_probe.py` whose
status is exactly `passed_idle_snapshot_only`, complete visible/allocation GPU
UUID inventory, selected UUID, and unoccupied snapshot prove the exact
allocation after Job start and before model construction; a free-form `passed`
flag is rejected. The
worker compares its
Job-controller label `JOB_UID` and Downward-API `POD_UID`, verifies
`reservation_gpu_hours = allocated_gpu_count * activeDeadlineSeconds / 3600`,
requires `deadline_utc == startTime + activeDeadlineSeconds`, rejects a future
start, requires global reservations not to exceed the approval in capped mode,
and caps operations to the remaining deadline. It checks the idle proof only
before model loading, then preserves the hash-bound same-allocation proof while
the server owns that GPU. Kubernetes `activeDeadlineSeconds` remains the
external final bound for hung descendants.

Workers never create, refund, reuse, reconcile, or authenticate these
coordinator receipts. A changed pod requires a newly issued allocation
receipt. Without a genuine coordinator-issued reservation, all learned-policy
launches remain blocked with exit code 44.

Both budget modes require the real hash-bound idle probe for the complete
allocated UUID inventory. Every device, not merely the selected simulator GPU,
must satisfy the same memory/utilization/process guard. The snapshot must be
measured after the Job starts, before model construction, and no more than
300 seconds before the startup check. It is not reinterpreted as current idle
capacity after the worker loads its own model. The allocation deadline must
equal Job start time plus `activeDeadlineSeconds`. Numeric-capped reservations
must include their own worst-case allocation and match the owner's actual cap.
All unfinished attempts recheck space, approvals, and remaining Job time;
completed partitions return without constructing an adapter.
## D1 runtime identity and cadence

The official DreamZero client is loaded only after three external identity
inputs are supplied: `SGW01_D1_SERVER_SOURCE_ROOT` must be the clean DreamZero
server checkout at commit `ab790c198fbce33503358efbbd4187ce9a89adf3`,
`SGW01_D1_CLIENT_SOURCE_ROOT` must be the clean RoboLab checkout at
`0aef241fb088ca21bb4ebd24448940ed56620d17`, and
`SGW01_D1_CHECKPOINT_PATH` must match the immutable checkpoint file manifest
and revision `96ad344138c66e82536422432ad742f015784942`.

The authorized source export is
`a6666f0aba72463cc4381c2bc0af14dfc956acb77ca728350211a421698fdfe6`;
the client and base-client file hashes are
`96de16927536f2b48427a6a2dcc3111d03204e1832e50e159cadf67b3fe956ac` and
`6d357550f55763d6c23dc7d9efb85af9e7821f5b0d0edf76213e6b2d9d7b3f29`.
These are qualification gates, not download instructions.

RoboLab's base client returns one processed action per `infer` call. The D1
wrapper configures `open_loop_horizon=8`; it calls the native client eight
times per SGW request, retains raw and postprocessed 24x8 chunks, and exposes
the postprocessed chunk for execution. The first call refreshes the native
cache and the next seven consume it. The wrapper also verifies that the eight
official infer returns equal the processed chunk prefix, then retains those
eight executed actions separately. Reset must evict the server session and
clear local chunk state. Live server trace, reset eviction, decoded-future
evidence, and checkpoint attestation remain required before D1 qualification.

## Independent qualification verification

`qualification_batch_verifier` checks the frozen batch/proposal/calibration
hashes, exact candidate and controller identities, every trial and warmup file,
raw-to-scored state correspondence, action bindings, physical timestamps,
reset comparisons, and full video decoding. It recomputes the unchanged scorer.
A strict stability failure is a verified physical rejection, not a technical
error. Missing/partial evidence never becomes a physical failure.

Warmup verification follows the frozen recorder's actual sampling: all 121
left-shoulder frames and three camera views at frames 0, 1, 10, 30, 60 and
120. That is 133 arrays plus one warmup video per reset, not 363 arrays.
CPU attempt am incorrectly required three views at every frame; its reports
are preserved as verifier failures and do not invalidate or rerun raw trials.

Run it in a CPU-only process against the persistent raw tree:

```bash
python -m experiments.workshops.spatial_grounding_v1.qualification_batch_verifier \
  --plan artifacts/workshops/spatial_grounding_v1/qualification_batches/lat-remaining-20260923.json \
  --expected-plan-sha256 f5007f84d26f27946b38f2053ca7d640b56e50e92ae3d09a0261feb45f88275e \
  --proposals artifacts/workshops/spatial_grounding_v1/proposals/lat-20260922.json \
  --controller-calibration artifacts/workshops/spatial_grounding_v1/controller_calibrations/lat-closed-pad-20260923.json \
  --raw-root /data/users/ali/sgw-01/qualification/lat-batch-20260923ak \
  --output <new-verification-report.json>
```

The report never releases a family. Source/launch attestation, historical
deduplication and model-runtime release remain separate gates. Reverification
reads existing evidence only; it never reruns a physical trial.
For a durable CPU Job, `--wait-until-utc <timezone-qualified-deadline>` verifies
new completed candidates once and fsyncs a compact per-candidate report beside
the final output. The bounded wait neither retries a simulator nor treats
unpublished candidates as physical failures. An incomplete final batch exits
nonzero while preserving every report.

## Owned D1 server binding

The owned D1 HTTP boundary is implemented by `dreamzero_producer.py` and
`dreamzero_wrapper_entrypoint.py`. Its native binding is deliberately not
guessed: `dreamzero_backend.py` requires a reviewed
`SGW01_D1_SERVER_FACTORY` after verifying the separate pinned DreamZero server
and RoboLab client checkouts plus the complete checkpoint manifest. The
synthetic backend tests exercise request binding, native session eviction,
action hashing, durable future retention, and the HTTP error boundary. The
exact server factory source still required for native qualification is the
DreamZero websocket/model constructor and its reset/session-eviction method
from the clean server checkout at
`ab790c198fbce33503358efbbd4187ce9a89adf3`; no server method names are
invented here.
