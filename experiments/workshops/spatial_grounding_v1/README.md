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

`runtime_preflight` is a zero-model renderer probe, not a fixture qualification.
`lat_workspace_capture` records the actual scene after that probe passes;
`lat_candidate_generator`, `robolab_lat_qualification` and
`model_blind_qualification` are the model-blind LAT qualification path.
HEIGHT/DIST have contracts and selection checks, **not qualified physical
fixtures**. Do not substitute repeated LAT layouts for those branches.

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
validated candidate slots; proposal and waypoint validation remain missing.
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
allocation before the Job starts; a free-form `passed` flag is rejected. The
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
