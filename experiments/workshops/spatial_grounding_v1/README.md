# SGW-01 implementation and launch boundary

The immutable handoff is in [`spec/`](spec/README.md). Its 18 prompts, 174
six-cell blocks and 1,044 planned episodes were reproduced byte-for-byte.
Neither the imported specification nor historical V2/V3 protocols are edited
by this implementation.

**No SGW-01 learned-policy request, behavioral episode, or released family
fixture exists yet.** Recorded model-blind candidate qualification is in progress;
individual physical passes do not release a family. Local synthetic tests are
engineering checks, not study evidence. Runtime factories, server trace provenance, full resets, physical-time
maps and the live simulator remain subject to qualification before release.
The user's current direction is **RoboLab only and full benchmark qualification
first**: no benchmark substitution or exploratory policy pilot. Manual scene
designs are prospective proposals, not new gates or permission to rewrite
frozen failures.

Current restart state and cluster evidence:

- [`STATUS.md`](../../../artifacts/workshops/spatial_grounding_v1/STATUS.md)
- [`continuation_state.json`](../../../artifacts/workshops/spatial_grounding_v1/continuation_state.json)
- [Portable scene-design table and reference pictures](../../../docs/sgw_scene_design/scene-design.html)

## Restart and reuse, rather than rebuild

The continuation state is the restart authority, not a previous chat or a
successful process exit. Inspect existing work before starting another worker:

```bash
git status --short
python3 - <<'PY'
import json
from pathlib import Path
state = json.loads(Path("artifacts/workshops/spatial_grounding_v1/continuation_state.json").read_text())
for field in ("updated_at_utc", "status", "next_authorized_step", "next_safe_command"):
    print(f"{field}: {state[field]}")
PY
```

The persistent study root is `/data/users/ali/sgw-01`; the existing native
environment, external repositories and checkpoints are under
`/data/users/ali/vla_wam/`. Use the exact paths and hashes in the retained
runtime/job receipts. Do not reinstall, redownload, replace assets, or update a
live pinned source checkout merely to resume an ablation.

Reuse qualified scene/controller, renderer, calibration and runtime evidence
only when its exact bindings still match. A wording-only ablation need not
rebuild an unchanged scene; changed geometry, controllers, interfaces or model
settings require the corresponding new checks and a prospective registration.
Never reuse a rejected/partial candidate as qualified or rerun a valid result.
The current absence of a released SGW family/runtime remains a real blocker:
this restart procedure is not a release or an exploratory-inference shortcut.

The portable design handoff contains editable HTML, a CSV, three compact
reference figures, provenance and the paper-informed proposal. Open its HTML
locally; it does not require the temporary Side chat web server. Raw rollout
videos, arrays, checkpoints and simulator collections remain on the PVC, not
in ordinary Git.

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
HEIGHT/DIST native candidate qualification is running, but **no family fixture
is released**. Do not substitute repeated LAT layouts for those branches.

The prospective generator includes geometric rejections within its 100-candidate
cap, records their reasons, and never refills rejected slots. Its table and
banana-clearance screen is not robot reachability or physical qualification.
Candidate qualification order and eventual layout selection use the frozen hash
order. Robot reachability and every reset must still be verified in the native
runtime before a family is released.

`tools.audit_sgw_family_capacity` computes a read-only necessary capacity bound
from the frozen 100-proposal HEIGHT/DIST plans, independently rechecked smoke,
and explicitly selected retained worker prefixes. Pass each prefix manifest with
`--prefix` and a fresh destination with `--output`. It checks retained byte
bindings, canonical verification digests, score projections and unique design
membership. The seeded pilot side needs 15 layouts, not 14; the other side needs
14. Geometric and valid physical rejections consume slots without refill.
Every unknown slot and every physical pass remains optimistically potentially
qualified. A sufficient upper bound is **not** historical freshness, physical
release, or model release. This audit neither changes the live queue nor
replaces its pending independent final raw compilation.
Its current prefix reader accepts complete six-trial candidates only; it
rejects other shapes rather than padding a terminal geometry prefix or a
partial episode. The independent collector handles those shapes separately.

`tools.prove_sgw_pi05_root_separation` compares the exact accepted DIST000
capture against the 108 source-bound V3-B002 final-cell post-settle root pairs.
Run it with `--output <new-path>`. It replays the historical producer/reset audit,
binds the native root aliases, metre units and binary32 API against the
registered versions, and derives an outward-rounded per-root error certificate
for the historically anchored subtraction/normalized-rotation code. The native
source projection is independently reproducible from installed distribution
RECORD hashes without importing a simulator.
The resulting nonmatches are conditional on that recorded API contract;
current native bytes are not presented as historical import-byte attestations.
The 108 records contain only two numerical root pairs and exclude no
constructor, settle-window, preflight or infrastructure populations. Neither
this necessary distance test nor its agreement case releases a fixture.

`tools.audit_sgw_pi05_stochastic_resets --output <new-path>` reconstructs all
864 named reset attestations from the 432 final V3-D001 cells and verifies
their registered initial-state identities. It distinguishes the inherited
Phase-A source map and separately attested policy server from the unanchored
D001 simulator producer. The one repeated numerical root pair is not an
independent-layout count or a qualified cross-frame comparison. The exporter
rehashes original raw/capture streams on the existing CPU accessor; no
simulator, model request, historical rewrite or release is involved.

`tools.prove_sgw_pi05_phase_a_roots --output <new-path>` handles Phase A
separately. It validates eight registered Git sources and the actual runtime
patch independently anchored by the historical runtime and pair manifests.
The full raw/capture/partial streams reconcile for 54 final cells and their
54 preserved setup resets. Its conditional native-API DIST000 comparisons reuse
the outward-rounded numerical certificate; neither this producer identity nor
its repeated numerical root pair may be substituted for D001 evidence.

The retained paper-informed design proposal targets the initial cube's
geometric center approximately 0.50 m from a measured robot root in translation
only. Derive the actor root from the measured local center offset and fixed
orientation; do not use authoring X or a zero-origin guess. This is a design
prior, not an added scorer threshold, universal optimum, native realization,
new gate, or modification of frozen candidates.

The separately disclosed `SGW-ENG-008` attempt is implemented by
`paper_engineering --registration <bound-json> --output-root <new-root>`.
It first captures the preserved LAT057 layout without controller actions,
measures the actual robot root and camera poses, then authors and captures a
new rigidly translated arrangement. Both goal displacements, orientations,
450-action controller and six fresh-reset checks are retained. Its opt-in
LAT scene path is hash-bound to the fresh capture; the original stock task
path is unchanged. Raw frames/videos, native geometry and all outcomes remain
on PVC. This is one engineering layout, not a refill or release of a frozen
pool. Unavailable arm/camera measurements are explicit, never safe defaults.
`native_successor_job` binds a single non-preempting GPU Job to the four
current worker nodes with hard anti-affinity to the original `bt` workload:
it cannot take a fifth lane or interrupt a current episode.

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

### Bounded historical lineage

`historical_layout_streaming.extract_state_payload(..., include_lineage=True)`
adds exact producer/input bindings and source-defined frame records to the
same-stream-hashed extraction. The default selection remains unchanged.
Lineage extraction is bounded and excludes unrelated environment or invocation
fields; it does not confer historical coverage or release authority.

Adding `include_population=True` retains the source-defined
`materialization_environment/fresh_reset` paths, environment lifecycles and
bounded scalar attempt outcomes. This is a separate v3 extraction contract;
the v1/v2 selections remain unchanged. The earlier 100-snapshot export omitted
materialization-environment resets and is explicitly not an exhaustive
population. Lifecycle accounting must also address earlier infrastructure
attempts; extraction alone cannot release fixtures or learned inference.

The expanded bx audit adds a separately hash-bound R012 geometry-preflight
receipt and matches all 101 recorded lifecycles to reset snapshots. Its 149
named snapshots do not cover the earlier R005 attempt01, whose ledger records
missing rank 1--3 scientific state payloads. Reproduce this bounded accounting:

```bash
.venv/bin/python -m tools.audit_sgw_historical_population \
  --export-root artifacts/workshops/spatial_grounding_v1/infrastructure/historical-population-20260923bx \
  --output /tmp/sgw-historical-population-audit.json
```

Reproduce the retained seven-source audit into a fresh output file:

```bash
.venv/bin/python tools/audit_sgw_historical_lineage.py \
  --export-root artifacts/workshops/spatial_grounding_v1/infrastructure/historical-lineage-20260923bv \
  --prior-export-root artifacts/workshops/spatial_grounding_v1/infrastructure/historical-state-fields-20260923bl \
  --inputs handoff/k8s/sgw01-ali-historical-lineage-inputs-20260923bv.json \
  --output /tmp/sgw-historical-lineage-audit.json
```

The audit checks exact Git objects, preserved prior selections, and named
frame evidence. Direct external-file hashes have a separate receipt. Neither
artifact establishes all asset dependencies, measured root-local geometry or
an exhaustive historical population. A failed historical reset need not have
been behaviorally accepted or assigned an SGW identity to be comparable.

`historical_root_nonmatch` proves only a necessary-condition exclusion in an
explicit common coordinate frame. It uses the shared, fixed 3 mm threshold and
validates every required root before returning a nonmatch. Nearby roots,
missing actors, malformed positions and frame mismatches remain unresolved.
It cannot establish duplication or exhaustive population coverage and is not
wired into the fixture release path.

The bounded proof for HEIGHT 003--006 against 100 source-bound snapshots is
reproducible without a simulator:

```bash
.venv/bin/python -m tools.prove_sgw_observed_root_nonmatches \
  --output /tmp/sgw-observed-root-nonmatches.json
```

Native and historical frame bindings are prerequisites to that proof. A
strictly separated actor root rules out that specific within-tolerance layout
match even when centroid offsets are unavailable; it does not prove that the
historical snapshot population is complete.

`historical_root_bounds` compares each prospective root with the nearest point
in an independently justified historical position interval. It retains the
fixed 3 mm componentwise threshold and validates every required actor before
returning a nonmatch. Its reported distances are conservative lower bounds,
not measured historical displacements; overlap remains unresolved.

The R005 attempt01 proof binds the exact clean-source launcher, producer,
comparison helper, frozen reference and 20 completed-reset lifecycle markers.
The historical gate was **5 mm Euclidean**, not SGW's 3 mm deduplication
tolerance. Its outward-rounded intervals rule out each of HEIGHT 003--006
against those 20 settled resets, without recovering missing point observations:

```bash
.venv/bin/python -m tools.prove_sgw_r005_reset_bounds \
  --output /tmp/sgw-r005-reset-bounds.json
```

This proof is separate from the earlier 400 observed-point comparisons.
It overlaps the retained rank4 reset evidence and does not cover later
constructed states, initialization transients or missing rank1--3
materialization payloads. Neither primitive is wired into fixture release.
The recovered pre-AppLauncher controller verification also binds the whole
historical `basic_recorders.py`; it does not attest the historical imported
IsaacLab getter bytes.

All twenty R005 attempt01 native `env_cfg.json` files are now retained
losslessly as one complete config plus byte deltas, including the environments
whose later materialization poses were lost. Replay their hashes, lifecycle
identities, exact source bindings and unchanged scored-reference contracts:

```bash
.venv/bin/python -m tools.audit_sgw_r005_object_inventory \
  --output /tmp/sgw-r005-object-inventory.json
```

Only recorder output directories differ between these configs. Each registers
the cube, bowl, two bananas and table, with the bowl as the success reference.
This does **not** establish a physical no-plate exclusion: the native importer
filters registered objects by an allowlist while spawning the complete USD
scene. Complete historical transitive asset identity remains unproven.
Configured initial poses are not recovered measured states. Moreover, the
historical materialization finalizer returns even when its gates fail, so
normal completion cannot extend the settled-reset bounds to constructed states.

GR00T Phase A now has a complete selection of the **54 final-manifest cell
initial resets and their 54 named warmups**, with each full state stream
reconciled before selection:

```bash
.venv/bin/python -m tools.audit_sgw_groot_reset_population \
  --output /tmp/sgw-groot-reset-population.json
```

These 108 records contain one distinct numerical cube/bowl root pair in their
declared robot-base frame, not 108 independent layouts. They do not cover
infrastructure-invalid attempts, preflights or constructor transients. The
retained runtime reset-sidecar patch preserves the original getter and
normalized-rotation ASTs, but its source files lack an independently established
historical hash anchor in the final manifest. No world-frame relabeling or
cross-frame exclusion is claimed.

The separate pi0.5 V3-B002 recovery binds all **108 final-cell post-settle
initial states** to their original reset attestations and a historically
recorded 16-file adapter digest. Reproduce its exact Git-source, queue,
initial-state, attestation-byte and reset-fingerprint checks with:

```bash
.venv/bin/python -m tools.audit_sgw_pi05_reset_population \
  --output /tmp/sgw-pi05-reset-population.json
```

Use the recorded Git objects, not current files, for the adapter digest.
The two logical pre-action reset calls correspond to one physical reset;
the second returns cached observations. The recovered records contain two
distinct numerical cube/bowl root pairs, not 108 independent layouts.
Source provenance does not establish complete constructor/settle/preflight/
infrastructure coverage or independently qualify numerical cross-frame
comparisons. No exclusion or SGW release follows from this audit alone.

The separate `historical_root_separation` primitive supplies a necessary
distance-based exclusion for future qualified inputs in different orthonormal
metre frames. A match within the unchanged componentwise tolerance bounds the
possible change in actor-pair separation. Both frames and each root's numerical
error bound must be independently justified; the primitive accounts for both
roots on both sides and returns only `nonmatch` or `unresolved`. It neither
establishes provenance nor releases fixtures, and has not been applied as a
qualified exclusion to these recovered GR00T records.

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

## Native simulator mailbox

`native_mailbox_receiver` is prospective A40-side infrastructure for one
registered attempt, not a behavioral release. A coordinator must create the
immutable identity and release-cell JSON records, bind their SHA-256 values,
and invoke the receiver with a new RWX PVC directory and finite deadline. It
validates those records before creating `AppLauncher`, then starts the
hash-bound joint-position environment. The B200 policy worker uses
`MailboxClient` with the same identity. Requests and responses are
exclusive-create, fsynced JSON manifests plus lossless NumPy arrays.
`create_mailbox_environment` is the explicit `ProductionAdapter` factory:
it requires a coordinator-provided mailbox root, immutable identity path, and
identity SHA-256 environment binding; it does not replace the direct backend.
Nested camera/proprio mappings retain their shape across the mailbox.

A receiver writes `receiver_complete.json` only after the close response is
durably published. It binds the identity, exact close command, command count,
and response hash. Native receiver exceptions are fsynced to
`receiver_failure.json` before environment/AppLauncher cleanup, so exit zero
without the completion record is never evidence of a complete attempt.
Environment and application cleanup failures retain separate traceback receipts
and invalidate completion. A partially failed environment close is never retried.
An external coordinator-side `verify_receiver_completion` must validate the
completion receipt, close-response hash, exact learned-policy attempt scope,
and absence of a failure receipt before it treats a receiver process as
complete. These receipts are not interchangeable with zero-model capture
artifacts or a scored behavioral completion.

Before importing or constructing `AppLauncher`, the receiver loads the actual
`SGW01_ENV_BINDING`, verifies its hash and selected
`candidate_file_sha256` against the identity, and compares the identity Job and
Pod UIDs with mandatory `JOB_UID`/`POD_UID` Downward-API values. The
coordinator must inject those values; an identity record merely claiming them
is insufficient.

The mailbox never starts a model, provides a network listener, retries an
action, or attests the remote simulator as a local policy process. A timeout,
duplicate command, identity/hash mismatch, malformed array, receiver fault, or
partial response closes the client attempt. Production use still requires the
existing fixture, release, resource, native-renderer, and remote ownership
qualification gates.

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

## Historical layout geometry evidence

`historical_layout_evidence` is additive to the earlier source inventories and
comparator. It resolves exact historical Git blobs and can join the 24
forecast-layout pose manifests to their hash-bound accepted gate receipts.
The resulting 144 object records preserve configured roots, settled roots,
settled quaternions, AABBs and derived root-local geometric offsets separately.
The historical field called `object_centers_robot_base_m` actually came from
`world.get_pose`; it is not used as a geometric center. Geometry comes from
the pinned producer's `world.get_bbox` extrema in the same environment-local
axes. Reset displacement is never substituted for a geometric offset.

Both export bytes and their independently pinned pose-manifest anchors are
verified before a join. Raw exports stay outside Git; the compact registry
retains provenance and JSON pointers. Set `SGW_HISTORICAL_POSE_EXPORT` and
`SGW_HISTORICAL_GATE_EXPORT` to the retained exports for the optional real-data
test, or pass `--pose-export` / `--gate-export` to regenerate the registry.
This recovers one historical cohort, **not exhaustive historical coverage or
permission to release any SGW family**. Historical poses are not replacement
candidates.

## Native joint-position execution boundary

`robolab_jointpos_environment:create_environment` is a simulator-process
factory accepting `cell` and an attempt-specific `evidence_root`. It requires
an already-running AppLauncher; it is **not** a remote simulator launcher.
`SGW01_ENV_BINDING` and `SGW01_ENV_BINDING_SHA256` bind clean study/RoboLab
checkouts, actual asset bytes, and each released cell's layout, fixture,
prompt, candidate file and scene seed. HEIGHT/DIST additionally require their
native scene-file hashes. The factory registers the native absolute joint
controller, preserves goal-independent termination, and checks actual reset
roots against 3 mm / 2 degree tolerances before a policy observation.

The production wrapper reuses the qualification measurement implementation:
actor roots remain distinct from geometric centers, COM velocity is transported
to the scored center, and support comes from attributed cube/object contact
forces, never a constant. Production support includes all imported non-gripper
surfaces rather than only the two scripted landing targets. Camera/proprioception
observations exclude simulator object state. N3 receives its official
one-based exterior-camera slots and leaves composition/resizing to the pinned
service; D1 receives native batched tensors before its unmodified official
extraction/padding path.

The production adapter resets the policy session once per attempt, closes the
previous simulator, and retains the full physical reset/warmup receipt. Viewport
FPS is derived from measured action timestamps (15 Hz for this native runtime),
not the render-only warmup's 30-FPS display convention. Nonuniform or absent
physical timing fails closed.

Both model paths have complete 450-action adapter/environment/recorder/scorer
integration tests with simulated physics and model computation, including the
real owned HTTP producers and trace readers. D1 uses the hash-verified official
client source; N3 image composition uses its exact exported source and CPU
PyTorch interpolation. These are engineering tests, not physical qualification
or behavioral episodes. Native AppLauncher lifecycle, cross-pod simulator
transport, D1 distributed startup/decode/time mapping and live runtime gates
remain unreleased. Neither factory nor passing tests authorizes a model launch.
The source-backed checks additionally accept `SGW01_NANO_SOURCE_AUDIT` and
require CPU PyTorch in the local test environment; model execution still uses
the separately pinned native environments.

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

The owned HTTP path is now exercised through the actual hash-verified RoboLab
constructor, image extraction/padding, request packing, eight-action cache,
gripper postprocessing and reset, followed by the concrete trace reader and
prediction recorder. Only simulator tensors, model computation and filesystem
identity are substituted in this offline test; it is not native-runtime
qualification. Set `SGW01_D1_SOURCE_AUDIT`, `SGW01_D1_IMAGE_SOURCE_AUDIT` and
`SGW01_D1_AR_SOURCE_AUDIT` to the authorized source exports to run these tests.
The image utility SHA-256 is
`aead0c246b696ce5feabbbc63df93f42762365cc6315269fa08ad5c98e1a3d94`.

Physical reset, wrapper reset and native session IDs remain distinct. Server
action hashes cover the raw chunk, while executed actions retain the official
gripper conversion. Registered sampling seeds are recorded separately from
the effective native seed. Historical latent-only traces remain explicitly
`latent_only_retained`, never decoded predictions or scored zeros. Native
distributed-worker startup, official video decode/time mapping and live
runtime release still require qualification.

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

The read-only server export proves the generic websocket boundary in
`eval_utils/policy_server.py` (source SHA
`5c541300759ac211aa00639223e707c80a12bf548520a70981162b8a0c534117`):
`WebsocketPolicyServer._handler` receives `endpoint`, dispatches
`policy.reset` for reset, and `policy.infer` for inference. It also proves the
conditional policy surface in `groot/vla/model/n1_5/sim_policy.py` (SHA
`c7b692b84a03a70adc7e0d21fb7632a9866285645e8d43c916100e6f5fb7497a`), where
`lazy_joint_forward_causal` returns unnormalized actions and `video_pred`.
The first server-surface export did not include the tracked 14B launcher or
its concrete policy-construction class. The later AR export supplies the
frozen 14B route described below; no 5B route is accepted here.

The AR export now supplies the exact 14B wrapper at
`socket_test_optimized_AR.py` (SHA
`7ef17f66064bac8defafc1a84551089b124546729a98be8c0515b33d2e159d48`).
`OfficialDreamZero14BBackend` follows its `ARDroidRoboarenaPolicy.infer`
frame accumulation, `GrootSimPolicy.lazy_joint_forward_causal` call, action
conversion, session-change reset, and explicit `reset` state clearing. The
entrypoint constructs the reviewed `GrootSimPolicy`, distributed signal group,
and AR wrapper only after identity checks. The checkpoint's
`num_inference_timesteps=4` is not the trained action-head sampler setting:
the action-head source constructs `num_inference_steps=16`, `seed=1140`, and
`cfg_scale=5.0`. The checkpoint's `action_dim=32` is the padded latent width;
the AR unnormalizer and wrapper expose 8-dimensional joint-plus-gripper
actions. The binding therefore observes the constructed action-head fields
after construction, retains padded width as native metadata, and validates the
actual returned action shape rather than overlaying protocol values. The
official AR route decodes accumulated latent chunks only through
`trained_model.action_head.vae.decode` with its native tiling parameters. The
producer retains separate decoded-`uint8` RGB and CPU-latent artifacts with
independent hashes and encodings. The provenance identifies decoded frames as
an accumulated native stream that includes context/past, not a request-local
forecast. Missing futures are `not_exposed`; decode failures retain the full
accumulated latent stream and record `decode_error`. BF16 latents are widened
losslessly to float32 storage with their original dtype recorded. Decoded
futures currently carry `time_mapping_status=unmapped`, so they are unscorable
until the released interface proves physical target time and action-prefix
mapping. The adapter and episode recorder preserve `decoded_unmapped` and
`decode_error` rather than relabeling them as mapped predictions or absent
evidence.

For release wiring, set `SGW01_D1_HTTP_URL` to the loopback URL owned by
`dreamzero_wrapper_entrypoint.py`. The official RoboLab client class remains
the cache/reset/postprocessing authority; its subclass only replaces the
network query and reset boundary with the owned HTTP producer. Each request
passes the SGW request/reset/cell bindings through that boundary, and the
producer's durable trace remains the attribution authority.

## D1 rank lifecycle

The HTTP wrapper is rank zero and remains the process launched by the SGW
runtime, so its PID/socket attestation stays exact. For a real two-rank launch,
set `SGW01_D1_WORLD_SIZE=2` (the entrypoint defaults the worker command to its
own exact `--rank-worker` path; `SGW01_D1_RANK_WORKER_ARGV` may override it only
with a reviewed equivalent). The entrypoint validates the loopback host,
attested source/entrypoint/checkpoint bytes, and exact
`SGW01_D1_MODEL_PATH` before spawning any worker. Set
`SGW01_D1_COLLECTIVE_TIMEOUT` to bound native process-group setup (default 300
seconds), and provide `SGW01_D1_RANK_LOG_DIR`,
`SGW01_D1_RANK_READY_DIR`, and an optional loopback
`SGW01_D1_MASTER_PORT`. `OwnedD1RankLifecycle` launches only ranks 1..N-1 in
their own process groups, passes the pinned rank identities and rendezvous
variables, waits for per-rank JSON readiness, and records bounded logs. A
rank-zero construction and readiness budget comes from the same
`SGW01_READINESS_TIMEOUT` used by the parent launcher (default 15 seconds);
native launches must explicitly budget checkpoint verification/loading.
The startup guard requires the POSIX main thread rather than silently running
unbounded elsewhere. Source/revision identity comes from verified checkout
and checkpoint evidence, not caller-supplied claim strings. A
worker readiness record must contain its rank plus the exact source commit and
checkpoint revision. It must run the exported conditional `WebsocketPolicyServer._worker_loop`; rank zero alone constructs
the HTTP listener. Readiness is checked against the frozen sampler values
(steps 16, seed 1140, CFG 5.0, output width 8), not merely a self-hash. The
HTTP server watchdog checks worker health during idle service periods and
preserves a worker failure instead of presenting a healthy listener. Startup
timeout, worker failure, and shutdown timeout all clean up only the owned
process groups. Decoder and future time-map qualification remain separate
gates.
