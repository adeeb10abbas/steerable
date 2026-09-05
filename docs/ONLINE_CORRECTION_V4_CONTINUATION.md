# Online correction V4 continuation

**Status:** `QUALIFYING` / `IMPLEMENTING` — prospective design freeze; no V4 policy inference has run.

This document is the machine-readable continuation companion to `docs/online_correction_v4/README.md`. Do not infer study state from chat history; use the committed artifacts under `artifacts/online_correction_v4/`.

## Authority

| Source | Role |
| --- | --- |
| `docs/online_correction_v4/` numbered docs | Scientific design, metrics, runbook |
| `docs/online_correction_v4/campaign.json` | Allocation and defaults |
| `docs/online_correction_v4/design_validation.json` | Design-only planning validation (unchanged semantics) |
| `artifacts/online_correction_v4/freeze_manifest.json` | Index of all freeze artifact hashes |
| `artifacts/online_correction_v4/continuation_state.json` | Active status, hashes, and next commands |
| `artifacts/online_correction_v4/` | Prospective freeze artifacts (queue, manifests, gate report) |

Historical V2/V3 protocol files remain immutable.

## Hash semantics

| Hash field | Meaning |
| --- | --- |
| `planning_manifest_sha256` | Pre-enrichment inventory from `tools/online_correction_v4.py build_manifest` (matches `design_validation.json` when campaign unchanged) |
| `frozen_queue_sha256` | Enriched `queue.jsonl` bytes including prompt text, hashes, and `queue_row_kind=new_episode` |
| `generation_parent_commit` | Git HEAD when the freeze builder last ran — **not** the commit containing freeze artifacts |

After merge, record a `git_receipt.freeze_commit` binding the commit that contains `artifacts/online_correction_v4/`.

## Prompt identity (C2 counterbalance)

Semantic prompt identity is **`prompt_id`**, not `prompt_sha256`. Under C2 reference binding, identical UTF-8 prompt text can name different physical A/B bowl identities when counterbalance swaps which color is “A”. The same words therefore may share one `prompt_sha256` while mapping to multiple `prompt_id` values.

- `prompt_sha256 = sha256(utf8(prompt_text))` — identical hashes iff byte-identical resolved text; different text must not share a hash.
- Episode-level binding uses `episode_id`, `prompt_id`, `prompt_text`, and `prompt_sha256`.
- Analysis must **never** aggregate, join, or key contrasts on `prompt_sha256` alone.
- `prompt_manifest.json` and `frozen_analysis_manifest.json` document this rule explicitly.

## Current status

| Field | Value |
| --- | --- |
| Lifecycle | `QUALIFYING` |
| Implementation | `IMPLEMENTING` |
| Release | `NOT_RELEASED` |
| Policy episodes executed | `0` |
| Confirmatory queue rows | `17664` |

### Infrastructure qualification on 2026-09-05

- The π0.5 A100-policy/A40-simulator lane passed startup qualification under
  driver `580.95.05`: checkpoint and source hashes, CUDA, Isaac RTX rendering,
  ffmpeg encode/decode, policy `/healthz`, pinned imports, and persistent
  storage checks all passed.
- The π0.5 B200 policy role passed the same policy-side checks. Its paired B200
  simulator did not schedule because all five B200 nodes reported insufficient
  GPU capacity, so no B200 lane qualification is claimed.
- Two earlier attempts are retained as infrastructure-invalid packaging
  failures. They exposed and led to fixing the missing runtime-script mount;
  no policy request or behavioral episode ran in any qualification attempt.
- Compact hashes and raw persistent-storage receipt URIs are recorded in
  `artifacts/online_correction_v4/qualification/20260905_pi05_lane_qualification.json`.
  All temporary Jobs, Services, and ConfigMaps were removed after evidence
  capture.

This clears only one infrastructure stratum. Runtime, geometry, trigger,
scorer, visibility/contact-sensor, prefix-replay, pilot, and miniature-campaign
gates remain unreleased.

### Current model-blind G2 preparation

`artifacts/online_correction_v4/setup/horizontal_reset_registry.candidate.json`
now prospectively binds all 128 registered horizontal reset seeds. It starts
from the committed zero-request V3 base-fixture calibration and applies an
independent seed-derived common x/y scene translation bounded to ±15 mm,
preserving all movable-object relative geometry. The candidate contains zero
model requests and zero behavioral episodes.

`tools/run_v4_horizontal_g2_seed.py` and
`experiments/online_correction_v4/model_blind_g2.py` implement the per-seed
two-reset/one-physical-reset, settle/stability, native-dt, neutral-layout,
policy-camera, and numeric task-frame checks. Each live seed also binds exact
camera intrinsics/extrinsics, projects left/front/up from the measured task
frame, and writes lossless source camera PNGs plus an annotated axis montage.
`tools/render_v4_horizontal_g2_k8s_jobs.py` renders 128 immutable
simulator-only A40 Jobs and no policy Service; its validator checks complete
seed coverage, commit/dt bindings, one-GPU isolation, and zero policy
endpoints. A complete G2 claim still requires all 128 live receipts and an
explicit review of an actual rendered axis montage.

Attempt `g2q20260905a` launched all 128 registered seed Jobs and was stopped
as infrastructure-invalid before any seed passed. The preserved attempt
contains 86 write-once infrastructure-failure markers and 42 Pods that never
created an episode output. Every one of the 66 Pods that Kubernetes marked
`Succeeded` had actually written the same failure:
`create_env() got an unexpected keyword argument 'rendering_type'`. The pinned
RoboLab factory requires `rendering_mode`, `renderer=realtime`; Isaac teardown
then masked the runner's failure return code with process exit 0. The adapter
now uses the pinned render contract, and each seed runs below
`tools/run_v4_g2_checked.py`, a non-Isaac parent that rejects a failure marker
or a missing/nonpassing seed receipt. Compact disposition and raw snapshot
hashes are in
`artifacts/online_correction_v4/qualification/20260905_horizontal_g2_attempt_g2q20260905a.json`.
No G2 seed, policy episode, or behavioral failure is claimed from this attempt.

The fresh one-seed smoke attempt `g2q20260905b` verified the parent checker:
the Pod exited 1 and Kubernetes marked the Job failed. Environment creation
passed the repaired renderer call, then reset failed because the live adapter
attempted to pass a CUDA world-pose tensor directly to NumPy. All pose and
velocity reads now explicitly detach and transfer to CPU first. The Job/Pod
objects, log, raw marker, and hashes are preserved under the compact receipt
`artifacts/online_correction_v4/qualification/20260905_horizontal_g2_smoke_g2q20260905b.json`.
This also remains infrastructure-only evidence with no passing G2 seed.

Smoke attempt `g2q20260905c` progressed into the model-blind settling loop and
exposed one more adapter boundary: the settle proxy supplied an already-batched
1x8 hold tensor to a converter that expected a Python tuple. The converter now
accepts either form while preserving the exact shape/device/dtype check. The
exit-1 Job and raw evidence are recorded in
`artifacts/online_correction_v4/qualification/20260905_horizontal_g2_smoke_g2q20260905c.json`;
it likewise contains no passing G2 seed or behavioral activity.

Smoke attempt `g2q20260905d` then reached native-period attestation and measured
the pinned horizontal RoboLab task at `0.06666666666666667` s (1/15 s), not the
provisional `0.05` s in the render spec. The exact-equality gate rejected the
attempt as intended. This model-blind post-result calibration is disclosed in
`artifacts/online_correction_v4/qualification/20260905_horizontal_g2_smoke_g2q20260905d.json`.
The live G2 CLI now requires an explicit native period, and the next fresh
attempt binds the measured value; no tolerance was relaxed and no G2 pass is
claimed from the calibration attempt.

The complete registered attempt `g2q20260905e` then ran all 128 model-blind
seeds on 12 A40 nodes with zero model requests. Exactly 64 seeds wrote passing
reset/camera/numeric-frame receipts and 64 failed the setup gate. Of the
failures, 62 exceeded the provisional 3 mm settled-position tolerance only at
the cube (observed rounded range 3.0–3.9 mm), one banana and one cube failed
the released stability threshold. Therefore G2 failed and remains incomplete:
do not record the visual-axis review as completing G2, do not execute G3, and
do not release the reset registry or policy inference. All raw seed outputs,
Job/Pod objects, logs, and an outcome index were retained before cluster
cleanup; compact hashes and disposition are in
`artifacts/online_correction_v4/qualification/20260905_horizontal_g2_wave_g2q20260905e.json`.
The 64 passing seed receipts remain valid model-blind setup evidence, while
none of the 64 failed seeds are behavioral failures.

The disclosed post-result amendment
`artifacts/online_correction_v4/setup/horizontal_g2_recalibration_amendment.candidate.json`
freezes a 5 mm registry-position tolerance for a fresh full attempt. This is
the observed 3.9 mm model-blind maximum rounded up to 4 mm plus a frozen 1 mm
guard band. The registry positions, jitter, native period, camera/frame
requirements, and all stability thresholds remain unchanged. Attempt
`g2q20260905e` remains failed; the amendment authorizes only a new 128-seed
zero-model-request G2 qualification at the pushed implementation commit
`54101be67cc1d51ec1144ac3a50a166c570079d4`.

That fresh attempt, `g2q20260905f`, completed all 128 seeds with 126 passing
receipts and no position-tolerance failures. Seeds `2100000052` and
`2100000101` reproduced the same banana and cube stability failures,
respectively, seen in attempt `e`. This confirms seed-specific unstable setup
geometry. G2 therefore still failed. Do not weaken the unchanged stability
thresholds and do not rerun attempt `f`; a new disclosed model-blind
seed-substitution or setup-geometry amendment is required. Full raw evidence
and compact hashes are recorded in
`artifacts/online_correction_v4/qualification/20260905_horizontal_g2_wave_g2q20260905f.json`.

After inspecting the montage,
`tools/record_v4_horizontal_g2_axis_review.py` records the four explicit
orientation assertions. `tools/compile_v4_horizontal_g2_aggregate.py` then
verifies every seed receipt, referenced artifact, runtime stratum, registry
hash, and axis review before producing the aggregate receipt. Even a passing
aggregate authorizes G2 only: the candidate reset registry remains
unreleased for policy use until G3 and the later gates pass.

`artifacts/online_correction_v4/setup/horizontal_g3_plan.candidate.json`
formula-closes the next model-blind gate without executing it. It binds all
128 reset seeds, the five descending scale candidates, 3,072 live path checks
per scale (15,360 maximum), and the 112 horizontal scripted checks required
for a final geometry candidate. The nine scripted reset cases are selected
from xy-jitter extrema while covering each counterbalance state twice. The
plan also freezes task-frame motion directions so left/right share one
physical axis and front/behind share the other; requested goal polarity does
not determine the independently balanced physical motion sign. Live G3
collision, support, reachable-workspace, and scripted grasp/place evidence is
still pending and cannot be replaced by this plan.

### Family disposition

| Family | Disposition | Status |
| --- | --- | --- |
| C1, C3–C7 | `pending_qualification` | `NOT_RELEASED` — runtime/geometry receipts pending |
| C2 | `hard_blocked` | `BLOCKED_SETUP` — verified common-prefix replay required |
| C8 | `hard_blocked` | `BLOCKED_RUNTIME` — GR00T Bridge/WidowX stack unverified |

Every queue row is a **registered new episode**. `reuse_episode_ids` are comparison-control links only; C3/C4 rows are not reuse-only aliases. C4 fast-schedule sham/move rows are new episodes because schedule differs from reused C1 controls.

## Exact next commands

No fresh cluster launch is authorized from the current artifacts. First freeze
a new disclosed model-blind amendment that replaces the two reproducibly
unstable reset seeds or corrects their setup geometry. G3, reset-registry
release, policy inference, and behavioral episodes remain prohibited.

```bash
python3 tools/online_correction_v4.py validate
python3 tools/build_online_correction_v4_freeze.py --out artifacts/online_correction_v4
python3 tools/validate_online_correction_v4.py
python3 -m unittest discover -s tests -p 'test_online_correction_v4*.py'
python3 tools/validate_vla_wam_v2_protocol.py
python3 tools/validate_vla_wam_v3_protocol.py
```

The live G2 seed entrypoint is supported only inside a fresh qualified
simulator Job at a clean pushed checkout:

```bash
python3 tools/run_v4_g2_checked.py \
  --expected-environment-seed "$ENV_SEED" -- \
python3 tools/run_v4_horizontal_g2_seed.py \
  --study-root "$STUDY_ROOT" \
  --robolab-root "$ROBOLAB_ROOT" \
  --reset-registry "$RESET_REGISTRY" \
  --reset-registry-sha256 "$RESET_REGISTRY_SHA256" \
  --environment-seed "$ENV_SEED" \
  --output-dir "$WRITE_ONCE_OUTPUT" \
  --expected-study-commit "$EXPECTED_STUDY_COMMIT" \
  --expected-driver-version "$EXPECTED_DRIVER_VERSION" \
  --native-control-dt-s "$NATIVE_CONTROL_DT_S" \
  --gpu-uuid "$GPU_UUID" \
  --pod "$POD_NAME" \
  --pod-uid "$POD_UID"
```

Each invocation covers one registered seed and explicitly remains incomplete
for G2 until aggregate seed coverage and rendered-axis review pass.

After all Jobs finish, inspect one hash-bound `axis_overlay_montage.png`, then
record and compile:

```bash
python3 tools/record_v4_horizontal_g2_axis_review.py \
  --seed-receipt "$G2_SEED_RECEIPT" \
  --axis-overlay "$G2_AXIS_MONTAGE" \
  --reviewer-identity "$REVIEWER_IDENTITY" \
  --left-axis-matches-fixed-robot-viewpoint \
  --front-axis-points-toward-robot \
  --up-axis-opposes-gravity \
  --labels-and-arrow-origins-visible \
  --out "$G2_AXIS_REVIEW"
python3 tools/compile_v4_horizontal_g2_aggregate.py \
  --reset-registry artifacts/online_correction_v4/setup/horizontal_reset_registry.candidate.json \
  --reset-registry-sha256 0c5fd649739cd19b74ec1874f306cf345a70fb110047294204295f9f53ced328 \
  --receipts-root "$G2_RECEIPTS_ROOT" \
  --axis-review "$G2_AXIS_REVIEW" \
  --out "$G2_AGGREGATE_RECEIPT"
```

## Freeze artifact index

| Artifact | Purpose |
| --- | --- |
| `protocol.json` | Frozen estimands; planning vs enriched queue hashes |
| `prompt_manifest.json` | Bare-noun resolved prompts (no duplicate articles) |
| `queue.jsonl` / `queue_manifest.json` | 17,664 new episodes + control link metrics |
| `seed_manifest.json` | Env/policy seed reservation + best-effort collision audit |
| `gate_report.json` | `hard_blocked_families` vs `pending_not_released_families`; historical seed receipt derived from seed audit |
| `freeze_manifest.json` / `continuation_state.json` | Hash index and continuation authority |

| `git_receipt` (in protocol/continuation) | Pending until post-merge commit binding |
| `runtime_manifest.json` | `NOT_RELEASED` stub |
| `setup_manifest.json` | `NOT_RELEASED` stub (fixture keys mirror `campaign.json`) |
| `launch_matrix.json` | `NOT_RELEASED` stub |
| `compiled_ledger/accepted_ledger.jsonl` | One accepted valid row per manifest episode after attempt compilation |
| `compiled_ledger/rejected_attempts.jsonl` | Infra-invalid, superseded, and corrupted attempt inventory |
| `compiled_ledger/accepted_ledger_manifest.json` | Transitive hashes and queue/control reconciliation report |

### Accepted-ledger compiler

`tools/compile_online_correction_v4_ledger.py` consumes write-once attempt directories with `COMPLETE.json` and `evidence_manifest.json`, verifies blob hashes, classifies infra-invalid vs behavioral outcomes, selects at most one verified valid attempt per episode using `latest_verified_valid_by_attempt_id` (no outcome peeking), reconciles control reuse against accepted source episodes, and atomically emits `accepted_ledger.jsonl`, `rejected_attempts.jsonl`, and `accepted_ledger_manifest.json`. C2 prefix/response fields are copied only when the runtime recorded them; confirmatory C2 analysis remains fail-closed until the contract is complete.

### Seed collision audit limitations

The historical seed collision audit in `seed_manifest.json` is **best-effort**: it regex-scans committed JSON/JSONL under the repository (excluding `artifacts/online_correction_v4/`) for env/policy seed fields. It does not scan binary blobs, external cluster storage, or uncommitted files. A passing audit means no collision was found in the scanned scope, not a proof of global uniqueness.

## Verification boundary

Passing `validate_online_correction_v4.py` certifies deterministic freeze structure (all 15 generated artifacts byte-stable or generation-parent-normalized), prompt invariants including C2 `prompt_sha256` byte-identity semantics, control-link semantics, historical protocol integrity, seed-manifest vs queue alignment, seed receipt derivation from collision audit, setup/runtime/launch stub cross-checks vs campaign, family disposition parity across gate report and continuation, continuation/freeze hash cross-checks, and gate-report seed receipt derivation from the seed audit. It does not authorize policy inference or family release.
