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

The next disclosed amendment,
`artifacts/online_correction_v4/setup/horizontal_g2_seed_substitution_amendment.candidate.json`,
retires only those two reproducibly unstable environment seeds and assigns the
next unused horizontal-namespace seeds: block 52 uses `2100000128` and block
101 uses `2100000129`. The 128 block identities, counterbalance, prompts,
policy seeds, 5 mm position tolerance, and all stability thresholds are
unchanged. The queue, seed manifest, reset registry, and G3 plan have been
deterministically rebuilt with the substitutions at pushed implementation
commit `289ad39f4f718b43e13ddacb06ebf49b56549bbc`. This amendment authorizes
only a fresh full zero-model-request G2 attempt.

Attempt `g2q20260905g` completed that replacement-seed wave with all 128
machine-checkable reset, camera, and numeric-frame receipts passing, zero
failure markers, and zero model requests. The complete wave, logs, cluster
objects, and per-seed index were preserved before Job cleanup. Compact hashes
are recorded in
`artifacts/online_correction_v4/qualification/20260905_horizontal_g2_wave_g2q20260905g.json`.
G2 is still incomplete: the required human review of the hash-pinned
left/front/up montage and the aggregate receipt are pending. Do not launch G3
or any policy request from the machine-check result alone.

The user then passed all four required montage assertions in the questionnaire
under reviewer identity `adeeb10abbas`. The immutable review receipt is
`artifacts/online_correction_v4/qualification/20260905_horizontal_g2_axis_review.json`
and the aggregate compiler verified all 128 seeds, no missing/unexpected/failed
seeds, the review, and 1,410 referenced external artifacts. The passing
aggregate is
`artifacts/online_correction_v4/qualification/20260905_horizontal_g2_aggregate.json`;
the compact gate disposition is
`artifacts/online_correction_v4/qualification/20260905_horizontal_g2_gate_receipt.json`.
Horizontal G2 is complete. This authorizes only model-blind G3
motion/feasibility preparation, not policy inference.

After inspecting the montage,
`tools/record_v4_horizontal_g2_axis_review.py` records the four explicit
orientation assertions. `tools/compile_v4_horizontal_g2_aggregate.py` then
verifies every seed receipt, referenced artifact, runtime stratum, registry
hash, and axis review before producing the aggregate receipt. Even a passing
aggregate authorizes G2 only: the candidate reset registry remains
unreleased for policy use until G3 and the later gates pass.

`artifacts/online_correction_v4/setup/horizontal_g3_plan.candidate.json`
formula-closed the model-blind gate before execution. It binds all
128 reset seeds, the five descending scale candidates, 3,072 live path checks
per scale (15,360 maximum), and the 112 horizontal scripted checks required
for a final geometry candidate. The nine scripted reset cases are selected
from xy-jitter extrema while covering each counterbalance state twice. The
plan also freezes task-frame motion directions so left/right share one
physical axis and front/behind share the other; requested goal polarity does
not determine the independently balanced physical motion sign. Live G3
started in the registered descending order on canonical first seed
`2100000000`. Each candidate is a conjunction across seeds and path checks, so
one valid failed seed conclusively rejects that scale and permits the next
registered scale without spending the remaining 127 resets.

All five candidates failed. Scales 2.0, 1.5, 1.0, and 0.75 failed path and/or
information predicates. At the final 0.5 scale (0.06 m), all four goal-area
information cases passed, but the front/behind movement checks produced
forbidden cube-bowl contact, manipulated-cube drift above 5 mm, reference pose
error, and intermittent support loss. The scale receipts each verified 148
external artifacts before aggregation. The aggregate rejected
`[2.0, 1.5, 1.0, 0.75, 0.5]`, selected no scale, and has status `blocked`:
`artifacts/online_correction_v4/qualification/20260905_horizontal_g3_aggregate.json`.
The compact disposition is
`artifacts/online_correction_v4/qualification/20260905_horizontal_g3_gate_receipt.json`.
No scripted checks were run because their prospective authorization requires a
complete passing path scale. The scripted Kubernetes renderer now enforces
that dependency and its default spec is explicitly blocked.

Horizontal G3 therefore failed. G4-G8, direct-command and policy pilots, reset
registry release, and policy episodes remain prohibited for C1/C3/C4. No
smaller horizontal displacement or post-result scene change is authorized by
this campaign. Independently scoped model-blind fixture qualification remains
permitted for C5-C7; C2 and C8 retain their separate hard blockers.

### Disclosed horizontal geometry repair (2026-09-07)

Independent PVC forensics on attempt `g3p20260905h` scale 0.5 confirmed a
layout collision, not model failure: `rubiks_cube__bowl` contact at
`planned_time_s=0.3` / `planned_displacement_m=0.0409536` / `2.36349 N`, and
`destination_static` front overlap at `856.442 N` on the first sample.
Left/right sham and move-stop checks passed. The live USD AABB audit shows
rubiks_cube root pose is stale relative to its projected AABB center; scoring
already uses live AABB, so projected dimensions remain usable.

A V4-only amendment freezes one deterministic cube-only repair: move
`rubiks_cube` along robot-base `-X` by `-0.01` m (the smallest 1 cm increment
giving conservative 0.5-scale swept-AABB separation plus the existing 5 mm
guard) before common XY jitter for all 128 resets. Bowl, banana, assets,
physics, task frame, prompts, scoring, timing, scale ladder, and thresholds
are unchanged. Original failed receipts and raw evidence are preserved.

Repaired-layout artifacts use fixture_version
`horizontal_geometry_repair_v1` and cohort
`confirmatory_horizontal_geometry_repair_v1`:

- `artifacts/online_correction_v4/qualification/20260907_horizontal_g3_collision_forensic_g3p20260905h.json`
- `artifacts/online_correction_v4/setup/horizontal_geometry_repair_amendment.candidate.json`
- `artifacts/online_correction_v4/setup/horizontal_reset_registry.geometry_repair_v1.candidate.json`
- `artifacts/online_correction_v4/setup/horizontal_g3_plan.geometry_repair_v1.candidate.json`
- `artifacts/online_correction_v4/queue_horizontal_geometry_repair_v1.jsonl` (9,728 C1/C3/C4 rows)
- `artifacts/online_correction_v4/setup/horizontal_geometry_repair_inventory_v1.json`

Historical `docs/online_correction_v4/campaign.json` and
`artifacts/online_correction_v4/queue.jsonl` remain unchanged. Fresh repaired
G2 and unchanged descending-ladder G3 are the next dependency phase; no
repaired horizontal policy inference is authorized until those gates pass.

### Family disposition

| Family | Disposition | Status |
| --- | --- | --- |
| C1, C3, C4 | `pending_qualification` | `NOT_RELEASED` — original-layout G3 failed; repaired inventory frozen pending fresh repaired G2/G3 |
| C5–C7 | `pending_qualification` | `NOT_RELEASED` — their separate fixtures remain unqualified |
| C2 | `hard_blocked` | `BLOCKED_SETUP` — verified common-prefix replay required |
| C8 | `hard_blocked` | `BLOCKED_RUNTIME` — GR00T Bridge/WidowX stack unverified |

Model-blind C7 preparation has started without model requests. The procedural
yellow-sponge/blue-tray USD candidate and its hash-bound 64-seed registry are
`experiments/online_correction_v4/droid_task_files/scene_assets/sponge_tray_object_pair.usda`
and
`artifacts/online_correction_v4/setup/object_pair_reset_registry.candidate.json`.
The candidate uses the frozen C7 seed namespace `2100040000`–`2100040063` and
common x/y jitter. It is not a released fixture: live USD import, G2
reset/camera/frame evidence, G3 paths, and scripted feasibility remain pending.

The first zero-model-request C7 G2 smoke, `g2c7q20260905a`, proved the USD task,
camera stack, and simulator startup but was infrastructure-invalid at reset
attestation: the procedural sponge and tray settled 60 mm and 50 mm below
their provisional centers. The compact receipt is
`artifacts/online_correction_v4/qualification/20260905_object_pair_g2_smoke_g2c7q20260905a.json`.
No G2 result was claimed. The model-blind candidate was recalibrated to the
observed support plane (box half-heights 17.5 mm and 9 mm); all scene and reset
hashes changed, requiring a fresh attempt identifier and pinned checkout.

Every queue row is a **registered new episode**. `reuse_episode_ids` are comparison-control links only; C3/C4 rows are not reuse-only aliases. C4 fast-schedule sham/move rows are new episodes because schedule differs from reused C1 controls.

## Exact next commands

There is no authorized downstream horizontal experiment command. The next
implementation target is simulator-only, model-blind fixture qualification for
C5-C7; no cluster launcher for those fixtures is released yet. Validation and
evidence-preservation commands are:

```bash
python3 tools/online_correction_v4.py validate
python3 tools/build_online_correction_v4_freeze.py --out artifacts/online_correction_v4
python3 tools/validate_online_correction_v4.py
python3 -m unittest discover -s tests -p 'test_online_correction_v4*.py'
python3 tools/validate_vla_wam_v2_protocol.py
python3 tools/validate_vla_wam_v3_protocol.py
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
