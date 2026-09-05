# V4 evidence preservation, scientific scope, and compatibility boundaries

**Status:** prospective campaign design; no V4 experiment is reported as run.

**Purpose:** extend the existing spatial-instruction study into online correction
under test-time scene changes while retaining its valid static evidence. Policy
weights and the complete episode instruction remain fixed. This is a behavioral
evaluation campaign, not online learning, fine-tuning, an architecture leaderboard,
or a general study of every form of steerability.

The useful extension is to ask whether corrections are selective to the named
reference and requested spatial relation, how that behavior changes as motion
demands increase, and whether execution timing explains a measured limitation.
The old results establish why final success alone is inadequate. The new results
must explain what happens between the scene intervention and placement completion.

## 1. Authority and source order

The design was prepared against repository main commit
`ce561e66f82e95055e39d3d7711691982f6b2086`. Before implementation, record the actual
checkout commit and read:

1. `AGENTS.md` and this V4 package's entry point.
2. `docs/VLA_WAM_V3_CONTINUATION.md` and
   `artifacts/vla_wam_shared_v3/continuation_state.json`.
3. `docs/VLA_WAM_STEERABILITY_V3_PROTOCOL.md` and the relevant cohort's final
   results, decision memo, evidence manifest, and amendments.
4. `docs/VLA_WAM_CONTINUATION.md`,
   `artifacts/vla_wam_shared_v2/continuation_state.json`, and
   `docs/VLA_WAM_STEERABILITY_V2_PROTOCOL.md`.
5. `docs/WORK_LAPTOP_B200_HANDOFF.md` before using a work laptop or cluster.

The B200 handoff contains useful environment and storage instructions but also
older phase-status text. It is not authority to reopen a completed or failed
cohort. A current cohort's final machine-readable continuation entry and
hash-bound result take precedence over an older narrative progress paragraph.

`paper/EVIDENCE_MAP.md` is a convenient claim-to-source index, but its opening
commit annotation refers to an earlier manuscript snapshot. Trace a number to
the primary result and verify its manifest at the execution checkout. This
package reviews committed summaries and provenance; it does not claim a new
independent audit of every raw PVC trajectory.

### Explicit V4 extension of the simulator-state rule

The user authorized moving scene objects during evaluation with fixed weights
and fixed episode prompts. V4 therefore introduces a **prospective, narrow
extension** to the historical rule that simulator state is used only after
actions for scoring and visualization:

- Ground-truth simulator state may additionally implement a frozen event
  trigger, execute the registered scene perturbation, check physical validity,
  and timestamp/log the intervention.
- State, object identities, target coordinates, scene graphs, trigger flags,
  success predicates, and privileged controller outputs must never be added to
  the learned policy's observations or prompt.
- Policy inputs remain the released interface's ordinary images, proprioception,
  and static instruction. The observation adapter must be auditable.
- The event trigger must not coach the policy, move the manipulated object into
  a grasp, rescue a dropped object, switch the instruction, or select a favorable
  motion from the policy's predicted future.
- A privileged controller is a separately labeled engineering/feasibility
  control. Its runs never enter learned-policy outcome denominators.

Record this scope in the V4 freeze and the additive repository operating
instructions. Do not rewrite V2/V3 rules or retroactively apply V4 exceptions to
historical results. The old ban on unlisted V3 expansions remains applicable to
V3; this separately identified V4 campaign is the authorized new work.

## 2. What the completed static results already establish

All lengths below are centimeters unless explicitly stated. LEFT-minus-RIGHT
signed endpoints and RIGHT-minus-LEFT requested-side depths are different
quantities; preserve the original signs in reanalysis.

| Evidence | Verified result and population | Role in the new paper | Limit that must remain visible |
| --- | --- | --- | --- |
| V3-B002, π0.5 movable-object position reflection | 27 matched reset/sampling-seed blocks, four cells each, 108 valid episodes. Control LEFT 4/27, RIGHT 25/27; reflected LEFT 25/27, RIGHT 9/27. RIGHT-minus-LEFT success changes from +77.8 to −59.3 percentage points. | Main-text static motivation: a single checkpoint receives opposite apparent directional diagnoses after changing object positions. | Specific DROID/RoboLab runtime and nonpublic historical checkpoint release. Reflection changes physical and visual geometry; it does not isolate a language module. |
| V3-B002 continuous behavior | Mean LEFT-minus-RIGHT signed endpoint difference +19.2 in control and +18.1 after reflection. Reflected-minus-control interaction −1.1, 95% CI [−7.8, +5.7]. Requested-side-depth interaction −34.6, 95% CI [−41.4, −28.5]. | Show that continuous instruction-responsive behavior and binary completion answer different questions. | A nonsignificant interaction does not prove invariant response. An ordered endpoint difference does not imply every individual trial completes correctly. |
| V3-B001, Cosmos3 Nano reflection | 108 valid episodes. Control LEFT/RIGHT 26/27 and 26/27; reflected 27/27 and 23/27. Endpoints are correctly ordered in 27/27 pairs in each layout. Requested-depth interaction −24.8, 95% CI [−32.4, −17.3]. | Public anchor and illustration of a geometry effect partly obscured by near-ceiling binary success. | Full-sample continuous analysis includes all valid failures. The separate all-four-cells-successful margin analysis has only 21 blocks and must retain that denominator. |
| V3-B003, DreamZero reflection | 108 valid episodes. Control LEFT/RIGHT 5/27 and 8/27; reflected 25/27 and 25/27. Requested-depth interaction −14.1, 95% CI [−19.9, −8.2]. | Supporting replication without making DreamZero a fourth new campaign policy. | Its scene dependence differs from π0.5's binary reversal. Do not describe every model as showing the same success-gap reversal. |
| V3-B005, Nano seven-position lateral sweep | 15 matched blocks × seven bowl positions × two directions = 210 valid episodes. Full-support requested-depth slope 1.125 m/m, 95% CI [0.719, 1.562], positive in 13/15 blocks. | Compact static dose-response context and a guide to feasible geometric scale. | The fitted zero crossing lies outside registered support. The middle-five-position slope is a post hoc sensitivity analysis. This is not a validated prediction of an unseen reversal or of a dynamic speed limit. |
| V3-C002-R001, π0.5 reference-inverted wording | 341 four-cell blocks, 1,364 valid episodes. Canonical endpoint difference +23.281; inverse +0.191, 95% CI [−0.884, +1.272]. Inverse-minus-canonical requested depth LEFT −9.783 and RIGHT −13.307; equivalence was not authorized. | Motivate a smaller, syntax-matched wording comparison within the new online study. | Historical prompts changed more than predicate/argument order. π0.5-only evidence in the registered symmetric object layout. R001 was a prospective operational repair after the original isolation gate failed. |
| V3-E004, symmetric object-layout cohort | 4,096 valid episodes and 2,048 pairs across separate checkpoint/arena slices. π0.5's full-endpoint binary gap changes from about +74.2 to +20.2 points; full-endpoint depth gap from about +17.0 to +5.75. Its separate 27-block core binary interaction is −51.9 points, 95% CI [−81.5, −22.2]. | Supporting evidence that a scene intervention can reduce a gap without removing it. Compress heavily in the main narrative. | The endpoint expansion and matched core have different inferential populations. The scene package includes a companion-object inventory transition. It is not positions-only reflection and not robot symmetry. |
| V3-E001, fixed-observation prompt/noise diagnostic | 336 valid model requests, zero executed behavioral episodes; 12/12 exact repeats bit-identical. | Reuse the interface-repeatability and seed-validation method before distributed V4 inference. | Prompt-response action RMS is not spatial correctness. DreamZero's zero same-prompt sampling denominator is undefined as a ratio, not evidence of an infinite effect. |
| V3-A and prior screen | Frozen V3-A contains 648 completed launch-authorized episodes; historical measurement audit covers 982 unique episodes with signed final lateral position available throughout. | Supplementary context and provenance for checkpoint selection. Reuse existing logged measures without new model inference. | Counts refer to different scopes and are not additive. Never present all legacy rows as a new balanced cross-model comparison. |

### Primary repository paths for this table

Use these paths rather than copying numbers from a chat transcript:

```text
artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/results/pi05_v3b002_report.json
artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/nano_v3b001_summary.json
artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003/results/dreamzero_v3b003_summary.json
artifacts/vla_wam_shared_v3/phase_b/nano_lateral_sweep_v3b005/results/nano_v3b005_dose_response_report.json
artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/activation_v4/final_analysis_v3/results/results.json
artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/activation_v4/final_analysis_v3/results/DECISION_MEMO.md
artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/results/results.json
artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/DECISION_MEMO.md
artifacts/vla_wam_shared_v3/phase_e/fixed_observation_prompt_noise_v3e001/results/compiled_results.json
artifacts/vla_wam_shared_v3/phase_e/fixed_observation_prompt_noise_v3e001/DECISION_MEMO.md
artifacts/vla_wam_shared_v3/measurement_coverage_audit.json
paper/EVIDENCE_MAP.md
docs/VLA_WAM_V3_CONTINUATION.md
```

The historical measurement audit is specific to the named legacy snapshot. Do
not relabel its 982 covered episodes as coverage of all later V3 or V4 results.

## 3. Failed gates are useful evidence, not reusable positive controls

### E002: the old privileged controller did not establish feasibility

V3-E002 completed 108 valid model-blind absolute-IK controller episodes with zero
learned-model requests. The deterministic waypoint recipe failed at pickup in
every episode. This is a failed controller diagnostic; it does not show that
LEFT and RIGHT are equally feasible, that either requested placement is
reachable from a valid grasp, or that a dynamic intervention is mechanically
easy. Do not reuse it as V4's mechanical ceiling or as a successful oracle.

Source:
`artifacts/vla_wam_shared_v3/phase_e/reference_controller_symmetry_v3e002/`,
with the completed status also documented in `docs/VLA_WAM_V3_CONTINUATION.md`.

### E005: the old second-arena language control failed

V3-E005 is complete at 108/108 valid LingBot-VA RoboTwin episodes and 54 matched
pairs across seven scene clusters. The preregistered endpoint-redirection hard
gate H4 failed at both layouts. Its rule required mean LEFT-minus-RIGHT endpoint
difference greater than 0.05 m and a scene-clustered 95% interval above zero.
The respective means were approximately +0.0235 m and −0.0821 m, and both
intervals included zero. H1–H3 remain withheld; the reflected extension was not
released. A failed positive control is not evidence that geometry has no effect.

Keep the failed gate in supporting material. Never unhide H1–H3, rerun the valid
cells, or substitute a different checkpoint into the old V3-E005 identity. A
new V4 robot-stack test needs its own checkpoint, task, population, eligibility
rules, and identifier, declared before its outcomes are inspected.

Sources:
`artifacts/vla_wam_shared_v3/phase_e/cross_arena_geometry_v3e005/DECISION_MEMO.md`
and the `phase_e_v3e005` entry in V3's `continuation_state.json`.

### E006: do not start V4 from an unvalidated constructed grasp

The original V3-E006 state-localization experiment stopped before learned-policy
inference because no canonical-grasp state passed its fixed validity gate. Its
publication decision explicitly withholds a stage-localization plot.

By E006-R012, all four reachable-pose diagnostics passed, but every one of four
registered grasp/carry candidate pairs failed at least one unchanged gate.
Across eight stages, physics passed **0/8**, while OOD, camera, companion, and
frame-identity checks passed **8/8** each. Cube angular-speed checks passed
0/8 and no-unintended-contact checks passed 0/8. There were **zero accepted
states, zero model requests, and zero behavioral episodes**. A patch handling
an extra validator argument did not convert these failures into accepted states.

V4 should observe naturally reached events during the policy's ordinary rollout
for its primary perturbation trigger. A valid robot pose or plausible rendered
image is insufficient evidence of a valid grasp. Trigger eligibility and the
unconditional probability of reaching that trigger must both be logged. A
post-grasp analysis is conditional on reaching the event and cannot stand in
for full task performance.

If a new snapshot/restore mechanism is used to branch after a naturally reached
event, validate simulator, contact, controller, policy memory, action buffer,
RNG, and observation-history restoration. A physics-only snapshot is not a
matched policy-state counterfactual. If this gate fails, use independently
replayed matched resets with event alignment and report their pre-intervention
differences; do not pretend the post-grasp states were identical.

Sources:

```text
artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006/V3E006_PUBLICATION_DECISION.md
artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r012/results/DECISION_MEMO.md
artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r012/results/results.json
```

## 4. How static evidence connects to the online campaign

Historical static experiments and new stationary controls serve different roles.

| Reuse or new control | What it answers | Required handling |
| --- | --- | --- |
| Reuse existing reflection, continuous-position, and selected wording results | Why a spatial evaluation needs measures beyond final success. | Regenerate figures from frozen artifacts; no new inference and no changed denominators. |
| New stationary control in each newly implemented relation/setup/runtime | Can this checkpoint perform the newly specified task without intervention? | New V4 identity and matched logging; this is a necessary control, not a legacy rerun. |
| New stationary control at the intervention's final reference position | Is the eventual goal region achievable when presented from the start? | Match object inventory, scoring, runtime, and allowable workspace; it does not equate the robot's post-grasp state with a reset state. |
| New event-aligned no-motion condition | What would behavior near the trigger look like without reference motion? | Same trigger and intervention machinery with zero displacement; avoid extra pause or observation refresh only in the motion arm. |
| New irrelevant-object or named-reference comparison | Is adjustment selective to the task-relevant object? | Keep the physical motion, visible inventory, and prompt carrier matched as far as the design permits. |

The intended paper progression is: **static ambiguity → online intervention →
goal-specific adjustment → limits and implementation consequences**. The
online conditions must preserve the relation at a declared placement-completion
event, not require indefinite maintenance after the policy already completes
the task. There is no need to expand every old LEFT/RIGHT ablation to every new
direction.

The agreed relation scope is four horizontal directions studied deeply
(LEFT, RIGHT, FRONT, BEHIND), two vertical directions tested in a focused feasible
supported setup (ABOVE, BELOW), and one containment bridge (INSIDE). Inverse
wording gives the same physical goal, e.g. cube INSIDE bowl versus bowl CONTAINS
cube. ON TOP requires support/contact and is not an interchangeable synonym
for ABOVE. Use the package's geometric definitions, not a VLM score, for policy
outcomes.

## 5. Checkpoint identity and deployment compatibility

The campaign's main matrix is authoritative for the final selection. The
intended structure is two main policies with substantially different released
systems and one focused second-stack policy. These are fixed-checkpoint
comparisons; architecture, training, data, controller, and embodiment effects
are not individually identified by their differences.

| Candidate role | Existing evidence and exact caution | Required gate before freezing V4 |
| --- | --- | --- |
| Main public WAM: Cosmos3 Nano Policy DROID | Existing release `nvidia/Cosmos3-Nano-Policy-DROID`, recorded revision `6706d7680581c255ff61e0f3bb49d90eac55c79e`; distinguish it from base Cosmos models, Edge Policy, and static Cosmos-Reason2 diagnostics. | Resolve the selected snapshot and all files, action statistics, source revisions, observation transforms, guidance settings, controller mapping, and cadence. Verify ordinary rollout and new event/motion hooks. |
| Main π0.5-style VLA | Historical key `pi05_droid_jointpos_polaris`; manuscript manifest abbreviation `v2a010-manifest-f5a56d9565f9381cc`; historical manuscript explicitly labels its checkpoint release nonpublic. V2-A010 used OpenPI `c23745b5ad24e98f66967ea795a07b2588ed6c79`, RoboLab `0aef241fb088ca21bb4ebd24448940ed56620d17`, horizon 15. | Verify artifact access and exact hashes on the workers. If a public joint-position-compatible replacement is selected, create a new V4 identity and disclose that it is not the historical checkpoint. A model family name or config string is insufficient provenance. |
| Focused second-stack candidate: GR00T N1.7 SimplerEnv Bridge / WidowX | This is distinct from old GR00T N1.7 DROID. The old DROID screen had LEFT 3/27, RIGHT 0/27, with 49/54 pickup failures. It is not an established online positive control. | Confirm the actual released Bridge checkpoint, access/license, action normalization, controller, camera views, simulator/reset compatibility, and report ordinary task competence without making a positive success rate an inclusion gate. Do not substitute an arbitrary GR00T checkpoint or imply a hierarchical policy was tested. |

A released interface's action units and controller semantics must be preserved.
Do not equate a standard real-robot joint-velocity checkpoint with a simulator
joint-position artifact or modify action scaling until a policy appears to
work. Legitimate adapter fixes need a logged, prospective runtime revision and
new invalidation boundary before confirmatory inference.

Do not compare raw action RMS across checkpoints as though all channels had the
same physical units. Behavior in world/robot coordinates and logged execution
timing are the comparable measurements. Generated futures are supplementary
only where the released interface exposes decodable frames; missing predictions
are unavailable, not zeros. Retain the native raw output where permitted so
later analysis does not require rerunning expensive policy inference.

### Evidence-based eligibility without model shopping

Before the large campaign, apply the package's same frozen engineering and
baseline-measurement and technical-validity rules to every nominated policy/setup. A low success rate or absent language effect alone does not block a technically valid family. Disclose all failed
candidates and setup attempts. No checkpoint or scene may be replaced after
inspecting confirmatory online outcomes. A blocked stack remains a documented
missing replication; it is not repaired by silently changing models.

Known historical competence is a reason to select Nano as an anchor, not
permission to assume it can execute vertical, containment, or multi-reference
tasks. Inability to pass a new setup gate is a meaningful scope limitation.
Completing large numbers of online trials from a floor-level static baseline
would not resolve the intended question.

## 6. Reuse rules for implementers and analysis agents

1. Preserve `artifacts/vla_wam_shared_v2/protocol.json` and
   `artifacts/vla_wam_shared_v3/protocol.json` byte-for-byte. Save a hash ledger of
   touched historical source artifacts before and after the V4 work.
2. Do not rerun a valid completed historical cell to obtain nicer video, a
   missing newly invented metric, or a more favorable success rate. Reanalyze
   existing logs where the measurement is actually recoverable; otherwise
   record it as unavailable for that cohort.
3. Do not assume old traces include observation capture timestamps, receipt
   times, buffer ownership, or reference motion. Those cannot be reconstructed
   from a final endpoint. Log V4's required fields prospectively.
4. Keep every valid learned-policy failure in the appropriate denominator.
   Distinguish infrastructure-invalid attempts, physics-invalid interventions,
   no-trigger episodes, and valid post-trigger failures without deleting any
   ledger entries.
5. Preserve exact cohort, checkpoint, runtime, reset, and seed identities.
   Repeated seed numbers across cohorts do not automatically make a paired
   comparison, and independent sampling seeds at one physical reset are not
   independent scene replicates.
6. Keep DROID/RoboLab, SimplerEnv/WidowX, and RoboTwin outcomes separate. A
   shared figure is permitted; pooled raw numerators and denominators are not.
7. Keep pilot and confirmatory seed namespaces disjoint. Acceptance of a
   bounded engineering pilot is not evidence supporting the paper's new
   behavioral hypotheses.
8. Do not promote failed H4 or E006 results into positive causal findings. Keep
   failed-gate dispositions visible in the final coverage table.
9. Commit compact results, manifests, hashes, selected bounded media, analysis
   code, and figures. Checkpoints, environments, full videos, action arrays,
   and raw simulator collections belong on persistent cluster storage.
10. Preserve unrelated working-tree changes. Cluster availability permits
    parallel execution of released cells, not bypassing validity gates or
    claiming machine-specific numeric behavior is identical without a check.

## 7. Deliverable from the evidence-preservation agent

Before manuscript integration, produce a compact machine-readable index with
one row per reused result:

```text
legacy_cohort_id
source_commit
source_path
source_sha256
runtime_or_checkpoint_identity
arena
population_definition
independent_unit
valid_episode_count
reported_measure_and_units
original_sign_convention
estimate_and_interval
claim_boundary
new_figure_path
new_figure_sha256
```

Regenerate the chosen static panels without policy inference, compare every
headline number against the frozen result file, and have a second agent review
the sign, units, denominator, and claim. A shared plot must visibly distinguish
historical static observations from the new confirmatory online cohort.

The campaign should finish with enough evidence to write either a positive or
a negative result. A useful negative finding could be a failure to make
reference-specific corrections despite established static competence; it must
not be confused with a setup that never reached valid manipulation states.
