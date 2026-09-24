# SGW-01: experiment coverage and remaining work

Compared with the live [Overleaf plan](https://www.overleaf.com/project/6ab2bdb39b30df4bbc98a15e) on 24 September 2026 UTC (23 September in New York).

**The planned matrix covers the stated language comparisons. Execution and analysis are not yet complete.** The clean-scene campaign supplies LAT and HEIGHT fixtures; it does not supply DIST, learned-policy results, or prediction accuracy. This document gives each missing part a concrete deliverable. It is an implementation handoff, not permission to expand the experiment or change running workers.

## 1. Exactly what the paper tests

| Question | Comparison | Required output | Extra learned episodes beyond the core queue |
|---|---|---|---:|
| Does changing the requested goal change the result? | Positive versus negative physical goal, separately within D, C and I | Signed physical endpoint separation, each goal's success and requested margin, paired distributions | 0 |
| Does the changed sentence construction matter? | C minus D, same layout and physical goal | Paired success and margin changes, split by goal | 0 |
| Does an equivalent reversed-reference description preserve the goal? | I minus C, same layout and physical goal | Primary success contrast; continuous margin contrast; success-to-failure transitions | 0 |
| Does the effect occur beyond left/right? | Repeat all three comparisons within LAT, HEIGHT and DIST | Separate model × family tables; side-stratified results | 0 |
| Is a generated future better than assuming nothing changes? | Prediction versus persistence on the same current/actual images and physical horizon | Relative-position error and persistence skill; target and anchors separately | 0; reuses recordings |
| Where do prediction and execution disagree? | Predicted and actual relation at the same measured time | Four agreement/disagreement categories plus unknown coverage and bounds | 0; reuses recordings |
| At what observable stage does behavior fail? | Pickup, transport, placement, release and terminal stability traces | Stage timelines; wrong-object/reference motion; first success versus fixed endpoint | 0; descriptive analysis |
| Are apparent prediction effects caused by stationary or unobservable frames? | Moving/stationary prefixes, first versus later requests, coverage by condition, alternate supported H | Prespecified sensitivity tables with denominators | 0; reuses recordings |

The language forms are **D = direct**, **C = subject-first clause**, **I = equivalent inverted clause**. I−C is primary; I−D alone is insufficient. D−C changes a bundle of wording features, including “Put” versus “Place … so that”; it does not isolate sentence length or syntax alone. I−C tests the specified reformulation, not an isolated neural parsing mechanism. DIST has a comparative clause and two anchors, so family differences also reflect different task demands.

HEIGHT means *higher/lower than the bowl*, with supported released placements. It does not cover image-up/down, direct overhead stacking or literal vertical velocity. DIST means *closer/farther from the bowl than from the plate*. It does not cover moving closer than the initial state, toward/away from the camera or egocentric depth. These distinctions belong in the paper, not just in an agent note.

## 2. Counted coverage

The checked `planned_cells.csv` contains 18 exact prompts and 174 six-condition blocks. Every block has D+/D−/C+/C−/I+/I− once. It contains every combination of two models, three families and the following stages:

| Stage | Layouts/family | Episodes/model/family | All six model-family branches |
|---|---:|---:|---:|
| P: recording pilot | 1 | 6 | 36 |
| D: development | 4 | 24 | 144 |
| C: confirmation | 24 | 144 | 864 |
| Total | 29 | 174 | **1,044** |

These are planned counts. The CSV deliberately remains `PLANNED_NOT_RELEASED`; execution evidence comes from separate immutable assignments, releases and completion receipts. No planned row is counted as a result. P/D never enter confirmation estimates. Both models must receive the same accepted physical layouts for a matched cohort.

Separate budgets: 12 fixed-input recording-check requests (six/model), up to three extra official-reference calls only if qualifying a patched D1 path, six scripted physical trials per candidate, and offline annotation/calibration. None is an additional independent learned-policy episode. The current clean campaign is a disclosed new fixture campaign after old failures; its candidate budget is not retroactively merged with the original 100-candidate pools.

## 3. Evidence available at this check

Sources: cluster branch `70a6bbaa3660992fc5998e5e38bcc812a32ab4a1`, scene branch `93fc172dfcd22fce1778e507dcefe0022557a36e`, and a workstation read at **02:17:50 UTC on 24 September**. The fetched cluster branch still describes **23:40 UTC on 23 September**; this is not a fresh cluster execution count.

| Part | Observed evidence | Still needed |
|---|---|---|
| Clean LAT template | Prototype 06 independently passes 6/6 scripted trials | Full distinct scene set and approach-side resolution below |
| Clean HEIGHT templates | Prototypes 04 and 05 each independently pass 6/6 | Full set with 12 upper-left and 12 upper-right confirmation layouts |
| Running clean LAT pool | 3 all-six passes; 1 valid physical rejection; next candidate running | 29 qualifying layouts; do not replace failed trials |
| Running clean HEIGHT pool | 3 all-six passes: 2 upper-left, 1 upper-right; next candidate running | 29 qualifying layouts with the frozen side quotas |
| DIST | Existing cluster implementation and partial old evidence; absent from this clean campaign | Independently verified final accounting, both anchor arrangements and a clean-appearance decision |
| N3/LAT MAIN P | Latest committed cluster snapshot: six assigned pilot episodes launched, first still in progress at that snapshot | Fresh completion receipts; do not infer current completion from that snapshot |
| N3 fixed-input recording checks | Six genuine responses and offline decode reproduction recorded in cluster artifacts | Not a behavioral denominator or physical-time alignment proof |
| D1 official conditional path | Adapter/producer implementation exists | Current native runtime, reset, decoding and timing receipts; historical custom s2 runs cannot substitute |
| Analysis | Behavioral compiler, statistical helpers and annotation record types exist | End-to-end reports and prediction metrics described below |

No confirmation cohort is established by these sources. Historical Nano/Edge wording results, reflection results and π0.5 inversion results remain separate motivating evidence.

## 4. Bounded tasks for execution agents

Each task should return artifact paths, source/asset hashes, actual counts, failures, and a concrete remaining blocker. A source file or a running process is not a completion receipt. These tasks can be assigned independently where their dependencies permit; no running workload is changed by this handoff.

### T1 — Finish and transfer the clean LAT/HEIGHT fixtures

**Input:** unchanged `SGW-CLEAN-20260924` plan SHA-256 `3fdd3ab3fecad77aafd67d5fac834b2a3f7a7df729a1a3888e1d94b2014cabd1`; existing workstation workers.

**Deliver:** per-family `assignments.json`, six-pass receipts for every selected layout, all failed candidates, measured geometry/appearance/camera identities, archive manifests and usable videos. HEIGHT must allocate one right-side pilot, two development layouts per side, and twelve confirmation layouts per side. Count passing candidates and *assigned* layouts separately.

**Acceptance:** 29 distinct qualifying candidates per family, side quotas checked from measured support heights/poses, reproducible restoration on the target simulator. Reproduce one exact accepted LAT fixture and one of each HEIGHT support orientation before expanding target-host collection. Engineering reproduction stays outside model denominators. Do not repeat already running workstation trials.

**Important unresolved LAT requirement:** spec §5.4 asks for balanced starting approach side. The current LAT generator translates one historical arrangement and its selector takes the first 29 passes without an approach-side field or quota. All authored cube y coordinates remain negative (about −0.239 to −0.119 m); no balanced approach-side claim follows. Both *goal* sides are tested, which is a different control. Define approach side relative to a documented robot/reset frame and check it from native captures. If a second stratum is required, register a separate finite mirrored/alternative geometry campaign before its outcomes; do not edit or relabel this running pool. Alternatively disclose the restricted workspace and explicitly revise that design claim before confirmation. Until resolved, the pool is usable engineering evidence, not proof of this counterbalance.

### T2 — Complete the DIST scene decision

**Input:** existing cluster DIST captures, candidate outcomes and administrative-pause ledger. Start by recovering/verifying completed evidence, not by restarting the search.

**Deliver:** one exact workable bowl-left scene and one bowl-right scene, each with six scripted checks, clear shoulder/wrist captures, then a finite assignment plan for 29 distinct fixtures. Cube starts on the bowl/plate perpendicular bisector within 5 mm; both released goals attain the existing 30 mm margin with both anchors undisturbed. Keep the plate visibly distinct from supports/pads.

**Acceptance:** 12 bowl-left/12 bowl-right confirmation fixtures, 2/2 development and the predeclared pilot side. Preserve all seven historically unstarted DIST slots as pending unless separately resumed under their own authority. The current workstation pool contains no DIST rows.

Use the same declared clean appearance for a unified three-family clean cohort. Applying the visual change to an old DIST layout creates a new visual condition and needs fresh rendering/qualification records. If DIST remains in the office appearance, report it as a separate visual cohort; do not attribute raw cross-family differences to the relation alone. Any new candidate pool is a disclosed amendment, not a refill of the old cap.

### T3 — Reconcile old pilot cells with the new appearance

**Input:** actual cluster cell ledger, existing MAIN N3/LAT P assignment and clean campaign assignments.

**Deliver:** one table mapping each model/family/stage/layout/cell to physical scene, appearance, runtime and completion record. Reserve the historical MAIN pilot against D/C reuse. A clean campaign's `P01` label is a proposed fixture role, not authorization to overwrite the already assigned MAIN P cells.

**Acceptance:** no completed/claimed cell is reassigned or rerun, no duplicate pilot counted, and clean versus office episodes remain distinguishable. State whether the old pilot is retained as technical development evidence or whether a separately registered clean recording pilot is needed. If new model cells are added, explicitly amend the budget; do not hide them inside 1,044. Both models share one frozen confirmation cohort. Runtime/recording qualification, not a favorable success rate, determines progression.

### T4 — Complete the physical prediction comparison

**Input:** original N3/D1 request inputs, returned/executed actions, decoded frames/latents and simulator clocks.

**Deliver:** a model-specific table mapping every usable generated frame to camera, request/reset ID, physical target time and executed action interval; a validated positive horizon H frozen on development; deterministic request selectors at floor[0.25(n−1)] and floor[0.75(n−1)] with deduplication; a localization/annotation export and persistence report.

**Acceptance:** align within the actual executed prefix (N3 up to 32 actions; D1 eight), including context-frame offsets and preprocessing/crop/packing transforms. Video presentation FPS alone is insufficient. Use the exact same view and physical time for predicted and actual images. Save actual policy-input arrays, not just a visualization camera. The scene has left/right shoulder cameras and one wrist camera, but availability does not prove what each model consumed. Current scripted trial videos are left-shoulder; wrist/right captures at warmup do not provide continuous wrist ground truth.

For normalized relative image coordinates u of each target-anchor pair, compute prediction error `||u_pred(H)−u_actual(H)||`, persistence error `||u_current−u_actual(H)||`, and skill = persistence error minus prediction error. Normalize by image diagonal consistently. Retain per-object localization errors; DIST includes both anchors. Two blinded raters and adjudication above 0.02 image diagonals are required. Average selected requests within an episode, then paired episodes within a layout. Continuous accuracy is conditional on visibility; report its denominator. Retain first requests for matched-input analysis and keep later closed-loop associations distinct.

**Current code gap:** `prediction_annotations.py` stores categorical labels and provenance, but not center coordinates, localization disagreement or persistence skill; these calculations are not present in the inspected SGW module. Implement and verify them before calling the prediction analysis covered. A failed time map blocks fidelity scoring for that branch, not preservation of its valid behavioral evidence.

### T5 — Validate relation labels, then measure disagreement

**Input:** qualified views/supports and T4's mapping. **Deliver:** 50 offline calibration images per family: 40 balanced unambiguous states (10 per sign for development and 10 per sign for held-out validation) plus 10 ambiguous/occluded variants. This resolves the prose's “40 … including 10” ambiguity consistently with `protocol.json`'s 50-state count; record this clarification before annotation. Keep development and held-out splits fixed.

**Acceptance:** at least 19/20 correct signs on unambiguous held-out states, with unknown allowed and not counted correct; no forced sign on genuinely ambiguous cases. A changed crop/view/label procedure needs separate validation, not reuse of another camera's score. Neither image-up nor 2D proximity automatically establishes world height/distance. Do not let actual future images or privileged state reveal the predicted label to annotators.

Then export prediction-consistent/execution-inconsistent, execution-consistent/prediction-inconsistent, both consistent, both inconsistent and unknown at matching H, with counts/coverage by model/family/form/goal and full-cohort binary missingness bounds. Compare forecast and actual state at H, not forecast H against task success at action 450. Freeze stationary/moving handling at the specified 0.02-image-diagonal threshold. No new model runs are needed.

### T6 — Finish the behavior and failure-stage reports

**Input:** all six cells per layout, per-control-step object/reference state, contacts, gripper status and true simulated timestamps.

**Deliver:** I−C primary and C−D secondary success/margin tables; each goal separately; endpoint separation; transition tables; stage timelines; first-success versus final-success results; safety/technical-missingness accounting. Use the existing pickup/release/stability predicates. Report the first observed failed criterion and overlapping failures; do not force a unique internal cause.

**Specific implementation correction:** `compile.py::_confirmation_estimates` currently labels `S(+)−S(−)` as `separation_by_form`. That is a difference in binary goal success, **not** the registered physical separation `r(+)−r(−)`. With `M=q*r`, the latter is `M(+)+M(−)`. Preserve a separately named success asymmetry if useful, and add the physical separation in metres with missing/censored endpoints handled explicitly. A regression example with both goals successful at +0.13/−0.13 m must report physical separation 0.26 m, not zero. Opposite-goal reversals must change its sign.

Wire existing bootstrap, sign-flip, Holm, censoring-bound and equivalence helpers into the actual outputs. `compile_primary_statistics` can currently label a branch complete from a nonempty subset: expose `n_complete/24`, pending/censored cells and conditional status; do not describe partial coverage as completed confirmation. The compiler's equivalence output is currently `None`; retain that until both prescribed success and margin criteria, including missingness, are evaluated. Use 20,000 layout resamples, 100,000 sign flips and the six-test correction, not frame-level replication. Give numerical estimates and CIs, not only helper availability.

### T7 — Freeze the final analysis outputs before confirmation

**Deliver:** one reproducible command producing the six model-family coverage rows, main language-effect table, prediction/persistence table, disagreement/coverage table, side-stratified and stage diagnostics, and paper figure inputs. Demonstrate it on a small labeled fixture dataset containing success, valid manipulation failure, anchor disturbance, safety censoring, technical missingness and unobservable forecasts. These are software checks, not experimental observations.

**Acceptance:** absent families/models render “not run/unavailable”; no substitution of historical custom DreamZero or π0.5; no missing predictions converted to zero; all statistics trace back to cell/request IDs. Alternative supported H, stationary/moving, first/later requests and per-goal effects are declared sensitivity analyses. Additional hypotheses discovered after confirmation require separate development and a new registered cohort if a confirmatory claim is intended.

## 5. What is intentionally not in the core queue

| Extension | Treatment |
|---|---|
| π0.5 action-only comparison | Optional fresh LAT control: 174 additional episodes on the same fixtures/forms, separately qualified. Historical 341-block inversion evidence is background only. This is not a causal ablation of world modeling. |
| Clean versus office background | We changed the experimental setting. Without matched learned-policy runs at identical geometry, this is not a background-robustness result. Do not add that factorial by default. |
| Wrist versus shoulder input / camera-frame instruction | Cameras remain fixed in the core study. Saving wrist images is not a camera ablation. Requires a separate interface-qualified design and budget. |
| Negation, free paraphrases, anchor-name substitutions, extra models, guidance sweeps | Optional, unqueued. D/C/I covers the specified descriptions, not general language robustness. |
| Stable-grasp/oracle intervention | Separate experiment; the historical attempted fixture failed validity. Stage timelines do not replace a causal intervention. |
| World model on versus off | Not established by comparing differently trained models. Do not claim this paper proves the causal value of world modeling. |

The core contribution requires both trustworthy language contrasts and useful, physically aligned information from generated futures. If only behavior is measurable, narrow the paper explicitly; do not quietly present a wording benchmark as a completed prediction study.

## 6. Immediate order

Keep the finite LAT/HEIGHT workers running unchanged. In parallel, prepare T2's DIST evidence decision, T3's cell/appearance reconciliation, T4–T5's prediction measurements and T6–T7's analysis outputs. Resolve LAT approach-side scope before confirmation assignment. None of those CPU/documentation tasks requires waiting for all 29 scenes. Do not start new learned policies or modify active cluster workers solely because this checklist names unfinished work.
