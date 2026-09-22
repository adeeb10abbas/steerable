# Spatial instructions in world–action models: experiment specification

Study **SGW-01**, version 1.1, 22 September 2026. Owner: Ali-Adeeb Abbas.

**Purpose:** Give execution agents a finite, resumable study of when spatial language changes predicted and executed behavior, and when equivalent descriptions fail to preserve the same goal.

**Status at handoff:** Specification and planned queue only. No SGW-01 model requests, fixture qualification, cluster deployment, or behavioral results have been completed. The worker interface described in `CLUSTER_RUNBOOK.md` must be implemented and qualified. This document replaces the September 12 forecast-only plan as the proposed next study; it does not modify any frozen V2/V3 experiment or stop any separately running job.

**Deliverables:** this specification; `protocol.json`; `prompts.json`; `planned_cells.csv`; `CLUSTER_RUNBOOK.md`; `AGENT_HANDOFF.md`; and `paper/main.tex`. The Overleaf research plan is https://www.overleaf.com/project/6ab2bdb39b30df4bbc98a15e. The prose manuscript is a plan, not a report of new findings.

## Workshop focus

The core paper is WAM-only, centered on whether generated futures remain reliable under spatial instruction changes. Our primary fit is Motion 6 (benchmark design), with a narrower connection to Motion 4 (prediction reliability for evaluation). `WORKSHOP_FIT.md` fixes the paper framing and the role of π0.5. Historical π0.5 results remain background, not a matched control. A fresh LAT baseline is optional and is not in this queue.

## 1. Question and contribution

When a spatial instruction fails, determine which observable property failed:

1. Changing the physical goal changes movement in the intended direction.
2. Restating the same goal preserves the requested outcome.
3. Reversing the subject and reference in a relational clause preserves the same physical relation.
4. Generated futures accurately describe the motion produced by the actions actually executed.

Compare these properties across lateral position, height, and relative distance. The proposed contribution is a set of controlled comparisons and the failure patterns they reveal in world–action models (WAMs). Generated futures supply a second measurable output. They are not a direct measurement of internal understanding, and a jointly generated future is not necessarily an upstream plan that causes the actions.

The principal language contrast is **reference-inverted minus syntax-matched wording**. Direct versus syntax-matched wording measures the effect of changing sentence construction. Opposite physical goals provide a directional control in every wording condition. Testing only direct versus inverted wording would mix these effects.

## 2. Evidence already available

Historical evidence is pinned to steerable commit `ce561e66f82e95055e39d3d7711691982f6b2086`. It is development context, not fresh confirmation data.

| Evidence | Completed observations | Appropriate use |
| --- | --- | --- |
| π0.5 reference inversion, V3-C002-R001 | 341 matched blocks in one symmetric scene; 1,364 episodes. Physical LEFT-minus-RIGHT endpoint contrast 23.28 cm [21.89, 24.56] with canonical wording and 0.19 cm [-0.88, 1.27] with inversion. LEFT success 210→151/341; RIGHT 297→240/341. | Background appendix only; excluded from the main WAM evidence table. This historical cohort is not a matched SGW-01 control. |
| Nano four phrasings, V3-C001 | 160 episodes, 20 seeds × four forms × two directions. Direct 39/40; short 39/40; outcome 35/40; desired-plus-negated-opposite 21/40. | Exploratory WAM wording sensitivity; forms share seeds. |
| Edge four phrasings, V3-C001 | 160 episodes. Direct LEFT/RIGHT 18/20 and 20/20; short 5/20 and 19/20. | Exploratory WAM wording sensitivity; not the reference-inversion test. |
| Nano reflection, V3-B001 | 108 episodes in two fixed layouts; success 52/54 and 50/54. | Geometry sensitivity and recording development. |
| DreamZero custom s2 reflection, V3-B003 | 108 episodes, effective noise fixed at 1140; success 13/54 and 50/54. | Geometry sensitivity of the disclosed custom configuration; not default DreamZero. |
| Canonical-grasp intervention, V3-E006 | State-validity checks failed; zero accepted states and zero model episodes. | No stage-localization finding. Do not treat a pre-grasped reset as already available. |

See `../results/existing_ablation_inventory.json` for the earlier 13-group WAM inventory. It did not include the π0.5 inversion cohort because its scope was WAM-only. Raw media availability must be checked separately from manifests. Historic endpoint measurements may depend on goal-triggered stopping; SGW-01 uses a common observation endpoint.

Source paths are stored in `protocol.json`. Do not rerun completed historical cells to make a new denominator.

## 3. Models

| ID | Configuration | Required identity |
| --- | --- | --- |
| N3 | Cosmos3 Nano Policy DROID, guidance 3 | `nvidia/Cosmos3-Nano-Policy-DROID`, revision `6706d7680581c255ff61e0f3bb49d90eac55c79e`; four denoising steps, shift 5, history length 1, conditioning FPS 15, resolution setting 480; joint-position interface; up to 32 actions executed per request. |
| D1 | DreamZero DROID, official conditional-action path | Checkpoint revision `96ad344138c66e82536422432ad742f015784942`; video guidance 5; 16 configured inference steps with actual evaluated/cache steps recorded; execute eight returned actions per replan; effective noise 1140. Resolve the official asset identifier from the pinned provenance before download. |

For D1, do not silently substitute the custom s2 action-guidance patch. A patched scale-1 path is acceptable only if identical-input actions and generated latents match the official conditional path. If not, label it as another configuration and do not run it in D1 cells.

Record model code commit, checkpoint hash, simulator/bridge commits, image digest, dependency versions, inference settings, returned/executed horizons, and effective randomness. These remain constant within each model's confirmation cohort. No cross-model ranking or causal claim about the benefit of world modeling is planned.

## 4. Relations and exact instruction contract

Let c be cube center, b bowl center, and p plate center in world coordinates; robot-left is +y and world-up is +z. Always move the cube. Both reference objects remain fixed task references, even when named first in a relational clause.

| Family | Positive goal | Negative goal | Signed relation r, in metres |
| --- | --- | --- | --- |
| LAT | cube left of bowl | cube right of bowl | c_y − b_y |
| HEIGHT | cube higher than bowl | cube lower than bowl | c_z − b_z |
| DIST | cube closer to bowl than plate | cube farther from bowl than plate | ||c−p||₂ − ||c−b||₂ |

HEIGHT tests relative height, not image-up/down, direct overhead stacking, or literal velocity commands. DIST tests relative distances to two visible anchors, not change from an unstated initial distance. These choices make the goal statically measurable and permit genuinely equivalent descriptions. Literal “move up/down” and “closer than before” are separate future tasks, not claimed as covered.

Three forms are crossed with both physical goals:

- **D, direct:** “Put the Rubik's cube to the left of the bowl.”
- **C, syntax-matched clause:** “Place the Rubik's cube so that the Rubik's cube is to the left of the bowl.”
- **I, inverted clause:** “Place the Rubik's cube so that the bowl is to the right of the Rubik's cube.”

`prompts.json` contains all 18 exact strings, physical-goal signs, and SHA-256 digests. Use those bytes, not generated paraphrases. For DIST, the equivalent inverted comparison is “the plate is farther from the Rubik's cube than the bowl is.” Swapping the arguments of distance alone does not reverse near/far: distance is symmetric.

The six conditions in a block share a physical reset and an effective policy draw where supported. Prompts stay static throughout an episode. No oracle action, progress-conditioned prompt, language coaching, or simulator geometry enters the model beyond its declared observation/proprioception interface.

## 5. Fixtures and qualification

Use DROID/RoboLab for every new core trial. Create new SGW-01 task definitions; do not alter the historical predicates. A fixture agent must produce immutable `fixtures/<family>/<layout_id>.json` files with actual asset hashes, all poses/orientations, support surfaces, cameras, collision geometry, reset procedure, and acceptance receipts before any model run in that fixture. Their SHA-256 values replace null pose hashes in a separately generated released queue; never edit the planned queue to pretend a layout was verified.

The following are design requirements, not claims that these scenes already exist:

1. **Neutral start:** |r(0)| ≤ 5 mm. Neither goal is already satisfied. The cube begins supported and ungrasped in all families.
2. **Goal margin:** require signed requested relation q·r ≥ 30 mm, where q=+1 or −1. Score and record the cube and every reference separately. If any reference center moves more than 5 mm from its initial position at any recorded step, fixed-endpoint success is S=0 (anchor-disturbance failure); retain the trial and its measured margins.
3. **Physical placement:** cube must have been picked up (≥30 mm above its initial center for three consecutive recorded control steps), detached from the gripper, and supported on a valid surface. In the final 0.5 simulated seconds, relation and release must hold, cube speed <0.02 m/s and angular speed <0.2 rad/s. A pure scene/reset technical failure remains separate from a valid manipulation failure.
4. **LAT:** use clear supported landing areas on both sides. Keep robot/cameras fixed, vary reachable cube/reference poses, and balance starting approach side across layouts.
5. **HEIGHT:** construct reachable lower and upper landing surfaces around a reference at intermediate height, with the cube initially at the reference's center height. Counterbalance which lateral side contains the upper surface (12 of each in confirmation). Both final height goals must permit released, supported placements. Do not ask for below-table placement or create a floating reference. Use explicit higher/lower wording to avoid an overhead interpretation.
6. **DIST:** bowl and visually distinct plate are both present and reachable placements exist nearer each. Start cube centers on the anchors' perpendicular bisector within the neutral tolerance. Counterbalance bowl/plate left-right arrangement (12 of each in confirmation). Prevent color, side, or object inventory from perfectly predicting the requested goal.
7. **Model-blind feasibility:** a scripted controller must execute both physical goals for each candidate, with three deterministic reset checks per goal, under the same 450-action cap. Require the predicate above and stable reset poses within 3 mm / 2 degrees. These are fixture receipts, not learned-policy evidence. Reject infeasible candidates before model outputs are inspected.
8. **Candidate selection:** freeze a seeded candidate generator, asset choices, reachable workspace bounds from the actual robot, and a maximum 100 candidates/family. Allocate qualifying candidates deterministically in hash order to P01, D01–D04, C01–C24, subject to the declared counterbalance. For HEIGHT (upper-support side) and DIST (bowl side), use one pilot layout with side fixed by a seeded coin, two layouts per side in development, and 12 per side in confirmation. Select the earliest qualified candidate in each required stratum; do not choose from outcomes. Preserve every candidate and rejection reason, including exhausted stratum counts. Reject any candidate duplicating a historical layout or another candidate within the reset tolerance. No model-performance-based selection. If 29 cannot qualify, mark the family blocked; do not fill with repeated states.
9. **Validation:** verify all six prompt-to-goal mappings against synthetic scene coordinates, reset determinism, deliberate wrong-side/anchor-motion/non-release examples, and goal-independent stopping. A family can proceed once its own checks pass; a HEIGHT block does not prevent LAT or DIST from running.

The exact workspace bounds, camera/time maps, and asset identities are measured implementation outputs. Inventing numeric poses without checking robot geometry would make the spec less executable, not more precise. Agents may resolve these engineering details under the above rules, record them, and continue without asking after every step. Changing the scientific comparisons, thresholds, models, or sample count requires a disclosed revision before new confirmation data are viewed.

## 6. Finite run matrix and ordering

| Stage | New layouts / family | Families | Models | Goals × forms | Episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| P: recording pilot | 1 | 3 | 2 | 2 × 3 | 36 |
| D: development | 4 | 3 | 2 | 2 × 3 | 144 |
| C: confirmation | 24 | 3 | 2 | 2 × 3 | 864 |
| Maximum core | 29 | 3 | 2 | 6 | **1,044** |

Each family alone is 348 episodes; confirmation is 288/family. A model-family branch is 174 episodes. There are 174 six-episode blocks in total. P and D never enter confirmation estimates. Candidate fixtures are shared across models; the independent sampling unit is the layout, not a frame, action request, or repeated condition. Twenty-four layouts/family is a bounded first study designed to estimate large effects, not a power claim for small equivalence margins.

`planned_cells.csv` freezes identifiers, prompts, seeds, and within-block order. Confirmation uses four six-block Latin cycles so each condition occupies each execution position four times per family/model. Both models use the same order for a layout. The queue environment seed initializes reset/simulator randomness after loading the separately qualified fixture; record the fixture-generation seed in its fixture manifest. It does not replace qualification or force a candidate pose. Check proposed seeds against existing run ledgers and runtime integer limits before release; DreamZero block labels are never described as independent noise draws.

Run order: qualify LAT and start its pilot/development; qualify DIST and HEIGHT independently; within a qualified model, finish intact six-cell blocks before changing family. P→D→C advancement depends on recorder/scorer/runtime correctness, not favorable success rates. Once development is frozen and all required receipts pass, agents may consume the entire finite released confirmation queue without per-block permission. No automatic sample expansion, rerun of valid failures, extra model, or training job.

**Execution:** 450 controller actions per episode with goal-triggered termination disabled. Log every success event but continue observing and controlling to action 450. Keep within-model control timing constant. Report simulated duration; 450 actions need not equal the same duration across models. A simulator safety termination is an observed censored model outcome, not a retry. Preserve full traces; never carry a censored endpoint forward to action 450.

Nonbehavioral recording checks: six fixed-input requests/model (same-input D+ twice and D− once, each with and without additional output decoding), giving **12 requests**. If decoding is mandatory, re-decode retained latents offline instead of changing inference. Add three official-reference requests only if qualifying a patched D1 scale-1 path. Record nondeterminism rather than pretending seeds control it. These calls are separate from 1,044 episodes. Scripted fixture checks are also counted separately.

## 7. Recordings and prediction measurements

Every episode must retain on persistent storage: exact prompt/hash; initial observation and simulator-state hash; all model inputs and camera names; returned and executed actions; request IDs; model/cache reset receipts; original decoded future frames and retained latent references when available; synchronized actual camera frames; per-control-step cube, bowl, plate, gripper and support states; contacts/release; success events; termination reason; timing; runtime identity; and a viewport video. Original frames and timestamps, not presentation composites, are the scientific data.

For every prediction frame record its intended physical target time and mapping to the associated action prefix. Validate the mapping using the released interface and recorder; frame number is not action number. Do not lengthen the executed chunk to manufacture forecast ground truth. Decode availability and physical alignment are separate checks. Unexecuted future portions can illustrate generated content but cannot receive a prediction-fidelity score.

Select two requests per episode at normalized request-index positions 0.25 and 0.75 (floor of fraction×(n−1), deduplicate for short episodes). Selection uses metadata only, includes failures, and never replaces an ambiguous request with an easier one. Preserve first requests additionally for matched-input comparisons. Later requests see diverged states; do not describe their differences as a prompt-only intervention.

Within each model choose H on development data as the largest documented exposed positive forecast time inside its unchanged executed prefix; freeze it before confirmation. Use the same H across forms and families for that model. If some family cannot be measured at H, record the limitation rather than change H after inspecting outcomes. Two blinded annotators localize target/reference centers in current, predicted and actual images; adjudicate disagreements >0.02 image diagonals. At most 1,728 selected confirmation requests and 10,368 initial frame judgments (two raters × three images), before adjudication. This is an annotation budget, not new inference.

**Primary prediction-fidelity measure:** for each identifiable object pair, relative 2D position error at matched H, normalized by image diagonal, compared with persistence. Positive skill = actual change from current minus prediction error. Keep families/models separate; score target and reference positions separately as a check against spurious reference motion. For DIST also track the plate. Persistence is the required equal-input baseline; constant velocity is secondary and uses extra history relative to Nano's history-1 input.

**Semantic forecast labels:** score whether the decoded future satisfies the requested relation only when its camera/view and visible supports make that relation identifiable. Monocular pixel-up does not establish 3D height; image proximity does not establish 3D distance. Validate a family-specific decoder/annotation procedure on 40 balanced rendered reference states (20 per physical sign; ten per sign development, ten per sign held-out validation), including 10 ambiguous/occluded variants scored separately. Require ≥95% correct sign on the 20 unambiguous validation states and no forced labels on ambiguous cases. These are offline rendering checks. A failed semantic-label check blocks that semantic prediction claim, not the valid behavioral branch. Label `unknown` and report coverage by condition. No auxiliary VLM answer is treated as the policy's internal understanding.

Report prediction-consistent/execution-inconsistent, execution-consistent/prediction-inconsistent, both consistent, both inconsistent, and unobservable categories at matching times. An initially stationary grasp phase is not evidence of ignored direction merely because neither cube nor forecast has moved. Long-horizon goal satisfaction and short-horizon prediction accuracy remain different measurements.

## 8. Analyses fixed before confirmation

For layout l, model m, family f, form k and goal q, let r be the observed relation and M=q·r the requested-side margin. Compute terminal M only when action 450 is observed. Compute the all-trial binary outcome S using the final stable predicate; safety-terminated trials have S=0 and censored M. Technical-invalid attempts receive no model outcome.

**Primary execution endpoint:** per-layout success contrast

`delta_S(l) = mean_q [S(l,I,q) − S(l,C,q)]`.

Report mean delta_S with 95% layout-cluster bootstrap intervals (20,000 resamples; all six cells stay together). Six model×family primary comparisons use Holm correction for paired sign-flip tests (100,000 fixed-seed draws); report the test's exchangeability assumption and all unadjusted/adjusted values. Missing technical cells leave a block incomplete; do not silently drop them or declare a complete branch.

**Key continuous secondary:** `delta_M(l) = mean_q [M(l,I,q) − M(l,C,q)]`. Also show each physical goal separately, the C−D construction effect, and `r(k,+)−r(k,−)` directional separation in each form. Signed average separation alone can hide mixtures of correct and reversed trials: include paired distributions and positive/negative/tie counts (tie tolerance 1 mm). Analyse scene-level variability; do not bootstrap individual frames as independent observations.

**Censoring:** report complete-case M as conditional. Bound full-cohort mean contrasts using the frozen fixture workspace bounds on r and every censored/missing terminal M; if bounds do not determine the sign, withhold the full-cohort sign claim. Report safety events by condition. Do not reconstruct a fixed-time endpoint from an early success termination.

**Prediction analyses:** persistence skill and semantic-label/behavior disagreement rates; two requests average within episode, then directions/forms within layout. Compare observed coverage by form/family. For a binary category with unobservable predictions, report full-cohort lower/upper bounds treating unknown cases as zero/one. Continuous estimates remain explicitly conditional on valid localization. Differences at first requests permit matched-input statements; later differences are closed-loop associations.

**Additional analyses requiring no new runs:** first-success versus fixed-endpoint summaries; pickup/transport/placement/release timelines; reference-object displacement; stationary versus moving prefixes; per-goal success transitions; alternative supported forecast time as a sensitivity analysis. Movement threshold is 0.02 image diagonals over H, fixed from calibration. Thresholds must not be selected to maximize an effect.

Do not conclude equivalence from a nonsignificant effect. Any equivalence claim requires both prespecified criteria: the 90% paired CI for delta_S wholly inside ±0.10 and the 90% paired CI for delta_M wholly inside ±0.02 m, with censoring/missingness accounted for. This establishes equivalence only for those endpoints and margins, not identical trajectories or general semantic understanding. The sample size may be insufficient; report inconclusive honestly.

## 9. Interpretation and follow-up rules

| Observation | Supported statement | Still unresolved |
| --- | --- | --- |
| C is worse than D; I resembles C | Complex sentence construction affects performance | Particular parsing, exposure, or control mechanism |
| I is worse than C | Relational argument reversal adds a cost under matched construction | Internal semantic representation or exact neural cause |
| Predicted relation correct, execution wrong | Prediction–execution disagreement at the measured horizon | Whether video caused action; whether language was understood |
| Both prediction and execution wrong | Error is visible in both outputs | Perception versus semantics versus shared dynamics error |
| Correct transport, loss during release | Failure becomes visible at release | Language may still affect the release strategy |
| Different relations fail differently | Family-specific behavior on these fixtures | General cognitive hierarchy across relations |

A failure-stage label describes when an observed criterion failed. It does not by itself identify why. “The model ignored language” requires evidence beyond low task success or small action differences.

Optional extensions are **not queued**: a fresh matched π0.5 LAT behavioral baseline (174 episodes, specified in `WORKSHOP_FIT.md`); Cosmos Edge replication; negation as a fourth form; explicit robot/camera frame contrasts; anchor-name substitution; literal up/down or initial-distance commands; a validated stable-grasp intervention; custom DreamZero s2 guidance. The earlier failed stable-grasp construction cannot be reused without new validation. If a new hypothesis is proposed after seeing confirmation, develop it on existing data and use a separately registered new cohort for confirmation.

## 10. Paper and stop conditions

The research manuscript should contain: question and prior evidence; relation/prompt design; models and fixture validity; execution and future measurements; then measured results once available. Planned result tables must say “not run.” Keep π0.5 out of the abstract and main evidence table; preserve its historical numbers in background materials. The primary figures and claims concern WAM execution and aligned prediction reliability. Cite STAGE and LIBERO-CF; a general distinction between semantics and action is already studied. No “first” claim is currently established.

Prioritize two main results: (1) per-family D/C/I completion and directional margins; (2) aggregate same-horizon prediction–execution agreement/disagreement, persistence skill and unknown coverage, illustrated by aligned cases. Use the three-instruction schematic as setup and failure-stage/object trajectories as diagnostic detail. A short-horizon forecast is not scored as a prediction of eventual task success. Never choose the most dramatic clips without also giving the selection rule and aggregate denominator.

The finite study ends at 1,044 valid model episodes or a documented blocked/reduced branch. Valid poor performance is a result and never triggers retry or automatic extra training. Cluster agents stop for exhausted retry limits, provenance drift, unsafe fixture/renderer behavior, exhausted storage, or resource budget. They keep other independent qualified branches progressing where safe. See `CLUSTER_RUNBOOK.md` for exact restart and handoff behavior.
