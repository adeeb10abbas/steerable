# V4 experiments: spatial instructions under test-time scene movement

**Status: prospective design, not executed results.** The current repository supports the historical evidence audit; the event scheduler, fixtures, and V4 measurement pipeline still require implementation and qualification. `campaign.json` fixes the allocation. `runtime_lock.template.json` names the physical and software details agents must resolve before release. Do not run V4 cells through an unchanged V3 runner and label them V4.

## 1. Scientific purpose and scope

The existing experiments establish that static layout and wording can change relational placement behavior and binary success. They do not establish whether a policy continues to use the named spatial reference after execution begins. V4 extends the same task and measurement infrastructure to answer that question with unchanged weights and one unchanged instruction per episode.

The main intervention is an externally imposed movement of a scene object after a natural grasp. Compare it with an otherwise identical rollout in which that object remains stationary. The crucial language test repeats **the same prescribed physical motion and stopping rule** while changing which object the instruction names as its reference. This separates generic movement response from response that depends on the requested relation and referent. Neither a low success rate nor motion toward a displaced object alone identifies spatial grounding.

Three research questions govern the paper:

1. **Does online correction depend on the spatial goal and named reference?** Measure completion, distance to the current valid placement region, and same-motion reference selectivity. C1 and C2 are primary.
2. **How consistently does this behavior survive equivalent wording and focused changes of relation, object, and robot stack?** C1 tests matched direct/inverse descriptions; C5–C8 are prespecified scope replications. C3 characterizes response to continued movement.
3. **Does shortening the executed action prefix improve correction under the same controlled observation-to-action delay?** C4 changes execution cadence while keeping the checkpoint and predicted action horizon fixed.

This is an evaluation of **online correction under test-time scene changes**, with a compact static foundation. It is not a new training method, a general steerability benchmark, a test of memory or instruction switching, or a claim that three checkpoints identify an architectural cause. The controlled simulator clock also prevents a claim about native real-time deployment speed.

## 2. Keep the completed work useful

Read [the source audit](00_EXISTING_EVIDENCE_AND_SCOPE.md) before implementing anything. Preserve V2/V3 protocols, accepted outcomes, original scorers, and continuation records.

| Existing evidence | Role in the revised paper | What it cannot replace |
| --- | --- | --- |
| Matched original/reflected layouts for pi0.5, Nano, and DreamZero | Compact motivating figure showing why a static success difference requires a geometric interpretation | Contemporaneous V4 stationary controls, online response measurements, or a cross-model architecture claim |
| Continuous final positions and Nano's lateral sweep | Establish the value of continuous geometry alongside success; inform workspace diagnostics | A validated held-out prediction; the historical fitted crossing is outside the observed range |
| Canonical/inverse wording study | Motivate a clean carrier-matched wording test; retain the historical result with its wording confound | A claim that reference frame alone caused the difference |
| Eleven-checkpoint screen and compound symmetric-scene cohort | Appendix/context with provenance and limited conclusions | New independent causal replications |
| Existing server adapters, state/action/video logging, manifests, Kubernetes lanes | Reuse implementation where behavior and provenance can be verified | A working V4 scheduler, validated grasp-state restoration, or qualified new fixtures |

Do not rerun the historical screen. Do not pool old and new controls: V4 changes prompts, timing, seeds, and first-placement scoring. The new stationary cells measure the baseline for the actual V4 treatment comparison. Show historical and prospective results in separate table panels.

## 3. Fixed checkpoint selection

| Role | Checkpoint and stack | Why this selection | Required qualification |
| --- | --- | --- | --- |
| Main anchor | Cosmos3 Nano, existing DROID/RoboLab integration | Existing static geometry and language evidence; public artifact; action/future interface can support detailed traces | Restore exact existing checkpoint and integration hashes; log all exposed predicted actions and decodable futures |
| Main confirmation | Existing pi0.5 DROID `pi05_droid_jointpos_polaris` artifact | Strong historical placement and reflection evidence; a distinct existing policy interface | This is the existing nonpublic artifact. Confirm access and exact identity; do not present it as a reproducible public pi0.5 release |
| Focused second-stack replication | GR00T N1.7 Bridge / SimplerEnv / WidowX | Adds a policy/robot/simulator stack different from DROID without multiplying every experiment | Verify the actual released Bridge checkpoint, model card, adapter, native control interface, and objects before freezing C8 |

Nano and pi0.5 receive C1–C6; Nano also receives C7. C8 is a focused bridge, not a third full factorial. No new DreamZero runs are required; its existing evidence remains useful. No fine-tuning, checkpoint selection based on these results, or silent replacement is allowed. If pi0.5 access or C8 integration is unavailable, mark the affected allocation blocked and state the resulting limitation. Do not substitute the poorly performing historical GR00T DROID interface for Bridge.

This comparison cannot attribute differences uniquely to architecture: training data, weights, action interface, and, in C8, embodiment and simulator also differ. Describe the actual systems rather than claiming a WAM-versus-VLA ranking.

## 4. Relation inventory and exact language

The six directional relations follow a conventional directional taxonomy. Containment adds one physical task with two equivalent argument orderings. EmbodimentSemantic's eight directed labels include `on_top` and `contains`; they are not eight independent terminal placement tasks. Here **above is different from support/contact**, and **contains is the inverse description of inside**.

| Physical goal | Direct clause | Equivalent inverse clause |
| --- | --- | --- |
| Left | the cube is left of the bowl | the bowl is right of the cube |
| Right | the cube is right of the bowl | the bowl is left of the cube |
| Front | the cube is in front of the bowl | the bowl is behind the cube |
| Behind | the cube is behind the bowl | the bowl is in front of the cube |
| Above | the cube is above the bowl | the bowl is below the cube |
| Below | the cube is below the bowl | the bowl is above the cube |
| Inside | the cube is inside the bowl | the bowl contains the cube |

Use exactly `Place the {object} so that {clause}.` The manipulated object remains the imperative's object in both forms. For every horizontal prompt append exactly: ` Use the robot's fixed viewpoint for left, right, front, and behind.` Resolve object names once in the fixture lock. C2 uses, for example, `the blue bowl` and `the yellow bowl`, with physical identities and positions counterbalanced. C7 substitutes sponge/tray. C8 substitutes the verified Bridge object names. Store the complete submitted UTF-8 prompt and its SHA-256; never reconstruct it from a condition label after the run.

“Inverse” reverses the relation arguments while preserving the same physical goal and reference frame. It is not a viewpoint change. `inside`/`contains` additionally changes predicate morphology; acknowledge that limitation. No negations, multiple carriers, new instructions after the event, “keep” task, or reference-frame sweep is added to this campaign.

The simulator's task-frame basis is frozen independently of camera perspective. Define front as the side toward the robot in the fixed robot-relative frame, behind away, left/right from that same viewpoint, and up opposite gravity. Render labeled axes and all eight submitted description forms for human review before inference. If the existing simulator convention differs, convert coordinates explicitly; do not change a sign until a plot looks correct.

## 5. Allocation: 17,664 new policy episodes

A block is an independent registered initial scene/reset seed, with matched conditions nested within it. More branches do not create more independent blocks. All blocks are fresh relative to historical evidence. C3 and C4 intentionally use the first 64 horizontal blocks from C1; their listed stationary/move-stop controls refer to existing episode IDs.

| ID | Question / fixture | New conditions per policy and block | Policies | Blocks | New episodes |
| --- | --- | --- | ---: | ---: | ---: |
| C1 | Horizontal goals, movement, wording | 4 goals × 2 wordings × 3 scenarios | 2 | 128 | 6,144 |
| C2 | Same motion, different named reference | 4 goals × 2 named references × sham/move-A | 2 | 128 | 4,096 |
| C3 | Continued movement | 4 goals × slow drift / fast drift / reversal; direct wording | 2 | 64 | 1,536 |
| C4 | Shorter executed prefix | 4 goals × sham / move-stop / fast drift / reversal; fast schedule, direct wording | 2 | 64 | 2,048 |
| C5 | Supported vertical relations | above/below × 2 wordings × 3 scenarios | 2 | 64 | 1,536 |
| C6 | Containment task | inside/contains descriptions × 3 scenarios | 2 | 64 | 768 |
| C7 | Object-pair bridge | 4 goals × 3 scenarios; sponge/tray, direct wording | 1 | 64 | 768 |
| C8 | Second-stack bridge | 4 goals × 3 scenarios; Bridge/WidowX, direct wording | 1 | 64 | 768 |
| **Total** | | | | | **17,664** |

Three scenarios means `original_sham`, `move_stop`, and `destination_static`. The 240 excluded engineering episodes are additional: 10 policy/fixture groups × (16 stationary + 8 movement). Model-blind geometry checks and privileged-controller trajectories are also additional and must be counted separately. Infrastructure attempts are not new scientific replications and do not replace valid failures.

The fixed n supports substantial effects and interpretable uncertainty, not guaranteed detection of small effects. At n=128, a worst-case independent binary rate has an approximate 95% half-width of 8.7 percentage points; a paired contrast can be wider, and a difference of paired differences wider still. See the explicit limits in the analysis specification. Do not resize after seeing confirmatory results or claim equivalence from a nonsignificant effect.

## 6. Common scene, intervention, and completion contract

### 6.1 Supported goal regions

Define a valid placement as a region, not a single desired offset. For horizontal relations, intersect the requested directional halfspace with the allowed supported workspace and object-size clearance. Orthogonal bounds come from the declared task surface and collision constraints, not an arbitrary narrow target strip. For example, left requires the manipulated object's appropriate projected extent to lie beyond the reference's corresponding extent plus the frozen clearance. Freeze the exact extent convention (oriented geometry versus conservative bounds), clearance, support erosion, and task-frame transform.

For above/below use the declared top/bottom supported shelf areas and a documented horizontal-overlap predicate. For containment use a true 3D admissible interior eroded by manipulated-object dimensions, including wall clearance, orientation, depth, and support. A centroid inside a bowl's bounding box is insufficient.

There may be many correct placements after the reference moves. A policy need not copy its displacement. A movement that leaves the planned placement valid need not cause any correction. The scorer must preserve that distinction. Do not condition the primary sample on the eventual sham endpoint becoming invalid.

### 6.2 Natural grasp, not reconstructed carry states

Run the ordinary policy from the registered reset. Establish a natural carry checkpoint using the frozen lift/coupling detector: initial defaults are at least 0.04 m lift above the initial supported height, 0.20 s sustained carry, and at most 0.01 m object-to-gripper relative drift over that dwell. Contact/gripper state strengthens the detector where available; when contact is not exposed, label it a kinematic proxy and verify it against pilot video. An object merely perched on a support or passively pushed upward is not a verified carry.

The trigger accesses simulator state only inside the experiment scheduler. It never sends state, a target, or a new instruction to the policy. Keep the prompt and policy inputs unchanged. The old privileged IK and grasp-state experiments did not validate a reliable alternate initialization; do not build the main study on them.

Trigger eligibility must occur by 40 simulated seconds. Event onset is the first scheduled standard-query boundary after eligibility, plus the block's phase fraction of the achieved **standard** query period. Quantize upward to native control ticks. Use fractions 0, 0.25, 0.50, 0.75 in a balanced block assignment. C4 uses the same standard-clock event anchor and phase even after switching to the fast query period. Record the actual quantized delay, phase, and natural-grasp time. If placement ends before the event, retain that outcome with `event_delivered=false`.

### 6.3 Clock and action queue

The requested emulated observation-to-action delay is 0.10 s, the standard query period 0.50 s, and the fast period 0.25 s. Quantize upward to native ticks and freeze each policy's achieved values. Capture an observation at virtual time t; pause physics while the server computes; advance physics over the emulated delay using the old action queue and evolving reference trajectory; deliver the response computed from observation t only at t+delay. A slow GPU increases wall-clock cost, not the simulated delay.

Predicted action horizon, output normalization, observation-history construction, and native action interpolation remain fixed. A change of query cadence changes how many predicted actions are executed before replacement, not the model's predicted sequence length. The queue must cover the largest registered interval plus delay. Initialization and safe-hold behavior are specified and qualified; queue underflow during ordinary operation is an infrastructure defect, not a policy failure. Do not skip simulator dynamics during the delay or give the response a later observation.

### 6.4 First-placement endpoint and time limits

This is a **first-placement attempt** evaluation. After verified carry, the first detachment lasting two native control ticks ends the policy phase, whether the placement is correct or incorrect. Record detachment onset at its first tick and confirmation at the second; stop imposed reference motion at the **confirmation** tick, without rewinding physics. Stop new policy requests and new predicted-action application, hold the last verified safe robot target, and simulate a fixed 1.0 s passive settling interval. Score the settled object against the reference frozen at confirmation. Verify support and stability over that interval using locked velocity/pose tolerances.

Premature drops, wrong-side releases, and unstable first placements remain failures; do not resume for rescue or retry until success. An accidentally lost grasp also counts as a first-attempt failure. A successful early placement stays successful; moving the reference afterward would turn “place” into a different task.

The active policy phase ends at `min(60 s from reset, event onset + 20 s)` when an event is delivered, or at 60 s when no event is delivered. At timeout, freeze imposed movement and terminate policy updates. A fixed 1 s adjudication interval is outside that active cap, so an archive can span up to 61 simulated seconds. A still-held object cannot become a success merely because its planar position is correct. Record both the active endpoint and settled adjudication state; use the analysis document's terminal-state rule consistently.

## 7. Exact movement profiles and balanced assignments

Let D be the frozen fixture displacement magnitude, e a unit vector in the fixed task plane, and `S(u)=10u^3-15u^4+6u^5`, clipped to [0,1]. Move the reference kinematically along `p_ref(t)=p_ref(0)+e*D*S(t/T)` for a single segment; rotation is fixed. Continue normal physics for all other objects. The external intervention does work on the reference and is not a claim about an autonomous moving object's natural dynamics.

| Profile | Prescribed translation from event onset | Nominal duration | Interpretation |
| --- | --- | ---: | --- |
| `original_sham` | Same scheduler calls, identity pose update, no translation | matched event window | Contemporaneous control, including scheduler overhead |
| `move_stop` / C2 `move_A` | 0 to +D with minimum-jerk interpolation, then stationary | 0.50 s | Primary finite perturbation |
| `destination_static` | Reference at the planned +D pose from reset; no online motion | whole episode | Physical competence at the planned destination, not an identical-prefix counterfactual |
| `slow_drift` | 0 to +D, then stationary | 4.0 s | Longer exposure to a moving reference |
| `fast_drift` | 0 to +D, then stationary | 1.0 s | Same displacement with a different speed-duration profile |
| `reversal` | 0 to +D over 2.0 s; +D to −0.5D over the next 2.0 s; then stationary | 4.0 s | Direction change with a non-original endpoint |

For each segment use its own smooth minimum-jerk interpolation. Actual motion stops at first-placement confirmation or the active timeout. Archive planned and measured reference poses, velocity, acceleration, path completion fraction, collision/contact events, and truncation reason. Destination-static always means the **planned move-stop destination**; it is not necessarily the attained terminal layout if the moving episode ends early.

Nominal D is 0.12 m for the DROID horizontal, two-reference, shelf, and object-pair fixtures; 0.08 m for containment and Bridge. Before any policy inference, choose the largest feasible scale from {2, 1.5, 1, 0.75, 0.5} under the model-blind geometry/controller checks below. Use one scale per fixture shared by both main policies, all wordings, and all assigned goals. Never choose a different scale because one model is less successful. Lock actual D, paths, speeds, and accelerations. If even half scale is physically invalid, block that fixture rather than silently making an easier test. Nominal values define one displacement unit, not the final selected amplitude; the ladder can choose 24 cm for a 12 cm fixture only if all geometric, swept-path, and controller checks pass. For the two primary fixtures, require at least 20% of the initial legal planar goal area to be removed in each registered shrinking-direction goal/reference case; report the goal-area overlap for every reset, goal, and motion sign. Expanding-direction cases can retain all old valid placements and have no removal requirement. If physical feasibility and this information gate cannot both hold at any candidate scale, block that primary fixture rather than hiding a narrow artificial target strip. No policy endpoint is used by this gate. Nominal values are design defaults, not a claim that the existing simulator has already passed them.

For C1, C3, C4, C7, C8, movement is along the requested relation's axis, with positive/negative physical direction balanced independently of the requested side across blocks. C5 moves the middle reference horizontally along the fixed left-right axis. C6 also uses balanced horizontal translation. C2 uses a diagonal `(±u_left ±u_front)/sqrt(2)` so the same physical trajectory is meaningful for both horizontal axes. Do not change the moved object's trajectory based on which reference is named.

Use a 16-block counterbalance cycle: `phase = block % 4`; `counterbalance = floor(block/4) % 4`. For C2, diagonals cycle (++,+−,−+,−−); color-to-ID mapping alternates and starting side alternates across these four counterbalance states. Each color, location, and diagonal is balanced marginally, but their interactions are not separately identifiable in this compact design. All named-reference and goal conditions occur within each block. For other fixtures, physical sign is positive for counterbalance 0/2 and negative for 1/3. Assign reset jitter independently using the registered environment seed. Do not use the same modulus bit for every nuisance variable and call them independent.

The finite profiles characterize sampled demands. Their durations and speeds covary at fixed displacement; results are effects of the registered **motion profile**, not an isolated causal speed law or a continuous maximum tracking speed. Reversal is secondary and analyzed by segment, not net displacement only.

## 8. Family-specific execution and inference

### C1 — Horizontal relation, layout, and wording

Use cube/bowl placement under left/right/front/behind, direct/inverse descriptions, and three scenarios. Share the reset and sampling seed within a block. Same-prompt original-sham and move-stop runs have an identical treatment-inactive prefix; destination-static starts from a different reference location and naturally may have a different prefix. The latter diagnoses static destination difficulty and cannot isolate only online adaptation.

Primary comparisons are moved versus sham completion and geometric violation, with the wording-by-movement interaction and its individual components. Report all four goals before balanced averaging. Good static performance followed by a movement cost is more informative than low performance in both. Low baseline competence remains a valid result and limits the eligible online conclusion; do not exclude that checkpoint after seeing it.

### C2 — Named reference changes the meaning of identical motion

Place two identically shaped, visibly distinct bowls A and B in a fixed, collision-free layout. Freeze their color/identity mapping in each block. The physical object A moves diagonally in the move-A condition; B stays stationary. Prescribe the exact same trajectory when the prompt names A and when it names B. First-placement termination can truncate realized motion differently; report the actual exposure and restrict identical-stimulus language to the verified common active interval. Both prompt conditions also have their own sham. The manipulated object remains the cube.

For each prompt, estimate improvement toward its current legal goal region relative to its same-prompt sham at planned event onset + 2.0 s, retaining early terminal states under the specified analysis extension. The primary anchor does not require a post-treatment changed observation; observation-anchored response timing is secondary. The primary selectivity contrast is that improvement with A named minus improvement with B named. Compute both terms against their appropriate current goal sets. Report each component: a positive difference alone could arise because movement harms the B-named condition. Also show world-coordinate motion, final success, and actual geometry. The moving object must not directly push the cube or robot in the planned path; observed policy-induced contacts are retained and identified as an alternative mechanism.

Do not transplant one prompt's post-grasp policy state into another prompt. Each instruction produces its own natural history. The motion/sham pair matches within prompt; the cross-prompt contrast pairs by reset block. The main eligible selectivity estimate uses the prespecified matched-eligible intersection, with all-assigned completion and eligibility counts alongside it. It is not a population-wide causal effect of language on all resets.

### C3 — Continued movement, with reused controls

Use the first 64 C1 blocks and direct wording. Add slow drift, fast drift, and reversal only. Reuse those blocks' C1 direct original-sham and move-stop episode IDs. Report event-aligned trajectories, helpful correction, missed or unnecessary responses, and first-placement outcomes. Do not require the relation during transport. Do not penalize completion before the planned motion finishes. Show the proportion of each profile actually exposed before placement, so waiting and early completion are visible.

### C4 — Execution cadence, with a sham at each cadence

At the natural carry checkpoint, switch the fast branch from the common standard schedule to the frozen fast schedule. Standard counterparts come from C1/C3; run only the four listed new fast-schedule scenarios. Use the same natural checkpoint, standard-clock event anchor, and physical motion across schedules. Compare each schedule's motion-versus-sham cost before comparing schedules. A faster schedule may change unperturbed behavior, so raw fast-versus-standard moved success alone is insufficient.

Keep delay, model horizon, observation/history interface, and control discretization fixed. The number and timing of observation updates naturally changes; this is the deployment intervention being tested. Measure actual compute cost separately. If the adapter cannot execute the fast schedule while preserving these semantics, block the C4 family and document the technical reason (the current release schema releases whole families, including all their assigned policies); do not replace it with a different model configuration.

### C5 — Supported above/below

Use three broad **stationary** shelves. The bowl sits on the middle shelf and translates horizontally; the cube's valid above/below placement lies on the top/bottom support, with the frozen horizontal overlap rule. Both goals must be reachable without traversing a shelf, and the bowl must not push a support or the carried object. Direct and inverse wording describe the same goal.

This expands relation semantics while preserving gravity and support. It does **not** test vertically moving targets or prove vertical tracking. Do not vertically reflect an unsupported tabletop scene or move the support carrying the manipulated object. If the shelf fixture is not physically feasible, mark C5 blocked and retain the narrower empirical scope.

### C6 — Inside / contains

Use one movable container with verified interior clearance and a manipulable object. Direct and inverse descriptions refer to the same insertion/placement goal. Translate the container during the natural carry phase, using sham and planned-destination static controls. Score stable full containment and release separately from horizontal approach geometry. End imposed motion at first release confirmation, so a later-moving container cannot revoke an already completed placement. This is a task bridge, not a new suite of outside/cover/contact relations.

### C7 — Object pair

Replace cube/bowl jointly with sponge/tray on the same robot stack, using Nano, all four horizontal goals, direct wording, and the three scenarios. Freeze physical dimensions, collision properties, visual assets, and pose distributions. Changes in both manipulated and reference objects are a **joint object-pair replication**, not an isolated appearance or object-role effect.

### C8 — Second stack

Use one qualified GR00T Bridge/WidowX checkpoint in its supported simulator/task integration. Resolve the manipulated/reference object names and legal workspace before policy pilots. Reproduce the horizontal three-scenario design with direct wording. Use an independently calibrated task frame and scoring geometry, with the same semantic and timing definitions where technically achievable. Report achieved clock quantization and fixture differences.

This is a replication across the combined policy/robot/simulator stack. Never pool its raw rates with DROID or interpret the difference as embodiment alone. A blocked Bridge release is reported as missing scope, not replaced by historical RoboTwin results that did not establish the same effect.

## 9. Model-blind preparation and engineering pilots

### 9.1 Deterministic preparation before policy inference

1. Export the existing base fixtures and their exact geometry; create the two-reference, shelf, containment, object-pair, and Bridge definitions. Resolve every path, object name, unit, and transform in the runtime lock.
2. Generate all registered resets from the reserved seed range. Validate support, initial contacts, joint limits, camera visibility, nonempty legal goal regions, and swept intervention collision clearance for **every** reset and all prescribed paths. Sample smooth paths at no more than 0.02 s intervals and use swept-volume bounds or finer adaptive checks wherever a thin obstacle could be missed. Pure geometric validity is not policy competence.
3. For each fixture, evaluate displacement scales in the fixed descending order. Use the largest scale that is jointly feasible for all its goals, directions, and reset definitions. Freeze it before any policy outcome is observed.
4. Validate a scripted/privileged controller separately on the canonical reset and eight preselected extreme reset states. It must actually grasp, transport, release, and stably place for each physical goal at original, midpoint, and endpoint reference positions. C2 checks each relation with **both** A and B named. For six fixtures with 4+8+2+1+4+4 goal/reference cases, this is **621 stationary scripted validation trajectories** (9 × 23 × 3) per final geometry candidate. Attempts on rejected candidates remain in the engineering log. Add one moving-reference trajectory per goal/reference case at the canonical reset (**23 movement checks**), for **644 controller checks** per final geometry candidate. These are model-blind feasibility controls, excluded from the policy episode total and from model performance tables.
5. A failing scripted controller does not prove task impossibility. Diagnose IK/controller implementation versus geometric infeasibility; preserve failed receipts. Release only after a valid controller establishes the tested setup, or explicitly block that fixture. The previous E002 controller failed to pick up and is not a feasibility certificate.
6. Independently verify scorer predicates against rendered valid/invalid placements, boundaries, unsupported hovering, wrong reference identity, partial containment, empty goal sets, early release, and lost logs. Freeze code, test fixtures, tolerances, and D_cap (the registered workspace diagonal).

The nine scripted reset cases are chosen from geometric extrema of the registered distribution before policy inference, not from policy failures. Their counterbalance assignments must cover all four C2 diagonal states and both color/position mappings; the exhaustive geometric sweep still checks every registered reset and path. Feasibility checks do not imply every policy trajectory can succeed. The controller's privileged state/target must never enter the tested policy API.

### 9.2 Fixed engineering policy pilots: 240 excluded episodes

Ten groups exist: Nano/pi0.5 on horizontal, reference-binding, vertical, and containment fixtures (eight groups), Nano on object-pair, and GR00T on the second stack. Each receives 16 stationary and 8 movement pilot episodes from a disjoint pilot seed pool. Round-robin the fixture's valid goals, wordings, and naming conditions; freeze the pilot manifest before inference. For horizontal groups include both query schedules and all motion profiles in the eight movement checks as applicable. Repeated engineering repair attempts are logged separately, not added as successful replicates.

These pilots establish correct serialization, real action application, valid images, clock/queue semantics, natural-trigger detection, release/scoring behavior, independent server reset, and complete video/state/action storage. Check a sham-versus-sham replay pair and a movement pair per policy/fixture group. No positive language effect, minimum task-success percentage, or observed correction is required to pass a **technical** gate. A policy that produces identical actions under different valid prompts is a possible scientific result. Failure to submit the prompts or apply returned actions is an implementation failure.

Do not tune prompts, displacement, timing, or sample size to pilot behavioral success. If a genuine implementation defect requires a repair, preserve the failed pilot and version the repaired code. If it changes a scientific condition, publish a pre-confirmatory amendment and regenerate the frozen inventory before release. Never mix pre-amendment and post-amendment cells as if they share a protocol.

## 10. Pairing, randomization, and one-campaign execution

Prefer full natural rollouts with a treatment-inactive prefix, matched reset and policy sampling seeds, and recorded prefix hashes. Full branch restoration is an optimization only after verifying simulator state, policy history/cache, RNG, action queues, controllers, clocks, and observation IDs. A simulator-only save state is not a policy checkpoint.

If exact replay is unavailable because the released runtime remains stochastic, a prospectively declared independent-natural-rollout fallback is allowed for C1 completion/geometric contrasts and secondary outcome summaries that do not assume a common prefix. C2 primary H selectivity requires verified same-prompt common prefixes and remains blocked without them. Record the fallback and randomized condition order in the runtime lock and use the corresponding paired-reset analysis. Preserve observed pre-event differences; do not use exact-prefix causal language or sham-subtraction latency detectors that assume identical pre-event trajectories. No fallback can silently change the estimand after results are inspected.

Environment seeds are reserved as `2100000000 + 10000 * fixture_slot + block_index`; pilot seeds use a separate base. Sampling seeds are a deterministic hash of the campaign namespace, policy, fixture, and block, excluding treatment, wording, and schedule to preserve common random numbers where possible. Validate the reservation against historical seed registries. Common sampling seeds do not imply identical post-prompt histories or exchangeable policy architectures.

Generate the complete manifest once, commit its hash, and schedule a deterministic randomized order within each isolated policy/fixture lane. Keep all branches of a block on a compatible software/hardware class; balance treatment order and retain GPU/driver/runtime metadata. Cluster availability changes wall time only. Leases and result writes are atomic, and reruns are restricted to documented infrastructure-invalid attempts. A valid no-grasp, no-event, no-response, wrong relation, or drop is final evidence.

The execution DAG is preparation → runtime qualification → excluded pilots → immutable release → independent family/block shards → completeness audit → frozen analysis → paper exports. Gates auto-advance when receipts pass; they do not require repeated user approval. Blocked secondary fixtures should not delay qualified primary families. C3/C4 reuse dependencies must be satisfied before their analysis closes. All agents use the same allocation and runtime lock, never independently inventing a “helpful” variant.

## 11. Required deliverables and stopping rule

For each released cell, preserve videos, state/action/timing traces, actual prompt, complete configuration, seed/prefix identifiers, intervention traces, all terminal predicates, raw exposed futures, and infrastructure-attempt history. Retain valid failures. The runbook defines storage and the analysis document defines schemas, estimands, and uncertainty.

Close the campaign when every planned episode is either accepted with complete required evidence or explicitly blocked/infra-unresolved with a reason; a scientific “complete” claim requires all declared primary cells. Run analysis from that locked inventory without selecting favorable conditions. Produce the fixed tables/figures in [04_PAPER_AND_RELEASE.md](04_PAPER_AND_RELEASE.md), report missing scope, and move to writing. New experiments require a separately versioned follow-up question, not an automatic search for a stronger result.

## 12. Primary literature and what it contributes

- [EmbodimentSemantic, arXiv:2607.00020](https://arxiv.org/html/2607.00020v1): directed spatial predicates, scene/language perturbation context, and limitations on interpreting action-level results without controlled comparisons. It already includes closed-loop VLA evaluation; do not claim otherwise. V4 changes terminal relational goals and isolates same-motion reference selectivity with policy rollouts.
- [Zampogiannis et al., qualitative spatial relations, ICRA 2015](https://kzampog.github.io/documents/ICRA2015_spatial_relations.pdf): basis for six directional relations, containment, and distinguishing support/contact from purely directional relations.
- [DynamicVLA](https://arxiv.org/html/2601.22153v1) and [DOMINO](https://arxiv.org/abs/2603.15620): moving-object manipulation and relational tasks already exist. V4 does not claim novelty from movement or left/right placement alone.
- [FASTER](https://arxiv.org/html/2603.19199v1) and [real-time chunking](https://arxiv.org/abs/2506.07339): motivate separating inference delay, query interval, and action execution. V4's controlled clock studies a specified deployment intervention, not an implementation of either method or native real-time performance.
- [A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation, TRI](https://arxiv.org/html/2507.05331v1): use explicit questions, concrete tasks, fixed checkpoint/configuration descriptions, task predicates, and matched comparison tables as a presentation guide. Cite evidence at the level actually tested.

The bounded literature review motivates this design; it is not proof of a literature-wide “first.” The final novelty claim must remain specific to the controlled semantic and behavioral comparisons supported by the completed data.
