# V4 measurement and analysis specification

Status: prospective specification. No V4 findings are asserted here. Read the experiment matrix and execution runbook alongside this document. The frozen machine-readable manifest must contain every constant named below before confirmatory inference starts. A remaining `TODO`, `null` required constant, or scorer mismatch blocks that cell family rather than silently selecting a default during analysis.

## 1. What the measurements must establish

The policy weights and episode prompt remain fixed. An external intervention moves a scene object during execution. The measurements distinguish:

1. Whether the assigned task is completed under the changed scene.
2. How far the physical outcome is from a valid placement, including failed episodes.
3. Whether the manipulated object's behavioral response depends on which object the unchanged instruction names as its reference.

They do not establish an internal representation, prove or disprove “understanding,” or assign failure to a neural module. A controller setting that improves completion establishes an effect of that deployment intervention in this setup; it does not by itself identify the original internal failure.

Use these three outcome types in the main paper. Richer logs support the secondary diagnostics below; do not promote every logged scalar to a headline metric.

| Main outcome | Unit | Population | Interpretation |
| --- | --- | --- | --- |
| Valid task completion, `success` | probability; contrasts in percentage points | all valid assigned reset blocks; also prespecified trigger-eligible population | Did the policy finish the assigned manipulation task? |
| Terminal geometric goal violation, `goal_violation_capped_m` | meters, also display centimeters; geometry-derived cap disclosed | every valid episode with required physical state coverage | How far is the object from the nearest valid placement region? Zero alone does not establish release or stable support. |
| Reference-selective goal improvement, `reference_selectivity_m` | meters of geometric improvement relative to sham | matched trigger-eligible branches with the fixed event protocol | Does the same visible object motion produce a different improvement toward the current valid goal when language names that object as its reference? Both within-reference effects must be shown. |

Correction timing, event response probability, integrated geometric error, failure composition, prompt invariance, and workload limits are secondary diagnostics. Report these when they explain an observed primary result; preserve all of them in the results archive.

## 2. Coordinate frames, relations, and valid goal regions

### 2.1 Coordinate contract

Freeze one robot-relative frame per robot stack. The basis vectors are `u_left`, `u_front`, and `u_up`; front points toward the robot under the stated viewing convention and up opposes gravity. Record the world-to-task transform numerically, its units, handedness, and calibration artifact hash. Keep cameras fixed unless a cell explicitly changes them. Do not infer the task frame from a rendered image or silently reuse the sign convention of another simulator.

At time `t`, record manipulated-object world pose `p_obj(t), R_obj(t)` and the named reference's world pose `p_ref(t), R_ref(t)`. Also record every candidate reference and distractor pose. Define:

```text
q(t) = p_obj(t) - p_ref(t)
y_left(t) = dot(u_left, q(t))
y_front(t) = dot(u_front, q(t))
y_up(t) = dot(u_up, q(t))
```

Keep the complete vector and both world trajectories. A change in `q` caused solely by moving the reference is not a robot correction.

### 2.2 Goal sets, not a unique desired offset

For relation `g`, define a geometric placement set `G_g(t)` from the actual named reference pose at `t`, the allowed support surfaces, manipulated-object dimensions, clearance, and the frozen task geometry. A position is admissible if a physically supported placement of the manipulated object at that position satisfies the requested relation and the non-directional placement bounds. Horizontal relations use a half-space beyond a robust object-extent margin intersected with the whole registered reachable support footprint. Orthogonal bounds come from the task surface; do not invent a narrow moving target box to force every scene change to require correction.

Compute geometric violation as the distance to this set:

```text
d_g(p, t) = min_{z in G_g(t)} ||p - z||_2
```

The minimizer is a scoring construction, not an oracle target given to the policy. Any admissible goal location is acceptable; the robot need not copy a moving reference's exact displacement or preserve an arbitrary initial offset.

Freeze a computable representation: analytic boxes/polygons for planar regions, supported shelf regions for vertical tasks, and object-footprint-eroded container interiors for containment. Document how orientation and collision clearance enter each set. For nonconvex sets, compute the minimum over the finite registered components and retain the chosen component ID. Verify the implementation against independent hand-checked and rendered cases.

| Goal family | Geometric set | Additional completion conditions |
| --- | --- | --- |
| Left/right/front/behind | relation region with declared orthogonal bounds, finite reachable workspace, object-size clearance, and allowed support | grasp/transport if required, release, stable support, no disallowed collision or boundary violation |
| Above/below | legal supported shelf or platform regions on the requested side of the reference | same release/support requirements; no unsupported hovering counted as placement |
| Inside | legal object-center/pose region inside container after erosion by manipulated-object footprint and wall clearance, plus permitted vertical depth | release and stable containment; intersection with the opening alone is insufficient |

“Inside” and “contains” are equivalent descriptions with reversed arguments, not opposite physical goals. “Above” is not automatically “on top of.” Do not use a scene-graph heuristic label as evidence of physical support.

Goal bounds make an under-specified natural-language instruction operational. Show those bounds in the paper and sensitivity appendix. They must be identical across compared cells except for the prespecified reference/layout transformation. Do not tighten a region after observing a policy's placements.

Freeze an empty-set rule. The model-blind intervention gate must establish nonempty legal regions throughout every planned motion, but a policy can subsequently displace an object into an infeasible arrangement. That remains behavioral evidence. Record `goal_set_empty=true`, its geometric cause, and task failure; never drop the episode as an infrastructure error. For the all-assigned continuous summary, use a predeclared capped distance `min(d_g, D_cap)`, with empty-set distance treated as above the cap and `D_cap` set from the registered workspace diagonal before inference. Archive uncapped finite distances and cap/empty-set indicators, report the capped fraction, and show a separate empty-set count. A capped value means at least that much violation or no legal placement, not an exactly observed distance. If no empty-set or capped observations occur, the summary equals ordinary nearest-goal distance.

### 2.3 Full versus diagnostic geometry

For planar response diagnostics, use the planar projection of the legal placement set so that ordinary carrying height does not dominate the signal. For the vertical replication, use the registered vertical slice with support and collision feasibility checked separately. For containment response, use the opening's admissible horizontal projection; completion still uses the full 3D containment predicate.

Name these `response_goal_projection` and `terminal_goal_set` separately. Never call projected containment error a successful insertion, or a vertical geometric relation a supported placement.

No relation is required to hold throughout transport. An object carried across the “wrong” side on its way to a valid final placement is not a task failure. Intermediate distances measure progress and correction, not compliance with an unstated trajectory constraint.

## 3. Task completion and terminal geometric error

### 3.1 First-placement time and stopping rule

This campaign evaluates the **first placement attempt after verified carrying**, not repeated recovery after release. Detect detachment using the frozen gripper/contact rule sustained for two native control ticks. Record both the first tick of the qualifying run (`t_detachment_onset`) and its detection tick (`t_release_detected`).

At detection, end the policy-controlled phase regardless of whether the placement looks correct. Stop the imposed reference motion at its actual current pose, issue no new policy requests, hold the robot's last safe target, and run exactly 1.0 s of passive physics for settling. Score the final manipulated-object pose against that frozen reference pose, with the registered support, release, and stability requirements. Record the hold command and physical settling trace. The freeze is performed at detection; the earlier onset timestamp is a timing annotation, not a claim that physics was changed retrospectively.

A wrong-side placement, drop, or unstable first release remains a failure. Do not allow a second attempt or regrasp to replace it. Temporary contact changes before the two-tick qualification are logged but are not yet terminal. Episodes that never satisfy the carry/release sequence run to the frozen timeout and retain their valid failure outcome.

Define `t_complete` as the end of the settling window and `placement_success` using the frozen predicates; the same terminal rule applies to an unsuccessful first placement. If release ends motion before the planned profile finishes, record `motion_truncated_by_release`. `destination_static` uses the planned destination; when motion truncates early, it is not an exact match to the attained terminal layout. Report planned and attained reference poses rather than making that false matching claim.

Once the first placement phase ends, later reference motion cannot retroactively turn it into failure. “Place” is a terminal goal. “Keep” and recovery after dropping would be different tasks and are outside this campaign.

### 3.2 Binary completion

```text
success = geometric_relation_correct
          AND required_manipulation_occurred
          AND released
          AND allowed_support_or_containment
          AND stable_for_registered_dwell
          AND no_registered_terminal_violation
```

Log every predicate separately, even when `success=false`. A policy that never grasps contributes a valid failure. An eligible policy that releases on the wrong side contributes a valid failure. A simulator crash without an observable outcome is an infrastructure-invalid attempt, not a zero success.

For the all-assigned estimand, intervention assignment is made before execution. If the physical trigger is never reached, record the actual task outcome and `event_delivered=false`. Do not drop the episode to make online performance look stronger. Report an all-assigned completion rate and a trigger-eligible completion rate side by side, with their numerators and denominators.

### 3.3 Terminal continuous outcome

For every first-placement attempt, successful or failed, use the settled object pose and reference pose frozen at detection. If the active phase ends without a qualifying release, record its active terminal state, run the common passive adjudication interval without synthesizing a release, and use the last valid settled physical state for terminal geometry. Such an episode remains a completion failure unless the required carry/release sequence actually occurred before the active cutoff. Preserve both active and settled endpoint fields. Define:

```text
goal_violation_m = d_g(p_obj(t_terminal), t_terminal)
goal_violation_capped_m = min(goal_violation_m, D_cap)
```

Include failed grasps, transport failures, wrong relations, and failed release if the required poses are valid. Use the capped field for the all-assigned estimate so policy-induced empty goal sets are not selectively removed, and provide uncapped distribution plots with their coverage explicitly named. Do not impute missing state from success. Do not convert all failures to one arbitrary distance. An object held above the correct planar area can have low planar violation while failing support/release; this is why full geometry and completion remain separate.

Archive signed coordinate offsets, object displacement from its initial pose, requested-side margin, distance to the closest valid goal, and failure predicates. Preserve historical endpoint fields as aliases only when the algebra and coordinate convention are exact. Keep V3's frozen scoring unchanged; a reanalysis under a V4 scorer must be labeled a separate derived analysis.

## 4. Matched response and reference selectivity

### 4.1 Pairing contract

For each language condition, compare the intervention branch with a sham branch from the same validated pre-intervention episode prefix. Preserve simulator state, policy observation history, action queue, controller state, policy cache where relevant, RNG states, task timer, and intervention clock. Matching only the visible object positions is insufficient.

Record a `prefix_id` and hashes of every restorable component. Demonstrate sham-versus-sham reproducibility in the engineering gate. **C2's primary selectivity analysis requires verified common prefixes**, through qualified deterministic fresh-session replay or a qualified full-state snapshot. If neither works, C2 remains blocked; do not substitute independent trajectories into its primary `H` estimator.

C1 may use a prospectively frozen independent-natural-rollout fallback with randomized condition order, retaining every start and pre-event state difference. Its completion and geometric-outcome contrasts then estimate reset-matched total protocol effects, not an exact post-grasp counterfactual. Do not use exact-prefix response detectors under that fallback, silently mix modes, or repair a valid no-trigger outcome by rerunning until it grasps.

Opposite prompts and different named references generally produce different prefixes. Their matching unit is the original reset block; do not require them to share a post-grasp state produced by a different instruction. The sham comparison is within the same prompt, and the semantic selectivity contrast then compares those within-prompt responses.

### 4.2 Primary reference-selective response

Let physical object `A` receive the same **prescribed** motion in two instruction conditions: `A` is the named reference in one and `B` is the named reference in the other. Swap identities/positions in the registered counterbalance so that one color or location cannot explain the result. Each condition has its own same-prompt sham.

Actual object motion is identical only over a verified common active interval. Different prompts can produce different robot trajectories and first-release times, and the stopping rule can truncate the physical stimulus at different times. The main C2 estimate is selectivity under the same prescribed intervention protocol, not necessarily under identical complete visual input. Report reference trajectories, actual exposure durations, truncation counts, and the predeclared common-active-window sensitivity analysis. Do not discard early-placement outcomes from the primary estimate to manufacture identical exposure.

The primary `h_response` is **2.0 s after the planned event onset**, with prespecified 1.0 s and 4.0 s sensitivity horizons. The common prefix fixes that onset before the intervention branches diverge, including the registered phase offset. Report native control-step counts as well as seconds. This anchor remains defined if release prevents motion or no qualifying changed observation reaches the policy. Let `G_move,c(t)` be the goal set for language condition `c` using the moving branch's actual named-reference pose. Score both branch object positions against that *same* goal set, using the capped planar response distance from Section 2:

```text
H_c(h) = d_cap(p_obj,sham,c(t_event_planned+h), G_move,c(t_event_planned+h))
         - d_cap(p_obj,move,c(t_event_planned+h), G_move,c(t_event_planned+h))
C_A(h) = H_named_A(h) - H_named_B(h)
```

The primary `reference_selectivity_m` is the equal-goal mean of the reset-paired `C_A` contrasts, with object identity/location and diagonal motion signs counterbalanced across blocks. Section 5.3 specifies how prespecified eligibility affects each goal's denominator. Positive `H` means actual object behavior improves proximity to the current legal goal relative to its same-prompt sham. Because the goal set is identical in the two terms, moving the reference alone with identical manipulated-object trajectories yields zero `H`.

Report `H_named_A` and `H_named_B` alongside `C_A`. A positive difference can arise solely because motion harms the irrelevant-reference condition; it does not by itself establish useful correction in the named-A condition. Keep each physical motion sign and goal relation visible in supporting tables before averaging. Interpret the response with first-placement success and terminal goal violation. No claim requires the relation to hold throughout carrying.

Use planar projection for C2 so carrying height does not dominate `H`. Archive full 3D values as secondary diagnostics. If policy-induced geometry makes the goal set empty, cap both distance terms, flag the event, and retain its behavioral failure; do not silently exclude it or call its zero improvement successful indifference.

Use an explicitly defined terminal extension when a branch ends before `h_response`: its settled terminal object pose and frozen reference pose are held constant for this *analysis array* after terminal settling, whether the attempt succeeds or fails. This does not run the policy longer or move the scene after release. Use actual passive-settling states when the horizon falls inside the settling interval. Flag these observations and show a sensitivity plot over the common physically observed pre-release window. Do not manufacture unobserved velocities or latencies from the extension.

Align primary paired trajectories by elapsed time after the shared planned event onset. Keep eligible early-release and unobserved-event branches in the estimate using the declared terminal extension. Do not require a post-intervention visibility event as a new inclusion criterion. Observation-anchored `H` and latency curves are secondary and must state their observed-event population. Interpolate poses only across short valid intervals using the frozen interpolation rule; do not interpolate across missing logs or terminal discontinuities.

If an implementation uses “move reference” versus “move distractor” without repeating the same physical object's prescribed motion under swapped reference naming, label the effect `reference_vs_distractor_response`, not the stronger same-prescribed-stimulus reference contrast. Different object positions can imply different occlusion and collision changes.

The finite move-and-stop C2 family is the confirmatory selectivity experiment; repeated-motion traces characterize behavior under increasing demands. Do not take near-zero net displacement across a reversal as absence of correction.

### 4.3 Actual object motion and unnecessary correction

For finite one-direction motion let `e_move` be its unit translation direction. Archive the supporting world-motion response:

```text
Delta_p_c(h) = p_obj,c(t_event_planned + h) - p_obj,c(t_event_planned)
M_c(h) = dot(e_move, Delta_p_move,c(h) - Delta_p_sham,c(h))
C_M(h) = M_named_A(h) - M_named_B(h)
```

`M` records actual manipulated-object displacement beyond sham along the imposed motion. `H` records whether the change is geometrically helpful. Neither can be replaced by a reference-relative coordinate change. For reversal profiles, calculate motion response within each monotone segment and show the trajectory instead of collapsing positive and negative legs into one net displacement.

Report these cases separately:

- Response toward the new goal, followed by valid completion.
- Response toward the new goal, followed by physical completion failure.
- Substantial motion response without improved goal geometry.
- Little response with already-valid final placement.
- Little response with invalid final placement.

A policy may legitimately ignore a movement if its planned placement still satisfies the instruction. Do not define responsiveness as always copying reference motion. The subset where a sham endpoint would become invalid under the moved-reference goal can be shown as a disclosed, outcome-conditioned diagnostic; it must not replace the all-assigned or prespecified eligible estimand.

## 5. Timing, response detection, and competing events

### 5.1 Record the complete timing chain

The event does not become visible to the policy when the simulator command is issued. Record:

The first qualifying changed observation is the first delivered policy-input frame captured after the moved object has actually translated at least **1 mm**, with a nonempty visible-object mask in at least one camera included in that request. Record actual input camera IDs, mask pixel counts, displacement, capture and delivery times, and observation hashes. Visibility is a simulator-derived measurement proxy used only by the evaluator; it is not a claim that the policy attended to the object or that a 1 mm change is perceptually distinguishable at its image resolution. A frame from a camera that the policy does not receive does not qualify. The 1 cm manipulated-object response threshold is a different detector.

| Field | Meaning |
| --- | --- |
| `t_trigger_eligible` | first time the frozen physical trigger holds for its required dwell |
| `t_event_planned` | event onset fixed at the common prefix, including phase offset; the primary response anchor |
| `t_intervention_command` | intervention controller receives the movement command, if the task remains active |
| `t_motion_actual_onset` | measured reference displacement crosses the motion-onset threshold |
| `t_observation_capture` | capture time of each sensor frame, in simulation and wall time |
| `first_changed_observation_id` | first observation that contains a change large enough to meet the frozen visibility/motion threshold |
| `t_changed_observation_available` | that observation reaches the policy input buffer |
| `t_policy_request_start/end` | inference call starts/returns, with referenced observation IDs |
| `t_first_updated_action_executed` | first executed action generated from a qualifying changed observation |
| `t_object_response` | first sustained qualifying object response, if observed |
| `t_detachment_onset`, `t_release_detected`, `t_complete`, `t_episode_end` | first-placement and competing events, with onset distinct from detection |

The primary campaign uses controlled simulator time: the standard query period is 0.50 s, C4's fast query period is 0.25 s, and the emulated observation-to-action delay is 0.10 s. Round each upward to an integer native control tick, freeze the achieved schedule, and report requested and achieved values. The C4 schedule switches only at the validated natural-grasp checkpoint, preserving its standard-schedule prefix.

At virtual time `t`, capture the observation and submit the inference request. Pause the simulator while wall-clock inference computes. Then advance physics and the external object path over `[t,t+delay)` while executing the already available action queue under the frozen queue-exhaustion rule. Only at `t+delay` may the new action chunk, computed from the observation at `t`, become executable. Do not provide the policy with an observation from the future delay interval or move the reference after a fresh observation but before stamping that observation's capture time. Unit-test this timeline against recorded request/action IDs.

Report latency from actual scene change to object response and its components: observation delay, the emulated inference/dispatch delay, queued old-action execution, and residual physical response. Log actual wall-clock inference separately as deployment cost. It is not the physical delay experienced in this controlled simulator-time experiment, and the results are not a real-time throughput claim. The 60 s episode cap, trigger deadline at 40 s, and maximum 20 s intervention window use simulator time; early placement completion remains allowed.

### 5.2 Response detector

The frozen displacement threshold is **1 cm sustained over two native control intervals**, with 0.5 cm and 2 cm sensitivity thresholds. Validate that sensor/physics resolution and sham-versus-sham noise permit this detector; a failed measurement gate blocks response-timing claims rather than authorizing an outcome-driven threshold change. Any accompanying velocity threshold must be geometry/timing-derived and documented before release.

For an exact-prefix pair, a qualifying response requires:

1. Manipulated-object world displacement differs from same-prompt sham by the frozen magnitude.
2. The difference persists for the required dwell.
3. The object is under the robot's control during the detected change, or the event is classified separately as a passive/contact-mediated response.

Classify the response as helpful only if the geometric improvement criterion also holds. Direct physical contact by the moving reference with the manipulated object or robot invalidates a “visually mediated correction” interpretation; it is a logged contact event and a valid behavioral outcome unless it violates the frozen intervention-validity gate. Do not infer helpful correction from joint action RMS alone.

### 5.3 Eligibility and denominators

Maintain this flow for every assigned block and branch:

```text
assigned -> valid rollout -> trigger eligible -> motion delivered
         -> changed observation delivered -> response / competing event / timeout
```

Every count must reconcile with the manifest. Publish the number at each stage. Repeated valid no-grasp outcomes are not infrastructure failures. An event that the policy never observes because it completes earlier is not an observed failure to react to the event.

Estimate the trigger-eligible intervention effect only within eligibility defined *before* the intervention branches diverge. The natural-grasp trigger requires at least 4 cm lift, 0.20 s of verified carrying, and relative object/gripper drift no greater than 1 cm under the validated physical detector. Simulator state evaluates the external experiment trigger and scoring; it is not extra policy input.

If different prompts reach different trigger states, their eligible populations can differ; disclose that for semantic contrasts. For C2, form a per-goal selectivity contrast where both named-reference conditions have qualified eligible common prefixes and valid assigned branch records. Do **not** further require motion delivery, changed-observation delivery, response, or survival until the response horizon. Estimate each goal's mean over its explicitly reported eligible reset population and give equal weight to the four goal means; bootstrap original reset blocks with their eligibility pattern intact. This avoids requiring all four goals to be eligible in every block or silently overweighting easier goals. If one goal has no estimable eligible pairs, the four-goal confirmatory aggregate is not estimable; do not silently drop that goal and relabel the remainder. Publish complete-prefix counts as a supplementary common-population analysis. Never condition the main estimate on eventual success, successful release, a useful-looking trajectory, or an invalid sham endpoint.

### 5.4 Censoring and competing events

By the frozen response deadline, classify each eligible event as:

- Helpful response observed.
- Other substantial response observed.
- First placement/release before qualifying response, split by whether passive settling subsequently succeeds.
- Grasp loss or terminal behavioral failure before response.
- No qualifying response by the administrative deadline.
- Infrastructure interruption or missing required observation/trajectory evidence.

First release, grasp loss, and terminal failure are competing events, not independent right-censoring. A mean response time over responding episodes hides nonresponse and early failure; never present it alone. Show response probability by the deadline with all competing-state proportions, then the conditional response-time distribution clearly labeled. A cumulative incidence curve is appropriate if event timing is adequately logged. Do not use ordinary Kaplan–Meier censoring for grasp loss and interpret it as the chance of correction. Passive object movement after the first release does not count as a policy correction, even though it contributes to the placement outcome.

Only fixed experiment timeout supplies administrative censoring of a still-running eligible episode. Infrastructure loss is missing evidence with its own ledger, not an uninformative biological/behavioral censoring assumption. Repeated movement events within one trajectory share a policy state and are not independent trials.

## 6. Motion stress and action-execution interventions

Increase one declared motion parameter at a time unless the matrix explicitly contains a small interaction comparison. Actual speed, displacement, acceleration, pause duration, reversal count, and frequency are measured from simulator poses; commanded values alone are insufficient. Normalize for workspace scale only as a supplementary display and never pool raw stack-specific success rates.

For the frozen minimum-jerk segment, `s(u)=10u^3-15u^4+6u^5`, `u=clip(t/T,0,1)`, and `p_ref(t)=p_start+s(u)*Delta`. The nominal peak speed is `1.875*||Delta||/T`. Report duration and peak speed, not only a label such as “fast.” For the 12 cm horizontal translation before any model-blind fixture scaling:

| Profile | Duration/waypoints | Nominal peak speed |
| --- | --- | --- |
| Acute move-and-stop | 0 to +12 cm in 0.5 s | 45 cm/s |
| Slow drift | 0 to +12 cm in 4.0 s | 5.625 cm/s |
| Fast drift | 0 to +12 cm in 1.0 s | 22.5 cm/s |
| Reversal | 0 to +12 cm at 2.0 s, then to -6 cm at 4.0 s | 11.25 cm/s outward and 16.875 cm/s on the return leg |

“Fast drift” is faster than slow drift but slower than the acute movement; it is not the campaign's maximum-speed condition. The reversal changes movement history as well as speed and does not isolate speed alone. Containment and the second-stack fixture use nominal 8 cm translations. Apply the model-blind largest-feasible scale from `{2.0,1.5,1.0,0.75,0.5}` consistently within a fixture, and report the attained values. No policy outcome selects the scale.

For the primary horizontal fixtures, publish `area(G_start intersect G_destination)/area(G_start)` for every relation, motion sign, and registered reset. The model-blind geometry gate requires at least 20% of the initial legal planar area to be removed in the designated shrinking-direction conditions for each horizontal axis. The corresponding expanding-direction condition can legitimately retain all old valid placements. The precise area calculation and fixture acceptance aggregation are frozen in the geometry receipt. If that gate cannot be met while all physical-validity gates hold, mark the primary fixture low-information/blocked; do not substitute a narrow hidden goal region or choose a displacement using observed policy placements. Passing this area gate does not guarantee that a particular policy's chosen endpoint will require correction.

C5 changes the reference **horizontally** on a fixed shelf arrangement while above/below goals retain supported upper/lower regions with the declared horizontal-overlap rule. It tests vertical relational goals under a moving scene, not correction to vertical reference motion. Its horizontal alignment requirement is part of the operational scoring definition and must be disclosed. Never describe it as a gravity-symmetric vertical reflection.

For each stress level report the three primary outcomes, event coverage, actual exposure duration, and latency components. Plot the entire prespecified grid. Do not report only the highest successful speed or fit a sharp universal threshold to sparse/nonmonotonic data.

If a practical limit is desired, freeze a completion criterion such as `>=80%` and use confidence intervals. A “supported operating level” requires the prespecified lower confidence bound to clear that criterion. If no level clears it, report “not established.” If performance is nonmonotonic, do not interpolate a monotone boundary without showing model mismatch. This is a limit for this checkpoint, controller, motion family, task, and setup.

For C4, the common matching checkpoint is the verified natural grasp, before the schedule intervention. Standard and fast trajectories may legitimately diverge between that checkpoint and the later movement onset. Within each fixed schedule, its movement/sham pair must still share its treatment-inactive prefix. Do not reject the intended cadence effect as a pre-event mismatch between schedules.

For C4, log both generated horizon and the number of actions actually executed before a newer chunk replaces it. Keep the generated horizon fixed and change the query cadence from the registered standard to fast schedule only after the shared natural-grasp checkpoint. Keep model weights, prompt, observation processing, and native controller settings otherwise fixed. Verify that changing execution cadence does not silently change timestep, action interpolation, normalization, input history, camera cadence, or stopping semantics. Equalize or disclose resource and wall-time conditions. A benefit can arise from fresher observations and more frequent replanning; it is not an isolated inference-speed effect.

## 7. Static evidence and wording analysis

V3 remains discovery evidence under its original manifests. Preserve the clean left/right reflection, continuous endpoint analyses, and wording limitations. Do not rerun a historical cell merely to fill a new table. New stationary controls are required for the new scene, prompt, task family, runtime, and intervention timing; their scientific role is a matched baseline for V4.

For left/right, retain positive-left signed endpoints and the right-minus-left success convention when reproducing historical figures. State every sign convention in captions. Do not merge the 27-block reflection core with the larger endpoint cohorts, or any DROID/RoboLab results with a second robot stack.

For equivalent wording, compare the same physical goal under direct and argument-inverted clauses with the same carrier, named reference, scene, event, and reset. Report per-goal paired effects on success, goal violation, and selectivity. Do not pool opposite-goal signed offsets in a way that cancels failures.

Similarity of endpoints is stronger than needed for two descriptions to specify the same valid goal. Report goal agreement and completion first. This campaign does not authorize a statistical equivalence claim: its fixed allocation targets large effects and does not guarantee narrow equivalence intervals. A future equivalence claim would require prespecified task-based margins and a valid equivalence test such as TOST; an ordinary nonsignificant difference is not evidence of invariance. Failure of exact endpoint equivalence is not automatically failure of semantic equivalence when both placements are valid.

## 8. Failure and missingness taxonomy

Keep all individual predicates and one mutually exclusive earliest decisive failure label. The recommended summary labels are `no_grasp`, `grasp_lost`, `transport_incomplete`, `wrong_goal_region`, `release_failed`, `support_or_containment_failed`, `timeout_without_completion`, and `success`. For ambiguous trajectories use `unresolved_behavioral_failure` until the frozen audit resolves them; do not infer a language failure from a wrong region alone.

The compact ledger also requires the helper's coarse `outcome.failure_stage`. Preserve the more detailed `outcome.failure_label` alongside it; the coarse field is a compatibility grouping, not a replacement for the recorded predicates.

| Detailed failure_label | Coarse failure_stage |
| --- | --- |
| success | none |
| no_grasp | pickup |
| grasp_lost / transport_incomplete | transport |
| wrong_goal_region | wrong_relation |
| release_failed / support_or_containment_failed | release |
| timeout_without_completion | timeout |
| collision-caused terminal failure | collision |
| model_output_invalid / unresolved_behavioral_failure | other |

Use the earliest decisive failure under the frozen predicate hierarchy. A timeout after an already diagnosed no-grasp failure remains `no_grasp` with a separate timeout flag; do not relabel it simply because the clock expired. The full trajectory retains all subsequent events. The helper checks only the compact structural contract; the independent scorer audit checks the detailed label, predicates, and this mapping.

In C2, record whether the endpoint satisfies the named-reference goal, the other-reference goal, both, or neither. An endpoint compatible only with the other reference is evidence of that geometric pattern, not proof that the network internally selected that reference. Overlapping goal sets legitimately produce the `both` category and do not distinguish binding from an endpoint alone. Do not force a mutually exclusive reference-confusion label where the geometry is ambiguous.

Separately retain intervention exposure labels: `trigger_not_reached`, `event_not_delivered`, `event_not_observed`, `event_delivered_valid`, `event_contact_mediated`, `motion_truncated_by_release`, and infrastructure-specific errors. These are not substituted for manipulation outcomes.

Infrastructure invalidity requires an objective prespecified reason: wrong checkpoint/hash, failed reset attestation, malformed action caused by a verified interface defect, simulator/service crash attributable to infrastructure, incorrect stimulus trajectory, missing required state/video/timing evidence, or proven scorer/runtime corruption. A valid action sequence that performs poorly is behavioral evidence. Never repeatedly retry a valid failure.

Freeze handling of native invalid policy outputs. If the correctly qualified model itself emits a nonfinite or invalid action with intact inputs and no infrastructure defect, stop before applying it, preserve the last valid physical state, and classify `model_output_invalid` as a behavioral/runtime failure under the deployed policy. Do not rerun until it produces a favorable finite output. An unresolved cause is quarantined pending objective adjudication, not silently called either success or infrastructure noise.

Each invalid attempt remains in an append-only ledger with its assigned cell ID, attempt ID, timestamps, observed prefix, raw hashes, reason, and exact rerun authorization. A replacement attempt receives a new attempt ID and the same immutable cell assignment. No exhausted or unresolved cell becomes a zero in the denominator.

## 9. Statistical units, planned contrasts, and uncertainty

### 9.1 Independent unit

The independent unit is the randomized reset block, not the action, video frame, generated sample, branch, event, or nominal seed string. A block contains all registered prompts, sham/intervention branches, motion signs, and runtime comparisons required for that contrast. Prefix-sharing branches are correlated by construction.

One physical reset repeated under several policy sampling seeds yields nested observations. Average at the declared block level or bootstrap the reset with all nested observations intact. A deterministic sampler does not create new independent evidence when its seed number changes. Log reset coordinates and policy RNG effectiveness separately.

Keep estimates per exact checkpoint and per robot stack. Architecture comparisons are descriptive because training data and other checkpoint choices differ. The second stack changes more than embodiment alone and does not identify a pure embodiment effect.

### 9.2 Effect estimates

For a matched contrast, compute the prescribed reset-paired contrast and report its mean with a reset-cluster bootstrap interval. C1 averages four goal contrasts within every complete assigned block; C2 averages the four goal-specific eligible means as specified in Section 5.3. Retain all condition rows and the eligibility pattern within each sampled block; if policies share a reset bank, preserve that matching for an explicit policy-difference analysis. Use 10,000 bootstrap resamples with the analysis RNG seed frozen in the manifest.

For binary outcomes report risk differences in percentage points, not only odds ratios. For continuous outcomes report centimeters and robust distribution plots; means are primary if frozen as such, medians are supplementary. Report raw counts, missing pairs, and effective block counts for each estimate. Ordinary 95% intervals are pointwise, not simultaneous coverage for the four primary hypotheses; label them accordingly alongside the Holm-adjusted tests.

Sparse eligible populations can make a C2 bootstrap replicate omit an entire goal. Count such resamples and do not silently discard them or impute that goal's effect as zero. If the registered estimator cannot be evaluated reliably over its bootstrap distribution, mark that confirmatory uncertainty estimate not estimable and retain the observed eligible counts and descriptive effects. Do not create new favorable episodes to repair statistical sparsity.

Wilson intervals are acceptable for a single rate only when its observations are independent Bernoulli units. Do not apply them to a pooled collection of branches or repeated events. Use the clustered estimator for averaged within-block rates and contrasts. When no successes or no discordant pairs occur, a degenerate bootstrap interval is insufficient; include an exact/binomial-compatible bound at the independent unit level where its assumptions apply, or report the limitation explicitly.

### 9.3 Primary contrast registry

The final matrix has two confirmatory contrast families, each evaluated separately for the two main policies:

1. **C1 wording × movement interaction in completion.** For each reset block and physical goal, calculate `(success_move,inverse - success_sham,inverse) - (success_move,direct - success_sham,direct)`, then average the four goals within the block. The destination-static condition is a separate difficulty control, not the sham in this interaction. A negative effect means movement reduces completion more under inverse wording in the declared contrast convention.
2. **C2 reference naming × same prescribed physical movement interaction.** Calculate the reference-selectivity contrast specified in Section 4. For each goal, average its reset-paired contrast over the prespecified eligible population, then give equal weight to the four goal means as required by Section 5.3. Preserve the registered motion-sign and object-identity balance and bootstrap the original reset blocks with eligibility masks intact.

These four tests form one Holm-controlled family at `alpha=0.05`. Use the frozen two-sided **studentized reset-block bootstrap null test**, with 10,000 resamples and analysis RNG seed **20260905**, paired with pointwise reset-cluster confidence intervals. Estimate standard errors by delete-one-reset-block jackknife of the complete estimator, for the observed dataset and within each whole-block bootstrap resample. Preserve all conditions and eligibility. Use `T_observed=theta_hat/SE_hat`, `T_star=(theta_star-theta_hat)/SE_star`, and `p=(1+count(abs(T_star)>=abs(T_observed)))/(B+1)`. The intervals are pointwise 95% percentile reset-block bootstrap intervals. This is a bootstrap approximation, not an exact randomization test. The implementation and centering/standard-error code must pass independent synthetic-data tests before release.

If the observed standard error is zero, or any required resample has an undefined estimator or undefined/zero standard error, output `p_value=null`, `test_status=not_estimable`, and the descriptive effect/coverage plus the count of affected resamples. Do not discard those resamples. If an estimator is undefined, its bootstrap interval is also not estimable. Do not invent a p-value or silently switch to a different hypothesis test after seeing the data. Preserve the four-hypothesis family; an unestimable test makes no rejection and is conservatively treated as 1 only for Holm bookkeeping, while its displayed p-value remains null. Do not independently sign-flip every branch or video frame.

The continuous goal-violation outcome explains the completion effects and remains a main estimated outcome. C3 motion stress and C4 execution-schedule effects are prespecified secondary analyses; their effect sizes and intervals are reported without quietly promoting their best comparison into a confirmatory claim. C5–C8 scope replications are estimated separately by family and stack. Exploratory per-relation, per-object, waveform, subgroup, and failure-stage comparisons are labeled as such.

If making an additional confirmatory claim using goal violation or a secondary outcome, amend the multiplicity registry before confirmatory inference. Report effect sizes and intervals whether or not the null is rejected.

No statistical equivalence test is in this campaign's primary registry. Do not add an “invariant” claim from an ordinary nonsignificant null test. A separately authorized future equivalence analysis would require a practically justified margin, estimand, test, and any multiplicity handling before its outcomes are examined.

### 9.4 Sample precision and one-campaign execution

The final allocation is 128 genuinely independent reset blocks for C1 and C2, and 64 for C3–C8. These are not interchangeable with 128 stochastic repeats of one reset. The eight families contain 17,664 assigned condition episodes; shared prefixes and reused C1/C3 controls reduce duplicate execution without creating additional independent evidence. Engineering pilots, privileged physical validation, and infrastructure-invalid attempts are accounted for separately. Use a fresh seed namespace disjoint from those pilots and historical cohorts.

At `n=128`, the approximate worst-case 95% half-width of one independent binomial rate is 8.7 percentage points. A paired risk-difference interval depends on discordance: approximately 12.3 points at 50% discordance near zero difference, and up to 17.3 points in the extreme. At `n=64`, a single rate's corresponding half-width is about 12.3 points. These are planning approximations, not promised final precision or power guarantees.

The C1 wording × movement interaction is a difference of two risk differences, with block-level range `[-2,2]` before averaging goals. Its worst-case approximate half-width can be 34.6 percentage points at 128 blocks; averaging four goals helps only to the extent that their outcomes are not perfectly correlated. Do not cite the single-rate precision as the interaction's precision. C2's effective sample size is also reduced when the required language-specific prefixes do not both reach the trigger.

The campaign accepts this precision limit in exchange for one fixed allocation. Do not resize the sample using pilot performance, interim results, or a promising p-value. Report a planning sensitivity calculation for the actual interaction and selectivity estimands using explicit variability/discordance assumptions. Failure to resolve a small effect leads to an inconclusive estimate, not extra outcome-driven runs or an equivalence claim. More available GPUs permit independent reset diversity and simultaneous planned conditions; they do not justify treating correlated repeats as sample size.

Engineering pilots establish feasibility, scorer resolution, timing, and variance assumptions. They are excluded from inference. Freeze all thresholds, conditions, allocation, exclusions, and analysis code after engineering validation and before confirmatory result inspection. Run one complete frozen campaign in parallel. Complete prespecified cells and replace only infrastructure-invalid attempts; do not stop or expand based on a promising p-value. Any later scientific change is a labeled amendment, not a rewrite of the original freeze.

## 10. Required raw records and derived tables

Store large arrays and videos on persistent experiment storage; commit manifests, hashes, compact tables, analysis code, and representative figure renderers to Git. All schemas are versioned. Missing optional fields are `null` with a reason, never a numeric zero.

### 10.1 `episodes.jsonl`

Required fields include:

```text
schema_version, protocol_id, protocol_sha256, episode_id, cell_id, attempt_id,
reset_block_id, physical_reset_id, prefix_id, branch_id, parent_attempt_id,
policy_id, checkpoint_uri, checkpoint_revision, checkpoint_sha256,
policy_code_commit, adapter_commit, simulator_commit, container_digest,
normalization_sha256, controller_config_sha256, scorer_sha256,
robot_stack, task_id, scene_id, scene_hash, frame_transform_sha256,
relation, manipulated_object_id, named_reference_id, moved_object_id,
prompt_id, prompt_text, prompt_sha256, wording_family,
reset_seed, policy_seed, effective_rng_verified, randomization_index,
motion_family, motion_sign, motion_amplitude_m, motion_speed_m_s,
motion_frequency_hz, motion_acceleration_m_s2, motion_profile_sha256,
runtime_mode, generated_action_horizon, executed_action_horizon,
control_dt_s, observation_period_s, simulation_clock_mode,
assigned, behavioral_valid, invalid_reason, trigger_eligible,
event_delivered, event_observed, motion_truncated_by_release,
success, t_terminal, terminal_reason, goal_violation_m,
goal_violation_capped_m, D_cap_m, goal_set_empty, goal_violation_cap_applied,
signed_left_m, signed_front_m, signed_up_m,
object_displacement_world_m, reference_displacement_world_m,
all_completion_predicates, failure_label, required_log_coverage,
raw_uri, raw_sha256, video_uri, video_sha256, event_log_sha256
```

Add the inherited reset/policy/cache/action-queue hashes for each branch. Policy input provenance must show that privileged simulator positions, goal labels, scorer outputs, and future reference trajectories never entered the policy input unless an explicitly separate diagnostic arm says so.

The compact accepted-result ledger uses `episode_id` as the immutable planned cell identifier and `attempt_id` for the execution attempt; `cell_id` may be an exact alias in richer records. It distinguishes `status=valid` from `status=infra_invalid`. The validator's required `outcome` object can contain the terminal scores/predicates listed here. Additional flat analysis tables are derived views, not separate sources of truth. Nonresponse latency is `null` with its competing-state reason and remains a valid behavioral row.

### 10.2 `trajectory` arrays, sampled at every executed control step

```text
simulation_time, monotonic_wall_time, physics_step, control_step,
object_pose_world, named_reference_pose_world, all_candidate_object_poses,
object_linear_angular_velocity, reference_linear_angular_velocity,
robot_joint_position_velocity, end_effector_pose, gripper_state,
contact_body_pairs_and_impulses, support_contacts,
commanded_action, executed_action, action_units, command_source_request_id,
action_chunk_id, action_index_in_chunk, pending_action_count,
observation_ids_used_by_action, observation_capture_times,
reference_controller_command, reference_actual_tracking_error,
goal_set_parameters, response_projection_parameters,
release_predicate, stability_predicate, grasp_predicate,
relation_predicates, containment_predicate, boundary_violations
```

Record enough geometry to regenerate every score with an independently versioned scorer. Sampling only final positions makes correction timing and passive-motion checks impossible. Sampling every inference call while executing many actions between calls is also insufficient.

### 10.3 `requests.jsonl` and `events.jsonl`

Requests bind exact observation IDs and hashes, prompt hash, input history/cache identity, generated horizon, action hash, RNG state identity, and start/return/dispatch/execution timestamps. Store the actual policy-input RGB/depth frames losslessly (only modalities the policy actually receives), their source camera/crop/resize metadata, and numeric proprioceptive inputs in content-addressed storage. An `observations.jsonl` index maps every request input to its URI, hash, timestamp, and history position. Hashes or compressed viewport video alone cannot reconstruct the submitted input. Preserve normalization/tokenization identities and any adapter transformation needed to reproduce it. Retain decoded future predictions only when the released interface exposes them; absent futures remain unavailable, never a zero-quality prediction. Mandatory viewport video covers every excluded engineering pilot and every main policy episode, including failures and passive settling.

Events contain eligibility, actual onset, delivery, observation, response, release, support loss, collision, timeout, and completion timestamps with detector versions and thresholds. Repeated events have `event_index` and the same parent episode/block identity. Record planned events that never occur and why.

### 10.4 Required derived files

- `coverage_by_cell.csv`: planned/completed/valid/invalid/eligible/delivered/observed counts and missingness.
- `episode_outcomes.parquet`: all complete outcome predicates and continuous scores, including failures.
- `event_outcomes.parquet`: motion exposure, response, competing event, and latency components.
- `paired_contrasts.parquet`: one row per reset block and registered contrast, with every contributing cell ID.
- `primary_results.csv`: estimates, intervals, raw/adjusted tests where specified, denominator, and contrast registry key.
- `scope_replications.csv`: relation, object, task, and stack-specific estimates without cross-stack raw pooling.
- `wording_results.csv`: per-goal matched direct/inverse contrasts; no unsupported equivalence labels.
- `failure_composition.csv`: behavior outcomes and intervention exposure, in separate columns.
- `timing_and_motion.csv`: actual versus commanded motion and observed/inference/queue/physical delays.
- `audit_report.json`: scorer tests, video review, coverage, deterministic replay, identity, and hash checks.
- `results_manifest.json`: hashes of all inputs, analysis code, environment, output tables, and figure sources.

## 11. Measurement acceptance before the full campaign

The engineering agent must demonstrate the following with small deterministic fixtures and excluded pilot rollouts:

1. Moving only the reference changes relative coordinates but produces zero manipulated-object world motion in a stationary-object fixture.
2. A valid placement anywhere inside the legal goal region receives zero geometric violation; no arbitrary exact target is required.
3. A wrong-direction but released and stably supported placement is geometrically wrong, not a grasp failure; a correct-direction held object fails release.
4. “Inside” versus “contains” yields identical goal sets for the same physical roles; inverse directional wording yields the identical set for the same physical goal.
5. The horizontal frame is verified by visible landmarks, and vertical scoring distinguishes above from support/contact.
6. A sham-versus-sham branch produces the expected baseline response variation and compatible cached-state/action-queue history.
7. A changed observation cannot be attributed to an action generated before that observation existed; clocks and request IDs reconcile.
8. A commanded motion rejected by the simulator does not masquerade as nonresponse to a delivered movement.
9. Early completion, no grasp, grasp loss, no response, and infrastructure loss enter their correct distinct denominators.
10. A completed placement is not rescored as failed after an unrequested later scene movement.
11. The same raw trajectory reproduces the same score and event labels on another machine under the pinned analysis environment.
12. Every planned contrast resolves to the intended unique cells and independent blocks; duplicated seeds/branches cannot inflate sample size.

Audit a prespecified random video sample stratified by family and outcome, with policy identity concealed where practical. Freeze the sample fraction and inter-rater disagreement rule before the full analysis. Separately review every infrastructure anomaly and a diagnostic sample of unusual behaviors. Report the random audit and anomaly review separately; attractive clips are not the audit sample.

Do not release the full queue until this report passes. This limited engineering stage is what makes a single large confirmatory campaign credible; it is not an invitation to run successive exploratory behavioral cohorts until the results look favorable.
