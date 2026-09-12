# World-model paper revision implementation plan

> The [ablation specification](../../ABLATION_SPEC.md) now fixes the exact
> matrix: 24 confirmation layout pairs, named N3/D1 primary configurations,
> optional D2 guidance cells, and annotation budgets including history and
> earlier-horizon images. It supersedes the ranges and conditional model
> substitutions in this earlier plan. Constant velocity is explicitly an
> observation-history baseline and has extra temporal input relative to Nano's
> history-length-one model interface.

> For agentic workers: use superpowers:subagent-driven-development or superpowers:executing-plans when implementation is requested. The current task delivers a research plan; it does not launch experiments.

**Goal:** Establish whether a world–action model's generated future predicts the object motion its own actions produce when the scene layout changes.

**Architecture:** Correct the existing behavioral analysis, qualify prediction/execution recordings, develop the measurement on separate scenes, then evaluate a frozen protocol on new layout pairs. Keep each model configuration and cohort separate.

**Tech stack:** Existing Python analysis and DROID/RoboLab integrations; decoded model frames, original execution RGB and simulator traces; matched image annotations; CoRL LaTeX manuscript.

**Spec:** The reviewer findings and research design below supersede the earlier eight-trial outline in ../../NEXT_EXPERIMENT.md.

Date: 2026-09-12. Status: proposed research plan, not a preregistration or a report of new results. Historical source: ce561e66f82e95055e39d3d7711691982f6b2086. Reviewed manuscript version: ea9d67d.

## 1. Recommended paper direction

**Question: When the scene changes, does a world–action model's generated future still describe the motion produced by its own actions?**

The existing 13/54 to 50/54 completion difference motivates this question. It is a result for a custom DreamZero action-guidance configuration, not released-default DreamZero. The current evidence does not establish prediction accuracy.

Three possible directions were considered:

| Direction | Value | Decision |
| --- | --- | --- |
| Repair the behavioral paper and add layouts | Necessary controls; still close to existing instruction-following benchmarks | Do the repairs, but use this as supporting evidence |
| Compare generated and executed motion under the same layout interventions | Directly tests a world-model output, using the existing task and integrations | Recommended main study |
| Change a controller to prevent failures using its forecasts | Could demonstrate practical value, but adds a new method and substantial validation | Stretch only after forecast usefulness is established |

The intended contribution is the response to a controlled scene change: does the forecast anticipate the change in executed motion, or show progress the jointly generated actions fail to realize? Beating a persistence baseline alone would be a modest result.

Working title: **Do World–Action Models Predict the Motion Their Actions Produce?**

Do not promise an award or choose the conclusion before observing the data. A replicated positive or negative result may be useful. An ambiguous result remains ambiguous.

## 2. What the review requires us to fix

| Review finding | Planned correction | Completion criterion |
| --- | --- | --- |
| DreamZero variant is inadequately identified | Carry action-guidance style/scale, video guidance, sampling, action horizon, checkpoint and integration versions into tables and methods | Every result identifies the actual configuration; no default-model claim for custom scale 2 |
| Goal-dependent stopping affects final positions | Add existing action/trajectory sensitivity checks; use a goal-independent evaluation schedule in the new study | Distinguish instruction effects from differences in measurement time |
| Only one base layout and its reflection | Select new layouts using physical validity criteria before examining target-model outcomes | Independent scene units replace repeated execution of one scene as the main evidence |
| DreamZero repeats have unexplained variability | Trace first divergence across observations, actions, simulator state, reset and scheduling records | Describe what varies; do not call fixed-seed repeats independent noise samples |
| Placement depth is overemphasized | Explain G as endpoint midpoint asymmetry and show expected reflection behavior | No suggestion that a sign reversal or greater depth is inherently better |
| Novelty is unclear | Compare with MESA, LIBERO-CF and prediction-based evaluation work | State the additional question this experiment answers |
| No world-model output is measured | Match true generated frames with actual execution at their physical target times | Main result concerns exposed prediction, not model names alone |

The old failure categories remain execution-stage observations, not labels of what the model understood.

### What can be recovered before new inference

Nano's V3 records list 738 decoded future sequences, retained step positions, action-request starts, executed-prefix lengths and original viewport-video identities. DreamZero records list future and execution-video identities; its compact episode table does not include all step trajectories. These are archive records, not confirmation that the media are currently accessible.

First try one complete Nano episode and one complete DreamZero episode from an authorized archive. Verify the original files and their timing. Publication videos that concatenate forecast horizons are presentation material, not a frame-alignment specification.

The review's Nano comparison just before either paired episode terminates retains 27/27 ordered pairs in each layout. Reproduce it as a retrospective sensitivity analysis, explicitly disclosing its outcome-dependent time. It is not a prospective fixed-horizon result. The early common horizons in the historical data contain almost no cube motion; later horizons exclude early-terminating episodes. Do not solve this by silently keeping only long episodes.

## 3. Recording feasibility before scaling up

Use the two recovered examples to check the complete measurement procedure. If a new model configuration or recorder requires qualification, run the proposed **eight-episode recording pilot**:

2 model configurations × 2 existing layouts × 2 commands = 8 episodes.

This is a new diagnostic cohort, not a rerun that replaces historical valid episodes. Its purpose is recording and measurement qualification, not significance.

Primary candidates:

- Cosmos 3 Nano Policy DROID, with the full evaluated inference configuration resolved and recorded.
- DreamZero-DROID through released/default inference, if executable and qualified. A local scale-1 equivalent may be described as equivalent only after verification against the official path.
- Custom DreamZero negative-branch action guidance scale 2 remains a separately named historical variant. Do not immediately double the new study to cover both guidance settings.

For each request retain the conditioning observations, original proposed actions, executed prefix, actual execution RGB and simulator state, generated frames and raw output identity, camera and resize/crop mapping, reset identity, timestamps, effective noise settings and all truncation events.

Qualification requires:

1. The video is a genuine model output generated with the recorded action proposal. Turning on decoding does not alter that proposal.
2. A documented mapping connects generated frame index to target physical time and action index. A matching file index is not enough.
3. The target time lies within the prefix actually executed before replanning. DreamZero's recorded interface returns 24 actions but executes eight; predictions beyond those eight are not observations of the actions that occurred.
4. The execution frame comes from the same camera and matching physical time, with known capture rate and any dropped frames. Do not compare a task-end image or post-reset image with a next-chunk prediction.
5. Object localization is sufficiently reproducible to resolve the movement at the exposed horizon. Use duplicate independent labels to estimate measurement noise.
6. The input/output repeatability checks identify effective seed behavior. Existing fixed-observation repeat and prompt-change probes are useful controls but do not identify every source of closed-loop variation.

Use one isolated DreamZero temporal session per server unless session isolation
has been demonstrated. The archived integration describes a prior multi-client
global-state incident with excluded attempts. Distinct client IDs alone do not
establish independent recurrent context. This is a new-run qualification rule,
not a claim that the retained historical cohort is contaminated.

Choose a shared physical horizon across models only when both actually support it. Otherwise run separate within-model studies at their native qualified horizons, with no cross-model accuracy ranking.

**If neither interface supports this comparison, stop this prediction study. If one does, prioritize one sound model study. If motion is too small to resolve, document that the interface cannot support this question at the current execution horizon; do not extend the action chunk merely to manufacture an evaluable forecast.** A changed chunk horizon would be a separate controller intervention.

A short future is not a forecast of the entire 450-action task. Use the term failure prediction only for events observable within that future, or for a separately evaluated prospective episode-level predictor.

## 4. New study and measurements

### Layouts and execution

Use several independent base arrangements, each paired with its reflection. Both commands begin from matched states within an arrangement. Sample coordinates within the physically qualified workspace, checking visibility, no initial collisions, and initially false task-success predicates. Accept/reject scenes without looking at target-model success or forecast quality; preserve the candidate list and rejection reasons. A scripted feasibility check is permissible as fixture qualification, never as coaching during a model episode.

First use **four development layout pairs**, separate from the historical scenes and confirmation scenes. Freeze geometry rules, horizon, annotation procedure, temporal sampling, minimum resolvable movement, analysis and final sample size after development and before any confirmation outcomes are inspected.

For new diagnostic executions, prefer the 450-action cap with goal-based termination disabled: continue the model after the first success and record goal attainment as an event. Observation/evaluation times must not be selected by which requested direction succeeds first. Preserve the historical success predicate when logging first attainment, while clearly disclosing the changed diagnostic stopping policy. These runs are not pooled with historical stopped episodes.

Record physical durations as well as action counts. Do not imply equal time or compute across models from the shared 450-action cap. Unsafe simulator states, crashes and truncations retain explicit reasons; do not compare unexecuted predictions with frozen states.

### Primary measurement

Use the same object definition on conditioning, generated and executed images. Let r0, rhat and r be the cube-minus-bowl image-plane position before acting, in the forecast, and after the matching executed prefix. Normalize both image coordinates by the image diagonal.

Per aligned sample:

**forecast skill = distance(r, r0) − distance(r, rhat).**

Positive skill means the generated future is closer to the actual outcome than predicting no change. This is a projected spatial measurement, not 3D grasp, contact, release or task success.

Use persistence as the primary baseline. Add constant velocity if preceding observations with verified timing are available to both methods; use only past observations, never a fitted velocity from the future. At the initial request with no history, define constant velocity as persistence rather than dropping the case.

The intended estimand gives equal weight to the four conditions in each layout
pair and equal weight to independent layout pairs. Average observable sampled
requests within each episode. If any of the four episodes has zero observable
comparisons, that layout's continuous four-condition mean is undefined: do not
renormalize onto the remaining conditions or assign zero. Report the continuous
estimate for fully represented layout pairs explicitly as a conditional estimate,
with the number contributing out of the number planned. Report missing conditions
and layouts alongside it, and calculate the win-indicator bounds below over the
full planned equal-condition layout structure. Do not present the conditional
estimate as performance on all sampled scenes. Estimate uncertainty by resampling
whole layout pairs, preserving their conditions and nested repeats. Report each
model separately. Frames and action requests are not independent scenes.

### Distinctive intervention analysis

Predefine two complementary checks:

- **Does the forecast track the layout effect?** Define actual relative-motion magnitude q = distance(r, r0) and predicted magnitude qhat = distance(rhat, r0) at the same fixed qualified horizon. Within each episode take the equally weighted mean over the same sampled, mutually observable requests. For each command and layout pair, compute actual delta = mean(q_reflected) - mean(q_original), predicted delta = mean(qhat_reflected) - mean(qhat_original), and their discrepancy. Show these paired values and their distribution across layouts. Use only fully represented condition pairs for the continuous contrast and report its coverage. There is no duration weighting: the sampling target is the uniform distribution of qualified request starts within an episode. Do not interpret agreement of these average magnitudes as correct individual trajectories; wrong directions and canceling errors are checked by the primary per-request position error. Separately inspect cube and bowl motion so a changed relation is not automatically attributed to cube transport.
- **Does it depict movement on stalled execution prefixes?** Compare predicted and actual displacement where executed movement is below the localization-noise threshold frozen during development. Report stationary and moving cases separately, as well as the overall result.

These analyses describe paired condition effects on the model's own experienced states; they do not isolate a particular neural mechanism. Opposing commands, scene intervention and a common recording protocol make this more informative than a generic video-quality score.

### Annotation, sampling and missingness

Save all recordings. For a bounded annotation workload, select up to four eligible requests per episode using a deterministic uniform draw from metadata, before image labels are viewed. Eligibility depends only on verified timing, executed action prefix and camera identity, never object visibility or forecast quality. Freeze the draw seed before labels; a selected image with missing/unclear objects stays selected and cannot be replaced. Use the same selected times for model and baselines. Retain the full request inventory, inclusion probabilities and episodes with zero eligible requests. Any later motion-enriched sample is a separately weighted secondary analysis, not a replacement for the primary draw.

Two independent raters label each selected image without its counterpart, model identity, requested command or success outcome. Freeze the landmark, occlusion and ambiguity rules on development data. Report initial disagreement and adjudication; do not present an unvalidated VLM scorer as ground truth. Simulator coordinates support execution-side checks, but may not be substituted for differently defined image landmarks.

For each condition report timing eligibility, localization coverage, unclear/generated-object cases, zero-eligible episodes and censoring reasons. The continuous skill estimate is conditional on observable matched cases. Do not invent coordinates for a missing or off-screen object. As a sensitivity check, also report the fraction of scheduled comparisons where the forecast beats persistence, assigning all unresolved comparisons first as non-wins and then as wins. Keep equality separate from a strict improvement. Compute lower/upper win means within each episode and preserve the full four-condition and layout weighting; an episode with no eligible comparison contributes an uncertainty interval [0, 1], not a fabricated measurement. These bounds describe a win indicator, not continuous localization accuracy. If execution itself is unobservable, state that no accuracy conclusion is available for those cases. Avoid a general accuracy claim if missingness could change the conclusion.

### Sample size and budget

| Stage | Proposed episodes for two usable models | Role |
| --- | ---: | --- |
| Recover existing examples | 0 new | Test whether archived material supports alignment |
| Recording pilot, if required | 8 | Qualify new instrumentation/configurations |
| Development: 4 layout pairs × 4 conditions × 2 models | 32 | Fix measurement and design |
| Bounded confirmation: 12 layout pairs × 4 × 2 | 96 | Narrow workshop study |
| Preferred confirmation: 24 layout pairs × 4 × 2 | 192 | Better scene coverage |

Choose either confirmation size once before running it; do not run 96 and then add more until a desired result appears. The preferred path is at most 232 new behavioral episodes before any separately scoped repeatability or intervention extension. One usable model halves development and confirmation counts; the initial eight-episode pilot may already have tested both.

These are planning targets, not a power calculation or GPU-hour estimate. Use development estimates of runtime, annotation coverage and between-layout variation to assess precision and cost. Default to 24 confirmation pairs when feasible. If only 12 fit the budget, narrow the claims. A second effective sampling seed can be added only in the frozen design and only for qualified stochastic interfaces; new layouts take priority over nominal fixed-noise repetitions.

With four selected requests per episode, preferred confirmation has at most 768 image triplets, or 4,608 individual image judgments from two raters before adjudication. Reuse identical images and their labels where valid. Measure annotation throughput during development and establish that budget alongside inference/storage cost.

Do not pad an ineligible stochastic configuration with repeated labels described as new model-noise draws. If closed-loop variability remains unexplained after trace comparison, run a separately bounded repeated-reset diagnostic before confirmation.

## 5. Decision criteria and paper deliverables

Decide whether the measurement is usable before examining confirmation effects. Keep the study running to its frozen completion regardless of whether interim examples look promising.

- Aligned, measurable forecasts and adequate precision: complete the planned study and report either sign of the result.
- Forecasts anticipate changed execution: describe what they capture and where they fail; do not infer that the system uses this prediction causally.
- Forecasts depict unrealized progress: quantify frequency, magnitude, coverage and scene consistency; do not select only striking videos.
- Inconsistent or imprecise results: report the uncertainty; do not infer equivalence or manufacture an award narrative.
- Failed alignment or insufficient observable motion: narrow or stop the prediction claim, preserving the repaired behavioral study.

For a four-page paper, target three figures and one compact results table:

1. Actual scene images and the prediction/action/execution timing diagram.
2. Forecast skill and predicted-versus-executed layout effects across independent scenes.
3. Representative correctly predicted and mismatched executions, chosen by a declared rule.
4. Table of configurations, scene counts, horizon, coverage, baseline errors and uncertainty.

The previous LEFT/RIGHT success table becomes motivation or supporting analysis. Include reproducibility details and artifacts without putting administrative language in the scientific argument.

Related work to address:

- [MESA](https://www.pair.toronto.edu/MESA/static/pdfs/draft.pdf): separates language following from completing a task; the new question concerns exposed future predictions.
- [LIBERO-CF](https://arxiv.org/abs/2602.17659): counterfactual instructions in fixed scenes; explain the role of executed-action-aligned forecasts.
- [WorldArena](https://arxiv.org/abs/2602.08971): functional evaluation of world models; distinguish prediction of this model's own executed actions under a paired scene intervention.
- [DreamZero](https://arxiv.org/abs/2602.15922): joint video/action generation and closed-loop execution; make configuration differences explicit.

The best workshop fit becomes Motion 4 (trustworthy model-based evaluation), with Motion 6 (limitations of benchmark scores) as the motivation. The [workshop page](https://do-robots-need-world-models.github.io/) currently leaves dates and submission details unconfirmed. Recheck those before a submission decision.

## 6. Execution checklist and files

All new work stays under workshops/corl2026_world_models/. Historical V2/V3 protocols, queues and artifacts are immutable. The following paths are relative to that workshop directory and describe future work unless already present.

### A. Correct and reproduce the current evidence

Files: analysis/extract_paper_results.py, analysis/analyze_stopping_controls.py (new), tests/test_paper_results.py, tests/test_stopping_controls.py (new), docs/MODEL_PROVENANCE.md, paper/main.tex and README.md.

- [ ] Preserve full DreamZero runtime fields and the exact fixed-observation probe caveat in extraction; test that dropping or changing the guidance identity is rejected.
- [ ] Reproduce the Nano pre-termination result from retained step rows, with common-step definition, physical-time caveat and eligibility counts.
- [ ] Include a synthetic instruction-independent trajectory that visits both sides: goal-stopped endpoints must not be accepted as proof of directional response.
- [ ] Report fixed-time historical coverage and cube/bowl contributions; do not select a flattering late horizon.
- [ ] Update model labels, expected reflection behavior, limitations and related work. Preserve the old PDF version through Git.
- [ ] Rebuild using the existing build_package.py entry point and check changed figures/pages.

### B. Qualify archive alignment

Files: analysis/align_v3_recordings.py (new), tests/test_v3_alignment.py (new), results/alignment_examples.json (new).

Read-only source reuse at the pinned repository revision: Nano
experiments/v3/cosmos_nano_phase_b/live_client.py and compile_cell.py; DreamZero
experiments/v3/dreamzero_phase_b/robolab_bridge.py, recover_future_trace.py and
experiments/v3/dreamzero_droid/future_retention.py. DreamZero's aggregate
source_behavioral_jsonl identity points to the fuller raw trace. Preserve that
identity when recovering it. The paths here refer to the parent study repository,
not new workshop-local files.

- [ ] Read original episode/media identities from the pinned V3 aggregates and recover two complete episodes from an authorized source.
- [ ] Reuse the normalized checks in analysis/evidence_audit.py without treating them as proof of raw timing.
- [ ] Implement the documented frame/action/physical-time mapping for each qualified interface.
- [ ] Reject synthetic examples with swapped cameras, incompatible object definitions, target times after replanning, dropped execution frames or post-reset endpoints.
- [ ] Produce one visually inspectable aligned request from each usable model, or a specific unsupported/missing-data result. Store raw media outside Git.

### C. Freeze and qualify new diagnostic recordings

Files: experiments/forecast_layout/recording_contract.json (new), experiments/forecast_layout/recording_adapter.py (new), tests/test_forecast_recording.py (new), results/recording_pilot.json (new).

- [ ] Reuse the pinned model integration after revalidating its runtime on the selected machine.
- [ ] Document effective guidance/noise settings, decoded-frame timing, action horizon, crop/camera identity, capture synchronization and goal-independent stopping.
- [ ] Verify decoding preserves proposed actions using matched inputs and effective noise, with exact comparison where deterministic.
- [ ] Run only the bounded recording pilot needed to qualify the new setup. Keep any infrastructure attempts separate.
- [ ] Report measured runtime/storage and unresolved repeatability before proposing confirmation costs.

### D. Freeze development and confirmation

Files: experiments/forecast_layout/design.json (new), experiments/forecast_layout/cells.jsonl (new), analysis/analyze_forecast_layout.py (new), tests/test_forecast_layout.py (new).

For fixture construction, reuse the pinned parent repository's
experiments/v3/cosmos_nano_phase_b/fixture_candidate.py and its model-blind
fixture checks. The Phase-E geometry and observation-replay helpers can support
qualification, but its existing symmetry generator is not a ready-made sampler
of independent layouts. Do not edit or activate those old queues.

- [ ] Generate physically valid independent layout pairs without target-model outcome selection.
- [ ] Use development only to finalize the measurement, sampling, noise threshold, scene count and runtime choice.
- [ ] Freeze the confirmation manifest, ordering, seed semantics, primary estimator and secondary contrasts before its first model request.
- [ ] Verify the analyzer preserves layout clusters, equal episode weights, all paired conditions and the missingness ledger.
- [ ] Include synthetic checks for perfect prediction, persistence, stationary execution with imagined motion, incomplete layout blocks and duplicated records.
- [ ] Run the fixed study; analyze only after the planned cohort is complete or report a clearly disclosed interrupted cohort.

### E. Rewrite around the measured finding and review again

Files: paper/main.tex, paper/references.bib, analysis/make_paper_figures.py, docs/DELIVERY_QA.md and the new compact result files.

- [ ] Build tables and figures from completed results.
- [ ] Choose the title and abstract to match the observed finding and qualified horizon.
- [ ] Recheck novelty, statistical scope, selection rules and all configuration labels.
- [ ] Compile and inspect the four-page paper, then obtain another skeptical review.
- [ ] Decide on submission only after authorship, eligibility and current venue requirements are settled.

No inference, remote machine access, annotation collection or manuscript revision was performed in this planning turn. GPU workstation use still requires the user's explicit machine instruction. A concrete compute budget follows the recording check; the counts above are proposals, not permission to launch a full queue.
