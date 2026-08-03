# Does the world model listen?

## A matched VLA/WAM study of direct language steerability

*Ali Adeeb Abbas · Senior Scientist, General Motors*
*Personal research; views are my own. Evidence package closed 2 August 2026.*

![Complete 160-episode steerability scorecard for the two tested checkpoints.](../artifacts/vla_wam_shared_v1/trajectory_evidence/social/steerability_scorecard_1600x900.png)

---

A model that changes its output when the instruction changes is not necessarily
a model that follows the instruction. Sampling noise moves an action tensor. A
swapped noun moves a generated video more than a swapped spatial relation does.
A first action chunk can point the right way and still wash out over a dozen
closed-loop replans. So the measurement that matters is not sensitivity but
compliance:

> With the observation, physical history, sampling schedule, task geometry and
> action schema held fixed, does changing the command cause the requested
> outcome?

This study answers that question for two public checkpoints under one matched
protocol, and asks what changes when the low-level policy is a world-action
model (WAM) that emits video as well as actions. The command taxonomy and
evaluation philosophy follow Chen et al.'s [Steerable Vision-Language-Action
Policies for Embodied Reasoning and Hierarchical
Control](https://arxiv.org/abs/2602.13193); the experimental design does not
reproduce their training intervention.

Full protocol, metric definitions, amendments, per-model notes and operational
measurements are in the companion
[**Protocol and limitations appendix**](VLA_WAM_STEERABILITY_APPENDIX.md).

## 1. Summary of results

**Neither tested checkpoint is robustly steerable across direction and
wording.** NVIDIA's Cosmos3 Edge DROID WAM completed 58/80 direct-language
episodes and Physical Intelligence's π0.5 DROID VLA completed 25/80. Both
totals conceal large handedness and syntax effects. π0.5 succeeded on 3/40 LEFT
requests versus 22/40 RIGHT; Cosmos on 22/40 LEFT versus 36/40 RIGHT, and fell
from 19/20 under declarative goal language to 10/20 when the same goal was
paired with a negated distractor.

**The dominant failure mode is grounded placement, not manipulation.** Of 77
failures, 53 picked up the correct cube and never entered the requested goal
region; 23 produced no verified cube interaction; one ended geometrically
inside the goal without satisfying the terminal release predicate. No rollout
entered the goal and later lost it.

**Sensitivity is real and is not the same as compliance.** On byte-identical
input and sampling seed, both endpoints reproduced a repeated request exactly
(action RMS 0.0; Cosmos future-video MAE 0.0) and both changed their output
when LEFT became RIGHT. For π0.5, however, moving the target relation from
before to after a negated clause changed actions *more* than swapping the
relation itself did, in every prompt family.

**Generated video is informative but is not yet a dependable semantic
monitor.** A prompt-blind visual scorer could label 421 of 752 Cosmos
action/future horizons. Among those, imagination and execution agreed in
413/421 (98.1%) — but the evaluator abstained on 72 of the 97 horizons in which
execution actually reached the requested relation. Only 25/97 positive
execution events support any precision or recall claim.

The result is a case study of two public systems and a reusable protocol, not a
model-class leaderboard. One checkpoint cannot represent a class.

## 2. What "steerable" has to mean

Six distinct claims are routinely collapsed into one word:

| Level | Claim | Minimum evidence |
| --- | --- | --- |
| 1 | Repeatability | Identical input and sampling state reproduce identical output |
| 2 | Sensitivity | A command change exceeds same-command noise |
| 3 | Semantic direction | The change points toward the requested outcome |
| 4 | Task competence | The checkpoint solves its released native task |
| 5 | Closed-loop control | A matched counterfactual completes the requested goal |
| 6 | Robust steerability | Control survives directions, seeds, scenes, wordings, objects and abstraction levels |

A pairwise-distance heatmap over prompts reaches level 2. A publishable claim
about steerability needs levels 3–5; a general claim needs level 6.

The distinction is stricter for a WAM than for a VLA. A VLA has one observable
obligation: produce actions that achieve the command. A WAM makes two coupled
promises — act, and imagine the future associated with the action — so it can
fail in four semantically different ways:

| Imagined future | Executed action horizon | Interpretation |
| --- | --- | --- |
| requested | requested | action and world prediction agree |
| requested | not requested | correct imagination, action does not realise it |
| not requested | requested | policy succeeds despite an inconsistent world prediction |
| not requested | not requested | neither output follows the requested relation |

Pixel MAE cannot separate these quadrants. Distinguishing them is the most
WAM-specific measurement in this study.

## 3. The matched comparison

Public robot models differ in training corpus, embodiment, action coordinates,
camera layout, horizon, parameter count, future representation and simulator.
Running each model's preferred benchmark and tabulating the percentages would
confound nearly everything. The shared grid instead fixes:

- the same DROID joint-position-plus-gripper action schema;
- the same simulator, task objects, camera geometry and 450-step limit;
- the same neutral object arrangement, in which neither LEFT nor RIGHT is
  initially true;
- the same ten policy-sampling seeds per prompt condition, with request seed
  `episode_seed × 1000 + replan_index`;
- the same success predicate, progression rubric and evidence compiler.

LEFT and RIGHT task variants differ only in the requested predicate and the
instruction. Both checkpoints retain their native action horizons — 15 for
π0.5, 32 for Cosmos — so that no horizon-matching distribution shift is
introduced to support a controller outside the question. Every episode uses one
episode-static direct task prompt: no subtask coach, no privileged oracle, no
mid-rollout prompt switching.

The registered grid is 2 checkpoints × 2 wordings × 2 directions × 10 seeds =
80 episodes. A separately disclosed, prospectively frozen post-interim tier
adds 80 more with declarative goal language and a contrastive instruction
containing both direction words. The four wordings:

```text
Put the rubiks cube to the {left|right} of the bowl                        (canonical)
Put the cube {left|right} of the bowl                                      (short)
The rubiks cube should end up to the {left|right} of the bowl              (declarative)
Put the rubiks cube to the {left|right} of the bowl, not to the {right|left} of the bowl   (contrastive)
```

The short wording drops "rubiks" and "to the", testing dependence on a
training-like caption. The declarative wording asks whether an end-state
description grounds like an imperative. The contrastive wording is the hardest
semantic test: both direction tokens are present and one relation is explicitly
negated, so a bag-of-direction-words response can move while a steerable policy
must respect the scope.

## 4. Closed-loop success

![Closed-loop success by checkpoint, wording and requested direction.](../artifacts/vla_wam_shared_v1/final_evidence/direct_language_success_with_intervals.png)

Across the full grid, π0.5 succeeded in **25/80** episodes and Cosmos in
**58/80**. These are descriptive checkpoint results at different native
horizons, not a class ranking. The direction breakdown is more informative than
either total: π0.5 was 3/40 LEFT versus 22/40 RIGHT; Cosmos was 22/40 LEFT
versus 36/40 RIGHT.

Individual cells show why "supports language" is too coarse a description.
Cosmos ranged from 10/10 (declarative LEFT, short RIGHT) to 1/10 (contrastive
LEFT). π0.5 ranged from 8/10 (canonical RIGHT) to 0/10 (canonical and
declarative LEFT). Every cell has ten trials, so the Beta(1,1) credible
intervals are wide by construction; the signal worth taking seriously is the
repeated, paired structure of the asymmetry, not a point estimate of its
population rate.

![Wording robustness matrix.](../artifacts/vla_wam_shared_v1/final_evidence/direct_prompt_robustness.png)

The stress tier separates two different failures. π0.5 retained its RIGHT
preference under declarative language but lost it under the contrastive form,
finishing at 2/10 in both directions. Cosmos handled the declarative form
almost perfectly (19/20), then fell to 10/20 once the same goals carried a
negated opposite. All nine discordant same-seed declarative/contrastive pairs
favoured declarative (two-sided exact McNemar *p* = 0.0039). Within Cosmos
contrastive trials, eight discordant direction pairs favoured RIGHT and none
favoured LEFT (*p* = 0.0078). These tests are exploratory and uncorrected. The
model did not simply become noisier: a scoped language change exposed a
strongly directional failure.

### Progression, not just termination

The paper's two-item spatial rubric awards persistent credit for picking up the
correct object and a second point for satisfying the requested placement.
Reporting it beside terminal success separates manipulation competence from
language-grounded placement:

| Checkpoint | Request | Correct pickup | Terminal placement | Mean two-item progression |
| --- | --- | ---: | ---: | ---: |
| π0.5 (VLA) | LEFT | 20/40 | 3/40 | 28.8% |
| π0.5 (VLA) | RIGHT | 37/40 | 22/40 | 73.8% |
| Cosmos (WAM) | LEFT | 39/40 | 22/40 | 76.2% |
| Cosmos (WAM) | RIGHT | 40/40 | 36/40 | 95.0% |

Pooled across directions, mean progression was 51.3% for π0.5 and 85.6% for
Cosmos. That does not mean Cosmos placed the cube correctly 85.6% of the time:
it earned the persistent pickup half-credit in 79/80 episodes. Strict
pick-then-place means were 51.3% and 85.0%; relation-only means were 51.3% and
86.3%. Their closeness confirms the distinction is requested placement versus
no requested placement, not an accounting artefact.

## 5. The failure is geometric

![Signed final cube offset toward the requested side.](../artifacts/vla_wam_shared_v1/final_evidence/direct_language_requested_side_offsets.png)

Binary success depends on a termination predicate, so the signed endpoint is a
useful independent check. Positive always means "toward the requested side".
Cosmos averaged +7.7 cm for canonical LEFT and +14.3 cm for declarative LEFT,
but **−5.9 cm** for short LEFT and **−9.8 cm** for contrastive LEFT: in those
two conditions the mean endpoint was on the opposite side. Its RIGHT means were
+34.7, +38.1, +36.5 and +31.8 cm. π0.5's LEFT means were all weakly positive
(+1.4 to +3.6 cm); its first three RIGHT means were +7.1 to +13.8 cm, and
contrastive RIGHT collapsed to +1.0 cm. The failures are geometric, not
threshold artefacts.

A score says whether a rollout ended correctly; it does not show how the
rollout failed. For every registered episode I transform the saved cube root
pose into the robot frame, place the bowl at the origin, and draw the full
executed path. Robot-left is left on the page. The shaded 45° cone is the goal
region; the dashed arrow is an illustrative direct route only — it is not an
oracle trajectory and enters no metric.

![Same scene and seed under declarative and contrastive language.](../artifacts/vla_wam_shared_v1/trajectory_evidence/social/first_seed_stress_landscape_1600x900.png)

![Every executed path and endpoint.](../artifacts/vla_wam_shared_v1/trajectory_evidence/blog/all_executed_paths_and_endpoints.png)

Each episode then receives one mutually exclusive terminal diagnosis: no cube
interaction; interaction without verified pickup; pickup without ever entering
the goal; entry followed by losing the relation; ending in the goal without
satisfying terminal release; or success.

![Terminal anatomy of every episode.](../artifacts/vla_wam_shared_v1/trajectory_evidence/blog/failure_progress_anatomy.png)

Of 77 failures, 53 picked up the cube and never entered the requested goal, 23
never produced a verified interaction, and one ended inside the goal without
satisfying the full terminal condition. That distribution rules out "the robot
cannot grasp" as a sufficient explanation: most failures progressed past pickup
and then placed or retained the cube in the wrong region. The [complete
filterable
gallery](../artifacts/vla_wam_shared_v1/trajectory_evidence/gallery/index.html)
exposes every seed, instruction, endpoint class, event step, source path and
rendered panel.

## 6. The byte-identical intervention

Closed-loop episodes share one exact physical reset fingerprint across the
robot and rigid objects, but that does **not** produce byte-identical first
observations. Objects settle by millimetres before the first recorded action,
and the realtime renderer is not deterministic.

![Conditioning-image variation across exact physical resets.](../artifacts/vla_wam_shared_v1/final_evidence/cosmos_conditioning_image_variation.png)

Because same-prompt resets already differ, the closed-loop opposite-prompt
action distance carries prompt, settling and renderer variation together. On
the first chunk, opposite-prompt separation was *smaller* than the same-prompt
baseline in every condition (effect-to-baseline ratios 0.118–0.373 for π0.5 and
0.409–0.622 for Cosmos). That does not show language had no effect; it shows
this particular contrast cannot isolate one. It stays a diagnostic. The causal
intervention is the fixed-observation probe, which holds input pixels, robot
state and sampling seed byte-for-byte constant.

![Exact-input direct task probe.](../artifacts/vla_wam_shared_v1/final_evidence/direct_task_exact_probe.png)

Both endpoints reproduced the repeated canonical request exactly: action RMS
0.0 for each model, and Cosmos future-video MAE also 0.0. With identical
observation bytes, changing LEFT to RIGHT produced nonzero action RMS for every
wording — establishing deterministic prompt sensitivity, and nothing more.

The word-order control is the warning. Moving the same target relation from
before to after the negated distractor changed π0.5 actions by 0.0077 (LEFT
target) and 0.0070 (RIGHT), larger than its LEFT-versus-RIGHT effect in every
prompt family (0.0024–0.0057). Cosmos's order effects, 0.0175 and 0.0122, were
comparable to its relation effects (0.0116–0.0201). Both checkpoints therefore
encode lexical scope and order in a way that can be at least as consequential
as the requested relation itself.

![Six-style fixed-observation command probe.](../artifacts/vla_wam_shared_v1/final_evidence/command_probe_action_sensitivity.png)

Extending the probe to all six command styles from the paper — task, subtask,
atomic motion, gripper trace, grounded point, combination — plus four negative
controls shows that both released interfaces react strongly to far more than
task captions, and that the largest tensor movement is not attached to the
richest semantic command. An unrelated task moved Cosmos further than any
spatial command did. These are interface diagnostics, not evidence of
obedience.

That limitation is itself instructive: a rich command interface requires a
metric suite rich enough to distinguish end-effector motion, grasp state,
object identity, traces, points and full task success. One left/right object
predicate cannot stand in for six command styles, so the semantic scorer below
is applied only where the cube–bowl relation is actually the command's target.

## 7. Does the WAM imagine what it executes?

Cosmos emits a 33-frame imagined video with each 32-action chunk. Frame 0 is
the conditioning image and never counts as forecast evidence; frames 8, 16, 24
and 32 are scored. A local
[Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) model
sees one camera panel at a time and receives only an object-localisation
request — find the multicoloured cube and the red bowl. It never sees the
policy instruction or the requested direction. The target relation comes from
the authoritative simulator task identity. Both over-shoulder cameras must
agree on the categorical relation and their reconstructed positions must agree
within a frozen 0.20 m threshold, otherwise the chunk abstains. This is
strictly offline measurement; the localiser never selects a command or feeds
anything back to either policy.

![Imagination versus execution across all 752 replan chunks.](../artifacts/vla_wam_shared_v1/final_evidence/cosmos_imagination_execution_quadrants.png)

Of 752 chunks, 421 received a certain label: 22 both imagined and executed the
request, 5 imagined only, 3 executed only, and 391 neither. The remaining 331
abstained. All 80 episodes had at least one certain chunk, but only 22 had any
certainly imagined-positive chunk, while 58 had at least one executed-positive
horizon.

![Frozen-order generated-future examples for each category.](../artifacts/vla_wam_shared_v1/semantic_future_visualization/blog/selected_semantic_future_examples.png)

Among certain chunks, imagination and execution agreed in 413/421 (98.1%). When
the future made a positive prediction, execution agreed in 22/27 (81.5%); among
executed-positive horizons with a certain future, 22/25 were imagined positive
(88.0%). Those conditionals sound strong until coverage is restored: execution
reached the requested relation in 97 chunks and the scorer abstained on 72 of
them. Only **25/97** positive execution events support either figure. The 391
neutral/neutral chunks dominate the aggregate agreement, because a full
pick-and-place usually cannot complete inside an early 32-action horizon.

![Semantic scorer threshold sensitivity.](../artifacts/vla_wam_shared_v1/final_evidence/semantic_threshold_sensitivity.png)

Replaying the labels at stricter thresholds reinforces the limitation rather
than rescuing the headline. At 0.10, 0.15 and the frozen 0.20 m cross-camera
thresholds, overall coverage was 5.5%, 34.8% and 56.0%; coverage on the 97
executed-positive horizons was 1.0%, 10.3% and 25.8%. Agreement among surviving
chunks stayed at 97.6%, 99.2% and 98.1%. A stricter evaluator preserves an
excellent-looking agreement rate by discarding almost every positive event.

The directional pattern is nevertheless real. Contrastive LEFT produced zero
certainly imagined-positive chunks across 146 replans while contrastive RIGHT
produced four; closed-loop success in the same conditions was 1/10 versus 9/10.
Conversely, declarative LEFT succeeded in all ten episodes but produced only one
episode with any certainly imagined-positive chunk. Generated video exposes some
genuine directional structure and is too short-horizon and too often unscorable
to serve as a reliable success forecast.

A frozen 24-sheet human audit reviewed 254 chunks and 1,016 frames. Cube or
bowl markers visibly jumped to the robot, table, background or a distractor
object under small-object and occluded futures. Cross-camera checks caught many
such cases and sampled certain directional labels were visually credible, but
two-camera agreement cannot guarantee object identity. The outcome is a
[qualified audit
pass](../artifacts/vla_wam_shared_v1/semantic_confirmation_audit.md), not an
evaluator-accuracy claim.

## 8. Other world-action models, kept separate

The shared-grid intervals contain only π0.5 and Cosmos. Earlier experiments on
other WAMs answer useful engineering questions but were not generated by the
frozen grid, so they are reported as a clearly separated retrospective tier and
never pooled.

| Model | Deployment-relevant scale | Evidence here | Verdict under this protocol |
| --- | --- | --- | --- |
| Light-WAM | 0.44B trainable + frozen 1.3B Wan backbone | release review only | highest-priority lightweight replication |
| UVA | ≈0.5B | early pairwise heatmap only | not established |
| Efficient-WAM-RT | 1B | 42 retrospective closed-loop episodes | usable causal-intervention core; strongly asymmetric |
| Fast-WAM | not normalised here | implementation audit + six-seed gate | language signal exists after repair; not robust |
| Cosmos3 Edge DROID | 4B | full direct-language grid | shared-benchmark WAM |
| LingBot-VA | 5.09B | retrospective native/swap gate | strong future-latent substrate; too slow for a core |
| DreamZero | ≈14B | setup and runtime experience only | later large-model confirmation |

Parameter labels are not made artificially comparable: "trainable" excludes
frozen inference weights, active parameters can exclude routed capacity, and a
download size is not a parameter count. The behavioural column, not the
smallest number in column two, drives the recommendation. Three of these models
are marked *not measured under this protocol*, which is not the same as zero
success. Per-model detail is in the
[appendix](VLA_WAM_STEERABILITY_APPENDIX.md#5-retrospective-wam-tier).

The practical recommendation has two layers. **Efficient-WAM-RT** is the most
productive rapid-intervention core available locally: roughly 0.137 s per warm
action chunk, and the only checkpoint here for which I have byte-identical
after-grasp command-switch evidence. **Cosmos** is the slower WAM cross-check
that supplies inspectable generated futures in a matched DROID setup, and π0.5
is the action-only VLA control. Light-WAM is the next checkpoint worth bringing
through the same protocol. None of these is ready to be treated as a reliable
free-form language-control layer, and none of these roles is a deployment
recommendation.

## 9. What the evidence does and does not establish

There is real capability here and the negative result should not erase it. Both
endpoints reproduced an identical request exactly; both changed their actions
under byte-identical LEFT/RIGHT interventions; Cosmos also changed its
generated future. Both solved substantial fractions of the released DROID task.
Cosmos's 19/20 declarative result is particularly useful: an end-state
description can control this checkpoint without an imperative verb or a
high-level coach.

The WAM interface also adds evidence that action-only evaluation cannot
provide. It lets an experiment separate a world prediction that anticipates the
requested relation from an action chunk that actually reaches it, and the saved
frames, overlays and executed state stay inspectable even when the localiser
abstains. That makes WAMs unusually promising substrates for monitoring,
planning and causal-intervention research — provided future quality and
action/future consistency are measured rather than assumed.

The study does **not** establish:

- that WAMs are more or less steerable than VLAs as a class (one shared-grid
  checkpoint per class);
- a reproduction of the paper's training intervention or its four
  generalisation splits;
- cross-scene or cross-object robustness;
- a clean separation of language grounding from learned motor or workspace
  handedness — the matched prompts, endpoint paths and word-order probe expose
  the asymmetry without fully decomposing it;
- fixed-observation distance as a success metric;
- closed-loop first-action separation as a pure language effect;
- an imagined pixel change as semantic compliance;
- anything about privileged oracles, subtask coaches or learned embodied
  reasoners — every analysed episode uses direct task language.

Scorer abstentions, setup failures, thermal exclusions and excluded batches are
all reported rather than dropped.

## 10. Design implications

The failures point to a concrete training recipe.

1. **Relabel trajectories at all six abstraction levels.** Task captions teach a
   policy to recognise familiar requests, not to expose a reusable control
   interface.
2. **Add paired counterfactuals.** The same scene and physical prefix should be
   paired with opposing spatial, object and gripper commands, and training
   should penalise an unchanged action/future when the requested predicate
   changes.
3. **Give grounding a structured interface.** Rendering `[x,y]` as text is easy
   and brittle. Points and traces should enter through a calibrated spatial
   channel with camera identity and coordinate convention explicit.
4. **Couple action and future predicates.** A WAM should be penalised when its
   imagined semantic state and the state reached by executing its action chunk
   disagree. Pixel reconstruction does not enforce this.
5. **Train command switching from identical prefixes.** Redirection from a
   shared physical prefix is more causal and more demanding than choosing a
   prompt at reset.
6. **Make the direct task interface robust before adding a high-level
   controller.** Declarative goals, negation, lexical distractors and
   equivalent paraphrases should map to the same objective. A coach can mask
   that defect; it cannot repair a model that does not ground the request.

The next version of the benchmark should keep the neutral-start seed grid and
add multiple object and reference pairs, mirror-reflected scene and robot-frame
controls to separate lexical direction from workspace handedness, all four
generalisation splits, task-specific semantic scorers for gripper displacement,
grasp/release, object identity, points and traces, counterbalanced contrastive
word order, and additional WAMs whenever their action schema can be matched
honestly. The core unit should remain a matched causal pair.

## 11. Assessment

The answer to the title is **yes, selectively and unreliably**. Both tested
checkpoints listen at the level of deterministic output sensitivity. Neither
provides robust semantic control across direction, wording and negation. Cosmos
is the stronger controller in this one matched DROID case, but its 58/80 total
coexists with a 1/10 contrastive-LEFT collapse; π0.5's 25/80 total is dominated
by a 22/40 RIGHT versus 3/40 LEFT split. A single aggregate score would conceal
the central result.

The methodological point generalises beyond these two checkpoints. Expected
paths and actual paths, successes and failures, imagined predicates and
executed predicates, prompt order, direction, abstentions and operating cost
all belong in the same evidence package. Once they are shown together, "the
output changed" stops looking like a result — and steerability becomes a claim
one can try to falsify.

---

## Reproducibility

The complete registered package is under `artifacts/vla_wam_shared_v1/`. The
machine-readable join is `final_evidence/compiled_evidence.json`; the human map
is `final_evidence/EVIDENCE_INDEX.md`. Every figure above is rendered from the
compiled evidence by `tools/render_study_figures.py`, which performs no
inference and derives no new statistic, so figures can be restyled without
re-running the study. Protocol files, amendments, implementation files and the
full operational-cost measurements are itemised in the
[appendix](VLA_WAM_STEERABILITY_APPENDIX.md).

## Primary external sources

- Chen et al., [*Steerable Vision-Language-Action Policies for Embodied Reasoning and Hierarchical Control*](https://arxiv.org/abs/2602.13193).
- Physical Intelligence, [*π0.5: a Vision-Language-Action Model with Open-World Generalization*](https://arxiv.org/abs/2504.16054); [OpenPI release](https://github.com/Physical-Intelligence/openpi).
- NVIDIA, [Cosmos release](https://github.com/NVIDIA/cosmos); [Cosmos3-Edge-Policy-DROID checkpoint](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID).
- Li et al., [*Efficient-WAM*](https://arxiv.org/abs/2606.10040); [code](https://github.com/jiajun613/Efficient-WAM); [RoboTwin checkpoint](https://huggingface.co/jiajun0613/Efficient-WAM_RoboTwin).
- Yuan et al., [*Fast-WAM*](https://arxiv.org/abs/2603.16666); [code](https://github.com/yuantianyuan01/FastWAM).
- Li et al., [*Causal World Modeling for Robot Control*](https://arxiv.org/abs/2601.21998); [LingBot-VA release](https://github.com/Robbyant/lingbot-va).
- Li et al., [*Unified Video Action Model*](https://arxiv.org/abs/2503.00200); [UVA release](https://github.com/ShuangLI59/unified_video_action).
- Li et al., [*Light-WAM*](https://arxiv.org/abs/2606.08242); [code](https://github.com/L1ziang/Light-WAM); [checkpoints](https://huggingface.co/l1ziang/lightwam-checkpoints).
- Ye et al., [*World Action Models are Zero-shot Policies*](https://arxiv.org/abs/2602.15922); [DreamZero release](https://github.com/dreamzero0/dreamzero).
