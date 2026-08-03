# Protocol and limitations appendix

Companion to [*Does the world model listen?*](VLA_VS_WAM_STEERABILITY_STUDY.md).
This document carries the full protocol, metric definitions, dated amendments,
per-model retrospective notes, operational measurements and evidence map. The
lead essay states the results; this states exactly how they were produced and
what they are not allowed to mean.

**Contents**

1. [Command interface and its provenance](#1-command-interface-and-its-provenance)
2. [Registered design and the disclosed scope change](#2-registered-design-and-the-disclosed-scope-change)
3. [Metrics](#3-metrics)
4. [The prompt-blind semantic future scorer](#4-the-prompt-blind-semantic-future-scorer)
5. [Retrospective WAM tier](#5-retrospective-wam-tier)
6. [Operational cost](#6-operational-cost)
7. [Integrity checks and amendments](#7-integrity-checks-and-amendments)
8. [Evidence map](#8-evidence-map)

---

## 1. Command interface and its provenance

Chen et al. train low-level policies on six command styles:

1. **Task** — complete task language, e.g. "put the carrot in the pot".
2. **Subtask** — a semantic component, e.g. "reach for the carrot".
3. **Atomic motion** — a low-level movement without task semantics, e.g. "move
   left", "open gripper".
4. **Gripper trace** — an image-space sequence of points for the gripper to
   follow.
5. **Point** — a grounded object or interaction location.
6. **Combination** — a hybrid of language, motion, point/trace and gripper
   state.

Their full evaluation contains in-distribution, motion, spatial and semantic
generalisation splits. The main closed-loop metric is success rate; their
multi-step in-context experiment reports task progression as the mean of a
task-specific list of binary rubric items. Those items need not occur in order,
and credit is revoked when a state is undone, except that first-interaction or
pickup credit persists. Their spatial example has two items: pick up the
correct object, then put it down in the correct location. Their learned
embodied reasoner is queried every five environment steps, their off-the-shelf
in-context VLM every twenty, and human-oracle interventions are separated by at
least two seconds. Grounded coordinates are normalised to 0–255.

This study copies the taxonomy and the progression rubric. It does **not**
reproduce the Bridge robot, the training mixture or the four evaluation splits;
it covers one spatial slice. It also deliberately does not copy the learned
reasoner or human-oracle experiments, which answer whether a high-level system
can select useful low-level commands. The question here is narrower: does the
released checkpoint itself ground the user's task language without privileged
state or a coach?

**Coordinate convention.** The paper appendix is internally inconsistent about
the coordinate origin: it defines the first coordinate as the column from the
left and the second as the row from the top, then calls `[0,0]` the top-right.
This benchmark adopts the conventional top-left origin implied by the
row/column definition and records that choice rather than guessing silently.
Point and trace coordinates are projected from simulator ground truth and
normalised to 0–255 before any model request.

## 2. Registered design and the disclosed scope change

The original design was committed before confirmation inference. Its direct
task grid:

```text
2 checkpoints × 2 wordings × 2 directions × 10 seeds = 80 episodes
```

After all 40 Cosmos outcomes in that grid and preliminary five-step evidence
were visible, I removed the privileged controller from the study. History was
not rewritten. The dated scope amendment
(`direct_language_scope_amendment_003.json`) lists every known outcome at the
time and freezes a separate stress tier *before* any of its episodes ran:

```text
2 checkpoints × 2 stress wordings × 2 directions × 10 new seeds = 80 episodes
```

The original canonical/short tier remains confirmatory. The
declarative/contrastive tier is post-interim and is never retroactively
described as part of the original preregistration. Retired five-step outputs
remain in an excluded/supporting ledger and contribute zero analysed episodes.

**Fixed-observation probes.** The six-style probe reuses one hash-pinned
neutral calibration image and one seed for both checkpoints. It covers task,
subtask, atomic motion, point, trace and combination commands plus exact
repeat, paraphrase, opposite relation, unrelated command, noun swap and
contradiction controls. A supplemental exact-input probe narrows to the four
task wordings and adds contrastive target-first and target-last variants, all
with input pixels, robot state and sampling seed held byte-for-byte fixed. That
probe can reveal a lexical order heuristic which closed-loop success
proportions alone cannot isolate.

**Scope of the lone subtask condition.** It is a one-request,
fixed-observation interface diagnostic. It is never selected from simulator
state, never switched during a rollout and never enters closed-loop success.
The study therefore has no subtask coach even though it documents whether the
released endpoint reacts to a subtask-form string.

## 3. Metrics

### Primary

**Binary success** is the official RoboLab requested-side termination: the cube
is inside the requested 45° robot-frame cone, within 0.1 m in height of the
bowl, and detached from the gripper.

**Paper-style progression** has two equally weighted items:

1. the correct Rubik's cube was successfully picked up at least once (credit
   persists);
2. the official requested-side success predicate is true, including release.

Each numerator and denominator is reported, direction and wording are kept
separate, and a 95% Beta(1,1) posterior credible interval accompanies every
observed success proportion. With ten episodes per cell, wide intervals are a
property of the design, not something to suppress. A central Bayesian credible
interval need not contain a boundary proportion such as 0/10, so interval
endpoints are drawn directly rather than as a non-negative error-bar distance
around the bar height.

### Declared secondary

- strict pick-then-place progression, requiring a post-pick transition into the
  requested relation;
- relation-only progression without the release requirement;
- signed final cube-minus-bowl offset, oriented so positive always means the
  requested side;
- first-chunk opposite-prompt RMS and same-prompt sampling RMS;
- paraphrase retention and directional asymmetry;
- inference time, wall time, GPU memory, disk and setup burden;
- for Cosmos, imagined/executed semantic quadrants and scorer coverage.

### Explicitly not success metrics

Action RMS and pixel MAE are diagnostics and are never promoted to success.
Exact paired McNemar tests are exploratory and reported without multiplicity
correction. Replan chunks from one episode are correlated, so semantic quadrant
rates are descriptive and receive no binomial confidence intervals.

## 4. The prompt-blind semantic future scorer

Cosmos emits a 33-frame imagined video with each 32-action chunk. Frame 0 is
the conditioning image and never counts as forecast evidence. Frames 8, 16, 24
and 32 are scored.

Those dimensions are not an arbitrary stretch of a 16-step model. The current
public model card separates a canonical 16-action configuration from a
32-action realtime PyTorch configuration; the locally pinned snapshot's own
`checkpoint.json` declares a 32-action chunk and 15 Hz conditioning, matching
the latter. The confirmation server uses four UniPC denoising steps, guidance
3.0 and the checkpoint's 8-D DROID joint-position-plus-gripper action schema.
The exact server command is preserved in the runbook.

A local [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
model sees one third-person camera panel at a time and receives only an
object-localisation request: find the multicoloured cube and the red bowl. It
never sees the policy instruction or the requested direction. The target
relation comes from the authoritative left/right simulator task identity, not
from asking the scorer to interpret the contrastive prompt's negation
(`semantic_target_parser_amendment_004.json`). Exact RoboLab camera intrinsics
and extrinsics project the two centroids to the table plane. Both
over-shoulder cameras must agree on the categorical relation and their two
reconstructed positions must agree within a frozen 0.20 m threshold.

**Labelling rule.** A chunk needs at least two reliable frames. At least 75%
requested frames means `imagined requested`; at most 25% means `did not imagine
requested`; the middle and insufficient-coverage cases abstain.

**Calibration.** All thresholds came from excluded 51xx calibration rollouts.
On 19 conditioning frames the two-camera relation agreed with simulator truth
on 18 (94.7%). The excluded dry run produced 78.9% chunk coverage and placed its
only positive quadrant on the final chunk of the successful-right rollout.
Confirmation labels are replayed at stricter 0.10 m and 0.15 m cross-camera
thresholds as a robustness audit; the frozen 0.20 m labels never change.

**Known limits.** The scorer is an auditable proxy, not ground truth for the
generated video. Occlusion, object deformation and the planar approximation can
force abstention or error. It scores the recorder's decoded 15 fps MP4, not a
pre-encoding latent or raw generator tensor, so video compression is part of
the measured interface. That is why coverage figures and contact-sheet overlays
ship beside the result, and why the human audit outcome is a qualified pass
rather than an evaluator-accuracy claim.

**Scope restriction.** The scorer is applied only where the cube–bowl relation
is the command's target: task, relation paraphrase/opposite, spatial-point and
combination commands. It does not score "move the gripper left" as though that
meant "put the cube left of the bowl". Subtask, atomic, trace, unrelated-object
and contradictory prompts remain sensitivity diagnostics unless a matching
task-specific semantic metric exists.

### Full quadrant table

At the frozen 0.20 m threshold, the 752 replan chunks divide as follows.
`Both` means the generated future and the executed 32-action horizon both reach
the request; `future only` and `execution only` are the mismatch modes;
`neither` is a certain negative; `uncertain` is an evaluator abstention.

| Wording | Request | Chunks | Certain coverage | Both | Future only | Execution only | Neither | Uncertain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| canonical | LEFT | 99 | 51/99 | 1 | 2 | 0 | 48 | 48 |
| canonical | RIGHT | 86 | 54/86 | 6 | 1 | 0 | 47 | 32 |
| short | LEFT | 125 | 71/125 | 1 | 0 | 1 | 69 | 54 |
| short | RIGHT | 64 | 31/64 | 5 | 2 | 1 | 23 | 33 |
| declarative | LEFT | 66 | 34/66 | 1 | 0 | 1 | 32 | 32 |
| declarative | RIGHT | 87 | 47/87 | 4 | 0 | 0 | 43 | 40 |
| contrastive | LEFT | 146 | 89/146 | 0 | 0 | 0 | 89 | 57 |
| contrastive | RIGHT | 79 | 44/79 | 4 | 0 | 0 | 40 | 35 |

### Exact-input semantic result

All 11 exact-input task-wording futures were reliably classified as cube–bowl
**neutral** over frames 8, 16, 24 and 32. Every LEFT/RIGHT condition changed
pixels; none imagined the requested terminal relation inside a single action
horizon. A full task normally takes several replans from reset, so this is not
a failed episode — it is direct evidence that nonzero future-video MAE cannot
be read as semantic goal completion.

The prompt-blind evaluator likewise classified all 16 rich-probe futures as
neutral. For the eight conditions where the cube–bowl relation was the declared
target — task, relation paraphrase/opposite, spatial point and combination —
that means 0/8 imagined the requested relation within one horizon. The other
eight do not have that target and are not counted as semantic failures.

## 5. Retrospective WAM tier

These experiments answer useful engineering questions but were not generated by
the shared frozen grid. They are never pooled into shared-grid intervals.

### [Efficient-WAM-RT](https://arxiv.org/abs/2606.10040) — fastest causal core

On one expert-valid RoboTwin scene and three diffusion seeds per native
direction, Efficient-WAM achieved 6/6 native successes and 2/6 matched
counterfactual successes. All 6/6 paired endpoints shifted in the requested
direction; median shifts were +15.2 cm for native-left → prompted-right and
−17.3 cm for native-right → prompted-left.

The stronger test changed language only after a verified grasp, preserving a
byte-identical 25-action prefix in all six matched groups. A full counterfactual
task succeeded 3/6, subtask 2/6, atomic motion 0/6 and combined motion-plus-
release 1/6; the same-direction control was 6/6. The aggregate conceals a
complete direction split: native-right → left was 3/3, native-left → right was
0/3.

*For:* ≈0.137 s per warm action chunk; ≈10 GB observed policy memory after
moving UMT5 to CPU; real same-history causal command interventions;
co-generated futures on a trainable 1B-scale core.
*Against:* positive evidence limited to one expert-valid scene; strong
directional and abstraction asymmetry; coarse future video; no demonstrated
structured point/trace interface; four of six static counterfactuals still
fail.

### [Fast-WAM](https://arxiv.org/abs/2603.16666) — a language knob that never reached inference

The release accepted `text_cfg_scale`, but action-only and joint inference did
not use it. Repairing the positive/negative guidance passes exposed a genuine
language signal. At the best post-hoc tested scale, prompt action RMS was
0.00532 versus 0.00793 sampling RMS — a ratio of 0.67. One clean matched
counterfactual succeeded in both directions, but the swapped success reproduced
on only 1/5 diffusion seeds.

*For:* the code path can be repaired and audited; one real counterfactual shows
usable signal exists; a useful example of testing an implementation rather than
trusting an API flag.
*Against:* prompt effect remains below sampling variation; post-hoc guidance
selection weakens the evidence tier; counterfactual robustness is 1/5; not a
dependable experimental controller.

### [LingBot-VA](https://arxiv.org/abs/2601.21998) — clearest imagined-future signal, slow control

LingBot repeated identical prompts exactly: action RMS and predicted-latent RMS
were both zero. Left/right predicted-video-latent RMS was 0.02733 while
normalised-action RMS was 0.00116. It solved both released native tasks but 0/2
swapped tasks. One swapped trajectory crossed to the requested side before
failing strict geometry/release; the other did not cross.

*For:* deterministic, language-dependent future latents; 2/2 native competence;
strong substrate for studying action/future coupling; fits one RTX 3090 after
sharing frozen VAE weights across two independent streaming-cache wrappers.
*Against:* ≈7 s per warm 16-action-plus-future chunk; 19,889 MiB (19.4 GiB)
PyTorch peak with little 3090 headroom; default matched swaps 0/2; higher action guidance did
not monotonically improve closed-loop control; CuRobo pools require one episode
per subprocess to avoid a second-episode OOM.

### Not measured under this protocol

[UVA](https://arxiv.org/abs/2503.00200) was the smallest attractive released
joint video/action checkpoint in the initial scan at roughly 0.5B parameters,
but there is no schema-correct matched closed-loop LIBERO result. Its early
pairwise heatmap is the motivating failure case for this study, not positive
evidence.

[Light-WAM](https://arxiv.org/abs/2606.08242) reports 0.44B **trainable**
parameters, 72.03 ms inference and 4.1 GiB peak memory, and provides LIBERO and
RoboTwin checkpoints. That trainable count excludes its frozen
Wan2.1-T2V-1.3B video backbone, so it is not a clean claim of a smaller total
deployed model. It is the highest-priority lightweight replication.

[DreamZero](https://dreamzero0.github.io/) is a useful large-model comparison,
but its ≈14B-scale deployment and different action/runtime stack make it a poor
rapid local core and an invalid participant in the shared DROID grid.

## 6. Operational cost

| Local system | Native output | Warm fixed-input request | Policy GPU point | Simulator GPU point | Mean guarded episode wall time |
| --- | --- | ---: | ---: | ---: | ---: |
| π0.5 DROID (VLA) | 15 actions | 0.145 s | 18,509 MiB | 7,570 MiB | 53–83 s |
| Cosmos3 Edge DROID (WAM) | 32 actions + 33 video frames | 5.526 s | 14,435 MiB | 7,949 MiB | 74–178 s |

The memory columns are steady-state point measurements, not peaks. The request
column comes from the exact fixed-observation probe and is the cleaner endpoint
comparison; the episode column includes simulator work and any thermal-guard
waiting. Locally pinned checkpoint directories occupied 12.44 GB for π0.5 and
9.17 GB for Cosmos. Both policies fit on one 24 GiB RTX 3090, but closed-loop
simulation simultaneously required the second card.

**Offline semantic pass.** This is an evaluation cost, not WAM inference
latency. Qwen processed 752 closed-loop chunks in 39 min 19 s, then the
16-condition and 11-condition probes in a further 1 min 35 s — 6,232
prompt-blind camera localisations in about 40 min 54 s on one 3090. Peak process
resident memory across stages was ≈5.12 GB. Localisation caches are keyed by
condition, so an interrupted audit resumes without repeating completed calls.

**Thermal confounds are real input confounds.** The accepted Cosmos
confirmation used one 3090 for the policy server and one for Isaac Sim. A live
snapshot during a valid request showed 14,435 MiB on the policy GPU and 7,949
MiB on the simulator GPU, with the policy 93% utilised; the simulator reported
software but not hardware thermal slowdown. The host driver was 535.309.01;
Isaac's Vulkan parser displayed 535.53.01 because its minor-version field
overflowed. A failed startup made zero policy requests and is preserved as an
excluded setup artefact; the valid run disabled only the erroneous version
check after `nvidia-smi` verified the actual driver.

That initial GPU assignment was not kept. The policy card reached 92 °C and
entered software thermal throttling. After moving the server to the cooler
second 3090, an exact seed-6100 replay showed why a casual resume would be
invalid: simulator state was byte-identical, but renderer output differed by
0.194/255 mean absolute pixel value and the first action chunk changed by RMS
0.0109. The interrupted paraphrase batch and the original 7/10-left,
9/10-right canonical batch were preserved as exclusions and both wordings were
rerun from seed 6100 under one common GPU assignment.

The common-role canonical rerun then exposed a second problem: the Isaac card
itself touched the preregistered 90 °C stop threshold after seven completed
left episodes. I stopped and excluded that entire directory, including three
chunks from the partial eighth episode, then froze a logged guard that pauses
only the simulator container at 87 °C, resumes it at 80 °C and still stops the
whole batch at 90 °C. Because a host pause could perturb wall-clock-dependent
realtime rendering, the already complete short-paraphrase batch was also
excluded and rerun under the same active guard. Simulated time, policy seeds,
model inputs and task configuration do not advance during a pause; episode wall
time does, and a pause can overlap the interval in which the client waits for a
response. Guarded closed-loop request timing is therefore an upper bound rather
than pure model latency, and separately measured warm probes are the cleaner
engineering comparison. Every cooling event ships as JSONL evidence and is not
subtracted from an unobserved phase.

The accepted π0.5 point measurement used 18,509 MiB on the policy 3090 and
7,570 MiB on the Isaac card, with no software or hardware thermal slowdown on
either card. Its four definitive thermal logs each contain a clean start/end
lifecycle with no cooling pause or emergency stop. Mean server request time in
the closed-loop logs was tightly grouped at 0.260–0.265 s, while the
fixed-input median was 0.145 s. Keeping those separate prevents simulator and
transport overhead from being reported as policy latency.

These details matter for the word *usable*. A model that fits only after a
hidden second-process allocation, takes minutes per intervention, or silently
ignores request seeds cannot support rapid causal experimentation even when its
paper metrics are good.

## 7. Integrity checks and amendments

**Initial-state identity.** All 160 analysed episodes share one exact physical
reset fingerprint across the robot and rigid objects. A fail-closed preview
exposed two hashes for the complete recorded reset group: Cosmos stores head
and right-shoulder camera poses that the π0.5 recorder omits. All 18 datasets
shared by both schemas are byte-identical; the difference is observation
bookkeeping, not a different scene
(`initial_state_schema_amendment_006.json`).

**Renderer variation.** That exact physical reset still does not produce
byte-identical first observations. The first two same-direction resets differed
by 3.50 mm at the cube, while the matched seed-6100 left/right centroids were
exact; their conditioning frames still differed by 1.60/255 MAE, and two
same-direction resets differed by 4.27/255. The closed-loop opposite-prompt
distance therefore contains prompt, settling *and* renderer variation, while the
same-prompt baseline contains sampling plus the same nuisance variation. Their
ratio is a sensitivity diagnostic, not a causal language estimate
(`observation_variation_amendment_001.json`).

**Derived-metric correction.** A second integrity check caught a subtler bug
before any confirmation future was semantically scored. The task predicate uses
rigid-object root poses, but the first compiler used rendered bounding-box
centroids, and cube rotation can shift those centroids across the 45° boundary.
The dated execution-geometry amendment switches endpoint and executed-state
relations to root poses in the robot frame, leaves visual calibration on visual
centroids, and changes no binary success, prompt, action, future or inclusion
decision (`execution_geometry_amendment_005.json`).

**Amendment ledger.**

| File | Purpose |
| --- | --- |
| `preregistration.json` | Frozen questions, grid, metrics, stopping rule |
| `direct_language_scope_amendment_003.json` | Outcome-timed scope disclosure; frozen declarative/contrastive stress grid |
| `metric_amendment_001.json` | Exact paper-style progression correction |
| `observation_variation_amendment_001.json` | Downgrades closed-loop action contrast after measured renderer variation |
| `initial_state_schema_amendment_006.json` | Separates exact physical reset identity from checkpoint-specific recorder schemas |
| `thermal_control_amendment_001.json` | Freezes pause/resume and emergency-stop behaviour |
| `thermal_timing_amendment_002.json` | Treats guarded client request timing as an upper bound |
| `semantic_target_parser_amendment_004.json` | Target resolution from matched task identity for prompts containing both direction words |
| `execution_geometry_amendment_005.json` | Source-aligned root-pose geometry for endpoint and executed-state relations |
| `trajectory_visualization_plan.json` | Complete-gallery policy, coordinate convention, deterministic social panel, retrospective-exemplar rule |
| `semantic_future_visualization_plan.json` | Frozen first-in-order example selection before confirmation scoring |
| `command_probe_plan.json` | Hash-pinned observation, coordinates, prompts, seed |
| `direct_task_command_probe_plan.json` | Exact-input task wording and contrastive word-order diagnostic |
| `semantic_future_calibration.json` | Every calibration point, threshold and localiser response |
| `run_manifest.json` | Local mapping for all eight direct-language conditions |

The same protocol caught an ignored guidance setting in Fast-WAM, renderer
variation in the shared grid, thermal-role confounds and a root-pose-versus-
bounding-box geometry bug. These are not model results, but they are exactly
the failures a reusable steerability benchmark should surface.

## 8. Evidence map

The complete registered package lives under `artifacts/vla_wam_shared_v1/`. The
machine-readable final join is `final_evidence/compiled_evidence.json`; the
human map is `final_evidence/EVIDENCE_INDEX.md`.

**Implementation files**

| File | Role |
| --- | --- |
| `tools/compile_vla_wam_evidence.py` | Closed-loop episode extraction |
| `tools/compile_vla_wam_study.py` | Fail-closed 160-episode join, integrity checks, paired diagnostics, robustness audit |
| `tools/render_study_figures.py` | All publication data figures, rendered from `compiled_evidence.json` only |
| `tools/figure_style.py` | Shared palette, typography and encoding contract for every figure |
| `tools/score_cosmos_semantic_futures.py` | Frozen prompt-blind semantic scorer |
| `tools/run_vla_wam_semantic_confirmation.sh` | Sequential GPU-1 scoring driver with resumable caches and per-stage timing logs |
| `tools/render_semantic_future_examples.py` | Deterministic future-frame strips and imagination/execution cards |
| `tools/run_fixed_observation_command_probe.py` | Shared command-style probe |
| `tools/render_trajectory_evidence.py` | Every-episode path renderer, machine-readable index, self-contained gallery, social exports |
| `tools/thermal_guard.py` | Logged simulator pause/resume and emergency stop |

**Figure provenance.** `render_study_figures.py` reads only
`compiled_evidence.json`. It performs no inference and derives no new
statistic, so a figure can be restyled without re-running the study and a
restyle cannot silently change a reported value. `compile_vla_wam_study.py`
calls into it, so the pipeline and a manual re-render share one code path.
Figure encoding is fixed in `figure_style.py`: hue encodes the checkpoint, tint
encodes the requested direction, and grey encodes absence of information
(abstention, uncertainty, out of scope).

**Raw outputs.** The raw RoboLab HDF5/log outputs stay in
`/home/ali/projects/RoboLab/output/`; absolute paths are recorded per episode in
the compiled evidence. `raw_evidence_manifest.csv` and
`supporting_evidence_manifest.csv` carry byte sizes and SHA-256 for every
prospective and supporting file. The retrospective tier remains under
`artifacts/wam_language_gate/` and is never silently mixed into the registered
direct-language estimates.
