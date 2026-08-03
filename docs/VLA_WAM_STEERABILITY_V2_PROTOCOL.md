# VLA/WAM language steerability v2

Status: the base protocol was **frozen before standardized v2 expansion
inference** on 3 August 2026 at 01:19:36 UTC. Three six-episode RoboTwin WAM
pilots and one six-episode π0-FAST DROID VLA pilot completed the
one-direction-only branch of the frozen adaptive gate. π0-FAST has since
completed its prospectively frozen ten-seed direct-command directional
confirmation. Efficient-WAM-RT has also completed the first prospective
RoboTwin confirmation pair; the three model-level WAM confirmations remain
pending. The RoboTwin ten-scene directional confirmation was frozen at
03:29:32 UTC after
disclosing all 18 WAM pilot outcomes and before any of its 42 new expansion
episodes. The machine-readable source of truth is
[`protocol.json`](../artifacts/vla_wam_shared_v2/protocol.json), and the media
selection rules are frozen separately in
[`media_selection_plan.json`](../artifacts/vla_wam_shared_v2/media_selection_plan.json).
The executable validator and its exact current check count are recorded in
[`protocol_validation.json`](../artifacts/vla_wam_shared_v2/protocol_validation.json).

This is a disclosed post-v1 extension. Every pi0.5/Cosmos result in the
160-episode DROID study and the small retrospective Efficient-WAM, FastWAM,
and LingBot-VA gates was known when v2 was designed. Those outcomes motivated
the model set and visuals; they are not presented as prospectively hidden. No
standardized v2 four-prompt pilot result existed for any expansion model at
the original freeze. Since then, Efficient-WAM-RT, FastWAM, and LingBot-VA have
completed the frozen direct-command gate. This document labels those observed
results separately from prospective follow-up cells.

## Research question

When does changing only the language change the requested physical outcome,
rather than merely changing an action tensor or an imagined video?

The v2 study expands breadth without pretending that DROID and RoboTwin are one
benchmark. It contains two matched arenas. Raw success, progression, endpoint,
and failure-stage comparisons remain inside an arena. Cross-arena figures may
compare only normalized model descriptors, and every mark must name its arena.

## Eight-model core

| Arena | Checkpoint | Category | Future interface | v2 status |
| --- | --- | --- | --- | --- |
| DROID | pi0.5 DROID | VLA | none | existing v1 reference |
| DROID | pi0-FAST DROID | VLA | none | confirmation complete: LEFT 1/10, RIGHT 10/10; wording eligible but deferred |
| DROID | GR00T N1.7 DROID | VLA | none | new expansion model |
| DROID | Cosmos3 Edge DROID | WAM | decoded video plus actions | existing v1 reference |
| RoboTwin | LingBot-VLA 4B | VLA | none | new expansion model |
| RoboTwin | Efficient-WAM-RT | WAM | coarse decoded video plus actions | direct gate complete; prospective pair03 adds LEFT 0/1, RIGHT 0/1 |
| RoboTwin | FastWAM | WAM | world-model training; action-only test time | direct gate complete: LEFT 1/3, RIGHT 0/3 |
| RoboTwin | LingBot-VA | WAM | joint video/action latent | direct gate complete: LEFT 3/3, RIGHT 0/3 |

The WAM label is not treated as a single interface. A generated future that a
human can inspect, a latent future representation, and a training-only future
objective provide different kinds of evidence. Figures and tables must say
which interface is present rather than using a generic WAM badge.

LaWAM, Light-WAM, pi0 DROID, and DreamZero remain second-wave candidates. They
are not assigned a zero: they are explicitly marked *not yet measured under
v2*. LaWAM's task policy weights are incomplete locally; Light-WAM lacks a
local integration audit; pi0 is deferred to limit first-wave family redundancy;
and DreamZero does not currently fit beside the simulator on the local two-3090
allocation.

## Four prompts, in plain language

The same scene, physical reset, task predicate, model, and paired seed are held
constant. Only the episode-static sentence changes.

### 1. Direct command

> Put the Rubik's cube to the **LEFT** of the bowl.

Question: can the model ground a direct task instruction? This is the reference
condition.

### 2. Short command

> Put the cube **LEFT** of the bowl.

Question: does grounding survive removed object detail and function words? It
removes “Rubik's” and “to the” without changing the requested relation.

### 3. Goal stated as an outcome

> The Rubik's cube should end up to the **LEFT** of the bowl.

Question: does a desired end-state description ground like an imperative? This
tests speech-act form, not a different physical goal.

### 4. Desired side plus negated opposite

> Put the Rubik's cube to the **LEFT** of the bowl, **not to the RIGHT** of the
> bowl.

Question: can the model resolve negation and semantic scope when both direction
words occur? This condition is contrastive, not contradictory. One relation is
requested; the opposite relation is explicitly rejected. A direction-word bag
sees both LEFT and RIGHT. A grounded policy must identify which relation is
inside the negated phrase.

The RIGHT prompts mirror these sentences exactly. For RoboTwin, every adapter
uses the same first `seen` object description from the simulator metadata. The
short command removes moved-object modifiers by using the canonical noun from
the raw object identifier; the other three forms use the full seen
descriptions. This keeps rendered prompt bytes identical across checkpoints.

## Token-order diagnostic

Closed-loop contrastive episodes use the desired relation first. A separate
hash-pinned observation probe compares it with equivalent target-last syntax:

> Do not put the Rubik's cube to the **RIGHT** of the bowl; put it to the
> **LEFT** of the bowl.

This asks whether equivalent meaning survives reversal of desired and negated
direction-token order. Action or future distance in this probe establishes
sensitivity and order dependence only. It is not a task-success metric.

## Matched arenas

### DROID/RoboLab

- Tasks: `RubiksCubeLeftOfBowlMatchedTask` and
  `RubiksCubeRightOfBowlMatchedTask`.
- One neutral cube/bowl reset; neither requested predicate starts true.
- Existing v1 seeds 6100–6109 for direct/short and 7200–7209 for
  outcome/contrastive prompts.
- Those v1 tiers do **not** contain exact-seed direct↔contrastive pairs. We do
  not substitute equal run numbers for equal seeds.
- New-model pilot seeds are 8300, 8301, and 8302 for every prompt form and both
  directions. The ten-seed expansion is 8300–8309. This makes every v2 wording
  and direction pair exact-seed matched.
- Every new pilot records the simulator viewport.
- Requested success is the official release-inside-the-45-degree-cone task
  termination.

The new DROID models use the exact reset and prompt grid already completed by
pi0.5 and Cosmos. Their native action representations and horizons remain
model-specific and are reported rather than forced into an artificial common
horizon.

### RoboTwin place-A-relative-to-B

- Available task classes are `place_a2b_left` and `place_a2b_right`, but they
  are **not** treated as matched scenes: the same integer seed samples different
  objects under the two task classes.
- Frozen pair 00 uses `place_a2b_left`, environment seed 4300000, and sampling
  seed 8400.
- Frozen pair 01 uses `place_a2b_right`, environment seed 4300001, and sampling
  seed 8401.
- Frozen pair 02 uses `place_a2b_left`, environment seed 4300002, and sampling
  seed 8402.
- Within each pair, both requested directions and all four prompt forms run in
  that one anchor task and one object layout. Equal sampling-seed integers make
  within-model repeats auditable; they do not imply identical random samples
  across architectures.
- One relation-aware success checker reads the requested prompt rather than the
  native task directory name.
- Every new pilot records the simulator video; exposed WAM futures are retained
  separately.

Existing RoboTwin single-scene gates used different prompts and partial seed
sets. They remain retrospective evidence and never enter v2 success rates.

This anchor-scene rule is pre-inference amendment `V2-A001`. It was recorded
after inspecting retrospective scene metadata and before completing any
standardized v2 episode; it changes no episode count. `V2-A002` likewise
standardizes object naming across model adapters before inference.

## Pilot and cost gate

The standardized first wave contains:

```text
6 expansion models × 4 prompt forms × 2 directions × 3 paired seeds
= 144 episodes
```

All valid failures remain evidence. The gate controls only *additional* spend:

1. **Technical invalidity:** preserve the logs, repair the setup, and rerun only
   the exact invalid cell. Infrastructure failure is not model failure.
2. **At least one direct-command success in each direction:** expand the model
   to ten seeds for all four prompts. A model with no language effect still
   expands because that is a scientifically important negative result.
3. **Direct-command success in one direction only:** run a ten-seed direct-
   command directional-bias confirmation before spending on the full wording
   grid.
4. **Zero direct-command success:** publish the six attempts as a base-
   competence failure and stop. Do not call an incapable base policy
   “language-unsteerable.”

The pilot media budget is 25 GiB. Raw episode collections stay outside ordinary
Git history; manifests, hashes, posters, compact publication clips, and the
renderer are versioned.

### Observed RoboTwin gate and frozen follow-up

All three locally runnable WAMs triggered branch 3:

| Model | Direct LEFT | Direct RIGHT | Gate |
| --- | ---: | ---: | --- |
| Efficient-WAM-RT | 2/3 | 0/3 | directional confirmation only |
| FastWAM | 1/3 | 0/3 | directional confirmation only |
| LingBot-VA | 3/3 | 0/3 | directional confirmation only |

Amendment `V2-A004` was recorded after those 18 outcomes were known. It does
not claim preregistration. It freezes environment seeds 4300000–4300009,
sampling seeds 8400–8409, alternating LEFT/RIGHT native anchors, and both
requested directions inside every anchor scene. The first three scenes are the
completed pilot. The same seven prospective scenes are used for every model,
adding 42 episodes and producing 60 total direct-command episodes when
complete. No wording cell is authorized.

Before loading another policy, all seven prospective fixtures were initialized
model-blind. Every fixture contained distinct moved/reference objects and
started outside both relation regions. The fixture report establishes only
technical validity and neutral starting geometry; it is not behavioral
evidence.

Efficient-WAM-RT pair03 is the first prospective pair completed after that
freeze. Its LEFT and RIGHT cells are both valid 400-action behavioral failures.
The first ten executed actions differ (RMS 0.0762), but the final
RIGHT-minus-LEFT object-to-target lateral offset is -0.0081 m, so the endpoint
ordering is anti-aligned with the requested language change. Both cells retain
simulator video and a five-frame decoded future; the thermal guard recorded a
49 C maximum with no pause or emergency. This is evidence for one scene, not a
model-level ten-scene result, and pair03 must not be rerun. The remaining
prospective WAM queue is 40 episodes.

### Observed π0-FAST DROID gate and selected follow-up

π0-FAST completed the exact six-episode DROID direct-command gate at
environment/sampling seeds 8300–8302. The prompt pair was the frozen sentence
“Put the Rubik's cube to the LEFT/RIGHT of the bowl.” The controller was static,
the open-loop horizon was ten actions, viewport video was enabled, and no
oracle action, subtask coach, dynamic prompt, or subtask progress checker was
used.

| Diagnostic | Observed result |
| --- | ---: |
| LEFT relation-and-release success | 0/3 |
| RIGHT relation-and-release success | 3/3 |
| LEFT / RIGHT verified pickup proxy | 2/3 / 3/3 |
| Same-seed first ten actions differ | 3/3 pairs |
| RIGHT endpoint is right of matched LEFT endpoint | 3/3 pairs |

The paired final lateral shifts were +0.374, +0.162, and +0.251 m toward robot
RIGHT. This supports command-conditioned physical redirection but not robust
bidirectional steerability. Under branch 3, the selected next experiment is
the ten-seed direct-command directional-bias confirmation. Short, outcome, and
contrastive prompt cells remain unauthorized until that gate resolves the base-
competence asymmetry.

The fixed-observation diagnostic is deliberately secondary. An exact duplicate
prompt produced RMS 0 while the older-v1-wording LEFT/RIGHT swap produced action
RMS 0.153. Because those diagnostic labels are not byte-identical to the v2
closed-loop prompts, they establish repeatability and label sensitivity only.

### Compiled π0-FAST direct-command directional confirmation

The prospectively frozen fourteen new cells at seeds 8303–8309 completed under
the same static direct prompts, ten-action open-loop controller, viewport-video
retention, and no-coach rule. Combined with the preserved seeds 8300–8302,
LEFT released requested placement is 1/10 (Wilson 95% [0.018, 0.404]) and
RIGHT is 10/10 ([0.722, 1.000]). All 20 behavioral episodes are valid; all ten
same-seed RIGHT-minus-LEFT endpoint shifts align with the requested direction.
One intermittent policy-GPU thermal-slowdown event makes seed-8305 LEFT wall
latency ineligible for operational aggregates, without changing its behavioral
failure. The compiled evidence is
[`pi0_fast_direct_confirmation.json`](../artifacts/vla_wam_shared_v2/pilot/results/pi0_fast_direct_confirmation.json)
with its CSV and Markdown companions; the external-log provenance and
latency-exclusion rule are hash-pinned in
[`pi0_fast_runtime_interventions.json`](../artifacts/vla_wam_shared_v2/pilot/directional_confirmation/pi0_fast_runtime_interventions.json).

This resolves π0-FAST direct competence in both directions and therefore makes
its four-wording grid eligible under the frozen adaptive rule. It does **not**
authorize a wording episode now: the grid is deliberately deferred until all
three WAM directional confirmations are compiled and a disclosed post-result
decision records the next authorized spend.

## Metrics

The frozen hypotheses are deliberately physical rather than tensor-level:

1. Mirroring LEFT to RIGHT should redirect the same-seed final endpoint in the
   requested physical direction.
2. Direct-command grounding should survive a shorter command, a declarative
   goal, and scope-sensitive negation when direct-command base competence is
   present.
3. LEFT and RIGHT performance should be symmetric; any gap is reported as a
   checkpoint-and-arena-specific bias.
4. For WAMs with decodable futures, the imagined relation should agree with the
   subsequently executed relation over the same horizon. Agreement is
   descriptive and does not prove that imagination caused the action.

Primary within-arena evidence:

- binary requested-relation success with raw numerator and denominator;
- persistent correct pickup plus released requested placement progression;
- same-seed paired endpoint displacement;
- mutually exclusive failure stages from no interaction through successful
  release.

Language diagnostics:

- short-command retention versus direct command;
- goal-as-outcome retention versus direct command;
- contrastive retention versus direct command;
- LEFT/RIGHT gap for every prompt form;
- exact paired discordance tables.

WAM-only evidence is conditional on the released interface. When a future is
decodable, the study reports prompt-blind imagined/executed relation quadrants,
coverage, and abstentions. FastWAM receives `not applicable` for test-time
imagined-video metrics; it is not assigned a neutral or zero value.

Latency, GPU memory, checkpoint bytes, auxiliary assets, simulator load,
thermal pauses, setup failures, and media bytes are part of usability.

## Reader-first figures

Every plot begins with a plain-language question and shows the exact prompt or
a nearby prompt key.

1. **What changed in the sentence?** Exact prompt cards, desired relation,
   negated distractor, held-constant variables, and the question each form
   answers.
2. **Did the robot obey?** Within-arena scorecards with prompt descriptions,
   raw successes, denominators, and uncertainty.
3. **Where did the object finish?** Physical bowl-relative lateral endpoints,
   with robot LEFT and RIGHT shown directly and matched seeds connected.
4. **How far did the behavior progress?** Started, transparent pickup proxy,
   requested region, and released success counts. These are observable stage
   counts rather than an assumed monotone funnel: a successful slide can fail
   the lift-based pickup proxy.
5. **Did the sentence redirect the same scene?** Same-seed expected-region and
   actual-path pairs.
6. **Did the WAM imagine what happened next?** Synchronized generated and
   executed horizon frames with abstentions visible.
7. **What did it cost?** Latency, memory, asset size, simulator burden, and
   future inspectability.

“Expected path” is never treated as one privileged trajectory. The normative
display is the requested goal region. Any dashed direct route is labeled
illustrative and never enters a metric.

## Video evidence

Every future pilot records a complete viewport video. Publication examples are
selected by the frozen rules in `media_selection_plan.json`: first LEFT
success, first RIGHT success, first post-pick placement failure, and the first
direct/contrastive same-seed reversal. If a category is absent, the gallery says
so rather than substituting a nicer clip.

Existing v1 DROID selections are retrospective but deterministic. Their saved
HDF5 actions are replayed with the recorded environment configuration. A replay
is rejected if the requested relation, binary outcome, failure stage, or final
lateral endpoint differs from the original by more than 2 cm. The exact policy
cell is rerun with video enabled when replay is not faithful.

Each article clip contains:

1. model, arena, exact prompt, seed, and requested-side diagram;
2. the complete robot rollout with requested region and object path overlay;
3. final endpoint, success/failure, and last verified progress stage;
4. for applicable WAMs, generated future versus the subsequently executed
   horizon, explicitly labeled as a horizon comparison.

The media index stores source paths, prompt text, requested relation, model,
arena, seed, outcome, endpoint, failure stage, source/replay status, and file
hashes. Every clip receives burned-in text, VTT captions, a poster, and alt
text.

The completed WAM gate additionally publishes one exact-pair explainer per
model: the first compiled pair with a LEFT success and matched RIGHT failure.
Each clip is exported at 1600×900 for the article/X and 1200×1200 for social
sharing. Both show the exact prompts, full outcome text, complete rollouts,
requested regions, and state-derived paths. When one rollout ends first, its
final frame is held while the other continues. The standalone
[`video gallery`](VLA_WAM_STEERABILITY_VIDEO_GALLERY.html) filters by future
interface. FastWAM's original capture used the wrong raw pixel dimensions; the
gallery explicitly records the four-packet head-camera reconstruction, and the
runner is repaired for all future captures.

The π0-FAST DROID gate publishes its first compiled LEFT-failure and matched
RIGHT-success pair under the same no-trimming rule. Its landscape and square
clips include the exact prompts, complete viewport rollouts, state-derived
Rubik's-cube paths, requested regions, endpoints, and failure stages. Its media
manifest is separate from the RoboTwin manifest so the arenas cannot be pooled
accidentally.

## Claim boundary

This is a multi-checkpoint, two-arena case study. Within-arena matched
comparisons apply to the tested checkpoints, prompts, seeds, scenes, action
interfaces, and success predicates. The study does not estimate a universal
VLA-versus-WAM class effect. DROID and RoboTwin raw success counts are never
pooled or placed on an unlabeled common leaderboard.

## Reproducible v2 implementation artifacts

- [`model_readiness.json`](../artifacts/vla_wam_shared_v2/model_readiness.json)
  records what is present locally, exact repository commits, checkpoint and
  auxiliary-asset bytes, and each model's next technical gate. A present file
  is setup evidence, not a steerability result.
- [`v1_droid_selection.json`](../artifacts/vla_wam_shared_v2/media/v1_droid_selection.json)
  and its CSV companion apply the frozen media rules to all 160 v1 episodes.
  Six unique episodes require validated viewport replay. Both exact-seed
  direct-to-contrastive slots are explicitly absent because v1 used disjoint
  seed tiers.
- [`figures_manifest.json`](../artifacts/vla_wam_shared_v2/figures/figures_manifest.json)
  hashes sixteen reader-first exports: prompt semantics, raw DROID obedience
  scorecards, paired DROID lateral endpoints, Efficient-WAM-RT path panels,
  cross-model RoboTwin stage counts, all nine WAM paired endpoints, the exact
  π0-FAST direct gate, and its three same-seed path pairs. Every figure is
  exported in landscape and square formats.
- [`pilot_grid.json`](../artifacts/vla_wam_shared_v2/pilot/pilot_grid.json)
  compiles the effective protocol into 144 unique cells and 72 exact LEFT/RIGHT
  pairs. The first execution batch is the 36-cell direct-command base-
  competence gate; the remaining 108 wording cells are conditional on the
  frozen gate rather than being launched blindly.
- [`efficient_wam_rt_direct_gate.md`](../artifacts/vla_wam_shared_v2/pilot/results/efficient_wam_rt_direct_gate.md)
  compiles the first six standardized RoboTwin episodes. Efficient-WAM-RT
  succeeds on 2/3 LEFT prompts and 0/3 RIGHT prompts; all six episodes satisfy
  the transparent verified-pickup proxy, so the four failures are post-pick
  placement failures. The frozen gate selects only a ten-scene direct-command
  directional-bias confirmation, not the four-wording sweep.
- [`fastwam_direct_gate.md`](../artifacts/vla_wam_shared_v2/pilot/results/fastwam_direct_gate.md)
  records FastWAM's matched result: 1/3 LEFT and 0/3 RIGHT. One additional
  LEFT run entered the requested region without completing release. Because
  the released action-only inference path emits no test-time future video, its
  imagination/execution fields are explicitly not applicable. This model also
  selects only the direct-command directional-bias confirmation.
- [`lingbot_va_direct_gate.md`](../artifacts/vla_wam_shared_v2/pilot/results/lingbot_va_direct_gate.md)
  records LingBot-VA's 3/3 LEFT and 0/3 RIGHT result. All six cells retain the
  first predicted latent. Three thermally paused cells remain valid behavioral
  episodes but are excluded from wall-latency aggregates in
  [`runtime_interventions.json`](../artifacts/vla_wam_shared_v2/pilot/runtime_interventions.json).
- [`pi0_fast_direct_gate.md`](../artifacts/vla_wam_shared_v2/pilot/results/pi0_fast_direct_gate.md)
  records the exact DROID result: 0/3 LEFT and 3/3 RIGHT, with all three paired
  endpoints shifted toward RIGHT. The raw RoboLab HDF5, environment configs,
  logs, and viewport videos stay outside ordinary Git; the compiled result
  hashes each one and versions the six state-derived trajectories.
- [`pi0_fast_directional_expansion.json`](../artifacts/vla_wam_shared_v2/pilot/pi0_fast_directional_expansion.json)
  is the prospective registry for the next fourteen DROID cells. It freezes
  seeds 8303--8309, exact prompts, checkpoint and repository revisions, video
  retention, and the no-coach static-controller rule before new inference.
- [`pi0_fast_direct_confirmation.json`](../artifacts/vla_wam_shared_v2/pilot/results/pi0_fast_direct_confirmation.json)
  compiles the completed twenty-episode direct-only confirmation without
  overwriting the six-episode gate: LEFT is 1/10, RIGHT is 10/10, all ten
  paired endpoints align, and one wall-latency-only intervention is recorded
  in its hash-bearing directional-confirmation ledger.
- [`directional_expansion.json`](../artifacts/vla_wam_shared_v2/pilot/directional_expansion.json)
  freezes the seven additional scene pairs and discloses the 18 known pilot
  outcomes. Its model-blind seven-scene setup audit is
  [`directional_fixture_validation.json`](../artifacts/vla_wam_shared_v2/pilot/directional_fixture_validation.json).
- [`efficient_wam_rt_pair03_integration.json`](../artifacts/vla_wam_shared_v2/pilot/directional_confirmation/efficient_wam_rt_pair03_integration.json)
  preserves the first prospective Efficient-WAM-RT pair as two valid failures,
  including paired action sensitivity, anti-aligned endpoint ordering, decoded
  future metadata, raw-output hashes, and its no-intervention thermal record.
- [`continuation_state.json`](../artifacts/vla_wam_shared_v2/continuation_state.json)
  and [`VLA_WAM_CONTINUATION.md`](VLA_WAM_CONTINUATION.md) provide the
  machine-readable queue and human restart guide. They name the exact next
  cells, launch commands, readiness blockers, stopping conditions, and
  required handoff updates for a fresh model or a new usage window.
- [`WORK_LAPTOP_B200_HANDOFF.md`](WORK_LAPTOP_B200_HANDOFF.md) and the
  [`repo_bundles` manifest](../handoff/repo_bundles/MANIFEST.json) make the
  external model/simulator integration commits and cluster continuation
  procedure portable without relying on chat history.
- [`media_index.json`](../artifacts/vla_wam_shared_v2/media/robotwin_wam_pairs/media_index.json)
  hashes three matched success/failure pairs in landscape and square formats,
  posters, captions, exact prompts, source trajectories, and the disclosed
  FastWAM pixel-layout repair.
- The separate π0-FAST
  [`media_index.json`](../artifacts/vla_wam_shared_v2/media/droid_pi0_fast_pairs/media_index.json)
  hashes the seed-8300 LEFT-failure/RIGHT-success explainer in both aspect
  ratios, its posters and captions, the exact source rollouts, and the frozen
  deterministic selection rule.
- [`execution_configs.json`](../artifacts/vla_wam_shared_v2/pilot/execution_configs.json)
  records the exact completed-pilot settings. Efficient-WAM-RT's entry was
  recorded retrospectively after its first six cells; FastWAM and LingBot-VA
  were frozen before their cells. The pre-episode SAPIEN startup failures and
  repair evidence remain separate in
  [`technical_events.json`](../artifacts/vla_wam_shared_v2/pilot/technical_events.json)
  and never enter a model denominator.

Regenerate and validate them from the repository root:

```bash
python3 tools/validate_vla_wam_v2_protocol.py \
  --write-report artifacts/vla_wam_shared_v2/protocol_validation.json
python3 tools/select_vla_wam_v2_media.py
python3 tools/render_vla_wam_v2_reader_figures.py
python3 tools/render_vla_wam_v2_robotwin_videos.py
python3 tools/render_vla_wam_v2_droid_videos.py
env -u DISPLAY CUDA_VISIBLE_DEVICES=1 \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  PYTHONPATH=/home/ali/projects/EfficientWAM-RoboTwin:/home/ali/lab/RoboTwin/envs/curobo/src \
  /home/ali/projects/Efficient-WAM/.venv/bin/python \
  tools/validate_robotwin_directional_fixtures.py
python3 tools/build_vla_wam_v2_pilot_grid.py
python3 tools/compile_vla_wam_v2_robotwin_pilot.py
```
