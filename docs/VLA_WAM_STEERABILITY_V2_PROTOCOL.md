# VLA/WAM language steerability v2

Status: **frozen before standardized v2 expansion inference** on 3 August
2026 at 01:19:36 UTC. The machine-readable source of truth is
[`protocol.json`](../artifacts/vla_wam_shared_v2/protocol.json), and the media
selection rules are frozen separately in
[`media_selection_plan.json`](../artifacts/vla_wam_shared_v2/media_selection_plan.json).
The executable validator currently passes 74 checks; its full report is
[`protocol_validation.json`](../artifacts/vla_wam_shared_v2/protocol_validation.json).

This is a disclosed post-v1 extension. Every pi0.5/Cosmos result in the
160-episode DROID study and the small retrospective Efficient-WAM, FastWAM,
and LingBot-VA gates was known when v2 was designed. Those outcomes motivated
the model set and visuals; they are not presented as prospectively hidden. No
standardized v2 four-prompt pilot result exists for any expansion model at the
time of this freeze.

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
| DROID | pi0-FAST DROID | VLA | none | new expansion model |
| DROID | GR00T N1.7 DROID | VLA | none | new expansion model |
| DROID | Cosmos3 Edge DROID | WAM | decoded video plus actions | existing v1 reference |
| RoboTwin | LingBot-VLA 4B | VLA | none | new expansion model |
| RoboTwin | Efficient-WAM-RT | WAM | coarse decoded video plus actions | standardized rerun required |
| RoboTwin | FastWAM | WAM | world-model training; action-only test time | standardized rerun required |
| RoboTwin | LingBot-VA | WAM | joint video/action latent with decodable future | standardized rerun required |

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
4. **How far did the behavior progress?** Interaction → pickup → requested
   region → release funnel.
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
  hashes six initial reader-first exports: prompt semantics, raw obedience
  scorecards, and all paired lateral endpoints in landscape and square formats.
- [`pilot_grid.json`](../artifacts/vla_wam_shared_v2/pilot/pilot_grid.json)
  compiles the effective protocol into 144 unique cells and 72 exact LEFT/RIGHT
  pairs. The first execution batch is the 36-cell direct-command base-
  competence gate; the remaining 108 wording cells are conditional on the
  frozen gate rather than being launched blindly.

Regenerate and validate them from the repository root:

```bash
python3 tools/validate_vla_wam_v2_protocol.py \
  --write-report artifacts/vla_wam_shared_v2/protocol_validation.json
python3 tools/select_vla_wam_v2_media.py
python3 tools/render_vla_wam_v2_reader_figures.py
python3 tools/build_vla_wam_v2_pilot_grid.py
```
