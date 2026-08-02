# VLA–WAM shared spatial benchmark v1

Status: **frozen prospectively on 2 August 2026 at 14:37:57 UTC**, before
recorder/scorer calibration and before any v1 confirmation episode. The
machine-readable source of truth is
`artifacts/vla_wam_shared_v1/preregistration.json`.

## Purpose

This benchmark closes the main comparability gap in the exploratory WAM study.
It evaluates a standard π0.5 DROID VLA and Cosmos3 Edge DROID WAM from the same
neutral RoboLab state, with the same task language, predicates, episode budget,
sampling-seed schedule, and task-progression rubric.

It is intentionally one spatial-generalization study. We will not relabel it as
a full reproduction of Chen et al.'s in-distribution, motion, spatial, and
semantic suites, because checkpoint training-set membership and equivalent
tasks are not established for the three omitted slices.

## Fixed confirmation grid

The static grid contains 80 episodes:

- 2 models: standard π0.5 DROID and Cosmos3 Edge DROID;
- 2 prompt wordings: canonical and short paraphrase;
- 2 requested relations: left and right;
- 10 prospective episode seeds: 6100–6109.

Each policy replan uses `episode_seed * 1000 + replan_index`. The same integer
schedule is used for the paired left/right prompts within a model. Integers do
not imply equivalent diffusion samples across architectures; they make repeats
within each architecture auditable.

The hierarchy add-on uses seeds 7100–7104. The predicate-oracle condition has
20 episodes and its matched static canonical control has another 20, for 40
episodes total. Both replan every five environment steps. The oracle issues a
grasp command before pickup, the requested spatial command while holding, and
a release command once the requested relation is true. This arithmetic
clarification was frozen before any hierarchy run in
`artifacts/vla_wam_shared_v1/hierarchy_amendment_001.json`; no model, task,
seed, prompt, metric, or stopping rule changed.

## Primary outcomes

The paper-aligned outcomes are requested-goal success and final task
progression. The two-item spatial rubric is:

1. correct cube picked up at least once; this credit persists;
2. correct cube put down in the requested left/right location.

Primary-source verification during inference corrected item 2 to include the
paper's explicit put-down requirement. The dated disclosure is in
`artifacts/vla_wam_shared_v1/metric_amendment_001.json`. Relation satisfaction
without release remains a secondary geometric diagnostic, not paper-aligned
task progression.

Every result is reported by model, wording, and direction with numerator,
denominator, and a 95% Beta(1,1) posterior interval. A stricter ordered
diagnostic requires pickup before placement so that pushing across a relation
boundary is not described as complete pick-and-place manipulation.

## WAM-only semantic outcome

Every recorded Cosmos replan stores its conditioning image, action chunk,
prompt, seed, and decoded future. Frame zero is conditioning and cannot count as
forecast evidence. The scorer tracks the cube and bowl in future frames and
asks whether the requested relation appears. It then compares that answer with
RoboLab ground truth after the corresponding executed actions.

The report must show the four quadrants:

- imagines requested, executes requested;
- imagines requested, executes opposite/not requested;
- does not imagine requested, executes requested;
- neither imagines nor executes requested.

Unreliable visual tracks are `uncertain`, not coerced to success or failure.
Coverage and audit contact sheets are mandatory.

The exact prompt-blind localizer, projection, abstention rules, excluded
calibration metrics, and evidence hashes were frozen separately in
`docs/SEMANTIC_FUTURE_SCORER_V1.md` before confirmation inference.

## Command-interface diagnostic

On a fixed initial observation and fixed sampling seed, both models receive the
six command styles from the paper: task, subtask, atomic motion, grounded point,
grounded gripper trace, and combination. Exact repeat, same-goal paraphrase,
matched opposite relation, and unrelated-command controls are included.

Point and trace coordinates must be measured in and linked to the actual source
camera image. Arbitrary coordinate-shaped text is invalid.

These fixed-observation probes diagnose interface sensitivity and semantic
direction. They are not substitutes for closed-loop success.

## Calibration and exclusions

Recorder and semantic-scorer calibration uses seed prefix 51xx and is excluded
from confirmation estimates. Detector thresholds and manual-audit rules are
frozen before seed 6100 is run. Earlier experiments are retrospective evidence
and remain in `artifacts/wam_language_gate`; they never enter v1 confidence
intervals.

Setup failures are excluded only with an error log demonstrating that no valid
policy episode ran. Full-horizon task failures remain model failures. Prompts,
seeds, thresholds, and stopping rules cannot be changed after inspecting
confirmation outcomes.

## Claim boundary

Even a clean result supports only a statement about these two checkpoints in
this shared spatial task. A VLA-versus-WAM class claim would require multiple
models per class, all four generalization slices, held-out scenes, and a larger
preregistered grid.
