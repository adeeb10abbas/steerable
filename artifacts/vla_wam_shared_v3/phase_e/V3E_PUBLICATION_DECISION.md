# Phase-E publication decision

Both registered controls now have hash-bearing outputs. Their claim boundaries
remain separate: E001 is a fixed-observation model diagnostic; E002 is a
model-blind controller diagnostic and not a learned-policy baseline.

## V3-E001 — fixed observation, prompt versus sampling noise

Three checkpoints × two layouts × 27 matched sampling seeds plus the four
registered exact repeats per checkpoint were completed: **336/336 valid model
requests, zero behavioral episodes**. Every model/layout cell has 27/27
matched LEFT/RIGHT action effects and bit-identical exact repeats. The safe
claim is reproducible prompt-conditioned action change at the same observation,
qualified by the same-prompt cross-seed distributions in
`fixed_observation_prompt_noise_v3e001/results/compiled_results.json`.
The effect-to-noise relation is not uniform: π0.5 is below measured sampling
variation, Nano is mixed by layout, and DreamZero has zero measured
same-prompt variation (ratio recorded as null/unbounded rather than assigned
an epsilon).

## V3-E002 — model-blind reference controller

The corrected RTX/Isaac gate selected 0.100 m under the frozen 45-degree
relation criterion. The four 27-seed queues completed: **108/108 valid
model-blind episodes, zero learned-model requests**. The single deterministic
waypoint recipe produced 0/27 success in every cell and `pick_failed` for all
episodes. This is a failed competence control, not evidence that the learned
policies are mechanically feasible or that their asymmetry is explained.

## Manuscript guidance

**Main paper:** include E001 as a fixed-state causal diagnostic if the
prompt-to-noise ratios and exact-repeat results are shown alongside the
estimand definition. Do not pool it with task success.

**Supplement:** include E002 as a transparent negative-control attempt with
the full failure taxonomy and controller claim boundary. It should not be used
to support a mechanical-feasibility conclusion.

**Omit:** any statement that E002 proves a learned-policy effect, that the
controller is symmetric, or that language “reaches the policy every time.”

## Exact replacement text

“At an identical settled observation, changing only the static directional
prompt produced a reproducible action change across 27 matched sampling seeds
for π0.5, Cosmos3 Nano Policy DROID, and DreamZero in both control and
position-mirrored layouts. Exact repeats were bit-identical; prompt effects
are reported relative to same-prompt cross-seed variation rather than as task
success.”

“A model-blind absolute-IK waypoint control was run on the same four fixtures
for 27 matched seeds per condition. Although the static target-manager gate
accepted 0.100 m under the frozen relation criterion, the uncalibrated recipe
picked up the cube in 0/108 episodes. We therefore treat this control as a
negative diagnostic and do not use it to infer mechanical feasibility.”

## Validation and provenance

Run after the final compact reports are copied:

```text
python3 tools/validate_vla_wam_v3_protocol.py
python3 tools/validate_v3e001.py
python3 tools/validate_v3e002.py
git diff --check
```

The compact reports include SHA-256 hashes of the PVC raw request/episode
files and the static-gate source. Checkpoints, environments, decoded future
videos, and full Nano request payloads remain outside Git on the PVC.
