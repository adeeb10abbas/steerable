# Phase-E publication decision

The registered Phase-E controls and the V3-E004 symmetric-object-layout cohort
now have hash-bearing outputs. Their claim boundaries remain separate: E001 is
a fixed-observation model diagnostic; E002 is a model-blind controller
diagnostic and not a learned-policy baseline; E003 is the original 27-seed
π0.5 null-control cohort; and E004 is the powered, five-checkpoint extension.

## V3-E003 — bilateral-symmetry null control

The registered π₀.₅ cohort completed **54/54 valid episodes** (27 matched
seeds × exact LEFT/RIGHT prompts). The symmetric-object layout had zero
measured symmetry residual, but the task-success gap remained: LEFT 13/27
(48.1%, Wilson 95% CI 30.7–66.0%) versus RIGHT 25/27 (92.6%, CI
76.6–97.9%). The paired McNemar discordance was 13 LEFT-failure/RIGHT-success
versus 1 in the opposite direction (exact p=0.00183). Requested-side depth
was +6.18 cm RIGHT−LEFT (20,000-resample CI +3.29 to +9.15 cm; sign test
22+/5−, p=0.00151), while endpoint redirection remained strong (LEFT−RIGHT
=23.77 cm). The preregistered small-gap interpretation is therefore not
supported; the experiment shows that object layout alone does not explain the
full directional gap. It does not identify the remaining contribution of
embodiment, camera placement, controller calibration, or policy distribution.
The complete memo and hash manifest are in
`bilateral_symmetry_null_control_v3e003/DECISION_MEMO.md` and
`bilateral_symmetry_null_control_v3e003/evidence_manifest.json`.

## V3-E001 — fixed observation, prompt versus sampling noise

Three checkpoints × two layouts × 27 matched sampling seeds plus the four
registered exact repeats per checkpoint were completed: **336/336 unique valid
model requests, zero behavioral episodes**. All 12 registered exact-repeat
comparisons are present, hash-equal, bit-identical, and have RMS 0.0. The raw
ledgers contain 224 invalid rows but only 112 unique invalid attempts after
duplicate provenance accounting. The effect-to-noise relation is not uniform:
π0.5 is below measured sampling variation, Nano is mixed by layout, and
DreamZero has zero measured same-prompt variation (ratio recorded as
null/unbounded rather than assigned an epsilon). The paired sign-flip test and
per-dimension vectors are in
`fixed_observation_prompt_noise_v3e001/results/compiled_results.json`.

## V3-E002 — model-blind reference controller

The corrected RTX/Isaac gate selected 0.100 m under the frozen 45-degree
relation criterion. The four 27-seed queues completed: **108/108 valid
model-blind episodes, zero learned-model requests**. The single deterministic
waypoint recipe produced 0/27 success in every cell and `pick_failed` for all
episodes. This is a failed competence control, not evidence that the learned
policies are mechanically feasible or that their asymmetry is explained. The
actual producing runner, environment revision, fixture hash, and lane
attestations are bound in the E002 evidence manifest; the checked-in launcher
remains fail-closed and was not the producer.

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
position-mirrored layouts. All 12 registered exact repeats were bit-identical;
prompt effects are reported relative to checkpoint-specific same-prompt
cross-seed variation and a paired systematic-shift test rather than as task
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
python3 tools/validate_v3e003_symmetry_gate.py
python3 tools/validate_v3e003.py
git diff --check
```

The compact reports include SHA-256 hashes of the PVC raw request/episode
files and the static-gate source. Checkpoints, environments, decoded future
videos, and full Nano request payloads remain outside Git on the PVC.

## V3-E004 — symmetric-object-layout cohort

V3-E004 completed **4,096/4,096 registered behavioral episodes and 2,048
matched LEFT/RIGHT pairs** across π0.5, Cosmos3 Nano Policy DROID, Cosmos3
Edge, DreamZero, and the separately analyzed FastWAM RoboTwin stretch slice.
The final source audit retained 79 discovery-only behavioral artifacts and 627
infrastructure-invalid attempts outside the denominator and rehashed all 4,096
selected raw episode sources.

The powered π0.5 result gives the cleanest interpretation. Object-layout
symmetry reduced the RIGHT-minus-LEFT binary-success gap from 74.2 percentage
points at s=0 to 20.2 points at s=1. On the registered 27-seed contrast, the
interaction was −51.9 points (95% paired-bootstrap CI −81.5 to −22.2; exact
p=0.00786). The requested-side-depth gap fell from +17.0 cm to +5.75 cm, with
a registered interaction of −12.4 cm (95% CI −19.7 to −4.88; p=0.00348).
Endpoint redirection remained intact (+23.9 cm at s=1; 335/341 positive pairs),
so the symmetric scene did not simply break prompt-conditioned motion.

The residual gap is scientifically important. Neither π0.5 equivalence test
passed: its s=1 binary 90% CI was +15.0 to +25.2 points against a ±15.56-point
margin, and its depth 90% CI was +4.85 to +6.66 cm against ±4.15 cm. No other
checkpoint earned an equivalence claim. FastWAM also failed the registered
endpoint-redirection positive control, so its large depth reversal is
descriptive rather than evidence of reliable language steering.

**Main paper:** report π0.5's powered attenuation and residual gap, with the
endpoint positive control and preregistered equivalence failure. The safe
headline is: *bilateral object-layout symmetry removed most, but not all, of
the directional performance asymmetry.*

**Supplement:** report the complete checkpoint table, graded Nano/π0.5
dose-response estimates, geometry/visibility gates, failure taxonomy, and the
arena-separated FastWAM stretch result.

**Omit:** claims that object layout fully explains the gap; that the remaining
effect is uniquely caused by training data; that object symmetry makes the
robot, cameras, or embodiment symmetric; or that a nonsignificant interaction
establishes equivalence.

### Exact manuscript replacement text

“Across 4,096 prospectively registered episodes, bilateral object-layout
symmetry substantially attenuated direction-dependent task performance while
preserving prompt-conditioned endpoint redirection in the powered π0.5 cohort.
For π0.5, the RIGHT-minus-LEFT success gap fell from 74.2 to 20.2 percentage
points, and the requested-side-depth gap fell from 17.0 to 5.75 cm. However,
neither residual contrast satisfied its preregistered equivalence margin.
Object geometry is therefore a major causal contributor to the observed
directional asymmetry, but it is not a complete explanation.”

The complete machine-readable result, figures, decision memo, and evidence
manifest are under `symmetric_layout_cohort_v3e004/`. DROID/RoboLab and
RoboTwin remain separate and are never pooled.
