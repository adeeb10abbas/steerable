# Nano H01 stopping-control reproduction

Implemented 12 September 2026. This is a retrospective sensitivity analysis of
historical Nano g3 V3-B001, with **zero new model requests or robot episodes**.
It reproduces the review's **27/27 ordered command pairs in each layout before
either paired episode terminates**. Its selected time remains outcome-dependent.

## Source and definitions

The analyzer reads the episode JSONL and coordinate/termination source code
using `git show` at the immutable commit
`ce561e66f82e95055e39d3d7711691982f6b2086`. It works with a sparse checkout and
does not require the absent raw-media paths. The episode source is:

`artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/nano_v3b001_episodes.jsonl`

- Git blob: `5c2f4a6fa4795ec054dbce2d4d68554245ceb4cb`
- SHA-256: `ce67aecb2705394e8161fa4b65df7130e62648e2f6d8b6a93408c8da685a7207`
- Size: 5,822,218 bytes.

Coordinate signs come from the pinned integration, rather than an image or
plot convention. `experiments/v3/cosmos_nano_phase_b/robolab_bridge.py::_sample`
transforms cube and bowl centers into robot-base coordinates;
`compile_cell.py::_in_cone` uses positive object-minus-reference y for LEFT and
its negative for RIGHT. The task's `_LeftTermination` and `_RightTermination`
use the corresponding robot-frame relation with detached release required.
The source compiler preserves the initial sample and each post-action sample;
valid unsuccessful episodes run to the 450-action cap. The JSON result retains
Git/blob/SHA-256 identities for these source files as well as the input JSONL.

Define `s(t) = cube_y(t) - bowl_y(t)` in meters, positive toward robot LEFT.
The ordered paired response is `D(t) = s_LEFT(t) - s_RIGHT(t)`, so `D > 0`
means the expected LEFT/RIGHT ordering. The retrospective common step is
the greatest step present in **both** episode traces and **strictly less than
both** `actions_executed` values. Selection uses observed indices before
checking position availability: a missing latest position cannot cause a
fallback to an earlier, more favorable sample. Separate terminal endpoints
use each run's own final step, including valid failures at the cap.

The fixed grid comprises every 32-action boundary below 450, plus step 450.
Each pair must contain the exact observed sample from both commands; a terminal
sample exactly at a fixed checkpoint is allowed. Missing/truncated traces are
never interpolated, extended, or filled forward. Eligible and missing seed IDs,
per-command eligibility and exclusion reasons, and full-denominator ordering
bounds are recorded for every checkpoint.

## Cohort and reproduced result

There are 108 valid episodes, 102 successes and six valid capped failures,
with no absent episodes, invalid episodes, missing termination fields, missing
coordinates, or gaps in the 22,080 retained state samples (21,972 actions plus
108 initial samples). All six valid failures remain included.

The experiment has **one prescribed base scene and its position reflection**:
two fixed layout states, each under 27 matched policy/environment seed labels
9400–9426. Each layout has exactly one retained initial-state hash shared by
all 54 of its episodes. The 27 seed blocks are not 27 independent scenes.

| Layout | Eligible / expected | Ordered | Mean D (m) | Median D (m) | Common-step min / median / max |
| --- | ---: | ---: | ---: | ---: | --- |
| Control | 27/27 | 27 | 0.3110371443822428 | 0.3073885738849640 | 89 / 133 / 290 |
| Position-reflected | 27/27 | 27 | 0.3584485597977484 | 0.3537990823388100 | 96 / 149 / 320 |

The mean selected action steps are 149.7037037037037 and 149.3703703703704,
respectively. Minimum D is 0.07832422852516174 m in control and
0.11178560741245747 m after reflection. At each run's separate terminal
endpoint, the mean D values are 0.4480567460672723 m and
0.4527369962867212 m (medians 0.4667481780052185 m and
0.4373619556427002 m), also ordered in 27/27 pairs each. The reduction when
measured before termination is a measurement-time sensitivity, not a new rollout.

## Fixed action-checkpoint coverage

Each eligibility denominator is 27 pairs. `Ordered` is the number with strictly
positive D among eligible pairs. A dash means unavailable, never zero motion.
Exact unrounded measurements and each metric's sample size are in the JSON/CSV.

| Step | Control eligible | Control ordered | Control mean D (m) | Reflected eligible | Reflected ordered | Reflected mean D (m) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 27 | 0 | 0 | 27 | 0 | 0 |
| 32 | 27 | 0 | 0 | 27 | 0 | 0 |
| 64 | 27 | 2 | -0.000002358 | 27 | 3 | -0.000037260 |
| 96 | 26 | 20 | 0.085672 | 27 | 25 | 0.094537 |
| 128 | 18 | 17 | 0.204485 | 17 | 16 | 0.250769 |
| 160 | 8 | 7 | 0.180849 | 5 | 4 | 0.223271 |
| 192 | 5 | 4 | 0.151596 | 4 | 3 | 0.232147 |
| 224 | 1 | 1 | 0.031111 | 2 | 1 | 0.025867 |
| 256 | 1 | 1 | 0.029626 | 2 | 1 | 0.119842 |
| 288 | 1 | 1 | 0.060593 | 1 | 1 | 0.020313 |
| 320 | 0 | 0 | — | 1 | 1 | 0.197799 |
| 352 | 0 | 0 | — | 0 | 0 | — |
| 384 | 0 | 0 | — | 0 | 0 | — |
| 416 | 0 | 0 | — | 0 | 0 | — |
| 448 | 0 | 0 | — | 0 | 0 | — |
| 450 | 0 | 0 | — | 0 | 0 | — |

At step 64, mean cube displacement from its own initial state is only
0.419/0.979 mm for control LEFT/RIGHT and 0.988/0.465 mm for reflected
LEFT/RIGHT. Thus early complete coverage does not show substantial transported
cube motion. Later samples condition on episodes that have not already stopped.
At step 450 the surviving individual episodes are control LEFT 1, control RIGHT 1,
reflected LEFT 0, and reflected RIGHT 4; none form a complete matched command pair.
The unavailable full-cohort ordering bounds there are [0, 1].

## Cube and bowl contributions

The exact decomposition is `D = cube_y_separation - bowl_y_separation`.
Change-from-initial versions are also retained. At the common pretermination step:

| Layout | Mean cube y separation (m) | Mean bowl y separation (m) | Mean cube displacement LEFT / RIGHT (m) | Mean bowl displacement LEFT / RIGHT (m) |
| --- | ---: | ---: | --- | --- |
| Control | 0.3112061367956577 | 0.0001689924134148492 | 0.1499083509994673 / 0.2316624020770421 | 0.0002046997572572578 / 0.00003323581258754284 |
| Position-reflected | 0.3586486062310912 | 0.0002000464333428277 | 0.2921653163718893 / 0.1544615846582201 | 0.00001613011482691829 / 0.0007587210814625766 |

Displacement denotes Euclidean distance from the initial center, not path
length. These summaries and their fixed-step counterparts use the same
eligible pairs as D; missing initial samples leave displacement unavailable
without deleting an otherwise observed offset.

## Changes and validation

New files only:

- `analysis/analyze_stopping_controls.py`: standard-library pinned-input analyzer.
- `tests/test_stopping_controls.py`: 11 behavioral and integration tests.
- `results/stopping_controls.json`: provenance, definitions, cohort/missingness,
  all checkpoint summaries, and all 54 pretermination pair records.
- `results/stopping_controls.csv`: 36 flat summary rows (two layouts ×
  16 fixed checkpoints plus terminal and pretermination comparisons).
- This implementation report.

Reproduce from the repository root:

```bash
python3 workshops/corl2026_world_models/analysis/analyze_stopping_controls.py
python3 -m unittest discover -s workshops/corl2026_world_models/tests -p test_stopping_controls.py
```

All 11 stopping tests pass. The full current workshop suite also passed all
43 tests, and `git diff --check` passed. Tests first failed because the analyzer was absent.
Coverage includes strict terminal exclusion with irregular observed steps,
missing/truncated samples without fill-forward, absent episodes with preserved
denominators, missing latest coordinates, missing initial/termination samples,
invalid episodes, duplicate cells/steps, incompatible coordinates and conflicting
termination/pair identities, pinned 27/27 results, the full fixed-step eligibility
grid, and a CLI round-trip writing JSON/CSV from pinned Git objects.

The synthetic instruction-independent path visits LEFT at step 2 and RIGHT at
step 4. Goal-stopped endpoints show an apparent D of 2 m, whereas the latest
common pretermination sample and a shared fixed step both have D = 0. This
guards against accepting goal-stopped endpoints alone as directional response.

## Limits

The retrospective common step still depends on episode outcomes. Action indices
do not establish equal physical duration or forecast-frame alignment. Later
fixed-checkpoint summaries have outcome-dependent missingness and cannot stand
in for the planned fixed-duration recordings. Exact sign counts have no
calibrated movement threshold; tiny early differences are not evidence of
meaningful transport. No independent-scene inference, forecast-accuracy result,
or replacement of the prospective fixed-duration control is claimed. Historical
V2/V3 files and model runners were not changed.
