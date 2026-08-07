# V3-E002 decision memo

Status: **complete model-blind diagnostic (108 valid episodes; zero learned-model requests).**

The corrected RTX/Isaac runtime uses the pinned RoboLab checkout and the
verified absolute-IK action contract. The static gate evaluated 0.075, 0.100,
0.150, and 0.200 m signed lateral targets across control and
position-mirrored layouts. Under the frozen 45-degree relation criterion,
0.075 and 0.100 m were feasible; the largest selected depth was 0.100 m.

All four matched 27-seed controller queues completed. The deterministic
waypoint recipe did not pick up the cube in any condition (0/27 each); these
are valid behavioral failures, not infrastructure exclusions. The resulting
diagnostic therefore does not establish controller competence or symmetry.
It does establish that this particular model-blind recipe is not a positive
control for the learned-policy claim. The selected-depth gate and all raw
episode rows are retained.

## Counts

| quantity | registered | completed |
|---|---:|---:|
| model-blind behavioral episodes | 108 | 108 valid |
| learned-model requests | 0 | 0 |
| infrastructure-invalid attempts | — | 0 |

## Claim boundary

Do not use these episodes to claim that the physical task is mechanically
easy. The controller's all-`pick_failed` outcome and endpoint errors show that
the present absolute-IK waypoint recipe is not a calibrated competence
baseline. The appropriate result is a failed model-blind competence control,
not evidence for a learned-policy direction effect.

`results.json` contains cell summaries, paired contrasts, gate provenance, and
source hashes. The raw JSONL episode ledgers remain committed here because
they are small and are the machine-readable evidence for this diagnostic.
