# Cosmos3 Edge DROID steerability pilot

Status: schema-correct offline diagnostics plus one matched closed-loop spatial
pair. This is pilot evidence, not an estimate of general steerability.

## Primary closed-loop outcomes

Checkpoint: `nvidia/Cosmos3-Edge-Policy-DROID` (4B). Simulator: RoboLab with
the DROID joint-position controller. Scene/model seed: 0.

The released native-left task establishes task competence:

| Condition | Success | Paper-style progression | Episode result |
| --- | ---: | ---: | --- |
| Released native left | 1/1 | 2/2 | Grasped cube; established left at step 319; released at 320; final lateral offset +8.04 cm |

The first same-scene right task was invalid because the cube already started
robot-right of the bowl. RoboLab treated each one-step termination as a physics
artifact and reset repeatedly. That run is excluded.

For the corrected matched pair, a deterministic reset event places the cube at
`(0.300653, 0.126572, 0.081132)` and the bowl at
`(0.442584, 0.126582, 0.077328)`: longitudinally separated with negligible
lateral offset, so neither relation is initially satisfied. The recorded robot,
cube, bowl, and banana initial-state arrays are byte-identical between tasks.
Only instruction and requested predicate differ.

| Neutral-start condition | Success | Paper-style progression | Raw score | Episode result |
| --- | ---: | ---: | ---: | --- |
| Left | 0/1 | 1/2 | 2/3 | Grasped at 141; never established left; truncated at 450; final lateral offset −11.79 cm |
| Right | 1/1 | 2/2 | 1.0 | Grasped at 62; established right at 106; released at 114; final lateral offset −24.09 cm |

The raw left score counts `object_dropped` at step 1 before a grasp. The
paper-aligned two-item rubric instead asks whether the correct object was picked
and whether it reached the requested relation. Rubric items need not be
completed in order, and pickup credit persists. First-chunk (32-action) RMS
between the two prompt conditions is 0.01648; RMS over their shared 114-step
horizon is 0.28413. The result is one-sided steering evidence: the successful
right command amplifies robot-right motion, while the left condition also
finishes on robot-right and fails.

The runner gap found during the invalid pilot was still repaired: an active
environment at runner-horizon exhaustion now serializes as `success=false` and
exports its HDF5 data without executing an extra action.

Raw results:

- Released native left:
  `/home/ali/projects/RoboLab/output/cosmos_edge_steerability_left_pilot/`
- Neutral exact matched pair:
  `/home/ali/projects/RoboLab/output/cosmos_edge_steerability_neutral_matched_pair/`
- Matched task definitions:
  `/home/ali/projects/RoboLab/robolab/tasks/benchmark/rubiks_cube_{left,right}_of_bowl_matched.py`

## Secondary fixed-observation diagnostics

The offline probe uses success episode 1040 from `nvidia/Cosmos3-DROID`, whose
task is to move a bowl left. Each request uses the same three real camera frames,
seven joint positions, gripper state, model sampling state, and server settings.
Five offsets in the episode were tested: 0.25, 1, 2, 3, and 4 seconds.

| Diagnostic | Result |
| --- | ---: |
| Exact-repeat action RMS | 0 at 5/5 offsets |
| Exact-repeat video pixel RMS | 0 at 5/5 offsets |
| Mean left/right action RMS | 0.03437 |
| Mean left/paraphrase action RMS | 0.03653 |
| Mean left/unrelated action RMS | 0.04810 |
| Opposite action delta > paraphrase | 1/5 offsets |
| Opposite action delta > unrelated | 1/5 offsets |
| Mean left/right video pixel RMS | 12.0209 |
| Mean left/paraphrase video pixel RMS | 11.0825 |
| Mean left/unrelated video pixel RMS | 15.5698 |
| Opposite video delta > paraphrase | 2/5 offsets |
| Opposite video delta > unrelated | 1/5 offsets |

Interpretation: Cosmos is exactly repeatable and prompt-dependent in this
configuration, but the opposite spatial command is not reliably more
distinctive than a same-goal paraphrase or an unrelated command. These
distances establish sensitivity, not correctness. Pixel RMS is particularly
weak because it does not identify which object moved or whether the requested
relation was established.

At the one-second observation, the full command-style probe produced:

| Condition vs canonical left | Action RMS | Video pixel RMS |
| --- | ---: | ---: |
| Right counterfactual | 0.03576 | 14.3125 |
| Left paraphrase | 0.05501 | 19.2518 |
| Grasp subtask | 0.02848 | 16.4437 |
| Atomic left | 0.03627 | 15.4316 |
| Gripper trace, text only | 0.10301 | 19.6252 |
| Point, text only | 0.05229 | 23.0371 |
| Combined left task | 0.01741 | 9.9423 |
| Unrelated drawer control | 0.07774 | 20.2037 |

The point and trace rows are negative capability probes. Chen et al. also
serialize coordinates as text tokens, but their coordinates are grounded in
the current image. These pilot coordinates were not calibrated to the selected
DROID view, so they measure response to coordinate-shaped text rather than a
valid grounded point/trace interface.

Probe implementation and raw outputs:

- `/home/ali/cosmos-framework/scripts/cosmos_droid_steerability_probe.py`
- `/home/ali/cosmos-framework/outputs/cosmos_droid_steerability_valid_ep1040_seed0/`
- `/home/ali/cosmos-framework/outputs/cosmos_droid_steerability_valid_ep1040_offset_*_seed0/`

## Visual evidence

- `cosmos_closed_loop_montage.jpg`: top row is released native-left success;
  middle is neutral-start left failure; bottom is matched neutral-start right
  success.
- `cosmos_real_droid_observation.jpg`: the three-view real DROID observation.
- `cosmos_imagined_future_grid.jpg`: decoded futures for the command-style
  probe. Visual coherence is not treated as task compliance.

## Engineering findings

- Server action schema must be `joint_pos`, dimension 8.
- RoboLab consumes 32 actions per open-loop chunk; a 16-action server response
  fails when the client requests action 17.
- Output guardrails are irrelevant to this action/video service and otherwise
  trigger a gated guardrail-model dependency.
- The host's 535.309.01 driver can appear as 535.53 through a 535-series Vulkan
  minor-version overflow; the test used a targeted Isaac driver-version check
  bypass after verifying the actual driver.
- The schema-correct model server used GPU 0; Isaac/RoboLab used GPU 1.

## Claim

Cosmos3 Edge DROID is runnable, competent in both directions across separate
pilots, and succeeds on the right member of an exact neutral-start pair. The
matched left member fails and both neutral endpoints move robot-right. Combined
with weak offline semantic selectivity, this is real but one-sided pilot
evidence—not robust steerability.
