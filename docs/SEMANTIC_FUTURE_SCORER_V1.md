# Frozen semantic-future scorer v1

Status: **frozen before confirmation inference** on 2 August 2026. Calibration
and dry-run evidence uses only the excluded seed-prefix 51xx rollouts. None of
the 6100--6109 confirmation episodes existed when these rules were committed.

## What is measured

For every Cosmos policy replan, RoboLab records the conditioning image, action
chunk, sampling seed, decoded future, and the exact environment step at which
the chunk begins. The scorer uses four non-conditioning future frames (8, 16,
24, and 32), estimates the Rubik's cube and red-bowl centers independently in
the two third-person camera panels, and projects them into the table plane.

The Qwen3-VL 2B localizer is deliberately prompt-blind: it sees one camera
image and an object-localization request, never the policy instruction or the
requested direction. The policy prompt therefore cannot leak into the visual
predicate label.

The semantic relation is the same 45-degree robot-frame cone used by the
RoboLab left/right task. Executed-state scoring additionally requires the
cube/bowl height difference to be at most 0.1 m. A future frame is reliable
only when:

- both object centers are present in both cameras;
- both cameras predict the same categorical relation; and
- neither object's cross-camera reconstructed position differs by more than
  0.20 m.

A chunk needs at least two reliable frames. It is `true` when at least 75% of
reliable frames show the requested relation, `false` when at most 25% do, and
`uncertain` otherwise. Frame zero is conditioning and never counts as future
evidence.

The execution comparison uses the state after the actions actually executed
from the chunk, at `start_step + open_loop_horizon - 1`. This supports four
semantic quadrants rather than treating any pixel change as steerability:

1. imagines requested and executes requested;
2. imagines requested and does not execute requested;
3. does not imagine requested and executes requested;
4. neither imagines nor executes requested.

Uncertain futures are reported as coverage failures and are never coerced into
success or failure.

## Frozen calibration

The localizer checkpoint is the local Qwen3-VL-2B-Instruct snapshot
`89644892e4d85e24eaac8bacfd4f463576704203`. Exact RoboLab camera intrinsics
and extrinsics perform the pinhole projection. Calibration estimates only:

- the shared cube/bowl centroid plane: 0.078552 m;
- a per-camera normalized-pixel bias: [28.5864, 21.3063] left and
  [31.3348, 36.0578] right;
- the 0.20 m reliability threshold, rounded upward from the excluded future
  dry run's 0.187 m 90th-percentile disagreement.

Across the 19 excluded conditioning frames, both camera relations matched
simulator ground truth on 18/19 frames (94.7%). The left-camera world-position
residual was 0.0276 m median and 0.0556 m at p90; the right-camera residual was
0.0269 m median and 0.0844 m at p90. The conditioning cross-camera
disagreement p90 was 0.0922 m.

The complete calibration, including every raw localizer response and projected
point, is in
`artifacts/vla_wam_shared_v1/semantic_future_calibration.json` (SHA-256
`56b7155fdb2eee1732e7636a104b2966323fcfc947c699cf3ac58635b232ace9`).

## Excluded dry-run behavior

Two seed-5100 Cosmos episodes were used: a full-horizon failed left command
(15 chunks) and a successful right command (4 chunks). The frozen scorer gave
15/19 certain chunks (78.9% coverage): 14 neither-imagines-nor-executes, one
imagines-and-executes, and four uncertain. The single positive quadrant is the
last chunk of the successful right episode. This is a useful sanity check, not
a performance estimate.

The auditable outputs are under
`artifacts/vla_wam_shared_v1/calibration_semantic_dry_run/`. The contact sheets
draw every prompt-blind localization (cyan cube, red bowl) over both cameras.
Their SHA-256 hashes are:

- failed-left sheet:
  `43e134d802b8c4b2eb9d9687519110fe38b45085475df4eaef6ad40fd896abb1`;
- successful-right sheet:
  `b7d83e2b62867eb9470971a624138b77b55bc2349bc0ca8f521061db28773d5c`.

## Known limitations

This is an auditable proxy, not perfect physical perception. Qwen localization
has non-zero bias, generated video can deform or occlude objects, and a planar
projection cannot recover object height. Requiring two-camera categorical
agreement limits false certainty at the cost of abstentions. The 0.20 m
cross-camera tolerance is loose relative to calibration error and was chosen
on a very small excluded sample. Confirmation results must therefore report
coverage, contact-sheet audits, and sensitivity to stricter thresholds. No
threshold may be changed after inspecting confirmation outcomes.

## Reproducibility pins

- scorer: `tools/score_cosmos_semantic_futures.py`;
- RoboLab recorder commit: `26c79ffb6a3096951991f490ff8af4d80a1073cf`;
- Cosmos server commit: `1439c1d5e45a23771e9b1a2ad8f40a5981ea86c0`;
- π0.5 request-seeding commit: `9e46d3aea26417bfb564227734b95d010aa827e5`.

All three model/evaluation changes make request-scoped sampling deterministic.
RoboLab uses `episode_seed * 1000 + replan_index`, so resume and paired-prompt
runs retain an inspectable seed schedule.
