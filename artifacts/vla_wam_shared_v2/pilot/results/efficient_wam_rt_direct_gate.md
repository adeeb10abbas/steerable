# Efficient-WAM-RT standardized direct-command pilot

Compiled at `2026-08-03T03:23:57.632176+00:00` from 6 executed episodes in 3 exact left/right scene pairs.

## Result

- LEFT: **2/3** requested-relation successes.
- RIGHT: **0/3** requested-relation successes.
- Overall: **2/6**.
- Prompt-ignored/native-task-completed failures: **1**.
- Wall-latency-valid episodes: **6/6**.

Direct-command success occurred for LEFT only, so the frozen gate calls for a ten-scene direct-command directional-bias confirmation before any four-wording sweep.

This six-episode pilot is a gate, not a stable rate estimate.

## Exact command pairs and endpoints

| Pair | LEFT | RIGHT | Final x: LEFT | Final x: RIGHT | RIGHT - LEFT | Response |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| pair00 | success | failure | -0.113 m | -0.194 m | -0.081 m | anti directed |
| pair01 | failure | failure | +0.236 m | -0.183 m | -0.419 m | anti directed |
| pair02 | success | failure | -0.123 m | -0.111 m | +0.012 m | aligned |

## Measurement boundary

The model receives only the frozen language prompt and observation. Simulator state is used after each action solely for scoring and visualization; no oracle actions, subtask coach, dynamic prompt, or online correction is used.

`verified_pickup_proxy` means at least 3 consecutive recorded states with the movable object lifted at least 0.03 m above its initial height while the gripper is reported closed. It is a transparent diagnostic proxy, not a learned semantic judgment.

Pixel differences between imagined futures are deliberately not used as semantic steerability evidence. The raw imagined artifacts are hash-locked here for later predicate scoring and imagination-versus-execution analysis.
