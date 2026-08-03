# FastWAM standardized direct-command pilot

Compiled at `2026-08-03T02:42:13.433587+00:00` from 6 executed episodes in 3 exact left/right scene pairs.

## Result

- LEFT: **1/3** requested-relation successes.
- RIGHT: **0/3** requested-relation successes.
- Overall: **1/6**.
- Prompt-ignored/native-task-completed failures: **1**.

Direct-command success occurred for LEFT only, so the frozen gate calls for a ten-scene direct-command directional-bias confirmation before any four-wording sweep.

This six-episode pilot is a gate, not a stable rate estimate.

## Exact command pairs and endpoints

| Pair | LEFT | RIGHT | Final x: LEFT | Final x: RIGHT | RIGHT - LEFT | Response |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| pair00 | failure | failure | -0.215 m | -0.178 m | +0.037 m | aligned |
| pair01 | failure | failure | +0.073 m | +0.027 m | -0.046 m | anti directed |
| pair02 | success | failure | -0.152 m | -0.123 m | +0.029 m | aligned |

## Measurement boundary

The model receives only the frozen language prompt and observation. Simulator state is used after each action solely for scoring and visualization; no oracle actions, subtask coach, dynamic prompt, or online correction is used.

`verified_pickup_proxy` means at least 3 consecutive recorded states with the movable object lifted at least 0.03 m above its initial height while the gripper is reported closed. It is a transparent diagnostic proxy, not a learned semantic judgment.

This inference interface emits actions but no test-time imagined-video artifact, so imagination-versus-execution metrics are recorded as not applicable rather than zero.
