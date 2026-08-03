# pi0-FAST DROID direct-command pilot

This is a six-episode, oracle-free base-competence gate—not a stable population rate.
The scene and seed are matched inside each pair; only `LEFT` versus `RIGHT` changes.

## What was tested

- LEFT: `Put the Rubik's cube to the left of the bowl.`
- RIGHT: `Put the Rubik's cube to the right of the bowl.`
- Environment/sampling seeds: 8300, 8301, and 8302.
- One episode-static prompt; no subtask coach, predicate oracle, or dynamic prompt.
- Primary outcome: released cube in the requested 45-degree bowl-relative region.
- Transparent pickup proxy: cube lifted at least 0.03 m for 3 consecutive recorded steps.

## Result

- LEFT: **0/3**.
- RIGHT: **3/3**.
- Overall: **3/6**.
- Same-seed endpoint redirection aligned with LEFT→RIGHT in **3/3** pairs.
- First-chunk action RMS was non-zero in **3/3** pairs.

| Seed | LEFT | RIGHT | LEFT endpoint | RIGHT endpoint | Redirected? |
| ---: | --- | --- | ---: | ---: | --- |
| 8300 | failure | success | +0.003 m | +0.377 m | aligned |
| 8301 | failure | success | +0.006 m | +0.168 m | aligned |
| 8302 | failure | success | -0.084 m | +0.167 m | aligned |

Negative endpoint values are robot LEFT; positive values are robot RIGHT.
The endpoint shifted rightward in all three pairs, but none of the LEFT runs completed. That is evidence of prompt-conditioned behavior plus a severe directional/base-competence asymmetry, not robust steerability.

## Frozen gate decision

**expand_direct_directional_bias_only.** Direct-command success occurred for RIGHT only. The frozen gate calls for the ten-seed direct-command directional-bias confirmation before any four-wording sweep.

All six videos, failures included, are retained. Simulator state was used only after action execution for scoring and visualization.
