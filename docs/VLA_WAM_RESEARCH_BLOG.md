# Can language steer robot policies left versus right?

Changing one word in a robot instruction should change what the robot does. We
tested that simple expectation across vision-language-action models (VLAs) and
world-action models (WAMs): from the same starting scene, ask for the same
object to be placed either **left** or **right** of the same target.

The result is more nuanced than a success-rate leaderboard. Most tested models
changed their actions when the direction changed, and many moved the object
endpoints in the requested order. Reliable task completion was much less
common. A policy can therefore be language-sensitive without being competent,
and it can imagine a plausible future without executing it successfully.

> **Claim boundary.** These are checkpoint- and simulator-specific results.
> DROID/RoboLab and RoboTwin use different tasks and success predicates, so
> their success rates are reported separately and never pooled.

## The experiment

Each matched pair begins from the same physical state and random seed. The only
experimental change is a static episode-level command:

- “Put the cube left of the bowl.”
- “Put the cube right of the bowl.”

There is no oracle, subtask coach, prompt switching, or progress-conditioned
language. Every valid episode keeps its full simulator video and executed
action trace. Failures remain evidence. Startup errors, rendering failures, and
partial runs are logged separately and excluded from behavioral denominators.

We report three different questions:

1. **Task success:** did the episode satisfy the requested spatial relation?
2. **Endpoint alignment:** within a matched pair, was the RIGHT endpoint to the
   right of the LEFT endpoint?
3. **Action distinctness:** did the model execute different actions after the
   language intervention?

The last two are sensitivity measures, not substitutes for success.

## Results

### DROID / RoboLab

| Model | Type | Valid episodes | LEFT | RIGHT | Aligned pairs | Distinct-action pairs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| π0-FAST DROID | VLA | 20 | 1/10 | 10/10 | 10/10 | 10/10 |
| GR00T N1.7 DROID | VLA | 6 | 0/3 | 0/3 | 3/3 | 3/3 |
| Cosmos3 Edge DROID | WAM | 6 | 3/3 | 3/3 | 3/3 | 3/3 |
| DreamZero DROID | WAM | 6 | 2/3 | 1/3 | 3/3 | 3/3 |
| π0.5 DROID — current-stack V2-A010 | VLA | 6 | 1/3 | 3/3 | 3/3 | 3/3 |
| Cosmos3 Nano Policy DROID — V2-A011 | WAM | 6 | 3/3 | 3/3 | 3/3 | 3/3 |

### RoboTwin place-A-relative-to-B

| Model | Type | Valid episodes | LEFT | RIGHT | Aligned pairs | Distinct-action pairs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LingBot-VLA 4B | VLA | 6 | 1/3 | 0/3 | 2/3 | not reported |
| Efficient-WAM-RT | WAM | 14 | 3/7 | 2/7 | 6/7 | 7/7 |
| FastWAM | WAM | 14 | 1/7 | 1/7 | 3/7 | 7/7 |
| LingBot-VA | WAM | 14 | 3/7 | 4/7 | 6/7 | 7/7 |
| Light-WAM | WAM | 6 | 1/3 | 0/3 | 1/3 | 3/3 |

The main pattern is visible without pooling arenas: action redirection is
common, but task success varies sharply by checkpoint and requested direction.
GR00T is the clearest example. All three matched pairs produced distinct
actions and correctly ordered endpoints, yet all six episodes failed the task.
π0-FAST shows a different limitation: it completed nearly every RIGHT request
but almost no LEFT request despite aligned endpoints in every pair.

Cosmos3 Edge and the separate current-stack Cosmos3 Nano Policy DROID gate
each completed all six bounded DROID episodes. Nano retained 37 decoded RGB
futures, and its three matched endpoint shifts all followed the requested
LEFT-to-RIGHT order. The RoboTwin WAMs were less reliable, although
Efficient-WAM-RT and LingBot-VA aligned six of seven endpoint pairs.

Two newer Cosmos3 base-model arms are frozen as feasibility work, not results.
Cosmos3 Super base V2-A012 and Cosmos3 Edge base V2-A013 each have an exact
three-request fixed-observation action-plus-future probe pending and zero
released behavioral cells. Edge base is distinct from the completed Edge
Policy DROID row above. A possible 10D CuRobo-derived controller is a separate
branch and remains blocked until its exact Franka+Robotiq asset and control
mapping are verified. Neither pending arm has gallery media.

## Dreaming is not executing

Some WAMs expose a future representation; others expose only actions or latent
features. We retain a future only when the model actually exposes it and never
turn a missing or latent-only future into a zero.

DreamZero is especially useful because its official decoder lets us inspect
what the model predicted. Its fixed-observation probe produced repeatable,
prompt-dependent imagined videos. In behavior, however, DreamZero completed
only three of six episodes. The evidence supports a narrow conclusion: it was
able to generate a direction-conditioned dream, but that did not guarantee
successful execution.

The [video gallery](VLA_WAM_STEERABILITY_VIDEO_GALLERY.html) places simulator
rollouts beside exposed imagined futures. It labels historical reference media
and missing behavioral media explicitly; no other model's clip is substituted.

## What this study does—and does not—show

It shows that language can redirect several evaluated policies under controlled
matched-state interventions. It also shows why “the actions changed” is not a
complete robotics result: changed actions may still miss the requested goal.

It does **not** establish a general VLA-versus-WAM ranking. The checkpoints,
action spaces, horizons, simulators, and future interfaces differ. The sample
sizes are deliberately bounded, and model families are not equally represented.
The useful comparison is within each checkpoint: LEFT versus RIGHT from the
same state.

## Reproducibility

The repository keeps the frozen protocol, exact episode registries,
hash-bearing compact evidence, invalid-attempt ledgers, and selected media in
Git. Full raw rollouts, environments, and checkpoints remain on persistent
storage rather than being committed.

- [Frozen protocol](VLA_WAM_STEERABILITY_V2_PROTOCOL.md)
- [Current cross-model evidence table](../artifacts/vla_wam_shared_v2/results/direct_command_cross_model_comparison.md)
- [Operational continuation state](../artifacts/vla_wam_shared_v2/continuation_state.json)
- [Full technical record](VLA_VS_WAM_STEERABILITY_STUDY.md)

The π0-FAST current-stack wording replication stopped at its release gate: on
a fixed observation, repeat LEFT was deterministic but LEFT and RIGHT produced
the same action tensor (RMS 0.0). It has zero behavioral episodes and contributes
neither successes nor failures. The separate π0.5 current-stack V2-A010 gate is
complete: all six episodes are valid, with LEFT 1/3 and RIGHT 3/3, three aligned
endpoint pairs, and three distinct action pairs. Its selected media is an actual
rollout, not historical v1 footage and not an imagined future. Cosmos3 Nano
Policy DROID V2-A011 is also complete as a separate current-stack gate (6/6
successes; 37 retained decoded futures). The gallery places Nano’s actual
execution beside a clearly labelled model prediction; predictions are not
additional trials. The Super-base and Edge-base feasibility records contain no
behavioral result or media and must not be added as zeros. None may be merged
with historical evidence. LaWAM was
withdrawn before inference and is outside the active study.
