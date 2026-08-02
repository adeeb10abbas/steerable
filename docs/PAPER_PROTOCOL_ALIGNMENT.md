# Alignment with Chen et al. (2026)

Source: William Chen et al., *Steerable Vision-Language-Action Policies for
Embodied Reasoning and Hierarchical Control*, arXiv:2602.13193. This document
records what the local VLA-WAM study reproduces, adapts, and does not claim to
reproduce.

## What the paper evaluates

The paper trains low-level VLAs on a mixture of six steering-command styles:

1. task;
2. subtask;
3. atomic motion;
4. gripper trace;
5. point;
6. combinations of those styles.

Its human-oracle and learned-reasoner experiments report success rate across
in-distribution, motion, spatial, and semantic generalization splits. Its
off-the-shelf in-context VLM experiments use harder multi-step tasks and report
task progression: the mean fraction of task-specific binary rubric items
completed. Rubric items need not be completed in order. Later actions revoke
credit when they undo progress, except that first object interaction or pickup
credit persists.

For the paper's spatial multi-step tasks, Table III gives two rubric items:

1. pick up the correct object;
2. put down the correct object in the correct location.

The learned embodied reasoner is re-queried every five environment steps. The
off-the-shelf in-context VLM is re-queried every 20 low-level steps and receives
observation/command history. The human oracle may intervene freely but must
wait at least two seconds between interventions. These are distinct protocols;
their cadences should not be conflated.

Appendix A states that grounded pixel coordinates are normalized to integers
from 0 to 255. The first value is the image column and the second is the row.
The local command probe preserves raw source-image pixels for visual audit but
serializes 0-to-255 coordinates in the model prompt.

The appendix text is internally inconsistent about the origin: it describes a
column measured from the left and a row measured from the top, then calls
`[0,0]` the top-right corner. We use the conventional top-left origin implied
by the first two definitions and retain the annotated source image so this
choice is auditable.

## Local mapping

| Paper concept | Local implementation | Alignment |
| --- | --- | --- |
| Requested-goal success | RoboLab success predicate: correct 45-degree left/right relation with cube released | Direct task-level analogue |
| Spatial task progression item 1 | Correct-cube `OBJECT_GRABBED_SUCCESS`, persistent | Direct analogue |
| Spatial task progression item 2 | RoboLab requested-side success, including gripper detachment | Direct analogue |
| No ordering requirement | Paper progression scores both items independently | Direct |
| Ordered pick-and-place | Additional post-pick relation-transition metric | Declared local extension |
| Four generalization splits | Neutral-start left/right DROID scene | Spatial slice only |
| Six command styles | Frozen fixed-observation probe | Interface diagnostic, not trained-style reproduction |
| Five-step learned reasoner | Not run | Deliberately outside the direct-grounding question |
| Twenty-step in-context VLM | Not run | Missing |
| Human oracle | Not run | Missing |
| 0-to-255 grounded coordinates | Simulator-projected points in actual left camera, converted to 0-to-255 | Direct serialization analogue |
| Standard-error bars | Numerators plus Beta(1,1) 95% posterior intervals | Different uncertainty summary, fully disclosed |

## Why the comparison remains useful

The standard π0.5 DROID VLA and Cosmos3 Edge DROID WAM receive the same image
state, prompts, actions schema, task predicates, episode seeds, and simulator
horizon. That controls several confounds that make ordinary cross-paper VLA/WAM
leaderboards uninterpretable. Cosmos additionally emits an imagined future, so
the study can ask whether the requested relation appears in prediction and
whether execution agrees.

The comparison does **not** show whether the paper's steering-command training
recipe benefits WAMs. Neither checkpoint was trained by that recipe, only two
model instances are tested, and three generalization splits are absent. A
positive result supports language responsiveness for these checkpoints on this
task; a negative result may reflect checkpoint data, prompt interface, action
head, scene shift, or control horizon rather than the whole model class.

## Source-driven amendments

Three corrections or scope amendments were made before their affected analyses
ran:

- `command_probe_amendment_001.json` converts grounded prompts from raw pixels
  to the paper's 0-to-255 range before any fixed-observation request;
- `metric_amendment_001.json` adds the paper's explicit put-down requirement to
  progression item 2 before final compilation;
- `direct_language_scope_amendment_003.json` retires the privileged coach and
  freezes two task-level prompt stressors before any associated rollout.

The metric correction occurred after the first Cosmos batch began, so it is
reported as an amendment rather than misrepresented as part of the initial
preregistration. It tightened the rubric and changed no policy input, episode,
or stopping rule.

The scope change happened after some original Cosmos and preliminary
five-step outcomes were visible, so the new declarative/contrastive grid is
reported as a prospectively frozen **post-interim stress tier**, not rewritten
into the original preregistration. This distinction is central to the study's
claim boundary.
