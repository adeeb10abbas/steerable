# A paper-faithful steerability benchmark for world-action models

Status: protocol v0.1. The command taxonomy and primary outcomes follow
Chen et al., *Steerable Vision-Language-Action Policies for Embodied Reasoning
and Hierarchical Control* (2026). The action/video diagnostics are additional
tests needed for world-action models (WAMs); they are not metrics reported by
that paper.

## What is being tested

A WAM is steerable only if changing a command can cause an appropriate change
in its imagined future **and** its executed behavior. These are separate claims:

1. **Repeatability:** identical inputs and sampling state reproduce.
2. **Sensitivity:** a command change alters the action or future prediction.
3. **Semantic direction:** that alteration is toward the requested outcome.
4. **Closed-loop control:** the altered behavior completes the requested goal.
5. **Robustness:** control survives seeds, scenes, objects, paraphrases, and
   command abstractions.

Only levels 4 and 5 support a headline claim that a WAM is steerable. Pixel
MAE, embedding distance, or action L2 can establish sensitivity, but not intent
or control.

## Command styles

Each executable task should be annotated with the six interfaces used in the
paper. A style may be reported as unsupported, but it must not be silently
replaced by another style.

| Style | Example | WAM-specific question |
| --- | --- | --- |
| Task | “Put the cube left of the bowl.” | Can one semantic goal condition both future and action? |
| Subtask | “Pick up the cube.” | Can a high-level controller compose familiar skills? |
| Atomic motion | “Move the gripper left.” | Does language expose local, low-level control? |
| Gripper trace | “Move through [(x1,y1), ...].” | Can an image-space path control action and imagined motion? |
| Point | “Grasp at <cube position>.” | Can grounding disambiguate the relevant object/location? |
| Combination | “Move left to <cube position>, then close the gripper.” | Do grounded and linguistic constraints compose? |

Commands are issued at task start and at verified state transitions such as
first grasp or first lift. Timer-only switching is exploratory, not a valid
hierarchical controller, because two rollouts can reach different physical
states at the same step.

## Primary outcomes: directly comparable to the paper

### Success rate

For model `m`, command style `c`, and evaluation split `g`:

```text
SR(m,c,g) = successful requested-goal episodes / attempted valid episodes
```

Report the numerator and denominator, not only a percentage. Pair trials by
scene seed, simulator seed, model sampling seed, and initial state. Give a 95%
Beta-binomial credible interval. Setup failures and non-expert-valid scenes are
reported separately and never counted as model failures.

Use the paper's four slices:

- **In-distribution:** familiar objects, layouts, task semantics, and wording.
- **Motion generalization:** a novel local motion or motion composition.
- **Spatial generalization:** a familiar skill with a changed spatial relation.
- **Semantic generalization:** a novel object/category/affordance reference.

### Task progression

Long-horizon episodes receive a task-specific success/fail rubric. The paper
does **not** require rubric items to be completed in order. At evaluation time:

```text
progress = credited rubric items / total rubric items
```

Credit for a stateful item is revoked if the policy later undoes it. The paper's
exception is first interaction/pickup credit, which persists. Its reported
metric is final average task progression across trials. Maximum progression and
area under the progression curve are useful WAM diagnostics, but are additions
to the paper rather than paper-primary metrics. If RoboLab's built-in subtask
score is used, preserve its raw score and additionally map its predicates into
the rubric rather than claiming the two definitions are identical. In
particular, an initially detached object must not earn “put down” credit before
the robot acts.

### Intervention protocol

Record the high-level intervention cadence in low-level environment steps.
The paper's learned embodied reasoner emits a command that the low-level policy
follows for **5 environment steps** before another high-level query. Its
in-context experiments run for **20 high-level steps** (25 for the second
long-horizon suite); 20 is an episode budget, not an intervention cadence. Its
human-oracle study allows interventions no more frequently than every 2
seconds. For WAM comparisons, add predicate-triggered interventions. Always
log:

- the exact command and style;
- the observation/action index at which it became active;
- the physical predicate that triggered it;
- whether the unexecuted action chunk was discarded;
- a hash of the trajectory prefix for matched interventions.

## Secondary outcomes: WAM diagnostics

These diagnose why a WAM succeeds or fails. They must not be mixed into the
paper-primary success table.

### Prompt effect relative to stochasticity

For the same observation, compare paired command distance with repeated-sample
distance:

```text
action_effect = RMS(action_A - action_B)
noise_floor   = median RMS(action_A(seed_i) - action_A(seed_j))
effect_ratio  = action_effect / noise_floor
```

Compute the same quantities for predicted-video latents and, where decoding is
stable, semantic object tracks. A ratio above one is evidence of a resolvable
prompt effect, not evidence that the effect is correct.

### Semantic future compliance

Evaluate the imagined future with the same requested-goal predicate used by
the simulator wherever possible: left/right relation, containment, contact,
release, or ordered subgoal completion. Report:

- predicted goal-predicate rate;
- predicted directional displacement;
- first predicted frame satisfying the predicate;
- agreement between predicted predicate and eventual executed predicate.

This is more informative than pixel MAE because two visually different videos
can represent the same plan, while a small pixel difference can hide a
task-critical relation change.

### Action–future consistency

The imagined robot/object motion should agree with the action chunk. Compare
the sign and coarse magnitude of predicted end-effector/object displacement
against forward-simulated execution. Report relation-predicate agreement and
trajectory correlation when coordinates are commensurate. A model that
imagines “left” while acting “right” is not a coherent WAM interface.

### Intervention latency and persistence

After a command switch, measure:

- chunks until the action first shifts in the requested direction;
- environment steps until the object/end-effector responds;
- whether the shift persists through the next replan;
- whether the new requested predicate is reached and retained.

## Required controls

Every prompt experiment includes:

1. exact-repeat determinism;
2. a semantic paraphrase;
3. the matched opposite relation or goal;
4. an unrelated but grammatical command;
5. a noun-only change that should not alter the relation;
6. a same-direction reissue at the intervention point;
7. an impossible or contradictory command, labeled as a safety/selectivity
   probe rather than a success trial.

The opposite-goal pair changes the smallest possible semantic unit while all
other inputs remain fixed. The same-direction reissue controls for chunk reset
and replanning effects.

## Cross-model reporting table

Every model gets one row per command style and split, with explicit support
status:

| Model | Interface support | Native SR | Counterfactual SR | Task progression | Future compliance | Action–future agreement | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Efficient-WAM | task/subtask/atomic/combination tested | measured | measured | pending rubric remap | coarse video | pending | RoboTwin |
| FastWAM | task tested; repaired text CFG | measured | measured | pending | available | pending | fragile across seeds |
| LingBot-VA | task tested | measured | measured | pending | latent future measured | pending | deterministic, slower |
| Cosmos3 Edge DROID | task supported; other styles under test | pending | pending | RoboLab score available | decoded video available | pending | official RoboLab path |
| Standard π0.5 DROID VLA | task tested | measured | measured | paper-style geometric proxy | not applicable | not applicable | non-WAM matched control |

“Unsupported” and “not yet tested” are distinct. Zero-shot failure on a
command style absent from a checkpoint's training data measures interface
bandwidth, not the value of the paper's command-mixture training method.

## Minimum evidence tiers

- **Pilot:** one valid scene, both goal directions, three matched model seeds.
- **Discovery:** at least five valid scenes and ten paired trials per condition.
- **Estimate:** preregistered task/seed grid with uncertainty intervals and no
  post-hoc prompt or guidance selection.
- **General steerability claim:** all four generalization slices, more than one
  command abstraction, and held-out scenes/objects.

Results below the relevant tier stay useful, but the article labels them with
the tier instead of converting them into a broad claim.
