# Does the world model listen?

## A stress test of steerability in world-action models

*Draft, 2 August 2026. Results are pilots unless a larger evidence tier is
explicitly stated.*

A robot model can accept a sentence, change its action tensor, and generate a
different-looking video without doing what the sentence asked. That distinction
sounds obvious. It becomes surprisingly easy to forget once the result is a
colorful pairwise-distance heatmap.

That was the mistake in my first experiment. I held an observation fixed, asked
for left, right, a paraphrase, and an unrelated task, then measured action L2 and
imagined-video pixel error. The plots showed non-zero differences. They did not
show language grounding. The unrelated command sometimes moved the output more
than the left/right counterfactual. A model that reacts to text is not
necessarily a model that is controlled by text.

So I rebuilt the test around a harder question:

> If I preserve the scene, physical history, model seed, and every word except
> the requested goal, does the robot reach the newly requested goal?

I tested four publicly runnable world-action models (WAMs): Efficient-WAM,
FastWAM, LingBot-VA, and NVIDIA's Cosmos3 Edge Policy for DROID. The answer is
not a clean yes or no. There are real signs of life. Efficient-WAM can redirect
an already grasped object after a command switch from an identical trajectory
prefix. Cosmos can solve a native spatial task in RoboLab. LingBot produces
deterministic, language-dependent future latents. But none of the tested
releases yet supports the broad, reliable, multi-abstraction steering interface
that the word *steerable* suggests.

My practical conclusion is:

> **Efficient-WAM is the best core for fast steerability research today.
> Cosmos3 Edge is the strongest systems-level comparison. Neither is yet a
> generally steerable WAM.**

That qualified conclusion is more useful than either extreme. “The models
ignore language” is false. “The models are language steerable” is also too
strong. The interesting research lives between those statements.

## What steerability should mean for a WAM

The benchmark design is adapted from Chen et al.'s
[Steerable Vision-Language-Action Policies for Embodied Reasoning and
Hierarchical Control](https://arxiv.org/abs/2602.13193). Their key idea is that
a useful low-level policy should accept commands at several levels of
abstraction, not only the task captions commonly found in robot datasets. They
train with six command styles:

1. **Task:** “Put the cube left of the bowl.”
2. **Subtask:** “Pick up the cube.”
3. **Atomic motion:** “Move the gripper left.”
4. **Gripper trace:** follow a sequence of image-space points.
5. **Point:** interact at a grounded image location.
6. **Combination:** compose language, motion, grounding, and gripper state.

This taxonomy becomes even more important for a WAM. A VLA policy has one
observable obligation: produce useful actions. A WAM makes two coupled
promises: produce actions and imagine the future associated with those
actions. Steering can therefore fail in at least three ways:

- the action does not follow the command;
- the imagined future does not follow the command;
- the action and imagined future follow different commands.

I use a six-level claim ladder to keep these possibilities separate.

| Level | Claim | Evidence required |
| --- | --- | --- |
| 1 | Repeatability | Exact input and sampling state reproduce |
| 2 | Sensitivity | A command change alters action or future beyond noise |
| 3 | Semantic direction | The change points toward the requested outcome |
| 4 | Task competence | The model solves the released native task |
| 5 | Closed-loop control | A matched counterfactual completes its requested goal |
| 6 | Robust steerability | Control survives scenes, seeds, objects, paraphrases, and abstractions |

Levels one and two are diagnostics. Levels five and six justify a claim about
steerability. Most attractive fixed-observation plots stop at level two.

## Using the paper's metrics without pretending the experiments are identical

Chen et al. report closed-loop **success rate** across in-distribution, motion,
spatial, and semantic generalization suites. For longer tasks controlled by a
high-level VLM, they also report rubric-based **task progression**. Those are
the primary metrics in this WAM study too.

For calibration inside their own benchmark, the full embodied-reasoner system
reports 83.7% ± 3.4% aggregate success, while their full in-context high-level
system reports 84% ± 3.7% aggregate task progression versus 48% ± 5.1% for
standard OpenVLA. Those numbers are a design target, not a baseline to compare
numerically with the tiny WAM pilots below: the tasks, robots, training data,
and sample counts differ.

For each model, command style, and split:

```text
success rate = requested-goal successes / valid attempted episodes
```

The denominator matters. A broken simulator, invalid expert scene, or wrong
action schema is a setup failure, not a model failure. Conversely, a rollout
that runs the complete valid horizon without satisfying the requested goal is
a failure even if the recorder forgets to serialize its timeout.

Long-horizon tasks use an ordered progression rubric. For a pick-and-place
relation, a simple version is:

1. reach and grasp the requested object;
2. transport it toward the requested relation;
3. establish the relation and release safely.

I record final progression, maximum progression, and area under the progression
curve. Final progression is the closest match to the paper. Maximum progression
distinguishes a robot that made progress and regressed from one that never
started. AUC rewards early, persistent progress. Simulator-native subtask scores
are retained but not automatically treated as identical to an ordered rubric.
For example, a “dropped” predicate can be true before the robot has ever grasped
the object, producing a misleading one-third raw score.

The WAM-specific quantities—action RMS, future-latent RMS, pixel RMS, prompt
effect relative to sampling noise, predicted-goal compliance, and
action–future agreement—are **secondary diagnostics**. They explain a success
or failure. They do not replace it.

This distinction changes how the results should be read. The current model
comparisons are predominantly pilots in the spatial-generalization slice. They
are not a completed reproduction of the paper's four-suite evaluation, and the
models were not trained with its two-million-command augmentation pipeline.

## A better experiment than “left versus right”

Left/right is a good minimal pair, but a bad complete benchmark.

It is good because the intervention is surgical. In a matched comparison, I
can reset the same simulator state and random seeds, preserve every noun and
verb, and flip only one relation token:

```text
Place the blue soap on the left of the tea-box.
Place the blue soap on the right of the tea-box.
```

That gives a clean causal question. If the endpoint changes, language probably
did something. If the endpoint crosses into the opposite requested success
region, language did something useful.

It is incomplete for four reasons. First, left and right may be asymmetric in a
dataset or camera frame. Second, a model may have memorized two task captions
without exposing reusable control. Third, relation changes do not test object
identity, affordances, or long-horizon composition. Fourth, left/right task
captions do not test lower-level intervention interfaces.

The expanded protocol therefore contains:

- an exact-repeat control;
- a semantic paraphrase;
- the matched opposite goal;
- an unrelated but grammatical command;
- a noun-only change that should preserve the relation;
- a same-direction command reissue at the intervention point;
- task, subtask, atomic, point, trace, and combined commands;
- command switches triggered by physical predicates, not only timers.

That last item matters. Two rollouts can be in different physical states at
step 25. A competent hierarchical controller should intervene after a verified
event such as grasp, lift, or release. Cycling prompts every fixed number of
steps confounds language with progress and can push a policy off-distribution.

## The four WAMs, at a glance

| Model | Strongest positive result | Main negative result | Practical verdict |
| --- | --- | --- | --- |
| **Efficient-WAM-RT** | 6/6 matched endpoints shifted in the requested direction; 3/6 mid-rollout full-task switches succeeded | Severe direction and abstraction asymmetry; coarse future video | Best fast experimental core |
| **FastWAM** | One clean matched left/right counterfactual after repairing text CFG | Prompt effect below sampling noise; counterfactual reproduced only 1/5 seeds | Useful debugging case, weak core |
| **LingBot-VA** | Exact repeat; strong language-dependent future latent; 2/2 native success | 0/2 swapped success; about 7 s per warm chunk | Strong semantic comparison, slow controller |
| **Cosmos3 Edge DROID** | Native-left success plus 1/2 success in an exact neutral-start left/right pair | Neutral-left failed and both neutral endpoints moved toward robot-right; offline selectivity was weak | Real one-sided signal, strong ecosystem comparison |

The public-interface coverage is much narrower than the model list initially
suggests:

| Model | Task | Subtask | Atomic | Point | Trace | Combination |
| --- | --- | --- | --- | --- | --- | --- |
| Efficient-WAM | Closed loop | Closed loop | Closed loop | No native structured test | No native structured test | Closed loop |
| FastWAM | Closed loop | Not tested | Not tested | Unsupported/not tested | Unsupported/not tested | Not tested |
| LingBot-VA | Closed loop | Not tested | Not tested | Unsupported/not tested | Unsupported/not tested | Not tested |
| Cosmos3 Edge | Closed loop | Offline text only | Offline text only | Text-only negative probe | Text-only negative probe | Offline text only |

“Unsupported,” “not tested,” and “tested but failed” are different outcomes.
Collapsing them to zero would unfairly compare an absent interface with an
implemented controller that attempted the command.

These rows are not rankings on one scalar. The checkpoints differ in training
data, action space, simulator, parameter count, generated-future
representation, and maturity of their evaluation stack. The comparison asks a
more practical question: *what kind of steerability evidence can each public
release support today?*

Two conspicuous candidates are absent from the evidence table. [UVA
(Unified Video Action Model)](https://arxiv.org/abs/2503.00200) remains the
smallest attractive released checkpoint at roughly 0.5B parameters, but I do
not yet have a schema-correct matched closed-loop result from its LIBERO policy.
[DreamZero](https://dreamzero0.github.io/) is much larger and operationally
costlier; it remains useful as a later scale comparison rather than the rapid
iteration core. Their omission is “not yet measured,” not a negative result.

## Efficient-WAM: the first genuinely useful experimental organism

[Efficient-WAM](https://arxiv.org/abs/2606.10040) is the most productive model
in this study because it is small enough to iterate on quickly and capable
enough to produce causal closed-loop evidence. The released RoboTwin checkpoint
runs on one RTX 3090, uses roughly 10 GB in the tested configuration after
moving UMT5 to CPU, and takes about 0.137 seconds for a warm action chunk.

### Static matched gate

On one expert-valid bell/bread scene, using three matched policy diffusion seeds
in each native direction:

- native requested-goal success: **6/6**;
- swapped requested-goal success: **2/6**;
- endpoint shifted in the requested direction: **6/6 matched pairs**;
- median native-left to prompted-right shift: **+15.2 cm**;
- median native-right to prompted-left shift: **−17.3 cm**.

This is the first result that survives the failed-heatmap critique. It is not
only output sensitivity: every paired endpoint moves in the sign requested by
the changed relation. Two swaps also meet the strict goal geometry and release
condition.

The limitation is just as important. Four of six counterfactuals still fail.
The evidence comes from one expert-valid layout, while attempted additional
scenes were not native-competent and therefore could not support a fair steering
test. This is a discovery result, not a cross-scene estimate.

### Identical-history command switch

The stronger experiment begins under the native task command. At the first
verified lifted-object, closed-gripper state, the unexecuted action-chunk suffix
is discarded and a new command is issued. In all six matched groups the switch
occurs at action 25, and the action prefix is byte-identical across conditions.

| Command after grasp | Strict success | Requested side reached | Favorable shift from native control |
| --- | ---: | ---: | ---: |
| Full counterfactual task | **3/6** | 3/6 | 4/6 |
| Counterfactual subtask | **2/6** | 3/6 | 3/6 |
| Atomic motion | **0/6** | 1/6 | 3/6 |
| Combined motion + release | **1/6** | 4/6 | 5/6 |
| Same-direction control | **6/6** | 6/6 | — |

![Efficient-WAM trajectories share an identical prefix and diverge after the
command switch.](../artifacts/wam_language_gate/hierarchical/identical_prefix_trajectory.png)

![Efficient-WAM endpoint outcomes by command
style.](../artifacts/wam_language_gate/hierarchical/command_style_endpoints.png)

The command causes different behavior after the same physical history. That is
causal steerability evidence. But the aggregate hides a striking directional
split: native-right to requested-left succeeds **3/3**, while native-left to
requested-right succeeds **0/3**. A headline that reported only 3/6 would miss
the most scientifically useful fact.

The abstraction results are equally revealing. “Move left” can move left
without placing an object. A combined command often reaches the desired side
but may fail distance, lateral tolerance, or release. Familiar task language is
more usable than the low-level interface a human might hope to expose.

### Pros

- Fast enough for large prompt and intervention sweeps on one 3090.
- Real closed-loop counterfactual successes, not only open-loop distances.
- Supports a clean same-prefix mid-rollout intervention experiment.
- Co-generated futures make action–future coherence measurable.
- The 1B scale makes fine-tuning on paired steering annotations plausible.

### Cons

- Current positive result is limited to one expert-valid scene.
- Strong left/right asymmetry suggests data or frame bias.
- Zero-shot atomic control is poor.
- Point and gripper-trace commands have no demonstrated native structured API.
- Coarse future video is less suitable for precise object-relation scoring.
- Timed multi-command control failed and drove the rollout off-manifold.

**Verdict:** Efficient-WAM is not a solved controller. It is the best model here
for doing the next experiment quickly enough to learn something.

## FastWAM: an advertised guidance knob that did nothing

[FastWAM](https://arxiv.org/abs/2603.16666) produced the most instructive
implementation failure. The release accepted a `text_cfg_scale` argument, but
the value never reached the action-only or joint denoising path. The API implied
language guidance while inference ignored the knob.

After implementing the missing positive and negative passes, the best tested
scale was 2.0. On the fixed-observation probe:

- prompt action RMS: **0.00532**;
- sampling action RMS: **0.00793**;
- prompt-to-sampling ratio: **0.67**.

There is a real language signal, but it is smaller than ordinary sampling
variation in that test. One seed produced a clean matched pair: the left prompt
finished at `dx=-0.195 m`, and the right prompt at `dx=+0.091 m`, both inside
their requested relation. Across five diffusion seeds, the counterfactual
success reproduced only once.

### Pros

- A genuine counterfactual can be elicited after the inference repair.
- The architecture exposes an explicit place to study text guidance.
- Joint action/video inference is useful for WAM-specific consistency tests.
- The bug itself gives a reproducible lesson: trace every control parameter into
  the actual denoising equation.

### Cons

- The released text-CFG behavior was nonfunctional in the tested paths.
- Prompt effect was below the sampling noise floor.
- The positive counterfactual was fragile and post-hoc scale selection risks
  overfitting the evaluation.
- A larger video expert plus action expert makes iteration heavier than
  Efficient-WAM.

**Verdict:** FastWAM contains usable language signal, but the current public
path is a better case study in evaluation hygiene than a reliable blog core.

## LingBot-VA: it imagines another future, but does not reliably execute it

[LingBot-VA](https://arxiv.org/abs/2601.21998) gives the cleanest evidence that
the predicted future depends on language. On one fixed observation and sampling
seed:

- exact-repeat action RMS: **0**;
- exact-repeat future-latent RMS: **0**;
- left/right future-latent RMS: **0.02733**;
- left/right normalized-action RMS: **0.00116**.

The checkpoint is also task competent: it solves both released native relation
tasks. Yet its default counterfactual score is **0/2**. In one swapped rollout,
the object crosses to the requested side, but the policy pushes the target,
keeps the grasp, and never completes the placement. In the other direction it
does not cross.

Increasing action CFG from 1 to 3 improves the open-loop prompt-to-sampling
action ratio from 0.29 to 0.49. It does not improve the primary metric: native
success remains 2/2 and swapped success remains 0/2. It even removes the one
directional crossing. More visible conditioning is not monotonically better
control.

### Pros

- Exact deterministic repeat makes prompt deltas easy to interpret.
- The clearest semantic imagined-future separation of the tested models.
- Native competence in both spatial directions.
- A strong substrate for studying alignment between imagined and executed
  futures.

### Cons

- No closed-loop swapped success in the tested pair.
- About seven seconds per warm 16-action-plus-future chunk.
- Single-3090 operation is possible but close to the memory limit.
- CuRobo CUDA graph pools require process-per-episode isolation in this setup.
- Larger open-loop prompt deltas did not predict better task success.

There was a worthwhile engineering result. The integration originally loaded
the same frozen 2.8 GB VAE twice to obtain two causal caches. Sharing one VAE's
weights across independent streaming wrappers preserves separate temporal state
and makes the full policy fit on a 24 GB 3090. The measured PyTorch peak was
about 19.9 GiB; driver-observed use approached the card's capacity.

**Verdict:** LingBot is the most interesting semantic and future-generation
comparison, but it is currently too slow and too weak under counterfactuals to
anchor rapid closed-loop steering work.

## Cosmos3 Edge: the ecosystem finally reaches the primary metric

[Cosmos3-Edge-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID)
is a 4B mixture-of-transformers policy that produces DROID joint-position
actions and an imagined video. NVIDIA also provides a
[Cosmos3-DROID dataset](https://huggingface.co/datasets/nvidia/Cosmos3-DROID)
and an official [RoboLab](https://github.com/NVlabs/RoboLab) client-server path.
That makes Cosmos attractive for a serious WAM benchmark: the model, action
schema, observations, simulator, task predicates, and output recorder can be
connected without inventing a surrogate controller.

### Offline command-style probe

I first used one successful real DROID episode with the correct schema: seven
joint positions, one gripper value, and three camera views. At five observation
offsets, the same model seed was run with a canonical left task, an exact repeat,
a right counterfactual, a left paraphrase, and an unrelated drawer command.

The exact repeat was perfect at every offset: action and video RMS were both
zero. The model is deterministically prompt-dependent. The semantic selectivity
was weak:

- mean left/right action RMS: **0.03437**;
- mean left/paraphrase action RMS: **0.03653**;
- mean left/unrelated action RMS: **0.04810**;
- the opposite command exceeded the paraphrase action delta at only **1/5**
  offsets;
- the opposite command exceeded the unrelated action delta at only **1/5**
  offsets.

The same pattern appears in imagined-video pixels. Opposite exceeded paraphrase
at 2/5 offsets and unrelated at 1/5. This proves neither that the right future
is wrong nor that the unrelated future is meaningful—pixel RMS is too blunt for
that. It does show why the heatmap cannot support a steering claim.

I also sent subtask, atomic, point, trace, and combined text. These are useful
capability probes, but the distinction is crucial: the RoboLab request has no
native structured coordinate field. A sentence containing `[320, 220]` is not
equivalent to the paper's grounded point interface.

![Cosmos3 Edge decoded future frames for the fixed-observation command-style
probe. Visual differences are secondary diagnostics, not success
evidence.](../artifacts/wam_language_gate/cosmos/cosmos_imagined_future_grid.jpg)

### Closed-loop matched tasks

The first native pilot succeeds: in RoboLab's released left-of-bowl task,
Cosmos grasps the cube, moves it from robot-right to robot-left, establishes the
relation at step 319, and releases at step 320. The final cube-minus-bowl lateral
offset is +8.04 cm. The raw and ordered progression scores are both 1.0.

My first attempt to create a matched right counterpart was invalid. The
released scene already starts with the cube to the bowl's robot-right. RoboLab
classifies a termination in the first two steps as a physics artifact and
resets it, so the apparent “static failure” was actually 450 silent one-step
resets. That rollout is excluded from the model results.

The corrected pair moves the cube to a neutral longitudinal position at reset,
where neither left nor right is satisfied. Left and right tasks use the same
scene, objects, episode length, event, simulator seed, model seed, and predicate
structure. Recorded robot and object initial-state arrays are byte-identical;
only the instruction and goal predicate change.

| Neutral-start condition | Success | Ordered final progression | Raw unordered score | Outcome |
| --- | ---: | ---: | ---: | --- |
| Left | **0/1** | **1/3** | 2/3 | Grasped at step 141; never established left; full 450-step horizon |
| Right | **1/1** | **1.0** | 1.0 | Grasped at step 62; established right at 106; released at 114 |

The first 32-action chunks differ with RMS 0.01648. By the matched 114-step
horizon, action RMS is 0.28413. The endpoint is even more informative: the
right prompt ends 24.09 cm to robot-right of the bowl and succeeds; the left
prompt also ends to robot-right, by 11.79 cm, and fails. The prompt changes
the trajectory and the successful right command amplifies motion in the
requested direction, but the pair exposes a strong rightward bias rather than
symmetric control.

![Cosmos3 Edge closed-loop rollouts. Top: released native-left success. Middle:
neutral-start left failure. Bottom: matched neutral-start right
success.](../artifacts/wam_language_gate/cosmos/cosmos_closed_loop_montage.jpg)

This is the clearest Cosmos sign of life in the study: one successful member of
an exact matched pair, plus separate competence in the other direction. It is
still only one model seed and one neutral geometry. Reporting 1/2 without the
direction split would overstate the result.

Bringing this evaluation up exposed three practical traps:

1. The action-policy server attempted to initialize an unused gated text
   guardrail; disabling guardrails for the action/video service removes that
   irrelevant dependency.
2. RoboLab executes 32 open-loop actions per request, so the server must return
   32—not 16—or the client exhausts its chunk.
3. The 535 driver branch can be displayed with a wrapped minor version in
   Vulkan. A targeted Isaac version-check bypass was needed after verifying the
   actual host driver rather than treating the displayed `535.53` as ground
   truth.

The evaluation runner also needed explicit horizon finalization so a still-active
failure serialized as `success=false` with its task score and HDF5 trajectory
instead of `null`.
These are not model-quality arguments. They are part of usability: a WAM that
requires several days of stack archaeology has a higher experimental cost than
a model that yields a controlled trial in an hour.

### Pros

- Official checkpoint, DROID schema, dataset, and RoboLab evaluation path.
- Generates both actions and decoded imagined futures.
- Demonstrated closed-loop competence in both directions across the native and
  neutral-start pilots.
- One requested-goal success in an exact, byte-identical-initial-state pair.
- RoboLab supplies predicates, videos, event logs, and task-progression hooks.
- Broader ecosystem makes it a valuable comparison to small research releases.

### Cons

- The neutral matched pair is one-sided: right succeeds, left fails, and both
  endpoints move robot-right.
- Offline prompt response was deterministic but poorly semantically selective.
- Point and trace commands are text-only hacks without a grounded interface.
- A 4B model and Isaac/RoboLab stack are heavier than Efficient-WAM.
- Chunk-length, guardrail, driver-reporting, and timeout issues complicate a
  supposedly official path.
- Three closed-loop episodes and one matched model seed are only a pilot.

**Verdict:** Cosmos has a real but one-sided closed-loop steering signal. It is
now a runnable and scientifically useful comparison, not yet a robustly
steerable winner. Its greatest near-term value is that it supports a credible
end-to-end benchmark with real task predicates and imagined futures.

## What the cross-model evidence says

### 1. Language hooks can be cosmetic

A prompt argument, guidance scale, or text encoder in a diagram does not prove
that the deployed inference path uses language effectively. FastWAM's dead CFG
path is the sharp example; weakly selective Cosmos deltas are the subtler one.

Every release should pass three cheap audits before expensive evaluation:

- exact-repeat determinism;
- prompt effect relative to sampling noise;
- source-level trace from argument to conditioning equation.

### 2. Sensitivity is not semantic selectivity

If an unrelated drawer command changes an output more than left versus right,
the model may be reacting to token or distribution distance rather than the
requested relation. The proper control is not “non-zero difference”; it is a
structured ranking:

```text
exact repeat < semantic paraphrase < matched opposite goal
```

That ordering will not hold at every observation, but a steerable interface
should satisfy it reliably and then pass closed loop.

### 3. Imagined and executed futures can disagree

LingBot makes this visible. A language-dependent future latent coexists with
zero swapped task success. A future generator can encode an alternative scene
without the action head supplying a stable route to it. Conversely, actions can
shift while a coarse predicted video hides the task-critical relation.

The central WAM metric should therefore be **predicate agreement**:

- does the imagined future satisfy the requested relation?
- does the executed future satisfy it?
- do those two answers agree?

Pixel similarity is at best a debugging aid.

### 4. Command abstraction is a bandwidth test

Efficient-WAM's full-task command works better than its atomic command. That
does not mean atomic steering is a bad idea. It means the public checkpoint's
language interface is shaped by its training distribution. A fair future study
must train on the command mixture before judging which abstraction is best.

Point and trace interfaces also cannot be evaluated by embedding coordinates in
a sentence and hoping. They need structured grounding tied to the current image
and camera geometry.

### 5. Asymmetry is a result, not noise to average away

Efficient-WAM's 3/3 versus 0/3 direction split and Cosmos's left-native success
versus right-counterfactual failure could come from task distribution, camera
frame, scene geometry, or policy bias. The next benchmark should reverse object
positions and camera viewpoints, then report both directions separately before
an aggregate.

### 6. Systems usability belongs in the comparison

“Fits on one 3090” can hide CPU-offloaded text encoders, duplicated VAEs,
process isolation, simulator memory, or a separate server GPU. Experimental
throughput determines how many paired trials can actually be run. That affects
the quality of scientific evidence, not only developer convenience.

## How I would train a genuinely steerable WAM

The most promising path is not more prompt engineering. It is a training and
evaluation loop built around counterfactual behavior.

### 1. Relabel diverse trajectories at multiple abstractions

For each trajectory, derive task, subtask, atomic motion, point, trace, and
combined commands, following the spirit of Chen et al. The command must describe
what the trajectory actually does at the labeled time—not merely paraphrase the
episode title.

### 2. Add paired counterfactuals

The same initial scene should contain successful rollouts for opposing spatial
relations, alternative objects, and different orderings. Without behavioral
support for both branches, the language variable is observational rather than
controllable.

### 3. Give grounding a real interface

Points and traces should be tensors or structured fields linked to the source
image and camera, not serialized coordinate strings. The WAM should condition
both its action and future heads on the same grounding representation.

### 4. Couple action and future predicates

Add losses or preference pairs that reward agreement between the commanded
predicate, imagined outcome, and forward-simulated action outcome. A visually
plausible future that contradicts the action should be treated as an error.

### 5. Train command switching explicitly

Sample interventions after verified states—grasp, lift, contact, partial
placement—and train the model to discard stale action plans. Include
same-direction reissues so the system cannot gain credit merely for resetting
its chunk.

### 6. Use a competent high-level controller

The paper's hierarchical result does not come from blindly rotating prompts.
The controller observes task progress and chooses an abstraction suited to the
failure. A WAM benchmark should compare predicate-triggered, five-step, and
twenty-step intervention cadences, while logging why each command was issued.

## The benchmark I would trust

A publishable estimate should include all four paper-aligned slices and report
the six command styles explicitly:

| Split | Example held-out factor | Primary question |
| --- | --- | --- |
| In-distribution | Familiar task, objects, layout, wording | Is the interface competent? |
| Motion generalization | New local motion or composition | Can low-level commands induce it? |
| Spatial generalization | New relation, pose, or camera | Does relational language control geometry? |
| Semantic generalization | New object, category, or affordance | Does grounding transfer? |

For every condition, publish numerator/denominator success, a 95% interval,
ordered task progression, seeds, exact prompts, command activation steps,
trajectory-prefix hashes, and invalid-scene counts. Secondary WAM tables should
contain prompt/noise effect ratios, predicted predicate success,
executed-predicate success, action–future agreement, intervention latency, and
persistence.

I use four evidence labels:

- **Pilot:** one valid scene, both directions, three matched model seeds.
- **Discovery:** at least five valid scenes and ten paired trials per condition.
- **Estimate:** preregistered task/seed grid with uncertainty intervals.
- **General steerability:** all four splits, multiple abstractions, held-out
  scenes and objects.

Most results in this article are pilots or discovery evidence. Keeping those
labels visible is not excessive caution. It prevents a useful observation from
turning into a false universal claim.

## Final assessment

There is enough signal to justify a WAM steerability program.

Efficient-WAM can respond causally to a mid-rollout language change after an
identical physical prefix, and it can occasionally complete the new goal. That
is real. FastWAM can produce a clean counterfactual after a broken guidance path
is repaired, but the result is fragile. LingBot can imagine language-dependent
futures with exact repeatability, but those futures do not reliably become
successful counterfactual actions. Cosmos can complete a native RoboLab spatial
task and the right member of an exact neutral-start pair through an official
end-to-end stack, while the matched left member fails and its offline prompt
changes lack selectivity.

The fairest one-line conclusion is:

> **Today's public WAMs expose fragments of steerability—sensitivity, directional
> motion, native competence, or imagined alternatives—but not yet a robust,
> general control interface.**

For the blog and the next experimental cycle, I would use Efficient-WAM as the
fast core, Cosmos as the ecosystem-scale comparison, LingBot as the
future-semantics comparison, and FastWAM as the cautionary implementation case.
Then I would fine-tune the small core on multi-abstraction, grounded, paired
counterfactual commands and evaluate it with success and task progression—not
another heatmap.

## Reproducibility map

- Benchmark contract: `docs/WAM_STEERABILITY_BENCHMARK.md`
- Efficient-WAM summary and figures:
  `artifacts/wam_language_gate/hierarchical/`
- Cosmos summary and rollout montage: `artifacts/wam_language_gate/cosmos/`
- Cross-model machine-readable summary: `artifacts/wam_language_gate/summary.json`
- Efficient-WAM raw experiments:
  `/home/ali/projects/Efficient-WAM/experiments/robotwin_language_gate/`
- FastWAM raw experiments:
  `/home/ali/projects/FastWAM/experiments/robotwin_language_gate/`
- LingBot raw experiments:
  `/home/ali/projects/lerobot-lingbot/experiments/lingbot_language_gate/`
- Cosmos offline probe: `/home/ali/cosmos-framework/scripts/cosmos_droid_steerability_probe.py`
- Cosmos closed loop: `/home/ali/projects/RoboLab/output/cosmos_edge_steerability_*`

Primary model and benchmark sources:
[Steerable Policies](https://arxiv.org/abs/2602.13193),
[Efficient-WAM](https://efficientwam.github.io/),
[FastWAM](https://yuantianyuan01.github.io/FastWAM/),
[LingBot-VA](https://github.com/robbyant/lingbot-va),
[Cosmos3 Edge DROID](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID),
and [RoboLab](https://github.com/NVlabs/RoboLab).
