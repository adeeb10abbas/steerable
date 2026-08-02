# The first small world-action model that actually moved when the word changed

Status: experiment-backed blog core. The default LingBot-VA gate is complete;
the action-guidance intervention is running. Keep the distinction between
directional evidence and successful counterfactual control explicit.

## The story in one paragraph

I wanted the smallest world-action model I could actually steer with language,
not merely one that accepted a string argument. A fixed-observation heatmap
made several models look alive, but a matched closed-loop test told a harsher
story. Efficient-WAM became the first useful experimental organism: on one
expert-valid scene, every left/right prompt swap shifted the endpoint in the
requested direction and 2/6 counterfactuals cleanly succeeded. FastWAM then
revealed a real implementation bug—its advertised text CFG argument never
reached inference—and repairing it exposed one valid but fragile
counterfactual. LingBot-VA supplies the strongest semantic imagined-future
signal: it fits on one RTX 3090 after a VAE-memory fix, exactly repeats
identical prompts, imagines substantially different latent futures for `left`
and `right`, and solves both released native tasks. At default guidance,
however, neither LingBot swap completes the opposite task. The useful result
is not “language steering solved”; it is a fast core model plus a reproducible
ladder for separating language sensitivity, task competence, directional
intervention, and robust steerability.

## Candidate title deck

- **The first small world-action model that actually moved when the word changed**
- **A world-action model can imagine the other future—but will it act on it?**
- **Your robot model accepts text. That does not mean it listens.**
- **I fixed a dead guidance knob and went looking for language-steerable world models.**

Suggested subtitle:

> A two-GPU overnight audit of UVA, Efficient-WAM, FastWAM, and LingBot-VA,
> using matched left/right counterfactuals instead of similarity heatmaps.

## The experimental contract

The central comparison changes one token and nothing else:

```text
Place the blue soap on the left of the tea-box.
Place the blue soap on the right of the tea-box.
```

For each pair:

1. Reset the exact RoboTwin task and environment seed.
2. Verify the same object identities and initial poses.
3. Reset the model sampling seed.
4. Change only the requested spatial relation.
5. Replace the task-name-bound evaluator with the same official geometric
   criteria parameterized by the requested relation.
6. Save action traces, generated-future latents, simulator video, and final
   object-minus-target displacement.

The official relation region is deliberately strict: planar distance must be
between 0.08 and 0.20 m, lateral error must be below 0.05 m, the sign of
`object_x - target_x` must match the prompt, and both grippers must be open.

## The claim ladder

Do not collapse these into a single “steerability score.”

| Level | Question | Evidence |
| --- | --- | --- |
| Determinism | Does the same prompt and RNG repeat? | Exact action and latent repeat |
| Sensitivity | Does changing the prompt change predictions? | Prompt delta versus sampling delta |
| Competence | Can the release solve its native task? | Official-geometry closed-loop success |
| Direction | Does a prompt swap move the outcome to the requested side? | Sign of final `dx` under a matched pair |
| Control | Does the swapped prompt complete the opposite goal? | Relation-aware closed-loop success |
| Robustness | Does control survive seeds and scenes? | Matched seed/scene grid |

The earlier heatmap failed because pairwise pixel or action distance answers
only the second question, and can be dominated by sampling noise, unrelated
nouns, or video appearance.

## Results so far

### Efficient-WAM (1B): the blog's core experimental model

On the expert-valid bell/bread scene (environment seed 4200000), across three
matched policy diffusion seeds:

- Native requested-goal success: **6/6**.
- Counterfactual requested-goal success: **2/6**.
- Endpoint shifted in the prompted direction: **6/6 matched pairs**.
- Median shift under native-left→prompt-right: **+15.2 cm**.
- Median shift under native-right→prompt-left: **−17.3 cm**.
- The cleanest pair succeeds in both directions: right at `dx=+0.101` in 76
  actions, left at `dx=-0.070` in 81 actions.

The checkpoint is also the practical iteration engine: roughly 10 GB observed
GPU memory in closed loop after moving UMT5 to CPU, about 0.137 seconds median
per warm action chunk, coarse generated futures, and one-3090 operation. That
combination—fast, causal, imperfect—is why it should anchor the blog and the
next experiments.

An earlier seed-4300000 scene produced 2/2 native successes and 0/2 swapped
successes. Two attempted cross-scene seeds were not expert-valid and neither
native condition moved the object, so they are setup failures rather than
evidence for or against language transfer. The positive claim remains scoped
to one expert-valid scene until a multi-scene expert-validation pass is run.

### FastWAM (5B video expert plus action expert): a latent signal behind a dead knob

The release accepted `text_cfg_scale` but never used it in action-only or joint
inference. After implementing the missing positive/negative passes and moving
UMT5 to CPU:

- Best fixed-observation scale in the tested sweep: CFG 2.0.
- Prompt action RMS: 0.00532; sampling RMS: 0.00793; ratio: 0.67.
- One clean matched counterfactual on the native-left scene:
  - `left`: success, `dx=-0.195`, `dy=-0.005`.
  - `right`: success, `dx=+0.091`, `dy=-0.015`.
- But the swapped success reproduced on only 1/5 diffusion seeds.

Interpretation: the released checkpoint contains usable language signal, and
the repair is technically real, but it is not a reliable steerable policy.
That makes a good section in the blog: “an API parameter is not a feature until
you trace it into the denoising equation.”

### LingBot-VA (5.09B): the strongest substrate, not yet a robust controller

Single fixed observation, sampling seed 42:

- Identical-prompt action RMS: **0**.
- Identical-prompt predicted-latent RMS: **0**.
- Left/right predicted-video-latent RMS: **0.02733** (max absolute **0.30078**).
- Left/right normalized-action RMS: **0.00116**.
- Warm 16-action-plus-future generation: about **7 seconds**.

Default closed loop:

| Native task | Prompt | Success | Actions | Final dx | Final dy |
| --- | --- | ---: | ---: | ---: | ---: |
| left | left | yes | 133 | -0.144 | -0.001 |
| left | right | no | 400 | +0.114 | +0.143 |
| right | right | yes | 144 | +0.128 | +0.012 |
| right | left | no | 400 | +0.178 | +0.001 |

The native-left swap is the important partial result. The object crosses to the
requested right side, but the rollout pushes the movable target, retains the
grasp, and never enters the complete success region. The native-right swap
does not cross. Therefore the defensible statement is:

> LingBot-VA shows deterministic, semantic future prediction and one clear
> prompt-conditioned directional intervention, while default closed-loop
> counterfactual success remains 0/2.

Action CFG is a useful control against fooling ourselves with the open-loop
metric. Raising it from 1 to 3 improves the prompt/sampling action-RMS ratio
from 0.29 to 0.49. Yet scale 3 still yields 2/2 native and 0/2 swapped
successes—and removes the left-task directional crossing. Amplifying the
one-chunk language delta does not monotonically improve closed-loop control.

## The 3090 engineering result

LingBot's RoboTwin camera layout uses a full-resolution head camera and two
half-resolution wrist cameras. The LeRobot integration loaded the same frozen
2.8 GB VAE twice to obtain independent causal caches. The cache belongs to the
streaming wrapper, not to the VAE weights, so sharing one VAE across two
wrappers preserves independent temporal state and removes the duplicate
weights.

That change makes the full policy plus CuRobo fit on one 24 GB RTX 3090:

- PyTorch peak allocation: about 19.9 GiB.
- Driver-observed peak during generation: about 24.1 GiB.
- UMT5-XXL stays on CPU and prompt embeddings are cached.

There is a second operational trap: closing RoboTwin does not release all
CuRobo CUDA-graph pools. A second LingBot episode in the same process OOMs.
The final evaluator therefore runs one episode per subprocess and aggregates
results after the process exits. This is worth including because it is exactly
the kind of detail that separates a reproducible WAM experiment from a model
card demo.

## Suggested article structure

1. Open with the bad heatmap and the question it could not answer.
2. Define the left/right matched intervention and the claim ladder.
3. Show Efficient-WAM's 6/6 directional shifts and the clean successful pair.
4. Trace FastWAM's unused CFG argument and show the repaired denoising path.
5. Introduce LingBot's deterministic future-latent result.
6. Put native and swapped rollouts side by side.
7. Explain the VAE-sharing and process-isolation fixes.
8. End with the honest result: Efficient-WAM is a usable research WAM, not a
   generally reliable language controller; LingBot is the stronger semantic
   comparison but currently slower and less successful under swaps.
9. Next experiment: fine-tune on explicit paired relation counterfactuals and
   evaluate across object, scene, and diffusion seeds.

## Figures to publish

1. **The failed heatmap** — label it “sensitivity is not control.”
2. **Claim ladder** — six levels from repeatability to robustness.
3. **FastWAM CFG sweep** — prompt RMS and sampling RMS versus guidance.
4. **LingBot imagined-future delta** — same observation, left/right predicted
   video, plus latent RMS.
5. **Four rollout strips** — native-left, swapped-right, native-right,
   swapped-left at matched seeds.
6. **Outcome plot** — final `(dx, dy)` overlaid on the valid left/right regions.
7. **Memory diagram** — duplicate VAE weights versus shared weights and two
   independent wrapper caches.

## Claim limits

- Do not call LingBot robustly steerable unless swapped success survives a
  preregistered seed and scene grid.
- Do not treat different imagined pixels as evidence of different task intent
  without the action and closed-loop controls.
- The current tests cover two relation tasks and one primary environment/model
  seed; they do not establish broad language grounding.
- The FastWAM counterfactual is real but post-hoc hyperparameter selection and
  1/5 robustness make it a discovery result, not an estimate of success rate.
- Report both CUDA allocator peak and driver-observed peak; the single-3090
  LingBot setup has little operational headroom.

## Durable experiment locations

- LingBot implementation and evidence:
  `/home/ali/projects/lerobot-lingbot/experiments/lingbot_language_gate/`
- FastWAM CFG repair and evidence:
  `/home/ali/projects/FastWAM/experiments/robotwin_language_gate/`
- Efficient-WAM matched gate:
  `/home/ali/projects/Efficient-WAM/experiments/robotwin_language_gate/`
- Full MP4/action/latent outputs are under each repository's ignored
  `outputs/robotwin_language_gate/` or `outputs/lingbot_language_gate/` tree.
