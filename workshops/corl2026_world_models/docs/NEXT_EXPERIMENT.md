# Does the predicted video show where the cube will end up?

> This initial outline is superseded by the
> [paper revision plan](superpowers/plans/2026-09-12-world-model-paper.md).
> The revised plan first checks archived V3 recordings, distinguishes the custom
> DreamZero configuration, addresses goal-dependent stopping, and limits each
> forecast comparison to its actual executed horizon. The eight-trial pilot is
> conditional on recording/configuration qualification needs.

The existing paper measures what the robot does. The most useful extension is
to test whether the model's generated future predicts those same outcomes.
This would connect the scene-layout result directly to the model's prediction
of the world.

## First: an eight-trial recording check

For Nano and DreamZero, run one trial in each of the four existing conditions:
original/reflected layout × LEFT/RIGHT command. This is eight trials total,
with the existing 450-action limit. This pilot tests whether the recordings
support the comparison; it is not the final experiment or a basis for significance.

Before execution, verify that each released interface actually exposes decoded
future images and documents their relation to the action sequence. Decoding must
not change the actions. If an interface cannot do this, stop that model's pilot
and record the limitation. Do not invent predicted images from action logs.

For every model request, save:

- The original conditioning image, generated frames and their camera identity.
- The complete proposed actions and the prefix that was actually executed.
- Original execution images and simulator state at every control step.
- Request, image, action and simulation timestamps; truncation and termination.
- Checkpoint/runtime versions, original modifications, prompts and reset state.

The pilot passes only when a generated frame can be matched to an original
execution image at the same physical time in the same view. A later conditioning
image can supply that observation if the timing is demonstrated. A trajectory
plot or newly rendered reconstruction is not original execution imagery.

## The actual comparison

For each eligible request, compare the cube and bowl in three images:
the observation before acting, the generated endpoint and the actual endpoint.
Use the same object definitions and camera view throughout.

The baseline predicts that the objects stay where they were before the robot
acted. The main question is whether the generated image predicts the eventual
cube–bowl relation better than this baseline. Report performance separately when
the objects change relation and when they stay in the same relation; otherwise
many easy stationary cases can hide poor predictions of movement.

Use independent image labels and retain unclear or missing images. Do not show
raters counterpart images, model names, requested directions or the old scores.
Treat image-plane relation prediction and full placement success as different
outcomes: a single projected relation cannot certify release, depth or contact.

## Confirmation after the recording check

Use new, fixed-in-advance object arrangements and matched original/reflected
pairs, with both commands in each arrangement. Keep the pilot and existing
study as development data. Choose and freeze the final number of arrangements,
seeds, annotation rubric and uncertainty calculation before inspecting the
confirmation results. The sample size should be based on desired precision and
the measured runtime; no numerical GPU-hour estimate is available yet.

New layouts address the present paper's main limitation: its repeated trials
come from just two physical arrangements. New model-noise draws are a separate
factor; do not call DreamZero's fixed-seed repetitions independent noise samples.
If the released interface supports seed control, qualify it explicitly before
including a stochastic replication claim.

The user has approved additional experiments if needed. The eight-trial pilot
is specified but **not launched**. The execution machine and resource limit
must be established before remote runs; current work has used local analysis
only. No paper result depends on the pilot having happened.
