# π0.5 DROID VLA baseline: matched spatial steering pilot

- Run date: 2 August 2026
- Evidence tier: below pilot (one scene, two canonical samples per direction,
  one short-form sample per direction)
- Checkpoint: `gs://openpi-assets-simeval/pi05_droid_jointpos`
- Policy checkout: OpenPI commit `aa64205`

## Why this run matters

This is the non-world-model VLA control for the WAM study. The public π0.5
DROID policy emits 15-step joint-position action chunks but no imagined future.
It was evaluated in the same RoboLab matched tasks used for Cosmos3 Edge. All
shared robot, object, and camera initial-state arrays are exactly equal across
the two model integrations; Cosmos records two additional cameras.

The scene starts with the Rubik's cube almost longitudinally aligned with the
bowl (`cube_y - bowl_y = -0.263 cm`), so neither requested left/right relation
is initially satisfied. Left and right conditions differ only in prompt and
goal predicate.

## Primary results

The two-item task-progression rubric follows the paper's spatial-generalization
form:

1. pick up the correct object (persistent credit); and
2. establish the requested final relation.

The paper does not require items to occur in order. I also report a stricter
pick-then-place score so that pushing the cube across the relation boundary
cannot masquerade as a complete manipulation.

| Prompt | Run | Binary SR | Picked | Requested relation | Paper-style progression | Strict pick-then-place | Final lateral offset |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Canonical left | 0 | 0 | no | no | 0/2 | 0/2 | +0.30 cm |
| Canonical left | 1 | 0 | no | no | 0/2 | 0/2 | −0.26 cm |
| Canonical right | 0 | 1 | yes, step 76 | yes, step 358 | 2/2 | 2/2 | −6.29 cm |
| Canonical right | 1 | 1 | no | yes, step 86 | 1/2 | 0/2 | −8.71 cm |
| Short left | 0 | 0 | no | no | 0/2 | 0/2 | +1.06 cm |
| Short right | 0 | 0 | yes, step 357 | no | 1/2 | 1/2 | **+19.43 cm (wrong side)** |

Positive lateral offsets are robot-left; negative offsets are robot-right.
Canonical binary success is 0/2 left and 2/2 right. Mean paper-style
progression is 0% left and 75% right; mean strict pick-then-place progression
is 0% left and 50% right.

The second canonical-right “success” is a benchmark warning. RoboLab terminates
once the right relation is true and the gripper is detached, even though the
log contains no successful cube pickup. Binary SR therefore overstates
pick-and-place competence. The paper-style and strict rubrics expose the
difference.

![Canonical and short-form π0.5 rollouts. Rows are canonical left, canonical
right, short left, and short right; columns are start, middle, and
end.](pi05_matched_rollout_montage.jpg)

## Does the first action chunk resolve the language change?

No. All four canonical rollouts have the same serialized initial-state hash:
`1b1ddf04672b0ac9dfdaeec96d84829c23a4fd3e250b3a62c01745a3cb3ab470`.
The public server advances its internal JAX sampling state between requests and
does not expose a per-request policy seed in this client path. These are
therefore same-prompt repeated samples—not a same-sampling-seed determinism
test. They estimate the sampling noise relevant to ordinary deployment. That
variation changes the first action chunk more than swapping left and right:

| Diagnostic | RMS |
| --- | ---: |
| Opposite prompts, run 0 | 0.0684 |
| Opposite prompts, run 1 | 0.0784 |
| Same left prompt, runs 0 vs 1 | 0.0972 |
| Same right prompt, runs 0 vs 1 | 0.0935 |
| Mean prompt effect | 0.0734 |
| Mean repeat noise floor | 0.0953 |
| Prompt effect / noise | **0.77** |

An effect-to-noise ratio below one means the open-loop left/right action delta
is not resolved above same-prompt variation in this sample. This does not erase
the one-sided closed-loop result, but it prevents interpreting first-chunk
action distance as clean language grounding.

The short prompts are not stable paraphrases. Canonical-to-short first-chunk
RMS is 0.0764 for left and 0.2994 for right. The short right instruction both
fails and ends on the wrong side, so larger action separation is actively
misleading here.

## Comparison with Cosmos3 Edge on the same task

| Model | Future output | Left | Right | Main interpretation |
| --- | --- | --- | --- | --- |
| Standard π0.5 DROID VLA | none | 0/2 binary; 0% progression | 2/2 binary; 75% progression | Right activates behavior; left stalls; wording brittle |
| Cosmos3 Edge DROID WAM | decoded video | 0/1 binary; 50% progression | 1/1 binary; 100% progression | Both act, but both endpoints favor robot-right |

Both checkpoints show one-sided right success from the neutral scene. That is
evidence of task-conditioned behavior, but not symmetric left/right control.
π0.5's right result repeats, although only one run completes a strict
pick-and-place. Cosmos has a deterministic action/future interface and stronger
left-side task progress, but its offline opposite-prompt deltas are less
selective than paraphrase or unrelated-command deltas at most tested offsets.

This pair does not establish that “VLAs are more steerable than WAMs” or the
reverse. It establishes something narrower and more useful: on a matched DROID
spatial pilot, architecture class does not remove directional and language-
interface bias. The trained command distribution matters more than whether a
model happens to expose an imagined video.

## Local cost and usability

- Two RTX 3090s were used locally: GPU 0 for the π0.5 server and GPU 1 for
  Isaac/RoboLab.
- The policy server occupied about 12.5 GB of GPU memory.
- Canonical 450-step failures took about 81–83 seconds wall time; the canonical
  successful runs took 16.5 and 68.4 seconds.
- The checkpoint consumes about 12 GB on disk; the OpenPI checkout and
  environment consume about 7.8 GB.
- No cloud GPU or paid API was used.

## Pros

- Public, native DROID checkpoint and a clean RoboLab client integration.
- Fits comfortably on one 24 GB GPU; a second local GPU can run Isaac.
- Canonical right behavior reproduces at the goal-predicate level.
- Fast enough for local matched-pair and prompt-control sweeps.
- Provides a useful non-WAM baseline under the same task geometry.

## Cons

- Left fails twice and does not make meaningful object progress.
- Same-prompt action variation exceeds the measured opposite-prompt effect.
- A short paraphrase destroys the canonical right success and reverses the
  endpoint direction.
- Binary task termination can award success to a push without a recorded pick.
- No imagined future, so future compliance and action–future agreement cannot
  be evaluated.
- Six episodes in one scene are far below a general steerability claim.

## Verdict

π0.5 is inexpensive and usable as the VLA control, but not a positive
steerability result. Its strongest signal is a reproducible preference for the
canonical right request. The absence of left behavior, sub-noise first-chunk
prompt effect, and paraphrase reversal make the correct label **one-sided task
activation**, not robust language grounding.

Machine-readable values are in `matched_pair_summary.json`; the analysis that
derives them from RoboLab HDF5 and event logs is in
`analyze_matched_pair.py`.
