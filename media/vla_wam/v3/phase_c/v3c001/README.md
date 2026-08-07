# Phase C phrasing media

This directory contains the outcome-independent publication slice for
`V3-C001`: the lowest preregistered seed (`8500`), all four prompt families,
and both matched directions. LEFT and RIGHT always start from the same reset.

## What each file shows

- `*_matched_actual.mp4`: the complete LEFT and RIGHT simulator rollouts side
  by side. If one ends first, its final frame is held; neither rollout is cut.
- `*_matched_local_predictions.mp4`: Cosmos-only companion media. Every
  exposed 33-frame request-local prediction horizon is shown in request order.
  These stitched horizons are model predictions, not execution and not one
  continuous full-task imagination.
- `*_poster.png`: first-frame preview for the corresponding video.
- `*_publication_media_manifest.json`: exact prompts, outcomes, source hashes,
  publication hashes, durations, and the outcome-independent selection rule.

GR00T N1.7 is action-only at this interface, so its directory correctly has no
prediction video. Cosmos3 Edge and Cosmos3 Nano each have separate actual and
local-prediction videos.

## Exact static prompts

| Family | LEFT | RIGHT |
|---|---|---|
| Direct instruction | `Put the Rubik's cube to the left of the bowl.` | `Put the Rubik's cube to the right of the bowl.` |
| Shortened instruction | `Put the cube left of the bowl.` | `Put the cube right of the bowl.` |
| Goal statement | `The Rubik's cube should end up to the left of the bowl.` | `The Rubik's cube should end up to the right of the bowl.` |
| Contrastive instruction | `Put the Rubik's cube to the left of the bowl, not to the right of the bowl.` | `Put the Rubik's cube to the right of the bowl, not to the left of the bowl.` |

These videos are illustrative evidence, not the statistical denominator. The
complete 20-seed-per-condition results are compiled separately, and DROID is
never pooled with RoboTwin.
