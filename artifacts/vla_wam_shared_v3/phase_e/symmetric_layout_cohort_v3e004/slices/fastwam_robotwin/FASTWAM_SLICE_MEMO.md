# V3-E004 FastWAM RoboTwin slice

**Status:** complete arena-separated stretch slice; descriptive only until the full registered V3-E004 cohort closes. RoboTwin is never pooled with DROID.

## Design

FastWAM was evaluated on 27 matched environment/sampling seeds (9400–9426), two object layouts (registered asymmetric s=0 and symmetric-object s=1), and two exact static prompts, for 108 valid behavioral episodes. Infrastructure-invalid setup and acceleration attempts remain outside the denominator. The s=1 fixture is symmetric in its movable-object layout, not in the robot embodiment.

- LEFT: `Put the small woodenblock to the left of the red playingcards box.`
- RIGHT: `Put the small woodenblock to the right of the red playingcards box.`

## Descriptive result

Binary task success was at floor under s=0 (LEFT 0/27; RIGHT 0/27) and remained very low under s=1 (LEFT 1/27; RIGHT 2/27). The continuous requested-depth contrast changed from +32.0 cm (95% CI +27.9 to +36.4) under s=0 to -6.2 cm (95% CI -10.7 to -1.7) under s=1. The paired s=1-minus-s=0 interaction was -38.2 cm (95% CI -44.5 to -32.0) (exact layout-label permutation p=1.49e-08).

That reversal is not sufficient for the registered equalisation interpretation. The endpoint-redirection positive control did not remain detectably positive: the paired LEFT-minus-RIGHT endpoint estimate was +1.3 cm (95% CI −2.9 to +6.2) at s=0 and +0.6 cm (95% CI −4.2 to +5.6) at s=1. Equivalence is also not claimed: both registered FastWAM estimands were classified as underpowered stretch analyses. The safe statement is therefore that the object-layout intervention strongly changed FastWAM's continuous depth contrast in this RoboTwin slice, while near-zero competence and a failed prompt-redirection positive control prevent attributing that change to reliable language steering.

Failure decomposition supports the competence boundary: s=0 contained 53 pick failures and one transport failure; s=1 contained 47 pick failures, four wrong-side failures, and three correct episodes.

## Figure

The complete slice figure reports binary success, the requested-depth interaction, the failed endpoint-redirection positive control, and failure composition together: [PNG](figures/v3e004_fastwam_robotwin_slice.png) · [SVG](figures/v3e004_fastwam_robotwin_slice.svg).

## Selected actual-rollout videos

These four clips are the complete matched seed-9413 layout-by-prompt set. They are illustrative actual simulator executions, not imagined futures and not a replacement for the 108-episode denominator.

| Layout | Exact prompt | Outcome | Clip |
|---|---|---|---|
| asymmetric s=0 | Put the small woodenblock to the left of the red playingcards box. | pick_failed | [seed9413_asymmetric_left.mp4](media/seed9413_asymmetric_left.mp4) |
| asymmetric s=0 | Put the small woodenblock to the right of the red playingcards box. | pick_failed | [seed9413_asymmetric_right.mp4](media/seed9413_asymmetric_right.mp4) |
| symmetric-object s=1 | Put the small woodenblock to the left of the red playingcards box. | success | [seed9413_symmetric_left.mp4](media/seed9413_symmetric_left.mp4) |
| symmetric-object s=1 | Put the small woodenblock to the right of the red playingcards box. | pick_failed | [seed9413_symmetric_right.mp4](media/seed9413_symmetric_right.mp4) |

## Claim boundary

This slice supports no VLA-versus-WAM comparison, no DROID/RoboTwin pooled rate, no equivalence statement, and no claim that the symmetric-object layout is a symmetric robot. Cross-checkpoint manuscript language remains withheld until the complete registered V3-E004 evidence is hash-closed.
