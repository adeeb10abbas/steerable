# Social launch kit: VLA/WAM language steerability

*Ali Adeeb Abbas · Senior Scientist, General Motors · personal research; views
are my own.*

This file is the post-ready companion to
[`VLA_VS_WAM_STEERABILITY_STUDY.md`](VLA_VS_WAM_STEERABILITY_STUDY.md). Use the
square assets for a LinkedIn carousel and the 1600×900 versions for an X thread
or link preview. Do not crop away the model-class boundary or personal-analysis
footer.

## Publication metadata

- **Title:** Does the world model listen? A matched steerability study of VLAs
  and WAMs
- **Description:** 160 matched robot rollouts reveal prompt sensitivity,
  directional bias, semantic failures, and the gap between imagined and
  executed futures.
- **Open Graph image:**
  `../artifacts/vla_wam_shared_v1/trajectory_evidence/social/steerability_scorecard_1600x900.png`
- **Suggested slug:** `does-the-world-model-listen-vla-wam-steerability`

## Recommended X thread

**Post 1 — scorecard**

> A robot model can react to a prompt without following it.
>
> I ran 160 matched, closed-loop episodes to test direct language steerability:
> π0.5 VLA vs Cosmos3 Edge WAM, same neutral scene, matched seeds, no coach or
> oracle.
>
> The result is much stranger than one success rate. 🧵

Attach: `../artifacts/vla_wam_shared_v1/trajectory_evidence/social/steerability_scorecard_1200x1200.png`

**Post 2 — what was controlled**

> Four static prompt forms × LEFT/RIGHT × 10 seeds × 2 checkpoints. Every
> episode used one unchanged task prompt at the model's native horizon. The
> physical robot/object reset arrays share one exact hash. No dynamic prompting
> entered the analysis.

Attach: `../artifacts/vla_wam_shared_v1/trajectory_evidence/social/first_seed_stress_square_1200x1200.png`

**Post 3 — closed-loop result**

> π0.5 completed 25/80 tasks; Cosmos completed 58/80. But totals hide the result:
> π0.5 was 3/40 LEFT vs 22/40 RIGHT; Cosmos was 22/40 LEFT vs 36/40 RIGHT.
> Cosmos went from 19/20 declarative to 10/20 contrastive—1/10 on contrastive
> LEFT.

**Post 4 — make failures visible**

> I plotted the full cube and gripper path for all 160 rollouts—not a success
> montage. Of 77 failures, 53 picked up the cube but never entered the requested
> goal, 23 never verified cube interaction, and 1 ended in-goal without terminal
> success.

Attach: `../artifacts/vla_wam_shared_v1/trajectory_evidence/social/failure_progress_anatomy_1200x1200.png`

**Post 5 — the WAM-only test**

> Cosmos produced 752 action+future chunks. A prompt-blind scorer labeled 421:
> 22 imagined+executed the request, 5 imagined only, 3 executed only, 391
> neither. But it covered just 25/97 executed-positive horizons. High agreement;
> severe visibility gap.

Attach: `../artifacts/vla_wam_shared_v1/semantic_future_visualization/social/wam_semantic_quadrants_1200x1200.png`

**Post 6 — what exact-input probes changed**

> Both endpoints were exactly repeatable and changed under LEFT/RIGHT prompts.
> Yet equivalent contrastive word order changed actions as much as—or more
> than—the requested relation. Prompt sensitivity is real. Semantic control is
> the harder claim.

**Post 7 — practical takeaway**

> My current stack: Efficient-WAM-RT for fast same-history interventions;
> Cosmos as the slower DROID WAM/future-prediction cross-check; π0.5 as the VLA
> control baseline. One checkpoint per class means this is a case study, not a
> universal VLA-vs-WAM ranking.

Link the full blog and evidence gallery in this post. Suggested tags:
`#Robotics #WorldModels #VLA #EmbodiedAI`.

## LinkedIn post

> **Does the world model actually listen?**
>
> Prompt sensitivity is not steerability. A model can change its action tensor
> or generated video when a word changes and still move the object to the wrong
> place.
>
> I built a matched, oracle-free study around that distinction: 160 closed-loop
> episodes, four direct task wordings, LEFT and RIGHT requests, ten matched seeds,
> one public VLA checkpoint (π0.5 DROID), and one public world-action checkpoint
> (Cosmos3 Edge DROID). Both saw the same neutral physical reset and used one
> unchanged task prompt per episode.
>
> The headline totals were 25/80 for π0.5 and 58/80 for Cosmos—but the aggregate
> is not the important result. π0.5 succeeded on 3/40 LEFT versus 22/40 RIGHT.
> Cosmos succeeded on 22/40 LEFT versus 36/40 RIGHT. Cosmos handled declarative
> end-state language at 19/20, then dropped to 10/20 when the prompt also named
> and negated the opposite relation; contrastive LEFT was 1/10.
>
> I recorded every success and failure as a robot-frame trajectory. The green
> region is the requested goal, the dark trace is the executed cube path, and the
> endpoint shows where it actually finished. Of 77 failures, 53 successfully
> picked up the cube but never entered the requested goal. That makes the core
> problem visible: it was usually placement grounding, not basic grasping.
>
> The WAM-only result was nuanced. Across 752 action/future horizons, a
> prompt-blind visual evaluator could issue 421 certain labels: 22 imagination
> and execution positives, five imagination-only mismatches, three execution-
> only mismatches, and 391 neutral/neutral horizons. Agreement among certain
> labels was 98.1%, but the evaluator covered only 25/97 horizons where
> execution actually reached the requested relation. Generated futures expose
> useful structure; they are not yet a dependable semantic success monitor.
>
> The practical outcome is a research stack, not a winner's podium: use
> Efficient-WAM-RT for rapid same-history causal interventions, Cosmos for a
> slower DROID WAM and generated-future cross-check, and π0.5 as the VLA control.
> The full article includes the protocol, exact-input probes, future/action
> agreement, thermal and memory costs, negative results, and a filterable gallery
> of all 160 episodes.
>
> This is personal research; views are my own. One checkpoint cannot represent a
> whole model class, so the study is a reusable benchmark and a detailed case
> study—not a universal VLA-versus-WAM ranking.

Recommended carousel order:

1. `steerability_scorecard_1200x1200.png`
2. `first_seed_stress_square_1200x1200.png`
3. `failure_progress_anatomy_1200x1200.png`
4. `wam_semantic_quadrants_1200x1200.png`

## Alt text

**Scorecard.** A four-by-four green heatmap comparing π0.5 VLA and Cosmos WAM
successes for LEFT and RIGHT commands under canonical, short, declarative, and
contrastive wording. Raw successes out of ten are printed in every cell. The
largest gaps are π0.5 canonical LEFT 0/10 versus RIGHT 8/10, and Cosmos
contrastive LEFT 1/10 versus RIGHT 9/10.

**Same-seed path comparison.** Four robot-frame path panels use the same scene
and seed 7200 for declarative and contrastive LEFT prompts. The green wedge is
the requested LEFT goal region; the pale red wedge is the opposite region; the
dark line is the executed cube path; the dashed green arrow is illustrative and
not scored. Cosmos succeeds under declarative wording and fails under
contrastive wording; both π0.5 trials fail.

**Failure anatomy.** Two stacked horizontal-bar panels account for every
episode by its last verified progress stage. Green is success, gold is pickup
without entering the goal, light gray is no cube interaction, and smaller
categories capture terminal mismatch or regressions. π0.5 has many no-
interaction and post-pick failures; most Cosmos failures occur after pickup.

**WAM future/action quadrants.** A dark two-by-two card shows one frozen-order
Cosmos generated-future example for each certain outcome: imagines and executes
the requested relation; imagines it while execution stays neutral; does not
imagine it while execution succeeds; and neither imagines nor executes it. Each
panel contains the two third-person camera views. Cyan circles mark the cube,
red circles mark the bowl, and colored borders plus text distinguish all four
outcomes without relying on color alone.

## Claim guardrails

- Say “one public checkpoint per class,” never “WAMs beat VLAs.”
- Call declarative and contrastive runs a prospectively frozen post-interim
  stress tier, not part of the original preregistration.
- Say the physical reset arrays are byte-identical; do not claim realtime
  conditioning pixels are identical across closed-loop launches.
- Treat exact-input action/video change as sensitivity. Closed-loop goal success
  and semantic future predicates carry the control claim.
- State that the dashed route is illustrative. The shaded region—not one path—is
  the expected goal.
- The offline Qwen localizer is an evaluator, not a coach or oracle, and its
  coverage plus human audit must accompany semantic-future claims.
- Keep the General Motors affiliation as author context; do not imply company
  sponsorship or endorsement.
