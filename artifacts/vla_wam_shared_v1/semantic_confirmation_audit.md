# Human audit of Cosmos semantic-future scoring

Status: **complete, qualified pass for descriptive use**

Completed: 2 August 2026

Frozen selection: `semantic_confirmation_audit_plan.json` plus
`semantic_confirmation_audit_amendment_002.json`

## Decision

The prompt-blind Qwen localizer is usable as an **abstaining descriptive
evaluator**, not as semantic ground truth. Across the frozen 24-sheet human
sample, the cube and bowl were usually marked plausibly when they were visible.
I found no gross systematic object-identity confusion, no evidence of prompt
following, and no obvious certain LEFT/RIGHT relation whose marked physical
objects visibly supported the opposite relation.

The audit also found real failures. When the generated cube was small, blurred,
deformed, or occluded by the robot, the cyan cube marker could jump to the
robot, table, banana, or background. The red bowl marker also occasionally
jumped under severe robot overlap. Cross-camera disagreement caught many of
these errors, but a two-camera consensus is not a proof of correct object
identity. Semantic results therefore ship with coverage, threshold sensitivity,
all contact sheets, and this limitation. No threshold or confirmation label was
changed after this review.

## Frozen detailed sample

The sample was selected without using semantic labels or rollout success: first,
middle, and final seed for every wording and direction. It covers **24 complete
episode sheets, 254 replan chunks, 1,016 scored future frames, and 2,032 camera
localizations**. The table records automatic accounting; the human review
covered every displayed frame in each linked sheet.

| Sheet | Request | Seed | Chunks | Reliable frames | Certain chunks | Reliable frame relations L / N / R |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| [canonical](semantic_confirmation/cosmos_canonical/audit/v1_cosmos_canonical__RubiksCubeLeftOfBowlMatchedTask/episode_000_contact_sheet.jpg) | LEFT | 6100 | 11 | 22/44 | 7/11 | 3 / 18 / 1 |
| [canonical](semantic_confirmation/cosmos_canonical/audit/v1_cosmos_canonical__RubiksCubeLeftOfBowlMatchedTask/episode_004_contact_sheet.jpg) | LEFT | 6104 | 15 | 23/60 | 8/15 | 0 / 23 / 0 |
| [canonical](semantic_confirmation/cosmos_canonical/audit/v1_cosmos_canonical__RubiksCubeLeftOfBowlMatchedTask/episode_009_contact_sheet.jpg) | LEFT | 6109 | 5 | 9/20 | 4/5 | 4 / 5 / 0 |
| [canonical](semantic_confirmation/cosmos_canonical/audit/v1_cosmos_canonical__RubiksCubeRightOfBowlMatchedTask/episode_000_contact_sheet.jpg) | RIGHT | 6100 | 12 | 25/48 | 7/12 | 0 / 23 / 2 |
| [canonical](semantic_confirmation/cosmos_canonical/audit/v1_cosmos_canonical__RubiksCubeRightOfBowlMatchedTask/episode_004_contact_sheet.jpg) | RIGHT | 6104 | 5 | 11/20 | 4/5 | 0 / 6 / 5 |
| [canonical](semantic_confirmation/cosmos_canonical/audit/v1_cosmos_canonical__RubiksCubeRightOfBowlMatchedTask/episode_009_contact_sheet.jpg) | RIGHT | 6109 | 15 | 37/60 | 10/15 | 0 / 36 / 1 |
| [short](semantic_confirmation/cosmos_vague/audit/v1_cosmos_vague__RubiksCubeLeftOfBowlMatchedTask/episode_000_contact_sheet.jpg) | LEFT | 6100 | 15 | 25/60 | 8/15 | 0 / 23 / 2 |
| [short](semantic_confirmation/cosmos_vague/audit/v1_cosmos_vague__RubiksCubeLeftOfBowlMatchedTask/episode_004_contact_sheet.jpg) | LEFT | 6104 | 15 | 33/60 | 10/15 | 0 / 10 / 23 |
| [short](semantic_confirmation/cosmos_vague/audit/v1_cosmos_vague__RubiksCubeLeftOfBowlMatchedTask/episode_009_contact_sheet.jpg) | LEFT | 6109 | 15 | 24/60 | 6/15 | 0 / 18 / 6 |
| [short](semantic_confirmation/cosmos_vague/audit/v1_cosmos_vague__RubiksCubeRightOfBowlMatchedTask/episode_000_contact_sheet.jpg) | RIGHT | 6100 | 5 | 8/20 | 3/5 | 0 / 6 / 2 |
| [short](semantic_confirmation/cosmos_vague/audit/v1_cosmos_vague__RubiksCubeRightOfBowlMatchedTask/episode_004_contact_sheet.jpg) | RIGHT | 6104 | 11 | 14/44 | 4/11 | 0 / 9 / 5 |
| [short](semantic_confirmation/cosmos_vague/audit/v1_cosmos_vague__RubiksCubeRightOfBowlMatchedTask/episode_009_contact_sheet.jpg) | RIGHT | 6109 | 9 | 13/36 | 4/9 | 0 / 10 / 3 |
| [declarative](semantic_confirmation/cosmos_declarative/audit/v1_cosmos_declarative__RubiksCubeLeftOfBowlMatchedTask/episode_000_contact_sheet.jpg) | LEFT | 7200 | 8 | 16/32 | 4/8 | 0 / 16 / 0 |
| [declarative](semantic_confirmation/cosmos_declarative/audit/v1_cosmos_declarative__RubiksCubeLeftOfBowlMatchedTask/episode_004_contact_sheet.jpg) | LEFT | 7204 | 7 | 13/28 | 4/7 | 1 / 12 / 0 |
| [declarative](semantic_confirmation/cosmos_declarative/audit/v1_cosmos_declarative__RubiksCubeLeftOfBowlMatchedTask/episode_009_contact_sheet.jpg) | LEFT | 7209 | 4 | 6/16 | 1/4 | 0 / 6 / 0 |
| [declarative](semantic_confirmation/cosmos_declarative/audit/v1_cosmos_declarative__RubiksCubeRightOfBowlMatchedTask/episode_000_contact_sheet.jpg) | RIGHT | 7200 | 15 | 53/60 | 13/15 | 0 / 52 / 1 |
| [declarative](semantic_confirmation/cosmos_declarative/audit/v1_cosmos_declarative__RubiksCubeRightOfBowlMatchedTask/episode_004_contact_sheet.jpg) | RIGHT | 7204 | 8 | 16/32 | 4/8 | 0 / 16 / 0 |
| [declarative](semantic_confirmation/cosmos_declarative/audit/v1_cosmos_declarative__RubiksCubeRightOfBowlMatchedTask/episode_009_contact_sheet.jpg) | RIGHT | 7209 | 4 | 8/16 | 3/4 | 0 / 8 / 0 |
| [contrastive](semantic_confirmation/cosmos_contrastive/audit/v1_cosmos_contrastive__RubiksCubeLeftOfBowlMatchedTask/episode_000_contact_sheet.jpg) | LEFT | 7200 | 15 | 21/60 | 7/15 | 1 / 20 / 0 |
| [contrastive](semantic_confirmation/cosmos_contrastive/audit/v1_cosmos_contrastive__RubiksCubeLeftOfBowlMatchedTask/episode_004_contact_sheet.jpg) | LEFT | 7204 | 15 | 31/60 | 9/15 | 0 / 27 / 4 |
| [contrastive](semantic_confirmation/cosmos_contrastive/audit/v1_cosmos_contrastive__RubiksCubeLeftOfBowlMatchedTask/episode_009_contact_sheet.jpg) | LEFT | 7209 | 15 | 39/60 | 12/15 | 1 / 37 / 1 |
| [contrastive](semantic_confirmation/cosmos_contrastive/audit/v1_cosmos_contrastive__RubiksCubeRightOfBowlMatchedTask/episode_000_contact_sheet.jpg) | RIGHT | 7200 | 15 | 33/60 | 10/15 | 17 / 15 / 1 |
| [contrastive](semantic_confirmation/cosmos_contrastive/audit/v1_cosmos_contrastive__RubiksCubeRightOfBowlMatchedTask/episode_004_contact_sheet.jpg) | RIGHT | 7204 | 11 | 19/44 | 6/11 | 0 / 16 / 3 |
| [contrastive](semantic_confirmation/cosmos_contrastive/audit/v1_cosmos_contrastive__RubiksCubeRightOfBowlMatchedTask/episode_009_contact_sheet.jpg) | RIGHT | 7209 | 4 | 8/16 | 2/4 | 0 / 7 / 1 |

In this selected sample, 507/1,016 frames passed both-camera reliability and
150/254 chunks received a certain imagined-predicate label. All 2,032 camera
responses parsed, and none returned a null object point. Unreliable frames can
carry more than one reason: 394 had camera relation disagreement, 227 exceeded
the cube cross-camera distance threshold, and 113 exceeded the bowl threshold.

## Full automatic population

The evaluator processed every one of the **752** registered Cosmos replan
chunks: 3,008 future frames and 6,016 camera responses. All responses parsed.
One response explicitly returned a null cube point
(`canonical RIGHT`, episode 2, chunk 7, frame 32, left camera); the scorer
abstained as designed. Across all frames, 1,460/3,008 passed the per-frame
two-camera rule. Overlapping rejection reasons were:

- camera relation disagreement: 1,201 frames;
- cube cross-camera disagreement above 0.20 m: 710 frames;
- bowl cross-camera disagreement above 0.20 m: 338 frames;
- missing localization: 1 frame.

At the chunk level, 421/752 received a certain label and 331/752 abstained.
Every one of the 80 episodes still had at least one certain chunk, but coverage
was much worse at the moments that matter most: only 25/97 horizons whose
executed state reached the requested relation had a certain future label.

## Concrete successes and failures in the audit

The sampled sheets contain useful successful localizations. For example,
canonical RIGHT seed 6104 tracks the cube into a visually right-of-bowl
relation, and the automatic relation remains consistent across the two camera
views. The evaluator also captures real wrong-way model behavior rather than
following the requested word: short LEFT seed 6104 repeatedly labels the
future cube as RIGHT, while contrastive RIGHT seed 7200 repeatedly labels it as
LEFT. Both rollouts physically end on the wrong side. Those examples are strong
evidence that the evaluator is reading the generated pixels rather than the
policy prompt it never receives.

Visible evaluator failures cluster around robot/object overlap. Representative
examples include declarative LEFT seed 7204 around chunks 3–6, declarative
RIGHT seed 7209 around chunks 1–3, and contrastive LEFT seed 7200 after chunk 4.
The cyan or red marker sometimes lands on the arm, tabletop, banana, or
background. Most of those frames are orange/uncertain in the sheets because
the views disagree, but the human audit cannot prove that every green frame is
correct. This is why the semantic scorer is never used as a controller, never
feeds the policy, and never supports a standalone accuracy claim.

## Publication boundary

The semantic quadrant table is publishable only with all of the following:

- the 55.98% chunk coverage and 25/97 positive-execution coverage;
- the frozen-threshold sensitivity analysis;
- the 24 linked sheets and all 80 automatically generated sheets;
- the statement that early neutral chunks are not failed episodes;
- the statement that replan chunks within an episode are correlated;
- the statement that the Qwen localizer is a fallible offline evaluator, not
  an oracle or subtask coach.

The audit caused **no relabeling, threshold change, episode exclusion, prompt
change, or rerun**.
