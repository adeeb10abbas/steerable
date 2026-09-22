# Workshop focus and the role of π0.5

SGW-01, version 1.1. Framing revision, 22 September 2026. The 1,044-cell WAM queue and all scientific thresholds are unchanged. No π0.5 cells are queued.

## The paper's question

**When instructions change, do a world–action model's generated futures remain a reliable account of what its robot actually does?**

Spatial language supplies controlled changes in goals and descriptions. The paper measures goal preservation in execution and the reliability of the model's accompanying predictions. A wording-sensitivity table by itself would leave the world-model contribution unclear.

Our proposed fit is primarily Motion 6 (benchmark design), with a narrower connection to Motion 4 (reliability of model-based evaluation). This is our interpretation of the [workshop's six motions](https://do-robots-need-world-models.github.io/), checked September 22. We test a WAM's own joint predictions and actions, not its ability to simulate arbitrary alternative policies, preserve policy rankings, or ensure safe deployment.

## What makes this a world-model paper

1. **Executed outcome:** use the same scenes and opposite goals to measure whether D/C/I wording preserves the intended placement.
2. **Prediction fidelity:** compare each model's decoded future with actual motion at the same physical time, within the executed action prefix; require the persistence baseline and report missingness.
3. **Prediction–execution disagreement:** make the existing same-horizon agreement/disagreement table central. A predicted relation can look appropriate while the executed relation differs, or an executed relation can be appropriate while the forecast is wrong. Report both and unobservable cases.
4. **What the forecast adds:** juxtapose these observations with the execution-only pickup/transport/release account. Establish the additional observable discrepancy; do not infer internal semantics or advertise an untested failure detector.

A generated short-horizon relation is not a forecast of final success after later replanning. We do not claim that the model's video causes its actions, that WAMs beat VLAs, or that world models are necessary. If the recordings cannot support aligned prediction measurements, the intended WAM-specific contribution remains incomplete; a large language effect does not repair that omission.

## π0.5 decision

**Keep the core paper WAM-only.** Move historical π0.5 numbers out of the abstract and main evidence table. Preserve them in the research-plan appendix and evidence inventory as background; omit them from the four-page submission unless one short contextual sentence is necessary.

| Candidate use | Decision | What it would establish |
| --- | --- | --- |
| Existing π0.5 inversion cohort | Historical context only | The phenomenon motivated our study in a different cohort. It is not a matched SGW-01 control. |
| Fresh π0.5 on matched SGW-01 fixtures | Optional behavioral baseline, not queued | Whether the same within-model wording effect also appears in an action-output policy. It has no decoded-future comparison. |
| π0.5 versus WAM as a causal test of world modeling | Unsupported | Training data, architectures, objectives and inference budgets are not controlled by this comparison. |

If a fresh baseline is later commissioned, first use the LAT branch: the same 1 pilot + 4 development + 24 confirmation layouts, three exact forms and both goals = 174 additional episodes (6/24/144). Use the same cameras, physical resets, goal predicates, stopping rule and controller timing where supported; record all interface and compute differences. Pin the exact checkpoint and wrapper and qualify them independently. Compare the I−C and C−D effects within each model on paired layouts, with uncertainty. Do not equate matching seed integers with matching random draws across architectures. Register the comparison and any multiplicity changes before examining its confirmation outcomes. The expanded total would be 1,218, only after a separate scope and resource decision. It must not delay or silently alter the existing WAM queue.

A causal world-model ablation would require a deliberately matched training comparison that controls data, architecture/capacity and compute as far as possible while changing predictive training/use. That is outside SGW-01.

## Submission shape

Target the research-paper track. The current [call](https://do-robots-need-world-models.github.io/) lists up to four pages in the CoRL template and an October 12, 2026 submission deadline. It welcomes unfinished and negative findings but excludes already-published work or work accepted to the CoRL 2026 main conference. OpenReview and archival details are still TBD. Do not assume references or appendices are exempt from the page limit.

Use a short motivation and setup, one main execution-effect figure, one aligned prediction–execution figure/table with aggregate coverage and persistence skill, and a focused discussion of what prediction inspection does and does not reveal. Full prompts, implementation details and historical π0.5 analyses remain in the working package; external supplementary material depends on the final submission rules. Keep all observed failures, including a finding that the predictions add little useful diagnostic information.
