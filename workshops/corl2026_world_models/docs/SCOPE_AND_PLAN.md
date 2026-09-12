# World-model workshop paper: scope and first research sprint

Date: 2026-09-12

## Accepted direction

Develop a world-model-only research submission for the CoRL 2026 workshop **Do Robots Need World Models?**, primarily Theme 6 (evaluation and benchmarking), with Theme 4 (models as evaluation tools) as motivation. Aim for Best Research Paper through a defensible result, useful method, and clear presentation. An award is an aspiration, not a promised outcome.

The user approved beginning this work after discussing paired prediction/execution evaluation, aligned time and spatial predicates, independent review, and explicit missingness. This document records that existing authorization; it does not initiate an experiment on external compute.

## Scientific question

When a world-action model jointly generates a predicted future and an action chunk, what evidence establishes that the forecast describes the outcome of those actions? Evaluate semantic agreement at the same times and under the same spatial definition, alongside coverage and informative baselines.

The primary existing case study is the historical Cosmos Edge DROID V1 corpus: 80 simulated episodes and 752 action/forecast chunks. The existing Qwen visual-localization audit is descriptive and fallible. Distinct Nano/DreamZero scene interventions may supply supporting evidence but are not pooled with this corpus or used as a substitute for prediction accuracy.

## Immutable evidence

- Repository: https://github.com/adeeb10abbas/steerable
- Source pin: `ce561e66f82e95055e39d3d7711691982f6b2086`
- Historical result, protocol, calibration and audit files remain unchanged.
- All new work stays under `workshops/corl2026_world_models/`.
- New analysis is a disclosed retrospective amendment. Known historical outcomes cannot be described as unseen or newly preregistered.
- No inference, new robot runs, remote compute, external messages, submission, or public release has been performed by this sprint.

## First-sprint deliverables and order

1. **Reconcile the evidence.** Verify all 752 chunk rows, episode identities, localization-cache links and hashes. Record what raw states and forecast images are present and what is missing. Recompute every historical count used in the manuscript.
2. **Specify and implement the analysis correction.** Reproduce the temporal and geometric mismatch. Use identical forecast/execution time and relation definitions when source data permits. Refuse to label incomplete or proxy comparisons as validated fidelity. Add regression checks for malformed, duplicate, missing and unaligned data.
3. **Challenge the contribution.** Compare against the nearest primary-source literature; define the claim beyond the established observation that visual quality need not imply control utility. Prepare a blinded annotation and held-out validation protocol.
4. **Produce the paper package.** Write an editable four-page working draft and figures generated from verified results; compile with the official CoRL template; render and inspect all pages. Include source and reproducible build instructions.
5. **Independent review and handoff.** Review the analysis and manuscript claims. Separate completed results from pending validation and identify the smallest concrete next evidence requirement.

## Completion conditions for this sprint

- A reproducible audit of the committed historical corpus, with no invented observations.
- An explicit decision on whether aligned prediction fidelity is computable from available data.
- A tested analysis entry point and exact recovery inventory if required data is absent.
- A substantive editable manuscript and readable PDF, labeled as a working draft.
- A research roadmap that targets stronger empirical evidence rather than stronger wording.

## Submission conditions beyond this sprint

Recover and verify required raw evidence; complete common-time/common-predicate scoring and independent labels; compare against justified simple baselines; obtain author and publication-eligibility decisions; recheck the workshop's final page, anonymization, archival and submission rules. The public call currently specifies up to four pages and leaves deadlines, OpenReview and archival status unsettled.
