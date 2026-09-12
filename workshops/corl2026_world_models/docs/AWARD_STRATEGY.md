# Strategy for a competitive WAM research paper

Prepared 2026-09-12. Research judgment and proposed gates, not an award
prediction. The current corpus is known retrospectively; no new validation is
claimed. The detailed measurement plan is in [ANALYSIS_PROTOCOL.md](ANALYSIS_PROTOCOL.md).

## Strongest central question

**Does a WAM's exposed future tell us what its action will change, beyond what
the current observation already tells us?**

The strongest contribution is a paired, time-aligned, coverage-aware audit of
incremental predictive evidence. A useful result may be positive or negative.
The paper should explain *when* a generated future supplies reliable evidence,
where it abstains, and whether it improves on persistence. It should not compete
as a generic critique of attractive but physically wrong videos, a WAM-versus-
VLA leaderboard, or a claim that world models are unnecessary.

The opportunity is to expose an evaluation failure that matters for robotics:
high selective agreement can coexist with very weak coverage of consequential
changes, and incompatible forecast/execution measurements can obscure this.
An award-level version must turn that observation into a validated method and
an independently replicated finding. A rewritten introduction alone cannot do
that.

## Evidence ladder and claims today

| Claim | Status and required wording |
| --- | --- |
| An old scorer reports 413/421 agreement | Supported historical computation, conditional on its certain subset and incompatible measurements; do not call this validated forecast fidelity. |
| Agreement is dominated by both-negative labels | Descriptive: 391/421 certain cases are both negative. Show all four cells and 331/752 abstentions. |
| Positive-event coverage is limited | Descriptive under the old execution predicate: 25/97 execution-positive chunks receive a certain forecast label. This is coverage, not 25/97 correctness. |
| The old measurement comparison is confounded | Supported by audited scorer definitions: time aggregation versus endpoint, visual centroids versus root poses, coordinate differences, and missing forecast height. |
| WAM futures fail to predict physical outcomes | Not established by those numbers. Needs identical time/entity/coordinate predicates and independent labels. |
| Generated futures predict changes beyond persistence | Open primary question, requiring the protocol's aligned comparison. |
| World models cause good/bad action or are unnecessary | Unsupported: joint future/action outputs provide no removal intervention or arbitrary action-counterfactual test. |
| Nano reflection and DreamZero symmetry results explain Cosmos fidelity | Unsupported. These are distinct checkpoint/cohort experiments and can only motivate separate questions. |

The 24 existing contact sheets may illustrate audit methodology and transparent
ambiguity. They cannot certify general accuracy, independence, or an unseen
validation result. Figures must distinguish historical computation, human
validation pending, and proposed experiments.

## What makes the paper difficult to dismiss

1. **Matched measurement:** expose conditioning time, generated target time,
   actual executed endpoint, camera/entity definitions, and exact missingness.
   An annotated alignment diagram should let a reader reconstruct one row.
2. **A demanding baseline:** show forecast and current-observation persistence
   on the same cases. Include class-conditional coverage and transitions, so
   stationary negatives cannot supply the entire narrative.
3. **Independent labels:** probability-sampled, blinded adjudication with
   unknowns retained; distinguish scorer validity from forecast validity.
4. **A reusable result:** release the manifest schema, alignment/scoring code,
   deterministic sample draw, weighting, clustered analysis, and an auditable
   media index under the relevant rights. Report hashes and actual availability.
5. **Independent replication:** a second exposed-future WAM under the same
   observable, with prospective freezing where possible and separate estimates.
   Heterogeneity is a finding; do not pool arenas or disguise incompatible
   interfaces as replication.

## Research versus position-paper gates

These are proposed editorial decision rules, not post hoc hypothesis tests.
Freeze them before the new blinded audit. Decisions concern claim strength and
future spending; they do not remove inconvenient observations.

| Gate | Research-paper requirement | If unmet |
| --- | --- | --- |
| Provenance | Recoverable hash-linked census with known time/camera mapping for every analyzed row | Report recoverability audit; do not estimate fidelity on guessed alignment. |
| Measurement | Identical observable on forecast and execution; at least 90% agreement on resolvability and relation in a separate rubric pilot, with disagreements examined blind | Refine rubric using development examples, freeze again, validate on separate images; otherwise position/methodology paper. |
| Missingness | Full flow diagram and positive/negative coverage; overall target population cannot be replaced silently by the readable subset | Narrow to the observable population or report bounds; abandon broad certificate claims. |
| Baseline comparison | Paired incremental BCC contrast with cluster-aware interval and transparent transitions | Case study or position paper; high agreement alone is insufficient. |
| Event information | Enough independent positive and negative blocks to estimate the contrast with intended precision | Report uncertainty and inconclusive result; do not count chunks as independent replication. |
| External check | Independently implemented second WAM with the same measurement and full protocol | Strong single-checkpoint case study remains possible, but award-scale generality is unsupported. |

The 90% rubric-pilot gate is an operational reproducibility target, not a
validated universal standard. State the pilot sample size and uncertainty; never
mistake passing it for proof of model fidelity. Prefer synthetic and clearly
separated calibration examples for rubric development.

Do not require a statistically significant failure of forecasts to proceed.
If aligned futures beat persistence with useful coverage, the central answer
becomes a positive certificate result. If they do not, a precise replicated
negative result is valuable. If the interval is broad, claim uncertainty.

## Two-stage plan with bounded cost

**Stage A: use existing evidence first.** Recover the full manifest and video/time
mapping; recompute only compatible endpoints; execute a blinded 160-chunk
probability audit; estimate the full annotation cost from observed practice
timings. Complete a census if affordable, otherwise preserve the sampling design.
Produce a reproducible endpoint dataset, confusion/coverage panel, persistence
comparison, uncertainty report, and a complete failure/missingness taxonomy.
Do not request GPU runs to compensate for an unresolved measurement problem.

**Stage B: validate an answer worth replicating.** Freeze a compatible independent
WAM cohort, count independent seed blocks for precision, and precommit the full
queue and budget. Existing untouched labels can support blinded annotation
validation even when behavioral outcomes were historically known; describe that
distinction exactly. New rollout collection needs separate authorization. Keep
all valid failures, retain videos and exact action/forecast timestamps, and score
only the horizon actually executed. No baseline requires secretly generating
counterfactual actions from an interface that does not support them.

Stop expanding after these two cohorts unless the result identifies a specific
unresolved mechanism with a feasible controlled experiment. More checkpoints
without comparable measurement add breadth without answering the central
question.

## Recommended narrative and figures

Lead with the operational question, then the exposed-future interface and
measurement contract. Show the historical number only as motivation for the
audit; place the aligned, independently checked endpoint result at the center
once it exists.

- Figure 1: one request, its conditioning observation, exact forecast endpoint,
  executed action window, and matched actual endpoint; identify unexecuted suffixes.
- Figure 2: all chunks as a denominator-preserving coverage/confusion display,
  with old and new measurement protocols clearly separated.
- Figure 3: paired forecast-versus-persistence certificate utility and transition
  performance, with episode/seed uncertainty and class coverage.
- Figure 4: independent cohort result and prespecified failure/ambiguity examples,
  selected by deterministic rules rather than visual appeal.

Geometry-reflection/symmetry evidence can occupy a short separate supporting
section if it sharpens the question about language, geometry, and control. It
must preserve its own cohort, intervention, estimator, and claim boundaries.
Omit it if page pressure would displace the main validation.

Until Stage A passes, the honest submission is a measurement case study or a
position paper arguing for this contract, with clearly labeled preliminary
evidence. The strongest research submission is the same question answered by
aligned data, useful baselines, independent labels, and replication. Neither
format can be promised a Best Research Paper award.
