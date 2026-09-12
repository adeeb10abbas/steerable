# Aligned future–execution audit: analysis protocol

Status: proposed timestamped retrospective amendment, 2026-09-12. The V1
results were inspected before this document was written. This is **not** a
preregistration of V1. No new human annotations, inference, or validation results
are asserted here. Original protocols and artifacts remain unchanged.

## Question and inferential boundary

Can an exposed WAM future certify the subsequent spatial outcome of its jointly
generated action chunk better than the observation already available at the
request boundary, at a useful coverage? The unit of prediction is a request-local
forecast at one aligned endpoint. The independent uncertainty unit is an
episode, or the complete matched-seed block when episodes share a fixture.

This tests incremental predictive evidence under the executed policy. It does
not test arbitrary action-conditioned counterfactuals, planner ranking, whether
the world model caused the action, or whether removing it would change control.
Results apply to the checkpoint, arena, camera, prompts, and horizon evaluated.

## Known evidence and amendment rationale

The reported Cosmos V1 corpus contains 80 episodes and 752 chunks. The old
scorer gave 413/421 agreement among certain labels: 22 both positive, five
future-only positive, three execution-only positive, and 391 both negative.
It abstained on 331/752 chunks. Only 25/97 execution-positive chunks received a
certain forecast label. These are historical scorer outputs, not independently
validated predictions. Always show their denominators.

The old comparison is unsuitable as the new primary endpoint: its future label
aggregates reliable frames 8/16/24/32 using a 75% rule, while its execution label
uses one endpoint. Forecast localization uses planar visual centroids; execution
uses robot-frame root poses and an additional height constraint. XY caches do
not recover missing Z or establish identical entity or coordinate definitions.
The existing 24-sheet inspection is a descriptive audit, not blinded validation.

## 1. Recoverability gate before any new labels

Build a census of every original chunk, retaining invalid/missing rows with
reason codes. Required columns are episode and matched-seed IDs; request ID;
prompt; model/runtime hashes; observation and forecast file hashes; requested and
executed action counts; forecast frame-to-action mapping; actual video/state
time mapping; camera identity and resolution; and original scorer labels.

For every included request establish a trace from the conditioning observation
at t0 to the last actually executed action tH. Derive the forecast frame that
represents **that same simulator time** from the adapter's documented timing,
not from equal frame indices or a visually convenient endpoint. Save the mapping
and its evidence. Check indexing, truncated final chunks, skipped video frames,
repeated conditioning frames, and any replanning before tH. A future of an
unexecuted suffix is not an observation of its counterfactual outcome.

Primary inclusion requires the same endpoint, camera/view definition, and entity
definition on both sides. If exact alignment cannot be established, classify
`alignment_unknown`; do not substitute the nearest nice-looking frame. A
separate sensitivity analysis may admit nearest recorded times within one
simulator control step, with the actual time error reported. If neither an
aligned video pair nor aligned physical states can be recovered, stop numerical
fidelity claims and publish the recoverability failure.

## 2. Freeze one common observable, not task success

The feasible media-only primary is a **projected spatial-relation certificate**.
Use the identical camera and image-based entity definition in forecast and
execution. Annotators mark the visible projected silhouette center of the cube,
the bowl center, and bowl width; complete occlusion, uncertain identity, or
insufficient visibility produces `unknown`. Do not estimate hidden centers by
using a later frame. For rotated or partially occluded objects, use the frozen
illustrated center-definition rubric; if the rubric cannot resolve the center,
retain uncertainty.

Define d = (cube_center_x − bowl_center_x) / bowl_width. Assign image-left when
d < −0.10, image-right when d > +0.10, and neutral otherwise. The 0.10 margin is
a proposed measurement deadband, not the historical 45-degree physical cone.
Freeze the rubric and margin after model-blind synthetic/calibration examples
and before new corpus labels. Report 0 and 0.20 margins as named sensitivities;
never select the best-looking margin.

Specify the camera-projected target side once from camera geometry and the
static LEFT/RIGHT command. If robot-relative directions cannot be mapped to
the declared image-side predicate, use three-class image-side prediction alone
and remove requested-side terminology. Perspective can make the projected
predicate differ from physical LEFT/RIGHT. A projected certificate measures
visual spatial evidence, not placement success, release, depth, or collision.

Only if both forecast and execution support the *same* physical entity centers,
camera calibration, coordinate transform, and XYZ measurement may a separate
physical predicate be evaluated. It must apply identical lateral, depth, and Z
thresholds to both sides. A planar assumption for the generated image does not
satisfy this gate during lifting. Simulator state is post-action scoring only.
Do not silently substitute the old root-pose task-success labels for media truth.

## 3. Primary endpoint and baselines

Let Y be the adjudicated execution-endpoint binary projected target relation,
F the forecast relation or abstention, and P the relation visible at t0 held
constant to tH. All labels use the same rubric. Prompt polarity is joined only
after independent image labeling. If the target-side mapping gate fails, freeze
the corresponding three-class macro-average before labels instead.

For each method M define balanced correct-and-covered rate:

    BCC(M) = 0.5 × [Pr(M = 1 | Y = 1) + Pr(M = 0 | Y = 0)].

Forecast abstention contributes no certificate to either numerator and remains
in the denominator whenever Y is resolvable. This is a certificate-utility
convention, not an assertion that an unobserved forecast is a behavioral error.
Execution-unknown cases have no gold Y and are never recoded as negative.
The primary contrast is BCC(F) − BCC(P), on the same execution-resolvable target
population. If a class is absent, the contrast is undefined, not perfect.

Required baselines, if their inputs are recoverable:

1. Current-observation persistence P, using only imagery available at t0.
2. Constant positive and constant negative certificates (each has BCC 0.5 when
   both classes exist), exposing prevalence-driven apparent agreement.
3. A prior-only predictor fitted on training episodes, reporting its proper
   probability score; it receives no execution-endpoint inputs.

Constant-velocity extrapolation is secondary only if at least two timestamped
pre-t0 observations and a common coordinate definition are available. Do not
manufacture motion from future frames. Action-aware fitted predictors require
logged action inputs and episode-disjoint training; they are optional and must
not be described as available until verified. No new model inference is needed
for the persistence or constant baselines.

Report alongside the primary contrast: every confusion cell; overall and
class-conditional forecast coverage; execution-label coverage; selective
agreement and balanced accuracy; positive predictive value; target-positive
sensitivity with abstention in its denominator; and performance on cases where
Y differs from P. This last transition subset is secondary, with its small
denominator and selection definition explicit. Also report three-class
image-left/neutral/image-right confusion to expose collapse hidden by polarity.

## 4. Blinded independent human adjudication

Stage A is an offline pilot audit, with no new inference. Prefer a full 752-chunk
label census if media are recoverable and cost permits. A bounded first pass is
160 chunks, probability sampled using seed 20260912 and canonical chunk-ID
sorting from the six historical strata below. Freeze and hash the sampling
script, population manifest, draw, and inclusion probabilities before annotation.

| Historical stratum | Population | Initial sample |
| --- | ---: | ---: |
| Both positive | 22 | 22 |
| Future-only positive | 5 | 5 |
| Execution-only positive | 3 | 3 |
| Both negative | 391 | 50 |
| Forecast abstention, execution positive | 72 | 40 |
| Forecast abstention, execution negative | 259 | 40 |

Verify these stratum counts against the immutable corpus before drawing. If
they differ, correct the disclosed manifest before labels, not after results.
Draw simple random samples without replacement within strata. The old labels
determine sampling only; they are not truth. Use inverse inclusion-probability
weights for corpus estimates. Publish both weighted estimates and raw sample
counts. A disagreement-enriched sample without weighting cannot estimate corpus
accuracy. Existing contact-sheet examples do not enter as an extra convenience
sample; if randomly selected, disclose prior exposure by authors.

Two independent raters, uninvolved in scorer construction and historical audit,
label separately randomized t0, forecast-endpoint, and execution-endpoint images.
Hide model name, prompt, old scores, paired identity, source-file names, and
counterpart image. Use identical presentation formatting and shuffled opaque IDs.
Rendering artifacts may reveal generated origin; measure raters' source guesses
and acknowledge imperfect source blinding instead of promising full blinding.
The same rater must not view counterpart images in an adjacent/identifiable batch.

Each rater records entity identity, centers, width, resolvability, and ambiguity
reason before any pair is joined. A third independent adjudicator resolves
disagreement using the same isolated-image rubric, still blind to pair and old
score. Allow `unknown` after adjudication; never force agreement. Report first-
pass disagreement, unknown frequency, center discrepancies, and adjudication
rate separately for forecast and execution. No human labels are supplied by the
authors or language-model guesses in this planning phase.

Use a separate training set of synthetic/model-blind examples for the rubric.
Keep audit labels hidden from scorer developers until their scorer/configuration
hash is frozen. If developers then tune thresholds, call the 160 images
development data; validate the tuned scorer on the remaining episode-disjoint
eligible data or an independent future cohort. Do not call a randomly held-out
part of previously inspected V1 a prospectively unseen model result.

## 5. Missingness, calibration, and uncertainty

Separate missing files, alignment unknown, forecast unreadable, execution
unreadable, and technical invalidity. Show their counts by episode, direction,
horizon, and historical outcome; these are informative measurement failures.
Never drop a valid model failure because it is visually inconvenient. Primary
inference is explicitly conditional on resolvable execution truth. Additionally
bound overall correctness by counting every execution-unknown case first as
incorrect and then as correct; class-balanced bounds require enumerating both
possible class assignments, not guessing class prevalence. Show complete-case
and conservative certificate results together.

A detector confidence score is not a probability that the forecast is correct.
Report scorer-vs-human error and coverage separately from forecast-vs-execution
fidelity. A reliability plot/Brier score requires a probabilistic certificate
trained/calibrated on separate episodes and evaluated untouched; with 80 episodes
use grouped cross-fitting and call this internal validation. Every paired seed
stays entirely within one fold. Fit all preprocessing and calibration on the
training fold. Show bin counts and uncertainty; do not claim calibrated validity
from one high selective-accuracy number.

Compute the primary contrast as paired predictions on identical rows. Use
10,000 resamples of whole episodes, or whole matched-seed blocks where relevant;
carry all selected chunks, paired methods, and sampling weights together. For
the stratified 160-chunk audit, account for the two-stage sampling design and
finite-population strata when reporting corpus uncertainty; a simple unweighted
chunk bootstrap is not acceptable. If that design-based implementation is not
available, report the weighted point estimate and clearly exploratory clustered
interval, and reserve primary inference for the full annotation census.

The corpus estimand is request-weighted; also report equal-episode weighting so
long failed episodes cannot silently dominate. Report a 95% interval for the
primary contrast; all secondary analyses are descriptive and named in advance.
Do not use 752 independent Bernoulli trials, chunk-level binomial intervals, or
choose a favorable endpoint after seeing its interval. Publish cluster counts
and sensitivity to leaving one episode/seed block out.

## 6. Cost-aware continuation and frozen validation

First spend only on manifest recovery, deterministic alignment, scoring existing
media, and the blinded audit. Measure annotation minutes per item on synthetic
practice material, then estimate full-census cost; do not quote unmeasured GPU
or annotation costs as receipts. Recoverability, timing, and annotation failures
are stop signals for further spend, not reasons to commission a wider benchmark.

A second WAM cohort is justified only after the measurement gate passes, the
audit finds the endpoint reproducible, and the answer remains scientifically
useful even if the forecast loses to persistence. Prefer existing media from an
independently released WAM with a decodable future and the same arena/predicate.
If its historical outcomes are known, label it retrospective replication.
Otherwise freeze checkpoint/runtime, complete seed list, prompts, horizon,
alignment rules, exclusions, scorer, human rubric, baselines, and analysis before
new outcomes or labels. Select the checkpoint for interface compatibility and
independent implementation, never for a promising previewed effect.

Choose a fixed episode/seed-block count by simulation of plausible event rates
and clustering using V1 only as development information. Require precision on
the primary contrast (target 95% interval half-width at most 0.10 BCC) and enough
positive/negative outcomes across independent blocks for a stable estimate.
Freeze the numerical N and maximum resource budget before launching; no stopping
when significance appears. Sparse events or failed precision produce an
inconclusive result and a disclosed follow-up, not unregistered extensions.

New inference, remote execution, recruitment, and spending remain future work;
this document authorizes none of them. V3 Nano/DreamZero geometry and symmetry
cohorts remain separate supporting evidence with their original estimands.
