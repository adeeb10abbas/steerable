# Forecast-layout blind annotation workflow

Status: implemented packaging, locked-response, and mechanical adjudication
contracts; no human labels or empirical movement threshold exist yet.

This workflow implements the annotation boundary in `ABLATION_SPEC.md` for
`WMF-ABLATION-001`. It does not localize objects, decide that development
measurement is usable, recruit raters, or turn a model/VLM output into truth.
Its purpose is to make those missing inputs explicit and to prevent identity or
outcome information from entering the rater packets.

The executable is
`../analysis/forecast_annotation_workflow.py`; its artifact contract schema is
`../experiments/forecast_layout/annotation_workflow.schema.json`.

## What is private and what a rater receives

The request inventory, selection manifest, image inventory, and restricted map
are analyst-only. They retain the exact source request, source image, source
video, model, layout, condition, episode, camera, alignment receipt, render
receipt, paths, and SHA-256 identities. They must never be sent to raters or a
blind adjudicator.

Each rater receives exactly one opaque batch directory at a time:

- `packet.json`, containing opaque image IDs, dimensions, and randomized order;
- `media/img_<opaque>.png`, one isolated, overlay-free image per item;
- the frozen rubric and model/condition-blind illustrated examples; and
- `response_template.json`.

The two rater streams use different opaque IDs and independently frozen orders.
No released batch contains two images from the same selected request. The same
rater must never have identified or simultaneous counterpart access. A prior
batch must be collected and access revoked before the next batch is released;
the rater-stream parent directory must never be distributed. Packet JSON contains
no model, layout, condition, command, success, counterpart, source request,
source video, episode, or original filename. Visual generation artifacts may
still reveal image origin, so every rater must record a source guess; this is a
measurement of imperfect blinding, not a promise that source is visually
indistinguishable.

Byte-identical sanitized PNGs are included once per rater after exact hash and
dimension checks. The restricted map retains every source/role alias, allowing
an identical label to be reused without erasing provenance.

## 1. Build the complete metadata-only request draw

The input `wmf-forecast-request-inventory-v1` must carry the complete planned
cell roster for D01-D04 or C01-C24 under the declared full or reduced cohort
branch, including valid, censored, technical-invalid, and not-run cells. Every
valid recording contributes a unique contiguous request-index sequence (15 for
a complete N3 cell and 57 for a complete D1 cell). `valid_complete` means
exactly 450 executed actions. A `valid_censored` safety-abort trace must record
1-449 actions; its request count and executed prefixes must reconstruct that
count using the qualified 32-action N3 or eight-action D1 chunk boundary. Every
complete, censored, or technical-invalid roster row must cite a signed recording
receipt by path and file hash. That receipt binds the stage, cell, recording,
model, layout, condition, status, source video, and a signed action manifest.
The action manifest contains one contiguous row per action actually executed,
including its action hash, owning request index, control timestamp, physics-step
identity, and camera-frame identity. Selection reloads both files, verifies all
hashes, reconstructs request starts/prefixes, and checks every action-to-request
assignment. Counts written only into the inventory are therefore insufficient.
The inventory includes time-ineligible
requests and cells that may have zero eligible requests. A
request carries actual timestamp error/tolerance and a hash-bound alignment
receipt. A per-model self-hashed alignment contract fixes H, generated frame,
executed action offset, control/capture timing, tolerance, camera crop, and
pixel dimensions. It must state whether the target lies in the executed prefix; model
output visibility, object visibility, forecast quality, success, scorer output,
or human labels are not accepted fields.

The algorithm and seed are already fixed by `ablation_spec.json`: select up to
four eligible requests per episode using seed `2026091302`. The implementation
uses a cross-runtime SHA-256 rank over seed, episode ID, and source request ID.
It records the full inventory, reasons for ineligibility, zero-eligible
episodes, and exact within-episode inclusion probability. An unclear selected
image is never replaced.

```sh
python3 workshops/corl2026_world_models/analysis/forecast_annotation_workflow.py \
  select-requests \
  --inventory /restricted/request_inventory.json \
  --output /restricted/request_selection.json
```

This command is run after a stage has produced its request inventory and before
any labels for that stage are inspected. For confirmation, the sampling
algorithm/seed must already be frozen before confirmation execution even though
the actual selected IDs cannot exist until afterward.

## 2. Freeze a real rubric on development material

The frozen rubric must identify, in words and model/condition-blind illustrated
examples:

- the visible projected silhouette center for the Rubik's cube and bowl;
- the visible projected bowl-width landmark;
- partial-occlusion and identity-uncertainty rules;
- explicit ambiguity codes including `none`; and
- the rule that hidden centers are never inferred.

Its illustrated examples must be model-blind, condition-blind, and outside the
confirmation cohort. The rubric cites a signed `illustrated_examples.json`
manifest whose only media entries are source-free `examples/example_NNN.png`
files. The validator checks every hash, dimension, PNG structure/metadata chunk,
and identity/outcome/source-free caption. It also requires a separate signed,
hash-bound pixel-blindness receipt from a named human reviewer covering the exact
media hashes and model, condition, instruction, outcome, path, counterpart, and
overlay leakage. PNG parsing can reject metadata but cannot determine whether
identifying words were drawn into pixels; only the required human visual review
addresses that risk. The code does not claim to cleanse or recognize embedded
pixel text. The same rubric hash must govern duplicate development
validation and confirmation. If the rubric changes after validation, validation
must be repeated under the new hash.

The development freeze must use status
`frozen_for_development_validation` and exactly this unresolved threshold block:

```json
{
  "status": "pending_duplicate_development_labels",
  "threshold_relative_image_diagonal": null,
  "development_summary_path": null,
  "development_summary_sha256": null
}
```

Any numerical value at this point is rejected. Validate the artifact before
making development packets:

```sh
python3 workshops/corl2026_world_models/analysis/forecast_annotation_workflow.py \
  validate-freeze --stage development --freeze /restricted/development_freeze.json
```

## 3. Build two independent development packets

The image inventory is produced only after request selection and alignment. It
contains one sanitized lossless PNG for each required role:

- `current`, `predicted`, and `executed` for every selected request;
- `preceding` when a real preceding observation exists;
- no synthetic preceding frame for request zero, whose constant-velocity
  baseline is persistence; and
- `early_predicted` and `early_executed` only when the fixed A9 earlier horizon
  is supported.

Each record binds its existing source image/video bytes, camera/crop, alignment
receipt, dimensions, and exact PNG hash through a signed structured rendering
receipt. The packager resolves and hashes both source and rendered files and
rejects any source/role/path swap, missing or extra
roles, overlays, non-PNG media, dimension/hash drift, unselected requests, or a
changed video/camera/alignment identity. It parses and CRC-checks the PNG and
allows only standard pixel/color/geometry chunks; text, EXIF, timestamp, ICC,
and other ancillary metadata cannot carry a source filename into the packet.
Separately, the image inventory must cite a signed human pixel-blindness receipt
whose reviewed hash/dimension population exactly equals the deduplicated rendered
media population. A false, missing, stale, or partial receipt fails closed.

Keep packet output and restricted output in separate directories:

```sh
python3 workshops/corl2026_world_models/analysis/forecast_annotation_workflow.py \
  package \
  --selection /restricted/request_selection.json \
  --image-inventory /restricted/annotation_images.json \
  --freeze /restricted/development_freeze.json \
  --packet-root /distribution/development_packets \
  --restricted-map /restricted/development_identity_map.json
```

Within each rater directory, distribute only one `batch_*` directory at a time;
collect its locked response and revoke access before releasing the next. Never
distribute a rater-directory parent or the common parent. Each rater fills every
response field, records cube and bowl resolvability separately, records
`unknown` rather than guessed coordinates, signs all
independence/blinding attestations, and locks the file. The two `rater_code`
values must represent distinct people.

## 4. Derive the empirical movement-resolution input

After both development responses are complete and locked, derive the specified
measurement convention:

```sh
python3 workshops/corl2026_world_models/analysis/forecast_annotation_workflow.py \
  summarize-development \
  --restricted-map /restricted/development_identity_map.json \
  --rater-a-response /restricted/rater_a_locked_responses \
  --rater-b-response /restricted/rater_b_locked_responses \
  --output /restricted/development_label_noise.json
```

For each image resolvable by both raters, the script computes each rater's
cube-minus-bowl vector divided by the image diagonal, then their Euclidean
disagreement. The operational A6 threshold is the 95th percentile using a
frozen type-7 linear quantile. Unknown-to-either images remain counted and do
not receive fabricated vectors. Each response input is a directory containing
exactly one locked JSON response per released batch. The output reports duplicate count,
categorical disagreement, source guesses, annotation time, per-image
disagreement, exact input hashes, and the empirical threshold.

That output deliberately has status
`EMPIRICAL_DEVELOPMENT_RESULT_REQUIRES_EXPLICIT_USABILITY_DECISION`. The code
does not decide that the number of resolvable duplicates, movement-to-noise
scale, rubric behavior, or cost is adequate.

The summary marks every exact first-pass label disagreement as requiring
adjudication and stores only hashes of the two rater codes. Build the actual
blind third-rater stream from either a development or confirmation restricted
map and the two locked first-pass response directories:

```sh
python3 workshops/corl2026_world_models/analysis/forecast_annotation_workflow.py \
  package-adjudication \
  --restricted-map /restricted/development_identity_map.json \
  --rater-a-response /restricted/rater_a_locked_responses \
  --rater-b-response /restricted/rater_b_locked_responses \
  --packet-root /distribution/development_adjudication_packets \
  --adjudication-map /restricted/development_adjudication_map.json
```

The restricted adjudication map binds the two first-pass response artifacts,
the original restricted map, exact disagreement set, packet manifests, opaque
IDs, and media hashes. It must never be distributed. The adjudicator receives
only one source-free batch at a time, cannot see first-pass labels, and supplies
a locked `adjudicator` response under a third canonical rater code. Merge only
after that response is complete:

```sh
python3 workshops/corl2026_world_models/analysis/forecast_annotation_workflow.py \
  merge-adjudication \
  --adjudication-map /restricted/development_adjudication_map.json \
  --adjudicator-response /restricted/adjudicator_locked_responses \
  --output /restricted/development_final_consensus.json
```

The resulting `wmf-forecast-final-consensus-v2` is derived mechanically: exact
first-pass agreements are copied unchanged, and every disagreement uses the
independent adjudicator label. The validator reopens every packet and media file,
verifies the response/manifest hashes and distinct rater identities, recomputes
the disagreement set, and reproduces the full signed consensus. Hand-authored
consensus booleans or labels cannot satisfy this gate. `unknown` remains valid.
These gates still do not assert that a person has performed the pending work.

## 5. Confirmation freeze is fail-closed

Before releasing C01-C24, an authorized scientific reviewer must inspect the
development recordings, coverage/disagreement summary, illustrated cases,
movement scale relative to the empirical threshold, and measured annotation
throughput. A confirmation freeze is valid only when it:

1. cites the exact development-summary file and SHA-256; validation reloads
   every locked response and packet/media artifact and reproduces the summary;
2. copies its exact 95th-percentile value without editing or rounding;
3. cites the same rubric hash used for duplicate validation;
4. preserves the exact development cohort branch, qualified model set, and
   per-model alignment-contract hashes, so a reduced N3 study cannot add D1; and
5. includes `measurement_usable: true` with an evidence-based reviewer
   rationale and the validated final-consensus artifact.

Validate it before confirmation execution:

```sh
python3 workshops/corl2026_world_models/analysis/forecast_annotation_workflow.py \
  validate-freeze --stage confirmation --freeze /restricted/confirmation_freeze.json
```

A missing summary, missing/changed threshold, cross-cohort/model/alignment drift,
unapproved measurement, missing adjudication, or changed rubric fails. If development cannot resolve motion at the qualified horizon,
retain that as a qualification result; do not extend the action chunk or invent
a threshold.

After confirmation execution, make its metadata-only selection and packets
using the already validated confirmation freeze. Do not inspect confirmation
labels or effects while deciding the design.

## Human dependencies that code cannot satisfy

The following are real pending inputs, not software defaults:

1. An illustrated rubric frozen by people using development/calibration images
   that contain no confirmation material or model/condition labels.
2. A named human visual reviewer who signs the exact-hash pixel-blindness
   receipts for the illustrated examples and each stage's rendered annotation
   media; metadata parsing is not a substitute.
3. Two genuinely independent human raters, each uninvolved in scorer
   construction and given only one source-free batch at a time.
4. A separate blind adjudicator for first-pass disagreement. The restricted map
   is transformed by `package-adjudication` into fresh isolated-image dispute
   packets; it must never be handed directly to the adjudicator. `unknown`
   remains permitted. The same build/respond/merge route applies after
   confirmation first-pass labels.
5. A scientific judgment that development localization and movement are usable,
   with the evidence and signer recorded in the confirmation freeze.

No language-model or unvalidated VLM labels satisfy these dependencies. The
current implementation prepares and validates both first-pass streams, the
development noise receipt, source-free adjudication streams for both stages,
and a mechanically reproducible final-consensus artifact. Nothing here claims
that human annotation, pixel-blindness review, adjudication, or scientific
approval occurred until their hash-bound inputs actually exist.

## Test

```sh
python3 -m unittest \
  workshops.corl2026_world_models.tests.test_forecast_annotation_workflow -v
```

The focused tests cover deterministic metadata-only selection, zero-eligible
episodes, forbidden outcome/visibility inputs, complete/censored action-request
bounds, full N3+D1 and reduced cohorts, null development thresholds,
source-hiding and simultaneous-counterpart exclusion, distinct rater orders,
source/render provenance, example-content leakage, exact PNG hash/dimension/role
gates, a separate human pixel-blindness receipt, locked response rules,
source-free stage-generic third-rater adjudication, media/manifest tamper
rejection, cross-cohort rejection, the type-7 q95 derivation, exact confirmation
binding, and real-fixture validation with JSON Schema Draft 2020-12.
