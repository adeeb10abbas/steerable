# Private annotation sampling handoff

**PROVISIONAL PRE-ANNOTATION DESIGN — media recovery, endpoint alignment,
rubric freeze and independent human annotation remain pending.**

The private manifests `../results/annotation_sample.json` and
`../results/annotation_sample.csv` contain 160 unique historical chunk IDs.
They are analyst materials, not rater sheets. Their historical strata are not
human truth and must not be exposed to raters or an adjudicator. No image packet,
alignment, counterpart pairing, human label, or asset availability is certified
by this draw.

## Draw and receipts

The draw follows the six strata in `ANALYSIS_PROTOCOL.md`, in that table's order.
It sorts source IDs canonically inside each stratum and uses simple random
sampling without replacement with Python `random.Random(20260912).sample`.
Selected IDs are sorted within stratum for the private manifest, which must
never be reused as rater presentation order. The runtime version is recorded.

| Historical stratum | N_h | n_h | Inclusion probability | Population weight |
| --- | ---: | ---: | ---: | ---: |
| Both positive | 22 | 22 | 22/22 | 22/22 |
| Future-only positive | 5 | 5 | 5/5 | 5/5 |
| Execution-only positive | 3 | 3 | 3/3 | 3/3 |
| Both negative | 391 | 50 | 50/391 | 391/50 |
| Forecast abstention, execution positive | 72 | 40 | 40/72 | 72/40 |
| Forecast abstention, execution negative | 259 | 40 | 40/259 | 259/40 |

The sample weights sum to 752 exactly by stratum design. Population and sample
counts, duplicate IDs, malformed booleans, and inconsistent historical quadrants
are checked before drawing. JSON provenance records SHA256 of the complete
population file, canonically sorted full population, sampling script, protocol,
canonical sample array, and emitted CSV. The draw hash covers the selected
records including weights and pending-state fields; the JSON describes its
exact serialization. These are reproducibility receipts, not asset verification.

Reproduce from the repository root:

```sh
python3 -m unittest discover -s workshops/corl2026_world_models/tests -p 'test_annotation_sample.py' -v
python3 workshops/corl2026_world_models/analysis/prepare_annotation_sample.py
```

Do not silently regenerate a different design after any new corpus labels are
seen. Record the final frozen protocol/script/population/draw hashes and
annotation-start timestamp before annotation. The present status explicitly
remains provisional.

## Prepare isolated image packets only after the recovery gate

**Additional execution-image gate:** the raw manifest has no separate actual
execution video/image receipts under the four selected Cosmos run roots.
RGB content inside the 80 HDF5 files is unverified. Recovering the existing
840 + 2,256 inventory entries therefore does not ensure that execution endpoint
images can be presented to raters. See `DATA_RECOVERY.md` and
`../results/execution_imagery_recoverability.json`. Require a hash-verified RGB
schema and decodable timestamped samples, independently recovered original
execution recordings with identity/hash/timing receipts, **or an original later
chunk's conditioning image whose metadata and observation/action timing prove
that it depicts the required endpoint in the same camera**, before making
packets. Preserve its image hash and alignment evidence. This conditioning-image
route is a candidate, not a verified pairing: terminal chunks, absent later
images, and unresolved time matches remain missing unless another original image
establishes that endpoint.
The existing generated-future contact sheets and state trajectory plots do not
satisfy this requirement. If actual imagery is unavailable, do not start the
projected human endpoint audit.

1. Recover original media and metadata and verify the historical hashes listed
   in `DATA_RECOVERY.md`. Trace each sampled request from conditioning time t0
   to its last actually executed action. Establish the forecast frame and
   execution observation at that same physical time, including truncation and
   camera identity. Record evidence and actual alignment error. Unresolved
   samples remain `alignment_unknown`; retain their sampled IDs and original
   weights, and do not substitute more legible chunks.
2. Freeze the common projected observable, illustrated center/width/unknown
   rubric, source-guess question, primary camera, target-side mapping, scorer
   configuration and any named sensitivities. Practice only on separately
   generated or model-blind calibration material. No language-model guesses
   become corpus annotations. Record authors' prior exposure if selected images
   overlap the earlier qualitative audit.
3. After alignment is verified, extract each chunk's t0, forecast endpoint and
   execution endpoint as **individual images**. For a single primary view,
   160 fully recoverable chunks would yield 480 images; this is a conditional
   planning count, not a count of existing assets. Use identical formatting and
   remove source names, overlays, prompts, task directions, timing labels,
   historical scores and any identifying metadata.
4. Assign fresh opaque image IDs in a separately frozen randomization step.
   Keep the mapping from IDs to source chunk, image role, old stratum and
   counterpart in restricted analyst storage. Rater packets contain only
   isolated images, opaque IDs, and the frozen rubric/response fields. They
   contain neither the sampling manifest nor contact sheets or paired images.
   Schedule randomized batches so counterparts are not adjacent or identifiable;
   use independently randomized orders for each rater. Report source guesses
   because visual generation artifacts may compromise source blinding.
5. Two independent raters, uninvolved in scorer construction and the previous
   audit, independently record object identity, visible centers, bowl width,
   resolvability and ambiguity reason for each isolated image. A separate
   adjudicator resolves disagreement while still blind to source pair and old
   label. Retain `unknown` when unresolved. Joining images and command polarity
   occurs only after independent labels are complete and locked.
6. Keep source/media absence, alignment unknown, forecast unreadability,
   execution unreadability and technical invalidity distinct. Report sample
   counts and weighted corpus estimates separately. The initial weights apply
   to the selected probability sample; missingness cannot be repaired by dropping
   cases, redistributing their weights or silently redrawing. Account for finite
   population strata and episode or matched-seed dependence as specified in the
   protocol. Do not report an unweighted enriched-sample accuracy as a corpus rate.

No packet distribution, recruitment, human labels, spending, inference, remote
execution, or model/scorer training is performed or authorized by this handoff.
