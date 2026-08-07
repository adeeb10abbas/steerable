# V3-E001 decision memo

Status: **complete (336 unique valid fixed-observation requests; zero
behavioral episodes).**

At each registered settled observation, the only intervention was the exact
static prompt: “Put the Rubik's cube to the left of the bowl.” versus “Put
the Rubik's cube to the right of the bowl.”. The four observation arrays are
hash-bound to the V3-B001 control and position-mirrored fixtures. π0.5,
Cosmos3 Nano Policy DROID, and DreamZero each supply 27 matched LEFT/RIGHT
pairs per layout and two exact-repeat rows per layout. No returned action was
executed.

## Exact-repeat and provenance repair

All **12/12** registered base/repeat comparisons are present and unique after
source-priority deduplication. Every pair has equal action shape, equal
action SHA-256, `np.array_equal == true`, and numerical RMS exactly 0.0.
The raw ledgers contain 224 infrastructure-invalid rows, but these collapse to
112 unique invalid attempts: 112 duplicate copies, two invalid source files,
and one unique invalid source-content hash. Nano's earlier v1/v2 request
shards also contain nine duplicate valid rows; the final nano_v3 repair shard
is selected deterministically and all duplicates remain accounted for.

## Prompt effect relative to sampling variability

The report separates native full returned action chunks from the executable
prefix contract. E001 is request-only, so no executable prefix was consumed;
that field is explicitly marked unavailable rather than silently equated with
the native chunk. Semantic FK is likewise structured as unavailable because
no verified state/action-frame mapping is bound to these request ledgers.

| checkpoint / layout | median prompt RMS | pooled median same-prompt RMS | ratio | fraction above noise p95 | paired shift p |
|---|---:|---:|---:|---:|---:|
| π0.5 / control | 0.00380 | 0.09214 | 0.041 | 0.000 | <1e-5 |
| π0.5 / mirrored | 0.00307 | 0.09217 | 0.033 | 0.000 | <1e-5 |
| Nano / control | 0.03491 | 0.04845 | 0.720 | 0.000 | 0.167 |
| Nano / mirrored | 0.07063 | 0.05925 | 1.192 | 0.037 | 0.00314 |
| DreamZero / control | 0.00923 | 0.00000 | null (unbounded) | 1.000 | <1e-5 |
| DreamZero / mirrored | 0.02348 | 0.00000 | null (unbounded) | 1.000 | <1e-5 |

The paired systematic test uses 100,000 within-seed sign flips. It evaluates
the norm of the mean LEFT-minus-RIGHT action shift and mean pairwise cosine
agreement of seed-level shifts; it is not a task-success test. The eight
action-dimension RMS vectors are retained for every model/layout, rather than
reported as a scalar mislabeled “per-dimension” error. Layout interactions use
20,000-resample paired bootstrap intervals and exact two-sided sign tests.

## Claim boundary

The safe manuscript sentence is: “At an identical observation, changing only
the directional prompt produced a reproducible action change across 27 matched
sampling seeds for each tested checkpoint and layout; the magnitude and
directional coherence of that change must be interpreted relative to the
checkpoint-specific same-prompt sampling distribution.” This is a fixed-state
prompt/noise diagnostic, not a task-success claim and not evidence that the
prompt effect exceeds sampling variability for every checkpoint.

Raw requests, decoded Nano futures, and DreamZero retained futures remain on
the PVC under the registered Phase-E raw root and are not committed. The
compact report is `results/compiled_results.json`; its source-file hashes and
duplicate accounting are part of the evidence manifest.
