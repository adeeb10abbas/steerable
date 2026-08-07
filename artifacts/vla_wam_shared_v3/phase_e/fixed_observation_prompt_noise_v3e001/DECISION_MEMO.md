# V3-E001 decision memo

Status: **complete (336 valid fixed-observation requests; zero behavioral episodes).**

At each registered settled observation, the only intervention was the exact
static prompt: “Put the Rubik's cube to the left of the bowl.” versus “Put the
Rubik's cube to the right of the bowl.”. The four observation arrays are
hash-bound to the V3-B001 control and position-mirrored fixtures. π0.5, Cosmos3
Nano Policy DROID, and DreamZero each supplied 27 matched LEFT/RIGHT pairs per
layout plus the registered exact repeats. No returned action was executed.

The compact compiler reports 27/27 matched prompt effects for every model and
layout, exact-repeat bit identity for all six model/layout cells, and retains
the raw request-file hashes. Nano decoded futures remain on the PVC; the
compact report retains action summaries only. DreamZero's first pass (112
infrastructure-invalid rows caused by an incorrect OpenPI-style client
coercion) is preserved separately and excluded from the denominator.
The prompt-to-noise diagnostic is checkpoint-specific: DreamZero's measured
same-prompt cross-seed RMS is zero (so the relative ratio is unbounded), while
π0.5 and Nano report nonzero same-prompt variation and their ratios are listed
without pooling action scales across model families.

## Counts

| quantity | registered | completed |
|---|---:|---:|
| model requests | 336 | 336 valid |
| behavioral episodes | 0 | 0 |
| infrastructure-invalid attempts | — | 224 retained, excluded |

## Primary interpretation

This is evidence for a fixed-state prompt intervention, not a task-success
claim. The safe manuscript sentence is: “At an identical observation,
changing only the directional prompt produced a reproducible action change in
all 27 matched sampling seeds for each tested checkpoint and layout; the
effect must be interpreted relative to the measured same-prompt cross-seed
variation in the compact report.”

## Evidence

`results/compiled_results.json` is the compact hash-bearing report. Raw
requests, decoded Nano futures, and DreamZero retained futures remain on the
PVC under the registered Phase-E raw root and are not committed.
