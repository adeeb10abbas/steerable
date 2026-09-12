# WAM semantic evidence audit

Audit date: 2026-09-12. Scope: the four V1 Cosmos confirmation cells and their
80 episodes / 752 replan chunks. Upstream commit `ce561e66` remains immutable.
This is a disclosed post-result measurement audit, not a new model experiment.

## What is verified

The standard-library auditor hashes 761 input files and verifies their Git blob
identities against upstream commit `ce561e66`. All 752 localization caches bind
to calibration SHA256
`56b7155fdb2eee1732e7636a104b2966323fcfc947c699cf3ac58635b232ace9`.
No matching historical SHA256 receipts for these 761 specific inputs were found
in the supporting manifest; current SHA256 values and upstream Git identities
are separate receipts, not a claim of historical SHA256 verification.

Every CSV row joins exactly one cache. Duplicate, missing, unexpected cache IDs,
wrong cell totals, missing episode IDs, frame-index changes, calibration-hash
changes, semantic discrepancies and source modifications cause failure.
The offline replay recomputes each frame's planar relation and cross-camera
distances from cached world XY, checks all stored frame reliability/reasons,
and reconstructs the legacy aggregation. It does not rerun localization,
unproject raw pixels again, inspect images, or independently validate execution
labels from absent trajectories.

All 3,008 frames and all 752 rows reconcile without discrepancy. Reliable frames:
1,460. Overlapping rejection reasons: 1,201 camera relation disagreement,
710 cube distance disagreement, 338 bowl distance disagreement, one missing
localization. The 0.20 m legacy counts are 22 both, 5 future only,
3 execution only, 391 neither and 331 abstentions. Coverage is 421/752;
positive-execution coverage is 25/97.

## Why the eight mismatches are not eight prediction errors

The original scorer compares different temporal and geometric quantities:

1. Lines 686–694 aggregate requested labels over reliable generated frames
   8, 16, 24 and 32, requiring at least two reliable frames and a requested
   fraction at least 0.75 (positive) or at most 0.25 (negative).
2. Lines 696–705 select one executed endpoint at
   `min(executed_step_start + open_loop_horizon - 1, state_length - 1)`.
   Frame/action alignment and terminal clipping cannot be checked from the CSV
   alone; `execution_end_step` does not recover start, horizon or trajectory length.
3. `_unproject_to_plane` returns **world XY**, on one fixed visual-centroid plane
   at 0.07855224609375 m. `_frame_semantics` applies its 45-degree direction cone
   directly to those XY coordinates. In contrast, execution uses cube and bowl
   **root poses**, rotates their relative displacement into the robot frame,
   and requires `abs(delta_z) <= 0.1 m`. The code's calibration prose calls the
   rule robot-frame, but predicted XY are not explicitly transformed there.
   Whether world and robot axes coincide in these actual episodes must be
   checked from the missing robot poses/configuration. A coordinate-system
   implementation difference is established; its empirical effect is unknown.
4. Prediction represents visible object centers; execution represents root
   origins. A rotated cube's bounding-box centroid need not equal its root pose,
   as the upstream `_episode_state` comments also explain.

Consequently neither the eight discordances nor the 413 concordances isolate
world-model prediction error. Localizer error, selection, temporal pooling,
geometry, and genuine model discrepancy can contribute.

## Historical-label arithmetic and sensitivity

On the **same 421 covered chunks**, observed label agreement is 413/421 (98.10%).
A constant-negative label agrees with the historical execution label on
396/421 (94.06%). The difference is 17/421 = 4.04 percentage points. These are
historical-label arithmetic; they do not establish predictive utility, accuracy
against human truth, or corrected fidelity.

Replaying the original threshold rule at 0.10 / 0.15 / 0.20 m yields
41 / 262 / 421 certain chunks and 1 / 10 / 25 covered positive executions,
with the executed-positive denominator fixed at 97. No threshold was retuned.

A separately named **posthoc endpoint-frame aggregation sensitivity** selects
only cached frame 32 when reliable. It yields 282/752 covered chunks,
18 both, 9 future only, 3 execution only, 252 neither, and 470 abstentions;
21/97 positive executions are covered. This changes aggregation and reliability
selection together and still compares against the geometrically asymmetric
legacy execution label. It does **not** establish temporal alignment or corrected
fidelity and should not replace the frozen table as though it did.

For descriptive coverage uncertainty only, 2,000 bootstrap replicates sample
whole **four-episode matched-seed blocks** with replacement within the 6100 and
7200 tiers, retaining every chunk and using pooled chunk ratios (seed 20260912).
There are 20 independent seed blocks: ten canonical/short × LEFT/RIGHT blocks
at seeds 6100–6109 and ten declarative/contrastive × LEFT/RIGHT blocks at seeds
7200–7209. For every CSV row the audit validates
`sampling_seed == (tier + episode_index) * 1000 + replan_index`, using the
documented upstream sampling convention (`docs/SEMANTIC_FUTURE_SCORER_V1.md`,
line 161), and checks all four wording/direction cells in each block.
Each replicate draws ten blocks within each tier. It never samples constituent
episodes or chunks independently.

The corrected 95% percentile interval is **53.61–58.30%** for legacy coverage
and **19.00–33.33%** for positive-execution coverage. Endpoint-sensitivity
coverage intervals are 33.29–41.34% overall and 12.87–29.82% for positive
executions. The earlier eight-stratum episode bootstrap broke shared-seed
dependence and is superseded. The intervals still exclude evaluator bias,
human annotation error, and deployment generalization; 80 episodes are not 80
independent replicates here.

## Reproduction and outputs

From the repository root:

```sh
python3 -m unittest discover -s workshops/corl2026_world_models/tests -v
python3 workshops/corl2026_world_models/analysis/evidence_audit.py
```

No third-party packages, model weights, GPU, inference, or network are required.
Regression tests reproduced the shared-block and adapter-reliability defects
before their fixes; all 13 evidence-audit tests now pass.
They cover cohort loss/duplication, temporal disagreement, abstention, symmetric
planar geometry, joint four-episode resampling with every chunk retained,
incomplete/inconsistent seed blocks, and the normalized adapter's
alignment/geometry/reliability/provenance gates. Equal-valued booleans and floats
are rejected as times on either side. Both sides require typed boolean
reliability; execution-unknown records are explicitly ineligible for fidelity,
while prediction uncertainty remains abstention even when its coordinates are
absent. A third regression first reproduced the missing-coordinate failure.
Reliable records still require finite numeric XY.

The adapter is a validation stub, not a corrected scorer. Passing it is
necessary but insufficient: source alignment and observable validity still need
independent evidence. No recovered records or new human labels were scored.

Outputs in `../results/`: `audit_summary.json`, `source_hashes.json`,
`reconciled_chunks.jsonl` (every cohort ID, original/sensitivity label and source
location), `legacy_mismatches.json` (all eight, without attribution),
`missing_data_inventory.json`, and `missing_media_inventory.json`.
The raw-data recovery boundary is described in `DATA_RECOVERY.md`.
