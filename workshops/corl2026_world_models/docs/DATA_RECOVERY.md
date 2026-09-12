# Exact data recovery boundary

Status: corrected like-for-like fidelity analysis is **not ready**. No GPU,
remote login, inference, data download, or new human annotation was performed.
Do not replace missing coordinates, alignment, or object labels with guesses.

`../results/missing_data_inventory.json` enumerates the minimum 840 raw-source
paths needed to recover execution geometry and audit alignment:

- 80 `run_N.hdf5` episode trajectories;
- 752 `predicted_chunks/episode_NNN/chunk_NNN/metadata.json` records;
- 8 `env_cfg.json` camera/coordinate configurations.

All 840 paths have exact historical bytes and SHA256 receipts in upstream
`artifacts/vla_wam_shared_v1/final_evidence/raw_evidence_manifest.csv`.
They are not retained in this sparse checkout. The paths identify historical
source locations; their existence on another host has not been checked.

`../results/missing_media_inventory.json` separately enumerates 2,256 supporting
raw files: each chunk's generated `future.mp4`, `conditioning.png`, and
`action.npy`, with historical hash receipts. Recovering these is needed for
visual evaluator validation, object-identity audits, and action/video interface
alignment checks. A receipt proves what bytes were recorded, not that the file
is presently recoverable or the observation is correct.

## Actual execution imagery is an additional unresolved gate

Recovering the 840 state/configuration files and 2,256 generated-media/action
files **does not guarantee actual endpoint imagery for human labels**.
`../results/execution_imagery_recoverability.json` records this gate for all
80 selected Cosmos episodes. The upstream raw manifest contains zero separate
execution video/image receipts anywhere under the four selected Cosmos run
roots. Its eight task directories contain 80 HDF5 trajectories, 80 episode logs
and eight configurations outside `predicted_chunks`. The manifest writer
includes `.mp4`, `.png` and `.jpg`, so this is not simply a video-suffix filter.

Whether those HDF5 files contain RGB or encoded images is **unverified**. The
semantic scorer reads only centroids and root poses; the trajectory renderer
reads poses and draws trajectory plots. Neither establishes an RGB dataset.
The 80 semantic contact sheets show annotated generated future frames, not
execution endpoint observations. The original V1 recorder implementation and a
complete per-episode RGB dataset inventory have not been established from the
retained evidence. These observations do not prove RGB is absent from HDF5.

Before any human execution labels, obtain a hash-verified HDF5 dataset inventory
and decodable image samples with camera IDs, frame counts and timestamps. If RGB
is absent, recover original execution recordings with exact source identities,
hashes and time mappings; their paths/hashes are not supplied by this manifest.
An additional recovery route is an original later chunk's `conditioning.png`.
It can supply the earlier chunk's actual endpoint image **if original metadata
and observation/action timing prove the same physical time, camera and entity
definition**. Retain the image hash and documented mapping. A subsequent request
index alone does not establish this match; observation timing offsets must be
resolved. Terminal chunks, missing later images, or an unproven time match remain
missing unless another original observation establishes the endpoint. This is a
candidate recovery route, not a claim that any such match has been verified.
Do not substitute trajectory plots or newly rendered simulator reconstructions
for original camera evidence. If no aligned actual imagery can be recovered,
the projected human endpoint annotation gate fails.

## Recovery and analysis sequence

1. Copy the listed artifacts from an explicitly authorized data source into a
   separate raw-data directory, preserving relative identity. Verify all bytes
   against the historical hashes before accepting them. Do not rerun any model
   to silently substitute for missing original evidence.
2. Verify exactly one `data/demo_0` group in each HDF5, as upstream requires.
   Recover `bbox/centroid/rubiks_cube`, `bbox/centroid/bowl`, and
   `states/articulation/robot/root_pose`; retain cube/bowl root poses separately
   for reproducing the historical execution labels.
3. Use each original metadata record and the released interface's documented
   frame/action convention to establish the physical sample corresponding to
   each prediction frame. Verify whether frame 0 is conditioning, whether the
   exposed horizon is 32 actions, and whether termination clips the endpoint.
   Record this mapping and its evidence; do not assume frame 32 equals a
   specific HDF5 index merely because its number is 32.
4. For a limited planar audit, apply the **same** world- or robot-frame
   visual-centroid XY cone to prediction and execution at the same physical
   sample. Disclose that this tests planar visual relations, not 3D placement
   or task success. Removing the execution height gate alone is insufficient:
   centroid/root identity and coordinate frame must also agree.
5. A true 3D relation requires trustworthy predicted height/depth and compatible
   object geometry, unavailable from the retained XY plane projections. Do not
   assign the fixed plane's artificial z as recovered object height. A human
   video audit requires a separately specified blind annotation procedure;
   the prior qualified visual audit is not a per-frame accuracy ground truth.
6. Preserve cohort IDs and abstentions, report coverage by execution outcome,
   compare baselines on the same denominator, and cluster uncertainty by whole
   matched-seed blocks. For this V1 corpus, each of 20 blocks contains four
   episodes (two wordings × LEFT/RIGHT); resample blocks within the 6100 and
   7200 tiers, retaining every constituent episode and chunk together.
   Disclose every post-result analysis separately from the frozen original.

## Normalized adapter contract

`analysis/evidence_audit.py::validate_pair` is a small fail-closed validation
adapter, not a trajectory extractor and not an independent alignment oracle.
Its record has an `id`, `prediction`, `execution`, and `provenance`. Both sides
must use the same nonnegative integer `time_index` (a common physical sample ID;
booleans and floats are rejected even when numerically equal to an integer),
`coordinate_frame`, and `geometry` = `visual_centroid_xy`, plus finite two-value
`cube` and `bowl` arrays whenever reliable. An unreliable forecast may omit
either coordinate or use null; any coordinates supplied must still be finite
numeric XY. No coordinates are fabricated to retain an abstention. Both sides require typed boolean `reliable`; omitted,
null, numeric, or string values are rejected. Prediction `reliable=false`
remains an abstention. Execution `reliable=false` produces an explicit
execution-unknown/ineligible error: this adapter accepts only pairs with
resolvable execution truth. The caller must preserve those rejected records in
the missingness ledger, never convert them to negative execution or silently
remove them from the recovery census.

Provenance requires `prediction`, `execution`, and `alignment` roles, each with
`path` and SHA256, verified against actual files under the caller's provenance
root. Hash verification confirms byte identity only. A reviewer must still
validate that the recorded coordinate system, object definition and physical
sample mapping are justified by those files. Passing this validation stub is necessary,
not sufficient, for a defensible corrected analysis; it is not a corrected
scorer. No recovered records have
been accepted or scored in this audit.
