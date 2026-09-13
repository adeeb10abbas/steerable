# Development physical-alignment job

`development_physical_alignment_jobs.py` is the detached CPU bridge between
completed development evidence and the annotation-media inventory. It derives
physical forecast targets by calling the reviewed
`freeze_development_release.write_bundle(..., confirmation_release=False)`.
It cannot freeze confirmation.

## Exact authority boundary

The job consumes these immutable prerequisites:

- passed compiler job `development-evidence-compiler-formal-004`, pinned by its
  published receipt hash, inner compiler receipt, 71-file raw bundle, 70-file
  semantic inventory, and N3/D1 freeze fragments;
- passed N3 timing job `timing-n3-development-sidecar-001` and passed
  missingness-aware D1 timing job `timing-d1-development-sidecar-002`, including
  their raw PVC sidecars;
- a later terminal passed camera-crop witness attempt, never diagnostic attempts
  001 or 002, plus its byte-identical published N3 and D1 signed crop contracts;
- the staged authoritative ablation spec, planned-cell CSV, freeze validator,
  timing validator, camera replay validator, compiler, and queue code.

The workstation builder requires a clean results-branch worktree at the exact
supplied results commit and exact `results/jobs/<job>/publish/...` paths. It
does not discover a newer receipt. The camera terminal receipt must use
`outputs.published_camera_crop_contracts`, set
`safe_for_physical_alignment_input: true`, preserve exact zero-science counts,
and qualify original camera pixels. Its signed per-model declarations must bind
the exact published file SHA-256/bytes, contract schema, model ID, crop ID, and
contract payload SHA-256; the files are reopened and matched to those
declarations again on the worker. Extra diagnostic or authority fields fail
closed. Until the exact terminal post-diagnostic success schema and receipt
exist, a real wave is intentionally impossible to emit; attempt 002 can never
be substituted.

After writing the four-file alignment bundle, the wrapper independently calls
`freeze_development_release.derive_bundle(..., require_annotation=False)` on
the immutable evidence manifest and requires byte-for-byte equality. It also
checks the complete per-model mapping schemas: all N3 frame/action targets
1..32, both D1 targets (1,3) and (2,6), every row's exact eligible/full-prefix
counts and finite residual bounds, model request semantics, native-clock
boundary, measured clock extrema/tolerance, camera contract, and all cell,
completion, journal, request, timing, and crop lineages. The mapping and
alignment IDs are literal frozen IDs, not mutually self-signed inputs.

## Build a descriptor after the camera terminal gate exists

```bash
python workshops/corl2026_world_models/experiments/forecast_layout/development_physical_alignment_jobs.py \
  build-wave \
  --study-commit "$STUDY_COMMIT" \
  --results-root "$RESULTS_WORKTREE" \
  --results-commit "$RESULTS_COMMIT" \
  --compiler-job-receipt "$RESULTS_WORKTREE/results/jobs/development-evidence-compiler-formal-004/publish/development_evidence_compiler_job_receipt.json" \
  --compiler-job-receipt-sha256 00329a3c991ee6eb0c68996ecaca1aed59ea3017329272cd82b4b5bfb5cee434 \
  --n3-timing-job-receipt "$RESULTS_WORKTREE/results/jobs/timing-n3-development-sidecar-001/publish/timing_job_receipt.json" \
  --n3-timing-job-receipt-sha256 7c4aa556d37f170308ef5529489936e3fd7fbce819909d8e1d5937c16965f9d9 \
  --d1-timing-job-receipt "$RESULTS_WORKTREE/results/jobs/timing-d1-development-sidecar-002/publish/timing_job_receipt.json" \
  --d1-timing-job-receipt-sha256 75490aab357d8290afa10f0e1a85437794a37c81f6a9a0524c94c746198da68d \
  --camera-job-receipt "$RESULTS_WORKTREE/results/jobs/$CAMERA_JOB_ID/publish/camera_crop_witness_job_receipt.json" \
  --camera-job-receipt-sha256 "$CAMERA_JOB_RECEIPT_SHA256" \
  --n3-camera-crop-contract "$RESULTS_WORKTREE/results/jobs/$CAMERA_JOB_ID/publish/n3_camera_crop_contract.json" \
  --n3-camera-crop-contract-sha256 "$N3_CROP_FILE_SHA256" \
  --d1-camera-crop-contract "$RESULTS_WORKTREE/results/jobs/$CAMERA_JOB_ID/publish/d1_camera_crop_contract.json" \
  --d1-camera-crop-contract-sha256 "$D1_CROP_FILE_SHA256" \
  --output /tmp/development-physical-alignment-wave.json
```

This command emits a descriptor only. Queue release remains a separate,
explicit operator step.

## Cluster outputs

Canonical raw evidence remains below
`control/jobs/development-physical-alignment-001/raw/`:

- `development_physical_alignment_input.json`;
- `development_release_evidence.json`;
- `physical_alignment_bundle/` containing exactly two mappings and two
  alignment contracts;
- `physical_alignment_evidence_manifest.json`, a signed lineage inventory.

Success is one atomic five-file publication:

- `n3_physical_alignment_receipt.json`;
- `n3_alignment_contract.json`;
- `d1_physical_alignment_receipt.json`;
- `d1_alignment_contract.json`;
- `development_physical_alignment_job_receipt.json`.

The four derived artifacts are byte-identical to the canonical raw bundle.
The transaction enforces the result publisher's 16 MiB per-file and 64 MiB
per-job limits before the publish directory becomes visible.
Any pre-publication failure leaves no partial success files and publishes only
`development_physical_alignment_job_failure.json`; partial staging is retained
under raw evidence for diagnosis.

Every receipt reports zero model loads, servers, requests, simulator starts,
resets, episodes, actions, behavioral cells, and labels. Alignment success only
authorizes those four files as inputs to the development annotation inventory.
It does not authorize rater distribution, resource freeze, annotation results,
confirmation dispatch, or a behavioral/prediction claim.
