# Forecast evidence analysis

Status: implementation and fail-closed contracts are ready; this document does
not assert that confirmation recordings or human labels exist.

`analysis/forecast_evidence_analysis.py` is the final confirmation analyzer for
`WMF-ABLATION-001`. It emits no forecast statistic until it has revalidated all
of the following:

1. The complete confirmation request-selection manifest, including every
   planned valid-complete, safety-censored, technical-invalid, and not-run cell.
   The annotation workflow reopens and hashes every cited recording receipt and
   action manifest and reconstructs the selected request inventory.
2. The signed development-to-confirmation release freeze. Validation reopens
   every native physical alignment/mapping receipt, fixes the analysis and
   request-sampling seeds, and proves that the complete development evidence,
   annotation decision, and measured resource budget actually released this
   model branch for confirmation.
3. The confirmation annotation freeze. Validation reproduces the development
   duplicate-label summary and final development consensus, verifies the
   empirical 95th-percentile movement threshold, and requires the explicit
   measurement-usability decision.
4. The confirmation restricted identity map and its exact hashes for the
   selection and freeze.
5. The final confirmation consensus. The analyzer reruns the mechanical merge
   from the two locked blind response streams and, when needed, the independent
   adjudicator stream. The reproduced document must be byte-for-byte equal to
   the supplied consensus.
6. Compact signed endpoint and history receipts used only for quantities not
   present in the annotation selection contract. These receipts cite the exact
   native recorder completion and fsync journal by path and SHA-256. Endpoint
   coordinates are reloaded from the hash-checked observation payloads, and
   history intervals are recomputed from the cited native camera clocks.

The exact constants and claim boundary are machine-readable in
`experiments/forecast_layout/forecast_analysis_contract.json`.

## Evidence manifest

The command accepts one signed `wmf-forecast-analysis-evidence-manifest-v1`.
Every source reference is an object with `path` and `sha256`; relative paths are
resolved from the manifest. The manifest must have this shape:

```json
{
  "schema_version": "wmf-forecast-analysis-evidence-manifest-v1",
  "study_id": "WMF-ABLATION-001",
  "stage": "confirmation",
  "cohort_branch": "full_two_model",
  "sources": {
    "ablation_spec": {"path": "...", "sha256": "..."},
    "development_release_freeze": {"path": "...", "sha256": "..."},
    "request_selection": {"path": "...", "sha256": "..."},
    "annotation_freeze": {"path": "...", "sha256": "..."},
    "restricted_map": {"path": "...", "sha256": "..."},
    "final_consensus": {"path": "...", "sha256": "..."}
  },
  "endpoint_trace_receipts": [
    {"path": "...", "sha256": "..."}
  ],
  "request_history_receipts": [
    {"path": "...", "sha256": "..."}
  ],
  "payload_sha256": "canonical unsigned-object SHA-256"
}
```

Use `forecast_evidence_analysis.sign_document` or the same canonical JSON rule
as the annotation workflow to fill `payload_sha256`. The endpoint list must
cover every roster cell whose final status is `valid_complete` or
`valid_censored`, exactly once. The history list must cover every selected
request that cites a real preceding observation, exactly once. Empty lists are
valid only when the corresponding required population is empty.

### Endpoint trace receipt

A `wmf-forecast-endpoint-trace-receipt-v1` binds one recording receipt and
action-manifest hash to robot-frame cube and bowl positions at observation zero,
action 450, and the first-success endpoint. For a complete no-success trace, the
alternative endpoint is exactly action 450. A safety-censored trace must set
`action_450` to null. Its last observed state is never carried forward. A
technical-invalid or unrun cell cannot have an endpoint receipt.

The exact keys are:

```text
schema_version, study_id, stage, cell_id, recording_id, model_id,
recording_status, recording_receipt_sha256, action_manifest_sha256,
executed_action_count, first_success_action_index, action_zero, action_450,
first_success_or_action_450, action_zero_observation_id,
action_450_observation_id, first_success_or_action_450_observation_id,
source_adapter_completion, source_adapter_journal, payload_sha256
```

Each non-null point has exactly `action_index`, `cube_robot_xyz`, and
`bowl_robot_xyz`. Coordinates must be finite three-vectors.

### Request history receipt

A `wmf-forecast-request-history-receipt-v1` binds the selected request and its
alignment receipt to the positive measured interval between the preceding and
current observations. Its exact keys are:

```text
schema_version, study_id, stage, source_request_id, cell_id,
alignment_receipt_id, alignment_receipt_sha256,
camera_id, preceding_observation_id, current_observation_id,
preceding_camera_capture_time_ns, current_camera_capture_time_ns,
preceding_physics_time_s, current_physics_time_s,
preceding_observation_interval_s, source_adapter_completion,
source_adapter_journal, payload_sha256
```

Request zero uses persistence as the constant-velocity baseline and therefore
must not receive a fabricated history receipt.

## Estimands and missingness

For each selected request the analyzer derives normalized image-plane cube,
bowl, and cube-minus-bowl vectors from adjudicated centers. Forecast error is
the Euclidean distance between predicted and executed relative vectors;
persistence error is the distance between current and executed vectors. Skill
is persistence error minus forecast error, so either positive or negative
results pass through unchanged.

Requests are averaged within their one episode/cell, then equally across the
four conditions, then across independent base-layout pairs. A continuous mean
and its 10,000-draw paired-layout bootstrap interval use only layout pairs with
all four condition means observable; every included and excluded layout ID is
reported. Models are always analyzed separately.

The full-design strict-win bounds retain all 96 planned confirmation cells per
model. An unresolved selected request is first treated as a non-win and then a
win. A zero-eligible, technical-invalid, unrun, or otherwise unmeasured episode
contributes `[0,1]`. This changes no object position and preserves equal
episode, condition, and layout weighting.

The output also includes:

- persistence and timestamped constant-velocity errors;
- actual and predicted reflected-minus-original motion contrasts with their
  paired discrepancy and signed per-layout values;
- action-450 versus first-success instruction-response controls;
- moving and stationary strata using the frozen development q95 threshold;
- separate relative, cube, and bowl motion/error components;
- an earlier-horizon result only at the exact frozen qualified target. Primary
  and earlier skill are paired within the same selected request before the
  episode/condition/layout aggregation, and the reported contrast is primary
  skill at H minus earlier skill. Unsupported qualification is kept distinct
  from qualified-but-unobservable human labels, and neither state emits a
  numerical contrast;
- per-condition request-inventory/timing eligibility, zero-eligible episodes,
  censor reasons, localization/ambiguity coverage, exact cell/request audits,
  and the per-model planned/complete/censored/technical-invalid/unrun table;
- the full hash-bound model configuration and frozen physical alignment fields
  in that sample-size table; and
- one example video per model/condition selected by a fixed metadata hash, with
  no label, success, error, or visual outcome in the selection rule.

A reduced `reduced_n3` or `reduced_d1` branch is labeled as a one-model study.
The absent model's 96 planned confirmation cells appear as an unqualified,
unrun branch; no replacement model is introduced.

## Run

```sh
python3 workshops/corl2026_world_models/analysis/forecast_evidence_analysis.py \
  --evidence-manifest /restricted/confirmation_analysis_evidence.json \
  --output /compact/forecast_confirmation_analysis.json
```

The output is itself canonically hash-bound. It explicitly forbids pooled model
claims, cross-model accuracy ranking, causal claims about world-model action
generation, and reinterpretation of technical invalidity as behavioral failure.
