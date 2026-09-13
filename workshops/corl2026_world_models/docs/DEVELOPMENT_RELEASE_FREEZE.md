# Development evidence and confirmation release freeze

Status: tooling implemented; no confirmation release is asserted by this
document.

`analysis/freeze_development_release.py` is the fail-closed bridge between the
D01--D04 recordings and confirmation admission. It emits a confirmation
release only after it authenticates the complete development cohort for every
qualified model. Software tests and recording pilots do not satisfy this gate.

## Physical alignment boundary

The recorder already retains the native control counter, physics step/time,
original-camera frame identity and original-camera capture timestamp for every
observation. The model request receipts retain the jointly generated future and
actions. Those facts alone do **not** define the physical target time of a
generated frame.

Each qualified model therefore needs a signed
`wmf-native-generated-target-timing-v1` receipt with:

- `status: qualified_from_native_runtime_metadata`;
- `time_source_kind: native_runtime_exposed_target_offsets`;
- the exact runtime field that exposes each target's elapsed physical time;
- every development request-receipt SHA-256 in canonical cell/request order;
- sorted `{generated_frame_index, target_physical_time_s,
  native_runtime_field}` rows;
- all three prohibitions set to `false`:
  `presentation_video_fps_used`,
  `conditioning_fps_used_as_target_timing`, and
  `generated_frame_index_interpreted_as_action_index`; and
- `clock_bridge: elapsed physical seconds from request current
  original-camera capture`.

The named field must exist in every cited official request receipt and contain
the same explicit ordered objects
`{generated_frame_index, target_physical_time_s}`. Merely naming a plausible
runtime field in the timing receipt is insufficient; the tool traverses and
compares the actual request JSON values.

The tool pairs each exposed target with the nearest recorded original-camera
capture after the request start, checks the native physics elapsed time as an
independent clock, and requires one consistent executed-action offset across
all full-prefix development requests. It takes the conservative global
tolerance

`min(half minimum positive native control interval,
half minimum positive original-camera capture interval)`.

It chooses the longest strictly positive qualified target inside the unchanged
executed prefix as primary H. A9, when supported, is the earliest strictly
positive qualified target below H. A generated index is never treated as an
action index.

## Current pinned-source audit: no timing release yet

The source/runtime inspection on 2026-09-13 found useful model-time structure,
but no authoritative decoded-frame-to-physical-target-time field. Therefore
the current evidence is **NO-GO** for alignment and confirmation:

- N3 commit `411d25b2e35bc441126f48c44a4b93e1c0564274`,
  `cosmos_framework/scripts/action_policy_server_robolab.py`, constructs
  `action_chunk_size + 1` vision slots, passes `conditioning_fps`, returns the
  action tensor, and optionally decodes `samples["vision"]`. The associated
  `cosmos_framework/inference/action.py` sequence plan and
  `cosmos_framework/data/vfm/action/datasets/droid_lerobot_dataset.py` training
  row timestamps show temporal intent, but none exposes target elapsed seconds
  for each generated output frame. The current workshop N3 request receipt in
  `experiments/forecast_layout/n3_behavioral_pilot_job.py` retains decoded
  shape, hashes and request timing, not such a target list.
- D1 commit `ab790c198fbce33503358efbbd4187ce9a89adf3`,
  `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`, exposes
  `num_frame_per_block`, `num_frames`, `action_horizon`, and returns
  `action_pred`/`video_pred`. The exact checkpoint configuration has
  `action_horizon: 24`, `num_frame_per_block: 2`, and
  `num_action_per_block: 24`; the DROID data configuration uses timestamped
  delta-index rows. Those are block/training semantics, not a decoded-frame
  physical-time map. `socket_test_optimized_AR.py` writes presentation videos
  at 5 FPS. The workshop `d1_instrumented_server.py` retains the exact
  block/action configuration, action/latent identities and offline decode, but
  no per-frame physical target offsets.
- RoboLab commit `0aef241fb088ca21bb4ebd24448940ed56620d17`,
  `robolab/registrations/droid/auto_env_registrations_jointpos.py`, configures
  `dt = 1/120`, `decimation = 8`, and `render_interval = 8`. These describe the
  nominal simulator/control/render clocks. `robolab/eval/episode.py` derives a
  saved-video FPS from them, which is presentation metadata and cannot supply
  the missing model-output target semantics.

Future requests can retain the missing evidence without changing inference:
the server may add a measurement-only, signed target-offset field sourced from
a separately qualified and pinned native temporal contract, while the
controller records the request's current original-camera identity/timestamp,
control step and physics time and later joins the executed-prefix captures.
The action and latent tensors remain untouched. Merely computing that field as
`frame_index / conditioning_fps`, pairing frames and actions by ordinal, or
using a decoded/presentation-video FPS does not qualify and is rejected by the
tool.

## Evidence manifest

The input is one `wmf-development-release-evidence-v1` JSON object:

```json
{
  "schema_version": "wmf-development-release-evidence-v1",
  "study_id": "WMF-ABLATION-001",
  "cohort_branch": "full_two_model",
  "qualified_model_ids": ["N3", "D1"],
  "ablation_spec": {"path": ".../ablation_spec.json", "sha256": "...", "bytes": 1},
  "planned_cells": {"path": ".../planned_cells.csv", "sha256": "...", "bytes": 1},
  "model_evidence": [
    {
      "model_id": "N3",
      "camera_crop_contract": {"path": "...", "sha256": "...", "bytes": 1},
      "generated_target_timing_receipt": {"path": "...", "sha256": "...", "bytes": 1},
      "development_cells": [
        {
          "cell_receipt": {"path": "...", "sha256": "...", "bytes": 1},
          "server_request_receipts": [
            {"path": "...", "sha256": "...", "bytes": 1}
          ],
          "resource_receipt": {"path": "...", "sha256": "...", "bytes": 1}
        }
      ]
    }
  ],
  "annotation": null,
  "resource_budget_policy": null
}
```

Use `reduced_n3`/`["N3"]` or `reduced_d1`/`["D1"]` only after that reduced
branch is scientifically qualified and disclosed. Each listed model must have
exactly the 16 selected development cells in `planned_cells.csv`. Every cell
must expose every official request receipt: 15 for N3 or 57 for D1.

The signed `wmf-camera-crop-contract-v1` must identify the original camera,
crop ID/operation, output width/height, and attest
`simulator_state_render_used: false`. The signed per-cell
`wmf-development-resource-measurement-v1` records complete episode wall time,
inference wall time, raw bytes, measured peak allocated/reserved GPU bytes, and
GPU count. Placeholder or projected development measurements do not qualify.

## Two-stage use

Derive alignment as soon as the complete native development evidence exists:

```bash
python workshops/corl2026_world_models/analysis/freeze_development_release.py \
  derive-alignment \
  --evidence /data/.../development_release_evidence.json \
  --output-dir /data/.../development_alignment_v1
```

This writes one signed physical-mapping receipt and one compact alignment
contract per model. The compact contract has the exact fields consumed by
`forecast_annotation_workflow.py`. Its contract hash can be frozen into the
development annotation workflow before labels begin.

After two locked independent first-pass responses, any required independent
adjudication, the development q95 summary and an explicit scientific usability
decision exist, replace `annotation: null` with hash-bound references to:

- the frozen rubric;
- the signed development duplicate-label summary;
- the mechanically merged final development consensus; and
- a decision containing `measurement_usable: true`, reviewer identity/time,
  basis and the exact development-summary file SHA-256.

Also provide `resource_budget_policy` with the measured headroom multiplier,
selected GM host/pool, per-model maximum parallel block counts, authorized
memory per GPU, exactly 96 confirmation cells per qualified model, and the
frozen confirmation annotation-judgment ceiling. Then run:

```bash
python workshops/corl2026_world_models/analysis/freeze_development_release.py \
  freeze-confirmation \
  --evidence /data/.../development_release_evidence.json \
  --output-dir /data/.../confirmation_release_v1
```

The output directory is created atomically and is never overwritten. If any
cell, request, clock, mapping, label, consensus, decision or measured resource
input is missing or inconsistent, the command exits 2 and creates no output
directory.

Every confirmation queue/server/cell entry point validates the final artifact
by exact file hash and model:

```python
validate_release_freeze(
    freeze_path,
    expected_sha256,
    expected_model="N3",
)
```

The release receipt means only that the frozen confirmation cohort is eligible
to start. It is not a behavioral episode, a model result, a label, or evidence
of forecast accuracy.
