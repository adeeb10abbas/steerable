# RoboTwin WAM v3 Phase-A adapter

This adapter covers only the prospective RoboTwin Phase-A queue:

- `efficient_wam_rt_robotwin`, `fastwam_robotwin`, and `lingbot_va_robotwin`
- `pair03` through `pair09`
- sampling replicates `r1` through `r9`
- one matched LEFT/RIGHT pair per invocation

Replicate `r0` is immutable v2 evidence. The loader rejects it before it can
construct a command. The adapter does not provide an all-pairs mode, a resume
flag, wording sweeps, confound fixtures, or any way to change the frozen
prompts, seeds, action cap, guidance, horizons, diffusion settings, or future
retention.

## Why a frame-release artifact is mandatory

The exact v2 runners record object/reference positions in SAPIEN world axes and
define native RoboTwin LEFT with a negative world-X delta. The shared v3 raw
schema defines its continuous lateral metric in robot-base axes, where positive
Y is robot LEFT. Those are not interchangeable labels.

Every live runtime manifest must therefore contain a model-blind calibration
record made before behavioral inference. It binds this planar transform:

```text
robot_base_x =  sapien_world_y + tx
robot_base_y = -sapien_world_x + ty
robot_base_z =  sapien_world_z + tz
```

The transform, its calibration fixture, checkpoint hash manifest, environment
lock, exact source commits, clean diff hashes, adapter source hashes, renderer,
and all five release-gate artifacts are covered by one canonical
`runtime_identity_sha256`. Missing frame provenance or any failed release gate
is an infrastructure failure, never a behavioral zero.

The runtime manifest schema is
`vla-wam-shared-v3-robotwin-runtime-identity-v1`. Its required top-level shape
is:

```json
{
  "schema_version": "vla-wam-shared-v3-robotwin-runtime-identity-v1",
  "study_id": "vla_wam_language_steerability_v3",
  "model_id": "fastwam_robotwin",
  "status": "passed_all_registered_release_gates",
  "runtime_id": "ali-fastwam-v3-runtime-2026-08-05",
  "phase_a_queue_sha256": "<frozen queue sha256>",
  "adapter_contract_sha256": "<adapter contract sha256>",
  "external_repository": {
    "path": "/data/users/ali/vla_wam/external/FastWAM",
    "commit": "068d3fd70c89df3726c09893f47b75a624b20c02",
    "diff_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  },
  "simulator_repository": {
    "path": "/absolute/pinned/robotwin/path",
    "commit": "<full 40-character commit>",
    "diff_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  },
  "checkpoint": {
    "id": "yuanty/fastwam/robotwin_uncond_3cam_384.pt",
    "revision": "<exact immutable revision>",
    "sha256": "<checkpoint or canonical checkpoint-set sha256>",
    "hash_gate_passed": true,
    "hash_manifest_artifact": {"path": "...", "sha256": "...", "bytes": 1}
  },
  "environment": {
    "lock_artifact": {"path": "...", "sha256": "...", "bytes": 1}
  },
  "simulator_version": "<exact version>",
  "renderer_backend": "<exact Vulkan/SAPIEN backend>",
  "adapter_files": {
    "wrapper": {"path": "<exact wrapper>", "sha256": "<frozen hash>"},
    "runner": {"path": "<exact runner>", "sha256": "<frozen hash>"}
  },
  "release_gates": {
    "model_blind_fixture_validation": {"status": "passed", "artifact": {"path": "...", "sha256": "...", "bytes": 1}},
    "exact_runtime_identity": {"status": "passed", "artifact": {"path": "...", "sha256": "...", "bytes": 1}},
    "raw_video_action_jsonl_write": {"status": "passed", "artifact": {"path": "...", "sha256": "...", "bytes": 1}},
    "fixed_observation_exact_repeat": {"status": "passed", "artifact": {"path": "...", "sha256": "...", "bytes": 1}},
    "fixed_observation_left_right_prompt_sensitivity": {"status": "passed", "artifact": {"path": "...", "sha256": "...", "bytes": 1}}
  },
  "measurement_transform": {
    "schema_version": "vla-wam-shared-v3-robotwin-frame-transform-v1",
    "source_frame_id": "sapien_world_xyz_m",
    "target_frame_id": "robot_base_object_minus_reference_xyz_m",
    "status": "passed_model_blind_before_behavior",
    "recorded_before_any_v3_behavioral_inference": true,
    "model_requests_during_validation": 0,
    "models_loaded_during_validation": 0,
    "rotation_source_to_target": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
    "translation_source_to_target_m": [0, 0, 0],
    "fixture_validation_artifact": {"path": "...", "sha256": "...", "bytes": 1},
    "transform_sha256": "<canonical hash of this object excluding transform_sha256>"
  },
  "runtime_identity_sha256": "<canonical hash of the full manifest excluding this field>"
}
```

## Dry run

The dry run validates the frozen queue, source files, runtime identity, release
artifacts, and transform. It prints the exact guarded command without loading a
model or simulator:

```bash
python3 -m experiments.v3.robotwin_wams.launcher dry-run \
  --model-id fastwam_robotwin --pair 3 --replicate 1 --gpu-index 0 \
  --runtime-manifest /data/users/ali/vla_wam/setup/fastwam_v3_runtime.json \
  --external-repository /data/users/ali/vla_wam/external/FastWAM \
  --simulator-repository /data/users/ali/vla_wam/external/FastWAM/third_party/RoboTwin \
  --attempt-dir /data/users/ali/vla_wam/raw/v3/fastwam/pair03/r01/attempt01 \
  --attempt-id fastwam-pair03-r01-attempt01
```

Replace `dry-run` with `execute` only after the protocol validator and live
cluster/PVC/GPU/renderer/egress gates pass. `execute` refuses an existing
attempt directory and always uses the process-group thermal guard. A repair is
a new immutable attempt directory for the same registered cell after manual
review; the adapter never resumes or overwrites a partial pair.

## Raw outputs and accounting

Each attempt retains the untouched native result, trajectory, action trace,
simulator video, and any exposed decoded/latent future. The compiler writes:

- `behavioral_episodes.jsonl` for complete behavioral cells;
- `infrastructure_attempts.jsonl` for missing, malformed, or partial cells;
- a post-close integrity manifest beside each JSONL;
- `attempt_manifest.json` binding the queue and runtime identity.

Every behavioral row preserves the frozen v2 failure stage and adds the v3
taxonomy plus action-indexed state, signed lateral offset, region-entry timing,
pickup timing, path length, action count, censoring, and contact-unavailable
provenance. A complete cell can remain behavioral evidence if its mate fails,
but the attempt manifest makes the incomplete matched pair explicit. If the
two retained initial states differ, both rows are invalidated from the matched
denominator.
