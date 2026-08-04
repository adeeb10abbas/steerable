# Cosmos3 Nano Policy DROID — V2-A011

This adapter is a separate six-cell DROID gate for
`nvidia/Cosmos3-Nano-Policy-DROID` at revision
`6706d7680581c255ff61e0f3bb49d90eac55c79e`. It reuses the completed Edge
integration's matched tasks, requested-relation scoring, video requirements,
action tracing, and generated-future retention. It never modifies or pools the
completed Cosmos3 Edge evidence.

Pinned software:

- Cosmos Framework: `411d25b2e35bc441126f48c44a4b93e1c0564274`
- RoboLab: `0aef241fb088ca21bb4ebd24448940ed56620d17`

## Asset gate

Resume the exact public checkpoint download on the ali-owned PVC:

```bash
/data/users/ali/vla_wam/envs/hf-tools/bin/hf download \
  nvidia/Cosmos3-Nano-Policy-DROID \
  --revision 6706d7680581c255ff61e0f3bb49d90eac55c79e \
  --local-dir /data/users/ali/vla_wam/checkpoints/cosmos3_nano_policy_droid
```

After it finishes, hash and finalize the registry without loading the model:

```bash
python3 tools/finalize_cosmos3_nano_registry.py \
  --registry artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_v2a011_registry.json \
  --checkpoint /data/users/ali/vla_wam/checkpoints/cosmos3_nano_policy_droid \
  --output artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_v2a011_registry.json
```

The finalizer rejects a wrong revision, missing indexed weights, nonempty
partial files, or an implausibly small payload. Do not start the server before
`checkpoint.hash_gate_passed` is true.

## Server and fixed-observation gate

Run the official pinned RoboLab policy server through the V2-A011 seed wrapper:

```bash
CUDA_VISIBLE_DEVICES=1 \
/data/users/ali/vla_wam/external/cosmos-framework/.venv/bin/python \
  experiments/cosmos/serve_nano_robolab_v2a011.py \
  --checkpoint-path /data/users/ali/vla_wam/checkpoints/cosmos3_nano_policy_droid \
  --hf-revision 6706d7680581c255ff61e0f3bb49d90eac55c79e \
  --host 0.0.0.0 --port 18011 --domain-name droid_lerobot \
  --decode-video --action-chunk-size 32 --action-dim 8 \
  --action-space joint_pos --history-length 1 --use-state \
  --conditioning-fps 15 --resolution 480 \
  --guidance 3 --num-steps 4 --shift 5
```

Use the same frozen conditioning image and state plan as the completed Edge
diagnostic, but write a new Nano-only output directory:

```bash
python experiments/cosmos/run_nano_fixed_observation_gate.py \
  --registry artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_v2a011_registry.json \
  --source-plan "$NANO_FIXED_SOURCE_PLAN" \
  --conditioning-image "$NANO_FIXED_CONDITIONING_PNG" \
  --output-dir /data/users/ali/vla_wam/raw/cosmos3_nano_droid/v2_a011/fixed_observation \
  --host 127.0.0.1 --port 18011
```

Behavior is released only if LEFT repeat actions and futures are bit-identical,
LEFT/RIGHT actions and futures differ, and all three responses echo seed 8300.

## Behavioral pairs

Run one seed pair per guarded invocation. Example for seed 8300:

```bash
python3 tools/native_process_group_thermal_guard.py --launch \
  --gpu-index 0 \
  --output /data/users/ali/vla_wam/raw/cosmos3_nano_droid/v2_a011/thermal/seed8300.jsonl \
  --ledger-output /data/users/ali/vla_wam/raw/cosmos3_nano_droid/v2_a011/runtime_interventions.json \
  --invalid-attempts-output /data/users/ali/vla_wam/raw/cosmos3_nano_droid/v2_a011/invalid_attempts.json \
  --model-id cosmos3_nano_policy_droid --pair-id droid_pair_seed_8300 \
  --environment-seed 8300 --sampling-seed 8300 \
  --requested-relation left --requested-relation right -- \
  /data/users/ali/vla_wam/external/RoboLab-11142d4/.venv/bin/python \
  experiments/cosmos/v2_nano_robolab_gate.py \
  --study-root /data/users/ali/vla_wam/src/steerable \
  --environment-seed 8300 --sampling-seed-base 8300 \
  --action-trace-dir /data/users/ali/vla_wam/raw/cosmos3_nano_droid/v2_a011/seed8300/actions \
  --future-trace-dir /data/users/ali/vla_wam/raw/cosmos3_nano_droid/v2_a011/seed8300/futures \
  --remote-host <B200_POD_IP> --remote-port 18011 \
  --task RubiksCubeLeftOfBowlMatchedTask RubiksCubeRightOfBowlMatchedTask \
  --num-envs 1 --num-runs 1 --headless --device cuda:0 \
  --video-mode viewport --disable-subtask \
  --instruction-controller static --instruction-type default \
  --open-loop-horizon 32 --environment-seed 8300 \
  --output-folder-name v2_cosmos_nano_seed8300_neutral
```

Repeat only for seeds 8301 and 8302 with matching environment/sampling seeds
and output paths. Every valid episode must retain viewport MP4, HDF5, exact
executed actions, every returned 32x8 chunk, and every exposed 33-frame future.
Resolve `<B200_POD_IP>` from the named ali-owned server pod immediately before
launch and verify port 18011 from each named RTX simulator pod; never assume or
reuse a stale address. The fixed-observation probe runs in the server pod and
therefore uses loopback.

Compile all six cells fail-closed:

```bash
python3 tools/compile_vla_wam_v2_cosmos_nano.py \
  --seeds 8300 8301 8302 \
  --robolab-output "$ROBOLAB_OUTPUT_ROOT" \
  --raw-root /data/users/ali/vla_wam/raw/cosmos3_nano_droid/v2_a011 \
  --trajectory-dir /data/users/ali/vla_wam/raw/cosmos3_nano_droid/v2_a011/trajectories \
  --registry artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_v2a011_registry.json \
  --fixed-observation-gate /data/users/ali/vla_wam/raw/cosmos3_nano_droid/v2_a011/fixed_observation/manifest.json \
  --invalid-attempts /data/users/ali/vla_wam/raw/cosmos3_nano_droid/v2_a011/invalid_attempts.json \
  --output artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_direct_gate.json \
  --compiled-at-git-head "$(git rev-parse HEAD)"
```

Infrastructure-invalid and partial attempts remain outside all behavioral
denominators. Missing generated futures are technical failures, never zeros.
