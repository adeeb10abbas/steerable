# Local VLA-WAM steerability runbook

This runbook reproduces the prospective study frozen in
`artifacts/vla_wam_shared_v1/preregistration.json`. It is host-specific by
design: the checkpoints, four repositories, and two physical RTX 3090 roles
are pinned in the evidence package.

## Safety and validity invariants

- Physical GPU 1 serves the policy. Physical GPU 0 runs Isaac Sim.
- Do not combine output from a different physical-GPU assignment. The renderer
  changed policy inputs by 0.194/255 MAE in the seed-6100 audit even though the
  simulator state was byte-identical.
- Stop a batch if either card reaches 90 C. Preserve interrupted output as an
  exclusion; restart the whole condition from seed 0 rather than mixing roles.
- Run one Isaac environment at a time.
- Static confirmation seeds are 6100--6109. Hierarchy seeds are 7100--7104.
- Every request uses `episode_seed * 1000 + replan_index`.
- Exact reset state does not imply identical first-observed object centroids or
  realtime-rendered pixels. Preserve every recorded conditioning frame and
  first centroid; treat closed-loop first-action distances as prompt plus
  settling/renderer sensitivity, and use the hash-pinned probe for an exact
  prompt intervention.
- Never reuse the excluded 51xx calibration episodes in a confirmation rate.
- `--output-folder-name` resumes existing runs. Begin a definitive condition
  only with an absent/empty registered output directory.

Monitor both cards while a batch is live:

```bash
watch -n 2 'nvidia-smi --query-gpu=index,temperature.gpu,fan.speed,utilization.gpu,memory.used,clocks_throttle_reasons.sw_thermal_slowdown,power.draw --format=csv,noheader'
```

## Repositories and checkpoints

```text
/home/ali/projects/steerable
/home/ali/projects/RoboLab
/home/ali/cosmos-framework
/home/ali/openpi-robolab

/home/ali/cosmos-framework/.cache/huggingface/hub/models--nvidia--Cosmos3-Edge-Policy-DROID/snapshots/3ea407af3e156c0af3b4bb6edd85842cc9a58777
/home/ali/.cache/openpi/openpi-assets-simeval/pi05_droid_jointpos
/home/ali/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
```

The checkpoint byte counts and aggregate SHA-256 values are in
`artifacts/vla_wam_shared_v1/checkpoint_provenance.json`.

## Shared Isaac container wrapper

All RoboLab commands below run from `/home/ali/projects/RoboLab` in the same
image and mounts:

```bash
docker run --rm --runtime nvidia --gpus device=0 --net host \
  -v /home/ali/projects/RoboLab/.cache/ov:/root/.cache/ov \
  -v /home/ali/projects/RoboLab/.cache/kit:/isaac-sim/kit/cache \
  -v /home/ali/projects/RoboLab:/workspace/robolab \
  robolab:codex-steerability -lc \
  '<COMMAND> --kit_args=--/rtx/verifyDriverVersion/enabled=false'
```

The version-check override is narrow. Host `nvidia-smi` reports driver
535.309.01; Isaac's Vulkan parser incorrectly renders its three-digit minor as
535.53.01. The excluded startup with the check enabled made zero requests.

## Cosmos3 Edge DROID

Start the decode-video server on physical GPU 1:

```bash
cd /home/ali/cosmos-framework
CUDA_VISIBLE_DEVICES=1 DS_IGNORE_CUDA_DETECTION=1 .venv/bin/python \
  -m cosmos_framework.scripts.action_policy_server_robolab \
  --checkpoint-path /home/ali/cosmos-framework/.cache/huggingface/hub/models--nvidia--Cosmos3-Edge-Policy-DROID/snapshots/3ea407af3e156c0af3b4bb6edd85842cc9a58777 \
  --port 8000 --seed 0 --decode-video --format-prompt-as-json True \
  --action-space joint_pos --action-dim 8 --action-chunk-size 32
```

Run the two static conditions through the shared container wrapper:

```bash
/workspace/isaaclab/_isaac_sim/python.sh policies/cosmos3/run.py \
  --task RubiksCubeLeftOfBowlMatchedTask RubiksCubeRightOfBowlMatchedTask \
  --num-envs 1 --num-runs 10 --headless --device cuda:0 --video-mode none \
  --output-folder-name v1_cosmos_canonical --instruction-type default \
  --sampling-seed-base 6100 --record-predictions

/workspace/isaaclab/_isaac_sim/python.sh policies/cosmos3/run.py \
  --task RubiksCubeLeftOfBowlMatchedTask RubiksCubeRightOfBowlMatchedTask \
  --num-envs 1 --num-runs 10 --headless --device cuda:0 --video-mode none \
  --output-folder-name v1_cosmos_vague --instruction-type vague \
  --sampling-seed-base 6100 --record-predictions
```

With Isaac stopped, run the frozen fixed-observation probe from the steerable
repository using the Cosmos environment:

```bash
cd /home/ali/projects/steerable
/home/ali/cosmos-framework/.venv/bin/python \
  tools/run_fixed_observation_command_probe.py \
  --plan artifacts/vla_wam_shared_v1/command_probe_plan.json \
  --model cosmos \
  --output-dir artifacts/vla_wam_shared_v1/command_probe/cosmos_gpu1

.venv/bin/python tools/compare_command_probe_hardware.py \
  --gpu0-probe artifacts/vla_wam_shared_v1/command_probe/cosmos \
  --gpu1-probe artifacts/vla_wam_shared_v1/command_probe/cosmos_gpu1 \
  --output artifacts/vla_wam_shared_v1/cosmos_gpu_assignment_audit.json
```

For the five-step hierarchy comparison, restart the same server without
`--decode-video`; these episodes do not consume generated futures. First run
the excluded seed-5190 one-episode oracle pilot. Then run:

```bash
/workspace/isaaclab/_isaac_sim/python.sh policies/cosmos3/run.py \
  --task RubiksCubeLeftOfBowlMatchedTask RubiksCubeRightOfBowlMatchedTask \
  --num-envs 1 --num-runs 5 --headless --device cuda:0 --video-mode none \
  --open-loop-horizon 5 --output-folder-name v1_cosmos_h5_static \
  --instruction-type default --instruction-controller static \
  --sampling-seed-base 7100

/workspace/isaaclab/_isaac_sim/python.sh policies/cosmos3/run.py \
  --task RubiksCubeLeftOfBowlMatchedTask RubiksCubeRightOfBowlMatchedTask \
  --num-envs 1 --num-runs 5 --headless --device cuda:0 --video-mode none \
  --open-loop-horizon 5 --output-folder-name v1_cosmos_h5_oracle \
  --instruction-type default --instruction-controller predicate_oracle \
  --sampling-seed-base 7100
```

## pi0.5 DROID

Start the policy server on physical GPU 1:

```bash
cd /home/ali/openpi-robolab
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/serve_policy.py \
  policy:checkpoint --policy.config=pi05_droid_jointpos \
  --policy.dir=/home/ali/.cache/openpi/openpi-assets-simeval/pi05_droid_jointpos \
  --port=8000
```

Run the fixed-observation probe with the same frozen plan:

```bash
cd /home/ali/projects/steerable
/home/ali/openpi-robolab/.venv/bin/python \
  tools/run_fixed_observation_command_probe.py \
  --plan artifacts/vla_wam_shared_v1/command_probe_plan.json \
  --model pi05 \
  --output-dir artifacts/vla_wam_shared_v1/command_probe/pi05
```

During a valid closed-loop policy request, capture one steady-state
`nvidia-smi` point measurement (temperature, memory, utilization, power, and
thermal-throttling flags for both physical cards), the server and simulator
PIDs/RSS, and the exact GPU roles in
`artifacts/vla_wam_shared_v1/operational_snapshot_pi05_confirmation.json`.
This is explicitly a point measurement rather than a peak-memory claim.

Run the static conditions through the shared container wrapper:

```bash
/workspace/isaaclab/_isaac_sim/python.sh policies/pi0_family/run.py \
  --policy pi05 \
  --task RubiksCubeLeftOfBowlMatchedTask RubiksCubeRightOfBowlMatchedTask \
  --num-envs 1 --num-runs 10 --headless --device cuda:0 --video-mode none \
  --open-loop-horizon 15 --output-folder-name v1_pi05_canonical \
  --instruction-type default --sampling-seed-base 6100

/workspace/isaaclab/_isaac_sim/python.sh policies/pi0_family/run.py \
  --policy pi05 \
  --task RubiksCubeLeftOfBowlMatchedTask RubiksCubeRightOfBowlMatchedTask \
  --num-envs 1 --num-runs 10 --headless --device cuda:0 --video-mode none \
  --open-loop-horizon 15 --output-folder-name v1_pi05_vague \
  --instruction-type vague --sampling-seed-base 6100
```

After the excluded seed-5190 one-episode oracle pilot, run:

```bash
/workspace/isaaclab/_isaac_sim/python.sh policies/pi0_family/run.py \
  --policy pi05 \
  --task RubiksCubeLeftOfBowlMatchedTask RubiksCubeRightOfBowlMatchedTask \
  --num-envs 1 --num-runs 5 --headless --device cuda:0 --video-mode none \
  --open-loop-horizon 5 --output-folder-name v1_pi05_h5_static \
  --instruction-type default --instruction-controller static \
  --sampling-seed-base 7100

/workspace/isaaclab/_isaac_sim/python.sh policies/pi0_family/run.py \
  --policy pi05 \
  --task RubiksCubeLeftOfBowlMatchedTask RubiksCubeRightOfBowlMatchedTask \
  --num-envs 1 --num-runs 5 --headless --device cuda:0 --video-mode none \
  --open-loop-horizon 5 --output-folder-name v1_pi05_h5_oracle \
  --instruction-type default --instruction-controller predicate_oracle \
  --sampling-seed-base 7100
```

## Prompt-blind future scoring

Stop both policy and Isaac processes before loading Qwen on physical GPU 1.
The calibration file is immutable. Score the two Cosmos static batches and
the fixed-observation probe:

```bash
cd /home/ali/projects/steerable
CUDA_VISIBLE_DEVICES=1 /home/ali/cosmos-framework/.venv/bin/python \
  tools/score_cosmos_semantic_futures.py score \
  --task-dir \
    /home/ali/projects/RoboLab/output/v1_cosmos_canonical/RubiksCubeLeftOfBowlMatchedTask \
    /home/ali/projects/RoboLab/output/v1_cosmos_canonical/RubiksCubeRightOfBowlMatchedTask \
  --calibration artifacts/vla_wam_shared_v1/semantic_future_calibration.json \
  --output-dir artifacts/vla_wam_shared_v1/semantic_confirmation/cosmos_canonical

CUDA_VISIBLE_DEVICES=1 /home/ali/cosmos-framework/.venv/bin/python \
  tools/score_cosmos_semantic_futures.py score \
  --task-dir \
    /home/ali/projects/RoboLab/output/v1_cosmos_vague/RubiksCubeLeftOfBowlMatchedTask \
    /home/ali/projects/RoboLab/output/v1_cosmos_vague/RubiksCubeRightOfBowlMatchedTask \
  --calibration artifacts/vla_wam_shared_v1/semantic_future_calibration.json \
  --output-dir artifacts/vla_wam_shared_v1/semantic_confirmation/cosmos_vague

CUDA_VISIBLE_DEVICES=1 /home/ali/cosmos-framework/.venv/bin/python \
  tools/score_cosmos_semantic_futures.py score-probe \
  --probe-dir artifacts/vla_wam_shared_v1/command_probe/cosmos_gpu1 \
  --calibration artifacts/vla_wam_shared_v1/semantic_future_calibration.json \
  --output-dir artifacts/vla_wam_shared_v1/command_probe/cosmos_gpu1_semantics
```

## Fail-closed compilation

Install the CPU-side evidence environment and run both compilers:

```bash
cd /home/ali/projects/steerable
uv pip install --python .venv/bin/python \
  -r tools/vla_wam_study_requirements.txt

.venv/bin/python tools/compile_vla_wam_evidence.py \
  --manifest artifacts/vla_wam_shared_v1/run_manifest.json \
  --output-dir artifacts/vla_wam_shared_v1/final_evidence

.venv/bin/python tools/compile_vla_wam_study.py \
  --study-root artifacts/vla_wam_shared_v1 \
  --closed-loop artifacts/vla_wam_shared_v1/final_evidence/closed_loop_summary.json \
  --output-dir artifacts/vla_wam_shared_v1/final_evidence
```

The first compiler must report 120 episodes and zero missing. The second must
verify 80 static episodes, 40 hierarchy episodes, one exact initial-state
fingerprint, 40 Cosmos first-conditioning images and their renderer variation,
16 probe conditions per model, frozen calibration/plan hashes, and a completed
semantic audit before the evidence package is publishable.
