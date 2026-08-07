#!/usr/bin/env bash
# Run complete GR00T V3-C001 seed blocks after the seed-8500 smoke passes.
#
# Every seed receives a fresh zero-action Isaac registration gate before its
# eight behavioral cells.  The queue stops at the first infrastructure error;
# completed behavioral seed blocks and partial infrastructure attempts remain
# on the PVC and are never overwritten.

set -euo pipefail

usage() {
  echo "usage: $0 --study-root PATH --execution-plan PATH --release-manifest PATH \\\n+    --registration-manifest PATH --raw-root PATH --runtime-python PATH \\\n+    --thermal-guard PATH --gpu-index INDEX [--remote-host HOST] \\\n+    [--remote-port PORT] [--seed-start N] [--seed-end N] [--attempt N]" >&2
}

study_root=
execution_plan=
release_manifest=
registration_manifest=
raw_root=
runtime_python=
thermal_guard=
gpu_index=
remote_host=127.0.0.1
remote_port=5555
seed_start=8501
seed_end=8519
attempt=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --study-root) study_root=$2; shift 2 ;;
    --execution-plan) execution_plan=$2; shift 2 ;;
    --release-manifest) release_manifest=$2; shift 2 ;;
    --registration-manifest) registration_manifest=$2; shift 2 ;;
    --raw-root) raw_root=$2; shift 2 ;;
    --runtime-python) runtime_python=$2; shift 2 ;;
    --thermal-guard) thermal_guard=$2; shift 2 ;;
    --gpu-index) gpu_index=$2; shift 2 ;;
    --remote-host) remote_host=$2; shift 2 ;;
    --remote-port) remote_port=$2; shift 2 ;;
    --seed-start) seed_start=$2; shift 2 ;;
    --seed-end) seed_end=$2; shift 2 ;;
    --attempt) attempt=$2; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done

for required in study_root execution_plan release_manifest registration_manifest raw_root runtime_python thermal_guard gpu_index; do
  if [[ -z ${!required} ]]; then
    usage
    exit 2
  fi
done

if (( seed_start < 8501 || seed_end > 8519 || seed_start > seed_end )); then
  echo "seed range must be within 8501..8519" >&2
  exit 2
fi

gate_root="$raw_root/phase_c_v3c001/groot_n17_droid_vla"
behavioral_root="$raw_root/behavioral/v3-c001/groot_n17_droid_vla"
attempt_tag=$(printf '%02d' "$attempt")
mkdir -p "$gate_root" "$behavioral_root"

run_guarded() {
  local attempt_root=$1
  shift
  mkdir -p "$attempt_root/cache/ov" "$attempt_root/cache/isaac" "$attempt_root/cache/kit" "$attempt_root/logs"
  "$runtime_python" "$thermal_guard" \
    --launch \
    --gpu-index "$gpu_index" \
    --output "$attempt_root/thermal_events.jsonl" \
    --poll-seconds 0.5 \
    -- env \
      CUDA_VISIBLE_DEVICES="$gpu_index" \
      OMNI_KIT_ACCEPT_EULA=YES \
      VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
      LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
      PYTHONPATH="$study_root${PYTHONPATH:+:$PYTHONPATH}" \
      OV_CACHE_ROOT="$attempt_root/cache/ov" \
      ISAACSIM_CACHE_PATH="$attempt_root/cache/isaac" \
      KIT_USER_DATA_ROOT="$attempt_root/cache/kit" \
      KIT_LOG_ROOT="$attempt_root/logs" \
      "$@"
}

for seed in $(seq "$seed_start" "$seed_end"); do
  bridge_preflight="$gate_root/behavioral_bridge_preflight_seed${seed}_attempt${attempt_tag}.json"
  task_registration="$gate_root/live_task_registration_seed${seed}_attempt${attempt_tag}.json"
  registration_attempt="$gate_root/task_registration_seed${seed}_attempt${attempt_tag}"
  behavior_attempt="$gate_root/behavioral_seed${seed}_attempt${attempt_tag}"
  runner_output="$behavioral_root/seed${seed}/_runner_attempt${attempt_tag}"
  launch_evidence="$gate_root/whole_seed_seed${seed}_attempt${attempt_tag}.json"

  if [[ ! -s "$bridge_preflight" ]]; then
    PYTHONPATH="$study_root${PYTHONPATH:+:$PYTHONPATH}" /usr/bin/python3 -m \
      experiments.v3.phase_c_four_phrasings.groot_behavioral_bridge \
      --study-root "$study_root" \
      --execution-plan "$execution_plan" \
      --release-manifest "$release_manifest" \
      --registration-manifest "$registration_manifest" \
      --seed "$seed" \
      --output "$bridge_preflight" \
      --preflight-only
  fi

  for fresh_path in "$task_registration" "$registration_attempt" "$behavior_attempt" "$runner_output" "$launch_evidence"; do
    if [[ -e "$fresh_path" ]]; then
      echo "refusing to overwrite retained seed-$seed path: $fresh_path" >&2
      exit 3
    fi
  done

  run_guarded "$registration_attempt" \
    "$runtime_python" "$study_root/experiments/v3/phase_c_four_phrasings/groot_task_registration_preflight.py" \
      --study-root "$study_root" \
      --bridge-preflight "$bridge_preflight" \
      --output "$task_registration" \
      --instruction-controller static \
      --num-envs 1 \
      --num-runs 1 \
      --video-mode viewport \
      --disable-subtask \
      --device cuda:0 \
      --headless \
      --renderer realtime \
      --rendering-type balanced \
      --kit_args=--/rtx/verifyDriverVersion/enabled=false

  [[ -s "$task_registration" ]] || { echo "seed-$seed task registration produced no evidence" >&2; exit 4; }

  run_guarded "$behavior_attempt" \
    "$runtime_python" "$study_root/experiments/v3/phase_c_four_phrasings/groot_live_bridge.py" \
      --study-root "$study_root" \
      --bridge-preflight "$bridge_preflight" \
      --task-registration "$task_registration" \
      --release-manifest "$release_manifest" \
      --registration-manifest "$registration_manifest" \
      --execution-plan "$execution_plan" \
      --runner-output-root "$runner_output" \
      --launch-evidence "$launch_evidence" \
      --remote-host "$remote_host" \
      --remote-port "$remote_port" \
      --open-loop-horizon 8 \
      --instruction-controller static \
      --num-envs 1 \
      --num-runs 1 \
      --video-mode viewport \
      --disable-subtask \
      --device cuda:0 \
      --headless \
      --renderer realtime \
      --rendering-type balanced \
      --kit_args=--/rtx/verifyDriverVersion/enabled=false

  [[ -s "$launch_evidence" ]] || { echo "seed-$seed behavioral block produced no evidence" >&2; exit 5; }
  echo "GR00T V3-C001 seed $seed complete: $launch_evidence"
done
