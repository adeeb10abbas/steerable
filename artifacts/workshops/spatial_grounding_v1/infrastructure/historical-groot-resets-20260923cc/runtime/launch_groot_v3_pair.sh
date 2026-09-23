#!/usr/bin/env bash
set -euo pipefail

readonly groot_seed="${1:?environment seed is required}"
readonly groot_attempt="${2:?attempt label is required}"
if [[ ! "${groot_seed}" =~ ^83[0-9][0-9]$ ]]; then
  printf 'refusing unexpected GR00T v3 seed: %s\n' "${groot_seed}" >&2
  exit 64
fi

readonly groot_root=/data/users/ali/vla_wam
readonly groot_raw_root="${groot_root}/raw/v3/groot_n17_droid/phase_a/seed${groot_seed}/${groot_attempt}"
readonly groot_output_name="v3_groot_seed${groot_seed}_${groot_attempt}"
readonly groot_robolab_output="${groot_root}/external/RoboLab-11142d4/output/${groot_output_name}"
readonly groot_bridge="${groot_root}/raw/v3/groot_n17_droid/runtime/instrumentation_attempt02/robolab_bridge_runtime.py"

if [[ -e "${groot_raw_root}" || -e "${groot_robolab_output}" ]]; then
  printf 'refusing to overwrite retained GR00T evidence for seed %s attempt %s\n' "${groot_seed}" "${groot_attempt}" >&2
  exit 65
fi

mkdir -p "${groot_raw_root}/thermal"
exec >>"${groot_raw_root}/guard_stdout.log" 2>>"${groot_raw_root}/guard_stderr.log"
printf '%s\n' "$$" >"${groot_raw_root}/guard.pid"

exec "${groot_root}/envs/groot-n17-b200/bin/python" \
  "${groot_root}/src/steerable/tools/native_process_group_thermal_guard.py" \
  --launch \
  --gpu-index 1 \
  --output "${groot_raw_root}/thermal/events.jsonl" \
  --ledger-output "${groot_raw_root}/thermal/groot_n17_droid_vla_interventions.json" \
  --invalid-attempts-output "${groot_raw_root}/thermal/groot_n17_droid_vla_invalid_attempts.json" \
  --model-id groot_n17_droid_vla \
  --pair-id "seed${groot_seed}" \
  --environment-seed "${groot_seed}" \
  --sampling-seed "${groot_seed}" \
  --requested-relation left \
  --requested-relation right \
  --poll-seconds 0.5 \
  -- \
  env \
  CUDA_VISIBLE_DEVICES=1 \
  PYTHONUNBUFFERED=1 \
  OMNI_KIT_ACCEPT_EULA=YES \
  NVIDIA_DRIVER_CAPABILITIES=all \
  VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  LD_LIBRARY_PATH="${groot_root}/envs/groot-render-libs/lib:${groot_root}/envs/robolab-native-libs-ubuntu2204/usr/lib/x86_64-linux-gnu:${groot_root}/envs/isaac-system-libs/lib:/usr/lib/x86_64-linux-gnu" \
  WARP_CACHE_PATH="${groot_root}/cache/v3/groot_warp" \
  XDG_CACHE_HOME="${groot_root}/cache/v3/groot_xdg" \
  PATH="${groot_root}/tools/git-lfs:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  "${groot_root}/envs/robolab-v2-isaac50/bin/python" \
  "${groot_bridge}" \
  --study-root "${groot_root}/src/steerable" \
  --environment-seed "${groot_seed}" \
  --sampling-seed-base "${groot_seed}" \
  --runtime-identity "${groot_root}/raw/v3/groot_n17_droid/runtime/runtime_identity.json" \
  --release-gate "${groot_root}/raw/v3/groot_n17_droid/preflight/seed8303_attempt02/release_gate.json" \
  --state-capture-dir "${groot_raw_root}/simulator/state_capture" \
  --action-trace-dir "${groot_raw_root}/actions" \
  --remote-host 127.0.0.1 \
  --remote-port 5555 \
  --open-loop-horizon 8 \
  --instruction-controller static \
  --condition both \
  --output-dir "${groot_raw_root}/simulator" \
  --output-folder-name "${groot_output_name}" \
  --num-envs 1 \
  --num-runs 1 \
  --video-mode viewport \
  --disable-subtask \
  --headless \
  --device cuda:0 \
  --renderer realtime \
  --rendering-type balanced \
  --kit_args=--/rtx/verifyDriverVersion/enabled=false
