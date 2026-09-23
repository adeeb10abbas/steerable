#!/usr/bin/env bash
set -euo pipefail

readonly groot_root=/data/users/ali/vla_wam
readonly groot_runtime="${groot_root}/raw/v3/groot_n17_droid/runtime/instrumentation_attempt02"
readonly groot_queue_root="${groot_root}/raw/v3/groot_n17_droid/runtime/queue_8304_8329_attempt01"
readonly groot_queue_log="${groot_queue_root}/queue_events.jsonl"

if [[ -e "${groot_queue_root}" ]]; then
  printf 'refusing to overwrite retained GR00T queue state: %s\n' "${groot_queue_root}" >&2
  exit 65
fi
mkdir -p "${groot_queue_root}"
exec >>"${groot_queue_root}/queue_stdout.log" 2>>"${groot_queue_root}/queue_stderr.log"
printf '%s\n' "$$" >"${groot_queue_root}/queue.pid"

groot_current_seed=none
groot_queue_failure() {
  readonly groot_status="$?"
  printf '{"event":"queue_failed","seed":"%s","exit_code":%d,"timestamp_utc":"%s"}\n' \
    "${groot_current_seed}" "${groot_status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${groot_queue_log}"
  exit "${groot_status}"
}
trap groot_queue_failure ERR

printf '{"event":"queue_started","first_seed":8304,"last_seed":8329,"timestamp_utc":"%s"}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${groot_queue_log}"

for groot_current_seed in $(seq 8304 8329); do
  printf '{"event":"pair_started","seed":%d,"timestamp_utc":"%s"}\n' \
    "${groot_current_seed}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${groot_queue_log}"

  bash "${groot_runtime}/launch_groot_v3_pair.sh" "${groot_current_seed}" attempt01

  "${groot_root}/envs/groot-n17-b200/bin/python" \
    "${groot_runtime}/validate_compile_groot_v3_pair.py" \
    --study-root "${groot_root}/src/steerable" \
    --raw-root "${groot_root}/raw/v3/groot_n17_droid/phase_a/seed${groot_current_seed}/attempt01" \
    --robolab-output "${groot_root}/external/RoboLab-11142d4/output/v3_groot_seed${groot_current_seed}_attempt01" \
    --runtime-identity "${groot_root}/raw/v3/groot_n17_droid/runtime/runtime_identity.json" \
    --seed "${groot_current_seed}"

  printf '{"event":"pair_validated","seed":%d,"manifest":"%s","timestamp_utc":"%s"}\n' \
    "${groot_current_seed}" \
    "${groot_root}/raw/v3/groot_n17_droid/phase_a/seed${groot_current_seed}/attempt01/compiled/seed${groot_current_seed}_pair_manifest.json" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${groot_queue_log}"
done

printf '{"event":"queue_completed","validated_pair_count":26,"timestamp_utc":"%s"}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${groot_queue_log}"
