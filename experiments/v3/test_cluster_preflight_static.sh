#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
script="$root/tools/vla_wam_v3_cluster_preflight.sh"
runbook="$root/experiments/v3/CLUSTER_PREFLIGHT_RUNBOOK.md"

bash -n "$script"
for required in '--context' '--namespace' '--pod' '--pvc-root' '--study-root' '--vulkan-icd' '--credential-gate-cmd' '--sapien-gate-cmd' \
  'nvidia-smi --query-gpu' 'nvidia-smi --query-compute-apps' 'render and capture a frame' \
  '/dev/nvidiactl' 'Do not assume that physical N is zero' \
  'https://github.com/' 'https://huggingface.co/' 'validate_vla_wam_v2_protocol.py' \
  'validate_vla_wam_v3_protocol.py' 'has_ali_owner_label' 'actual_pod_name' \
  'exact ali owner/user value' \
  'This did not reserve GPUs or authorize inference'; do
  rg -F --quiet -- "$required" "$script"
done
if rg -F --quiet 'test -c /dev/nvidia0' "$script"; then
  printf 'ERROR: preflight must not assume Kubernetes exposes physical GPU zero.\n' >&2
  exit 1
fi

if rg -n -- '--all-namespaces|kget get pods|kubectl.*\b(create|delete|scale)\b' "$script"; then
  printf 'ERROR: preflight contains a forbidden cluster mutation or enumeration.\n' >&2
  exit 1
fi
rg -F --quiet 'not live verification' "$runbook"
rg -F --quiet '/etc/vulkan/icd.d/nvidia_icd.json' "$runbook"
rg -F --quiet 'VULKAN_ICD must be an absolute in-pod path' "$script"
if rg -n -- 'vulkaninfo' "$script"; then
  printf 'ERROR: preflight must not require the unavailable vulkaninfo utility.\n' >&2
  exit 1
fi

v2_line=$(rg -n 'validate_vla_wam_v2_protocol.py' "$script" | cut -d: -f1)
v3_line=$(rg -n 'validate_vla_wam_v3_protocol.py' "$script" | cut -d: -f1)
egress_line=$(rg -n "say 'explicit Vulkan ICD path and external egress'" "$script" | cut -d: -f1)
credential_line=$(rg -n "say 'model credential or local snapshot gate'" "$script" | cut -d: -f1)
gpu_line=$(rg -n "say 'GPU availability" "$script" | cut -d: -f1)
sapien_line=$(rg -n "say 'model-specific headless SAPIEN gate'" "$script" | cut -d: -f1)
[[ "$v2_line" -lt "$gpu_line" && "$v3_line" -lt "$gpu_line" ]]
[[ "$egress_line" -lt "$credential_line" && "$credential_line" -lt "$gpu_line" && "$gpu_line" -lt "$sapien_line" ]]
printf 'PASS: static v3 preflight safety checks passed.\n'
