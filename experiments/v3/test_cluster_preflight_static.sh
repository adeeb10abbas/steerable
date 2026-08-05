#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
script="$root/tools/vla_wam_v3_cluster_preflight.sh"
runbook="$root/experiments/v3/CLUSTER_PREFLIGHT_RUNBOOK.md"

bash -n "$script"
for required in '--context' '--namespace' '--pod' '--pvc-root' '--study-root' '--vulkan-icd' '--sapien-gate-cmd' \
  'nvidia-smi --query-gpu' 'nvidia-smi --query-compute-apps' 'vulkaninfo --summary' \
  'https://github.com/' 'https://huggingface.co/' 'validate_vla_wam_v2_protocol.py' \
  'validate_vla_wam_v3_protocol.py' 'has_ali_owner_label' 'exact ali owner/user value' \
  'This did not reserve GPUs or authorize inference'; do
  rg -F --quiet -- "$required" "$script"
done

if rg -n -- '--all-namespaces|kget get pods|kubectl.*\b(create|delete|scale)\b' "$script"; then
  printf 'ERROR: preflight contains a forbidden cluster mutation or enumeration.\n' >&2
  exit 1
fi
rg -F --quiet 'not live verification' "$runbook"
rg -F --quiet '/etc/vulkan/icd.d/nvidia_icd.json' "$runbook"
rg -F --quiet 'VULKAN_ICD must be an absolute in-pod path' "$script"

v2_line=$(rg -n 'validate_vla_wam_v2_protocol.py' "$script" | cut -d: -f1)
v3_line=$(rg -n 'validate_vla_wam_v3_protocol.py' "$script" | cut -d: -f1)
gpu_line=$(rg -n "say 'GPU availability" "$script" | cut -d: -f1)
[[ "$v2_line" -lt "$gpu_line" && "$v3_line" -lt "$gpu_line" ]]
printf 'PASS: static v3 preflight safety checks passed.\n'
