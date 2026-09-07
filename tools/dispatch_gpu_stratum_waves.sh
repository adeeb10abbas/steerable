#!/usr/bin/env bash
# Gated V4 GPU-stratum smoke dispatch — full waves blocked until smoke registry passes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CTX=prod-dcwi-warrenq1-vmkub007
NS=211247-prod
POD=211247-sz5vjy-vla4-b200-4gpu
OUT=artifacts/online_correction_v4/execution/gpu_widen_20260908
SPEC_G2=artifacts/online_correction_v4/execution/g2_horizontal_repair_v2_20260908/rendered-a10080-smoke/.dispatch-g2r20260908a10080a.render-spec.json

echo "== sync cluster study checkouts =="
python3 tools/sync_v4_cluster_study_checkouts.py \
  --kube-context "$CTX" \
  --namespace "$NS" \
  --publisher-pod "$POD" \
  --apply

dispatch_smoke() {
  local attempt="$1"
  local gpu="$2"
  python3 tools/dispatch_v4_model_blind_k8s_bundle.py \
    --spec "$SPEC_G2" \
    --output-root "$OUT/g2-smoke-${attempt}" \
    --gpu-product "$gpu" \
    --attempt-id "$attempt" \
    --max-seed-jobs 1 \
    --smoke \
    --publisher-pod "$POD" \
    --create
}

# Smokes only — full waves require a passed registry entry per fixture/GPU/gate.
# Re-run individual smokes after checkout sync; omit attempts already running.
# dispatch_smoke g2gpu20260908a40b NVIDIA-A40
# dispatch_smoke g2gpu20260908a10040 NVIDIA-A100-SXM4-40GB
# dispatch_smoke g2gpu20260908b200 NVIDIA-B200

echo "Smoke dispatch helper loaded; uncomment target smokes after verifying checkout pin manifest."
