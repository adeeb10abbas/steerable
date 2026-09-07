#!/usr/bin/env bash
# Gated V4 GPU-stratum smoke dispatch with isolated study checkouts.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CTX=prod-dcwi-warrenq1-vmkub007
NS=211247-prod
POD=211247-sz5vjy-vla4-b200-4gpu
OUT=artifacts/online_correction_v4/execution/gpu_widen_20260908
SPEC_G2=artifacts/online_correction_v4/execution/g2_horizontal_repair_v2_20260908/rendered-a10080-smoke/.dispatch-g2r20260908a10080a.render-spec.json

echo "== verify legacy shared checkouts are not mutable =="
python3 tools/sync_v4_cluster_study_checkouts.py \
  --kube-context "$CTX" \
  --namespace "$NS" \
  --publisher-pod "$POD"

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

echo "Isolated checkout dispatch helper loaded; smokes provision per-attempt study roots automatically."
