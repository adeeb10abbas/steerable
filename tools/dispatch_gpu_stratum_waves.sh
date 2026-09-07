#!/usr/bin/env bash
# Capacity dispatch helper — run from repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CTX=prod-dcwi-warrenq1-vmkub007
NS=211247-prod
OUT=artifacts/online_correction_v4/execution/gpu_widen_20260908
SRC_G2=artifacts/online_correction_v4/execution/g2_horizontal_repair_v2_20260908/rendered-a10080-smoke/v4-g2-horizontal-g2r20260908a10080a-11feca0806
SRC_C=deploy/k8s/v4_lane_bundle/rendered-g3-containment/v4-g3-containment-g3c6p20260906a-30dffd5752
SRC_V=deploy/k8s/v4_lane_bundle/rendered-g3-vertical/v4-g3-vertical-g3c5p20260906a-3210363d9b

replicate_smoke() {
  local label="$1" attempt="$2" gpu="$3"
  local dir="$OUT/g2-smoke-${label}"
  rm -rf "$dir"
  python3 tools/replicate_v4_k8s_bundle_gpu_stratum.py \
    --source-bundle "$SRC_G2" \
    --output-root "$dir" \
    --from-attempt-id g2r20260908a10080a \
    --to-attempt-id "$attempt" \
    --from-gpu-product NVIDIA-A100-SXM4-80GB \
    --to-gpu-product "$gpu" \
    --max-seeds 1
  local bundle
  bundle=$(ls -d "$dir"/v4-g2-horizontal-*)
  python3 tools/prepare_v4_k8s_output_parents.py --bundle-root "$bundle" --kube-context "$CTX"
  kubectl --context "$CTX" -n "$NS" create -k "$bundle"
}

replicate_wave() {
  local label="$1" attempt="$2" gpu="$3" from="$4" src="$5"
  local dir="$OUT/${label}"
  rm -rf "$dir"
  python3 tools/replicate_v4_k8s_bundle_gpu_stratum.py \
    --source-bundle "$src" \
    --output-root "$dir" \
    --from-attempt-id "$from" \
    --to-attempt-id "$attempt" \
    --to-gpu-product "$gpu"
  local bundle
  bundle=$(ls -d "$dir"/v4-g3-*)
  python3 tools/prepare_v4_k8s_output_parents.py --bundle-root "$bundle" --kube-context "$CTX"
  kubectl --context "$CTX" -n "$NS" create -k "$bundle"
}

replicate_smoke a40 g2gpu20260908a40b NVIDIA-A40
replicate_smoke a10040 g2gpu20260908a10040 NVIDIA-A100-SXM4-40GB
replicate_smoke b200 g2gpu20260908b200 NVIDIA-B200
replicate_wave containment-a10040 g3c6p20260908a10040 NVIDIA-A100-SXM4-40GB g3c6p20260906a "$SRC_C"
replicate_wave containment-b200 g3c6p20260908b200 NVIDIA-B200 g3c6p20260906a "$SRC_C"
replicate_wave vertical-a10040 g3c5p20260908a10040 NVIDIA-A100-SXM4-40GB g3c5p20260906a "$SRC_V"
replicate_wave vertical-b200 g3c5p20260908b200 NVIDIA-B200 g3c5p20260906a "$SRC_V"
