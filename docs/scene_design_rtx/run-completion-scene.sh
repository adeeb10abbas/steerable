#!/usr/bin/env bash
# Run one authored scene. A fresh, uniquely named output is mandatory.
set -euo pipefail
TASK_ROOT=${SGW_SCENE_ROOT:-/home/ali/sgw-scene-design-20260923}
DESIGN=${1:?absolute design JSON path}
GPU=${2:?physical GPU index}
OUTPUT=${3:?absolute new evidence directory}
CODE_ROOT=${SGW_SCENE_CODE_ROOT:-$TASK_ROOT/steerable}
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT:$TASK_ROOT/RoboLab" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU" OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$TASK_ROOT/cache/xdg" WARP_CACHE_PATH="$TASK_ROOT/cache/warp" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
STUDY=artifacts/workshops/spatial_grounding_v1
exec "$TASK_ROOT/venv/bin/python" -m experiments.workshops.spatial_grounding_v1.scene_completion_runner \
  --design "$DESIGN" --output "$OUTPUT" --robolab-root "$TASK_ROOT/RoboLab" \
  --workspace "$STUDY/infrastructure/a40-20260922z-workspace.json" \
  --assets-manifest "$TASK_ROOT/evidence/assets.json" \
  --calibration "$STUDY/controller_calibrations/lat-closed-pad-20260923.json" \
  --headless --num-envs 1 --device cuda:0 --renderer realtime --rendering-type balanced --rendering_mode balanced \
  '--kit_args=--/rtx/verifyDriverVersion/enabled=false'
