#!/usr/bin/env bash
set -euo pipefail

# Run the frozen prompt-blind Cosmos future scorer sequentially on physical
# GPU 1. Localization caches make an interrupted invocation resumable without
# changing the frozen calibration or semantic decision rules.

readonly STUDY_ROOT="/home/ali/projects/steerable"
readonly ROBOLAB_OUTPUT="/home/ali/projects/RoboLab/output"
readonly PYTHON_BIN="/home/ali/cosmos-framework/.venv/bin/python"
readonly SCORER="${STUDY_ROOT}/tools/score_cosmos_semantic_futures.py"
readonly CALIBRATION="${STUDY_ROOT}/artifacts/vla_wam_shared_v1/semantic_future_calibration.json"
readonly SEMANTIC_ROOT="${STUDY_ROOT}/artifacts/vla_wam_shared_v1/semantic_confirmation"
readonly PROBE_ROOT="${STUDY_ROOT}/artifacts/vla_wam_shared_v1/command_probe"

run_closed_loop() {
  local condition="$1"
  local source_folder="$2"
  local output_dir="${SEMANTIC_ROOT}/${condition}"
  mkdir -p "${output_dir}"
  echo "[semantic-confirmation] closed-loop ${condition}"
  CUDA_VISIBLE_DEVICES=1 /usr/bin/time -v -o "${output_dir}/runtime.txt" \
    "${PYTHON_BIN}" "${SCORER}" score \
      --task-dir \
        "${ROBOLAB_OUTPUT}/${source_folder}/RubiksCubeLeftOfBowlMatchedTask" \
        "${ROBOLAB_OUTPUT}/${source_folder}/RubiksCubeRightOfBowlMatchedTask" \
      --calibration "${CALIBRATION}" \
      --output-dir "${output_dir}" \
      2>&1 | tee "${output_dir}/score.log"
}

run_probe() {
  local source_name="$1"
  local output_name="$2"
  local output_dir="${PROBE_ROOT}/${output_name}"
  mkdir -p "${output_dir}"
  echo "[semantic-confirmation] fixed-observation ${source_name}"
  CUDA_VISIBLE_DEVICES=1 /usr/bin/time -v -o "${output_dir}/runtime.txt" \
    "${PYTHON_BIN}" "${SCORER}" score-probe \
      --probe-dir "${PROBE_ROOT}/${source_name}" \
      --calibration "${CALIBRATION}" \
      --output-dir "${output_dir}" \
      2>&1 | tee "${output_dir}/score.log"
}

cd "${STUDY_ROOT}"

run_closed_loop cosmos_canonical v1_cosmos_canonical
run_closed_loop cosmos_vague v1_cosmos_vague
run_closed_loop cosmos_declarative v1_cosmos_declarative
run_closed_loop cosmos_contrastive v1_cosmos_contrastive
run_probe cosmos_gpu1 cosmos_gpu1_semantics
run_probe direct_task_cosmos direct_task_cosmos_semantics

echo "[semantic-confirmation] complete"
