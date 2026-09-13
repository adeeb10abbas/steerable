"""Unreleased-candidate reflected-layout RIGHT calibration task."""

import os
from pathlib import Path

from fixture_tasks import build_candidate_gate_task_class


WMFGateReflectedRightTask = build_candidate_gate_task_class(
    candidate_pool_path=Path(os.environ["WMF_FORECAST_CANDIDATE_POOL"]),
    candidate_pool_sha256=os.environ["WMF_FORECAST_CANDIDATE_POOL_SHA256"],
    candidate_id=os.environ["WMF_FORECAST_CANDIDATE_ID"],
    layout_arm="reflected",
    command="right",
    class_name="WMFGateReflectedRightTask",
)
