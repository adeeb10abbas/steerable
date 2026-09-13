"""Frozen-manifest reflected-layout LEFT task."""

import os
from pathlib import Path

from fixture_tasks import build_timeout_only_task_class


WMFForecastReflectedLeftTask = build_timeout_only_task_class(
    manifest_path=Path(os.environ["WMF_FORECAST_POSE_MANIFEST"]),
    manifest_sha256=os.environ["WMF_FORECAST_POSE_MANIFEST_SHA256"],
    layout_pair_id=os.environ["WMF_FORECAST_LAYOUT_PAIR_ID"],
    layout_arm="reflected",
    command="left",
    class_name="WMFForecastReflectedLeftTask",
)
