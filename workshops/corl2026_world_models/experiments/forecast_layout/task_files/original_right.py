"""Frozen-manifest original-layout RIGHT task."""

import os
from pathlib import Path

from fixture_tasks import build_timeout_only_task_class


WMFForecastOriginalRightTask = build_timeout_only_task_class(
    manifest_path=Path(os.environ["WMF_FORECAST_POSE_MANIFEST"]),
    manifest_sha256=os.environ["WMF_FORECAST_POSE_MANIFEST_SHA256"],
    layout_pair_id=os.environ["WMF_FORECAST_LAYOUT_PAIR_ID"],
    layout_arm="original",
    command="right",
    class_name="WMFForecastOriginalRightTask",
)
