#!/usr/bin/env python3
"""Live Isaac positive control for horizontal NaturalGraspDetector."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from v4_natural_grasp_live_control_runner import FIXTURE_DEFAULTS, main  # noqa: E402

FIXTURE_ID = "horizontal"
_defaults = FIXTURE_DEFAULTS[FIXTURE_ID]
RECEIPT_SCHEMA = _defaults["receipt_schema"]
CANONICAL_ENV_SEED = _defaults["environment_seed"]
DEFAULT_SCALE = 0.5
DEFAULT_GOAL = _defaults["goal"]


if __name__ == "__main__":
    raise SystemExit(main(fixture_id=FIXTURE_ID))
