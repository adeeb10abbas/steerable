#!/usr/bin/env python3
"""Tests for V4 code-identity aggregate audit receipts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import audit_v4_code_identity_aggregate as audit  # noqa: E402


def _resolve(short: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", short],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return completed.stdout.strip()


def test_audit_never_allows_mixed_runtime_stratum_aggregation() -> None:
    try:
        base = _resolve("0e8ab82")
        head = _resolve("96ccf4a")
    except subprocess.CalledProcessError:
        pytest.skip("repair pin commits unavailable in this checkout")
    report = audit.audit_code_identity_aggregate(
        base_commit=base,
        head_commit=head,
        repo_root=ROOT,
    )
    assert report["aggregate_mixing_allowed"] is False
    assert report["compiler_would_accept_mixed_stratum"] is False
    assert report["re_run_required_for_homogeneous_aggregate"] is True


def test_audit_flags_droid_robolab_diff_between_repair_pins() -> None:
    try:
        base = _resolve("0e8ab82")
        head = _resolve("96ccf4a")
    except subprocess.CalledProcessError:
        pytest.skip("repair pin commits unavailable in this checkout")
    report = audit.audit_code_identity_aggregate(
        base_commit=base,
        head_commit=head,
        repo_root=ROOT,
    )
    assert "experiments/online_correction_v4/droid_robolab.py" in report[
        "model_blind_check_touched_files"
    ]
    assert report["disclosure_only_model_blind_path_clean"] is False


def test_require_natural_grasp_blocks_other_fixtures_without_entry() -> None:
    import v4_dispatch_gates as gates

    with pytest.raises(gates.DispatchGateError, match="horizontal.*missing passed controls"):
        gates.require_natural_grasp_live_control(fixture_id="horizontal")
