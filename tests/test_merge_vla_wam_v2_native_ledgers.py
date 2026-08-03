from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "merge_vla_wam_v2_native_ledgers.py"
sys.path.insert(0, str(ROOT / "tools"))
from merge_vla_wam_v2_native_ledgers import merge_ledgers  # noqa: E402


SCHEMA = "vla-wam-shared-v2-native-thermal-invalid-attempts-v1"
MODEL = "lingbot_va_robotwin"


def write_ledger(path: Path, events: list[dict]) -> None:
    path.write_text(json.dumps({"schema_version": SCHEMA, "events": events}) + "\n")


def event(event_id: str, started: str, *, model_id: str = MODEL) -> dict:
    return {"id": event_id, "model_id": model_id, "started_at_utc": started}


def test_merge_is_deterministic_and_preserves_relation_specific_events(tmp_path: Path) -> None:
    later = tmp_path / "later.json"
    earlier = tmp_path / "earlier.json"
    write_ledger(later, [event("right-stall", "2026-08-03T16:19:28Z")])
    write_ledger(earlier, [event("pair03-left", "2026-08-03T15:40:47Z")])
    payload = merge_ledgers([later, earlier], MODEL)
    assert [row["id"] for row in payload["events"]] == ["pair03-left", "right-stall"]
    assert payload["events"][1]["id"] == "right-stall"


def test_merge_rejects_duplicate_ids_and_cross_model_events(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_ledger(first, [event("duplicate", "2026-08-03T15:00:00Z")])
    write_ledger(second, [event("duplicate", "2026-08-03T16:00:00Z")])
    with pytest.raises(RuntimeError, match="Duplicate native ledger event id"):
        merge_ledgers([first, second], MODEL)
    write_ledger(second, [event("foreign", "2026-08-03T16:00:00Z", model_id="fastwam_robotwin")])
    with pytest.raises(RuntimeError, match="expected 'lingbot_va_robotwin'"):
        merge_ledgers([first, second], MODEL)


def test_cli_requires_model_specific_output_filename(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    write_ledger(source, [])
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--input",
            str(source),
            "--model-id",
            MODEL,
            "--output",
            str(tmp_path / "invalid_attempts.json"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Model-specific output filename" in result.stderr
