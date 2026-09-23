"""Offline, fail-closed verifier for a finished HEIGHT/DIST campaign design."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import imageio.v2 as iio

from .fixtures import ACTION_CAP
from .family_campaign import SCHEMA
from .prospective_family_designs import _digest


def verify_design(*, campaign_path: Path, design_id: str, root: Path, output: Path) -> dict[str, Any]:
    """Verify family binding, six trial identities, retained videos, and postprocess inputs."""

    if output.exists():
        raise FileExistsError("refusing to overwrite family verification")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if campaign.get("schema_version") != SCHEMA or campaign.get("campaign_sha256") != _digest(campaign, "campaign_sha256"):
        raise ValueError("campaign identity differs")
    job = next((row for row in campaign["jobs"] if row["design_id"] == design_id), None)
    if not isinstance(job, Mapping) or job.get("status") != "blocked_pending_candidate_overlay_and_fresh_zero_model_capture":
        raise ValueError("design is not a campaign qualification job")
    candidate_capture = root / "candidate_capture.json"
    qualification = root / "qualification.json"
    if not candidate_capture.is_file() or not qualification.is_file():
        raise ValueError("candidate capture and qualification outputs are both required")
    capture = json.loads(candidate_capture.read_text(encoding="utf-8"))
    result = json.loads(qualification.read_text(encoding="utf-8"))
    if capture.get("family") != campaign["family"] or result.get("family") != campaign["family"]:
        raise ValueError("family binding differs from campaign")
    if capture.get("model_request_count") != 0 or capture.get("behavioral_episode_count") != 0:
        raise ValueError("candidate capture is not zero-model")
    if result.get("action_cap") != ACTION_CAP or result.get("controller_identity", {}).get("recipe") is None:
        raise ValueError("qualification lacks calibrated 450-action controller identity")
    expected = [(sign, reset) for sign in (1, -1) for reset in range(3)]
    checks = result.get("checks")
    if not isinstance(checks, list) or [(row.get("goal_sign"), row.get("reset_index")) for row in checks] != expected:
        raise ValueError("qualification must retain both goals and three fresh resets per goal")
    trial_reports = []
    for check in checks:
        trial = root / "trials" / f"goal-{check['goal_sign']:+d}" / f"reset-{check['reset_index']}" / "trial.json"
        if not trial.is_file():
            raise ValueError("qualification trial receipt is missing")
        receipt = json.loads(trial.read_text(encoding="utf-8"))
        if receipt.get("goal_sign") != check["goal_sign"] or receipt.get("reset_index") != check["reset_index"]:
            raise ValueError("trial identity differs from qualification aggregate")
        if receipt.get("actions_executed") != ACTION_CAP or receipt.get("observed_actions") != ACTION_CAP:
            raise ValueError("trial did not retain the full calibrated action sequence")
        video = _bound_file(trial.parent, receipt.get("viewport_video"))
        _decode_video(video, ACTION_CAP + 1)
        warmup = receipt.get("reset_receipt", {}).get("render_only_warmup")
        warmup_video = _bound_file(root, warmup.get("viewport_video") if isinstance(warmup, Mapping) else None)
        _decode_video(warmup_video, 121)
        trial_sha256 = _sha256(trial)
        trial_reports.append({
            "goal_sign": check["goal_sign"], "reset_index": check["reset_index"],
            "trial_sha256": trial_sha256, "sha256": trial_sha256, "bytes": trial.stat().st_size,
        })
    value = {
        "schema_version": "sgw-01-family-campaign-verification-v1",
        "campaign_sha256": campaign["campaign_sha256"],
        "design_id": design_id,
        "family": campaign["family"],
        "evidence_root": str(root.resolve()),
        "candidate_capture_sha256": _sha256(candidate_capture),
        "qualification_sha256": _sha256(qualification),
        "candidate_capture": _record(candidate_capture),
        "qualification": _record(qualification),
        "trials": [
            {**row, "path": str((root / "trials" / f"goal-{row['goal_sign']:+d}" / f"reset-{row['reset_index']}" / "trial.json").resolve())}
            for row in trial_reports
        ],
        "status": "verified_evidence_not_fixture_release",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "release_permitted": False,
    }
    value["verification_sha256"] = _digest(value, "verification_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def _bound_file(root: Path, record: Any) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError("video receipt is missing")
    path = Path(record.get("path", ""))
    path = path if path.is_absolute() else root / path
    if not path.is_file() or path.stat().st_size != record.get("bytes") or _sha256(path) != record.get("sha256"):
        raise ValueError("retained video differs from receipt")
    return path


def _decode_video(path: Path, frames: int) -> None:
    with iio.get_reader(path) as reader:
        decoded = sum(1 for _ in reader)
    if decoded != frames:
        raise ValueError("retained video frame count differs from physical trace")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}
