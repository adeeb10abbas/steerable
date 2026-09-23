"""Compile a finite, non-launching HEIGHT/DIST qualification campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .prospective_family_capture import verify_capture_artifacts
from .prospective_family_designs import CANDIDATE_STATUS, PLAN_SCHEMA, _digest

SCHEMA = "sgw-01-family-finite-campaign-v1"


def compile_campaign(*, plan_path: Path, baseline_captures: Mapping[str, Path], output: Path) -> dict[str, Any]:
    """Create immutable job descriptors; deliberately does not launch them."""

    if output.exists():
        raise FileExistsError("refusing to overwrite a finite family campaign")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("plan_sha256") != _digest(plan, "plan_sha256"):
        raise ValueError("prospective design plan is malformed")
    if not 1 <= plan.get("design_slot_count", 0) <= 100:
        raise ValueError("campaign requires a fixed 1..100-slot plan")
    if set(baseline_captures) != {"left", "right"}:
        raise ValueError("both independently verified baseline captures are required")
    verified = {side: verify_capture_artifacts(Path(path)) for side, path in baseline_captures.items()}
    jobs = []
    for row in plan["designs"]:
        if row["status"] == "prospective_design_requires_zero_model_capture":
            jobs.append({
                "design_id": row["design_id"],
                "family": plan["family"],
                "side": row["side"],
                "status": "blocked_pending_candidate_overlay_and_fresh_zero_model_capture",
                "candidate_overlay_status": CANDIDATE_STATUS,
                "fixed_trial_contract": {
                    "controller": "existing_calibrated_family_abs_ik",
                    "action_cap": 450,
                    "goal_signs": [1, -1],
                    "fresh_resets_per_goal": 3,
                    "total_scripted_trials": 6,
                    "retry_permitted": False,
                    "retain_trial_and_warmup_video": True,
                },
                "required_postprocess": {
                    "status": "blocked_pending_external_exit_zero_and_artifact_verification",
                    "required_output": "family_verification.json",
                },
            })
        else:
            jobs.append({
                "design_id": row["design_id"], "family": plan["family"], "side": row["side"],
                "status": "geometrically_rejected_slot_no_refill", "retry_permitted": False,
            })
    value = {
        "schema_version": SCHEMA,
        "family": plan["family"],
        "plan": {"path": str(plan_path.resolve()), "sha256": _sha256(plan_path), "plan_sha256": plan["plan_sha256"]},
        "baseline_capture_verification": verified,
        "fixed_slot_count": len(plan["designs"]),
        "jobs": jobs,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "status": "compiled_not_authorized_to_launch_or_release",
        "release_permitted": False,
    }
    value["campaign_sha256"] = _digest(value, "campaign_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def verify_external_postprocess(*, campaign_path: Path, output: Path, returncode: int, verification_path: Path) -> dict[str, Any]:
    """Record only a successful external verifier that produced a bound result."""

    if output.exists():
        raise FileExistsError("refusing to overwrite external postprocess receipt")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if campaign.get("schema_version") != SCHEMA or campaign.get("campaign_sha256") != _digest(campaign, "campaign_sha256"):
        raise ValueError("campaign identity differs")
    if returncode != 0:
        raise RuntimeError("external postprocess failed; no success-shaped receipt is written")
    if not verification_path.is_file():
        raise RuntimeError("external postprocess exited zero without its verification artifact")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("campaign_sha256") != campaign["campaign_sha256"]:
        raise RuntimeError("external verification is not bound to this campaign")
    value = {
        "schema_version": "sgw-01-family-external-postprocess-v1",
        "campaign_sha256": campaign["campaign_sha256"],
        "verification": {"path": str(verification_path.resolve()), "sha256": _sha256(verification_path)},
        "external_returncode": returncode,
        "status": "external_postprocess_verified_not_fixture_release",
        "release_permitted": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
