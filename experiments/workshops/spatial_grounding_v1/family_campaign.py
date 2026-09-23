"""Compile a finite, non-launching HEIGHT/DIST qualification campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .prospective_family_capture import verify_capture_artifacts
from .prospective_family_designs import CANDIDATE_STATUS, PLAN_SCHEMA, _digest

SCHEMA = "sgw-01-family-finite-campaign-v1"


def compile_campaign(
    *, plan_path: Path, baseline_captures: Mapping[str, Path], baseline_reviews: Mapping[str, Path], output: Path,
) -> dict[str, Any]:
    """Create immutable job descriptors; deliberately does not launch them."""

    if output.exists():
        raise FileExistsError("refusing to overwrite a finite family campaign")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _validate_finite_plan(plan)
    if set(baseline_captures) != {"left", "right"}:
        raise ValueError("both independently verified baseline captures are required")
    if set(baseline_reviews) != {"left", "right"}:
        raise ValueError("both independent reviewed baseline decisions are required")
    verified = {
        side: _bound_baseline_capture(
            plan, side, Path(baseline_captures[side]), verify_capture_artifacts(Path(baseline_captures[side])),
        )
        for side in ("left", "right")
    }
    reviewed = {
        side: _reviewed_baseline(
            Path(baseline_reviews[side]), Path(baseline_captures[side]), plan["baselines"][side]["capture"],
            family=plan["family"], side=side,
        )
        for side in ("left", "right")
    }
    jobs = []
    for row in plan["designs"]:
        if row["status"] == "prospective_design_requires_zero_model_capture":
            jobs.append({
                "design_id": row["design_id"],
                "family": plan["family"],
                "side": row["side"],
                "plan_path": str(plan_path.resolve()),
                "plan_sha256": plan["plan_sha256"],
                "design_sha256": _digest(row),
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
        elif row["status"] == "prospective_design_rejected_geometrically":
            jobs.append({
                "design_id": row["design_id"], "family": plan["family"], "side": row["side"],
                "status": "geometrically_rejected_slot_no_refill", "retry_permitted": False,
            })
        else:
            raise ValueError(f"campaign plan has unknown design status: {row['status']!r}")
    value = {
        "schema_version": SCHEMA,
        "family": plan["family"],
        "plan": {"path": str(plan_path.resolve()), "sha256": _sha256(plan_path), "plan_sha256": plan["plan_sha256"]},
        "baseline_capture_verification": verified,
        "baseline_scene_review": reviewed,
        "fixed_slot_count": len(plan["designs"]),
        "jobs": jobs,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "status": "compiled_native_visual_setup_baselines_not_authorized_to_launch_or_release",
        "release_permitted": False,
    }
    value["campaign_sha256"] = _digest(value, "campaign_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def _reviewed_baseline(
    path: Path, capture_path: Path, plan_capture: Mapping[str, Any], *, family: str, side: str,
) -> dict[str, Any]:
    """Bind one sealed native-visual judgment to the exact baseline receipt.

    Native visual setup is intentionally narrower than candidate, physical,
    policy, or category-recognition acceptance.  This gate permits only
    prospective design authoring from the reviewed baseline.
    """

    if not path.is_file():
        raise FileNotFoundError(f"independent baseline scene review is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    scenes = value.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("independent review lacks sealed scene judgments")
    expected_scene_suffix = f"{family.lower()}-{side}"
    matches = [
        row for row in scenes
        if isinstance(row, Mapping)
        and isinstance(row.get("capture"), Mapping)
        and row["capture"].get("sha256") == _sha256(capture_path)
        and isinstance(row.get("scene"), str)
        and row["scene"].lower().endswith(expected_scene_suffix)
    ]
    if len(matches) != 1:
        raise ValueError("sealed review must select exactly one scene for the baseline capture")
    judgment = matches[0]
    if (
        judgment.get("disposition") != "ACCEPT_NATIVE_VISUAL_SETUP_ONLY"
    ):
        raise ValueError("sealed review scene family/side is not native-visual accepted for this plan")
    capture = judgment["capture"]
    if (
        not isinstance(capture, Mapping)
        or capture.get("sha256") != _sha256(capture_path)
        or capture.get("sha256") != plan_capture.get("sha256")
        or str(capture_path.resolve()) != plan_capture.get("path")
    ):
        raise ValueError("baseline scene review is not bound to its capture receipt")
    return {
        "path": str(path.resolve()), "sha256": _sha256(path),
        "scene": judgment["scene"], "capture_sha256": capture["sha256"],
        "disposition": judgment["disposition"],
        "scope": "native_visual_setup_only_not_candidate_physical_or_model_acceptance",
    }


def _validate_finite_plan(plan: Mapping[str, Any]) -> None:
    """Require every registered fixed slot exactly once before campaign compilation."""

    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("plan_sha256") != _digest(plan, "plan_sha256"):
        raise ValueError("prospective design plan is malformed")
    count = plan.get("design_slot_count")
    designs = plan.get("designs")
    if not isinstance(count, int) or not 1 <= count <= 100 or not isinstance(designs, list) or len(designs) != count:
        raise ValueError("campaign requires a complete fixed 1..100-slot design list")
    identifiers = [row.get("design_id") for row in designs if isinstance(row, Mapping)]
    if len(identifiers) != count or any(not isinstance(item, str) or not item for item in identifiers) or len(set(identifiers)) != count:
        raise ValueError("campaign design list has duplicate or missing design IDs")
    accepted = [
        row["design_id"] for row in designs
        if row.get("status") == "prospective_design_requires_zero_model_capture"
    ]
    if plan.get("accepted_design_ids") != accepted or plan.get("accepted_design_count") != len(accepted):
        raise ValueError("campaign accepted design registry differs from fixed design list")
    if plan.get("geometric_rejection_count") != count - len(accepted):
        raise ValueError("campaign geometric rejection count differs from fixed design list")
    baselines = plan.get("baselines")
    if not isinstance(baselines, Mapping) or set(baselines) != {"left", "right"}:
        raise ValueError("campaign plan lacks both registered baseline bindings")


def _bound_baseline_capture(
    plan: Mapping[str, Any], side: str, path: Path, verified: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind supplied capture bytes to the plan's exact side-specific reference."""

    binding = plan["baselines"][side]
    capture = binding.get("capture") if isinstance(binding, Mapping) else None
    if not isinstance(capture, Mapping):
        raise ValueError(f"campaign plan {side} baseline lacks capture binding")
    if str(path.resolve()) != capture.get("path") or _sha256(path) != capture.get("sha256"):
        raise ValueError(f"supplied {side} baseline capture differs from the plan binding")
    receipt = verified.get("receipt") if isinstance(verified, Mapping) else None
    if not isinstance(receipt, Mapping) or receipt.get("sha256") != capture["sha256"]:
        raise ValueError(f"verified {side} baseline receipt differs from the plan binding")
    return dict(verified)


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
    _validate_external_verification(verification, campaign["campaign_sha256"])
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


def _validate_external_verification(verification: Mapping[str, Any], campaign_sha256: str) -> None:
    """Reject success-shaped output unless the external verifier retained evidence."""

    if (
        verification.get("schema_version") != "sgw-01-family-campaign-verification-v1"
        or verification.get("campaign_sha256") != campaign_sha256
        or verification.get("status") != "verified_evidence_not_fixture_release"
        or verification.get("release_permitted") is not False
        or verification.get("model_request_count") != 0
        or verification.get("behavioral_episode_count") != 0
        or verification.get("verification_sha256") != _digest(verification, "verification_sha256")
    ):
        raise RuntimeError("external verification is not a bound campaign evidence result")
    required_hashes = ("candidate_capture_sha256", "qualification_sha256")
    if any(not isinstance(verification.get(key), str) or len(verification[key]) != 64 for key in required_hashes):
        raise RuntimeError("external verification lacks bound capture and qualification evidence")
    trials = verification.get("trials")
    expected = [(sign, reset) for sign in (1, -1) for reset in range(3)]
    early = verification.get("physical_geometry_rejection")
    if early is not None:
        if (
            not isinstance(early, Mapping) or early.get("rejection_scope") != "reset"
            or (early.get("goal_sign"), early.get("reset_index")) not in expected
            or early.get("controller_actions_executed") != 0
            or not isinstance(early.get("reason"), str) or not early["reason"]
        ):
            raise RuntimeError("external verification has malformed early rejection")
        expected = expected[:expected.index((early["goal_sign"], early["reset_index"])) + 1]
    if not isinstance(trials, list) or len(trials) != len(expected):
        raise RuntimeError("external verification lacks six retained trial evidence records")
    identities = [(row.get("goal_sign"), row.get("reset_index")) for row in trials if isinstance(row, Mapping)]
    if identities != expected:
        raise RuntimeError("external verification trial identities differ from the fixed contract")
    if any(not isinstance(row.get("trial_sha256"), str) or len(row["trial_sha256"]) != 64 for row in trials):
        raise RuntimeError("external verification has malformed trial evidence hashes")
    for key, expected_hash in (
        ("candidate_capture", verification["candidate_capture_sha256"]),
        ("qualification", verification["qualification_sha256"]),
    ):
        _verify_evidence_file(verification.get(key), expected_hash)
    for row in trials:
        _verify_evidence_file(row, row["trial_sha256"])


def _verify_evidence_file(record: Any, expected_sha256: str) -> None:
    if not isinstance(record, Mapping):
        raise RuntimeError("external verification lacks retained evidence file record")
    path = Path(str(record.get("path", "")))
    if (
        not path.is_file()
        or record.get("sha256") != expected_sha256
        or record.get("bytes") != path.stat().st_size
        or _sha256(path) != expected_sha256
    ):
        raise RuntimeError("external verification evidence file differs from its retained record")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
