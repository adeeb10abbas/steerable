"""Finite, fail-closed runner for one frozen HEIGHT/DIST campaign slot.

This module only prepares immutable inputs and verifies child-process outputs.
It never creates Kubernetes resources, selects GPUs, or releases a fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Callable, Mapping, Sequence

from .family_campaign import SCHEMA, _sha256
from .family_campaign_verifier import verify_design
from .prospective_family_designs import _digest, author_candidate_overlay

CALIBRATION_SHA256 = "107442ccca01c4ac44ec6e1cb9674d51dbcd8663288a851dc54fa91a124f93d7"
EXECUTOR_SCHEMA = "sgw-01-family-campaign-executor-v1"


def _fsync_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _campaign(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA or raw.get("campaign_sha256") != _digest(raw, "campaign_sha256"):
        raise ValueError("campaign identity differs")
    if not isinstance(raw.get("jobs"), list) or not 1 <= len(raw["jobs"]) <= 100:
        raise ValueError("campaign has no finite 1..100 slot registry")
    if raw.get("model_request_count") != 0 or raw.get("behavioral_episode_count") != 0:
        raise ValueError("campaign must remain zero-model")
    return raw


def _materialize(*, plan: Path, design_id: str, manifest: Path, capture: Path, calibration: Path, output: Path) -> dict[str, Any]:
    # Fixture ownership deliberately stays in height_dist_proposals. Import
    # only after a native child has produced independently verified capture.
    from .height_dist_proposals import materialize_campaign_candidate
    return materialize_campaign_candidate(
        plan_path=plan, design_id=design_id, candidate_manifest_path=manifest,
        candidate_capture_path=capture, controller_calibration_path=calibration, output=output,
    )


def _run_child(command: Sequence[str], *, label: str, root: Path, values: Mapping[str, str]) -> None:
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise ValueError(f"{label} child command is required")
    # Do not apply ``str.format`` to arbitrary child source snippets: native
    # command arguments may legitimately contain JSON braces. Only documented
    # named placeholders are substituted.
    expanded = []
    for item in command:
        for key, value in values.items():
            item = item.replace("{" + key + "}", value)
        expanded.append(item)
    result = subprocess.run(expanded, cwd=root, check=False, capture_output=True, text=True)
    _fsync_json(root / f"{label}-process.json", {
        "schema_version": EXECUTOR_SCHEMA, "label": label, "argv": expanded,
        "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
    })
    if result.returncode != 0:
        raise RuntimeError(f"{label} native child failed")


def run_slot(
    *, campaign_path: Path, index: int, root: Path, controller_calibration: Path,
    capture_command: Sequence[str], qualification_command: Sequence[str],
    materialize: Callable[..., dict[str, Any]] = _materialize,
    verify: Callable[..., dict[str, Any]] = verify_design,
) -> dict[str, Any]:
    """Execute one registered slot exactly once; no retry/refill semantics exist."""

    campaign = _campaign(campaign_path)
    if type(index) is not int or not 0 <= index < len(campaign["jobs"]):
        raise ValueError("index is not a registered campaign slot")
    if root.exists():
        raise FileExistsError("refusing to reuse a campaign slot evidence root")
    if not controller_calibration.is_file() or _sha256(controller_calibration) != CALIBRATION_SHA256:
        raise ValueError("controller calibration differs from the exact frozen calibration")
    job = campaign["jobs"][index]
    if not isinstance(job, Mapping):
        raise ValueError("campaign job is malformed")
    root.mkdir(mode=0o700, parents=True)
    receipt = root / "executor-receipt.json"
    try:
        common = {
            "campaign": str(campaign_path.resolve()), "campaign_sha256": campaign["campaign_sha256"],
            "index": str(index), "design_id": str(job["design_id"]), "root": str(root.resolve()),
            "calibration": str(controller_calibration.resolve()),
        }
        if job.get("status") == "geometrically_rejected_slot_no_refill":
            value = {
                "schema_version": EXECUTOR_SCHEMA, "campaign_sha256": campaign["campaign_sha256"],
                "index": index, "design_id": job["design_id"], "family": campaign["family"],
                "status": "geometric_rejection_accounted_slot_no_refill",
                "model_request_count": 0, "behavioral_episode_count": 0, "release_permitted": False,
            }
            _fsync_json(receipt, value)
            return value
        if job.get("status") != "blocked_pending_candidate_overlay_and_fresh_zero_model_capture":
            raise ValueError("campaign slot has unsupported status")
        plan = Path(campaign["plan"]["path"])
        if not plan.is_file() or _sha256(plan) != campaign["plan"]["sha256"]:
            raise ValueError("campaign plan source differs from immutable campaign binding")
        overlay = root / "candidate-overlay.usda"
        manifest = root / "candidate-overlay.json"
        author_candidate_overlay(plan=json.loads(plan.read_text(encoding="utf-8")), design_id=str(job["design_id"]),
                                 output=overlay, manifest_output=manifest)
        values = {**common, "overlay": str(overlay), "overlay_manifest": str(manifest),
                  "capture": str(root / "candidate_capture.json"), "candidate": str(root / "candidate.json"),
                  "qualification": str(root / "qualification.json")}
        _run_child(capture_command, label="capture", root=root, values=values)
        capture = Path(values["capture"])
        if not capture.is_file():
            raise RuntimeError("capture native child exited zero without candidate capture output")
        from .prospective_family_capture import verify_capture_artifacts
        verify_capture_artifacts(capture)
        materialize(plan=plan, design_id=str(job["design_id"]), manifest=manifest, capture=capture,
                    calibration=controller_calibration, output=Path(values["candidate"]))
        _run_child(qualification_command, label="qualification", root=root, values=values)
        guard = root / "preaction-geometry-guard.json"
        if not guard.is_file():
            raise RuntimeError("qualification child exited without fresh pre-action geometry guard")
        guard_value = json.loads(guard.read_text(encoding="utf-8"))
        expected_resets = [(sign, reset) for sign in (1, -1) for reset in range(3)]
        checks = guard_value.get("checks")
        if (guard_value.get("design_id") != job["design_id"]
                or guard_value.get("candidate_sha256") != _sha256(Path(values["candidate"]))
                or guard_value.get("status") != "measured_banana_geometry_valid_before_actions"
                or guard_value.get("actions_started") is not False
                or not isinstance(checks, list)
                or [(item.get("goal_sign"), item.get("reset_index")) for item in checks if isinstance(item, Mapping)] != expected_resets
                or any(item.get("status") != "measured_banana_geometry_valid_before_actions"
                       or item.get("actions_started") is not False
                       or not isinstance(item.get("raw_reset_sha256"), str)
                       or len(item["raw_reset_sha256"]) != 64 for item in checks)):
            raise RuntimeError("qualification pre-action geometry guard is absent or does not bind measured candidate")
        verification = verify(campaign_path=campaign_path, design_id=str(job["design_id"]), root=root,
                              output=root / "family_verification.json")
        value = {
            "schema_version": EXECUTOR_SCHEMA, "campaign_sha256": campaign["campaign_sha256"],
            "index": index, "design_id": job["design_id"], "family": campaign["family"],
            "verification_sha256": verification["verification_sha256"],
            "status": "externally_verified_candidate_slot_not_fixture_or_behavioral_release",
            "model_request_count": 0, "behavioral_episode_count": 0, "release_permitted": False,
        }
        _fsync_json(receipt, value)
        return value
    except BaseException as error:
        _fsync_json(root / "executor-failure.json", {
            "schema_version": EXECUTOR_SCHEMA, "campaign_path": str(campaign_path.resolve()),
            "index": index, "error_type": type(error).__name__, "error": str(error),
            "traceback": "".join(traceback.format_exception(error)),
            "model_request_count": 0, "behavioral_episode_count": 0, "release_permitted": False,
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--controller-calibration", type=Path, required=True)
    parser.add_argument("--capture-command-json", required=True)
    parser.add_argument("--qualification-command-json", required=True)
    args = parser.parse_args()
    capture, qualification = json.loads(args.capture_command_json), json.loads(args.qualification_command_json)
    if not isinstance(capture, list) or not isinstance(qualification, list):
        raise ValueError("native child commands must be JSON string arrays")
    run_slot(campaign_path=args.campaign, index=args.index, root=args.root,
             controller_calibration=args.controller_calibration,
             capture_command=capture, qualification_command=qualification)


if __name__ == "__main__":
    main()
