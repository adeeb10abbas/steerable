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
import signal
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

from .family_campaign import SCHEMA, _sha256
from .family_campaign_verifier import verify_design
from .prospective_family_designs import _digest, author_candidate_overlay

CALIBRATION_SHA256 = "107442ccca01c4ac44ec6e1cb9674d51dbcd8663288a851dc54fa91a124f93d7"
EXECUTOR_SCHEMA = "sgw-01-family-campaign-executor-v1"


@dataclass(frozen=True)
class PhysicalGeometryRejection:
    """Measured candidate/reset geometry rejection, never infrastructure loss."""

    design_id: str
    candidate_sha256: str
    rejection_scope: str
    goal_sign: int | None
    reset_index: int | None
    raw_reset_sha256: str
    reason: str
    controller_actions_executed: int = 0

    def __post_init__(self) -> None:
        if (self.rejection_scope not in {"candidate", "reset"}
                or self.controller_actions_executed != 0
                or not self.reason
                or len(self.candidate_sha256) != 64
                or len(self.raw_reset_sha256) != 64):
            raise ValueError("physical geometry rejection lacks measured zero-action evidence")
        if self.rejection_scope == "reset" and (self.goal_sign not in {-1, 1} or self.reset_index not in range(3)):
            raise ValueError("reset rejection lacks its registered trial identity")


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


def _run_child(
    command: Sequence[str], *, label: str, root: Path, values: Mapping[str, str], timeout_seconds: int,
) -> None:
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise ValueError(f"{label} child command is required")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 7200:
        raise ValueError(f"{label} child requires a finite 1..7200 second wall-time bound")
    # Do not apply ``str.format`` to arbitrary child source snippets: native
    # command arguments may legitimately contain JSON braces. Only documented
    # named placeholders are substituted.
    expanded = []
    for item in command:
        for key, value in values.items():
            item = item.replace("{" + key + "}", value)
        expanded.append(item)
    stdout_path, stderr_path = root / f"{label}.stdout.log", root / f"{label}.stderr.log"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(
            expanded, cwd=root, stdout=stdout, stderr=stderr, start_new_session=True,
        )
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
        stdout.flush(); os.fsync(stdout.fileno())
        stderr.flush(); os.fsync(stderr.fileno())
    _fsync_json(root / f"{label}-process.json", {
        "schema_version": EXECUTOR_SCHEMA, "label": label, "argv": expanded,
        "returncode": returncode, "timed_out": timed_out, "timeout_seconds": timeout_seconds,
        "stdout": {"path": str(stdout_path.resolve()), "sha256": _sha256(stdout_path), "bytes": stdout_path.stat().st_size},
        "stderr": {"path": str(stderr_path.resolve()), "sha256": _sha256(stderr_path), "bytes": stderr_path.stat().st_size},
    })
    if timed_out:
        raise TimeoutError(f"{label} native child exceeded its wall-time bound")
    if returncode != 0:
        raise RuntimeError(f"{label} native child failed")


def _preaction_rejection(
    value: Mapping[str, Any], *, design_id: str, candidate_sha256: str, candidate_capture_sha256: str, root: Path,
) -> PhysicalGeometryRejection | None:
    _validate_guard_common(
        value, design_id=design_id, candidate_sha256=candidate_sha256,
        candidate_capture_sha256=candidate_capture_sha256, root=root,
    )
    if value.get("status") != "physical_geometry_rejection_before_actions":
        return None
    if value.get("controller_actions_executed") != 0:
        raise RuntimeError("physical geometry rejection guard is malformed")
    rejection = PhysicalGeometryRejection(
        design_id=str(value.get("design_id")), candidate_sha256=str(value.get("candidate_sha256")),
        rejection_scope=str(value.get("rejection_scope")), goal_sign=value.get("goal_sign"),
        reset_index=value.get("reset_index"), raw_reset_sha256=str(value["raw_reset"]["sha256"]),
        reason=str(value.get("reason")), controller_actions_executed=value.get("controller_actions_executed"),
    )
    if rejection.design_id != design_id or rejection.candidate_sha256 != candidate_sha256:
        raise RuntimeError("physical geometry rejection does not bind measured candidate")
    return rejection


def _validate_guard_common(
    value: Mapping[str, Any], *, design_id: str, candidate_sha256: str, candidate_capture_sha256: str, root: Path,
) -> None:
    raw = value.get("raw_reset")
    if (value.get("schema_version") != "sgw-01-family-preaction-geometry-guard-v1"
            or value.get("design_id") != design_id or value.get("candidate_sha256") != candidate_sha256
            or value.get("candidate_capture_sha256") != candidate_capture_sha256
            or not isinstance(raw, Mapping) or not isinstance(raw.get("path"), str)
            or not isinstance(raw.get("sha256"), str) or len(raw["sha256"]) != 64
            or type(raw.get("bytes")) is not int):
        raise RuntimeError("qualification per-trial pre-action geometry guard is malformed")
    expected = (root / "state-0000.json").resolve()
    path = Path(raw["path"])
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if (path != expected or not path.is_file() or path.stat().st_size != raw["bytes"]
            or _sha256(path) != raw["sha256"]):
        raise RuntimeError("qualification pre-action raw reset evidence is absent or hash-mismatched")


def _trial_guards(
    root: Path, *, design_id: str, candidate_sha256: str, candidate_capture_sha256: str,
) -> PhysicalGeometryRejection | None:
    """Validate guards exactly where each independent reset occurs."""
    identities = [(goal_sign, reset_index) for goal_sign in (1, -1) for reset_index in range(3)]
    for offset, (goal_sign, reset_index) in enumerate(identities):
            trial_root = root / f"goal-{goal_sign:+d}" / f"reset-{reset_index}"
            guard = trial_root / "preaction-geometry-guard.json"
            if not guard.is_file():
                raise RuntimeError("qualification child omitted a per-trial pre-action geometry guard")
            value = json.loads(guard.read_text(encoding="utf-8"))
            rejection = _preaction_rejection(
                value, design_id=design_id, candidate_sha256=candidate_sha256,
                candidate_capture_sha256=candidate_capture_sha256, root=trial_root,
            )
            if rejection is not None:
                if rejection.goal_sign != goal_sign or rejection.reset_index != reset_index:
                    raise RuntimeError("physical geometry rejection has a different trial identity")
                for future_goal, future_reset in identities[offset + 1:]:
                    future = root / f"goal-{future_goal:+d}" / f"reset-{future_reset}"
                    if future.exists() and any(future.glob("action-*.npy")):
                        raise RuntimeError("physical geometry rejection was followed by controller actions")
                return rejection
            _validate_guard_common(
                value, design_id=design_id, candidate_sha256=candidate_sha256,
                candidate_capture_sha256=candidate_capture_sha256, root=trial_root,
            )
            if (value.get("goal_sign") != goal_sign or value.get("reset_index") != reset_index
                    or value.get("status") != "measured_banana_geometry_valid_before_actions"
                    or value.get("controller_actions_executed") != 0):
                raise RuntimeError("qualification per-trial pre-action geometry guard is malformed")
    return None


def run_slot(
    *, campaign_path: Path, index: int, root: Path, controller_calibration: Path,
    capture_command: Sequence[str], qualification_command: Sequence[str],
    child_timeout_seconds: int = 1800,
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
        manifest = root / "candidate_manifest.json"
        author_candidate_overlay(plan=json.loads(plan.read_text(encoding="utf-8")), design_id=str(job["design_id"]),
                                 output=overlay, manifest_output=manifest)
        values = {**common, "overlay": str(overlay), "overlay_manifest": str(manifest),
                  "capture": str(root / "candidate_capture.json"), "candidate": str(root / "candidate.json"),
                  "qualification": str(root / "qualification.json")}
        _run_child(capture_command, label="capture", root=root, values=values,
                   timeout_seconds=child_timeout_seconds)
        capture = Path(values["capture"])
        if not capture.is_file():
            raise RuntimeError("capture native child exited zero without candidate capture output")
        from .prospective_family_capture import verify_capture_artifacts
        verify_capture_artifacts(capture)
        materialize(plan=plan, design_id=str(job["design_id"]), manifest=manifest, capture=capture,
                    calibration=controller_calibration, output=Path(values["candidate"]))
        _run_child(qualification_command, label="qualification", root=root, values=values,
                   timeout_seconds=child_timeout_seconds)
        candidate_sha256 = _sha256(Path(values["candidate"]))
        candidate = json.loads(Path(values["candidate"]).read_text(encoding="utf-8"))
        candidate_capture_sha256 = candidate.get("metadata", {}).get("candidate_capture_sha256")
        if not isinstance(candidate_capture_sha256, str) or len(candidate_capture_sha256) != 64:
            raise RuntimeError("materialized candidate lacks bound capture hash")
        rejected = _trial_guards(
            root, design_id=str(job["design_id"]), candidate_sha256=candidate_sha256,
            candidate_capture_sha256=candidate_capture_sha256,
        )
        if rejected is not None:
            value = {
                "schema_version": EXECUTOR_SCHEMA, "campaign_sha256": campaign["campaign_sha256"],
                "index": index, "design_id": job["design_id"], "family": campaign["family"],
                "status": "physical_geometry_rejection_accounted_slot_no_refill",
                "physical_rejection": asdict(rejected),
                "model_request_count": 0, "behavioral_episode_count": 0, "release_permitted": False,
            }
            _fsync_json(receipt, value)
            return value
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
    parser.add_argument("--child-timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    capture, qualification = json.loads(args.capture_command_json), json.loads(args.qualification_command_json)
    if not isinstance(capture, list) or not isinstance(qualification, list):
        raise ValueError("native child commands must be JSON string arrays")
    run_slot(campaign_path=args.campaign, index=args.index, root=args.root,
             controller_calibration=args.controller_calibration,
             capture_command=capture, qualification_command=qualification,
             child_timeout_seconds=args.child_timeout_seconds)


if __name__ == "__main__":
    main()
