#!/usr/bin/env python3
"""Bind a passed zero-request R002 pair gate into an E004 lane release."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.r002_orientation_tolerance import (  # noqa: E402
    CORRECTED_TOLERANCE_RAD,
    ORIGINAL_CONTROL_ASSET_SHA256,
    load_amendment as load_r002_amendment,
    validate_runtime_attestation,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.request0_replay import (  # noqa: E402
    LANE_PREFLIGHT_SCHEMA,
    file_record,
    load_amendment as load_r001_amendment,
    validate_lane_preflight,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.runtime_contract import (  # noqa: E402
    RuntimeContractError,
    load_runtime_bundle,
    sha256_file,
    validate_lane_release,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeContractError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeContractError(f"invalid finite JSON: {path}: {exc}") from exc
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _bound_record(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} file record is missing")
    path = Path(str(value.get("path"))).resolve()
    record = file_record(path)
    for key in ("path", "bytes", "sha256"):
        _require(value.get(key) == record[key], f"{label} changed for {key}")
    return record


def _validate_report(
    report_path: Path,
    *,
    relation: str,
    model_id: str,
    lane_pod_uid: str,
    lane_gpu_uuid: str,
    registration_sha256: str,
    queue_sha256: str,
    candidate_sha256: str,
    r002_amendment_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _load(report_path)
    expected = {
        "schema_version": "vla-wam-shared-v3e004-standalone-model-blind-droid-gate-v2",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E004",
        "passed": True,
        "model_id": model_id,
        "pod_uid": lane_pod_uid,
        "gpu_uuid": lane_gpu_uuid,
        "registration_sha256": registration_sha256,
        "queue_sha256": queue_sha256,
        "model_request_count": 0,
        "behavioral_action_count": 0,
        "behavioral_episode_count": 0,
    }
    for key, wanted in expected.items():
        _require(report.get(key) == wanted, f"R002 {relation} report differs for {key}")
    cell_id = report.get("registered_cell_id")
    _require(
        isinstance(cell_id, str) and cell_id.endswith(f":s000:{relation}"),
        f"R002 {relation} report is not the registered s=0 direction",
    )
    _require(report.get("candidate_sha256") == candidate_sha256, f"R002 {relation} candidate changed")
    attestation = validate_runtime_attestation(
        report.get("live_orientation_realisation_tolerance_amendment"),
        amendment_sha256=r002_amendment_sha256,
        symmetry_level_s=0.0,
    )
    _require(
        attestation["control_scene_asset"]["sha256"] == ORIGINAL_CONTROL_ASSET_SHA256,
        f"R002 {relation} did not bind the original control asset",
    )
    live_gate_record = _bound_record(report.get("live_scene_gate"), f"R002 {relation} live gate")
    live_gate = _load(Path(live_gate_record["path"]))
    gate_attestation = validate_runtime_attestation(
        live_gate.get("orientation_tolerance_attestation"),
        amendment_sha256=r002_amendment_sha256,
        symmetry_level_s=0.0,
    )
    _require(gate_attestation == attestation, f"R002 {relation} report/live-gate attestations differ")
    return report, live_gate_record


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("registration", "queue", "candidate"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--base-lane-release", type=Path, required=True)
    parser.add_argument("--base-lane-release-sha256", required=True)
    parser.add_argument("--request0-replay-amendment", type=Path, required=True)
    parser.add_argument("--request0-replay-amendment-sha256", required=True)
    parser.add_argument("--r002-amendment", type=Path, required=True)
    parser.add_argument("--r002-amendment-sha256", required=True)
    parser.add_argument("--left-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--lane-pod-uid", required=True)
    parser.add_argument("--lane-gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = load_runtime_bundle(
        registration_path=args.registration,
        registration_sha256=args.registration_sha256,
        queue_path=args.queue,
        queue_sha256=args.queue_sha256,
        candidate_path=args.candidate,
        candidate_sha256=args.candidate_sha256,
    )
    release = validate_lane_release(
        args.base_lane_release,
        args.base_lane_release_sha256,
        bundle=bundle,
        model_id=args.model_id,
        lane_pod_uid=args.lane_pod_uid,
        lane_gpu_uuid=args.lane_gpu_uuid,
    )
    load_r001_amendment(args.request0_replay_amendment, args.request0_replay_amendment_sha256)
    load_r002_amendment(
        args.r002_amendment,
        args.r002_amendment_sha256,
        registration_sha256=args.registration_sha256,
        queue_sha256=args.queue_sha256,
        candidate_sha256=args.candidate_sha256,
    )
    left, left_gate = _validate_report(
        args.left_report,
        relation="left",
        model_id=args.model_id,
        lane_pod_uid=args.lane_pod_uid,
        lane_gpu_uuid=args.lane_gpu_uuid,
        registration_sha256=args.registration_sha256,
        queue_sha256=args.queue_sha256,
        candidate_sha256=args.candidate_sha256,
        r002_amendment_sha256=args.r002_amendment_sha256,
    )
    right, right_gate = _validate_report(
        args.right_report,
        relation="right",
        model_id=args.model_id,
        lane_pod_uid=args.lane_pod_uid,
        lane_gpu_uuid=args.lane_gpu_uuid,
        registration_sha256=args.registration_sha256,
        queue_sha256=args.queue_sha256,
        candidate_sha256=args.candidate_sha256,
        r002_amendment_sha256=args.r002_amendment_sha256,
    )
    _require(left.get("matched_pair_id") == right.get("matched_pair_id"), "R002 reports are not one matched pair")
    left_identity = left.get("request0_replay", {}).get("pair_identity_sha256")
    right_identity = right.get("request0_replay", {}).get("pair_identity_sha256")
    _require(isinstance(left_identity, str) and len(left_identity) == 64, "R002 pair identity is invalid")
    _require(left_identity == right_identity, "R002 LEFT/RIGHT pair identities differ")

    output = dict(release)
    output["captured_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    output["study_execution_commit"] = left.get("study_commit")
    output["request0_replay_preflight"] = {
        "schema_version": LANE_PREFLIGHT_SCHEMA,
        "amendment": file_record(args.request0_replay_amendment),
        "left_report": file_record(args.left_report),
        "right_report": file_record(args.right_report),
        "pair_identity_sha256": left_identity,
    }
    attestation = left["live_orientation_realisation_tolerance_amendment"]
    output["r002_preflight"] = {
        "amendment": file_record(args.r002_amendment),
        "effective_live_orientation_realisation_tolerance_rad": CORRECTED_TOLERANCE_RAD,
        "original_control_asset_sha256": ORIGINAL_CONTROL_ASSET_SHA256,
        "left_report": file_record(args.left_report),
        "right_report": file_record(args.right_report),
        "left_live_gate": left_gate,
        "right_live_gate": right_gate,
        "pair_identity_sha256": left_identity,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "runtime_attestation": attestation,
    }
    evidence = dict(output.get("evidence", {}))
    evidence["bound_live_scene_gate"] = left_gate
    left_snapshot = args.left_report.resolve().parent / "live_scene_snapshot.json"
    if left_snapshot.is_file():
        evidence["live_scene_snapshot"] = file_record(left_snapshot)
    viewport = left.get("viewport_video")
    if isinstance(viewport, Mapping):
        evidence["renderer_viewport_video"] = _bound_record(viewport, "R002 LEFT viewport")
    output["evidence"] = evidence
    output["gates"] = {
        **output["gates"],
        "request0_left_capture_right_replay_preflight": True,
    }
    output["release_boundary"] = (
        "Bridge repeats per-cell live geometry/camera and R002 orientation/asset gates "
        "in the same simulator process immediately before request zero."
    )

    validate_lane_preflight(
        output,
        amendment_sha256=args.request0_replay_amendment_sha256,
        model_id=args.model_id,
        lane_pod_uid=args.lane_pod_uid,
        lane_gpu_uuid=args.lane_gpu_uuid,
    )
    destination = args.output.resolve()
    _require(not destination.exists(), f"refusing to overwrite lane release: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_lane_release(
        destination,
        sha256_file(destination),
        bundle=bundle,
        model_id=args.model_id,
        lane_pod_uid=args.lane_pod_uid,
        lane_gpu_uuid=args.lane_gpu_uuid,
    )
    print(json.dumps({
        "output": str(destination),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "matched_pair_id": left.get("matched_pair_id"),
        "pair_identity_sha256": left_identity,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
