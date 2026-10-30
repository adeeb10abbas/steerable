#!/usr/bin/env python3
"""Audit all compact fixed-observation capture receipts without reading raw PVC data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


RESULTS_COMMIT = "12839900e5c46299e11fff4813e8602949055b86"
REPOSITORY = Path(__file__).resolve().parents[3]
NAMESPACE = "wmf_ablation_001_20260912"
RAW_ROOT = PurePosixPath("/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912")
LAYOUTS = ("P00",) + tuple(f"D{i:02d}" for i in range(1, 5)) + tuple(
    f"C{i:02d}" for i in range(1, 25)
)
EXPECTED_SOURCE_INDEX_SHA256 = "701ad1341ad75d7c10a60f3e1d8aa90072e66c9efd9e5e32678d4f39a01f88fc"
EXPECTED_DESCRIPTOR_INDEX_SHA256 = "21e8565813ef4465162acb1f8aecbbc3d9c9d1d45eb7490074b27eed20c5532f"
PILOT_COMMIT = "ce5ad4282d2392245f843d5270a9656d4e99f842"
LAYOUT_COMMIT = "148b0bad38382b32667d7cf2484773c11fa744ad"
D1_ARRAY_KEYS = [
    "observation/exterior_image_0_left",
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
    "observation/joint_position",
    "observation/cartesian_position",
    "observation/gripper_position",
]
N3_ARRAY_KEYS = [
    "observation/image",
    "observation/joint_position",
    "observation/gripper_position",
]
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def strict_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def load_json(payload: bytes, source: str) -> dict:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON: {source}") from error
    require(type(value) is dict, f"Expected JSON object: {source}")
    return value


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def compact_index_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def git_blob(path: str) -> bytes:
    return subprocess.check_output(
        [
            "git",
            "--no-replace-objects",
            "-C",
            str(REPOSITORY),
            "show",
            f"{RESULTS_COMMIT}:{path}",
        ],
        timeout=30,
    )


def stage(layout: str) -> str:
    if layout == "P00":
        return "pilot"
    if layout.startswith("D"):
        return "development"
    return "confirmation"


def expected_candidate(layout: str) -> str:
    ordinal = "01" if layout in {"P00", "C01", "C17", "C24"} else "00"
    return f"{layout}__candidate_{ordinal}"


def expected_seed(layout: str) -> int:
    if layout == "P00":
        return 2026091000
    if layout.startswith("D"):
        return 2026091100 + int(layout[1:])
    return 2026091200 + int(layout[1:])


def validate_descriptor(
    layout: str, job_id: str, kind: str, descriptor: object, suffix: str
) -> dict:
    require(type(descriptor) is dict, f"Descriptor is not an object: {layout}/{kind}")
    require(
        set(descriptor) == {"bytes", "path", "sha256"},
        f"Descriptor keys changed: {layout}/{kind}",
    )
    size, path, digest = (
        descriptor["bytes"],
        descriptor["path"],
        descriptor["sha256"],
    )
    require(type(size) is int and size > 0, f"Invalid byte count: {layout}/{kind}")
    require(type(path) is str and path.startswith("/"), f"Invalid path: {layout}/{kind}")
    require(type(digest) is str and SHA256_RE.fullmatch(digest), f"Invalid SHA: {layout}/{kind}")
    pure = PurePosixPath(path)
    require(".." not in pure.parts, f"Unsafe path: {layout}/{kind}")
    expected_root = RAW_ROOT / "fixed_observations" / job_id
    require(pure.is_relative_to(expected_root), f"Path escaped job root: {layout}/{kind}")
    require(path.endswith(suffix), f"Unexpected path suffix: {layout}/{kind}")
    return {
        "layout_pair_id": layout,
        "kind": kind,
        "bytes": size,
        "path": path,
        "sha256": digest,
    }


def audit() -> dict:
    sources, descriptors, layouts = [], [], []
    for layout in LAYOUTS:
        job_id = f"fixed-observation-{layout.lower()}-001"
        git_path = f"results/jobs/{job_id}/publish/fixed_observation_job_receipt.json"
        payload = git_blob(git_path)
        digest = hashlib.sha256(payload).hexdigest()
        source = {
            "layout_pair_id": layout,
            "git_path": git_path,
            "bytes": len(payload),
            "sha256": digest,
        }
        sources.append(source)
        value = load_json(payload, git_path)
        require(
            value.get("schema_version") == "wmf-forecast-layout-fixed-observation-job-v1"
            and value.get("study_namespace") == NAMESPACE
            and value.get("layout_pair_id") == layout
            and value.get("job_id") == job_id,
            f"Receipt identity mismatch: {layout}",
        )
        require(
            value.get("status") == "passed"
            and value.get("child_started") is True
            and value.get("exit_code") == 0
            and value.get("child_exit_code") == 0,
            f"Receipt did not pass: {layout}",
        )
        require(
            value.get("model_request_count") == 0
            and value.get("behavioral_action_count") == 0,
            f"Capture created policy activity: {layout}",
        )
        require(value.get("candidate_id") == expected_candidate(layout), f"Candidate changed: {layout}")
        require(value.get("environment_seed") == expected_seed(layout), f"Seed changed: {layout}")
        expected_commit = PILOT_COMMIT if layout == "P00" else LAYOUT_COMMIT
        require(value.get("study_commit") == expected_commit, f"Study commit changed: {layout}")
        require(
            value.get("claim_boundary")
            == f"One current {layout} physical observation and exact preprocessing only; no model request or behavioral episode.",
            f"Claim boundary changed: {layout}",
        )
        for field in ("gate_receipt_sha256", "pose_manifest_sha256"):
            require(
                type(value.get(field)) is str and SHA256_RE.fullmatch(value[field]),
                f"Invalid {field}: {layout}",
            )
        gpu = value.get("gpu_identity")
        require(
            type(gpu) is dict
            and gpu.get("name") == "NVIDIA B200"
            and gpu.get("preexisting_compute_process_count") == 0
            and type(gpu.get("uuid")) is str,
            f"Historical GPU identity changed: {layout}",
        )
        models = value.get("model_fixtures")
        require(type(models) is dict and set(models) == {"D1", "N3"}, f"Fixture set changed: {layout}")
        require(models["D1"].get("array_keys") == D1_ARRAY_KEYS, f"D1 keys changed: {layout}")
        require(models["N3"].get("array_keys") == N3_ARRAY_KEYS, f"N3 keys changed: {layout}")
        require(
            models["D1"].get("interface")
            == "official conditional DreamZero six-array request fixture"
            and models["N3"].get("interface")
            == "official packed observation/image plus joint/gripper arrays",
            f"Fixture interface changed: {layout}",
        )
        row_descriptors = [
            validate_descriptor(layout, job_id, "stderr", value["child_logs"]["stderr"], "/child.stderr.log"),
            validate_descriptor(layout, job_id, "stdout", value["child_logs"]["stdout"], "/child.stdout.log"),
            validate_descriptor(layout, job_id, "D1_fixture", models["D1"]["fixture"], "/capture/d1_official_wire_observation.npz"),
            validate_descriptor(layout, job_id, "N3_fixture", models["N3"]["fixture"], "/capture/n3_official_wire_observation.npz"),
            validate_descriptor(layout, job_id, "capture_receipt", value["raw_capture_receipt"], "/capture/capture_receipt.json"),
        ]
        descriptors.extend(row_descriptors)
        layouts.append(
            {
                "layout_pair_id": layout,
                "stage": stage(layout),
                "candidate_id": value["candidate_id"],
                "environment_seed": value["environment_seed"],
                "study_commit": value["study_commit"],
                "finished_at_utc": value["finished_at_utc"],
                "historical_gpu_uuid": gpu["uuid"],
                "gate_receipt_sha256": value["gate_receipt_sha256"],
                "pose_manifest_sha256": value["pose_manifest_sha256"],
                "compact_receipt": source,
                "raw_capture_receipt": value["raw_capture_receipt"],
                "D1_fixture": models["D1"]["fixture"],
                "N3_fixture": models["N3"]["fixture"],
            }
        )

    source_index = compact_index_bytes(sources)
    descriptor_index = compact_index_bytes(descriptors)
    require(
        hashlib.sha256(source_index).hexdigest() == EXPECTED_SOURCE_INDEX_SHA256,
        "Pinned compact-receipt index changed",
    )
    require(
        hashlib.sha256(descriptor_index).hexdigest() == EXPECTED_DESCRIPTOR_INDEX_SHA256,
        "Pinned declared-raw descriptor index changed",
    )
    require(len(sources) == len({row["git_path"] for row in sources}) == 29, "Receipt paths not unique")
    require(len(descriptors) == len({row["path"] for row in descriptors}) == 145, "Raw paths not unique")
    require(len({row["sha256"] for row in descriptors}) == 145, "Raw descriptor hashes not unique")
    require(len({row["gate_receipt_sha256"] for row in layouts}) == 29, "Gate hashes not unique")
    require(len({row["pose_manifest_sha256"] for row in layouts}) == 29, "Pose hashes not unique")
    require(len({row["environment_seed"] for row in layouts}) == 29, "Seeds not unique")
    stage_counts = Counter(row["stage"] for row in layouts)
    candidate_counts = Counter(row["candidate_id"].rsplit("_", 1)[-1] for row in layouts)
    gpu_counts = Counter(row["historical_gpu_uuid"] for row in layouts)
    descriptor_kinds = {}
    for kind in sorted({row["kind"] for row in descriptors}):
        rows = [row for row in descriptors if row["kind"] == kind]
        descriptor_kinds[kind] = {
            "count": len(rows),
            "declared_bytes": sum(row["bytes"] for row in rows),
            "unique_paths": len({row["path"] for row in rows}),
            "unique_sha256": len({row["sha256"] for row in rows}),
        }
    require(stage_counts == {"pilot": 1, "development": 4, "confirmation": 24}, "Stage counts changed")
    require(candidate_counts == {"00": 25, "01": 4}, "Candidate counts changed")
    return {
        "schema_version": "wmf-fixed-observation-capture-inventory-audit-v1",
        "results_commit": RESULTS_COMMIT,
        "scope": "All 29 accepted fixed-observation capture jobs: P00, D01-D04 and C01-C24.",
        "verification_level": "Pinned compact Git receipt bytes plus strict semantic and declared raw path/byte/SHA binding; no current raw-PVC revalidation.",
        "capture_job_accounting": {
            "jobs": len(layouts),
            "model_requests": 0,
            "behavioral_actions": 0,
        },
        "layout_inventory": {
            "total": len(layouts),
            "by_stage": dict(sorted(stage_counts.items())),
            "candidate_00": candidate_counts["00"],
            "candidate_01": candidate_counts["01"],
            "unique_environment_seeds": len({row["environment_seed"] for row in layouts}),
        },
        "compact_receipts": {
            "count": len(sources),
            "bytes": sum(row["bytes"] for row in sources),
            "unique_paths": len({row["git_path"] for row in sources}),
            "index_canonical_bytes": len(source_index),
            "index_sha256": hashlib.sha256(source_index).hexdigest(),
        },
        "declared_raw_artifacts": {
            "count": len(descriptors),
            "declared_bytes": sum(row["bytes"] for row in descriptors),
            "unique_paths": len({row["path"] for row in descriptors}),
            "unique_sha256": len({row["sha256"] for row in descriptors}),
            "index_canonical_bytes": len(descriptor_index),
            "index_sha256": hashlib.sha256(descriptor_index).hexdigest(),
            "by_kind": descriptor_kinds,
        },
        "historical_runtime_identity": {
            "gpu_model": "NVIDIA B200",
            "unique_gpu_uuids": len(gpu_counts),
            "capture_jobs_by_gpu_uuid": dict(sorted(gpu_counts.items())),
            "boundary": "Historical receipt identity only; no current worker, GPU or allocation state is inferred.",
        },
        "layouts": layouts,
        "claim_boundaries": {
            "raw_availability": "Raw paths, byte counts and SHA-256 values are declarations authenticated by compact receipts; raw PVC bytes were not read or rehashed.",
            "fixture": "One accepted physical-observation receipt and exact declared N3/D1 preprocessing fixture identity per layout; rejected candidates remain separate evidence.",
            "science": "No model request, behavioral action, policy episode, forecast score, label, resource qualification or confirmation release follows.",
        },
        "new_model_requests": 0,
        "new_behavioral_actions": 0,
        "new_behavioral_cells": 0,
        "labels_created": 0,
        "forecast_accuracy_estimated": False,
        "raw_pvc_revalidated": False,
        "confirmation_released": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, help="Verify an existing canonical audit without modifying it")
    args = parser.parse_args()
    encoded = canonical_bytes(audit())
    if args.check:
        require(args.check.read_bytes() == encoded, "Saved fixed-observation inventory audit differs")
        print("Verified 29 compact capture receipts and 145 unique declared raw artifacts; fixture evidence only, no new science.")
    else:
        print(encoded.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
