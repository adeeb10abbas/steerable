#!/usr/bin/env python3
"""Compile complete live-seed and rendered-axis evidence for a V4 DROID G2 fixture."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.droid_task_files.reset_registry import (  # noqa: E402
    MODEL_BLIND_CANDIDATE_STATUS,
    load_reset_registry,
)
from experiments.online_correction_v4.model_blind_g2 import (  # noqa: E402
    axis_review_schema,
    canonical_json_bytes,
    compile_aggregate_receipt,
    seed_receipt_schema,
    sha256_file,
)


def _load_canonical(path: Path, *, schema: str) -> dict[str, Any]:
    body = path.read_bytes()
    value = json.loads(body)
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise ValueError(f"{path}: schema differs")
    if body != canonical_json_bytes(value):
        raise ValueError(f"{path}: JSON is not canonical")
    return value


def _verify_artifact_records(value: Any, *, label: str) -> int:
    verified = 0
    if isinstance(value, Mapping):
        if {"path", "sha256", "bytes"}.issubset(value):
            path = Path(str(value["path"]))
            if not path.is_file():
                raise ValueError(f"{label}: artifact is missing: {path}")
            if sha256_file(path) != value["sha256"]:
                raise ValueError(f"{label}: artifact hash differs: {path}")
            if path.stat().st_size != value["bytes"]:
                raise ValueError(f"{label}: artifact byte count differs: {path}")
            verified += 1
        for key, item in value.items():
            verified += _verify_artifact_records(item, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            verified += _verify_artifact_records(
                item, label=f"{label}[{index}]"
            )
    return verified


def _runtime_stratum(receipt: Mapping[str, Any]) -> dict[str, Any]:
    runtime = receipt.get("runtime_identity")
    if not isinstance(runtime, Mapping):
        raise ValueError("seed receipt lacks runtime identity")
    study = runtime.get("study_checkout")
    robolab = runtime.get("robolab_checkout")
    gpu = runtime.get("gpu")
    if not all(isinstance(item, Mapping) for item in (study, robolab, gpu)):
        raise ValueError("seed receipt runtime identity is incomplete")
    return {
        "study_commit": study.get("commit"),
        "robolab_commit": robolab.get("commit"),
        "gpu_name": gpu.get("name"),
        "driver_version": gpu.get("driver_version"),
        "gate_entrypoint_sha256": runtime.get("gate_entrypoint_sha256"),
        "gate_core_sha256": runtime.get("gate_core_sha256"),
        "droid_robolab_sha256": runtime.get("droid_robolab_sha256"),
        "reset_registry_sha256": runtime.get("reset_registry_sha256"),
    }


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_json_bytes(dict(payload))
    with path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())


def compile_receipts(
    *,
    registry_path: Path,
    registry_sha256: str,
    receipts_root: Path,
    axis_review_path: Path,
    output_path: Path,
    fixture_id: str = "horizontal",
) -> dict[str, Any]:
    registry = load_reset_registry(
        registry_path=str(registry_path.resolve()),
        registry_sha256=registry_sha256,
        required_status=MODEL_BLIND_CANDIDATE_STATUS,
        expected_fixture_id=fixture_id,
    )
    receipt_paths = sorted(receipts_root.resolve().rglob("g2_seed_receipt.json"))
    if not receipt_paths:
        raise ValueError("no G2 seed receipts found")
    receipts: list[dict[str, Any]] = []
    receipt_files: dict[str, Any] = {}
    runtime_stratum: dict[str, Any] | None = None
    verified_artifact_count = 0
    for path in receipt_paths:
        receipt = _load_canonical(path, schema=seed_receipt_schema(fixture_id))
        if receipt.get("fixture_id") != fixture_id:
            raise ValueError(f"{path}: fixture differs")
        if receipt.get("reset_registry_sha256") != registry.registry_sha256:
            raise ValueError(f"{path}: reset registry hash differs")
        current_stratum = _runtime_stratum(receipt)
        if runtime_stratum is None:
            runtime_stratum = current_stratum
        elif current_stratum != runtime_stratum:
            raise ValueError(f"{path}: runtime stratum differs")
        verified_artifact_count += _verify_artifact_records(
            receipt.get("artifacts"), label=str(path)
        )
        seed = receipt.get("environment_seed")
        receipt_files[str(seed)] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        receipts.append(receipt)

    axis_path = axis_review_path.resolve()
    axis_review = _load_canonical(
        axis_path,
        schema=axis_review_schema(fixture_id),
    )
    verified_artifact_count += _verify_artifact_records(
        {
            "source_seed_receipt": axis_review.get("source_seed_receipt"),
            "source_axis_overlay": axis_review.get("source_axis_overlay"),
        },
        label=str(axis_path),
    )
    aggregate = compile_aggregate_receipt(
        expected_env_seeds=registry.positions_by_env_seed,
        seed_receipts=receipts,
        axis_review=axis_review,
        fixture_id=fixture_id,
    )
    aggregate.update(
        {
            "reset_registry": {
                "path": str(registry_path.resolve()),
                "sha256": registry.registry_sha256,
                "bytes": registry_path.resolve().stat().st_size,
                "status": registry.status,
            },
            "runtime_stratum": runtime_stratum,
            "seed_receipt_files_by_env_seed": receipt_files,
            "axis_review_file": {
                "path": str(axis_path),
                "sha256": sha256_file(axis_path),
                "bytes": axis_path.stat().st_size,
            },
            "verified_external_artifact_count": verified_artifact_count,
            "release_boundary": (
                f"A passing aggregate completes {fixture_id} G2 only. G3-G8 and a "
                "released reset registry/runtime lock remain required before policy inference."
            ),
        }
    )
    _write_exclusive(output_path.resolve(), aggregate)
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-id", default="horizontal")
    parser.add_argument("--reset-registry", type=Path, required=True)
    parser.add_argument("--reset-registry-sha256", required=True)
    parser.add_argument("--receipts-root", type=Path, required=True)
    parser.add_argument("--axis-review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compile_receipts(
        registry_path=args.reset_registry,
        registry_sha256=args.reset_registry_sha256,
        receipts_root=args.receipts_root,
        axis_review_path=args.axis_review,
        output_path=args.out,
        fixture_id=args.fixture_id,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
