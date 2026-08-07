#!/usr/bin/env python3
"""Build the independent V3-B005 behavioral release after nine probe requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.v3.cosmos_nano_lateral_sweep.live_support import (
    validate_fixed_observation_report,
    verify_behavioral_release_gate,
    verify_live_runtime_identity,
)
from experiments.v3.cosmos_nano_lateral_sweep.runtime_adapter import (
    AMENDMENT_ID,
    EXPECTED_FILENAMES,
    MODEL_ID,
    RELEASE_GATE_SCHEMA,
    STUDY_ID,
    load_release_bundle,
    sha256_file,
)


def _record(path: Path) -> dict[str, object]:
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"release input is missing or empty: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def build_release_gate(
    *,
    study_root: Path,
    manifest: Path,
    manifest_sha256: str,
    runtime_manifest: Path,
    fixed_observation_report: Path,
) -> dict[str, object]:
    release = load_release_bundle(
        manifest,
        expected_manifest_sha256=manifest_sha256,
    )
    runtime = verify_live_runtime_identity(
        runtime_manifest,
        study_root=study_root,
        release=release,
    )
    fixed = validate_fixed_observation_report(
        fixed_observation_report,
        release=release,
        runtime=runtime,
    )
    artifact_root = Path(manifest).resolve().parent
    physical_path = artifact_root / EXPECTED_FILENAMES["physical_gate"]
    return {
        "schema_version": RELEASE_GATE_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "prospective_artifact_sha256": release.hashes,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "runtime_identity": _record(runtime_manifest),
        "physical_gate": _record(physical_path),
        "physical_gate_passed": True,
        "fixed_observation_report": _record(fixed_observation_report),
        "fixed_observation_release_passed": True,
        "fixed_observation_probe_levels": fixed["probe_levels"],
        "fixed_observation_probe_sequence": fixed["probe_sequence"],
        "model_request_count_before_release": 9,
        "behavioral_episode_count_before_release": 0,
        "behavioral_release": True,
        "authorized_behavioral_cell_count": 210,
        "release_boundary": (
            "Only the exact 210 V3-B005 cells may run in frozen whole-seed order. "
            "The nine fixed-observation requests are diagnostics, not behavior."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--fixed-observation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    gate = build_release_gate(
        study_root=args.study_root,
        manifest=args.manifest,
        manifest_sha256=args.manifest_sha256,
        runtime_manifest=args.runtime_manifest,
        fixed_observation_report=args.fixed_observation_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    release = load_release_bundle(
        args.manifest,
        expected_manifest_sha256=args.manifest_sha256,
    )
    runtime = verify_live_runtime_identity(
        args.runtime_manifest,
        study_root=args.study_root,
        release=release,
    )
    verify_behavioral_release_gate(args.output, release=release, runtime=runtime)
    print(json.dumps(_record(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
