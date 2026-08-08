"""Build the hashable V3-E004 layout candidate from a registered input spec.

The builder is model-blind and emits no registration on its own.  It refuses
to invent the s=0 companion policy: the input must explicitly bind the exact
B001 control and the counterfactual interpolation anchor for every companion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .layout_contract import (
    LayoutContractError,
    PoseSE2,
    SymmetryWeights,
    build_candidate,
    canonical_json_bytes,
)


INPUT_SCHEMA = "vla-wam-shared-v3e004-layout-builder-input-v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )


def _poses(value: Mapping[str, Any], label: str) -> dict[str, PoseSE2]:
    if not isinstance(value, Mapping) or not value:
        raise LayoutContractError(f"{label} must be a nonempty pose map")
    return {
        str(name): PoseSE2.from_json(pose, f"{label}.{name}")
        for name, pose in value.items()
    }


def build_from_spec(spec: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    if spec.get("schema_version") != INPUT_SCHEMA:
        raise LayoutContractError("layout builder input schema changed")
    if spec.get("registered_before_inference") is not True:
        raise LayoutContractError("layout builder input was not preregistered")
    if spec.get("model_request_count") != 0 or spec.get("behavioral_episode_count") != 0:
        raise LayoutContractError("layout builder input is not model-blind")
    source_bindings = spec.get("source_bindings")
    if not isinstance(source_bindings, Mapping) or not source_bindings:
        raise LayoutContractError("source bindings are required")
    for relative, expected_sha256 in source_bindings.items():
        path = (repo_root / str(relative)).resolve()
        try:
            path.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise LayoutContractError(f"source binding escapes repository: {relative}") from exc
        if not path.is_file() or _sha(path) != expected_sha256:
            raise LayoutContractError(f"source binding changed: {relative}")
    candidate = build_candidate(
        control_poses=_poses(spec.get("control_poses", {}), "control_poses"),
        symmetric_poses=_poses(spec.get("symmetric_poses", {}), "symmetric_poses"),
        companion_counterfactual_s0_poses=_poses(
            spec.get("companion_counterfactual_s0_poses", {}),
            "companion_counterfactual_s0_poses",
        ),
        orientation_invariant_objects=spec.get("orientation_invariant_objects", []),
        mirror_pairs=[tuple(row) for row in spec.get("mirror_pairs", [])],
        midline_objects=spec.get("midline_objects", []),
        target_object=str(spec.get("target_object", "")),
        reference_object=str(spec.get("reference_object", "")),
        expected_cameras=spec.get("expected_cameras", []),
        robot_base_xy_m=spec.get("robot_base_xy_m", []),
        weights=SymmetryWeights.from_json(spec.get("asymmetry_weights", {})),
        s0_frozen_control_attestation=spec.get("s0_frozen_control_attestation", {}),
        realisation_position_tolerance_m=spec.get("realisation_position_tolerance_m"),
        realisation_orientation_tolerance_rad=spec.get("realisation_orientation_tolerance_rad"),
    )
    output = candidate.to_json()
    output["builder_input_sha256"] = hashlib.sha256(canonical_json_bytes(spec)).hexdigest()
    output["source_bindings"] = dict(sorted((str(key), str(value)) for key, value in source_bindings.items()))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite candidate: {args.output}")
    if _sha(args.input) != args.input_sha256:
        raise SystemExit("layout builder input SHA-256 mismatch")
    value = build_from_spec(_finite_json(args.input), repo_root=args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(value))
    print(json.dumps({"candidate": str(args.output.resolve()), "sha256": _sha(args.output)}, indent=2))


if __name__ == "__main__":
    main()
