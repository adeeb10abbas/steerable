"""Bound a named DIST/history comparison under the registered native API contract."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import json
import math
from pathlib import Path

from experiments.workshops.spatial_grounding_v1.historical_root_nonmatch import RootPosition, _validated_position
from experiments.workshops.spatial_grounding_v1.historical_root_separation import (
    MetricFrameContract, prove_root_separation_nonmatch,
)
from tools.audit_sgw_historical_lineage import require
from tools.audit_sgw_pi05_reset_population import compile_audit as audit_resets
from tools.prove_sgw_r005_reset_bounds import read_bound, sha

ROOT = Path(__file__).resolve().parents[1]
INFRA = ROOT / "artifacts/workshops/spatial_grounding_v1/infrastructure"
NATIVE = INFRA / "native-metric-20260923cg"
HISTORY = INFRA / "historical-pi05-resets-20260923cf"
NATIVE_BINDING = {"bytes": 9125, "sha256": "c5d5a81504d2eec4ff6ce86ccde569ca8401dbb924539def73af93f7c36835e0"}
HISTORICAL_FRAME = "v3b002_registered_native_robot_base_root_m"
PROSPECTIVE_FRAME = "robolab_environment_local_world_axes_m"
ACTORS = ("rubiks_cube", "bowl")


def _upper_float(value: Fraction) -> float:
    result = float(value)
    require(math.isfinite(result), "roundoff bound overflowed")
    return math.nextafter(result, math.inf) if Fraction.from_float(result) < value else result


def rotation_roundoff_bound(recorded_roots: list) -> dict:
    """Bound the source's subtraction and normalized quaternion arithmetic.

    Native transforms are binary32. Budget subtraction more conservatively
    at binary16 precision, including complete flushing of its subnormal range.
    The quaternion is converted to binary64 before normalization. Finite
    nonzero binary32 quaternion components cannot overflow or underflow the
    binary64 norm calculation.

    Normalization error is bounded by gamma(16). Perturbing the displayed
    quadratic rotation polynomial contributes at most 16*d + 8*d*d in
    operator norm. Each component has fewer than 32 dependent operations;
    16*gamma(32)*(1+d)**2 bounds evaluation error in Euclidean norm, including
    the absolute magnitudes of all three vector terms.

    Invert this near-isometry bound using the observed L1 norm (an upper
    bound on its Euclidean norm), then solve the subtraction-error inequality.
    Three minimum-normal half values dominate Euclidean subnormal-flush error.
    Fractions keep the certificate exact until its outward-rounded output.
    """
    require(isinstance(recorded_roots, list) and bool(recorded_roots), "missing recorded roots")
    maximum = Fraction(0)
    for value in recorded_roots:
        position = _validated_position(value)
        require(position is not None, "invalid recorded root")
        norm_upper = sum((abs(Fraction.from_float(x)) for x in position), Fraction(0))
        require(norm_upper <= 1, "root is outside the certificate's one-metre L1 domain")
        maximum = max(maximum, norm_upper)
    unit64 = Fraction(1, 2**53)

    def gamma(count: int) -> Fraction:
        return count * unit64 / (1 - count * unit64)

    delta = gamma(16)
    rotation = 16 * delta + 8 * delta**2 + 16 * gamma(32) * (1 + delta)**2
    require(rotation < 1, "rotation error does not permit inversion")
    intermediate = maximum / (1 - rotation)
    unit16, minimum_normal16 = Fraction(1, 2**11), Fraction(1, 2**14)
    subtraction = (unit16 * intermediate + 3 * minimum_normal16) / (1 - unit16)
    total = subtraction + rotation * intermediate
    return {
        "maximum_recorded_root_l1_m": _upper_float(maximum),
        "native_transform_dtype": "binary32",
        "subtraction_budget_significand_bits": 11,
        "subnormal_flush_budget_m_per_coordinate": float(minimum_normal16),
        "rotation_arithmetic": "binary64_normalized_quaternion",
        "rotation_relative_error_upper_bound": _upper_float(rotation),
        "per_root_euclidean_error_upper_bound_m": _upper_float(total),
        "exact_error_bound_fraction": [str(total.numerator), str(total.denominator)],
        "scope": "Finite native binary32 roots/quaternion; finite recorded output with L1 norm at most1m.",
    }


def compile_proof() -> dict:
    historical = audit_resets()
    require(historical == json.loads((HISTORY / "audit.json").read_bytes()),
            "source-bound historical audit no longer reproduces")
    native = read_bound(NATIVE / "native_contract.json", NATIVE_BINDING)
    require(native["native_root_tensor_dtype"] == "float32" and native["stage_units_in_meters"] == 1.0,
            "native precision or metre contract differs")
    require(native["robolab_getter_commit"] == "0aef241fb088ca21bb4ebd24448940ed56620d17"
            and native["robolab_getter_sha256"] == "a4c12dc07673b0733c53990244e86577d16fa98888772a6c93934903f9699bbf",
            "native actor-root getter differs")
    for name, record in native["native_sources"].items():
        expected = "5.0.0.0" if name == "PhysX" else "2.2.0"
        require(record["distribution"]["version"] == expected, "native API version differs")
    for cls in ("RigidObjectData", "ArticulationData"):
        methods = native["native_sources"][cls]["methods"]
        require(methods[f"{cls}.root_pos_w"]["returns"] == ["self.root_link_pos_w"]
                and methods[f"{cls}.root_link_pos_w"]["returns"] == ["self.root_link_pose_w[:, :3]"],
                "native root aliases changed")
    runtime = json.loads((ROOT / "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/gates/runtime_identity.json").read_bytes())
    require(runtime["simulator_version"] == "Isaac Sim 5.0.0.0 / Isaac Lab 2.2.0 / RoboLab 0.2.1"
            and runtime["robolab_commit"] == native["robolab_getter_commit"],
            "historical and native registered API contracts differ")

    values = {name: read_bound(NATIVE / name, binding) for name, binding in native["files"].items()}
    candidate, capture = values["candidate.json"], values["candidate_capture.json"]
    verification = values["family_verification.json"]
    require(candidate["candidate_id"] == "DIST-CANDIDATE-SGW-DIST-DESIGN-000"
            and verification["family"] == candidate["family"] == "DIST",
            "named prospective candidate differs")
    require(verification["candidate_capture_sha256"] == native["files"]["candidate_capture.json"]["sha256"]
            and verification["materialized_candidate_sha256"] == native["files"]["candidate.json"]["sha256"],
            "original verification does not bind the native roots")
    require(capture["environment_origin_world_xyz_m"] == [0, 0, 0]
            and capture["robolab_commit"] == native["robolab_getter_commit"]
            and capture["study_source_commit"] == "e43f5b4054d1b65d192daa734fa9d99427b2eca3",
            "native capture is outside the bound zero-origin contract")
    prospective = {}
    for actor in ACTORS:
        root = capture["objects"][actor]["root_position_env_local_xyz_m"]
        require(root == candidate["object_poses"][actor]["position_m"], "candidate roots differ from capture")
        prospective[actor] = RootPosition(tuple(root), PROSPECTIVE_FRAME)
    resets = json.loads((HISTORY / "resets.json").read_bytes())["entries"]
    roundoff = rotation_roundoff_bound([
        row["sample"][key] for row in resets for key in ("object_xyz", "reference_xyz")
    ])
    frames = MetricFrameContract(
        HISTORICAL_FRAME, PROSPECTIVE_FRAME,
        roundoff["per_root_euclidean_error_upper_bound_m"], 0.0,
    )
    comparisons = []
    for row in resets:
        roots = {
            actor: RootPosition(tuple(row["sample"][key]), HISTORICAL_FRAME)
            for actor, key in zip(ACTORS, ("object_xyz", "reference_xyz"), strict=True)
        }
        result = prove_root_separation_nonmatch(roots, prospective, actor_pair=ACTORS, frames=frames)
        comparisons.append({"registered_cell_id": row["registered_cell_id"], "arm": row["arm"], **asdict(result)})
    return {
        "schema_version": "sgw-01-pi05-named-metric-separation-proof-v1",
        "producer_sha256": sha(Path(__file__).read_bytes()),
        "primitive_sha256": sha((ROOT / "experiments/workshops/spatial_grounding_v1/historical_root_separation.py").read_bytes()),
        "historical_audit_sha256": sha((HISTORY / "audit.json").read_bytes()),
        "native_contract_sha256": NATIVE_BINDING["sha256"],
        "prospective_candidate_id": candidate["candidate_id"],
        "prospective_candidate_sha256": native["files"]["candidate.json"]["sha256"],
        "frames": asdict(frames), "numerical_certificate": roundoff,
        "prospective_error_basis": "Same native root_pos_w with exact zero-origin subtraction and lossless binary32-to-JSON round trip.",
        "comparison_count": len(comparisons),
        "nonmatches_under_registered_api_contract": sum(row["status"] == "nonmatch" for row in comparisons),
        "unresolved_count": sum(row["status"] == "unresolved" for row in comparisons),
        "distinct_historical_numerical_root_pairs": historical["distinct_numerical_root_pairs"],
        "comparisons": comparisons,
        "qualification_basis": "Historically bound producer plus the registered native metre/root/precision API semantics.",
        "historical_native_import_bytes_independently_attested": False,
        "historical_population_coverage_complete": False,
        "release_permitted": False, "new_model_requests": 0, "new_behavioral_episodes": 0,
        "claim_boundary": (
            "Necessary nonmatches for one named SGW candidate against108 named final-cell post-settle states, "
            "conditional on the recorded native API contract. Current native source files match distribution "
            "RECORD; their historical imported bytes are not independently attested. No constructor, settle, "
            "preflight or infrastructure population coverage, no duplication proof and no fixture/model release."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compile_proof()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
