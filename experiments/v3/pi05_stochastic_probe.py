#!/usr/bin/env python3
"""V3-D001 pi0.5 effective-sampling-seed fixed-observation probe.

This script issues raw policy requests only. It never creates a simulator or
executes an action. Every response is appended and hashed before the next
request so an interrupted probe remains diagnosable and outside behavioral
denominators.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np


STUDY_ID = "vla_wam_language_steerability_v3"
REGISTRATION_ID = "V3-D001"
MODEL_ID = "pi05_current_stack_droid"
REGISTRATION_SHA256 = "899a52c79355919210d56fa8f31d944f8a373e1e184650ee8974d62acfd6c788"
PHASE_D_REGISTRY_SHA256 = "e319f8dcaefa6803ca46989313ba737834eef1dd531c1898aeee5fa816a28ad9"
SCOPE_CORRECTION_SHA256 = "b8969639a1c45f5fd8981c5e053f170a8a6ddac5ae7ffd2185e08ff40f751b9e"
RUNTIME_SCHEMA = "vla-wam-shared-v3d001-pi05-runtime-attestation-v1"
ACTION_SHAPE = (15, 8)
SEED_INDICES = tuple(range(8))
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}


class ProbeError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ProbeError(f"non-finite {value}")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProbeError(f"{label} must be an object")
    return value


def validate_inputs(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    root = args.study_root.resolve()
    registration_path = root / "artifacts/vla_wam_shared_v3/prospective_tier_b/pi05_stochastic_eligibility_v3d001.json"
    phase_d_path = root / "artifacts/vla_wam_shared_v3/stochastic_rollout_registry.json"
    correction_path = root / "artifacts/vla_wam_shared_v3/prospective_tier_b/pi05_stochastic_v3d001_eight_repeat_correction.json"
    if sha256_file(registration_path) != REGISTRATION_SHA256:
        raise ProbeError("V3-D001 registration hash changed")
    if sha256_file(phase_d_path) != PHASE_D_REGISTRY_SHA256:
        raise ProbeError("Phase-D stochastic registry hash changed")
    if sha256_file(correction_path) != SCOPE_CORRECTION_SHA256:
        raise ProbeError("V3-D001 eight-repeat correction hash changed")
    registration = load_json(registration_path, "V3-D001 registration")
    if (
        registration.get("registration_id") != REGISTRATION_ID
        or registration.get("model_id") != MODEL_ID
        or registration.get("eligibility_probe", {}).get("shared_candidate_sampling_seed_indices") != list(SEED_INDICES)
        or registration.get("eligibility_probe", {}).get("exact_prompts") != PROMPTS
        or registration.get("release_gate", {}).get("behavioral_release") is not False
        or registration.get("release_gate", {}).get("model_request_release") is not False
    ):
        raise ProbeError("V3-D001 registration contract mismatch")
    runtime = load_json(args.runtime_attestation, "runtime attestation")
    expected_runtime = {
        "schema_version": RUNTIME_SCHEMA,
        "study_id": STUDY_ID,
        "registration_id": REGISTRATION_ID,
        "model_id": MODEL_ID,
        "openpi_commit": "c23745b5ad24e98f66967ea795a07b2588ed6c79",
        "openpi_config": "pi05_droid_jointpos_polaris",
        "server_source_sha256": "cd415e3a98da977f395242c24bb8f3d3187eb4cc3bf53c5dc659d190e6934051",
        "checkpoint_manifest_sha256": "f5a56d9565f9381ccdeeaa165b0495dab6d17a81836cc7b01c5fbc6ab89e74ca",
        "checkpoint_hash_gate_passed": True,
        "registration_sha256": REGISTRATION_SHA256,
        "phase_d_registry_sha256": PHASE_D_REGISTRY_SHA256,
        "scope_correction_sha256": SCOPE_CORRECTION_SHA256,
    }
    for key, wanted in expected_runtime.items():
        if runtime.get(key) != wanted:
            raise ProbeError(f"runtime attestation mismatch for {key}")
    claimed = runtime.get("runtime_attestation_sha256")
    body = dict(runtime)
    body.pop("runtime_attestation_sha256", None)
    if claimed != sha256_bytes(canonical_json_bytes(body)):
        raise ProbeError("runtime attestation self-hash mismatch")
    if sha256_file(args.runtime_attestation) != args.runtime_attestation_sha256:
        raise ProbeError("runtime-attestation file hash mismatch")
    fixed = load_json(args.fixture_manifest, "fixed-observation manifest")
    if sha256_file(args.fixture_manifest) != args.fixture_manifest_sha256:
        raise ProbeError("fixed-observation manifest hash mismatch")
    if (
        fixed.get("schema_version") != "vla-wam-v2a010-pi05-current-fixed-observation-v1"
        or fixed.get("robolab_commit") != "0aef241fb088ca21bb4ebd24448940ed56620d17"
        or fixed.get("environment_seed") != 8300
        or fixed.get("reset_count") != 2
        or fixed.get("neutral_reset_contract") != {"left_predicate_at_reset": False, "right_predicate_at_reset": False}
    ):
        raise ProbeError("fresh fixed-observation manifest contract mismatch")
    fixture_path = Path(fixed["fixture_path"]).resolve()
    if fixture_path != args.fixture.resolve() or sha256_file(fixture_path) != fixed.get("fixture_sha256"):
        raise ProbeError("fixed-observation NPZ binding mismatch")
    with np.load(fixture_path, allow_pickle=False) as data:
        observation = {key: np.asarray(data[key]) for key in data.files}
    expected_keys = {
        "observation/exterior_image_1_left", "observation/wrist_image_left",
        "observation/joint_position", "observation/gripper_position",
    }
    if set(observation) != expected_keys:
        raise ProbeError("fixed observation has an unexpected tensor key set")
    return observation, runtime, fixed


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")


def evaluate_sample_hashes(hashes: dict[str, dict[int, list[str]]]) -> tuple[dict[str, dict[str, Any]], bool]:
    """Apply the prospectively registered within-seed repeat/cross-seed rule."""

    direction_metrics: dict[str, dict[str, Any]] = {}
    passed = True
    for relation in ("left", "right"):
        if set(hashes.get(relation, {})) != set(SEED_INDICES):
            raise ProbeError(f"{relation} sample-hash seed set mismatch")
        if any(len(values) != 2 for values in hashes[relation].values()):
            raise ProbeError(f"{relation} requires exactly two samples per seed index")
        repeat_equal = {
            str(index): values[0] == values[1] for index, values in hashes[relation].items()
        }
        representative = [values[0] for values in hashes[relation].values()]
        unique = len(set(representative))
        row = {
            "exact_repeat_bit_identical_by_seed_index": repeat_equal,
            "all_exact_repeats_bit_identical": all(repeat_equal.values()),
            "unique_raw_policy_samples_across_seed_indices": unique,
            "at_least_two_seed_indices_bitwise_distinct": unique >= 2,
            "representative_action_sha256_by_seed_index": {
                str(index): hashes[relation][index][0] for index in SEED_INDICES
            },
        }
        row["passed"] = row["all_exact_repeats_bit_identical"] and row["at_least_two_seed_indices_bitwise_distinct"]
        passed = passed and row["passed"]
        direction_metrics[relation] = row
    return direction_metrics, passed


def run(args: argparse.Namespace) -> dict[str, Any]:
    from openpi_client import websocket_client_policy

    observation, runtime, fixed = validate_inputs(args)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite V3-D001 evidence: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    raw_dir = args.output_dir / "raw_policy_samples"
    raw_dir.mkdir()
    records_path = args.output_dir / "raw_sample_records.jsonl"
    client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    metadata = client.get_server_metadata()
    expected_metadata = {
        "v2a010_sampling_contract": "required_request_field:sampling_seed",
        "v2a010_openpi_commit": "c23745b5ad24e98f66967ea795a07b2588ed6c79",
        "v2a010_config": "pi05_droid_jointpos_polaris",
    }
    for key, wanted in expected_metadata.items():
        if metadata.get(key) != wanted:
            raise ProbeError(f"live pi0.5 server metadata mismatch for {key}")
    hashes: dict[str, dict[int, list[str]]] = {
        relation: {index: [] for index in SEED_INDICES} for relation in PROMPTS
    }
    request_index = 0
    for relation in ("left", "right"):
        for seed_index in SEED_INDICES:
            for repeat in ("a", "b"):
                response = client.infer({
                    **{key: np.array(value, copy=True) for key, value in observation.items()},
                    "prompt": PROMPTS[relation],
                    "sampling_seed": seed_index,
                })
                if response.get("v2a010_sampling_seed") != seed_index:
                    raise ProbeError("live pi0.5 server did not echo exact sampling-seed index")
                action = np.asarray(response.get("actions"), dtype=np.float32)
                if action.shape != ACTION_SHAPE or not np.isfinite(action).all():
                    raise ProbeError(f"raw pi0.5 sample is not finite {ACTION_SHAPE}")
                path = raw_dir / f"{request_index:02d}_{relation}_seed{seed_index}_{repeat}.npy"
                np.save(path, action, allow_pickle=False)
                digest = sha256_file(path)
                hashes[relation][seed_index].append(digest)
                append_jsonl(records_path, {
                    "schema_version": "vla-wam-shared-v3d001-pi05-raw-sample-v1",
                    "study_id": STUDY_ID,
                    "registration_id": REGISTRATION_ID,
                    "model_id": MODEL_ID,
                    "request_index": request_index,
                    "relation": relation,
                    "prompt": PROMPTS[relation],
                    "sampling_seed_index": seed_index,
                    "exact_repeat_label": repeat,
                    "action_path": str(path.resolve()),
                    "action_sha256": digest,
                    "action_bytes": path.stat().st_size,
                    "action_shape": list(action.shape),
                    "action_dtype": str(action.dtype),
                    "behavioral_episode": False,
                })
                request_index += 1
    direction_metrics, passed = evaluate_sample_hashes(hashes)
    report = {
        "schema_version": "vla-wam-shared-v3d001-pi05-stochastic-eligibility-result-v1",
        "study_id": STUDY_ID,
        "registration_id": REGISTRATION_ID,
        "model_id": MODEL_ID,
        "status": "eligible_effective_sampling_seed" if passed else "ineligible_zero_behavior_release",
        "passed": passed,
        "model_request_count": request_index,
        "behavioral_episode_count": 0,
        "registration_sha256": REGISTRATION_SHA256,
        "phase_d_registry_sha256": PHASE_D_REGISTRY_SHA256,
        "scope_correction_sha256": SCOPE_CORRECTION_SHA256,
        "runtime_attestation": {
            "path": str(args.runtime_attestation.resolve()),
            "sha256": args.runtime_attestation_sha256,
            "runtime_attestation_sha256": runtime["runtime_attestation_sha256"],
        },
        "fresh_fixed_observation": {
            "manifest_path": str(args.fixture_manifest.resolve()),
            "manifest_sha256": args.fixture_manifest_sha256,
            "fixture_path": str(args.fixture.resolve()),
            "fixture_sha256": fixed["fixture_sha256"],
        },
        "server_metadata": metadata,
        "sampling_seed_indices": list(SEED_INDICES),
        "exact_prompts": PROMPTS,
        "request_order": "LEFT indices0..7 a,b then RIGHT indices0..7 a,b",
        "direction_metrics": direction_metrics,
        "raw_sample_records": {
            "path": str(records_path.resolve()),
            "sha256": sha256_file(records_path),
            "bytes": records_path.stat().st_size,
            "rows": request_index,
        },
        "release_boundary": (
            "Eligible result permits construction of a separate hash-bound 8-rollout-per-condition, 432-cell registry; this probe itself releases and executes zero behavior."
            if passed else
            "Ineligible result permanently closes V3-D001 behavior at zero stochastic episodes."
        ),
    }
    report_path = args.output_dir / "eligibility_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "evidence_manifest.json":
            files.append({"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manifest = {
        "schema_version": "vla-wam-shared-v3d001-pi05-stochastic-eligibility-manifest-v1",
        "study_id": STUDY_ID,
        "registration_id": REGISTRATION_ID,
        "status": report["status"],
        "model_request_count": request_index,
        "behavioral_episode_count": 0,
        "files": files,
    }
    manifest_path = args.output_dir / "evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"report": str(report_path), "report_sha256": sha256_file(report_path), "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path), "passed": passed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--runtime-attestation", type=Path, required=True)
    parser.add_argument("--runtime-attestation-sha256", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--fixture-manifest-sha256", required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except BaseException as exc:
        if args.output_dir.exists():
            failure = args.output_dir / "infrastructure_failure.json"
            if not failure.exists():
                failure.write_text(json.dumps({
                    "schema_version": "vla-wam-shared-v3d001-infrastructure-failure-v1",
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "behavioral_episode_count": 0,
                    "excluded_from_behavioral_denominators": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
