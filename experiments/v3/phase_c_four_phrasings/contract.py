#!/usr/bin/env python3
"""Fail-closed contracts for the V3-C001 four-phrasing experiment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


EXPERIMENT_ID = "V3-C001"
SCHEMA_PREFIX = "vla-wam-shared-v3c-four-phrasings"
RANDOMIZATION_NAMESPACE = "vla_wam_v3_phase_c_v3c001_joint_order_v1"
SEEDS = tuple(range(8500, 8520))
RELATIONS = ("left", "right")
PROMPT_FORMS = (
    "direct_command",
    "short_command",
    "goal_as_outcome",
    "desired_plus_negated_opposite",
)
PROMPTS = {
    "direct_command": {
        "left": "Put the Rubik's cube to the left of the bowl.",
        "right": "Put the Rubik's cube to the right of the bowl.",
    },
    "short_command": {
        "left": "Put the cube left of the bowl.",
        "right": "Put the cube right of the bowl.",
    },
    "goal_as_outcome": {
        "left": "The Rubik's cube should end up to the left of the bowl.",
        "right": "The Rubik's cube should end up to the right of the bowl.",
    },
    "desired_plus_negated_opposite": {
        "left": "Put the Rubik's cube to the left of the bowl, not to the right of the bowl.",
        "right": "Put the Rubik's cube to the right of the bowl, not to the left of the bowl.",
    },
}
MODEL_CONTRACTS = {
    "groot_n17_droid_vla": {
        "checkpoint": "nvidia/GR00T-N1.7-DROID",
        "checkpoint_revision": "05e7cc97e40dbd33b0890c35cc0214fcb0547ab5",
        "phase_a_runtime_identity_sha256": "1c9515daaae3b7298310694bd5b9eb0ecdbffb5c71df747f5e1cb0d0e711be64",
        "action_cap": 450,
        "action_horizon": 8,
        "future_contract": "action_only_no_decodable_future",
        "fixed_observation_future_required": False,
    },
    "cosmos3_edge_policy_droid": {
        "checkpoint": "nvidia/Cosmos3-Edge-Policy-DROID",
        "checkpoint_revision": "3ea407af3e156c0af3b4bb6edd85842cc9a58777",
        "phase_a_runtime_identity_sha256": "e92f68c02345042190a415a67e3eafbb12b35fded6d59d77074c74cb28ef1940",
        "action_cap": 450,
        "action_horizon": 32,
        "future_contract": "decoded_rgb_33_frames_when_exposed",
        "fixed_observation_future_required": True,
    },
    "cosmos3_nano_policy_droid": {
        "checkpoint": "nvidia/Cosmos3-Nano-Policy-DROID",
        "checkpoint_revision": "6706d7680581c255ff61e0f3bb49d90eac55c79e",
        "phase_a_runtime_identity_sha256": "d4bc4ab7d03fd1d1041f0bcc384d34321f3bd7b16c0c4cf517b62b8a1a2160e2",
        "action_cap": 450,
        "action_horizon": 32,
        "future_contract": "decoded_rgb_33_frames_when_exposed",
        "fixed_observation_future_required": True,
    },
}


class ContractError(ValueError):
    """Raised when a Phase-C artifact violates a frozen contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_sha256(prompt: str) -> str:
    return sha256_bytes(prompt.encode("utf-8"))


def randomized_conditions(model_id: str, seed: int) -> list[tuple[str, str]]:
    if model_id not in MODEL_CONTRACTS:
        raise ContractError(f"unregistered model_id: {model_id}")
    if seed not in SEEDS:
        raise ContractError(f"unregistered seed: {seed}")
    conditions = [(form, relation) for form in PROMPT_FORMS for relation in RELATIONS]
    return sorted(
        conditions,
        key=lambda item: sha256_bytes(
            "\0".join(
                (RANDOMIZATION_NAMESPACE, model_id, str(seed), item[0], item[1])
            ).encode("utf-8")
        ),
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ContractError(f"{path}:{line_number} is not an object")
                rows.append(value)
    return rows


def validate_cells(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(rows)
    if len(rows) != 480:
        raise ContractError(f"expected 480 cells, found {len(rows)}")
    expected_ids: set[str] = set()
    by_block: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        model_id = row.get("model_id")
        seed = row.get("seed")
        form = row.get("prompt_family")
        relation = row.get("relation")
        if model_id not in MODEL_CONTRACTS or seed not in SEEDS:
            raise ContractError(f"unregistered model/seed in cell: {row}")
        if form not in PROMPT_FORMS or relation not in RELATIONS:
            raise ContractError(f"unregistered prompt condition in cell: {row}")
        prompt = PROMPTS[form][relation]
        if row.get("prompt") != prompt or row.get("prompt_sha256") != prompt_sha256(prompt):
            raise ContractError(f"prompt bytes changed for {model_id}/{seed}/{form}/{relation}")
        if row.get("environment_seed") != seed or row.get("sampling_seed") != seed:
            raise ContractError(f"seed identity changed for {model_id}/{seed}")
        cell_id = f"v3c001:droid:{model_id}:seed{seed}:{form}:{relation}"
        if row.get("registered_cell_id") != cell_id or cell_id in expected_ids:
            raise ContractError(f"invalid or duplicate registered_cell_id: {row.get('registered_cell_id')}")
        expected_ids.add(cell_id)
        by_block.setdefault((model_id, seed), []).append(row)
    for (model_id, seed), block in by_block.items():
        if len(block) != 8:
            raise ContractError(f"seed block {model_id}/{seed} has {len(block)} cells")
        ordered = sorted(block, key=lambda row: row["within_seed_execution_order"])
        observed = [(row["prompt_family"], row["relation"]) for row in ordered]
        if [row["within_seed_execution_order"] for row in ordered] != list(range(1, 9)):
            raise ContractError(f"seed block {model_id}/{seed} does not use orders 1..8")
        if observed != randomized_conditions(model_id, seed):
            raise ContractError(f"seed block {model_id}/{seed} randomization changed")
    if set(by_block) != {(model, seed) for model in MODEL_CONTRACTS for seed in SEEDS}:
        raise ContractError("cell set does not contain all 60 registered seed blocks")
    return rows


@dataclass(frozen=True)
class ReleasedModel:
    model_id: str
    runtime_identity_sha256: str
    release_manifest_sha256: str


def validate_release_manifest(
    path: Path, *, model_id: str, registration_manifest_sha256: str
) -> ReleasedModel:
    release = load_json(path)
    expected = MODEL_CONTRACTS.get(model_id)
    if expected is None:
        raise ContractError(f"unregistered model_id: {model_id}")
    failures: list[str] = []

    def valid_proof(reference: Any) -> bool:
        if not isinstance(reference, dict):
            return False
        proof_path_value = reference.get("proof_path")
        proof_sha256 = reference.get("proof_sha256")
        if not isinstance(proof_path_value, str) or not isinstance(proof_sha256, str):
            return False
        proof_path = Path(proof_path_value)
        return proof_path.is_file() and len(proof_sha256) == 64 and sha256_file(proof_path) == proof_sha256

    if release.get("schema_version") != f"{SCHEMA_PREFIX}-release-v1":
        failures.append("schema_version")
    if release.get("experiment_id") != EXPERIMENT_ID or release.get("model_id") != model_id:
        failures.append("experiment/model identity")
    if release.get("registration_manifest_sha256") != registration_manifest_sha256:
        failures.append("registration manifest binding")
    runtime = release.get("runtime_identity", {})
    if runtime.get("semantic_sha256") != expected["phase_a_runtime_identity_sha256"]:
        failures.append("exact Phase-A runtime identity")
    if runtime.get("checkpoint") != expected["checkpoint"] or runtime.get("checkpoint_revision") != expected["checkpoint_revision"]:
        failures.append("checkpoint revision")
    runtime_path_value = runtime.get("path")
    runtime_file_sha256 = runtime.get("file_sha256")
    if (
        not isinstance(runtime_path_value, str)
        or not isinstance(runtime_file_sha256, str)
        or not Path(runtime_path_value).is_file()
        or sha256_file(Path(runtime_path_value)) != runtime_file_sha256
    ):
        failures.append("runtime identity file proof")
    phase_a = release.get("phase_a_direct_release", {})
    if phase_a.get("passed") is not True or phase_a.get("runtime_identity_match") is not True:
        failures.append("Phase-A direct release proof")
    if not valid_proof(phase_a):
        failures.append("Phase-A release proof file/hash")
    required_gates = (
        "prompt_byte_hash",
        "fixed_observation_exact_repeat",
        "fixed_observation_prompt_only_sensitivity",
        "raw_video_action_jsonl_state_write",
    )
    gates = release.get("gates", {})
    for gate in required_gates:
        if gates.get(gate, {}).get("passed") is not True or not valid_proof(gates.get(gate)):
            failures.append(gate)
    fixed = gates.get("fixed_observation_prompt_only_sensitivity", {})
    if sorted(fixed.get("prompt_forms", [])) != sorted(PROMPT_FORMS):
        failures.append("all four fixed-observation prompt forms")
    if release.get("behavioral_release") is not True:
        failures.append("behavioral_release")
    if failures:
        raise ContractError("Phase-C model remains unreleased: " + ", ".join(failures))
    return ReleasedModel(
        model_id=model_id,
        runtime_identity_sha256=runtime["semantic_sha256"],
        release_manifest_sha256=sha256_file(path),
    )


def select_whole_seed_blocks(
    rows: Iterable[dict[str, Any]], *, model_id: str, lane_index: int, lane_count: int
) -> list[dict[str, Any]]:
    if lane_count < 1 or not 0 <= lane_index < lane_count:
        raise ContractError("lane_index must be in [0, lane_count)")
    validated = validate_cells(rows)
    selected_seeds = {
        seed
        for seed in SEEDS
        if int(sha256_bytes(f"{EXPERIMENT_ID}\0{model_id}\0{seed}".encode())[:16], 16)
        % lane_count
        == lane_index
    }
    return [
        row
        for row in validated
        if row["model_id"] == model_id and row["seed"] in selected_seeds
    ]
