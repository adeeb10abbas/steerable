#!/usr/bin/env python3
"""Frozen queue and runtime checks shared by Cosmos3 v3 Phase-A tools."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STUDY_ID = "vla_wam_language_steerability_v3"
ARENA = "droid_robolab"
PHASE = "A_direct_command_matched_pairs"
AUTHORIZED_SEEDS = frozenset(range(8303, 8330))
MODELS = frozenset({"cosmos3_edge_policy_droid", "cosmos3_nano_policy_droid"})
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
FROZEN_GROUNDED_OBSERVATION_SHA256 = "2a431b0fa288890b3509b314c0351c91123d5f64b237678fed972848e29cd55b"
EMPTY_DIFF_SHA256 = hashlib.sha256(b"").hexdigest()
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

MODEL_CONTRACTS = {
    "cosmos3_edge_policy_droid": {
        "checkpoint_id": "nvidia/Cosmos3-Edge-Policy-DROID",
        "checkpoint_revision": "3ea407af3e156c0af3b4bb6edd85842cc9a58777",
        "server_repository_commit": "a904d2d36b774a51dd06ff9ff906816b1a04f579",
        "robolab_repository_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17",
        "v2_registry": "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_edge_droid_v2_registry.json",
        "v2_registry_sha256": "a12e4dd6668cc7c234d4b7e597885c4f04072bfd601dbc4b7a13821e2f8a49cd",
        "server_port": 18010,
        "sampling_seed_echo_required": False,
    },
    "cosmos3_nano_policy_droid": {
        "checkpoint_id": "nvidia/Cosmos3-Nano-Policy-DROID",
        "checkpoint_revision": "6706d7680581c255ff61e0f3bb49d90eac55c79e",
        "server_repository_commit": "411d25b2e35bc441126f48c44a4b93e1c0564274",
        "robolab_repository_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17",
        "v2_registry": "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_v2a011_registry.json",
        "v2_registry_sha256": "4b7fc1f7a98d73b3cd2995d32b926fc2bba4175f9b03a8131d81a78873b03eba",
        "server_port": 18011,
        "sampling_seed_echo_required": True,
    },
}

PINNED_REPO_FILES = {
    "experiments/cosmos/v2_robolab_client.py": "c9936139ee6192f6647db16a8a58a3080c5d3c5ceb64c702286c85dea2009afa",
    "experiments/cosmos/serve_robolab_without_guardrails.py": "02bc8836bd2a2ec009287487ee03e8bb810da0c1e07c94794faed84d3dc8f93b",
    "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_left_of_bowl_matched.py": "9c4d90be770266bac3ba5242b743098348c565ee622179b5e88fa2af0c4891bc",
    "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_right_of_bowl_matched.py": "ba0eb879590960c57976dd1b749c4ebbd3e86054e152ca5af014ac1bc2b6d02a",
}
ADAPTER_CONTRACT_FILES = (
    "experiments/v3/cosmos_droid/contract.py",
    "experiments/v3/cosmos_droid/client.py",
    "experiments/v3/cosmos_droid/run_pair.py",
    "experiments/v3/cosmos_droid/fixed_observation_gate.py",
    "experiments/v3/cosmos_droid/compile_pair.py",
    "experiments/v3/cosmos_droid/preflight.py",
    "experiments/v3/cosmos_droid/record_infrastructure.py",
    "experiments/v3/cosmos_droid/serve_nano.py",
)


class ContractError(ValueError):
    """Raised when a launch would depart from the registered v3 cell."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read JSON contract {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"JSON contract must be an object: {path}")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class AuthorizedPair:
    model_id: str
    seed: int
    pair_id: str
    left: dict[str, Any]
    right: dict[str, Any]
    queue_sha256: str

    def cell(self, relation: str) -> dict[str, Any]:
        if relation == "left":
            return self.left
        if relation == "right":
            return self.right
        raise ContractError("relation must be left or right")


def load_authorized_pair(study_root: Path, model_id: str, seed: int) -> AuthorizedPair:
    """Return the exact registered pair, rejecting nonlaunchable or unmatched rows."""

    study_root = Path(study_root).resolve()
    if model_id not in MODELS:
        raise ContractError(f"unsupported Cosmos model: {model_id}")
    if type(seed) is not int or seed not in AUTHORIZED_SEEDS:
        raise ContractError("v3 Cosmos Phase-A seeds are exactly 8303..8329")
    queue_path = study_root / "artifacts/vla_wam_shared_v3/phase_a_cells.jsonl"
    manifest_path = study_root / "artifacts/vla_wam_shared_v3/phase_a_cells_manifest.json"
    manifest = _load_json(manifest_path)
    observed_queue_hash = sha256_file(queue_path)
    if manifest.get("queue_sha256") != observed_queue_hash:
        raise ContractError("Phase-A queue hash does not match its committed manifest")
    launch_rule = manifest.get("launch_rule", "")
    if "status=authorized_new" not in launch_rule:
        raise ContractError("Phase-A launch rule is missing the authorized_new gate")

    rows = []
    for line_number, line in enumerate(queue_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(f"invalid queue JSON on line {line_number}: {error}") from error
        if row.get("model_id") == model_id and row.get("environment_seed") == seed:
            rows.append(row)
    if len(rows) != 2 or {row.get("relation") for row in rows} != {"left", "right"}:
        raise ContractError("registered seed does not resolve to exactly one LEFT/RIGHT pair")
    by_relation = {row["relation"]: row for row in rows}
    pair_ids = {row.get("pair_id") for row in rows}
    reset_ids = {row.get("reset_identity") for row in rows}
    runtime_requirements = {
        json.dumps(row.get("runtime_identity_requirement"), sort_keys=True) for row in rows
    }
    for relation, row in by_relation.items():
        expected_cell = f"v3:droid:{model_id}:seed{seed}:{relation}"
        expected_prompt_hash = hashlib.sha256(PROMPTS[relation].encode()).hexdigest()
        checks = {
            "study_id": STUDY_ID,
            "arena": ARENA,
            "phase": PHASE,
            "cell_id": expected_cell,
            "environment_seed": seed,
            "sampling_seed": seed,
            "status": "authorized_new",
            "execution_status": "authorized_after_all_registered_release_gates",
            "prompt": PROMPTS[relation],
            "prompt_family": "direct_command",
            "prompt_sha256": expected_prompt_hash,
            "success_predicate_id": "v2_frozen_droid_robolab_release_inside_45deg_requested_relation",
        }
        for key, expected in checks.items():
            if row.get(key) != expected:
                raise ContractError(f"queue row {expected_cell} has unexpected {key}")
    if len(pair_ids) != 1 or len(reset_ids) != 1 or len(runtime_requirements) != 1:
        raise ContractError("LEFT/RIGHT rows are not matched on pair, reset, and runtime identity")
    return AuthorizedPair(
        model_id=model_id,
        seed=seed,
        pair_id=next(iter(pair_ids)),
        left=by_relation["left"],
        right=by_relation["right"],
        queue_sha256=observed_queue_hash,
    )


def verify_repository_pins(study_root: Path, model_id: str) -> dict[str, str]:
    """Verify immutable v2 adapter/task inputs and exact model registry."""

    root = Path(study_root).resolve()
    expected = dict(PINNED_REPO_FILES)
    if model_id == "cosmos3_nano_policy_droid":
        expected["experiments/cosmos/serve_nano_robolab_v2a011.py"] = (
            "4d78af3e1fb4705b40ac36a803d49756ca6c98e512861f7f8d05b58ebc04b6f4"
        )
    spec = MODEL_CONTRACTS[model_id]
    expected[spec["v2_registry"]] = spec["v2_registry_sha256"]
    observed: dict[str, str] = {}
    for relative, expected_hash in expected.items():
        path = root / relative
        if not path.is_file():
            raise ContractError(f"missing pinned repository input: {relative}")
        observed[relative] = sha256_file(path)
        if observed[relative] != expected_hash:
            raise ContractError(f"pinned repository input changed: {relative}")
    return observed


def compute_adapter_contract_hash(study_root: Path) -> str:
    """Hash every executable v3 Cosmos adapter file in a stable inventory."""

    root = Path(study_root).resolve()
    inventory = []
    for relative in ADAPTER_CONTRACT_FILES:
        path = root / relative
        if not path.is_file():
            raise ContractError(f"missing v3 adapter file: {relative}")
        inventory.append({"path": relative, "sha256": sha256_file(path)})
    return hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_runtime_identity(
    study_root: Path, model_id: str, runtime_manifest_path: Path
) -> dict[str, Any]:
    """Verify a live-runtime manifest against the exact prior serving stack."""

    if model_id not in MODELS:
        raise ContractError(f"unsupported Cosmos model: {model_id}")
    pins = verify_repository_pins(study_root, model_id)
    runtime = _load_json(Path(runtime_manifest_path))
    spec = MODEL_CONTRACTS[model_id]
    expected_values = {
        "schema_version": "vla-wam-shared-v3-cosmos-runtime-identity-v1",
        "study_id": STUDY_ID,
        "model_id": model_id,
        "checkpoint_identifier": spec["checkpoint_id"],
        "checkpoint_revision": spec["checkpoint_revision"],
        "external_repository_commit": spec["server_repository_commit"],
        "external_repository_diff_hash": EMPTY_DIFF_SHA256,
        "simulator_repository_commit": spec["robolab_repository_commit"],
        "simulator_repository_diff_hash": EMPTY_DIFF_SHA256,
    }
    for key, expected in expected_values.items():
        if runtime.get(key) != expected:
            raise ContractError(f"runtime identity mismatch for {key}")
    for key in (
        "checkpoint_sha256", "environment_lock_hash", "adapter_contract_hash",
        "runtime_identity_sha256",
    ):
        _require_sha256(runtime.get(key), key)
    for key in ("simulator_version", "renderer_backend"):
        if not isinstance(runtime.get(key), str) or not runtime[key].strip():
            raise ContractError(f"runtime identity requires non-empty {key}")
    if runtime.get("checkpoint_hash_gate_passed") is not True:
        raise ContractError("checkpoint hash gate has not passed")
    if runtime.get("adapter_contract_hash") != compute_adapter_contract_hash(study_root):
        raise ContractError("adapter_contract_hash does not match the live v3 adapter files")
    if runtime.get("repository_pins") != pins:
        raise ContractError("runtime repository_pins do not match the committed adapter inputs")
    identity_payload = {
        key: value for key, value in runtime.items() if key != "runtime_identity_sha256"
    }
    identity_hash = hashlib.sha256(
        json.dumps(identity_payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if runtime["runtime_identity_sha256"] != identity_hash:
        raise ContractError("runtime_identity_sha256 does not bind the manifest fields")
    return runtime


def verify_release_gate(
    release_manifest_path: Path, *, pair: AuthorizedPair, runtime_identity_sha256: str
) -> dict[str, Any]:
    """Require the model-specific repeat/sensitivity gate before behavior."""

    gate = _load_json(Path(release_manifest_path))
    expected = {
        "schema_version": "vla-wam-shared-v3-cosmos-fixed-observation-v1",
        "study_id": STUDY_ID,
        "model_id": pair.model_id,
        "status": "passed",
        "queue_sha256": pair.queue_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "conditions": ["left", "left_exact_repeat", "right"],
        "conditioning_image_sha256": FROZEN_GROUNDED_OBSERVATION_SHA256,
    }
    for key, value in expected.items():
        if gate.get(key) != value:
            raise ContractError(f"fixed-observation release gate mismatch for {key}")
    metrics = gate.get("metrics")
    if not isinstance(metrics, dict):
        raise ContractError("fixed-observation gate lacks metrics")
    if metrics.get("left_repeat_action_rms") != 0.0:
        raise ContractError("exact LEFT repeat is not action-deterministic")
    if metrics.get("left_repeat_future_pixel_mae") != 0.0:
        raise ContractError("exact LEFT repeat is not future-deterministic")
    if not isinstance(metrics.get("left_right_action_rms"), (int, float)) or metrics["left_right_action_rms"] <= 0:
        raise ContractError("LEFT/RIGHT action sensitivity did not pass")
    if not isinstance(metrics.get("left_right_future_pixel_mae"), (int, float)) or metrics["left_right_future_pixel_mae"] <= 0:
        raise ContractError("LEFT/RIGHT future sensitivity did not pass")
    return gate
