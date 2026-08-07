#!/usr/bin/env python3
"""Fail-closed shared contract for the released V3-B008/B009 Nano lanes.

The two amendments deliberately use different ports and fresh server processes.
This module validates their already-committed releases without authorizing any
behavior by itself.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


STUDY_ID = "vla_wam_language_steerability_v3"
MODEL_ID = "cosmos3_nano_policy_droid"
MODEL_REPOSITORY = "nvidia/Cosmos3-Nano-Policy-DROID"
CHECKPOINT_REVISION = "6706d7680581c255ff61e0f3bb49d90eac55c79e"
COSMOS_REPOSITORY_COMMIT = "411d25b2e35bc441126f48c44a4b93e1c0564274"
ROBOLAB_REPOSITORY_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
PINNED_CHECKPOINT_PATH = Path("/data/users/ali/vla_wam/checkpoints/cosmos3_nano_policy_droid")
ACTION_CHUNK_STEPS = 32
ACTION_DIM = 8
ACTION_CAP = 450
ACTION_SPACE = "joint_position_8d"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

CONFIG: dict[str, dict[str, Any]] = {
    "V3-B008": {
        "port": 18018,
        "seed_range": tuple(range(9700, 9727)),
        "arms": ("target_start_left", "target_start_center", "target_start_right"),
        "relations": ("left", "right"),
        "cells": 162,
        "cell_schema": "vla-wam-shared-v3b008-cell-v1",
        "manifest_schema": "vla-wam-shared-v3b008-release-manifest-v1",
        "manifest_sha256": "93026ce05b0b68f2d054a6b7d26c41cb711ec29bc6f92f5a1a40cd49ef26ad4d",
        "cells_sha256": "2ccc9eaa787dac0f6f5f94b2e3c91d09952daa5692da48effe55bd638032a311",
        "gate_sha256": "a256113c8a53186ad2842555804a491a2bc759f22bdb14bc5c89a44b1d88cb46",
        "cell_prefix": "v3b008:nano:start_side",
        "release_dir": "artifacts/vla_wam_shared_v3/prospective_tier_b/releases/v3b008",
        "cells_file": "v3b008_cells.jsonl",
    },
    "V3-B009": {
        "port": 18019,
        "seed_range": tuple(range(9800, 9827)),
        "arms": ("cube_target_bowl_reference", "bowl_target_cube_reference"),
        "relations": ("left", "right"),
        "cells": 108,
        "cell_schema": "vla-wam-shared-v3b009-cell-v1",
        "manifest_schema": "vla-wam-shared-v3b009-release-manifest-v1",
        "manifest_sha256": "940dc170865dc9df338590687dba658c33eb645ca2b80294c4862bf6ba18efa0",
        "cells_sha256": "ab2cda86e5cebfab23b1e2b683f023f5e94682e67041a5c304f0f435d10c52ad",
        "gate_sha256": "d102e99cda8a38d1c936488c886fe2317cddf5cf13cf7eca149a010de97314b0",
        "cell_prefix": "v3b009:nano:role_swap",
        "release_dir": "artifacts/vla_wam_shared_v3/prospective_tier_b/releases/v3b009",
        "cells_file": "v3b009_cells.jsonl",
    },
}

CONTRACT_FILES = (
    "experiments/v3/cosmos_nano_tier_b/runtime_contract.py",
    "experiments/v3/cosmos_nano_tier_b/server.py",
    "experiments/v3/cosmos_nano_tier_b/fixed_observation_gate.py",
    "experiments/v3/cosmos_nano_tier_b/build_behavioral_release_gate.py",
    "experiments/v3/cosmos_nano_tier_b/bind_runtime.py",
    "experiments/v3/cosmos_nano_tier_b/serve_v3b008_nano.py",
    "experiments/v3/cosmos_nano_tier_b/serve_v3b009_nano.py",
)


class ContractError(ValueError):
    """Raised before inference when any released binding differs."""


def fail(message: str) -> None:
    raise ContractError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not finite canonical JSON: {exc}") from exc


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
        value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda v: fail(f"non-finite {v}"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        fail(f"{label} must be a JSON object")
    return value


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _row_sha(row: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(row))


@dataclass(frozen=True)
class AuthorizedCell:
    row: dict[str, Any]
    cell_sha256: str

    @property
    def cell_id(self) -> str:
        return self.row["cell_id"]

    @property
    def seed(self) -> int:
        return self.row["seed"]

    @property
    def arm(self) -> str:
        return self.row["arm"]

    @property
    def relation(self) -> str:
        return self.row["relation"]


@dataclass(frozen=True)
class ReleaseBundle:
    amendment_id: str
    manifest_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    cells_path: Path
    cells_sha256: str
    cells: tuple[AuthorizedCell, ...]
    by_cell_id: dict[str, AuthorizedCell]

    @property
    def config(self) -> dict[str, Any]:
        return CONFIG[self.amendment_id]

    def cell(self, cell_id: str) -> AuthorizedCell:
        try:
            return self.by_cell_id[cell_id]
        except KeyError as exc:
            raise ContractError(f"cell is not in the exact {self.amendment_id} release: {cell_id}") from exc

    def probe_cell(self, arm: str, relation: str) -> AuthorizedCell:
        seed = self.config["seed_range"][0]
        cell_id = f'{self.config["cell_prefix"]}:seed{seed}:{arm}:{relation}'
        return self.cell(cell_id)

    def release_fingerprint(self, cell: AuthorizedCell) -> str:
        return sha256_bytes(canonical_json_bytes({
            "schema_version": "vla-wam-shared-v3b008-v3b009-nano-cell-fingerprint-v1",
            "study_id": STUDY_ID,
            "amendment_id": self.amendment_id,
            "release_manifest_sha256": self.manifest_sha256,
            "cells_sha256": self.cells_sha256,
            "model_blind_gate_sha256": self.config["gate_sha256"],
            "cell_id": cell.cell_id,
            "cell_sha256": cell.cell_sha256,
            "prompt": cell.row["prompt"],
            "fixture_positions_robot_base_m": cell.row["fixture_positions_robot_base_m"],
        }))


def load_release(study_root: Path, amendment_id: str, manifest_path: Path) -> ReleaseBundle:
    if amendment_id not in CONFIG:
        fail(f"unsupported amendment {amendment_id}")
    cfg = CONFIG[amendment_id]
    root = Path(study_root).resolve()
    expected_manifest = (root / cfg["release_dir"] / "release_manifest.json").resolve()
    path = Path(manifest_path).resolve()
    if path != expected_manifest:
        fail(f"{amendment_id} manifest path differs from committed release")
    if sha256_file(path) != cfg["manifest_sha256"]:
        fail(f"{amendment_id} manifest hash mismatch")
    manifest = load_json(path, "release manifest")
    if (
        manifest.get("schema_version") != cfg["manifest_schema"]
        or manifest.get("amendment_id") != amendment_id
        or manifest.get("study_id") != STUDY_ID
        or manifest.get("status") != "exact_queue_released_zero_cells_launched"
        or manifest.get("counts", {}).get("cells") != cfg["cells"]
        or manifest.get("counts", {}).get("launched") != 0
    ):
        fail(f"{amendment_id} release manifest contract mismatch")
    cells_path = (path.parent / cfg["cells_file"]).resolve()
    if sha256_file(cells_path) != cfg["cells_sha256"]:
        fail(f"{amendment_id} cells hash mismatch")
    lines = cells_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != cfg["cells"] or any(not line for line in lines):
        fail(f"{amendment_id} cells must contain exactly {cfg['cells']} non-empty rows")
    cells: list[AuthorizedCell] = []
    seen: set[str] = set()
    by_block: dict[int, list[dict[str, Any]]] = {}
    for line in lines:
        row = json.loads(line, parse_constant=lambda v: fail(f"non-finite {v}"))
        seed, arm, relation = row.get("seed"), row.get("arm"), row.get("relation")
        if (
            row.get("schema_version") != cfg["cell_schema"]
            or row.get("study_id") != STUDY_ID
            or row.get("amendment_id") != amendment_id
            or row.get("model_id") != MODEL_ID
            or seed not in cfg["seed_range"]
            or arm not in cfg["arms"]
            or relation not in cfg["relations"]
            or row.get("prompt_mode") != "static_episode_prompt"
            or row.get("behavioral_status") != "authorized_not_launched"
            or row.get("model_blind_gate_sha256") != cfg["gate_sha256"]
        ):
            fail(f"{amendment_id} cell contract mismatch: {row.get('cell_id')}")
        expected_id = f'{cfg["cell_prefix"]}:seed{seed}:{arm}:{relation}'
        if row.get("cell_id") != expected_id or expected_id in seen:
            fail(f"{amendment_id} duplicate or malformed cell id: {row.get('cell_id')}")
        seen.add(expected_id)
        by_block.setdefault(seed, []).append(row)
        cells.append(AuthorizedCell(row=row, cell_sha256=_row_sha(row)))
    expected_per_seed = len(cfg["arms"]) * len(cfg["relations"])
    if set(by_block) != set(cfg["seed_range"]):
        fail(f"{amendment_id} seed set mismatch")
    for seed, rows in by_block.items():
        if len(rows) != expected_per_seed:
            fail(f"{amendment_id} seed {seed} is not a complete matched block")
        conditions = {(row["arm"], row["relation"]) for row in rows}
        expected = {(arm, relation) for arm in cfg["arms"] for relation in cfg["relations"]}
        if conditions != expected:
            fail(f"{amendment_id} seed {seed} condition set mismatch")
        if sorted(row["execution_order_index_within_seed"] for row in rows) != list(range(expected_per_seed)):
            fail(f"{amendment_id} seed {seed} execution order mismatch")
    return ReleaseBundle(
        amendment_id=amendment_id,
        manifest_path=path,
        manifest=manifest,
        manifest_sha256=cfg["manifest_sha256"],
        cells_path=cells_path,
        cells_sha256=cfg["cells_sha256"],
        cells=tuple(cells),
        by_cell_id={cell.cell_id: cell for cell in cells},
    )


def compute_contract_sha256(study_root: Path) -> str:
    root = Path(study_root).resolve()
    inventory = []
    for relative in CONTRACT_FILES:
        path = root / relative
        if not path.is_file():
            fail(f"missing Nano Tier-B runtime source: {relative}")
        inventory.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return sha256_bytes(canonical_json_bytes(inventory))


def load_runtime(path: Path, *, study_root: Path, release: ReleaseBundle) -> dict[str, Any]:
    runtime = load_json(path, "runtime identity")
    expected = {
        "schema_version": "vla-wam-shared-v3b008-v3b009-nano-runtime-v1",
        "study_id": STUDY_ID,
        "amendment_id": release.amendment_id,
        "model_id": MODEL_ID,
        "checkpoint_identifier": MODEL_REPOSITORY,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "external_repository_commit": COSMOS_REPOSITORY_COMMIT,
        "external_repository_diff_hash": EMPTY_SHA256,
        "simulator_repository_commit": ROBOLAB_REPOSITORY_COMMIT,
        "simulator_repository_diff_hash": EMPTY_SHA256,
        "release_manifest_sha256": release.manifest_sha256,
        "cells_sha256": release.cells_sha256,
        "model_blind_gate_sha256": release.config["gate_sha256"],
        "contract_sha256": compute_contract_sha256(study_root),
        "server_port": release.config["port"],
        "action_space": ACTION_SPACE,
        "action_chunk_shape": [ACTION_CHUNK_STEPS, ACTION_DIM],
        "open_loop_horizon": ACTION_CHUNK_STEPS,
        "action_cap": ACTION_CAP,
        "instruction_controller": "static",
    }
    for key, wanted in expected.items():
        if runtime.get(key) != wanted:
            fail(f"runtime identity mismatch for {key}")
    supplied = _require_sha(runtime.get("runtime_identity_sha256"), "runtime identity")
    body = dict(runtime)
    body.pop("runtime_identity_sha256")
    if supplied != sha256_bytes(canonical_json_bytes(body)):
        fail("runtime identity self-hash mismatch")
    for key in (
        "checkpoint_sha256", "environment_lock_sha256", "phase_a_runtime_identity_sha256",
        "phase_a_runtime_manifest_sha256", "phase_a_adapter_contract_hash",
    ):
        _require_sha(runtime.get(key), key)
    return runtime


def write_runtime_identity(
    *, study_root: Path, release: ReleaseBundle, base_runtime_manifest: Path, output: Path
) -> dict[str, Any]:
    # Use an authoritative existing verifier rather than trusting copied fields.
    base_path = Path(base_runtime_manifest).resolve()
    unverified = load_json(base_path, "base Nano runtime identity")
    if unverified.get("schema_version") == "vla-wam-shared-v3b005-nano-runtime-identity-v1":
        from experiments.v3.cosmos_nano_lateral_sweep.live_support import verify_live_runtime_identity
        from experiments.v3.cosmos_nano_lateral_sweep.runtime_adapter import load_release_bundle

        b005_manifest = (
            Path(study_root).resolve()
            / "artifacts/vla_wam_shared_v3/phase_b/nano_lateral_sweep_v3b005/nano_lateral_v3b005_manifest.json"
        )
        b005_release = load_release_bundle(
            b005_manifest,
            expected_manifest_sha256="47c426f13146591d1a0bde60136e124eb5818cd8d44ef312f0f8fa82ad1623a1",
        )
        base = verify_live_runtime_identity(base_path, study_root=Path(study_root).resolve(), release=b005_release)
        phase_a_identity_sha = base["phase_a_runtime_identity_sha256"]
        phase_a_manifest_sha = base["phase_a_runtime_manifest_sha256"]
        phase_a_adapter_hash = base["phase_a_adapter_contract_hash"]
        environment_lock_sha = base["environment_lock_sha256"]
    else:
        from experiments.v3.cosmos_droid.contract import verify_runtime_identity

        base = verify_runtime_identity(Path(study_root).resolve(), MODEL_ID, base_path)
        phase_a_identity_sha = base["runtime_identity_sha256"]
        phase_a_manifest_sha = sha256_file(base_path)
        phase_a_adapter_hash = base["adapter_contract_hash"]
        environment_lock_sha = base["environment_lock_hash"]
    expected_base = {
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "checkpoint_identifier": MODEL_REPOSITORY,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_hash_gate_passed": True,
        "external_repository_commit": COSMOS_REPOSITORY_COMMIT,
        "external_repository_diff_hash": EMPTY_SHA256,
        "simulator_repository_commit": ROBOLAB_REPOSITORY_COMMIT,
        "simulator_repository_diff_hash": EMPTY_SHA256,
    }
    for key, wanted in expected_base.items():
        if base.get(key) != wanted:
            fail(f"Phase-A Nano runtime mismatch for {key}")
    payload = {
        "schema_version": "vla-wam-shared-v3b008-v3b009-nano-runtime-v1",
        "study_id": STUDY_ID,
        "amendment_id": release.amendment_id,
        "model_id": MODEL_ID,
        "checkpoint_identifier": MODEL_REPOSITORY,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": base["checkpoint_sha256"],
        "checkpoint_hash_gate_passed": True,
        "external_repository_commit": COSMOS_REPOSITORY_COMMIT,
        "external_repository_diff_hash": EMPTY_SHA256,
        "simulator_repository_commit": ROBOLAB_REPOSITORY_COMMIT,
        "simulator_repository_diff_hash": EMPTY_SHA256,
        "environment_lock_sha256": environment_lock_sha,
        "release_manifest_sha256": release.manifest_sha256,
        "cells_sha256": release.cells_sha256,
        "model_blind_gate_sha256": release.config["gate_sha256"],
        "contract_sha256": compute_contract_sha256(study_root),
        "server_port": release.config["port"],
        "action_space": ACTION_SPACE,
        "action_chunk_shape": [ACTION_CHUNK_STEPS, ACTION_DIM],
        "open_loop_horizon": ACTION_CHUNK_STEPS,
        "action_cap": ACTION_CAP,
        "instruction_controller": "static",
        "phase_a_runtime_identity_sha256": phase_a_identity_sha,
        "phase_a_runtime_manifest_sha256": phase_a_manifest_sha,
        "phase_a_adapter_contract_hash": phase_a_adapter_hash,
        "simulator_version": base["simulator_version"],
        "renderer_backend": base["renderer_backend"],
    }
    payload["runtime_identity_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        fail(f"refusing to overwrite runtime identity: {output}")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def validate_server_cli(amendment_id: str, argv: list[str]) -> dict[str, Any]:
    import argparse

    cfg = CONFIG[amendment_id]
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--hf-revision", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--domain-name", required=True)
    parser.add_argument("--decode-video", action="store_true")
    parser.add_argument("--action-chunk-size", type=int, required=True)
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--action-space", required=True)
    parser.add_argument("--history-length", type=int, required=True)
    parser.add_argument("--use-state", action="store_true")
    parser.add_argument("--conditioning-fps", type=int, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--guidance", type=float, required=True)
    parser.add_argument("--num-steps", type=float, required=True)
    parser.add_argument("--shift", type=float, required=True)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        raise ContractError(f"{amendment_id} Nano server CLI differs") from exc
    observed = vars(args)
    observed["checkpoint_path"] = str(args.checkpoint_path.resolve())
    expected = {
        "checkpoint_path": str(PINNED_CHECKPOINT_PATH), "hf_revision": CHECKPOINT_REVISION,
        "host": "0.0.0.0", "port": cfg["port"], "domain_name": "droid_lerobot",
        "decode_video": True, "action_chunk_size": 32, "action_dim": 8,
        "action_space": "joint_pos", "history_length": 1, "use_state": True,
        "conditioning_fps": 15, "resolution": 480, "guidance": 3.0,
        "num_steps": 4.0, "shift": 5.0,
    }
    if observed != expected:
        changed = [key for key, wanted in expected.items() if observed.get(key) != wanted]
        fail(f"{amendment_id} Nano server CLI mismatch for {', '.join(changed)}")
    return observed


def validate_behavioral_release_gate(path: Path, *, release: ReleaseBundle, runtime: Mapping[str, Any]) -> dict[str, Any]:
    gate = load_json(path, "behavioral release gate")
    expected = {
        "schema_version": "vla-wam-shared-v3b008-v3b009-nano-behavioral-release-v1",
        "study_id": STUDY_ID,
        "amendment_id": release.amendment_id,
        "model_id": MODEL_ID,
        "status": "passed_behavioral_release",
        "behavioral_release": True,
        "release_manifest_sha256": release.manifest_sha256,
        "cells_sha256": release.cells_sha256,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "model_blind_gate_sha256": release.config["gate_sha256"],
    }
    for key, wanted in expected.items():
        if gate.get(key) != wanted:
            fail(f"behavioral release gate mismatch for {key}")
    fixed = gate.get("fixed_observation_gate", {})
    if fixed.get("status") != "passed" or fixed.get("behavioral_episode_count") != 0:
        fail("behavioral release gate lacks a passing zero-behavior fixed-observation gate")
    if fixed.get("model_request_count") != len(release.config["arms"]) * 3:
        fail("fixed-observation model-request count mismatch")
    if sha256_file(Path(fixed.get("path", ""))) != _require_sha(fixed.get("sha256"), "fixed gate sha"):
        fail("fixed-observation gate hash mismatch")
    return gate
