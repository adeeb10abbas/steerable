"""Fail-closed Cosmos runtime envelope for the registered V3-E004 cohort.

This module is intentionally free of Cosmos, Isaac, and RoboLab imports.  It
binds the exact committed E004 registration/candidate/queue, the immutable
Phase-A runtime identity, one new behavioral cell, and one gated live session
before a model request can cross the client/server boundary.

The queue is the seed authority.  There is deliberately no broad numeric seed
allow-list: extension seeds are accepted only when an exact new-behavior cell
with that seed exists in the committed E004 queue.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess
from typing import Any, Iterable, Mapping

import numpy as np


STUDY_ID = "vla_wam_language_steerability_v3"
AMENDMENT_ID = "V3-E004"
REGISTRATION_RELATIVE = (
    "artifacts/vla_wam_shared_v3/phase_e/"
    "symmetric_layout_cohort_v3e004/registration.json"
)
QUEUE_RELATIVE = (
    "artifacts/vla_wam_shared_v3/phase_e/"
    "symmetric_layout_cohort_v3e004/queue.jsonl"
)
CANDIDATE_RELATIVE = (
    "artifacts/vla_wam_shared_v3/phase_e/"
    "symmetric_layout_cohort_v3e004/layout/candidate.json"
)
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
ACTION_CHUNK_STEPS = 32
ACTION_DIM = 8
ACTION_CAP = 450
MAX_REQUESTS_PER_SESSION = math.ceil(ACTION_CAP / ACTION_CHUNK_STEPS)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "cosmos3_nano_policy_droid": {
        "cell_prefix": "v3e004:nano:",
        "checkpoint": "nvidia/Cosmos3-Nano-Policy-DROID",
        "checkpoint_revision": "6706d7680581c255ff61e0f3bb49d90eac55c79e",
        "checkpoint_sha256": "cf76fcba7008061ecf95ec08b1b21815a6ffcb2ae9878fa11fb64a5eafb2e246",
        "checkpoint_path": "/data/users/ali/vla_wam/checkpoints/cosmos3_nano_policy_droid",
        "phase_a_runtime_identity_sha256": "d4bc4ab7d03fd1d1041f0bcc384d34321f3bd7b16c0c4cf517b62b8a1a2160e2",
        "server_repository_commit": "411d25b2e35bc441126f48c44a4b93e1c0564274",
        "robolab_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17",
        "server_port": 18011,
        "environment_id": "cosmos3_nano_policy_droid_exact_env",
    },
    "cosmos3_edge_policy_droid": {
        "cell_prefix": "v3e004:edge:",
        "checkpoint": "nvidia/Cosmos3-Edge-Policy-DROID",
        "checkpoint_revision": "3ea407af3e156c0af3b4bb6edd85842cc9a58777",
        "checkpoint_sha256": "b58d38088b3baad884a44ff9587ba10584a573f15e2cf7b08b836336cb53e48e",
        "checkpoint_path": "/data/users/ali/vla_wam/checkpoints/cosmos3_edge_policy_droid",
        "phase_a_runtime_identity_sha256": "e92f68c02345042190a415a67e3eafbb12b35fded6d59d77074c74cb28ef1940",
        "server_repository_commit": "a904d2d36b774a51dd06ff9ff906816b1a04f579",
        "robolab_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17",
        "server_port": 18010,
        "environment_id": "cosmos3_edge_policy_droid_exact_env",
    },
}

SOURCE_RELATIVES = (
    "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/cosmos_runtime.py",
    "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/cosmos_client.py",
    "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/cosmos_server.py",
    "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/serve_nano.py",
    "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/serve_edge.py",
)

REQUEST_METADATA_FIELDS = frozenset({
    "v3e004_live_stack",
    "study_id",
    "amendment_id",
    "model_id",
    "registered_cell_id",
    "cell_sha256",
    "session_id",
    "session_sha256",
    "session_manifest_path",
    "session_manifest_sha256",
    "registration_commit",
    "registration_sha256",
    "queue_sha256",
    "candidate_sha256",
    "runtime_identity_sha256",
    "phase_a_runtime_identity_sha256",
    "request_index",
    "action_step_start",
    "model_input_sha256",
    "request_binding_sha256",
})

RESPONSE_METADATA_FIELDS = frozenset({
    "v3e004_live_stack",
    "study_id",
    "amendment_id",
    "model_id",
    "registered_cell_id",
    "cell_sha256",
    "session_id",
    "session_sha256",
    "sampling_seed",
    "request_index",
    "model_input_sha256",
    "request_binding_sha256",
    "model_output_sha256",
    "response_binding_sha256",
    "runtime_identity_sha256",
    "phase_a_runtime_identity_sha256",
})


class CosmosRuntimeError(ValueError):
    """Raised before inference when an E004 Cosmos binding differs."""


def fail(message: str) -> None:
    raise CosmosRuntimeError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CosmosRuntimeError(f"value is not finite canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda token: fail(f"{label} contains non-finite {token}"),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CosmosRuntimeError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        fail(f"{label} must be a JSON object")
    return value


def _git(root: Path, *arguments: str, text: bool = False) -> bytes | str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"cannot verify E004 registration commit: git {' '.join(arguments)} failed")
    return result.stdout


def _committed_bytes(root: Path, commit: str, relative: str) -> bytes:
    value = _git(root, "show", f"{commit}:{relative}")
    if not isinstance(value, bytes):  # defensive for the type checker
        fail("git returned text for a binary-safe registration read")
    return value


@dataclass(frozen=True)
class AuthorizedCell:
    row: dict[str, Any]
    cell_sha256: str

    @property
    def cell_id(self) -> str:
        return self.row["cell_id"]

    @property
    def model_id(self) -> str:
        return self.row["model_id"]

    @property
    def seed(self) -> int:
        return self.row["sampling_seed"]

    @property
    def relation(self) -> str:
        return self.row["relation"]

    @property
    def symmetry_level(self) -> float:
        return float(self.row["symmetry_level_s"])


@dataclass(frozen=True)
class RegistrationBundle:
    study_root: Path
    registration_commit: str
    registration: dict[str, Any]
    registration_sha256: str
    queue_sha256: str
    candidate_sha256: str
    rows: tuple[dict[str, Any], ...]
    cells: tuple[AuthorizedCell, ...]
    by_cell_id: dict[str, AuthorizedCell]

    def cell(self, cell_id: str, *, model_id: str | None = None) -> AuthorizedCell:
        try:
            cell = self.by_cell_id[cell_id]
        except KeyError as exc:
            raise CosmosRuntimeError(
                f"cell is not a new registered E004 Cosmos behavior cell: {cell_id}"
            ) from exc
        if model_id is not None and cell.model_id != model_id:
            fail(f"E004 cell {cell_id} belongs to {cell.model_id}, not {model_id}")
        return cell


def _validate_runtime_requirement(row: Mapping[str, Any], model_id: str) -> None:
    spec = MODEL_SPECS[model_id]
    runtime = row.get("runtime_identity_requirement")
    if not isinstance(runtime, Mapping):
        fail(f"{row.get('cell_id')} lacks runtime_identity_requirement")
    required = {
        "checkpoint": spec["checkpoint"],
        "checkpoint_revision": spec["checkpoint_revision"],
        "phase_a_runtime_identity_sha256": spec["phase_a_runtime_identity_sha256"],
        "action_horizon": ACTION_CHUNK_STEPS,
        "action_cap": ACTION_CAP,
    }
    for key, wanted in required.items():
        if runtime.get(key) != wanted:
            fail(f"{row.get('cell_id')} runtime requirement mismatch for {key}")
    if model_id == "cosmos3_nano_policy_droid":
        if runtime.get("server_repository_commit") != spec["server_repository_commit"]:
            fail(f"{row.get('cell_id')} Nano server repository commit changed")
        if runtime.get("robolab_commit") != spec["robolab_commit"]:
            fail(f"{row.get('cell_id')} Nano RoboLab commit changed")
    else:
        if runtime.get("checkpoint_sha256") != spec["checkpoint_sha256"]:
            fail(f"{row.get('cell_id')} Edge checkpoint hash changed")


def _validate_cell_row(
    row: dict[str, Any], *, candidate_sha256: str
) -> AuthorizedCell | None:
    model_id = row.get("model_id")
    if model_id not in MODEL_SPECS:
        return None
    cell_id = row.get("cell_id")
    relation = row.get("relation")
    seed = row.get("sampling_seed")
    level = row.get("symmetry_level_s")
    if (
        row.get("schema_version") != "vla-wam-shared-v3e004-cell-v1"
        or row.get("study_id") != STUDY_ID
        or row.get("amendment_id") != AMENDMENT_ID
        or row.get("arena") != "droid_robolab"
        or row.get("execution_mode") != "new_behavioral_episode"
        or row.get("static_episode_prompt") is not True
    ):
        # Preserved closed controls are intentionally ineligible for a new
        # request.  Any other malformed Cosmos row is an error, not a skip.
        if row.get("execution_mode") == "preserved_closed_control_evidence":
            return None
        fail(f"malformed or non-new E004 Cosmos cell: {cell_id}")
    if relation not in PROMPTS or row.get("prompt") != PROMPTS[relation]:
        fail(f"{cell_id} prompt bytes changed")
    if row.get("prompt_sha256") != sha256_bytes(PROMPTS[relation].encode("utf-8")):
        fail(f"{cell_id} prompt SHA-256 changed")
    if type(seed) is not int or row.get("environment_seed") != seed:
        fail(f"{cell_id} environment/sampling seed mismatch")
    if isinstance(level, bool) or not isinstance(level, (int, float)) or not math.isfinite(float(level)):
        fail(f"{cell_id} has invalid symmetry level")
    level = float(level)
    if level not in {0.0, 0.25, 0.5, 0.75, 1.0}:
        fail(f"{cell_id} has an unregistered symmetry level")
    if model_id == "cosmos3_edge_policy_droid" and level not in {0.0, 1.0}:
        fail(f"{cell_id} Edge level is outside its registered s=0/s=1 cohort")
    token = f"s{round(level * 100):03d}"
    prefix = MODEL_SPECS[model_id]["cell_prefix"]
    expected = f"{prefix}seed{seed}:{token}:{relation}"
    if cell_id != expected:
        fail(f"E004 Cosmos cell id changed: expected {expected}, observed {cell_id}")
    if row.get("layout_candidate_sha256") != candidate_sha256:
        fail(f"{cell_id} layout candidate hash changed")
    if row.get("success_predicate_id") != (
        "v2_frozen_droid_robolab_release_inside_45deg_requested_relation"
    ):
        fail(f"{cell_id} success predicate changed")
    _validate_runtime_requirement(row, model_id)
    return AuthorizedCell(
        row=dict(row), cell_sha256=sha256_bytes(canonical_json_bytes(row))
    )


def load_registration_bundle(
    study_root: Path, *, registration_commit: str
) -> RegistrationBundle:
    """Load only bytes present together in one exact committed registration.

    The supplied commit must be a full object id, be an ancestor of the
    checkout, and contain byte-identical registration, candidate, and queue
    files.  A dirty or subsequently rewritten registered artifact fails.
    """

    root = Path(study_root).resolve()
    if not COMMIT_RE.fullmatch(registration_commit):
        fail("V3-E004 registration commit must be a full lowercase 40-hex object id")
    resolved = str(_git(root, "rev-parse", f"{registration_commit}^{{commit}}", text=True)).strip()
    if resolved != registration_commit:
        fail("V3-E004 registration commit did not resolve exactly")
    _git(root, "merge-base", "--is-ancestor", registration_commit, "HEAD")

    committed: dict[str, bytes] = {}
    for relative in (REGISTRATION_RELATIVE, QUEUE_RELATIVE, CANDIDATE_RELATIVE):
        data = _committed_bytes(root, registration_commit, relative)
        path = root / relative
        if not path.is_file() or path.read_bytes() != data:
            fail(f"registered E004 artifact differs from commit {registration_commit}: {relative}")
        committed[relative] = data

    registration_path = root / REGISTRATION_RELATIVE
    queue_path = root / QUEUE_RELATIVE
    candidate_path = root / CANDIDATE_RELATIVE
    registration = _load_json(registration_path, "E004 registration")
    expected_registration = {
        "schema_version": "vla-wam-shared-v3e004-registration-v1",
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "status": "prospectively_registered_zero_e004_model_requests_or_behavioral_episodes",
        "model_request_count_before_registration": 0,
        "behavioral_episode_count_before_registration": 0,
        "success_predicates_frozen": True,
    }
    for key, wanted in expected_registration.items():
        if registration.get(key) != wanted:
            fail(f"E004 registration mismatch for {key}")
    queue_sha = sha256_bytes(committed[QUEUE_RELATIVE])
    candidate_sha = sha256_bytes(committed[CANDIDATE_RELATIVE])
    queue_meta = registration.get("queue", {})
    layout_meta = registration.get("layout", {})
    if (
        queue_meta.get("path") != QUEUE_RELATIVE
        or queue_meta.get("sha256") != queue_sha
        or queue_meta.get("bytes") != len(committed[QUEUE_RELATIVE])
        or layout_meta.get("candidate_path") != CANDIDATE_RELATIVE
        or layout_meta.get("candidate_sha256") != candidate_sha
    ):
        fail("E004 registration no longer binds the exact candidate/queue bytes")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        committed[QUEUE_RELATIVE].decode("utf-8").splitlines(), start=1
    ):
        if not line:
            fail(f"E004 queue contains a blank row at line {line_number}")
        try:
            row = json.loads(line, parse_constant=lambda token: fail(f"non-finite {token}"))
        except json.JSONDecodeError as exc:
            raise CosmosRuntimeError(f"invalid E004 queue row {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            fail(f"E004 queue row {line_number} is not an object")
        rows.append(row)
    if queue_meta.get("rows") != len(rows):
        fail("E004 queue row count changed")

    cells: list[AuthorizedCell] = []
    seen: set[str] = set()
    for row in rows:
        cell = _validate_cell_row(row, candidate_sha256=candidate_sha)
        if cell is None:
            continue
        if cell.cell_id in seen:
            fail(f"duplicate E004 Cosmos cell id: {cell.cell_id}")
        seen.add(cell.cell_id)
        cells.append(cell)
    if not cells or {cell.model_id for cell in cells} != set(MODEL_SPECS):
        fail("committed E004 queue lacks new Nano or Edge behavior cells")

    return RegistrationBundle(
        study_root=root,
        registration_commit=registration_commit,
        registration=registration,
        registration_sha256=sha256_bytes(committed[REGISTRATION_RELATIVE]),
        queue_sha256=queue_sha,
        candidate_sha256=candidate_sha,
        rows=tuple(rows),
        cells=tuple(cells),
        by_cell_id={cell.cell_id: cell for cell in cells},
    )


def validate_runtime_payload(payload: Mapping[str, Any], *, model_id: str) -> dict[str, Any]:
    """Validate one self-hashed runtime derived from the exact Phase-A stack."""

    if model_id not in MODEL_SPECS:
        fail(f"unsupported E004 Cosmos model: {model_id}")
    spec = MODEL_SPECS[model_id]
    expected = {
        "study_id": STUDY_ID,
        "model_id": model_id,
        "checkpoint_identifier": spec["checkpoint"],
        "checkpoint_revision": spec["checkpoint_revision"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "checkpoint_hash_gate_passed": True,
        "external_repository_commit": spec["server_repository_commit"],
        "external_repository_diff_hash": EMPTY_SHA256,
        "simulator_repository_commit": spec["robolab_commit"],
        "simulator_repository_diff_hash": EMPTY_SHA256,
    }
    for key, wanted in expected.items():
        if payload.get(key) != wanted:
            fail(f"{model_id} runtime identity mismatch for {key}")
    semantic = payload.get(
        "phase_a_runtime_identity_sha256", payload.get("runtime_identity_sha256")
    )
    if semantic != spec["phase_a_runtime_identity_sha256"]:
        fail(f"{model_id} runtime is not bound to the registered Phase-A identity")
    identity = require_sha256(payload.get("runtime_identity_sha256"), "runtime_identity_sha256")
    body = dict(payload)
    body.pop("runtime_identity_sha256")
    if identity != sha256_bytes(canonical_json_bytes(body)):
        fail(f"{model_id} runtime identity self-hash mismatch")
    environment_hash = payload.get("environment_lock_sha256", payload.get("environment_lock_hash"))
    require_sha256(environment_hash, "environment lock hash")
    return dict(payload)


def load_runtime_identity(path: Path, *, model_id: str) -> dict[str, Any]:
    return validate_runtime_payload(_load_json(path, "Cosmos runtime identity"), model_id=model_id)


def source_inventory(study_root: Path) -> tuple[list[dict[str, Any]], str]:
    root = Path(study_root).resolve()
    rows: list[dict[str, Any]] = []
    for relative in SOURCE_RELATIVES:
        path = root / relative
        if not path.is_file() or path.stat().st_size <= 0:
            fail(f"missing E004 Cosmos runtime source: {relative}")
        rows.append({
            "path": relative,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    return rows, sha256_bytes(canonical_json_bytes(rows))


def _gate_record(path: Path, *, label: str, bundle: RegistrationBundle, cell: AuthorizedCell) -> dict[str, Any]:
    path = Path(path).resolve()
    value = _load_json(path, f"{label} gate")
    lane_release = value.get("schema_version") == "vla-wam-shared-v3e004-droid-lane-release-v1"
    if lane_release:
        lane_flag = {
            "raw_write": "raw_pvc_write",
            "renderer": "renderer_viewport_video",
        }.get(label)
        if (
            lane_flag is None
            or value.get("status") != "passed_pre_request_gates_except_per_cell_live_snapshot"
            or value.get("gates", {}).get(lane_flag) is not True
        ):
            fail(f"{label} lane-release gate did not pass")
    elif value.get("passed") is not True:
        fail(f"{label} gate did not pass")
    if value.get("model_request_count", 0) != 0:
        fail(f"{label} gate was not completed before model inference")
    if value.get("behavioral_episode_count", 0) != 0:
        fail(f"{label} gate was not model-blind")
    if value.get("candidate_sha256", bundle.candidate_sha256) != bundle.candidate_sha256:
        fail(f"{label} gate candidate hash mismatch")
    if value.get("model_id", cell.model_id) != cell.model_id:
        fail(f"{label} gate model mismatch")
    if value.get("registered_cell_id", cell.cell_id) != cell.cell_id:
        fail(f"{label} gate cell mismatch")
    if label == "live_camera_reset":
        scene = value.get("scene")
        if not isinstance(scene, Mapping):
            scene = value.get("compiled_gate", {}).get("scene", {})
        if not isinstance(scene, Mapping) or not math.isclose(
            float(scene.get("symmetry_level_s", math.inf)), cell.symmetry_level, abs_tol=1e-12
        ):
            fail("live camera/reset gate symmetry level differs from the cell")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def build_session_manifest(
    *,
    bundle: RegistrationBundle,
    cell: AuthorizedCell,
    runtime: Mapping[str, Any],
    runtime_manifest_path: Path,
    session_id: str,
    attempt_id: str,
    gate_paths: Mapping[str, Path],
    initial_state_sha256: str,
) -> dict[str, Any]:
    """Build (but do not write) a zero-request, all-gates-passed session."""

    if cell.cell_id not in bundle.by_cell_id:
        fail("session cell is not part of the loaded E004 registration")
    if not isinstance(session_id, str) or not session_id.strip():
        fail("E004 Cosmos session_id must be non-empty")
    if not isinstance(attempt_id, str) or not attempt_id.strip():
        fail("E004 Cosmos attempt_id must be non-empty")
    require_sha256(initial_state_sha256, "initial_state_sha256")
    runtime = validate_runtime_payload(runtime, model_id=cell.model_id)
    runtime_path = Path(runtime_manifest_path).resolve()
    if sha256_file(runtime_path) != sha256_bytes(
        json.dumps(runtime, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    ):
        # Runtime manifests in the repository are pretty-printed, while a live
        # manifest may use canonical compact JSON.  Re-read and compare the
        # object, then bind its actual bytes rather than assuming formatting.
        if _load_json(runtime_path, "Cosmos runtime identity") != runtime:
            fail("runtime object differs from its bound manifest path")
    required_gates = {"static_layout", "live_camera_reset", "raw_write", "renderer"}
    if set(gate_paths) != required_gates:
        fail("session must bind static, live camera/reset, raw-write, and renderer gates")
    gates = {
        label: _gate_record(path, label=label, bundle=bundle, cell=cell)
        for label, path in sorted(gate_paths.items())
    }
    sources, source_sha = source_inventory(bundle.study_root)
    payload: dict[str, Any] = {
        "schema_version": "vla-wam-shared-v3e004-cosmos-session-v1",
        "status": "all_registered_model_blind_runtime_gates_passed_zero_requests",
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": cell.model_id,
        "registered_cell_id": cell.cell_id,
        "cell_sha256": cell.cell_sha256,
        "session_id": session_id,
        "attempt_id": attempt_id,
        "registration_commit": bundle.registration_commit,
        "registration_sha256": bundle.registration_sha256,
        "queue_sha256": bundle.queue_sha256,
        "candidate_sha256": bundle.candidate_sha256,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "phase_a_runtime_identity_sha256": MODEL_SPECS[cell.model_id][
            "phase_a_runtime_identity_sha256"
        ],
        "runtime_manifest": {
            "path": str(runtime_path),
            "sha256": sha256_file(runtime_path),
            "bytes": runtime_path.stat().st_size,
        },
        "model_specific_environment": MODEL_SPECS[cell.model_id]["environment_id"],
        "initial_state_sha256": initial_state_sha256,
        "gates": gates,
        "source_inventory": sources,
        "source_inventory_sha256": source_sha,
        "model_request_count_before_session": 0,
        "behavioral_episode_count_before_session": 0,
        "inference_launched": False,
    }
    payload["session_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def validate_session_manifest(
    path: Path,
    *,
    bundle: RegistrationBundle,
    cell: AuthorizedCell,
    runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    session_path = Path(path).resolve()
    session = _load_json(session_path, "E004 Cosmos session")
    supplied_sha = require_sha256(session.get("session_sha256"), "session_sha256")
    body = dict(session)
    body.pop("session_sha256")
    if supplied_sha != sha256_bytes(canonical_json_bytes(body)):
        fail("E004 Cosmos session self-hash mismatch")
    sources, source_sha = source_inventory(bundle.study_root)
    expected = {
        "schema_version": "vla-wam-shared-v3e004-cosmos-session-v1",
        "status": "all_registered_model_blind_runtime_gates_passed_zero_requests",
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": cell.model_id,
        "registered_cell_id": cell.cell_id,
        "cell_sha256": cell.cell_sha256,
        "registration_commit": bundle.registration_commit,
        "registration_sha256": bundle.registration_sha256,
        "queue_sha256": bundle.queue_sha256,
        "candidate_sha256": bundle.candidate_sha256,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "phase_a_runtime_identity_sha256": MODEL_SPECS[cell.model_id][
            "phase_a_runtime_identity_sha256"
        ],
        "model_specific_environment": MODEL_SPECS[cell.model_id]["environment_id"],
        "source_inventory": sources,
        "source_inventory_sha256": source_sha,
        "model_request_count_before_session": 0,
        "behavioral_episode_count_before_session": 0,
        "inference_launched": False,
    }
    for key, wanted in expected.items():
        if session.get(key) != wanted:
            fail(f"E004 Cosmos session mismatch for {key}")
    for key in ("session_id", "attempt_id"):
        if not isinstance(session.get(key), str) or not session[key].strip():
            fail(f"E004 Cosmos session lacks {key}")
    require_sha256(session.get("initial_state_sha256"), "session initial_state_sha256")
    runtime_record = session.get("runtime_manifest", {})
    runtime_path = Path(runtime_record.get("path", "")).resolve()
    if (
        not runtime_path.is_file()
        or runtime_record.get("sha256") != sha256_file(runtime_path)
        or runtime_record.get("bytes") != runtime_path.stat().st_size
        or _load_json(runtime_path, "session runtime identity") != dict(runtime)
    ):
        fail("session runtime-manifest proof changed")
    gates = session.get("gates", {})
    if set(gates) != {"static_layout", "live_camera_reset", "raw_write", "renderer"}:
        fail("session gate inventory changed")
    for label, record in gates.items():
        gate_path = Path(record.get("path", "")).resolve()
        expected_record = _gate_record(
            gate_path, label=label, bundle=bundle, cell=cell
        )
        if record != expected_record:
            fail(f"session {label} gate proof changed")
    return session, sha256_file(session_path)


def _hash_update(digest: "hashlib._Hash", value: Any) -> None:
    """Update a digest with a typed, deterministic representation."""

    if value is None:
        digest.update(b"N")
    elif isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
    elif isinstance(value, int) and not isinstance(value, bool):
        raw = str(value).encode("ascii")
        digest.update(b"I" + len(raw).to_bytes(8, "big") + raw)
    elif isinstance(value, float):
        if not math.isfinite(value):
            fail("non-finite float cannot be evidence-hashed")
        digest.update(b"F" + struct.pack(">d", value))
    elif isinstance(value, str):
        raw = value.encode("utf-8")
        digest.update(b"S" + len(raw).to_bytes(8, "big") + raw)
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        digest.update(b"Y" + len(raw).to_bytes(8, "big") + raw)
    elif isinstance(value, Mapping):
        digest.update(b"D" + len(value).to_bytes(8, "big"))
        if not all(isinstance(key, str) for key in value):
            fail("evidence-hashed mappings require string keys")
        for key in sorted(value):
            _hash_update(digest, key)
            _hash_update(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"L" + len(value).to_bytes(8, "big"))
        for item in value:
            _hash_update(digest, item)
    else:
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.generic):
            _hash_update(digest, value.item())
            return
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                fail("object arrays cannot be evidence-hashed")
            array = np.ascontiguousarray(value)
            dtype = array.dtype.str.encode("ascii")
            shape = canonical_json_bytes(list(array.shape))
            digest.update(
                b"A"
                + len(dtype).to_bytes(8, "big")
                + dtype
                + len(shape).to_bytes(8, "big")
                + shape
                + array.nbytes.to_bytes(8, "big")
                + array.tobytes(order="C")
            )
            return
        if hasattr(value, "model_dump"):
            _hash_update(digest, value.model_dump())
            return
        fail(f"unsupported value in evidence hash: {type(value).__name__}")


def hash_value(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_update(digest, value)
    return digest.hexdigest()


def native_request(request: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in request.items() if key not in REQUEST_METADATA_FIELDS}


def native_response(response: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in response.items() if key not in RESPONSE_METADATA_FIELDS}


def _request_binding_payload(
    *,
    bundle: RegistrationBundle,
    cell: AuthorizedCell,
    runtime: Mapping[str, Any],
    session: Mapping[str, Any],
    request_index: int,
    action_step_start: int,
    model_input_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "vla-wam-shared-v3e004-cosmos-request-binding-v1",
        "registered_cell_id": cell.cell_id,
        "cell_sha256": cell.cell_sha256,
        "session_id": session["session_id"],
        "session_sha256": session["session_sha256"],
        "registration_commit": bundle.registration_commit,
        "registration_sha256": bundle.registration_sha256,
        "queue_sha256": bundle.queue_sha256,
        "candidate_sha256": bundle.candidate_sha256,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "sampling_seed": cell.seed,
        "request_index": request_index,
        "action_step_start": action_step_start,
        "model_input_sha256": model_input_sha256,
    }


def add_request_envelope(
    native: Mapping[str, Any],
    *,
    bundle: RegistrationBundle,
    cell: AuthorizedCell,
    runtime: Mapping[str, Any],
    session: Mapping[str, Any],
    session_manifest_path: Path,
    session_manifest_sha256: str,
    request_index: int,
    action_step_start: int,
) -> dict[str, Any]:
    if REQUEST_METADATA_FIELDS.intersection(native):
        fail("native Cosmos request collides with E004 evidence metadata")
    if native.get("prompt") != cell.row["prompt"] or native.get("sampling_seed") != cell.seed:
        fail("native Cosmos request prompt/seed differs from the registered cell")
    if request_index < 0 or request_index >= MAX_REQUESTS_PER_SESSION:
        fail("E004 Cosmos request index is outside the 450-action cap")
    if action_step_start != request_index * ACTION_CHUNK_STEPS:
        fail("E004 Cosmos request is not on a contiguous 32-action boundary")
    require_sha256(session_manifest_sha256, "session_manifest_sha256")
    model_input_sha = hash_value(native)
    binding = _request_binding_payload(
        bundle=bundle,
        cell=cell,
        runtime=runtime,
        session=session,
        request_index=request_index,
        action_step_start=action_step_start,
        model_input_sha256=model_input_sha,
    )
    metadata = {
        "v3e004_live_stack": "cosmos_hash_bound_v1",
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": cell.model_id,
        "registered_cell_id": cell.cell_id,
        "cell_sha256": cell.cell_sha256,
        "session_id": session["session_id"],
        "session_sha256": session["session_sha256"],
        "session_manifest_path": str(Path(session_manifest_path).resolve()),
        "session_manifest_sha256": session_manifest_sha256,
        "registration_commit": bundle.registration_commit,
        "registration_sha256": bundle.registration_sha256,
        "queue_sha256": bundle.queue_sha256,
        "candidate_sha256": bundle.candidate_sha256,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "phase_a_runtime_identity_sha256": MODEL_SPECS[cell.model_id][
            "phase_a_runtime_identity_sha256"
        ],
        "request_index": request_index,
        "action_step_start": action_step_start,
        "model_input_sha256": model_input_sha,
        "request_binding_sha256": sha256_bytes(canonical_json_bytes(binding)),
    }
    return {**native, **metadata}


def validate_request_envelope(
    request: Mapping[str, Any],
    *,
    bundle: RegistrationBundle,
    runtime: Mapping[str, Any],
    model_id: str,
    expected_request_index: int,
    session_root: Path,
) -> tuple[AuthorizedCell, dict[str, Any], str, dict[str, Any]]:
    cell_id = request.get("registered_cell_id")
    if not isinstance(cell_id, str):
        fail("E004 Cosmos request lacks registered_cell_id")
    cell = bundle.cell(cell_id, model_id=model_id)
    session_path_value = request.get("session_manifest_path")
    if not isinstance(session_path_value, str) or not session_path_value:
        fail("E004 Cosmos request lacks session_manifest_path")
    session_path = Path(session_path_value).resolve()
    allowed = Path(session_root).resolve()
    if session_path != allowed and allowed not in session_path.parents:
        fail("E004 Cosmos session manifest escapes the configured raw session root")
    session, session_file_sha = validate_session_manifest(
        session_path, bundle=bundle, cell=cell, runtime=runtime
    )
    native = native_request(request)
    model_input_sha = hash_value(native)
    binding = _request_binding_payload(
        bundle=bundle,
        cell=cell,
        runtime=runtime,
        session=session,
        request_index=expected_request_index,
        action_step_start=expected_request_index * ACTION_CHUNK_STEPS,
        model_input_sha256=model_input_sha,
    )
    expected = {
        "v3e004_live_stack": "cosmos_hash_bound_v1",
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": model_id,
        "registered_cell_id": cell.cell_id,
        "cell_sha256": cell.cell_sha256,
        "session_id": session["session_id"],
        "session_sha256": session["session_sha256"],
        "session_manifest_sha256": session_file_sha,
        "registration_commit": bundle.registration_commit,
        "registration_sha256": bundle.registration_sha256,
        "queue_sha256": bundle.queue_sha256,
        "candidate_sha256": bundle.candidate_sha256,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "phase_a_runtime_identity_sha256": MODEL_SPECS[model_id][
            "phase_a_runtime_identity_sha256"
        ],
        "request_index": expected_request_index,
        "action_step_start": expected_request_index * ACTION_CHUNK_STEPS,
        "model_input_sha256": model_input_sha,
        "request_binding_sha256": sha256_bytes(canonical_json_bytes(binding)),
    }
    for key, wanted in expected.items():
        if request.get(key) != wanted:
            fail(f"E004 Cosmos request mismatch for {key}")
    if native.get("prompt") != cell.row["prompt"] or native.get("sampling_seed") != cell.seed:
        fail("E004 Cosmos native prompt/seed differs from its registered cell")
    if expected_request_index >= MAX_REQUESTS_PER_SESSION:
        fail("E004 Cosmos request begins beyond the 450-action cap")
    return cell, session, session_file_sha, native


def add_response_envelope(
    native: Mapping[str, Any],
    *,
    cell: AuthorizedCell,
    runtime: Mapping[str, Any],
    session: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    if RESPONSE_METADATA_FIELDS.intersection(native):
        fail("native Cosmos response collides with E004 evidence metadata")
    output_sha = hash_value(native)
    response_binding = {
        "schema_version": "vla-wam-shared-v3e004-cosmos-response-binding-v1",
        "registered_cell_id": cell.cell_id,
        "cell_sha256": cell.cell_sha256,
        "session_id": session["session_id"],
        "session_sha256": session["session_sha256"],
        "request_index": request["request_index"],
        "model_input_sha256": request["model_input_sha256"],
        "request_binding_sha256": request["request_binding_sha256"],
        "model_output_sha256": output_sha,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
    }
    metadata = {
        "v3e004_live_stack": "cosmos_hash_bound_v1",
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": cell.model_id,
        "registered_cell_id": cell.cell_id,
        "cell_sha256": cell.cell_sha256,
        "session_id": session["session_id"],
        "session_sha256": session["session_sha256"],
        "sampling_seed": cell.seed,
        "request_index": request["request_index"],
        "model_input_sha256": request["model_input_sha256"],
        "request_binding_sha256": request["request_binding_sha256"],
        "model_output_sha256": output_sha,
        "response_binding_sha256": sha256_bytes(canonical_json_bytes(response_binding)),
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "phase_a_runtime_identity_sha256": MODEL_SPECS[cell.model_id][
            "phase_a_runtime_identity_sha256"
        ],
    }
    return {**native, **metadata}


def validate_response_envelope(
    response: Mapping[str, Any],
    *,
    cell: AuthorizedCell,
    runtime: Mapping[str, Any],
    session: Mapping[str, Any],
    pending_request: Mapping[str, Any],
) -> dict[str, Any]:
    native = native_response(response)
    output_sha = hash_value(native)
    response_binding = {
        "schema_version": "vla-wam-shared-v3e004-cosmos-response-binding-v1",
        "registered_cell_id": cell.cell_id,
        "cell_sha256": cell.cell_sha256,
        "session_id": session["session_id"],
        "session_sha256": session["session_sha256"],
        "request_index": pending_request["request_index"],
        "model_input_sha256": pending_request["model_input_sha256"],
        "request_binding_sha256": pending_request["request_binding_sha256"],
        "model_output_sha256": output_sha,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
    }
    expected = {
        "v3e004_live_stack": "cosmos_hash_bound_v1",
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": cell.model_id,
        "registered_cell_id": cell.cell_id,
        "cell_sha256": cell.cell_sha256,
        "session_id": session["session_id"],
        "session_sha256": session["session_sha256"],
        "sampling_seed": cell.seed,
        "request_index": pending_request["request_index"],
        "model_input_sha256": pending_request["model_input_sha256"],
        "request_binding_sha256": pending_request["request_binding_sha256"],
        "model_output_sha256": output_sha,
        "response_binding_sha256": sha256_bytes(canonical_json_bytes(response_binding)),
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "phase_a_runtime_identity_sha256": MODEL_SPECS[cell.model_id][
            "phase_a_runtime_identity_sha256"
        ],
    }
    for key, wanted in expected.items():
        if response.get(key) != wanted:
            fail(f"E004 Cosmos response mismatch for {key}")
    return native


def ensure_exact_server_cli(model_id: str, argv: Iterable[str]) -> dict[str, Any]:
    """Check the invariant serving arguments while preserving official CLIs.

    Edge and Nano use different external commits and environments, so their
    official parsers are not forced through one synthetic argparse schema.
    This gate verifies the identity-critical options shared by the released
    Phase-A/Phase-C stacks and the Edge JSON-prompt requirement.
    """

    if model_id not in MODEL_SPECS:
        fail(f"unsupported E004 Cosmos model: {model_id}")
    tokens = list(argv)

    def option(name: str) -> str | None:
        for index, token in enumerate(tokens):
            if token == name:
                if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                    fail(f"server CLI option {name} lacks a value")
                return tokens[index + 1]
            if token.startswith(name + "="):
                return token.split("=", 1)[1]
        return None

    spec = MODEL_SPECS[model_id]
    required = {
        "--checkpoint-path": spec["checkpoint_path"],
        "--port": str(spec["server_port"]),
        "--action-chunk-size": str(ACTION_CHUNK_STEPS),
        "--action-dim": str(ACTION_DIM),
        "--action-space": "joint_pos",
    }
    for name, wanted in required.items():
        observed = option(name)
        if name == "--checkpoint-path" and observed is not None:
            observed = str(Path(observed).resolve())
        if observed != wanted:
            fail(f"{model_id} server CLI mismatch for {name}")
    if "--decode-video" not in tokens:
        fail(f"{model_id} server must retain every exposed decoded future")
    if model_id == "cosmos3_nano_policy_droid":
        if option("--hf-revision") != spec["checkpoint_revision"]:
            fail("Nano server CLI checkpoint revision changed")
        if option("--domain-name") != "droid_lerobot":
            fail("Nano server CLI domain changed")
    else:
        if str(option("--format-prompt-as-json")).lower() not in {"true", "1"}:
            fail("Edge server must preserve the released JSON prompt transport")
    return {
        "model_id": model_id,
        "checkpoint_path": spec["checkpoint_path"],
        "checkpoint_revision": spec["checkpoint_revision"],
        "server_port": spec["server_port"],
        "environment_id": spec["environment_id"],
        "decoded_future_required": True,
        "action_chunk_shape": [ACTION_CHUNK_STEPS, ACTION_DIM],
    }
