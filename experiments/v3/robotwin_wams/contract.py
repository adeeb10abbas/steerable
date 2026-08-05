#!/usr/bin/env python3
"""Frozen queue and runtime contract for v3 RoboTwin WAM replication.

The v2 runners expose SAPIEN world coordinates while the shared v3 episode
schema deliberately scores continuous fields in a robot-base frame.  A
model-blind, hash-bound world-to-robot-base transform is therefore a mandatory
release artifact.  Missing or ambiguous frame provenance is a hard error; it
must never be papered over by renaming the source axes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STUDY_ID = "vla_wam_language_steerability_v3"
ARENA = "robotwin"
BEHAVIORAL_ARENA = "robotwin_place_a2b"
PHASE = "A_direct_command_sampling_replication"
QUEUE_SCHEMA = "vla-wam-shared-v3-phase-a-cells-v1"
RUNTIME_SCHEMA = "vla-wam-shared-v3-robotwin-runtime-identity-v1"
TRANSFORM_SCHEMA = "vla-wam-shared-v3-robotwin-frame-transform-v1"
MEASUREMENT_FRAME_ID = "robot_base_object_minus_reference_xyz_m"
SOURCE_FRAME_ID = "sapien_world_xyz_m"
QUEUE_RELATIVE = Path("artifacts/vla_wam_shared_v3/phase_a_cells.jsonl")
QUEUE_MANIFEST_RELATIVE = Path(
    "artifacts/vla_wam_shared_v3/phase_a_cells_manifest.json"
)
ROBOTWIN_REGISTRY_RELATIVE = Path(
    "artifacts/vla_wam_shared_v3/robotwin_direct_registry.json"
)
V2_PROTOCOL_RELATIVE = Path("artifacts/vla_wam_shared_v2/protocol.json")
EXPECTED_QUEUE_SHA256 = "8350b98f958424b56b66e67e8c70ec3951d27f4ae257476d6f08c0aaa873cb7c"
EXPECTED_ROBOTWIN_REGISTRY_SHA256 = (
    "2a840a6eaa418980f8237f5f8ab522028d4b4453fc2133dd0c0180ac9d6be8b5"
)
EXPECTED_V3_PROTOCOL_SHA256 = (
    "0e1a6465c96178e0c768c9398fe003c6617456b5101cfe8ce068283a8a7572d2"
)
EMPTY_DIFF_SHA256 = hashlib.sha256(b"").hexdigest()
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

AUTHORIZED_PAIRS = frozenset(range(3, 10))
AUTHORIZED_REPLICATES = frozenset(range(1, 10))
RELATIONS = ("left", "right")
REQUIRED_RAW_OUTPUTS = [
    "simulator_viewport_video",
    "executed_action_trace",
    "raw_result_jsonl",
    "fixture_manifest",
    "final_state",
    "frozen_failure_stage",
    "v3_failure_taxonomy_fields",
]
RUNTIME_MUST_RECORD = {
    "external_repository_commit",
    "external_repository_diff_hash",
    "checkpoint_identifier",
    "checkpoint_sha256",
    "environment_lock_hash",
    "adapter_contract_hash",
    "simulator_version",
    "renderer_backend",
}
RELEASE_GATES = (
    "model_blind_fixture_validation",
    "exact_runtime_identity",
    "raw_video_action_jsonl_write",
    "fixed_observation_exact_repeat",
    "fixed_observation_left_right_prompt_sensitivity",
)

PAIR_FIXTURES = {
    3: ("place_a2b_right", "small woodenblock", "red playingcards box"),
    4: ("place_a2b_left", "plastic mouse", "blue stapler"),
    5: ("place_a2b_right", "box of playingcards", "rubikscube"),
    6: ("place_a2b_left", "coffee box", "red playingcards box"),
    7: ("place_a2b_right", "golden bread", "blue stapler"),
    8: ("place_a2b_left", "box with cards inside", "black phone"),
    9: ("place_a2b_right", "rubikscube", "brown woodenblock"),
}

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "efficient_wam_rt_robotwin": {
        "source_commit": "b0b6cfabcbd68d18888866e958c677ce640f0412",
        "simulator_commit": "0bd8e76fde3afcffa4b30a3e3e8f92a206aa66cc",
        "checkpoint_id": "jiajun0613/Efficient-WAM_RoboTwin/Efficient-WAM-RT",
        "wrapper_path": "experiments/robotwin_language_gate/run_gate_3090.sh",
        "wrapper_sha256": "4bf8447f817e1bb1c4ffef4374f29a17e098d4b1f054b8e6a5ba17056f3c62b7",
        "runner_path": "experiments/robotwin_language_gate/closed_loop_language_gate.py",
        "runner_sha256": "ab479fa8560c8ac50433f80fb6394f5d6a5f9aa370e1693a2210b78a4a45b028",
        "gpu_environment_variable": "EFFICIENT_WAM_GPU",
        "future_interface": "decoded_future_video",
    },
    "fastwam_robotwin": {
        "source_commit": "068d3fd70c89df3726c09893f47b75a624b20c02",
        "simulator_commit": None,
        "checkpoint_id": "yuanty/fastwam/robotwin_uncond_3cam_384.pt",
        "wrapper_path": "experiments/robotwin_language_gate/run_gate_3090.sh",
        "wrapper_sha256": "da73ed99d3fa7b6e6f6ccd1725f877b0722b3d572e7dd7cd021de95c28de4d2a",
        "runner_path": "experiments/robotwin_language_gate/closed_loop_language_gate.py",
        "runner_sha256": "77f40086ec319c34fe41a0c208b5155ff0f297f6b615668ae95d695c3f0541dc",
        "gpu_environment_variable": "FASTWAM_GPU",
        "future_interface": "action_only_not_applicable",
    },
    "lingbot_va_robotwin": {
        "source_commit": "d42efbc04e502057dab4b18bb14770cc48e85131",
        "simulator_commit": None,
        "checkpoint_id": "lerobot/lingbot_va_robotwin",
        "wrapper_path": "experiments/lingbot_language_gate/run_gate_3090.sh",
        "wrapper_sha256": "e09aa0c3125e4d95ff5f0c34a21a87b8bde20b96bd058d065fa0536b5e4d1fce",
        "runner_path": "experiments/lingbot_language_gate/closed_loop_language_gate.py",
        "runner_sha256": "9b500a3d2f9910cee5add5dc1f3bf7129602c8805e700f728de9afc78480186d",
        "gpu_environment_variable": "LINGBOT_GPU",
        "future_interface": "latent_only_future_not_decodable",
    },
}


class AdapterError(ValueError):
    """Raised before a launch or compile can depart from the frozen contract."""


def _fail(message: str) -> None:
    raise AdapterError(message)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _require_sha256(value: Any, name: str) -> str:
    if not _is_sha256(value):
        _fail(f"{name} must be a lowercase SHA-256 digest")
    return value


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AdapterError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        _fail(f"{path} must contain one JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError as error:
        raise AdapterError(f"cannot read JSONL {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            _fail(f"blank JSONL row at {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise AdapterError(f"invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            _fail(f"{path}:{line_number} must contain an object")
        rows.append(row)
    return rows


def _git_output(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        text=False,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.decode(errors="replace").strip()
        _fail(f"cannot verify Git repository {repository}: {detail}")
    return completed.stdout


def _verify_live_git(repository: Path, *, commit: str, diff_hash: str) -> None:
    if not repository.is_dir():
        _fail(f"live Git repository is missing: {repository}")
    observed_commit = _git_output(repository, "rev-parse", "HEAD").decode().strip()
    if observed_commit != commit:
        _fail(f"live Git commit mismatch at {repository}")
    observed_diff = _git_output(
        repository, "diff", "--no-ext-diff", "--binary", "HEAD", "--"
    )
    if hashlib.sha256(observed_diff).hexdigest() != diff_hash:
        _fail(f"live tracked Git diff mismatch at {repository}")


def _artifact(value: Any, name: str, *, verify_file: bool = True) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{name} must be an artifact object")
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        _fail(f"{name}.path must be non-empty")
    digest = _require_sha256(value.get("sha256"), f"{name}.sha256")
    size = value.get("bytes")
    if type(size) is not int or size < 0:
        _fail(f"{name}.bytes must be a non-negative integer")
    if verify_file:
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            _fail(f"{name} is missing: {path}")
        if path.stat().st_size != size or sha256_file(path) != digest:
            _fail(f"{name} integrity mismatch: {path}")
    return value


def prompt_for(pair_number: int, relation: str) -> str:
    if pair_number not in PAIR_FIXTURES or relation not in RELATIONS:
        _fail("prompt requires pair03..09 and relation left/right")
    _, movable, reference = PAIR_FIXTURES[pair_number]
    return f"Put the {movable} to the {relation} of the {reference}."


@dataclass(frozen=True)
class AuthorizedPair:
    model_id: str
    pair_number: int
    replicate: int
    pair_id: str
    anchor_task: str
    environment_seed: int
    policy_seed: int
    reset_id: str
    left: dict[str, Any]
    right: dict[str, Any]
    queue_sha256: str

    def cell(self, relation: str) -> dict[str, Any]:
        if relation == "left":
            return self.left
        if relation == "right":
            return self.right
        _fail("relation must be left or right")


def _verify_frozen_queue_inputs(study_root: Path) -> str:
    queue_path = study_root / QUEUE_RELATIVE
    manifest = _load_object(study_root / QUEUE_MANIFEST_RELATIVE)
    queue_sha = sha256_file(queue_path)
    registry_sha = sha256_file(study_root / ROBOTWIN_REGISTRY_RELATIVE)
    protocol_sha = sha256_file(study_root / "artifacts/vla_wam_shared_v3/protocol.json")
    if queue_sha != EXPECTED_QUEUE_SHA256 or manifest.get("queue_sha256") != queue_sha:
        _fail("Phase-A queue does not match the frozen committed hash")
    if manifest.get("queue_file") != str(QUEUE_RELATIVE):
        _fail("Phase-A queue_file is not the frozen relative path")
    if manifest.get("study_id") != STUDY_ID:
        _fail("Phase-A manifest study_id mismatch")
    sources = manifest.get("source_registry_sha256")
    if not isinstance(sources, dict):
        _fail("Phase-A manifest lacks source registry hashes")
    if (
        registry_sha != EXPECTED_ROBOTWIN_REGISTRY_SHA256
        or sources.get("robotwin") != registry_sha
        or protocol_sha != EXPECTED_V3_PROTOCOL_SHA256
        or sources.get("protocol") != protocol_sha
    ):
        _fail("Phase-A queue source registry/protocol hashes do not match the freeze")
    registry = _load_object(study_root / ROBOTWIN_REGISTRY_RELATIVE)
    if (
        registry.get("schema_version")
        != "vla-wam-shared-v3-robotwin-direct-registry-v1"
        or registry.get("study_id") != STUDY_ID
        or registry.get("status")
        != "frozen_before_any_v3_robotwin_behavioral_inference"
        or registry.get("models") != list(MODEL_SPECS)
        or registry.get("scene_pairs") != list(range(3, 10))
    ):
        _fail("RoboTwin direct registry is not the frozen v3 registry")
    return queue_sha


def load_authorized_pair(
    study_root: Path, model_id: str, pair_number: int, replicate: int
) -> AuthorizedPair:
    """Resolve exactly two launchable rows; r0 is deliberately unrepresentable."""

    root = Path(study_root).resolve()
    if model_id not in MODEL_SPECS:
        _fail(f"unsupported RoboTwin v3 model: {model_id}")
    if type(pair_number) is not int or pair_number not in AUTHORIZED_PAIRS:
        _fail("v3 RoboTwin pairs are exactly pair03..pair09")
    if replicate == 0:
        _fail("replicate r0 is immutable preserved evidence and MUST NOT be rerun")
    if type(replicate) is not int or replicate not in AUTHORIZED_REPLICATES:
        _fail("new v3 RoboTwin replicates are exactly r1..r9")
    queue_sha = _verify_frozen_queue_inputs(root)
    expected_pair_id = (
        f"v3:robotwin:{model_id}:pair{pair_number:02d}:replicate{replicate:02d}"
    )
    selected = [
        row
        for row in _load_jsonl(root / QUEUE_RELATIVE)
        if row.get("pair_id") == expected_pair_id
    ]
    if len(selected) != 2 or {row.get("relation") for row in selected} != set(RELATIONS):
        _fail("registered pair must resolve to exactly one LEFT and one RIGHT row")
    by_relation = {str(row["relation"]): row for row in selected}
    anchor, movable, reference = PAIR_FIXTURES[pair_number]
    environment_seed = 4_300_000 + pair_number
    policy_seed = 8_400 + pair_number + 100 * replicate
    reset_id = (
        f"v3:robotwin:pair{pair_number:02d}:anchor_{anchor}:"
        f"environment_seed_{environment_seed}"
    )
    expected_runtime = None
    for relation in RELATIONS:
        row = by_relation[relation]
        expected_cell = f"{expected_pair_id}:{relation}"
        prompt = prompt_for(pair_number, relation)
        checks = {
            "schema_version": QUEUE_SCHEMA,
            "study_id": STUDY_ID,
            "arena": ARENA,
            "phase": PHASE,
            "model_id": model_id,
            "cell_id": expected_cell,
            "pair_id": expected_pair_id,
            "scene_pair": pair_number,
            "replicate": replicate,
            "anchor_task": anchor,
            "environment_seed": environment_seed,
            "sampling_seed": policy_seed,
            "prompt_family": "direct_command",
            "prompt_fixture_id": "v2_pairs03_09_first_seen_object_description_freeze",
            "seen_object_description": movable,
            "seen_reference_description": reference,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "relation": relation,
            "reset_identity": reset_id,
            "status": "authorized_new",
            "execution_status": "authorized_after_all_registered_release_gates",
            "success_predicate_id": (
                "v2_frozen_robotwin_relation_aware_detached_release_requested_relation"
            ),
            "required_raw_outputs": REQUIRED_RAW_OUTPUTS,
        }
        for key, expected in checks.items():
            if row.get(key) != expected:
                _fail(f"frozen queue mismatch for {expected_cell}.{key}")
        requirement = row.get("runtime_identity_requirement")
        if not isinstance(requirement, dict):
            _fail(f"missing runtime identity requirement for {expected_cell}")
        if (
            requirement.get("model_id") != model_id
            or requirement.get("left_right_must_match") is not True
            or requirement.get("requirement_id")
            != "v3:robotwin:runtime_identity_and_checkpoint_hash"
            or set(requirement.get("must_record", [])) != RUNTIME_MUST_RECORD
        ):
            _fail(f"runtime identity requirement mismatch for {expected_cell}")
        encoded_runtime = json.dumps(requirement, sort_keys=True, separators=(",", ":"))
        if expected_runtime is None:
            expected_runtime = encoded_runtime
        elif expected_runtime != encoded_runtime:
            _fail("LEFT/RIGHT runtime identity requirements differ")
    if by_relation["left"]["reset_identity"] != by_relation["right"]["reset_identity"]:
        _fail("LEFT/RIGHT do not share the exact frozen reset")
    return AuthorizedPair(
        model_id=model_id,
        pair_number=pair_number,
        replicate=replicate,
        pair_id=expected_pair_id,
        anchor_task=anchor,
        environment_seed=environment_seed,
        policy_seed=policy_seed,
        reset_id=reset_id,
        left=by_relation["left"],
        right=by_relation["right"],
        queue_sha256=queue_sha,
    )


def adapter_contract_sha256(study_root: Path, model_id: str) -> str:
    if model_id not in MODEL_SPECS:
        _fail(f"unsupported RoboTwin v3 model: {model_id}")
    root = Path(study_root).resolve()
    contract = {
        "study_id": STUDY_ID,
        "model_id": model_id,
        "queue_sha256": _verify_frozen_queue_inputs(root),
        "v2_protocol_sha256": sha256_file(root / V2_PROTOCOL_RELATIVE),
        "model_spec": MODEL_SPECS[model_id],
        "measurement_source_frame": SOURCE_FRAME_ID,
        "measurement_target_frame": MEASUREMENT_FRAME_ID,
        "behavioral_schema": "vla-wam-shared-v3-raw-episode-v1",
        "infrastructure_schema": "vla-wam-shared-v3-infrastructure-attempt-v1",
    }
    return canonical_sha256(contract)


def _validate_rotation(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 3:
        _fail("measurement_transform.rotation_source_to_target must be 3x3")
    rotation: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            _fail("measurement_transform.rotation_source_to_target must be 3x3")
        parsed = []
        for component in row:
            if type(component) not in {int, float} or not math.isfinite(float(component)):
                _fail("measurement transform rotation entries must be finite")
            parsed.append(float(component))
        rotation.append(parsed)
    for i in range(3):
        for j in range(3):
            dot = sum(rotation[i][k] * rotation[j][k] for k in range(3))
            expected = 1.0 if i == j else 0.0
            if not math.isclose(dot, expected, abs_tol=1e-9):
                _fail("measurement transform rotation must be orthonormal")
    determinant = (
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if not math.isclose(determinant, 1.0, abs_tol=1e-9):
        _fail("measurement transform rotation must be right-handed (determinant +1)")
    # The v2 native relation calls SAPIEN world -X LEFT.  The shared v3 frame
    # calls robot-base +Y LEFT and uses robot-base Z for pickup height.  This
    # exact planar mapping prevents a calibrated artifact from silently
    # changing either scientific convention.
    expected_rotation = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    for observed_row, expected_row in zip(rotation, expected_rotation, strict=True):
        for observed, expected in zip(observed_row, expected_row, strict=True):
            if not math.isclose(observed, expected, abs_tol=1e-9):
                _fail(
                    "measurement transform must map SAPIEN world -X to robot-base +Y "
                    "and preserve vertical Z"
                )
    return rotation


def validate_measurement_transform(
    value: Any, *, verify_artifacts: bool = True
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("runtime identity requires measurement_transform")
    checks = {
        "schema_version": TRANSFORM_SCHEMA,
        "source_frame_id": SOURCE_FRAME_ID,
        "target_frame_id": MEASUREMENT_FRAME_ID,
        "status": "passed_model_blind_before_behavior",
        "recorded_before_any_v3_behavioral_inference": True,
        "model_requests_during_validation": 0,
        "models_loaded_during_validation": 0,
    }
    for key, expected in checks.items():
        if value.get(key) != expected:
            _fail(f"measurement transform mismatch for {key}")
    _validate_rotation(value.get("rotation_source_to_target"))
    translation = value.get("translation_source_to_target_m")
    if not isinstance(translation, list) or len(translation) != 3:
        _fail("measurement transform translation must contain three values")
    if any(
        type(component) not in {int, float} or not math.isfinite(float(component))
        for component in translation
    ):
        _fail("measurement transform translation must be finite")
    _artifact(
        value.get("fixture_validation_artifact"),
        "measurement_transform.fixture_validation_artifact",
        verify_file=verify_artifacts,
    )
    transform_hash = _require_sha256(
        value.get("transform_sha256"), "measurement_transform.transform_sha256"
    )
    payload = {key: item for key, item in value.items() if key != "transform_sha256"}
    if transform_hash != canonical_sha256(payload):
        _fail("measurement_transform.transform_sha256 does not bind the transform")
    return value


def transform_xyz(transform: dict[str, Any], xyz: Any) -> list[float]:
    if not isinstance(xyz, list) or len(xyz) != 3:
        _fail("source XYZ must contain exactly three values")
    source = []
    for component in xyz:
        if type(component) not in {int, float} or not math.isfinite(float(component)):
            _fail("source XYZ values must be finite")
        source.append(float(component))
    rotation = transform["rotation_source_to_target"]
    translation = transform["translation_source_to_target_m"]
    return [
        sum(float(rotation[i][j]) * source[j] for j in range(3))
        + float(translation[i])
        for i in range(3)
    ]


def verify_runtime_identity(
    study_root: Path,
    model_id: str,
    runtime_manifest_path: Path,
    *,
    external_repository: Path,
    simulator_repository: Path,
    verify_live_files: bool = True,
) -> dict[str, Any]:
    """Verify exact model, repository, checkpoint, release, and frame identity."""

    if model_id not in MODEL_SPECS:
        _fail(f"unsupported RoboTwin v3 model: {model_id}")
    root = Path(study_root).resolve()
    external = Path(external_repository).expanduser().resolve()
    simulator = Path(simulator_repository).expanduser().resolve()
    runtime = _load_object(Path(runtime_manifest_path))
    spec = MODEL_SPECS[model_id]
    expected = {
        "schema_version": RUNTIME_SCHEMA,
        "study_id": STUDY_ID,
        "model_id": model_id,
        "status": "passed_all_registered_release_gates",
        "phase_a_queue_sha256": _verify_frozen_queue_inputs(root),
        "adapter_contract_sha256": adapter_contract_sha256(root, model_id),
    }
    for key, expected_value in expected.items():
        if runtime.get(key) != expected_value:
            _fail(f"runtime identity mismatch for {key}")
    runtime_id = runtime.get("runtime_id")
    if not isinstance(runtime_id, str) or not runtime_id.strip():
        _fail("runtime identity requires a non-empty runtime_id")
    source = runtime.get("external_repository")
    if not isinstance(source, dict):
        _fail("runtime identity requires external_repository")
    if (
        Path(str(source.get("path", ""))).expanduser().resolve() != external
        or source.get("commit") != spec["source_commit"]
        or source.get("diff_hash") != EMPTY_DIFF_SHA256
    ):
        _fail("external repository path/commit/clean-diff identity mismatch")
    simulator_source = runtime.get("simulator_repository")
    if not isinstance(simulator_source, dict):
        _fail("runtime identity requires simulator_repository")
    if Path(str(simulator_source.get("path", ""))).expanduser().resolve() != simulator:
        _fail("simulator repository path mismatch")
    if not isinstance(simulator_source.get("commit"), str) or not GIT_COMMIT_RE.fullmatch(
        simulator_source["commit"]
    ):
        _fail("simulator_repository.commit must be a full 40-character Git commit")
    if spec["simulator_commit"] and simulator_source.get("commit") != spec["simulator_commit"]:
        _fail("simulator repository commit mismatch")
    if simulator_source.get("diff_hash") != EMPTY_DIFF_SHA256:
        _fail("simulator repository must have the frozen empty diff hash")
    checkpoint = runtime.get("checkpoint")
    if not isinstance(checkpoint, dict):
        _fail("runtime identity requires checkpoint")
    if checkpoint.get("id") != spec["checkpoint_id"]:
        _fail("checkpoint identifier mismatch")
    if not isinstance(checkpoint.get("revision"), str) or not checkpoint["revision"].strip():
        _fail("checkpoint revision must be exact and non-empty")
    _require_sha256(checkpoint.get("sha256"), "checkpoint.sha256")
    if checkpoint.get("hash_gate_passed") is not True:
        _fail("checkpoint hash gate has not passed")
    _artifact(
        checkpoint.get("hash_manifest_artifact"),
        "checkpoint.hash_manifest_artifact",
        verify_file=verify_live_files,
    )
    environment = runtime.get("environment")
    if not isinstance(environment, dict):
        _fail("runtime identity requires environment")
    _artifact(
        environment.get("lock_artifact"),
        "environment.lock_artifact",
        verify_file=verify_live_files,
    )
    for key in ("simulator_version", "renderer_backend"):
        if not isinstance(runtime.get(key), str) or not runtime[key].strip():
            _fail(f"runtime identity requires non-empty {key}")
    files = runtime.get("adapter_files")
    if not isinstance(files, dict):
        _fail("runtime identity requires adapter_files")
    if verify_live_files:
        branch = _git_output(root, "branch", "--show-current").decode().strip()
        if branch != "codex/wam-language-steerability":
            _fail("study repository is not on codex/wam-language-steerability")
        if _git_output(root, "status", "--porcelain=v1", "--untracked-files=all"):
            _fail("study repository worktree must be clean before behavioral inference")
        _verify_live_git(
            external,
            commit=source["commit"],
            diff_hash=source["diff_hash"],
        )
        _verify_live_git(
            simulator,
            commit=simulator_source["commit"],
            diff_hash=simulator_source["diff_hash"],
        )
    for name in ("wrapper", "runner"):
        record = files.get(name)
        if not isinstance(record, dict):
            _fail(f"runtime adapter_files.{name} is missing")
        expected_relative = spec[f"{name}_path"]
        expected_hash = spec[f"{name}_sha256"]
        expected_path = (external / expected_relative).resolve()
        if (
            Path(str(record.get("path", ""))).expanduser().resolve() != expected_path
            or record.get("sha256") != expected_hash
        ):
            _fail(f"runtime adapter_files.{name} path/hash mismatch")
        if verify_live_files:
            if not expected_path.is_file() or sha256_file(expected_path) != expected_hash:
                _fail(f"live frozen adapter {name} is missing or changed")
            if name == "wrapper" and not expected_path.stat().st_mode & 0o111:
                _fail("live frozen adapter wrapper is not executable")
    gates = runtime.get("release_gates")
    if not isinstance(gates, dict) or set(gates) != set(RELEASE_GATES):
        _fail("runtime identity must contain exactly the registered release gates")
    for gate_name in RELEASE_GATES:
        gate = gates[gate_name]
        if not isinstance(gate, dict) or gate.get("status") != "passed":
            _fail(f"release gate {gate_name} has not passed")
        _artifact(
            gate.get("artifact"),
            f"release_gates.{gate_name}.artifact",
            verify_file=verify_live_files,
        )
    validate_measurement_transform(
        runtime.get("measurement_transform"), verify_artifacts=verify_live_files
    )
    runtime_hash = _require_sha256(
        runtime.get("runtime_identity_sha256"), "runtime_identity_sha256"
    )
    payload = {
        key: item for key, item in runtime.items() if key != "runtime_identity_sha256"
    }
    if runtime_hash != canonical_sha256(payload):
        _fail("runtime_identity_sha256 does not bind the complete manifest")
    return runtime
