"""Fail-closed registry and runtime contract for V3-E005 LingBot-VA.

This module is deliberately model- and simulator-free.  It binds the exact
108-cell registration, preserves whole four-cell seed blocks while sharding,
and revalidates the previously released LingBot/RoboTwin runtime without
importing a DROID task or success predicate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping

from experiments.v3.robotwin_wams.contract import (
    AdapterError as PhaseAAdapterError,
    verify_runtime_identity as verify_phase_a_runtime_identity,
)


STUDY_ID = "vla_wam_language_steerability_v3"
AMENDMENT_ID = "V3-E005"
MODEL_ID = "lingbot_va_robotwin"
ARENA = "robotwin"
REGISTRATION_SCHEMA = "vla-wam-shared-v3e005-registration-v1"
CELL_SCHEMA = "vla-wam-shared-v3e005-cell-v1"
REGISTRATION_SHA256 = "6886dae4bfcc6dc5f2bcefa1e0788dccf0a9ef1cde89000998b0e63c30c745c9"
QUEUE_SHA256 = "bba7df41b6f5ee23f6460f910b1eac64f0bf20bf734b5546f2b4275b95b01786"
REGISTRATION_RELATIVE = Path(
    "artifacts/vla_wam_shared_v3/phase_e/cross_arena_geometry_v3e005/registration.json"
)
QUEUE_RELATIVE = Path(
    "artifacts/vla_wam_shared_v3/phase_e/cross_arena_geometry_v3e005/queue.jsonl"
)
EXTERNAL_COMMIT = "d42efbc04e502057dab4b18bb14770cc48e85131"
SIMULATOR_COMMIT = "0aeea2d669c0f8516f4d5785f0aa33ba812c14b4"
MODEL_BLIND_GATE_SCHEMA = "vla-wam-shared-v3e005-seven-scene-model-blind-gate-v1"
EMPTY_DIFF_SHA256 = hashlib.sha256(b"").hexdigest()
SEEDS = tuple(range(9400, 9427))
LEVELS = (0.0, 1.0)
RELATIONS = ("left", "right")
SCENES = tuple(range(3, 10))
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class E005ContractError(ValueError):
    """Raised before E005 can depart from its registered contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E005ContractError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    require(resolved.is_file(), f"required artifact is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise E005ContractError(f"cannot read JSON object {path}: {error}") from error
    require(isinstance(value, dict), f"{path} must contain one JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError as error:
        raise E005ContractError(f"cannot read JSONL {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        require(bool(line.strip()), f"blank JSONL row at {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise E005ContractError(
                f"invalid JSON at {path}:{line_number}: {error}"
            ) from error
        require(isinstance(row, dict), f"{path}:{line_number} must be an object")
        rows.append(row)
    return rows


def _git(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        detail = completed.stderr.decode(errors="replace").strip()
        raise E005ContractError(f"cannot inspect Git repository {repository}: {detail}")
    return completed.stdout


def verify_git_identity(
    repository: Path,
    expected_commit: str,
    *,
    require_untracked_clean: bool = False,
) -> None:
    resolved = Path(repository).expanduser().resolve()
    require(resolved.is_dir(), f"Git repository is missing: {resolved}")
    observed = _git(resolved, "rev-parse", "HEAD").decode().strip()
    require(observed == expected_commit, f"Git commit drift at {resolved}")
    arguments = ["status", "--porcelain=v1"]
    if not require_untracked_clean:
        arguments.append("--untracked-files=no")
    status = _git(resolved, *arguments)
    require(not status, f"Git worktree is not clean: {resolved}")


@dataclass(frozen=True)
class RegisteredCell:
    row: dict[str, Any]

    @property
    def cell_id(self) -> str:
        return str(self.row["cell_id"])

    @property
    def environment_seed(self) -> int:
        return int(self.row["environment_seed"])

    @property
    def sampling_seed(self) -> int:
        return int(self.row["sampling_seed"])

    @property
    def scene_number(self) -> int:
        return int(str(self.row["scene_id"]).rsplit("_", 1)[1])

    @property
    def scene_id(self) -> str:
        return str(self.row["scene_id"])

    @property
    def symmetry_level(self) -> float:
        return float(self.row["symmetry_level_s"])

    @property
    def relation(self) -> str:
        return str(self.row["relation"])

    @property
    def level_code(self) -> str:
        return f"s{int(round(100.0 * self.symmetry_level)):03d}"

    @property
    def prompt(self) -> str:
        return str(self.row["prompt"])

    @property
    def anchor_task(self) -> str:
        return str(self.row["anchor_task"])

    @property
    def matched_layout_pair_id(self) -> str:
        return str(self.row["matched_layout_pair_id"])


@dataclass(frozen=True)
class RegisteredBundle:
    study_root: Path
    registration_path: Path
    queue_path: Path
    registration: dict[str, Any]
    cells: tuple[RegisteredCell, ...]
    registration_sha256: str
    queue_sha256: str

    def cell(self, cell_id: str) -> RegisteredCell:
        selected = [cell for cell in self.cells if cell.cell_id == cell_id]
        require(len(selected) == 1, f"cell id is not registered exactly once: {cell_id}")
        return selected[0]

    def seed_block(self, seed: int) -> tuple[RegisteredCell, ...]:
        selected = tuple(cell for cell in self.cells if cell.environment_seed == seed)
        require(len(selected) == 4, f"seed {seed} is not an exact four-cell block")
        return selected


def _validate_cell(row: Mapping[str, Any], runtime_requirement: Mapping[str, Any]) -> None:
    required_scalars = {
        "schema_version": CELL_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "arena": ARENA,
        "execution_mode": "new_behavioral_episode",
        "static_episode_prompt": True,
        "success_predicate_id": "frozen_v3_robotwin_relation_aware_success",
        "outcome_coordinate_contract": "frozen_robotwin_native_lateral_axis_and_region",
        "layout_coordinate_contract": "calibrated_robot_frame_y_midline",
    }
    cell_id = str(row.get("cell_id", "<missing>"))
    for key, expected in required_scalars.items():
        require(row.get(key) == expected, f"{cell_id}.{key} changed")
    seed = row.get("environment_seed")
    sample = row.get("sampling_seed")
    level = row.get("symmetry_level_s")
    relation = row.get("relation")
    require(type(seed) is int and seed in SEEDS, f"{cell_id}: seed outside 9400..9426")
    require(sample == seed, f"{cell_id}: environment/sampling seed mismatch")
    require(type(level) in {int, float} and float(level) in LEVELS, f"{cell_id}: level drift")
    require(relation in RELATIONS, f"{cell_id}: relation drift")
    scene = 3 + ((seed - SEEDS[0]) % len(SCENES))
    scene_id = f"robotwin_pair_{scene:02d}"
    anchor = "place_a2b_right" if scene % 2 else "place_a2b_left"
    level_code = f"{int(float(level) * 100):03d}"
    require(row.get("scene_id") == scene_id, f"{cell_id}: scene assignment drift")
    require(row.get("scene_cluster_id") == scene_id, f"{cell_id}: scene cluster drift")
    require(row.get("anchor_task") == anchor, f"{cell_id}: anchor task drift")
    require(
        row.get("cell_id")
        == f"v3e005:lingbot:seed{seed}:scene{scene:02d}:s{level_code}:{relation}",
        f"{cell_id}: cell id drift",
    )
    require(
        row.get("matched_seed_id") == f"v3e005:lingbot:seed{seed}",
        f"{cell_id}: matched seed id drift",
    )
    require(
        row.get("matched_layout_pair_id")
        == f"v3e005:lingbot:seed{seed}:s{level_code}",
        f"{cell_id}: matched layout pair id drift",
    )
    prompt = row.get("prompt")
    require(isinstance(prompt, str) and prompt, f"{cell_id}: prompt is empty")
    require(
        hashlib.sha256(prompt.encode()).hexdigest() == row.get("prompt_sha256"),
        f"{cell_id}: prompt hash drift",
    )
    require(
        row.get("runtime_identity_requirement") == runtime_requirement,
        f"{cell_id}: runtime identity requirement drift",
    )


def load_registered_bundle(
    study_root: Path,
    *,
    registration_path: Path | None = None,
    queue_path: Path | None = None,
) -> RegisteredBundle:
    root = Path(study_root).expanduser().resolve()
    registration_file = (
        Path(registration_path).expanduser().resolve()
        if registration_path is not None
        else root / REGISTRATION_RELATIVE
    )
    queue_file = (
        Path(queue_path).expanduser().resolve()
        if queue_path is not None
        else root / QUEUE_RELATIVE
    )
    registration_hash = sha256_file(registration_file)
    queue_hash = sha256_file(queue_file)
    require(registration_hash == REGISTRATION_SHA256, "E005 registration hash drift")
    require(queue_hash == QUEUE_SHA256, "E005 queue hash drift")
    registration = load_object(registration_file)
    require(registration.get("schema_version") == REGISTRATION_SCHEMA, "wrong E005 registration schema")
    require(registration.get("study_id") == STUDY_ID, "wrong E005 study id")
    require(registration.get("amendment_id") == AMENDMENT_ID, "wrong E005 amendment")
    require(
        registration.get("status")
        == "registered_before_any_e005_model_request_or_behavioral_episode",
        "E005 registration is not prospective",
    )
    require(registration.get("model_request_count_before_registration") == 0, "registration followed a model request")
    require(registration.get("behavioral_episode_count_before_registration") == 0, "registration followed behavior")
    queue_record = registration.get("queue")
    require(isinstance(queue_record, dict), "registration lacks queue binding")
    require(queue_record.get("sha256") == queue_hash, "registration does not bind queue hash")
    require(queue_record.get("rows") == 108, "registration does not bind 108 cells")
    runtime_requirement = registration.get("runtime_identity_requirement")
    require(isinstance(runtime_requirement, dict), "registration lacks runtime identity requirement")
    rows = load_jsonl(queue_file)
    require(len(rows) == 108, "E005 queue must contain exactly 108 rows")
    require(len({row.get("cell_id") for row in rows}) == 108, "E005 cell ids are not unique")
    for row in rows:
        _validate_cell(row, runtime_requirement)
    observed = {
        (int(row["environment_seed"]), float(row["symmetry_level_s"]), str(row["relation"]))
        for row in rows
    }
    expected = {
        (seed, level, relation)
        for seed in SEEDS
        for level in LEVELS
        for relation in RELATIONS
    }
    require(observed == expected, "E005 queue does not cover the exact 27x2x2 grid")
    order = {relation: index for index, relation in enumerate(RELATIONS)}
    cells = tuple(
        RegisteredCell(dict(row))
        for row in sorted(
            rows,
            key=lambda item: (
                int(item["environment_seed"]),
                float(item["symmetry_level_s"]),
                order[str(item["relation"])],
            ),
        )
    )
    return RegisteredBundle(
        study_root=root,
        registration_path=registration_file,
        queue_path=queue_file,
        registration=registration,
        cells=cells,
        registration_sha256=registration_hash,
        queue_sha256=queue_hash,
    )


def shard_seed_blocks(
    bundle: RegisteredBundle,
    *,
    shard_index: int,
    shard_count: int,
) -> tuple[tuple[RegisteredCell, ...], ...]:
    require(type(shard_count) is int and 1 <= shard_count <= len(SEEDS), "shard_count must be in 1..27")
    require(type(shard_index) is int and 0 <= shard_index < shard_count, "shard_index is outside shard_count")
    blocks: list[tuple[RegisteredCell, ...]] = []
    for position, seed in enumerate(SEEDS):
        if position % shard_count != shard_index:
            continue
        block = bundle.seed_block(seed)
        require(
            {(cell.symmetry_level, cell.relation) for cell in block}
            == {(level, relation) for level in LEVELS for relation in RELATIONS},
            f"seed {seed}: shard would not retain its exact four cells",
        )
        blocks.append(block)
    require(bool(blocks), "requested shard contains no registered seed block")
    return tuple(blocks)


def verify_runtime_identity(
    bundle: RegisteredBundle,
    runtime_manifest_path: Path,
    *,
    external_repository: Path,
    simulator_repository: Path,
    expected_study_commit: str,
    verify_live_files: bool,
) -> dict[str, Any]:
    """Bind the old released LingBot stack to the new, separate E005 queue."""

    require(COMMIT_RE.fullmatch(expected_study_commit) is not None, "expected study commit must be full SHA")
    try:
        runtime = verify_phase_a_runtime_identity(
            bundle.study_root,
            MODEL_ID,
            Path(runtime_manifest_path),
            external_repository=Path(external_repository),
            simulator_repository=Path(simulator_repository),
            # The Phase-A verifier hard-codes branch main.  E005 is a separate
            # preregistered branch, so live Git/file checks are repeated below.
            verify_live_files=False,
        )
    except PhaseAAdapterError as error:
        raise E005ContractError(f"released Phase-A runtime identity failed: {error}") from error
    expected = bundle.registration["runtime_identity_requirement"]
    checks = {
        "model_id": runtime.get("model_id"),
        "checkpoint_id": runtime.get("checkpoint", {}).get("id"),
        "checkpoint_revision": runtime.get("checkpoint", {}).get("revision"),
        "checkpoint_manifest_sha256": runtime.get("checkpoint", {})
        .get("hash_manifest_artifact", {})
        .get("sha256"),
        "environment_lock_sha256": runtime.get("environment", {})
        .get("lock_artifact", {})
        .get("sha256"),
        "adapter_contract_sha256": runtime.get("adapter_contract_sha256"),
        "external_repository_commit": runtime.get("external_repository", {}).get("commit"),
        "simulator_repository_commit": runtime.get("simulator_repository", {}).get("commit"),
        "runtime_payload_sha256": runtime.get("runtime_identity_sha256"),
    }
    for key, observed in checks.items():
        require(observed == expected.get(key), f"E005 runtime identity mismatch for {key}")
    renderer = str(runtime.get("renderer_backend", ""))
    require("headless SAPIEN Vulkan" in renderer, "E005 renderer is not headless SAPIEN Vulkan")
    require("RTX PRO 6000" in renderer, "E005 renderer is not the registered RTX PRO 6000 runtime")
    if verify_live_files:
        verify_git_identity(bundle.study_root, expected_study_commit, require_untracked_clean=True)
        verify_git_identity(Path(external_repository), EXTERNAL_COMMIT)
        verify_git_identity(Path(simulator_repository), SIMULATOR_COMMIT)
        spec_files = runtime.get("adapter_files")
        require(isinstance(spec_files, dict), "runtime lacks adapter file records")
        for name in ("wrapper", "runner"):
            record = spec_files.get(name)
            require(isinstance(record, dict), f"runtime lacks adapter_files.{name}")
            path = Path(str(record.get("path", ""))).expanduser().resolve()
            require(path.is_file(), f"live Phase-A {name} is missing")
            require(sha256_file(path) == record.get("sha256"), f"live Phase-A {name} hash drift")
    return runtime


def validate_bound_artifact(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    require(SHA256_RE.fullmatch(expected_sha256) is not None, f"{label} expected SHA-256 is invalid")
    record = file_record(path)
    require(record["sha256"] == expected_sha256, f"{label} SHA-256 mismatch")
    return load_object(Path(path))


def validate_candidate_binding(
    candidate: Mapping[str, Any],
    *,
    bundle: RegisteredBundle,
    candidate_sha256: str,
) -> None:
    require(candidate.get("study_id") == STUDY_ID, "layout candidate study id mismatch")
    require(candidate.get("amendment_id") == AMENDMENT_ID, "layout candidate amendment mismatch")
    require(candidate.get("arena") == ARENA, "layout candidate arena mismatch")
    require(candidate.get("model_id") in {None, MODEL_ID}, "layout candidate model mismatch")
    require(
        candidate.get("registration_sha256") == bundle.registration_sha256,
        "candidate registration binding mismatch",
    )
    require(candidate.get("queue_sha256") == bundle.queue_sha256, "candidate queue binding mismatch")
    require(candidate.get("candidate_sha256") in {None, candidate_sha256}, "candidate self hash mismatch")
    require(candidate.get("model_request_count", 0) == 0, "candidate followed a model request")
    require(candidate.get("behavioral_episode_count", 0) == 0, "candidate followed behavior")


def validate_model_blind_gate_binding(
    gate: Mapping[str, Any],
    *,
    bundle: RegisteredBundle,
    candidate_sha256: str,
) -> None:
    require(gate.get("schema_version") == MODEL_BLIND_GATE_SCHEMA, "model-blind gate schema mismatch")
    require(gate.get("study_id") == STUDY_ID, "model-blind gate study id mismatch")
    require(gate.get("amendment_id") == AMENDMENT_ID, "model-blind gate amendment mismatch")
    require(gate.get("arena") == ARENA, "model-blind gate arena mismatch")
    require(gate.get("model_id") == MODEL_ID, "model-blind gate model mismatch")
    passed = gate.get("passed") is True or str(gate.get("status", "")).startswith("passed")
    require(passed, "model-blind gate did not pass")
    require(gate.get("registration_sha256") == bundle.registration_sha256, "gate registration binding mismatch")
    require(gate.get("queue_sha256") == bundle.queue_sha256, "gate queue binding mismatch")
    require(gate.get("candidate_sha256") == candidate_sha256, "gate candidate binding mismatch")
    require(
        gate.get("simulator_repository_commit") == SIMULATOR_COMMIT,
        "model-blind gate simulator commit mismatch",
    )
    for key in (
        "model_request_count",
        "model_action_request_count",
        "behavioral_episode_count",
    ):
        require(gate.get(key, 0) == 0, f"model-blind gate {key} is not zero")
    scene_ids = {f"robotwin_pair_{number:02d}" for number in SCENES}
    scenes = gate.get("scenes")
    require(isinstance(scenes, Mapping) and set(scenes) == scene_ids, "model-blind gate scene inventory drift")
    repeat_count = gate.get("reset_gate", {}).get(
        "repeat_count_per_scene_level_relation"
    )
    require(type(repeat_count) is int and repeat_count >= 2, "model-blind gate reset repeats are insufficient")
    expected_per_scene = repeat_count * len(LEVELS) * len(RELATIONS)
    require(gate.get("scene_count") == len(scene_ids), "model-blind gate scene count drift")
    require(
        gate.get("reset_count") == len(scene_ids) * expected_per_scene,
        "model-blind gate reset count drift",
    )
    for scene_id in sorted(scene_ids):
        scene = scenes[scene_id]
        require(isinstance(scene, Mapping) and scene.get("scene_id") == scene_id, "gate scene row drift")
        layouts = scene.get("resolved_layouts")
        require(isinstance(layouts, Mapping) and set(layouts) == {"0.00", "1.00"}, "gate layout inventory drift")
        resets = scene.get("reset_evidence")
        require(isinstance(resets, list) and len(resets) == expected_per_scene, "gate reset evidence count drift")
        observed = [
            (
                float(row.get("symmetry_level_s")),
                row.get("relation"),
                int(row.get("repeat_index")),
            )
            for row in resets
            if isinstance(row, Mapping)
        ]
        expected = [
            (level, relation, repeat)
            for level in LEVELS
            for repeat in range(repeat_count)
            for relation in RELATIONS
        ]
        require(sorted(observed) == sorted(expected), "gate reset evidence grid drift")
        require(
            all(row.get("validation", {}).get("passed") is True for row in resets),
            "gate contains an unvalidated live reset",
        )
