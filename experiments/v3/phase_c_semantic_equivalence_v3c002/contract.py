"""Pure, hash-bound contract helpers for the prospective V3-C002 cohort."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


STUDY_ID = "vla_wam_language_steerability_v3"
AMENDMENT_ID = "V3-C002"
REGISTRATION_SCHEMA = "vla-wam-shared-v3c002-registration-v1"
CELL_SCHEMA = "vla-wam-shared-v3c002-cell-v1"
WORDING_GATE_SCHEMA = "vla-wam-shared-v3c002-wording-gate-v1"
RELEASE_GATE_SCHEMA = "vla-wam-shared-v3c002-release-gate-v1"
ARENA = "droid_robolab"
MODEL_ID = "pi05_current_stack_droid"
LAYOUT_LEVEL = 1.0
SUCCESS_PREDICATE_ID = "v2_frozen_droid_robolab_release_inside_45deg_requested_relation"
PROMPT_CONDITIONS = (
    "canonical_left",
    "inverse_reference_left",
    "canonical_right",
    "inverse_reference_right",
)
PHYSICAL_GOALS = ("left", "right")
SEED_START = 12000
SEED_END = 12340
SEEDS = tuple(range(SEED_START, SEED_END + 1))
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """A value is outside the frozen V3-C002 contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def read_finite_json(path: Path) -> Any:
    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError(f"not finite UTF-8 JSON: {path}: {exc}") from exc


def file_binding(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    require(path.is_file(), f"required source file does not exist: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def registered_prompts() -> dict[str, dict[str, Any]]:
    """Return prompts with goal metadata; never recover a goal by parsing text."""

    prompts = {
        "canonical_left": ("left", "left", "Put the Rubik's cube to the left of the bowl."),
        "inverse_reference_left": (
            "left",
            "right",
            "Place the Rubik's cube so that the bowl is to the right of the Rubik's cube.",
        ),
        "canonical_right": ("right", "right", "Put the Rubik's cube to the right of the bowl."),
        "inverse_reference_right": (
            "right",
            "left",
            "Place the Rubik's cube so that the bowl is to the left of the Rubik's cube.",
        ),
    }
    return {
        condition: {
            "condition": condition,
            "physical_goal": physical_goal,
            "surface_direction_word": word,
            "prompt": prompt,
            "prompt_utf8_hex": prompt.encode("utf-8").hex(),
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        }
        for condition, (physical_goal, word, prompt) in prompts.items()
    }


def deterministic_condition_order(seed: int) -> tuple[str, ...]:
    """A hash-ranked, fixed randomization; no run-time RNG state is used."""

    require(seed in SEEDS, f"seed {seed} is outside V3-C002")
    return tuple(
        sorted(
            PROMPT_CONDITIONS,
            key=lambda condition: sha256_bytes(f"V3-C002|queue-order-v1|{seed}|{condition}".encode("utf-8")),
        )
    )


def request_seed(episode_seed: int, replan_index: int) -> int:
    require(episode_seed in SEEDS, "episode seed is unregistered")
    require(type(replan_index) is int and replan_index >= 0, "replan index must be a non-negative integer")
    return episode_seed * 1000 + replan_index


@dataclass(frozen=True)
class C002Cell:
    row: Mapping[str, Any]

    @property
    def cell_id(self) -> str:
        return str(self.row["cell_id"])

    @property
    def seed(self) -> int:
        return int(self.row["episode_seed"])

    @property
    def condition(self) -> str:
        return str(self.row["prompt_condition"])

    @property
    def physical_goal(self) -> str:
        return str(self.row["physical_goal"])

    @property
    def block_id(self) -> str:
        return str(self.row["seed_block_id"])

    @property
    def row_sha256(self) -> str:
        return canonical_json_sha256(self.row)


def validate_cell(row: Any, *, candidate_sha256: str) -> C002Cell:
    require(isinstance(row, dict), "queue row must be an object")
    for key, expected in (
        ("schema_version", CELL_SCHEMA),
        ("study_id", STUDY_ID),
        ("amendment_id", AMENDMENT_ID),
        ("arena", ARENA),
        ("model_id", MODEL_ID),
        ("execution_mode", "new_behavioral_episode"),
        ("layout_source_amendment", "V3-E004"),
        ("symmetry_level_s", LAYOUT_LEVEL),
        ("success_predicate_id", SUCCESS_PREDICATE_ID),
    ):
        require(row.get(key) == expected, f"queue row {row.get('cell_id')} differs for {key}")
    seed = row.get("episode_seed")
    require(type(seed) is int and seed in SEEDS, "episode_seed is not registered")
    condition = row.get("prompt_condition")
    prompts = registered_prompts()
    require(condition in prompts, "prompt condition is not registered")
    prompt = prompts[str(condition)]
    for key in ("physical_goal", "surface_direction_word", "prompt", "prompt_utf8_hex", "prompt_sha256"):
        require(row.get(key) == prompt[key], f"queue prompt metadata differs for {key}")
    require(row.get("layout_candidate_sha256") == candidate_sha256, "queue layout digest changed")
    require(row.get("environment_seed") == seed and row.get("sampling_seed") == seed, "within-block seeds changed")
    require(row.get("seed_block_id") == f"v3c002:seed{seed}", "seed block id changed")
    order = deterministic_condition_order(seed)
    require(row.get("execution_order") == list(order), "execution order changed")
    require(row.get("execution_order_index") == order.index(str(condition)), "execution order index changed")
    require(row.get("request_seed_formula") == "episode_seed * 1000 + replan_index", "request seed formula changed")
    return C002Cell(row=row)


def load_cells(*, registration_path: Path, queue_path: Path) -> tuple[dict[str, Any], list[C002Cell]]:
    registration = read_finite_json(registration_path)
    require(isinstance(registration, dict), "registration must be an object")
    require(registration.get("schema_version") == REGISTRATION_SCHEMA, "registration schema changed")
    require(registration.get("study_id") == STUDY_ID and registration.get("amendment_id") == AMENDMENT_ID, "registration identity changed")
    registration_queue = registration.get("queue")
    require(isinstance(registration_queue, dict), "registration queue binding is missing")
    require(registration_queue.get("sha256") == sha256_file(queue_path), "queue digest does not match registration")
    require(registration_queue.get("bytes") == Path(queue_path).stat().st_size, "queue bytes do not match registration")
    layout = registration.get("e004_s1_layout")
    require(isinstance(layout, dict), "E004 layout binding is missing")
    candidate_sha256 = layout.get("candidate_sha256")
    require(isinstance(candidate_sha256, str) and _SHA256.fullmatch(candidate_sha256), "candidate digest is invalid")
    rows = []
    for number, line in enumerate(Path(queue_path).read_text(encoding="utf-8").splitlines(), 1):
        require(line.strip(), f"queue has a blank line at {number}")
        try:
            row = json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ContractError(f"invalid queue JSON on line {number}: {exc}") from exc
        rows.append(validate_cell(row, candidate_sha256=candidate_sha256))
    require(len(rows) == 1364, "C002 must have 1,364 queue rows")
    require(len({cell.cell_id for cell in rows}) == len(rows), "queue cell IDs are not unique")
    for seed in SEEDS:
        block = [cell for cell in rows if cell.seed == seed]
        require(len(block) == 4 and {cell.condition for cell in block} == set(PROMPT_CONDITIONS), f"seed block {seed} is incomplete")
    return registration, rows


def require_released_gate(
    *, registration_path: Path, queue_path: Path, release_gate_path: Path
) -> tuple[dict[str, Any], list[C002Cell], dict[str, Any]]:
    registration, cells = load_cells(registration_path=registration_path, queue_path=queue_path)
    require(registration.get("registration_status") == "registered_after_two_human_wording_agreements", "behavioral registration is not active")
    gate = read_finite_json(release_gate_path)
    require(isinstance(gate, dict) and gate.get("schema_version") == RELEASE_GATE_SCHEMA, "release gate schema changed")
    require(gate.get("passed") is True and gate.get("status") == "passed_pre_request_release", "release gate has not passed")
    for key, path in (("registration", registration_path), ("queue", queue_path)):
        binding = gate.get(key)
        require(isinstance(binding, dict), f"release gate lacks {key} binding")
        require(binding.get("sha256") == sha256_file(path), f"release gate {key} digest mismatch")
    wording = gate.get("wording_gate")
    require(isinstance(wording, dict) and wording.get("passed") is True, "independent wording gate has not passed")
    return registration, cells, gate


def grouped_shard(cells: Sequence[C002Cell], *, shard_index: int, shard_count: int) -> list[C002Cell]:
    require(type(shard_index) is int and type(shard_count) is int and 0 <= shard_index < shard_count, "invalid shard")
    selected_seeds = {
        seed
        for seed in SEEDS
        if int(sha256_bytes(f"V3-C002|shard-v1|{seed}".encode("utf-8"))[:16], 16) % shard_count == shard_index
    }
    result = [cell for cell in cells if cell.seed in selected_seeds]
    for seed in selected_seeds:
        require(len([cell for cell in result if cell.seed == seed]) == 4, "shard split a seed block")
    return result


def finite_number(value: Any, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(float(value)), f"{label} must be finite")
    return float(value)
