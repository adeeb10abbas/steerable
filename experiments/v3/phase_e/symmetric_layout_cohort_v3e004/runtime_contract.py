"""Hash-bound runtime contract shared by the V3-E004 DROID lanes.

This module intentionally imports no simulator or model package.  It is used
by queue launchers, live-reset adapters, and episode compilers to ensure that
all three prospective inputs (registration, queue, and layout candidate) are
identical before a policy request can be made.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


REGISTRATION_SCHEMA = "vla-wam-shared-v3e004-registration-v1"
CELL_SCHEMA = "vla-wam-shared-v3e004-cell-v1"
LANE_RELEASE_SCHEMA = "vla-wam-shared-v3e004-droid-lane-release-v1"
STUDY_ID = "vla_wam_language_steerability_v3"
AMENDMENT_ID = "V3-E004"
ARENA = "droid_robolab"
NEW_EPISODE_MODE = "new_behavioral_episode"
MODEL_IDS = frozenset(
    {
        "pi05_current_stack_droid",
        "cosmos3_nano_policy_droid",
        "dreamzero_droid_action_cfg",
        "cosmos3_edge_policy_droid",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RuntimeContractError(ValueError):
    """A runtime input is outside the registered E004 release."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _finite_json(path: Path) -> Any:
    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeContractError(f"not finite UTF-8 JSON: {path}: {exc}") from exc


def _bound_path(path: Path, expected_sha256: str, label: str) -> Path:
    path = Path(path).resolve()
    _require(_SHA256.fullmatch(expected_sha256 or "") is not None, f"{label} digest is invalid")
    _require(path.is_file(), f"{label} does not exist: {path}")
    _require(sha256_file(path) == expected_sha256, f"{label} SHA-256 mismatch")
    return path


@dataclass(frozen=True)
class E004Cell:
    row: Mapping[str, Any]

    @property
    def cell_id(self) -> str:
        return str(self.row["cell_id"])

    @property
    def model_id(self) -> str:
        return str(self.row["model_id"])

    @property
    def relation(self) -> str:
        return str(self.row["relation"])

    @property
    def symmetry_level_s(self) -> float:
        return float(self.row["symmetry_level_s"])

    @property
    def environment_seed(self) -> int:
        return int(self.row["environment_seed"])

    @property
    def sampling_seed(self) -> int:
        return int(self.row["sampling_seed"])

    @property
    def matched_pair_id(self) -> str:
        return str(self.row["matched_pair_id"])

    @property
    def row_sha256(self) -> str:
        return canonical_json_sha256(self.row)


@dataclass(frozen=True)
class E004RuntimeBundle:
    registration_path: Path
    registration_sha256: str
    registration: Mapping[str, Any]
    queue_path: Path
    queue_sha256: str
    candidate_path: Path
    candidate_sha256: str
    cells: tuple[E004Cell, ...]

    def cell(self, cell_id: str) -> E004Cell:
        matches = [cell for cell in self.cells if cell.cell_id == cell_id]
        _require(len(matches) == 1, f"cell is absent or duplicated in E004 queue: {cell_id}")
        return matches[0]

    def droid_new_cells(self, model_id: str) -> tuple[E004Cell, ...]:
        _require(model_id in MODEL_IDS, f"model is not an E004 DROID checkpoint: {model_id}")
        rows = tuple(
            cell
            for cell in self.cells
            if cell.model_id == model_id
            and cell.row["arena"] == ARENA
            and cell.row["execution_mode"] == NEW_EPISODE_MODE
        )
        _require(rows, f"no new DROID behavioral rows registered for {model_id}")
        return rows


def _validate_cell(row: Any, candidate_sha256: str) -> E004Cell:
    _require(isinstance(row, dict), "queue row must be an object")
    expected = {
        "schema_version": CELL_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
    }
    for key, value in expected.items():
        _require(row.get(key) == value, f"queue row {row.get('cell_id')} differs for {key}")
    _require(isinstance(row.get("cell_id"), str) and row["cell_id"], "cell_id is invalid")
    _require(row.get("relation") in {"left", "right"}, "relation must be left or right")
    _require(row.get("arena") in {ARENA, "robotwin"}, "queue arena is invalid")
    _require(row.get("execution_mode") == NEW_EPISODE_MODE, "execution mode is invalid")
    _require(type(row.get("environment_seed")) is int and row["environment_seed"] >= 0, "environment seed is invalid")
    _require(type(row.get("sampling_seed")) is int and row["sampling_seed"] >= 0, "sampling seed is invalid")
    level = row.get("symmetry_level_s")
    _require(type(level) in (int, float) and math.isfinite(float(level)), "symmetry level is invalid")
    _require(float(level) in {0.0, 0.25, 0.5, 0.75, 1.0}, "symmetry level is unregistered")
    prompt = row.get("prompt")
    _require(isinstance(prompt, str) and prompt, "prompt is invalid")
    _require(hashlib.sha256(prompt.encode("utf-8")).hexdigest() == row.get("prompt_sha256"), "prompt digest mismatch")
    if row.get("arena") == ARENA:
        _require(row.get("model_id") in MODEL_IDS, "DROID row model is not authorized")
        _require(row.get("layout_candidate_sha256") == candidate_sha256, "DROID row layout candidate digest changed")
        _require(row.get("success_predicate_id") == "v2_frozen_droid_robolab_release_inside_45deg_requested_relation", "DROID predicate changed")
    return E004Cell(row=row)


def load_runtime_bundle(
    *,
    registration_path: Path,
    registration_sha256: str,
    queue_path: Path,
    queue_sha256: str,
    candidate_path: Path,
    candidate_sha256: str,
) -> E004RuntimeBundle:
    registration_path = _bound_path(registration_path, registration_sha256, "registration")
    queue_path = _bound_path(queue_path, queue_sha256, "queue")
    candidate_path = _bound_path(candidate_path, candidate_sha256, "candidate")
    registration = _finite_json(registration_path)
    _require(isinstance(registration, dict), "registration must be an object")
    _require(registration.get("schema_version") == REGISTRATION_SCHEMA, "registration schema changed")
    _require(registration.get("study_id") == STUDY_ID and registration.get("amendment_id") == AMENDMENT_ID, "registration identity changed")
    _require(registration.get("status") == "prospectively_registered_zero_e004_model_requests_or_behavioral_episodes", "registration is not the prospective zero-request freeze")
    queue_record = registration.get("queue")
    _require(isinstance(queue_record, dict), "registration queue binding is missing")
    _require(queue_record.get("sha256") == queue_sha256, "registration does not bind queue digest")
    _require(queue_record.get("bytes") == queue_path.stat().st_size, "registration queue byte count changed")
    layout = registration.get("layout")
    _require(isinstance(layout, dict) and layout.get("candidate_sha256") == candidate_sha256, "registration does not bind candidate digest")
    rows: list[E004Cell] = []
    for line_number, line in enumerate(queue_path.read_text(encoding="utf-8").splitlines(), 1):
        _require(line.strip() != "", f"queue contains a blank line at {line_number}")
        try:
            value = json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeContractError(f"queue row {line_number} is invalid: {exc}") from exc
        rows.append(_validate_cell(value, candidate_sha256))
    _require(len(rows) == queue_record.get("rows"), "registration queue row count changed")
    ids = [cell.cell_id for cell in rows]
    _require(len(ids) == len(set(ids)), "queue cell IDs are not unique")
    return E004RuntimeBundle(
        registration_path=registration_path,
        registration_sha256=registration_sha256,
        registration=registration,
        queue_path=queue_path,
        queue_sha256=queue_sha256,
        candidate_path=candidate_path,
        candidate_sha256=candidate_sha256,
        cells=tuple(rows),
    )


def validate_lane_release(
    path: Path,
    expected_sha256: str,
    *,
    bundle: E004RuntimeBundle,
    model_id: str,
    lane_pod_uid: str,
    lane_gpu_uuid: str,
) -> dict[str, Any]:
    """Validate the pre-request lane gates that are independent of live reset.

    The per-cell live geometry/camera gate is intentionally not represented by
    this manifest: it is generated after the physical reset and immediately
    before request zero by :mod:`live_snapshot_adapter`.
    """

    path = _bound_path(path, expected_sha256, "lane release")
    value = _finite_json(path)
    _require(isinstance(value, dict), "lane release must be an object")
    expected = {
        "schema_version": LANE_RELEASE_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "status": "passed_pre_request_gates_except_per_cell_live_snapshot",
        "registration_sha256": bundle.registration_sha256,
        "queue_sha256": bundle.queue_sha256,
        "candidate_sha256": bundle.candidate_sha256,
        "model_id": model_id,
        "lane_pod_uid": lane_pod_uid,
        "lane_gpu_uuid": lane_gpu_uuid,
    }
    for key, wanted in expected.items():
        _require(value.get(key) == wanted, f"lane release differs for {key}")
    gates = value.get("gates")
    _require(isinstance(gates, dict), "lane release gate map is missing")
    for name in (
        "exact_runtime_identity",
        "renderer_viewport_video",
        "raw_pvc_write",
        "model_endpoint",
        "single_isaac_process",
    ):
        _require(gates.get(name) is True, f"lane release gate did not pass: {name}")
    runtime = value.get("runtime_identity")
    _require(isinstance(runtime, dict), "runtime identity is missing")
    _require(_SHA256.fullmatch(str(runtime.get("sha256", ""))) is not None, "runtime identity digest is invalid")
    return dict(value)


def shard_cells(cells: Iterable[E004Cell], *, shard_index: int, shard_count: int) -> tuple[E004Cell, ...]:
    """Return a stable whole-pair shard; LEFT/RIGHT are never split."""

    _require(type(shard_count) is int and shard_count > 0, "shard_count must be positive")
    _require(type(shard_index) is int and 0 <= shard_index < shard_count, "shard_index is outside shard_count")
    grouped: dict[str, list[E004Cell]] = {}
    for cell in cells:
        grouped.setdefault(cell.matched_pair_id, []).append(cell)
    output: list[E004Cell] = []
    for pair_index, pair_id in enumerate(sorted(grouped)):
        pair = grouped[pair_id]
        _require(len(pair) == 2 and {cell.relation for cell in pair} == {"left", "right"}, f"registered matched pair is incomplete: {pair_id}")
        if pair_index % shard_count == shard_index:
            output.extend(sorted(pair, key=lambda cell: int(cell.row["execution_order_index_within_model_seed"])))
    return tuple(output)
