"""Immutable SGW-01 release and queue contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


class ContractError(ValueError):
    """Raised when an execution release is incomplete or has drifted."""


STAGE_EPISODES = {"P": 6, "D": 24, "C": 144}
REQUIRED_RELEASE_FILES = (
    "protocol.json", "prompts.json", "queue.jsonl", "fixtures.json",
    "runtime_binding.json", "release_receipt.json", "hashes.json",
)
REQUIRED_BINDING_FIELDS = {
    "context", "namespace", "resource_owner", "worker_image_digest", "pvc_name",
    "pvc_mount_path", "pvc_access_mode", "lock_test_receipt", "model_gpu_counts",
    "cpu_memory_limits", "node_gpu_type", "cluster_version", "source_commit",
    "model_code_commits", "simulator_commit", "renderer_receipt",
    "persistent_write_receipt", "checkpoint_hashes", "user_resource_budget",
    "budget_source", "policy_ports", "cache_reset_receipt", "frame_time_mapping_hashes",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def validate_stage_authorizations(authorizations: Any) -> dict[str, dict[str, str]]:
    """Require current hash-bound, passed technical receipts; never outcome gates."""
    if not isinstance(authorizations, dict):
        raise ContractError("stage authorizations must contain receipt records")
    required = {"direct_command_fixed_input_gate", "P", "D", "C"}
    if set(authorizations) != required:
        raise ContractError("stage authorizations must bind direct gate and P/D/C receipts only")
    validated: dict[str, dict[str, str]] = {}
    for name in required:
        receipt = authorizations[name]
        if not isinstance(receipt, dict) or not isinstance(receipt.get("path"), str) or not isinstance(receipt.get("sha256"), str):
            raise ContractError(f"{name} authorization lacks a hash-bound receipt")
        path = Path(receipt["path"])
        if not path.is_file() or sha256_file(path) != receipt["sha256"]:
            raise ContractError(f"{name} authorization receipt is missing or changed")
        value = load_json(path, f"{name} authorization receipt")
        if value.get("status") not in {"passed", "qualified", "accepted"}:
            raise ContractError(f"{name} authorization receipt is not technically passed")
        validated[name] = {"path": str(path.resolve()), "sha256": receipt["sha256"]}
    return validated


@dataclass(frozen=True)
class Cell:
    row: Mapping[str, Any]

    @property
    def cell_id(self) -> str:
        return str(self.row["cell_id"])

    @property
    def block_id(self) -> str:
        return str(self.row["block_id"])

    @property
    def model(self) -> str:
        return str(self.row["model"])

    @property
    def family(self) -> str:
        return str(self.row["family"])

    @property
    def stage(self) -> str:
        return str(self.row["stage"])


@dataclass(frozen=True)
class Release:
    root: Path
    release_id: str
    hashes: Mapping[str, str]
    cells: tuple[Cell, ...]
    binding: Mapping[str, Any]

    def partition(self, model: str, family: str, stage: str) -> tuple[Cell, ...]:
        if stage not in STAGE_EPISODES:
            raise ContractError(f"unknown stage: {stage}")
        cells = tuple(c for c in self.cells if (c.model, c.family, c.stage) == (model, family, stage))
        if len(cells) != STAGE_EPISODES[stage]:
            raise ContractError(f"partition {model}/{family}/{stage} has {len(cells)}, expected {STAGE_EPISODES[stage]}")
        return tuple(sorted(cells, key=lambda c: int(c.row["within_block_order"])))


def _load_cells(path: Path, release_id: str) -> tuple[Cell, ...]:
    rows: list[Cell] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise ContractError(f"blank queue line {number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid queue line {number}") from exc
        required = {"cell_id", "block_id", "model", "family", "stage", "layout_id",
                    "prompt_id", "prompt", "prompt_sha256", "within_block_order",
                    "fixture_sha256", "runtime_sha256", "time_map_sha256", "release_id"}
        if not isinstance(row, dict) or not required.issubset(row):
            raise ContractError(f"queue line {number} lacks immutable cell fields")
        if row["release_id"] != release_id or row.get("status") != "RELEASED":
            raise ContractError(f"queue line {number} is not released for this release")
        if row["stage"] not in STAGE_EPISODES or row["model"] not in {"N3", "D1"}:
            raise ContractError(f"queue line {number} has invalid model/stage")
        if row["family"] not in {"LAT", "HEIGHT", "DIST"} or not all(row[key] for key in ("fixture_sha256", "runtime_sha256", "time_map_sha256")):
            raise ContractError(f"queue line {number} lacks qualified runtime or fixture binding")
        if hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
            raise ContractError(f"queue line {number} prompt hash mismatch")
        rows.append(Cell(row))
    if not rows or len({c.cell_id for c in rows}) != len(rows):
        raise ContractError("queue must contain unique released cells")
    return tuple(rows)


def load_release(release: Path) -> Release:
    root = Path(release).resolve()
    if not root.is_dir():
        raise ContractError(f"release directory does not exist: {root}")
    missing = [name for name in REQUIRED_RELEASE_FILES if not (root / name).is_file()]
    if missing:
        raise ContractError(f"release missing files: {', '.join(missing)}")
    hashes = load_json(root / "hashes.json", "release hash manifest")
    if set(hashes) != set(REQUIRED_RELEASE_FILES) - {"hashes.json"}:
        raise ContractError("hash manifest must bind every immutable release input")
    for name, expected in hashes.items():
        if not isinstance(expected, str) or sha256_file(root / name) != expected:
            raise ContractError(f"immutable release hash mismatch: {name}")
    receipt = load_json(root / "release_receipt.json", "release receipt")
    release_id = receipt.get("release_id")
    if not isinstance(release_id, str) or not release_id:
        raise ContractError("release receipt lacks release_id")
    binding = load_json(root / "runtime_binding.json", "runtime binding")
    missing_binding = sorted(REQUIRED_BINDING_FIELDS - set(binding))
    if missing_binding:
        raise ContractError(f"runtime binding missing fields: {', '.join(missing_binding)}")
    if "@sha256:" not in str(binding["worker_image_digest"]):
        raise ContractError("worker image must be immutable by digest")
    if binding.get("resource_owner") != receipt.get("resource_owner"):
        raise ContractError("release receipt resource owner differs from binding")
    cells = _load_cells(root / "queue.jsonl", release_id)
    return Release(root, release_id, hashes, cells, binding)


def verify_completion_pointer(release: Release, pointer: Path) -> dict[str, Any]:
    value = load_json(pointer, "completion pointer")
    if value.get("release_id") != release.release_id or not isinstance(value.get("result"), dict):
        raise ContractError("completion pointer release identity mismatch")
    result = value["result"]
    result_path = Path(str(result.get("path", "")))
    if not result_path.is_file() or result.get("sha256") != sha256_file(result_path):
        raise ContractError("completion pointer result is absent or corrupt")
    manifest_path = Path(str(value.get("manifest_path", "")))
    if not manifest_path.is_file() or value.get("manifest_sha256") != sha256_file(manifest_path):
        raise ContractError("completion pointer manifest is absent or corrupt")
    manifest = load_json(manifest_path, "attempt artifact manifest")
    if (manifest.get("release_id") != release.release_id or manifest.get("complete") is not True
            or manifest.get("release_hashes") != dict(release.hashes)
            or manifest.get("result") != load_json(result_path, "attempt result")):
        raise ContractError("completion artifact manifest release mismatch")
    return value


def partition_blocks(cells: Iterable[Cell]) -> dict[str, tuple[Cell, ...]]:
    blocks: dict[str, list[Cell]] = {}
    for cell in cells:
        blocks.setdefault(cell.block_id, []).append(cell)
    result = {key: tuple(sorted(value, key=lambda cell: int(cell.row["within_block_order"]))) for key, value in blocks.items()}
    if any(len(block) != 6 for block in result.values()):
        raise ContractError("each released block must contain exactly six cells")
    return result
