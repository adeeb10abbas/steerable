"""Pure, hash-bound contract helpers for the prospective V3-C002 cohort."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence


STUDY_ID = "vla_wam_language_steerability_v3"
AMENDMENT_ID = "V3-C002"
REGISTRATION_SCHEMA = "vla-wam-shared-v3c002-registration-v4"
CELL_SCHEMA = "vla-wam-shared-v3c002-cell-v4"
WORDING_GATE_SCHEMA = "vla-wam-shared-v3c002-wording-gate-v1"
RELEASE_GATE_SCHEMA = "vla-wam-shared-v3c002-release-gate-v4"
SOURCE_PUSH_GATE_SCHEMA = "vla-wam-shared-v3c002-source-push-gate-v1"
PHYSICAL_GATE_SCHEMA = "vla-wam-shared-v3c002-model-blind-physical-gate-v1"
SMOKE_GATE_SCHEMA = "vla-wam-shared-v3c002-excluded-smoke-gate-v1"
ISOLATION_GATE_SCHEMA = "vla-wam-shared-v3c002-two-lane-isolation-gate-v1"
LANE_GATE_SCHEMA = "vla-wam-shared-v3c002-lane-release-manifest-v1"
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
REPO_ROOT = Path(__file__).resolve().parents[3]


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


def repo_file_binding(path: Path) -> dict[str, Any]:
    binding = file_binding(path)
    try:
        binding["path"] = str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError as exc:
        raise ContractError(f"committed evidence must be inside the repository: {path}") from exc
    return binding


def resolve_binding_path(record: Mapping[str, Any]) -> Path:
    path = Path(str(record.get("path", "")))
    return path if path.is_absolute() else REPO_ROOT / path


def validate_file_binding(record: Any, label: str) -> dict[str, Any]:
    require(isinstance(record, Mapping), f"{label} binding is missing")
    path = resolve_binding_path(record)
    require(path.is_file(), f"{label} artifact does not exist: {path}")
    require(record.get("bytes") == path.stat().st_size, f"{label} artifact byte count changed")
    require(record.get("sha256") == sha256_file(path), f"{label} artifact digest changed")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": str(record["sha256"])}


def validate_exact_runtime_contract(value: Any) -> str:
    require(isinstance(value, dict), "exact E004 runtime contract is missing")
    digest = value.get("contract_sha256")
    unsigned = {key: item for key, item in value.items() if key != "contract_sha256"}
    require(digest == canonical_json_sha256(unsigned), "exact E004 runtime contract digest changed")
    require(value.get("source_amendment") == "V3-E004" and value.get("symmetry_level_s") == 1.0, "runtime source/layout changed")
    dependencies = value.get("dependency_bindings")
    require(isinstance(dependencies, dict), "runtime dependency bindings are missing")
    required_groups = ("checkpoint", "runtime", "policy_server", "controller", "action_interface", "camera_configuration", "horizon", "scorer", "raw_writer", "renderer")
    for group in required_groups:
        records = dependencies.get(group)
        require(isinstance(records, list) and records, f"exact runtime contract lacks {group} bindings")
        for index, record in enumerate(records):
            validate_file_binding(record, f"runtime {group}[{index}]")
    values = value.get("identity_values")
    require(isinstance(values, dict), "exact runtime identity values are missing")
    for key in (
        "model_id", "arena", "checkpoint", "checkpoint_manifest_sha256", "checkpoint_digest",
        "openpi_commit", "robolab_commit", "source_commit", "action_dim", "action_horizon",
        "action_cap", "policy_cameras", "simulator_identity", "renderer_backend",
    ):
        require(key in values, f"exact runtime identity lacks {key}")
    require(values["model_id"] == MODEL_ID and values["arena"] == ARENA, "exact runtime model/arena changed")
    require(values["action_dim"] == 8 and values["action_horizon"] == 15 and values["action_cap"] == 450, "action interface/horizon changed")
    require(isinstance(values["policy_cameras"], list) and len(values["policy_cameras"]) == 4, "policy camera set changed")
    digests = value.get("component_digests")
    require(isinstance(digests, dict), "runtime component digests are missing")
    for group in required_groups:
        expected = canonical_json_sha256(dependencies[group])
        require(digests.get(group) == expected, f"runtime {group} component digest changed")
    return str(digest)


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


def validate_cell(row: Any, *, candidate_sha256: str, runtime_contract_sha256: str) -> C002Cell:
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
    require(row.get("exact_runtime_contract_sha256") == runtime_contract_sha256, "queue exact runtime contract changed")
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
    runtime_contract_sha256 = validate_exact_runtime_contract(registration.get("exact_e004_pi05_runtime"))
    lineage = registration.get("source_lineage")
    require(isinstance(lineage, dict), "prospective source lineage is missing")
    require(lineage.get("required_base_commit") == "18a2bf0200183647291cc7aeb1fe89997b3fb82f", "required source base changed")
    require(lineage.get("recorded_before_any_model_request") is True and lineage.get("model_requests_at_recording") == 0 and lineage.get("behavioral_episodes_at_recording") == 0, "source lineage was not recorded prospectively")
    require(lineage.get("replacement_commit") == registration["exact_e004_pi05_runtime"]["identity_values"]["source_commit"], "replacement/runtime source commits differ")
    rows = []
    for number, line in enumerate(Path(queue_path).read_text(encoding="utf-8").splitlines(), 1):
        require(line.strip(), f"queue has a blank line at {number}")
        try:
            row = json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ContractError(f"invalid queue JSON on line {number}: {exc}") from exc
        rows.append(validate_cell(row, candidate_sha256=candidate_sha256, runtime_contract_sha256=runtime_contract_sha256))
    require(len(rows) == 1364, "C002 must have 1,364 queue rows")
    require(len({cell.cell_id for cell in rows}) == len(rows), "queue cell IDs are not unique")
    for seed in SEEDS:
        block = [cell for cell in rows if cell.seed == seed]
        require(len(block) == 4 and {cell.condition for cell in block} == set(PROMPT_CONDITIONS), f"seed block {seed} is incomplete")
    return registration, rows


def _bound_json(record: Any, label: str, *, schema: str, status: str) -> dict[str, Any]:
    binding = validate_file_binding(record, label)
    value = read_finite_json(Path(binding["path"]))
    require(isinstance(value, dict), f"{label} must be a JSON object")
    require(value.get("schema_version") == schema, f"{label} schema changed")
    require(value.get("status") == status and value.get("passed") is True, f"{label} has not passed")
    return value


def _verify_pushed_source_commit(source_gate: Mapping[str, Any]) -> None:
    commit = source_gate.get("source_commit")
    branch = source_gate.get("branch")
    remote = source_gate.get("remote")
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "source gate commit is invalid")
    require(isinstance(branch, str) and branch and isinstance(remote, str) and remote, "source gate branch/remote are missing")
    local = subprocess.run(["git", "rev-parse", commit], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    require(local == commit, "source gate commit is unavailable locally")
    remote_rows = subprocess.run(["git", "ls-remote", "--heads", remote, branch], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    heads = [row.split()[0] for row in remote_rows if row.split()]
    require(len(heads) == 1, "bound remote branch is missing or ambiguous")
    head = heads[0]
    available = subprocess.run(["git", "cat-file", "-e", f"{head}^{{commit}}"], cwd=REPO_ROOT, capture_output=True).returncode == 0
    if not available:
        subprocess.run(["git", "fetch", "--no-tags", remote, branch], cwd=REPO_ROOT, check=True, capture_output=True)
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", commit, head], cwd=REPO_ROOT).returncode == 0
    require(ancestor, "source gate commit is not contained in the pushed remote branch")


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
    wording = _bound_json(gate.get("wording_gate"), "wording gate", schema=WORDING_GATE_SCHEMA, status="passed_two_authorized_independent_human_readers_agree_same_endpoint")
    readers = wording.get("reader_attestations")
    require(isinstance(readers, list) and len(readers) == 2 and len({record.get("reader_id") for record in readers if isinstance(record, dict)}) == 2, "wording gate lacks two independent readers")
    source_gate = _bound_json(gate.get("source_push_gate"), "source push gate", schema=SOURCE_PUSH_GATE_SCHEMA, status="passed_source_commit_pushed")
    require(source_gate.get("pushed") is True and source_gate.get("registration_sha256") == sha256_file(registration_path), "source push gate identity changed")
    _verify_pushed_source_commit(source_gate)
    lineage = registration.get("source_lineage")
    require(isinstance(lineage, dict) and lineage.get("recorded_before_any_model_request") is True, "source lineage is missing or retrospective")
    require(lineage.get("required_base_commit") == "18a2bf0200183647291cc7aeb1fe89997b3fb82f", "required source base changed")
    require(lineage.get("replacement_commit") == source_gate.get("source_commit") == registration["exact_e004_pi05_runtime"]["identity_values"]["source_commit"], "runtime/source-push/replacement commit differs")
    physical = _bound_json(gate.get("physical_gate"), "model-blind physical gate", schema=PHYSICAL_GATE_SCHEMA, status="passed_exact_e004_model_blind_physical_preflight")
    for key in ("physical_scene", "full_reset", "policy_cameras", "raw_writer", "renderer"):
        require(physical.get(key) is True, f"physical gate lacks passed {key}")
    require(physical.get("model_requests") == 0 and physical.get("behavioral_episodes") == 0, "physical gate was not model blind")
    require(physical.get("exact_runtime_contract_sha256") == registration["exact_e004_pi05_runtime"]["contract_sha256"], "physical gate runtime contract changed")
    smoke = _bound_json(gate.get("excluded_smoke_gate"), "excluded smoke gate", schema=SMOKE_GATE_SCHEMA, status="passed_excluded_four_cell_smoke")
    require(smoke.get("excluded_from_behavioral_denominators") is True and smoke.get("completed_cells") == 4, "smoke was not an excluded four-cell block")
    isolation = _bound_json(gate.get("two_lane_isolation_gate"), "two-lane isolation gate", schema=ISOLATION_GATE_SCHEMA, status="passed_two_lane_fixed_observation_isolation")
    require(isolation.get("fixed_observation_equal") is True and isolation.get("fixed_prompt_equal") is True and isolation.get("request_seed_equal") is True, "isolation inputs differed")
    require(isolation.get("outputs_match") is True and isolation.get("lane_state_isolated") is True, "two-lane isolation did not pass")
    lanes = gate.get("lane_manifests")
    require(isinstance(lanes, list) and len(lanes) >= 2, "release gate lacks exact lane manifests")
    lane_values = [_bound_json(record, f"lane manifest {index}", schema=LANE_GATE_SCHEMA, status="passed_lane_release") for index, record in enumerate(lanes)]
    require(len({value.get("lane_id") for value in lane_values}) == len(lane_values), "lane IDs are not unique")
    require(len({value.get("simulator_pod_uid") for value in lane_values}) == len(lane_values), "simulator pods are not isolated")
    require(len({value.get("simulator_gpu_uuid") for value in lane_values}) == len(lane_values), "simulator GPUs are not isolated")
    for value in lane_values:
        for key in ("simulator_pod_uid", "simulator_gpu_uuid", "policy_server_pod_uid", "policy_server_gpu_uuid", "container_identity", "runtime_identity", "raw_root", "server_process_identity", "server_lock_identity"):
            require(isinstance(value.get(key), str) and value.get(key), f"lane manifest lacks {key}")
        require(type(value.get("server_port")) is int and 1024 <= value["server_port"] <= 65535, "lane server port is invalid")
        require(value.get("source_commit") == source_gate.get("source_commit"), "lane source commit differs from pushed source")
        require(value.get("exact_runtime_contract_sha256") == registration["exact_e004_pi05_runtime"]["contract_sha256"], "lane exact runtime contract changed")
        require(value.get("registration_sha256") == sha256_file(registration_path) and value.get("queue_sha256") == sha256_file(queue_path), "lane registration/queue identity changed")
    require(len({value["server_port"] for value in lane_values}) == len(lane_values), "lane server ports are not unique")
    require(len({value["raw_root"] for value in lane_values}) == len(lane_values), "lane raw roots are not unique")
    require(len({value["policy_server_pod_uid"] for value in lane_values}) == len(lane_values), "policy server pods are not isolated")
    require(len({value["policy_server_gpu_uuid"] for value in lane_values}) == len(lane_values), "policy server GPUs are not isolated")
    require(len({value["server_process_identity"] for value in lane_values}) == len(lane_values), "policy server processes are not isolated")
    require(len({value["server_lock_identity"] for value in lane_values}) == len(lane_values), "policy server locks are not isolated")
    allocated_gpu_uuids = [value[key] for value in lane_values for key in ("simulator_gpu_uuid", "policy_server_gpu_uuid")]
    require(len(set(allocated_gpu_uuids)) == len(allocated_gpu_uuids), "a GPU UUID is shared across C002 lane roles")
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
