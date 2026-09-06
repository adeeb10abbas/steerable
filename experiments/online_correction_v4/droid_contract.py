"""Fail-closed launch contracts for the V4 DROID/RoboLab live adapter layer."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.online_correction_v4.contracts import EpisodeManifestRow, TimingConfig


STUDY_ID = "online_correction_v4"
ADAPTER_CONTRACT_FILES: tuple[str, ...] = (
    "experiments/online_correction_v4/droid_contract.py",
    "experiments/online_correction_v4/droid_reset.py",
    "experiments/online_correction_v4/droid_simulator.py",
    "experiments/online_correction_v4/droid_nano_policy.py",
    "experiments/online_correction_v4/droid_pi05_policy.py",
    "experiments/online_correction_v4/droid_bindings.py",
)

NANO_POLICY_ID = "cosmos3_nano_droid"
PI05_POLICY_ID = "pi05_droid"
SUPPORTED_DROID_POLICIES = (NANO_POLICY_ID, PI05_POLICY_ID)

NANO_ACTION_CHUNK_STEPS = 32
PI05_ACTION_CHUNK_STEPS = 15
ACTION_DIM = 8
NANO_ACTION_SHAPE = (NANO_ACTION_CHUNK_STEPS, ACTION_DIM)
PI05_ACTION_SHAPE = (PI05_ACTION_CHUNK_STEPS, ACTION_DIM)

RESET_SCHEMA = "v4-droid-reset-attestation-v1"
WRITER_CONTRACT_SCHEMA = "v4-droid-writer-contract-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
TODO_PREFIX = "TODO_"


class PrefixMode(str, Enum):
    DETERMINISTIC_FRESH_SESSION_REPLAY = "deterministic_fresh_session_replay"
    QUALIFIED_FULL_STATE_SNAPSHOT = "qualified_full_state_snapshot"
    INDEPENDENT_NATURAL_ROLLOUT_FALLBACK = "independent_natural_rollout_fallback"
    FRESH_SESSION_DETERMINISTIC_REPLAY = DETERMINISTIC_FRESH_SESSION_REPLAY
    COMPLETE_STATE_SNAPSHOT_BRANCH = QUALIFIED_FULL_STATE_SNAPSHOT


class DroidContractError(ValueError):
    """Raised before live execution when any frozen binding is missing or invalid."""


def _fail(message: str) -> None:
    raise DroidContractError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_adapter_contract_sha256(study_root: Path) -> str:
    root = Path(study_root).resolve()
    inventory: list[dict[str, str]] = []
    for relative in ADAPTER_CONTRACT_FILES:
        path = root / relative
        if not path.is_file():
            _fail(f"missing V4 DROID adapter source: {relative}")
        inventory.append({"path": relative, "sha256": sha256_file(path)})
    return sha256_bytes(canonical_json_bytes(inventory))


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _reject_todo(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip() or value.startswith(TODO_PREFIX):
        _fail(f"{label} is missing or unreleased")


@dataclass(frozen=True)
class WriterContract:
    schema_version: str
    output_parent_uri: str
    viewport_video_required: bool
    write_once_attempt_directories: bool
    incremental_fsync_required: bool
    required_streams: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> WriterContract:
        if value.get("schema_version") != WRITER_CONTRACT_SCHEMA:
            _fail("writer contract schema mismatch")
        _reject_todo(value.get("output_parent_uri"), "writer contract output_parent_uri")
        streams = value.get("required_streams")
        if not isinstance(streams, list) or not streams:
            _fail("writer contract required_streams must be a nonempty list")
        return cls(
            schema_version=str(value["schema_version"]),
            output_parent_uri=str(value["output_parent_uri"]),
            viewport_video_required=bool(value.get("viewport_video_required", True)),
            write_once_attempt_directories=bool(value.get("write_once_attempt_directories", True)),
            incremental_fsync_required=bool(value.get("incremental_fsync_required", True)),
            required_streams=tuple(str(item) for item in streams),
        )


@dataclass(frozen=True)
class PolicyRuntimeBinding:
    policy_id: str
    checkpoint_sha256: str
    checkpoint_uri: str
    runtime_image_digest: str
    integration_commit: str
    native_control_dt_s: float
    achieved_delay_s: float
    achieved_standard_query_period_s: float
    achieved_fast_query_period_s: float
    prediction_horizon_actions: int
    policy_reset_and_history_contract_uri: str

    @classmethod
    def from_runtime_lock(cls, policy_id: str, value: Mapping[str, Any]) -> PolicyRuntimeBinding:
        _require_sha256(value.get("checkpoint_sha256"), f"policy {policy_id}.checkpoint_sha256")
        for key in (
            "checkpoint_uri",
            "runtime_image_digest",
            "policy_reset_and_history_contract_uri",
        ):
            _reject_todo(value.get(key), f"policy {policy_id}.{key}")
        integration = value.get("integration_commit")
        if not isinstance(integration, str) or not HEX40_RE.fullmatch(integration):
            _fail(f"policy {policy_id}.integration_commit must be a 40-char git commit")
        dt = value.get("native_control_dt_s")
        if not isinstance(dt, (int, float)) or isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            _fail(f"policy {policy_id}.native_control_dt_s must be a positive finite number")
        achieved_keys = (
            "achieved_delay_s",
            "achieved_standard_query_period_s",
            "achieved_fast_query_period_s",
        )
        achieved: dict[str, float] = {}
        for key in achieved_keys:
            item = value.get(key)
            if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(item) or item <= 0:
                _fail(f"policy {policy_id}.{key} must be a positive finite number")
            achieved[key] = float(item)
        horizon = value.get("prediction_horizon_actions")
        if type(horizon) is not int or horizon <= 0:
            _fail(f"policy {policy_id}.prediction_horizon_actions must be a positive integer")
        return cls(
            policy_id=policy_id,
            checkpoint_sha256=_require_sha256(value["checkpoint_sha256"], "checkpoint_sha256"),
            checkpoint_uri=str(value["checkpoint_uri"]),
            runtime_image_digest=str(value["runtime_image_digest"]),
            integration_commit=integration,
            native_control_dt_s=float(dt),
            achieved_delay_s=achieved["achieved_delay_s"],
            achieved_standard_query_period_s=achieved["achieved_standard_query_period_s"],
            achieved_fast_query_period_s=achieved["achieved_fast_query_period_s"],
            prediction_horizon_actions=horizon,
            policy_reset_and_history_contract_uri=str(value["policy_reset_and_history_contract_uri"]),
        )


@dataclass(frozen=True)
class FixtureRuntimeBinding:
    fixture_id: str
    geometry_sha256: str
    scorer_sha256: str
    reset_registry_sha256: str
    geometry_uri: str
    scorer_uri: str
    reset_registry_uri: str
    calibration_scale: float
    d_cap_m: float

    @classmethod
    def from_runtime_lock(cls, fixture_id: str, value: Mapping[str, Any]) -> FixtureRuntimeBinding:
        for key in ("geometry_sha256", "scorer_sha256", "reset_registry_sha256"):
            _require_sha256(value.get(key), f"fixture {fixture_id}.{key}")
        for key in (
            "geometry_uri",
            "scorer_uri",
            "reset_registry_uri",
            "frame_transform_uri",
            "goal_geometry_and_tolerances_uri",
            "trigger_release_detector_uri",
            "intervention_trajectory_registry_uri",
            "scoring_and_visibility_thresholds_uri",
        ):
            _reject_todo(value.get(key), f"fixture {fixture_id}.{key}")
        scale = value.get("calibration_scale")
        d_cap = value.get("D_cap_m")
        if not isinstance(scale, (int, float)) or isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
            _fail(f"fixture {fixture_id}.calibration_scale must be positive")
        if not isinstance(d_cap, (int, float)) or isinstance(d_cap, bool) or not math.isfinite(d_cap) or d_cap <= 0:
            _fail(f"fixture {fixture_id}.D_cap_m must be positive")
        return cls(
            fixture_id=fixture_id,
            geometry_sha256=str(value["geometry_sha256"]),
            scorer_sha256=str(value["scorer_sha256"]),
            reset_registry_sha256=str(value["reset_registry_sha256"]),
            geometry_uri=str(value["geometry_uri"]),
            scorer_uri=str(value["scorer_uri"]),
            reset_registry_uri=str(value["reset_registry_uri"]),
            calibration_scale=float(scale),
            d_cap_m=float(d_cap),
        )


@dataclass(frozen=True)
class RuntimeLockBinding:
    schema_version: int
    campaign_id: str
    config_sha256: str
    manifest_sha256: str
    release_status: str
    released_families: tuple[str, ...]
    runner_entrypoint: str
    runner_sha256: str
    prefix_mode: PrefixMode
    prefix_mode_receipt_sha256: str
    writer_contract: WriterContract
    policies: dict[str, PolicyRuntimeBinding]
    fixtures: dict[str, FixtureRuntimeBinding]
    raw: dict[str, Any]

    @property
    def is_released(self) -> bool:
        return self.release_status == "RELEASED" and bool(self.released_families)

    @property
    def is_pilot_released(self) -> bool:
        return (
            self.release_status == "PILOT_RELEASED"
            and bool(self.released_families)
        )


@dataclass(frozen=True)
class LaunchArgs:
    manifest_path: Path
    runtime_lock_path: Path
    episode_id: str
    attempt_id: str
    output_dir: Path
    dry_run: bool
    validate_only: bool
    policy_host: str | None = None
    policy_port: int | None = None


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DroidContractError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    return value


def load_manifest_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DroidContractError(f"invalid manifest row {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            _fail(f"manifest row {line_number} must be an object")
        rows.append(row)
    if not rows:
        _fail("manifest must contain at least one episode row")
    return rows, digest


def find_manifest_row(rows: Sequence[Mapping[str, Any]], episode_id: str) -> dict[str, Any]:
    matches = [dict(row) for row in rows if row.get("episode_id") == episode_id]
    if not matches:
        _fail(f"episode_id is not registered in manifest: {episode_id}")
    if len(matches) > 1:
        _fail(f"episode_id is not unique in manifest: {episode_id}")
    return matches[0]


def validate_runtime_lock(
    lock_path: Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_config_sha256: str | None = None,
) -> RuntimeLockBinding:
    if not lock_path.is_file():
        _fail(f"runtime lock is missing: {lock_path}")
    raw = load_json(lock_path, "runtime lock")
    if raw.get("schema_version") != 1:
        _fail("runtime lock schema_version must be 1")
    if raw.get("campaign_id") != STUDY_ID:
        _fail("runtime lock campaign_id mismatch")
    config_sha = _require_sha256(raw.get("config_sha256"), "runtime lock config_sha256")
    manifest_sha = _require_sha256(raw.get("manifest_sha256"), "runtime lock manifest_sha256")
    if expected_config_sha256 is not None and config_sha != expected_config_sha256:
        _fail("runtime lock config_sha256 does not match campaign config")
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        _fail("runtime lock manifest_sha256 does not match supplied manifest")
    runner = raw.get("runner")
    if not isinstance(runner, dict):
        _fail("runtime lock runner binding is required")
    _reject_todo(runner.get("entrypoint"), "runtime lock runner.entrypoint")
    runner_sha = _require_sha256(runner.get("sha256"), "runtime lock runner.sha256")
    prefix_mode_raw = raw.get("prefix_mode")
    try:
        prefix_mode = PrefixMode(str(prefix_mode_raw))
    except ValueError as exc:
        raise DroidContractError(
            "runtime lock prefix_mode must be one of "
            f"{[item.value for item in PrefixMode]}"
        ) from exc
    prefix_receipt = raw.get("prefix_mode_receipt_sha256")
    _require_sha256(prefix_receipt, "runtime lock prefix_mode_receipt_sha256")
    writer_raw = raw.get("writer_contract")
    if not isinstance(writer_raw, dict):
        _fail("runtime lock writer_contract is required")
    writer = WriterContract.from_mapping(writer_raw)
    policies_raw = raw.get("policies")
    fixtures_raw = raw.get("fixtures")
    if not isinstance(policies_raw, dict) or not isinstance(fixtures_raw, dict):
        _fail("runtime lock policies and fixtures must be objects")
    policies = {
        name: PolicyRuntimeBinding.from_runtime_lock(name, value)
        for name, value in policies_raw.items()
        if name in SUPPORTED_DROID_POLICIES
    }
    fixtures = {
        name: FixtureRuntimeBinding.from_runtime_lock(name, value)
        for name, value in fixtures_raw.items()
    }
    released = tuple(str(item) for item in raw.get("released_families", ()))
    return RuntimeLockBinding(
        schema_version=int(raw["schema_version"]),
        campaign_id=str(raw["campaign_id"]),
        config_sha256=config_sha,
        manifest_sha256=manifest_sha,
        release_status=str(raw.get("release_status", "NOT_RELEASED")),
        released_families=released,
        runner_entrypoint=str(runner["entrypoint"]),
        runner_sha256=runner_sha,
        prefix_mode=prefix_mode,
        prefix_mode_receipt_sha256=str(prefix_receipt),
        writer_contract=writer,
        policies=policies,
        fixtures=fixtures,
        raw=raw,
    )


def validate_manifest_row_against_lock(
    row: Mapping[str, Any],
    *,
    lock: RuntimeLockBinding,
    manifest_sha256: str,
) -> EpisodeManifestRow:
    if lock.manifest_sha256 != manifest_sha256:
        _fail("manifest SHA-256 does not match runtime lock binding")
    manifest = EpisodeManifestRow.from_manifest_dict(row)
    if manifest.campaign != lock.campaign_id:
        _fail("manifest row campaign mismatch")
    if manifest.family not in lock.released_families:
        _fail(f"family {manifest.family} is not released in runtime lock")
    policy_id = manifest.factors.get("policy")
    if policy_id not in SUPPORTED_DROID_POLICIES:
        _fail(f"episode policy {policy_id!r} is not a supported DROID adapter policy")
    if policy_id not in lock.policies:
        _fail(f"runtime lock lacks released policy binding for {policy_id}")
    if manifest.fixture not in lock.fixtures:
        _fail(f"runtime lock lacks fixture binding for {manifest.fixture}")
    return manifest


def validate_prefix_mode_for_family(prefix_mode: PrefixMode, family_id: str) -> None:
    if family_id == "C2" and prefix_mode is PrefixMode.INDEPENDENT_NATURAL_ROLLOUT_FALLBACK:
        _fail("C2 requires verified common prefixes; independent rollout fallback is blocked")


def validate_launch_args(args: LaunchArgs) -> None:
    for path, label in (
        (args.manifest_path, "manifest"),
        (args.runtime_lock_path, "runtime lock"),
    ):
        if not path.is_file():
            _fail(f"{label} path does not exist: {path}")
    if not args.episode_id.strip():
        _fail("episode_id is required")
    if not args.attempt_id.strip():
        _fail("attempt_id is required")
    if not args.output_dir.is_absolute():
        _fail("output_dir must be an absolute path")
    if args.dry_run and not args.validate_only and args.output_dir.exists():
        _fail("dry-run refuses an existing output directory")


def build_launch_plan(
    args: LaunchArgs,
    *,
    study_root: Path,
    campaign_config_path: Path,
) -> dict[str, Any]:
    validate_launch_args(args)
    config_raw = load_json(campaign_config_path, "campaign config")
    config_sha = sha256_file(campaign_config_path)
    rows, manifest_sha = load_manifest_rows(args.manifest_path)
    lock = validate_runtime_lock(
        args.runtime_lock_path,
        expected_config_sha256=config_sha,
        expected_manifest_sha256=manifest_sha,
    )
    row = find_manifest_row(rows, args.episode_id)
    manifest = validate_manifest_row_against_lock(row, lock=lock, manifest_sha256=manifest_sha)
    if not lock.is_released:
        if not lock.is_pilot_released:
            _fail("runtime lock is not released for live execution")
        if manifest.cohort != "engineering_pilot":
            _fail(
                "PILOT_RELEASED runtime lock authorizes only engineering_pilot rows"
            )
    validate_prefix_mode_for_family(lock.prefix_mode, manifest.family)
    policy_id = manifest.factors["policy"]
    policy_binding = lock.policies[policy_id]
    fixture_binding = lock.fixtures[manifest.fixture]
    adapter_sha = compute_adapter_contract_sha256(study_root)
    timing = TimingConfig.from_mapping(config_raw["timing"])
    return {
        "schema_version": "v4-droid-launch-plan-v1",
        "study_id": STUDY_ID,
        "episode_id": manifest.episode_id,
        "attempt_id": args.attempt_id,
        "family": manifest.family,
        "fixture": manifest.fixture,
        "policy_id": policy_id,
        "prefix_mode": lock.prefix_mode.value,
        "prefix_group_id": manifest.prefix_group_id,
        "env_seed": manifest.env_seed,
        "policy_seed": manifest.policy_seed,
        "config_sha256": lock.config_sha256,
        "manifest_sha256": lock.manifest_sha256,
        "adapter_contract_sha256": adapter_sha,
        "runner_entrypoint": lock.runner_entrypoint,
        "runner_sha256": lock.runner_sha256,
        "writer_contract": writer_contract_summary(lock.writer_contract),
        "policy_binding": policy_binding.__dict__,
        "fixture_binding": fixture_binding.__dict__,
        "timing": timing.__dict__,
        "dry_run": args.dry_run,
        "validate_only": args.validate_only,
        "output_dir": str(args.output_dir),
    }


def writer_contract_summary(writer: WriterContract) -> dict[str, Any]:
    return {
        "schema_version": writer.schema_version,
        "output_parent_uri": writer.output_parent_uri,
        "viewport_video_required": writer.viewport_video_required,
        "write_once_attempt_directories": writer.write_once_attempt_directories,
        "required_streams": list(writer.required_streams),
    }


def expected_action_shape(policy_id: str) -> tuple[int, int]:
    if policy_id == NANO_POLICY_ID:
        return NANO_ACTION_SHAPE
    if policy_id == PI05_POLICY_ID:
        return PI05_ACTION_SHAPE
    _fail(f"unsupported policy action shape lookup: {policy_id}")
    raise AssertionError("unreachable")
