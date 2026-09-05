"""V4 campaign coordinator: dependency-aware dispatch planning and lane rendering.

Plans shard independent execution groups across qualified hardware strata, render
fresh immutable Kubernetes lane bundles, and optionally create cluster objects.
Never applies or patches existing Jobs. Fails closed when the runtime lock is not
released. Scheduling uses only technical receipt status, never behavioral outcomes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Optional, Sequence

from experiments.online_correction_v4.attempts import AttemptClassifier, RetryPolicy
from experiments.online_correction_v4.contracts import GroupExecutionState
from experiments.online_correction_v4.droid_contract import (
    DroidContractError,
    RuntimeLockBinding,
    sha256_file,
    validate_runtime_lock,
)
from experiments.online_correction_v4.leases import GroupLease, GroupLeaseStore, LeaseConflict
from experiments.online_correction_v4.registry import CampaignRegistry, ExecutionGroup
from experiments.online_correction_v4.scheduler import CampaignScheduler


SCHEMA_VERSION = "v4-campaign-coordinator-plan-v1"
COORDINATION_STATE_SCHEMA = "v4-coordination-state-v1"
DISPATCH_MANIFEST_SCHEMA = "v4-lane-dispatch-v1"
GROUP_RECEIPT_SUFFIX = ".group_receipt.json"
K8S_NAME_RE = re.compile(r"^\s+name:\s+(.+)\s*$", re.MULTILINE)
LANE_DISPATCH_SCRIPT = "scripts/run_online_correction_v4_lane_dispatch.py"
SIMULATOR_LAUNCH_CONFIG_MOUNT = "/opt/v4-lane/config/simulator-launch.json"
NOOP_SIMULATOR_ARGV = frozenset({("/usr/bin/true",), ("/bin/true",)})
QUALIFICATION_SCOPE = "infrastructure_qualification_only_no_scientific_behavior"
V4_RUNNER_MARKER = "online_correction_v4"


class DispatchMode(str, Enum):
    BEHAVIORAL = "behavioral"
    QUALIFICATION_ONLY = "qualification_only"


class CoordinatorError(Exception):
    """Base error for campaign coordination."""


class CoordinatorBlockedError(CoordinatorError):
    """Raised when release or binding preconditions are not satisfied."""


@dataclass(frozen=True)
class ClusterBinding:
    kube_context: str
    namespace: str
    pvc: str
    output_parent: str

    def validate_for_create(self) -> None:
        for label, value in (
            ("kube_context", self.kube_context),
            ("namespace", self.namespace),
            ("pvc", self.pvc),
            ("output_parent", self.output_parent),
        ):
            if not isinstance(value, str) or not value.strip():
                raise CoordinatorBlockedError(f"{label} is required for --create")
        if not Path(self.output_parent).is_absolute():
            raise CoordinatorBlockedError("output_parent must be an absolute path")


@dataclass(frozen=True)
class ExecutionConfig:
    max_infra_retries: int = 3
    lane_quarantine_threshold: int = 5
    authorized_storage_bytes: Optional[int] = None
    estimated_bytes_per_episode: int = 500_000_000
    estimated_bytes_per_infra_retry: int = 50_000_000

    @classmethod
    def from_launch_matrix(cls, payload: Mapping[str, Any]) -> ExecutionConfig:
        dispatch = payload.get("dispatch") or {}
        budget = payload.get("resource_budget") or {}
        return cls(
            max_infra_retries=int(dispatch.get("max_infra_retries_per_episode", 3)),
            lane_quarantine_threshold=int(dispatch.get("lane_quarantine_threshold", 5)),
            authorized_storage_bytes=(
                int(budget["authorized_storage_bytes"])
                if budget.get("authorized_storage_bytes") is not None
                else None
            ),
            estimated_bytes_per_episode=int(
                budget.get("estimated_bytes_per_episode", 500_000_000)
            ),
            estimated_bytes_per_infra_retry=int(
                budget.get("estimated_bytes_per_infra_retry", 50_000_000)
            ),
        )


@dataclass(frozen=True)
class LaneStratum:
    lane_id: str
    hardware_stratum: str
    spec_template: dict[str, Any]
    template_root: Path

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], *, repo_root: Path) -> LaneStratum:
        lane_id = str(row.get("lane_id") or "")
        hardware_stratum = str(row.get("hardware_stratum") or "")
        if not lane_id or not hardware_stratum:
            raise CoordinatorBlockedError("qualified_lanes entries require lane_id and hardware_stratum")
        template = row.get("lane_spec")
        template_path = row.get("lane_spec_template_path")
        if isinstance(template, dict):
            spec_template = dict(template)
            template_root = (repo_root / "deploy/k8s/v4_lane_bundle").resolve()
        elif isinstance(template_path, str) and template_path.strip():
            path = Path(template_path)
            if not path.is_absolute():
                path = (repo_root / path).resolve()
            if not path.is_file():
                raise CoordinatorBlockedError(f"lane spec template missing: {path}")
            spec_template = json.loads(path.read_text(encoding="utf-8"))
            template_root = path.parent.resolve()
        else:
            raise CoordinatorBlockedError(
                f"lane {lane_id}: lane_spec or lane_spec_template_path is required"
            )
        return cls(
            lane_id=lane_id,
            hardware_stratum=hardware_stratum,
            spec_template=spec_template,
            template_root=template_root,
        )


@dataclass(frozen=True)
class GroupReceipt:
    group_id: str
    manifest_sha256: str
    accepted_episode_ids: tuple[str, ...]
    partial_episode_ids: tuple[str, ...]
    status: str
    receipt_path: Path

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"


@dataclass(frozen=True)
class TeardownObject:
    kind: str
    name: str
    namespace: str
    labels: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "namespace": self.namespace,
            "labels": dict(self.labels),
        }


@dataclass(frozen=True)
class TeardownInventory:
    kube_context: str
    objects: tuple[TeardownObject, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "kube_context": self.kube_context,
            "objects": [item.as_dict() for item in self.objects],
        }


@dataclass(frozen=True)
class CoordinationState:
    lane_infra_failures: Counter[str]
    episode_infra_failures: Counter[str]
    reserved_attempt_ids: frozenset[str]
    attempt_index: int

    @classmethod
    def empty(cls, *, attempt_index: int = 1) -> CoordinationState:
        return cls(
            lane_infra_failures=Counter(),
            episode_infra_failures=Counter(),
            reserved_attempt_ids=frozenset(),
            attempt_index=attempt_index,
        )


@dataclass(frozen=True)
class LaneAssignment:
    lane_id: str
    hardware_stratum: str
    attempt_id: str
    group_ids: tuple[str, ...]
    remaining_episode_ids: tuple[str, ...]
    spec_path: Optional[str]
    bundle_root: Optional[str]
    launch_hash: Optional[str]
    teardown_inventory: Optional[dict[str, Any]]
    dispatch_manifest_path: Optional[str] = None
    acquired_group_leases: tuple[str, ...] = ()
    pvc_binding_root: Optional[str] = None
    create_status: str = "planned"


@dataclass(frozen=True)
class CampaignPlan:
    schema_version: str
    release_status: str
    manifest_sha256: str
    queue_sha256: str
    config_sha256: str
    dispatchable_group_count: int
    pending_group_count: int
    complete_group_count: int
    partial_group_count: int
    lane_assignments: tuple[LaneAssignment, ...]
    quarantined_lanes: tuple[str, ...]
    blocked_groups: tuple[dict[str, Any], ...]
    storage_budget: dict[str, Any]
    resume_receipt_count: int
    scheduling_inputs: dict[str, Any]
    dispatch_mode: str
    behavioral_episode_count: int
    effect_size_peeking: bool = False
    partial_create_wave: bool = False
    create_wave_receipt: Optional[dict[str, Any]] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "release_status": self.release_status,
            "manifest_sha256": self.manifest_sha256,
            "queue_sha256": self.queue_sha256,
            "config_sha256": self.config_sha256,
            "dispatchable_group_count": self.dispatchable_group_count,
            "pending_group_count": self.pending_group_count,
            "complete_group_count": self.complete_group_count,
            "partial_group_count": self.partial_group_count,
            "dispatch_mode": self.dispatch_mode,
            "behavioral_episode_count": self.behavioral_episode_count,
            "lane_assignments": [
                {
                    "lane_id": item.lane_id,
                    "hardware_stratum": item.hardware_stratum,
                    "attempt_id": item.attempt_id,
                    "group_ids": list(item.group_ids),
                    "remaining_episode_ids": list(item.remaining_episode_ids),
                    "spec_path": item.spec_path,
                    "bundle_root": item.bundle_root,
                    "launch_hash": item.launch_hash,
                    "teardown_inventory": item.teardown_inventory,
                    "dispatch_manifest_path": item.dispatch_manifest_path,
                    "acquired_group_leases": list(item.acquired_group_leases),
                    "pvc_binding_root": item.pvc_binding_root,
                    "create_status": item.create_status,
                }
                for item in self.lane_assignments
            ],
            "partial_create_wave": self.partial_create_wave,
            "create_wave_receipt": dict(self.create_wave_receipt) if self.create_wave_receipt else None,
            "quarantined_lanes": list(self.quarantined_lanes),
            "blocked_groups": list(self.blocked_groups),
            "storage_budget": dict(self.storage_budget),
            "resume_receipt_count": self.resume_receipt_count,
            "scheduling_inputs": dict(self.scheduling_inputs),
            "effect_size_peeking": self.effect_size_peeking,
        }


@dataclass
class CoordinatorInputs:
    runtime_lock_path: Path
    queue_path: Path
    queue_manifest_path: Path
    launch_matrix_path: Path
    campaign_config_path: Path
    group_receipts_dir: Optional[Path] = None
    evidence_root: Optional[Path] = None
    lane_infra_failures: Counter[str] = field(default_factory=Counter)
    episode_infra_failures: Counter[str] = field(default_factory=Counter)
    coordination_state_path: Optional[Path] = None
    group_lease_root: Optional[Path] = None
    render_output_root: Optional[Path] = None
    cluster_binding: Optional[ClusterBinding] = None
    execution_config: ExecutionConfig = field(default_factory=ExecutionConfig)
    attempt_index: int = 1
    qualification_only: bool = False
    repo_root: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.repo_root is None:
            self.repo_root = Path(__file__).resolve().parents[2]
        if self.coordination_state_path is not None:
            state = load_coordination_state(self.coordination_state_path)
            self.lane_infra_failures = Counter(state.lane_infra_failures)
            self.episode_infra_failures = Counter(state.episode_infra_failures)
            if self.attempt_index == 1 and state.attempt_index > 1:
                self.attempt_index = state.attempt_index


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CoordinatorBlockedError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CoordinatorBlockedError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CoordinatorBlockedError(f"{label} must be a JSON object")
    return payload


def load_group_receipts(receipts_dir: Path) -> list[GroupReceipt]:
    if not receipts_dir.is_dir():
        return []
    receipts: list[GroupReceipt] = []
    for path in sorted(receipts_dir.glob(f"*{GROUP_RECEIPT_SUFFIX}")):
        payload = load_json(path, "group receipt")
        group_id = str(payload.get("group_id") or "")
        manifest_sha = str(payload.get("manifest_sha256") or "")
        status = str(payload.get("status") or "")
        if not group_id or not manifest_sha or status not in {"complete", "partial"}:
            raise CoordinatorError(f"invalid group receipt: {path}")
        accepted = tuple(str(item) for item in payload.get("accepted_episode_ids") or ())
        partial = tuple(str(item) for item in payload.get("partial_episode_ids") or ())
        receipts.append(
            GroupReceipt(
                group_id=group_id,
                manifest_sha256=manifest_sha,
                accepted_episode_ids=accepted,
                partial_episode_ids=partial,
                status=status,
                receipt_path=path,
            )
        )
    return receipts


def load_coordination_state(path: Path) -> CoordinationState:
    payload = load_json(path, "coordination state")
    if payload.get("schema_version") != COORDINATION_STATE_SCHEMA:
        raise CoordinatorBlockedError(
            f"coordination state schema must be {COORDINATION_STATE_SCHEMA}"
        )
    lane_failures = Counter(
        {str(k): int(v) for k, v in (payload.get("lane_infra_failures") or {}).items()}
    )
    episode_failures = Counter(
        {str(k): int(v) for k, v in (payload.get("episode_infra_failures") or {}).items()}
    )
    reserved = frozenset(str(item) for item in payload.get("reserved_attempt_ids") or ())
    attempt_index = int(payload.get("attempt_index", 1))
    if attempt_index < 1:
        raise CoordinatorBlockedError("coordination state attempt_index must be >= 1")
    return CoordinationState(
        lane_infra_failures=lane_failures,
        episode_infra_failures=episode_failures,
        reserved_attempt_ids=reserved,
        attempt_index=attempt_index,
    )


def coordination_state_from_inputs(inputs: CoordinatorInputs) -> CoordinationState:
    if inputs.coordination_state_path is not None:
        return load_coordination_state(inputs.coordination_state_path)
    return CoordinationState(
        lane_infra_failures=Counter(inputs.lane_infra_failures),
        episode_infra_failures=Counter(inputs.episode_infra_failures),
        reserved_attempt_ids=frozenset(),
        attempt_index=inputs.attempt_index,
    )


def load_durable_group_leases(
    lease_root: Path,
    *,
    manifest_sha256: str,
) -> dict[str, GroupLease]:
    leases_dir = lease_root / "leases"
    if not leases_dir.is_dir():
        return {}
    active: dict[str, GroupLease] = {}
    for path in sorted(leases_dir.glob("*.lease")):
        payload = load_json(path, "group lease")
        group_id = str(payload.get("group_id") or "")
        owner_lane = str(payload.get("owner_lane") or "")
        attempt_id = str(payload.get("attempt_id") or "")
        lease_manifest = str(payload.get("manifest_sha256") or "")
        if not group_id or not owner_lane or not attempt_id:
            raise CoordinatorError(f"invalid durable group lease: {path}")
        if lease_manifest != manifest_sha256:
            raise CoordinatorBlockedError(
                f"durable group lease manifest mismatch for {group_id}"
            )
        active[group_id] = GroupLease(
            group_id=group_id,
            owner_lane=owner_lane,
            attempt_id=attempt_id,
            acquired_at_monotonic=float(payload.get("acquired_at_monotonic", 0.0)),
            lease_path=path,
            manifest_sha256=lease_manifest,
        )
    return active


def apply_durable_group_leases(scheduler: CampaignScheduler, leases: Mapping[str, GroupLease]) -> None:
    for group_id, lease in leases.items():
        if group_id not in scheduler.registry.by_execution_group:
            raise CoordinatorBlockedError(f"unknown group in durable lease: {group_id}")
        scheduler.active_leases[group_id] = lease
        scheduler.group_states[group_id] = GroupExecutionState.LEASED


def collect_reserved_attempt_ids(
    *,
    coordination_state: CoordinationState,
    lease_root: Optional[Path],
    render_output_root: Optional[Path],
) -> set[str]:
    reserved = set(coordination_state.reserved_attempt_ids)
    if lease_root is not None:
        leases_dir = lease_root / "leases"
        if leases_dir.is_dir():
            for path in leases_dir.glob("*.lease"):
                payload = json.loads(path.read_text(encoding="utf-8"))
                attempt_id = payload.get("attempt_id")
                if isinstance(attempt_id, str) and attempt_id:
                    reserved.add(attempt_id)
    if render_output_root is not None and render_output_root.is_dir():
        for child in render_output_root.iterdir():
            parts = child.name.split("-")
            for part in parts:
                if part.startswith("attempt") and part[7:].isdigit():
                    reserved.add(part)
    return reserved


def allocate_attempt_id(
    *,
    reserved: set[str],
    start_index: int,
) -> tuple[str, int]:
    index = start_index
    while True:
        attempt_id = f"attempt{index:04d}"
        if attempt_id not in reserved:
            reserved.add(attempt_id)
            return attempt_id, index + 1
        index += 1


def _contains_v4_runner(path: str) -> bool:
    lowered = path.lower()
    return V4_RUNNER_MARKER in lowered and "online_correction_v4.py" not in lowered


def assert_behavioral_lane_spec(spec: Mapping[str, Any]) -> None:
    if spec.get("qualification_only") is True:
        raise CoordinatorBlockedError(
            "qualification_only lane spec cannot masquerade as behavioral dispatch"
        )
    simulator = spec.get("simulator")
    if not isinstance(simulator, dict):
        raise CoordinatorBlockedError("behavioral lane spec requires simulator bindings")
    argv = tuple(simulator.get("experiment_argv") or ())
    if argv in NOOP_SIMULATOR_ARGV:
        raise CoordinatorBlockedError(
            "qualification /usr/bin/true simulator argv cannot masquerade as behavioral dispatch"
        )
    binding_paths = {
        str(row.get("path"))
        for row in simulator.get("file_bindings") or []
        if isinstance(row, dict) and row.get("path")
    }
    argv_paths = {item for item in argv if isinstance(item, str) and item.startswith("/")}
    runner_hits = [path for path in argv_paths | binding_paths if _contains_v4_runner(path)]
    if not runner_hits:
        raise CoordinatorBlockedError(
            "behavioral lane spec must bind an online_correction_v4 lane dispatch runner"
        )
    dispatch = spec.get("dispatch")
    if not isinstance(dispatch, dict):
        raise CoordinatorBlockedError("behavioral lane spec requires immutable dispatch assignment")
    for key in ("group_ids", "episode_ids", "dispatch_manifest_path"):
        if key not in dispatch:
            raise CoordinatorBlockedError(f"behavioral lane spec dispatch missing {key}")
    manifest_path = str(dispatch["dispatch_manifest_path"])
    if manifest_path not in binding_paths:
        raise CoordinatorBlockedError("behavioral lane spec must bind dispatch manifest path")
    if manifest_path not in argv_paths:
        raise CoordinatorBlockedError("behavioral lane spec argv must reference dispatch manifest path")


def assert_qualification_lane_spec(spec: Mapping[str, Any]) -> None:
    if spec.get("qualification_only") is not True:
        raise CoordinatorBlockedError("qualification lane spec must set qualification_only=true")
    argv = tuple((spec.get("simulator") or {}).get("experiment_argv") or ())
    if argv not in NOOP_SIMULATOR_ARGV:
        raise CoordinatorBlockedError("qualification lane spec must keep /usr/bin/true simulator argv")


def validate_rendered_bundle_scope(bundle_root: Path, *, behavioral: bool) -> None:
    configmap_path = bundle_root / "configmap.yaml"
    if not configmap_path.is_file():
        raise CoordinatorError(f"rendered bundle missing configmap: {configmap_path}")
    text = configmap_path.read_text(encoding="utf-8")
    if behavioral:
        if QUALIFICATION_SCOPE in text:
            raise CoordinatorBlockedError(
                "rendered behavioral bundle still carries qualification-only launch scope"
            )
        if '"/usr/bin/true"' in text or "'/usr/bin/true'" in text:
            raise CoordinatorBlockedError(
                "rendered behavioral bundle still references /usr/bin/true simulator argv"
            )
        if "online_correction_v4" not in text:
            raise CoordinatorBlockedError(
                "rendered behavioral bundle lacks online_correction_v4 runner binding"
            )
    else:
        if QUALIFICATION_SCOPE not in text:
            raise CoordinatorBlockedError(
                "rendered qualification bundle lacks qualification-only launch scope"
            )


def stage_binding_source(source: Path, staging_root: Path) -> Path:
    staging_root.mkdir(parents=True, exist_ok=True)
    destination = staging_root / source.name
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise CoordinatorBlockedError(f"conflicting staged binding already exists: {destination}")
        return destination
    shutil.copy2(source, destination)
    return destination


def pvc_binding_root_path(output_parent: str, lane_id: str, attempt_id: str) -> str:
    parent = output_parent.rstrip("/")
    return f"{parent}/.coord-bindings/{lane_id}/{attempt_id}"


def publish_staged_bindings_to_pvc(*, local_binding_root: Path, pvc_binding_root: str) -> None:
    pvc_root = Path(pvc_binding_root)
    if not pvc_root.is_absolute():
        raise CoordinatorBlockedError(f"pvc binding root must be absolute: {pvc_binding_root}")
    pvc_root.mkdir(parents=True, exist_ok=False)
    for source in sorted(local_binding_root.iterdir()):
        if not source.is_file():
            continue
        destination = pvc_root / source.name
        if destination.exists():
            raise CoordinatorBlockedError(f"refusing to overwrite pvc binding: {destination}")
        shutil.copy2(source, destination)


def write_coordination_state(path: Path, state: CoordinationState) -> None:
    payload = {
        "schema_version": COORDINATION_STATE_SCHEMA,
        "lane_infra_failures": dict(state.lane_infra_failures),
        "episode_infra_failures": dict(state.episode_infra_failures),
        "reserved_attempt_ids": sorted(state.reserved_attempt_ids),
        "attempt_index": state.attempt_index,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def rollback_acquired_leases(lease_store: GroupLeaseStore, leases: Sequence[GroupLease]) -> None:
    for lease in leases:
        if lease_store.verify(lease):
            lease_store.release(lease)


def build_dispatch_manifest_payload(
    *,
    lock: RuntimeLockBinding,
    queue_path: Path,
    runtime_lock_path: Path,
    campaign_config_path: Path,
    lane_id: str,
    attempt_id: str,
    policy_id: str,
    group_ids: Sequence[str],
    episode_ids: Sequence[str],
    local_binding_root: Path,
    pvc_binding_root: str,
) -> tuple[dict[str, Any], Path, str]:
    runner_source = (Path(__file__).resolve().parents[2] / lock.runner_entrypoint).resolve()
    if not runner_source.is_file():
        raise CoordinatorBlockedError(f"released runner source missing: {runner_source}")
    pvc_root = pvc_binding_root.rstrip("/")
    queue_name = "queue.jsonl"
    lock_name = "runtime_lock.json"
    config_name = "campaign.json"
    manifest_name = "lane_dispatch_manifest.json"
    runner_name = runner_source.name
    stage_binding_source(queue_path, local_binding_root)
    stage_binding_source(runtime_lock_path, local_binding_root)
    stage_binding_source(campaign_config_path, local_binding_root)
    stage_binding_source(runner_source, local_binding_root)
    pvc_queue = f"{pvc_root}/{queue_name}"
    pvc_lock = f"{pvc_root}/{lock_name}"
    pvc_config = f"{pvc_root}/{config_name}"
    pvc_runner = f"{pvc_root}/{runner_name}"
    pvc_manifest = f"{pvc_root}/{manifest_name}"
    payload = {
        "schema_version": DISPATCH_MANIFEST_SCHEMA,
        "manifest_sha256": lock.manifest_sha256,
        "queue_path": pvc_queue,
        "runtime_lock_path": pvc_lock,
        "campaign_config_path": pvc_config,
        "runner_entrypoint": pvc_runner,
        "runner_sha256": lock.runner_sha256,
        "lane_id": lane_id,
        "attempt_id": attempt_id,
        "policy_id": policy_id,
        "group_ids": list(group_ids),
        "episode_ids": list(episode_ids),
        "qualification_only": False,
        "one_episode_per_process": True,
    }
    local_manifest = local_binding_root / manifest_name
    local_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload, local_manifest, pvc_manifest


def build_lane_spec(
    *,
    template: Mapping[str, Any],
    cluster: ClusterBinding,
    lane_id: str,
    attempt_id: str,
    lock: RuntimeLockBinding,
    assignment_groups: Sequence[ExecutionGroup],
    remaining_episode_ids: Sequence[str],
    qualification_only: bool,
    queue_path: Path,
    runtime_lock_path: Path,
    campaign_config_path: Path,
    repo_root: Path,
    local_binding_root: Path,
    pvc_binding_root: str,
) -> dict[str, Any]:
    spec = json.loads(json.dumps(template))
    spec["kube_context"] = cluster.kube_context
    spec["namespace"] = cluster.namespace
    spec["pvc"] = cluster.pvc
    spec["output_parent"] = cluster.output_parent
    spec["lane_id"] = lane_id
    spec["attempt_id"] = attempt_id

    if qualification_only:
        spec["qualification_only"] = True
        assert_qualification_lane_spec(spec)
        spec.pop("dispatch", None)
        return spec

    policies = {group.policy for group in assignment_groups}
    if len(policies) != 1:
        raise CoordinatorError(
            "each lane assignment must contain one policy execution group batch"
        )
    policy_id = next(iter(policies))
    if policy_id not in lock.policies:
        raise CoordinatorBlockedError(f"runtime lock lacks policy binding for {policy_id}")
    policy_binding = lock.policies[policy_id]
    image_digest = policy_binding.runtime_image_digest
    if image_digest.startswith("sha256:"):
        image_digest = image_digest.split(":", 1)[1]
    spec["image_sha256"] = image_digest
    for role in ("policy", "simulator"):
        role_doc = spec.get(role)
        if isinstance(role_doc, dict):
            role_doc["checkpoint_sha256"] = policy_binding.checkpoint_sha256

    spec["qualification_only"] = False
    group_ids = [group.group_id for group in assignment_groups]
    dispatch_payload, _local_manifest, pvc_manifest_path = build_dispatch_manifest_payload(
        lock=lock,
        queue_path=queue_path,
        runtime_lock_path=runtime_lock_path,
        campaign_config_path=campaign_config_path,
        lane_id=lane_id,
        attempt_id=attempt_id,
        policy_id=policy_id,
        group_ids=group_ids,
        episode_ids=list(remaining_episode_ids),
        local_binding_root=local_binding_root,
        pvc_binding_root=pvc_binding_root,
    )
    dispatch_script_source = (repo_root / "deploy/k8s/v4_lane_bundle" / LANE_DISPATCH_SCRIPT).resolve()
    if not dispatch_script_source.is_file():
        raise CoordinatorBlockedError(f"lane dispatch runner missing: {dispatch_script_source}")
    dispatch_script_local = stage_binding_source(dispatch_script_source, local_binding_root)
    pvc_root = pvc_binding_root.rstrip("/")
    pvc_dispatch_script = f"{pvc_root}/{dispatch_script_source.name}"
    simulator = spec.setdefault("simulator", {})
    runtime_sim = spec.get("runtime", {}).get("simulator", {})
    sim_python = runtime_sim.get("python_bin")
    if not isinstance(sim_python, str) or not sim_python:
        raise CoordinatorBlockedError("simulator runtime.python_bin is required for behavioral dispatch")
    simulator["experiment_argv"] = [
        sim_python,
        pvc_dispatch_script,
        "--dispatch-manifest",
        pvc_manifest_path,
        "--launch-config",
        SIMULATOR_LAUNCH_CONFIG_MOUNT,
    ]
    existing_bindings = list(simulator.get("file_bindings") or [])
    bound_paths = {str(row.get("path")) for row in existing_bindings if isinstance(row, dict)}
    binding_specs = (
        (dispatch_script_local, pvc_dispatch_script),
        (local_binding_root / "lane_dispatch_manifest.json", pvc_manifest_path),
        (local_binding_root / "queue.jsonl", f"{pvc_root}/queue.jsonl"),
        (local_binding_root / "runtime_lock.json", f"{pvc_root}/runtime_lock.json"),
        (local_binding_root / "campaign.json", f"{pvc_root}/campaign.json"),
        (local_binding_root / Path(lock.runner_entrypoint).name, f"{pvc_root}/{Path(lock.runner_entrypoint).name}"),
    )
    for source_path, mount_path in binding_specs:
        if not source_path.is_file():
            continue
        mount = str(mount_path)
        if mount in bound_paths:
            continue
        existing_bindings.append({"source": str(source_path.resolve()), "path": mount})
        bound_paths.add(mount)
    simulator["file_bindings"] = existing_bindings
    spec["dispatch"] = {
        "schema_version": DISPATCH_MANIFEST_SCHEMA,
        "group_ids": group_ids,
        "episode_ids": list(remaining_episode_ids),
        "dispatch_manifest_path": pvc_manifest_path,
        "runner_entrypoint": lock.runner_entrypoint,
        "runner_sha256": lock.runner_sha256,
        "manifest_sha256": lock.manifest_sha256,
        "policy_id": policy_id,
        "dispatch_manifest": dispatch_payload,
    }
    assert_behavioral_lane_spec(spec)
    spec.pop("dispatch", None)
    return spec


def verify_queue_binding(
    *,
    queue_path: Path,
    queue_manifest_path: Path,
    expected_manifest_sha256: str,
) -> str:
    manifest_doc = load_json(queue_manifest_path, "queue manifest")
    queue_sha = sha256_file(queue_path)
    for key in ("queue_sha256", "frozen_queue_sha256"):
        declared = manifest_doc.get(key)
        if isinstance(declared, str) and declared != queue_sha:
            raise CoordinatorBlockedError(
                f"queue bytes do not match queue manifest {key}"
            )
    planning_sha = str(manifest_doc.get("planning_manifest_sha256") or "")
    if planning_sha and planning_sha != expected_manifest_sha256:
        raise CoordinatorBlockedError(
            "runtime lock manifest_sha256 does not match queue manifest planning hash"
        )
    return queue_sha


def load_qualified_lanes(launch_matrix: Mapping[str, Any], *, repo_root: Path) -> list[LaneStratum]:
    if launch_matrix.get("release_status") == "NOT_RELEASED":
        raise CoordinatorBlockedError("launch_matrix.json is NOT_RELEASED")
    lanes_raw = launch_matrix.get("qualified_lanes")
    if not isinstance(lanes_raw, list) or not lanes_raw:
        raise CoordinatorBlockedError("launch_matrix qualified_lanes must be nonempty when released")
    return [LaneStratum.from_mapping(row, repo_root=repo_root) for row in lanes_raw]


def apply_group_receipts(
    scheduler: CampaignScheduler,
    receipts: Sequence[GroupReceipt],
    *,
    manifest_sha256: str,
) -> None:
    for receipt in receipts:
        if receipt.manifest_sha256 != manifest_sha256:
            raise CoordinatorBlockedError(
                f"group receipt manifest mismatch for {receipt.group_id}"
            )
        if receipt.group_id not in scheduler.registry.by_execution_group:
            raise CoordinatorBlockedError(f"unknown group in receipt: {receipt.group_id}")
        for episode_id in receipt.accepted_episode_ids:
            scheduler.record_valid_episode(episode_id)
        if receipt.partial_episode_ids:
            scheduler.group_states[receipt.group_id] = GroupExecutionState.PARTIAL
            scheduler.interrupted_groups.add(receipt.group_id)
        elif receipt.is_complete:
            group = scheduler.registry.by_execution_group[receipt.group_id]
            if not scheduler._group_remaining_episodes(group):
                scheduler.group_states[receipt.group_id] = GroupExecutionState.COMPLETE


def episode_retry_exhausted(
    episode_id: str,
    *,
    episode_infra_failures: Counter[str],
    execution_config: ExecutionConfig,
) -> bool:
    return episode_infra_failures[episode_id] >= execution_config.max_infra_retries


def remaining_episodes_for_group(
    group: ExecutionGroup,
    scheduler: CampaignScheduler,
    *,
    episode_infra_failures: Counter[str],
    execution_config: ExecutionConfig,
) -> list[str]:
    remaining: list[str] = []
    for row in group.rows:
        if not scheduler.family_dispatchable(row.family):
            continue
        if row.episode_id in scheduler.completed_episodes:
            continue
        if episode_retry_exhausted(
            row.episode_id,
            episode_infra_failures=episode_infra_failures,
            execution_config=execution_config,
        ):
            continue
        remaining.append(row.episode_id)
    return remaining


def classify_group_episodes(
    group: ExecutionGroup,
    scheduler: CampaignScheduler,
    *,
    episode_infra_failures: Counter[str],
    execution_config: ExecutionConfig,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return dispatchable episode IDs and blocked episode records for one group."""
    dispatchable: list[str] = []
    blocked: list[dict[str, Any]] = []
    for episode_id in remaining_episodes_for_group(
        group,
        scheduler,
        episode_infra_failures=episode_infra_failures,
        execution_config=execution_config,
    ):
        row = scheduler.registry.get(episode_id)
        missing_controls = [
            control_id
            for control_id in row.reuse_episode_ids
            if control_id not in scheduler.accepted_episodes
        ]
        if missing_controls:
            blocked.append(
                {
                    "group_id": group.group_id,
                    "episode_id": episode_id,
                    "reason": "control_dependencies_unsatisfied",
                    "missing_control_episode_ids": missing_controls,
                }
            )
            continue
        dispatchable.append(episode_id)
    return dispatchable, blocked


def dispatchable_group_units(
    scheduler: CampaignScheduler,
    *,
    episode_infra_failures: Counter[str],
    execution_config: ExecutionConfig,
) -> tuple[list[tuple[ExecutionGroup, list[str]]], list[dict[str, Any]]]:
    units: list[tuple[ExecutionGroup, list[str]]] = []
    blocked_records: list[dict[str, Any]] = []
    for group in scheduler.registry.iter_groups():
        if group.group_id in scheduler.active_leases:
            continue
        dispatchable, blocked = classify_group_episodes(
            group,
            scheduler,
            episode_infra_failures=episode_infra_failures,
            execution_config=execution_config,
        )
        blocked_records.extend(blocked)
        if not dispatchable:
            continue
        state = scheduler.group_states.get(group.group_id, GroupExecutionState.UNLEASED)
        if state not in {GroupExecutionState.UNLEASED, GroupExecutionState.PARTIAL}:
            continue
        units.append((group, dispatchable))
    return units, blocked_records


def aggregate_blocked_groups(blocked_records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in blocked_records:
        group_id = str(record["group_id"])
        entry = grouped.setdefault(
            group_id,
            {
                "group_id": group_id,
                "reason": "control_dependencies_unsatisfied",
                "blocked_episode_ids": [],
                "missing_control_episode_ids": [],
            },
        )
        entry["blocked_episode_ids"].append(str(record["episode_id"]))
        for control_id in record.get("missing_control_episode_ids") or ():
            if control_id not in entry["missing_control_episode_ids"]:
                entry["missing_control_episode_ids"].append(str(control_id))
    return tuple(grouped.values())


def shard_group_units(
    units: Sequence[tuple[ExecutionGroup, list[str]]],
    lanes: Sequence[LaneStratum],
) -> dict[str, list[tuple[ExecutionGroup, list[str]]]]:
    if not lanes:
        return {}
    buckets: dict[str, list[tuple[ExecutionGroup, list[str]]]] = {lane.lane_id: [] for lane in lanes}
    lane_ids = [lane.lane_id for lane in lanes]
    for group, episode_ids in sorted(units, key=lambda item: item[0].group_id):
        digest = hashlib.sha256(group.group_id.encode("utf-8")).hexdigest()
        lane_id = lane_ids[int(digest[:8], 16) % len(lane_ids)]
        buckets[lane_id].append((group, episode_ids))
    return buckets


def estimate_storage_need(
    *,
    episode_count: int,
    infra_retry_count: int,
    execution_config: ExecutionConfig,
) -> int:
    return (
        episode_count * execution_config.estimated_bytes_per_episode
        + infra_retry_count * execution_config.estimated_bytes_per_infra_retry
    )


def storage_budget_allows(
    *,
    pending_episodes: int,
    pending_infra_retries: int,
    bytes_already_used: int,
    execution_config: ExecutionConfig,
) -> tuple[bool, dict[str, Any]]:
    need = estimate_storage_need(
        episode_count=pending_episodes,
        infra_retry_count=pending_infra_retries,
        execution_config=execution_config,
    )
    authorized = execution_config.authorized_storage_bytes
    summary = {
        "authorized_storage_bytes": authorized,
        "bytes_already_used": bytes_already_used,
        "estimated_pending_bytes": need,
        "estimated_total_bytes": bytes_already_used + need,
    }
    if authorized is None:
        summary["status"] = "unbounded"
        return True, summary
    allowed = bytes_already_used + need <= authorized
    summary["status"] = "within_budget" if allowed else "exceeded"
    return allowed, summary


def measure_evidence_bytes(evidence_root: Optional[Path]) -> int:
    if evidence_root is None or not evidence_root.is_dir():
        return 0
    total = 0
    for path in evidence_root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def parse_k8s_objects(bundle_root: Path, *, kube_context: str) -> TeardownInventory:
    objects: list[TeardownObject] = []
    mapping = {
        "configmap.yaml": "ConfigMap",
        "policy-job.yaml": "Job",
        "simulator-job.yaml": "Job",
        "policy-service.yaml": "Service",
    }
    for filename, kind in mapping.items():
        path = bundle_root / filename
        if not path.is_file():
            raise CoordinatorError(f"rendered bundle missing {filename}")
        text = path.read_text(encoding="utf-8")
        name_match = K8S_NAME_RE.search(text)
        namespace_match = re.search(r"^\s+namespace:\s+(.+)\s*$", text, re.MULTILINE)
        if not name_match or not namespace_match:
            raise CoordinatorError(f"cannot parse object identity from {path}")
        name = json.loads(name_match.group(1)) if name_match.group(1).startswith('"') else name_match.group(1)
        namespace = json.loads(namespace_match.group(1)) if namespace_match.group(1).startswith('"') else namespace_match.group(1)
        labels: dict[str, str] = {}
        for key in ("v4-lane-id", "v4-attempt-id", "v4-config-sha", "v4-lane-role"):
            label_match = re.search(rf"^\s+{re.escape(key)}:\s+(.+)\s*$", text, re.MULTILINE)
            if label_match:
                raw = label_match.group(1)
                labels[key] = json.loads(raw) if raw.startswith('"') else raw
        objects.append(
            TeardownObject(kind=kind, name=str(name), namespace=str(namespace), labels=labels)
        )
    return TeardownInventory(kube_context=kube_context, objects=tuple(objects))


def import_lane_renderer():
    import importlib.util

    renderer_path = Path(__file__).resolve().parents[2] / "tools/render_v4_k8s_lane_bundle.py"
    spec = importlib.util.spec_from_file_location("render_v4_k8s_lane_bundle", renderer_path)
    if spec is None or spec.loader is None:
        raise CoordinatorError("unable to load V4 lane renderer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_lane_bundle(
    *,
    spec_path: Path,
    output_root: Path,
    renderer: Any | None = None,
) -> dict[str, str]:
    render_mod = renderer or import_lane_renderer()
    return render_mod.render(spec_path.resolve(), output_root.resolve())


def kubectl_create_bundle(
    *,
    bundle_root: Path,
    kube_context: str,
    kubectl_runner: Callable[..., subprocess.CompletedProcess] | None = None,
) -> subprocess.CompletedProcess:
    runner = kubectl_runner or subprocess.run
    return runner(
        [
            "kubectl",
            "--context",
            kube_context,
            "create",
            "-k",
            str(bundle_root.resolve()),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _blocked_family_map(
    lock: RuntimeLockBinding,
    *,
    declared_families: set[str],
) -> dict[str, str]:
    raw_blocked = lock.raw.get("blocked_families") or {}
    blocked = {str(k): str(v) for k, v in raw_blocked.items()}
    released = set(lock.released_families)
    for family in declared_families:
        if family not in released:
            blocked.setdefault(family, blocked.get(family, "withheld by runtime lock"))
    return blocked


def build_scheduler(
    registry: CampaignRegistry,
    lock: RuntimeLockBinding,
    receipts: Sequence[GroupReceipt],
) -> CampaignScheduler:
    scheduler = CampaignScheduler(registry=registry)
    released = set(lock.released_families)
    blocked = _blocked_family_map(lock, declared_families=registry.families_present())
    errors = scheduler.set_release_state(released=released, blocked=blocked)
    if errors:
        raise CoordinatorBlockedError("; ".join(errors))
    apply_group_receipts(scheduler, receipts, manifest_sha256=lock.manifest_sha256)
    return scheduler


def plan_campaign(
    inputs: CoordinatorInputs,
    *,
    render_bundles: bool = False,
    create_on_cluster: bool = False,
    kubectl_runner: Callable[..., subprocess.CompletedProcess] | None = None,
    renderer: Any | None = None,
) -> CampaignPlan:
    if create_on_cluster and inputs.cluster_binding is None:
        raise CoordinatorBlockedError("--create requires explicit cluster binding")
    if create_on_cluster and inputs.cluster_binding is not None:
        inputs.cluster_binding.validate_for_create()
    if (
        create_on_cluster
        and not inputs.qualification_only
        and inputs.group_lease_root is None
    ):
        raise CoordinatorBlockedError(
            "behavioral --create requires --group-lease-root for durable group leases"
        )

    coordination_state = coordination_state_from_inputs(inputs)
    dispatch_mode = (
        DispatchMode.QUALIFICATION_ONLY
        if inputs.qualification_only
        else DispatchMode.BEHAVIORAL
    )

    config_sha = sha256_file(inputs.campaign_config_path)
    try:
        lock = validate_runtime_lock(
            inputs.runtime_lock_path,
            expected_config_sha256=config_sha,
        )
    except DroidContractError as exc:
        raise CoordinatorBlockedError(str(exc)) from exc

    if not lock.is_released:
        raise CoordinatorBlockedError("runtime lock is NOT_RELEASED")

    queue_sha = verify_queue_binding(
        queue_path=inputs.queue_path,
        queue_manifest_path=inputs.queue_manifest_path,
        expected_manifest_sha256=lock.manifest_sha256,
    )
    lock = validate_runtime_lock(
        inputs.runtime_lock_path,
        expected_config_sha256=config_sha,
        expected_manifest_sha256=lock.manifest_sha256,
    )

    registry = CampaignRegistry.from_manifest_path(inputs.queue_path)
    if registry.config_sha256 != lock.config_sha256:
        raise CoordinatorBlockedError("queue config_sha256 does not match runtime lock")

    launch_matrix = load_json(inputs.launch_matrix_path, "launch matrix")
    execution_config = inputs.execution_config
    if execution_config.authorized_storage_bytes is None:
        execution_config = ExecutionConfig.from_launch_matrix(launch_matrix)
    lanes = load_qualified_lanes(launch_matrix, repo_root=inputs.repo_root or Path("."))

    receipts = (
        load_group_receipts(inputs.group_receipts_dir)
        if inputs.group_receipts_dir is not None
        else []
    )
    scheduler = build_scheduler(registry, lock, receipts)
    durable_leases: dict[str, GroupLease] = {}
    lease_store: GroupLeaseStore | None = None
    if inputs.group_lease_root is not None:
        lease_store = GroupLeaseStore(inputs.group_lease_root.resolve())
        durable_leases = load_durable_group_leases(
            inputs.group_lease_root,
            manifest_sha256=lock.manifest_sha256,
        )
        apply_durable_group_leases(scheduler, durable_leases)

    quarantined = tuple(
        sorted(
            lane_id
            for lane_id, count in coordination_state.lane_infra_failures.items()
            if count >= execution_config.lane_quarantine_threshold
        )
    )
    active_lanes = [lane for lane in lanes if lane.lane_id not in quarantined]

    blocked_groups: list[dict[str, Any]] = []
    dispatch_units: list[tuple[ExecutionGroup, list[str]]] = []
    blocked_records: list[dict[str, Any]] = []
    if dispatch_mode is DispatchMode.BEHAVIORAL:
        dispatch_units, blocked_records = dispatchable_group_units(
            scheduler,
            episode_infra_failures=coordination_state.episode_infra_failures,
            execution_config=execution_config,
        )
        blocked_groups.extend(aggregate_blocked_groups(blocked_records))

    if (
        dispatch_mode is DispatchMode.BEHAVIORAL
        and dispatch_units
        and not active_lanes
    ):
        raise CoordinatorBlockedError(
            "all qualified lanes are quarantined while dispatchable work remains"
        )

    pending_episode_count = sum(len(episodes) for _, episodes in dispatch_units)
    pending_retry_count = sum(
        min(count, execution_config.max_infra_retries)
        for count in coordination_state.episode_infra_failures.values()
    )
    bytes_used = measure_evidence_bytes(inputs.evidence_root)
    allowed, storage_summary = storage_budget_allows(
        pending_episodes=pending_episode_count,
        pending_infra_retries=pending_retry_count,
        bytes_already_used=bytes_used,
        execution_config=execution_config,
    )
    if dispatch_mode is DispatchMode.BEHAVIORAL and not allowed:
        raise CoordinatorBlockedError("storage budget would be exceeded by pending dispatch")

    reserved_attempt_ids = collect_reserved_attempt_ids(
        coordination_state=coordination_state,
        lease_root=inputs.group_lease_root,
        render_output_root=inputs.render_output_root,
    )
    lane_attempt_index = coordination_state.attempt_index

    assignments: list[LaneAssignment] = []
    cluster = inputs.cluster_binding
    repo_root = inputs.repo_root or Path(".")
    partial_create_wave = False
    create_wave_receipt: Optional[dict[str, Any]] = None

    if dispatch_mode is DispatchMode.QUALIFICATION_ONLY:
        target_lanes = active_lanes or lanes
        for lane in target_lanes:
            attempt_id, lane_attempt_index = allocate_attempt_id(
                reserved=reserved_attempt_ids,
                start_index=lane_attempt_index,
            )
            rendered_spec_path: Optional[str] = None
            bundle_root: Optional[Path] = None
            launch_hash: Optional[str] = None
            teardown: Optional[dict[str, Any]] = None
            if render_bundles or create_on_cluster:
                if cluster is None:
                    raise CoordinatorBlockedError("cluster binding is required to render lane bundles")
                if inputs.render_output_root is None:
                    raise CoordinatorBlockedError("render_output_root is required to render lane bundles")
                bundle_root = (
                    inputs.render_output_root
                    / f"{lane.lane_id}-{attempt_id}-{lane.hardware_stratum}"
                ).resolve()
                if bundle_root.exists():
                    raise CoordinatorBlockedError(
                        f"refusing to overwrite existing bundle directory: {bundle_root}"
                    )
                bundle_root.mkdir(parents=True, exist_ok=False)
                local_binding_root = (bundle_root / ".bindings").resolve()
                pvc_root = pvc_binding_root_path(cluster.output_parent, lane.lane_id, attempt_id)
                spec = build_lane_spec(
                    template=lane.spec_template,
                    cluster=cluster,
                    lane_id=lane.lane_id,
                    attempt_id=attempt_id,
                    lock=lock,
                    assignment_groups=(),
                    remaining_episode_ids=(),
                    qualification_only=True,
                    queue_path=inputs.queue_path,
                    runtime_lock_path=inputs.runtime_lock_path,
                    campaign_config_path=inputs.campaign_config_path,
                    repo_root=repo_root,
                    local_binding_root=local_binding_root,
                    pvc_binding_root=pvc_root,
                )
                spec_path = (
                    lane.template_root / f".coord-{lane.lane_id}-{attempt_id}-render-spec.json"
                ).resolve()
                if spec_path.exists():
                    raise CoordinatorBlockedError(
                        f"refusing to overwrite existing render spec: {spec_path}"
                    )
                rendered_spec_path = str(spec_path)
                try:
                    spec_path.write_text(
                        json.dumps(spec, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    launch_hash = sha256_file(spec_path)
                    render_lane_bundle(spec_path=spec_path, output_root=bundle_root, renderer=renderer)
                    validate_rendered_bundle_scope(bundle_root, behavioral=False)
                finally:
                    spec_path.unlink(missing_ok=True)
                inventory = parse_k8s_objects(bundle_root, kube_context=cluster.kube_context)
                teardown = inventory.as_dict()
                if create_on_cluster:
                    result = kubectl_create_bundle(
                        bundle_root=bundle_root,
                        kube_context=cluster.kube_context,
                        kubectl_runner=kubectl_runner,
                    )
                    if result.returncode != 0:
                        raise CoordinatorError(
                            f"kubectl create failed for {bundle_root}: {result.stderr.strip()}"
                        )
            assignments.append(
                LaneAssignment(
                    lane_id=lane.lane_id,
                    hardware_stratum=lane.hardware_stratum,
                    attempt_id=attempt_id,
                    group_ids=(),
                    remaining_episode_ids=(),
                    spec_path=rendered_spec_path if (render_bundles or create_on_cluster) else None,
                    bundle_root=str(bundle_root) if bundle_root else None,
                    launch_hash=launch_hash,
                    teardown_inventory=teardown,
                    create_status=(
                        "created"
                        if create_on_cluster and teardown is not None
                        else ("rendered" if render_bundles and bundle_root else "planned")
                    ),
                )
            )
    else:
        buckets = shard_group_units(dispatch_units, active_lanes)
        wave_receipt_created: list[dict[str, Any]] = []
        for lane in active_lanes:
            if partial_create_wave:
                break
            lane_units = buckets.get(lane.lane_id, [])
            if not lane_units:
                continue
            by_policy: dict[str, list[tuple[ExecutionGroup, list[str]]]] = {}
            for group, episode_ids in lane_units:
                by_policy.setdefault(group.policy, []).append((group, episode_ids))
            for policy_id in sorted(by_policy):
                policy_units = by_policy[policy_id]
                groups = [group for group, _episode_ids in policy_units]
                remaining_episodes = [
                    episode_id for _group, episode_ids in policy_units for episode_id in episode_ids
                ]
                attempt_id, lane_attempt_index = allocate_attempt_id(
                    reserved=reserved_attempt_ids,
                    start_index=lane_attempt_index,
                )
                assignment_leases: list[GroupLease] = []
                cluster_created = False
                spec_path: Optional[Path] = None
                rendered_spec_path = None
                bundle_root = None
                launch_hash = None
                teardown = None
                dispatch_manifest_path: Optional[str] = None
                pvc_root: Optional[str] = None
                acquired_group_ids: list[str] = []
                create_status = "planned"

                try:
                    if render_bundles or create_on_cluster:
                        if cluster is None:
                            raise CoordinatorBlockedError("cluster binding is required to render lane bundles")
                        if inputs.render_output_root is None:
                            raise CoordinatorBlockedError("render_output_root is required to render lane bundles")
                        bundle_root = (
                            inputs.render_output_root
                            / f"{lane.lane_id}-{attempt_id}-{lane.hardware_stratum}"
                        ).resolve()
                        if bundle_root.exists():
                            raise CoordinatorBlockedError(
                                f"refusing to overwrite existing bundle directory: {bundle_root}"
                            )
                        bundle_root.mkdir(parents=True, exist_ok=False)
                        local_binding_root = (bundle_root / ".bindings").resolve()
                        pvc_root = pvc_binding_root_path(cluster.output_parent, lane.lane_id, attempt_id)
                        spec = build_lane_spec(
                            template=lane.spec_template,
                            cluster=cluster,
                            lane_id=lane.lane_id,
                            attempt_id=attempt_id,
                            lock=lock,
                            assignment_groups=groups,
                            remaining_episode_ids=remaining_episodes,
                            qualification_only=False,
                            queue_path=inputs.queue_path,
                            runtime_lock_path=inputs.runtime_lock_path,
                            campaign_config_path=inputs.campaign_config_path,
                            repo_root=repo_root,
                            local_binding_root=local_binding_root,
                            pvc_binding_root=pvc_root,
                        )
                        dispatch_manifest_path = f"{pvc_root}/lane_dispatch_manifest.json"
                        spec_path = (
                            lane.template_root / f".coord-{lane.lane_id}-{attempt_id}-render-spec.json"
                        ).resolve()
                        if spec_path.exists():
                            raise CoordinatorBlockedError(
                                f"refusing to overwrite existing render spec: {spec_path}"
                            )
                        rendered_spec_path = str(spec_path)
                        try:
                            spec_path.write_text(
                                json.dumps(spec, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8",
                            )
                            launch_hash = sha256_file(spec_path)
                            render_lane_bundle(spec_path=spec_path, output_root=bundle_root, renderer=renderer)
                            validate_rendered_bundle_scope(bundle_root, behavioral=True)
                        finally:
                            spec_path.unlink(missing_ok=True)
                        inventory = parse_k8s_objects(bundle_root, kube_context=cluster.kube_context)
                        teardown = inventory.as_dict()
                        create_status = "rendered"

                        if create_on_cluster:
                            publish_staged_bindings_to_pvc(
                                local_binding_root=local_binding_root,
                                pvc_binding_root=pvc_root,
                            )
                            if lease_store is not None:
                                for group in groups:
                                    if group.group_id in durable_leases:
                                        raise CoordinatorBlockedError(
                                            f"group {group.group_id} is already leased durably"
                                        )
                                    try:
                                        lease = lease_store.acquire(
                                            group_id=group.group_id,
                                            owner_lane=lane.lane_id,
                                            attempt_id=attempt_id,
                                            manifest_sha256=lock.manifest_sha256,
                                        )
                                    except LeaseConflict as exc:
                                        raise CoordinatorBlockedError(str(exc)) from exc
                                    assignment_leases.append(lease)
                                    acquired_group_ids.append(group.group_id)
                                    durable_leases[group.group_id] = lease
                                    scheduler.active_leases[group.group_id] = lease
                                    scheduler.group_states[group.group_id] = GroupExecutionState.LEASED
                            result = kubectl_create_bundle(
                                bundle_root=bundle_root,
                                kube_context=cluster.kube_context,
                                kubectl_runner=kubectl_runner,
                            )
                            if result.returncode != 0:
                                raise CoordinatorError(
                                    f"kubectl create failed for {bundle_root}: {result.stderr.strip()}"
                                )
                            cluster_created = True
                            create_status = "created"
                            wave_receipt_created.append(
                                {
                                    "lane_id": lane.lane_id,
                                    "attempt_id": attempt_id,
                                    "group_ids": [group.group_id for group in groups],
                                    "bundle_root": str(bundle_root),
                                }
                            )

                    assignments.append(
                        LaneAssignment(
                            lane_id=lane.lane_id,
                            hardware_stratum=lane.hardware_stratum,
                            attempt_id=attempt_id,
                            group_ids=tuple(group.group_id for group in groups),
                            remaining_episode_ids=tuple(remaining_episodes),
                            spec_path=rendered_spec_path if (render_bundles or create_on_cluster) else None,
                            bundle_root=str(bundle_root) if bundle_root else None,
                            launch_hash=launch_hash,
                            teardown_inventory=teardown,
                            dispatch_manifest_path=dispatch_manifest_path,
                            acquired_group_leases=tuple(acquired_group_ids),
                            pvc_binding_root=pvc_root,
                            create_status=create_status,
                        )
                    )
                except Exception as exc:
                    if assignment_leases and lease_store is not None and not cluster_created:
                        rollback_acquired_leases(lease_store, assignment_leases)
                    if wave_receipt_created:
                        partial_create_wave = True
                        create_wave_receipt = {
                            "status": "partial_create_wave",
                            "created_assignments": list(wave_receipt_created),
                            "failed_assignment": {
                                "lane_id": lane.lane_id,
                                "attempt_id": attempt_id,
                                "policy_id": policy_id,
                                "group_ids": [group.group_id for group in groups],
                                "error": str(exc),
                            },
                        }
                        break
                    raise
            if partial_create_wave:
                break

    complete_groups = sum(
        1
        for state in scheduler.group_states.values()
        if state is GroupExecutionState.COMPLETE
    )
    partial_groups = sum(
        1
        for state in scheduler.group_states.values()
        if state is GroupExecutionState.PARTIAL
    )
    pending_groups = sum(
        1
        for group in scheduler.registry.by_execution_group.values()
        if remaining_episodes_for_group(
            group,
            scheduler,
            episode_infra_failures=coordination_state.episode_infra_failures,
            execution_config=execution_config,
        )
    )

    behavioral_episode_count = (
        0
        if dispatch_mode is DispatchMode.QUALIFICATION_ONLY
        else sum(len(item.remaining_episode_ids) for item in assignments)
    )
    classifier = AttemptClassifier(retry_policy=RetryPolicy(max_retries=execution_config.max_infra_retries))

    if inputs.coordination_state_path is not None and (assignments or partial_create_wave):
        write_coordination_state(
            inputs.coordination_state_path,
            CoordinationState(
                lane_infra_failures=Counter(coordination_state.lane_infra_failures),
                episode_infra_failures=Counter(coordination_state.episode_infra_failures),
                reserved_attempt_ids=frozenset(reserved_attempt_ids),
                attempt_index=lane_attempt_index,
            ),
        )

    return CampaignPlan(
        schema_version=SCHEMA_VERSION,
        release_status=lock.release_status,
        manifest_sha256=lock.manifest_sha256,
        queue_sha256=queue_sha,
        config_sha256=lock.config_sha256,
        dispatchable_group_count=len(dispatch_units),
        pending_group_count=pending_groups,
        complete_group_count=complete_groups,
        partial_group_count=partial_groups,
        lane_assignments=tuple(assignments),
        quarantined_lanes=quarantined,
        blocked_groups=tuple(blocked_groups),
        storage_budget=storage_summary,
        resume_receipt_count=len(receipts),
        scheduling_inputs={
            "max_infra_retries": execution_config.max_infra_retries,
            "lane_quarantine_threshold": execution_config.lane_quarantine_threshold,
            "accepted_episode_count": len(scheduler.accepted_episodes),
            "retry_policy_allows_behavioral_retry": False,
            "classifier_retry_policy_max": classifier.retry_policy.max_retries,
            "durable_group_lease_count": len(durable_leases),
            "reserved_attempt_id_count": len(reserved_attempt_ids),
            "coordination_state_loaded": inputs.coordination_state_path is not None,
        },
        dispatch_mode=dispatch_mode.value,
        behavioral_episode_count=behavioral_episode_count,
        effect_size_peeking=False,
        partial_create_wave=partial_create_wave,
        create_wave_receipt=create_wave_receipt,
    )
