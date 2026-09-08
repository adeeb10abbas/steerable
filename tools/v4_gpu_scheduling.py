#!/usr/bin/env python3
"""Shared V4 GPU product scheduling helpers for Kubernetes lane renderers.

Model-blind G2/G3 simulator jobs may run on any product in
``MODEL_BLIND_SIMULATOR_POOL`` after per-product determinism attestation.
Policy episode lanes remain bound to the qualified hardware stratum recorded
in each fixture's released runtime lock and G4 receipt.

Placement policies (``c8_a40_spread``, ``c6_a10040_spread``) add topology
spread and soft anti-affinity so fragmented single-GPU nodes bin-pack better
without rebinding hardware strata.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

GPU_PRODUCT_RE = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}[A-Za-z0-9]$|^[A-Za-z0-9]$"

# Qualified primary for V3/V4 DROID Isaac sim; not a universal scientific lock.
DEFAULT_MODEL_BLIND_SIMULATOR_PRODUCT = "NVIDIA-A40"

# Products validated for Isaac Lab / PhysX sim after determinism attestation.
MODEL_BLIND_SIMULATOR_POOL: tuple[str, ...] = (
    "NVIDIA-A40",
    "NVIDIA-A100-SXM4-80GB",
    "NVIDIA-A100-SXM4-40GB",
    "NVIDIA-B200",
)

PRODUCT_TO_DISPLAY_NAME: dict[str, str] = {
    "NVIDIA-A40": "NVIDIA A40",
    "NVIDIA-A100-SXM4-80GB": "NVIDIA A100-SXM4-80GB",
    "NVIDIA-A100-SXM4-40GB": "NVIDIA A100-SXM4-40GB",
    "NVIDIA-B200": "NVIDIA B200",
}

DISPLAY_NAME_TO_PRODUCT: dict[str, str] = {
    display: product for product, display in PRODUCT_TO_DISPLAY_NAME.items()
}

# Frozen C7 confirmatory stratum; do not widen without a new G4 receipt.
POLICY_EPISODE_HARDWARE_STRATUM = {
    "hardware_stratum": "a10080-policy_a40-simulator",
    "policy_gpu_product": "NVIDIA-A100-SXM4-80GB",
    "policy_gpu_display_name": "NVIDIA A100-SXM4-80GB",
    "simulator_gpu_product": "NVIDIA-A40",
    "simulator_gpu_display_name": "NVIDIA A40",
}


class GpuSchedulingError(ValueError):
    """Raised when GPU scheduling configuration is invalid."""


def display_name_for_product(product: str) -> str:
    try:
        return PRODUCT_TO_DISPLAY_NAME[product]
    except KeyError as exc:
        raise GpuSchedulingError(f"unknown gpu_product: {product!r}") from exc


def resolve_model_blind_scheduling(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve simulator GPU scheduling fields from a render spec."""
    primary = str(spec.get("gpu_product") or DEFAULT_MODEL_BLIND_SIMULATOR_PRODUCT)
    if primary not in MODEL_BLIND_SIMULATOR_POOL:
        raise GpuSchedulingError(
            f"gpu_product {primary!r} is outside MODEL_BLIND_SIMULATOR_POOL"
        )
    allowlist_raw = spec.get("gpu_product_allowlist")
    if allowlist_raw is None:
        allowlist = [primary]
    else:
        if not isinstance(allowlist_raw, list) or not allowlist_raw:
            raise GpuSchedulingError("gpu_product_allowlist must be a nonempty list")
        allowlist = [str(item) for item in allowlist_raw]
        unknown = [item for item in allowlist if item not in MODEL_BLIND_SIMULATOR_POOL]
        if unknown:
            raise GpuSchedulingError(
                f"gpu_product_allowlist contains unknown products: {unknown}"
            )
        if primary not in allowlist:
            raise GpuSchedulingError("gpu_product must appear in gpu_product_allowlist")
    expected_name = str(
        spec.get("expected_gpu_name") or display_name_for_product(primary)
    )
    allowed_names = [display_name_for_product(item) for item in allowlist]
    if expected_name not in allowed_names:
        raise GpuSchedulingError(
            f"expected_gpu_name {expected_name!r} is not in the allowlist display names"
        )
    return {
        "gpu_product": primary,
        "gpu_product_allowlist": allowlist,
        "expected_gpu_name": expected_name,
        "allowed_gpu_names": allowed_names,
        "homogeneous_stratum_required": len(allowlist) > 1,
    }


def yaml_scalar(value: str) -> str:
    if not value or any(ch in value for ch in ":{}[],&*#?|-<>=!%@\\"):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def render_pod_gpu_scheduling_yaml(
    *,
    gpu_product: str,
    gpu_product_allowlist: Sequence[str] | None,
    indent: str = "      ",
) -> list[str]:
    """Emit nodeSelector or nodeAffinity YAML lines for one GPU request."""
    products = list(gpu_product_allowlist or [gpu_product])
    rows = [f"{indent}nodeSelector:", f'{indent}  node-role.kubernetes.io/worker-gpu: ""']
    if len(products) == 1:
        rows.append(
            f"{indent}  nvidia.com/gpu.product: {yaml_scalar(products[0])}"
        )
        return rows
    rows.append(f"{indent}affinity:")
    rows.append(f"{indent}  nodeAffinity:")
    rows.append(f"{indent}    requiredDuringSchedulingIgnoredDuringExecution:")
    rows.append(f"{indent}      nodeSelectorTerms:")
    rows.append(f"{indent}      - matchExpressions:")
    rows.append(f"{indent}        - key: nvidia.com/gpu.product")
    rows.append(f"{indent}          operator: In")
    rows.append(f"{indent}          values:")
    for product in products:
        rows.append(f"{indent}          - {yaml_scalar(product)}")
    return rows


def preflight_gpu_fields(scheduling: Mapping[str, Any]) -> dict[str, Any]:
    """Launch-config GPU fields consumed by startup_preflight.py."""
    return {
        "expected_gpu_name": scheduling["expected_gpu_name"],
        "allowed_gpu_names": list(scheduling["allowed_gpu_names"]),
    }


def validate_pod_gpu_scheduling(
    pod_spec: Mapping[str, Any],
    *,
    gpu_product: str,
    gpu_product_allowlist: Sequence[str] | None,
) -> None:
    """Fail closed unless pod scheduling matches the rendered manifest binding."""
    products = list(gpu_product_allowlist or [gpu_product])
    node_selector = pod_spec.get("nodeSelector") or {}
    if not isinstance(node_selector, dict):
        raise GpuSchedulingError("pod nodeSelector must be an object")
    if node_selector.get("node-role.kubernetes.io/worker-gpu") != "":
        raise GpuSchedulingError("pod must select worker-gpu nodes")
    if len(products) == 1:
        if node_selector.get("nvidia.com/gpu.product") != products[0]:
            raise GpuSchedulingError(
                f"pod must pin nvidia.com/gpu.product={products[0]!r}"
            )
        return
    affinity = pod_spec.get("affinity") or {}
    if not isinstance(affinity, dict):
        raise GpuSchedulingError("pod with GPU allowlist must set affinity.nodeAffinity")
    node_affinity = affinity.get("nodeAffinity") or {}
    required = node_affinity.get("requiredDuringSchedulingIgnoredDuringExecution") or {}
    terms = required.get("nodeSelectorTerms") or []
    if not terms:
        raise GpuSchedulingError("pod nodeAffinity terms are missing")
    expressions = terms[0].get("matchExpressions") or []
    product_expr = next(
        (
            expr
            for expr in expressions
            if isinstance(expr, dict) and expr.get("key") == "nvidia.com/gpu.product"
        ),
        None,
    )
    if not isinstance(product_expr, dict) or product_expr.get("operator") != "In":
        raise GpuSchedulingError("pod must require nvidia.com/gpu.product In allowlist")
    values = product_expr.get("values")
    if not isinstance(values, list) or sorted(str(v) for v in values) != sorted(products):
        raise GpuSchedulingError("pod GPU allowlist differs from manifest")


# --- Placement policies for per-node bin-packing across fragmented GPU pools ---

DEFAULT_PROTECT_LIST = (
    __import__("pathlib").Path(__file__).resolve().parents[1]
    / "artifacts/online_correction_v4/execution/gpu_widen_20260908/gpu_sweep_protect_list_20260908.json"
)

PRODUCTIVE_C8_LANE_IDS = frozenset({"c8m13", "c8m14"})

PLACEMENT_POLICY_SCHEMA_VERSION = "v4-gpu-placement-policy-v1"

# Registered spread profiles. Renderer and enforce tooling share these keys.
PLACEMENT_POLICIES: dict[str, dict[str, Any]] = {
    "c8_a40_spread": {
        "schema_version": PLACEMENT_POLICY_SCHEMA_VERSION,
        "policy_id": "c8_a40_spread",
        "gpu_product": "NVIDIA-A40",
        "spread_family_label": "c8-a40",
        "topology_key": "kubernetes.io/hostname",
        "max_skew": 1,
        "when_unsatisfiable": "ScheduleAnyway",
        "roles": ("policy", "simulator"),
        # Sim-first: spread and schedule simulator before policy consumes scattered slots.
        "schedule_order": ("simulator", "policy"),
        "spread_roles": ("simulator",),
        "defer_roles_until_partner_running": ("policy",),
        "hardware_stratum": "a40-single-container-groot-bridge",
        "lane_id_prefixes": ("c8m", "g7c8p"),
    },
    "c6_a10040_spread": {
        "schema_version": PLACEMENT_POLICY_SCHEMA_VERSION,
        "policy_id": "c6_a10040_spread",
        "gpu_product": "NVIDIA-A100-SXM4-40GB",
        "spread_family_label": "c6-a10040-sim",
        "topology_key": "kubernetes.io/hostname",
        "max_skew": 1,
        "when_unsatisfiable": "ScheduleAnyway",
        "roles": ("simulator",),
        "schedule_order": ("simulator", "policy"),
        "spread_roles": ("simulator",),
        "defer_roles_until_partner_running": ("policy",),
        "defer_partner_gpu_product": "NVIDIA-B200",
        "hardware_stratum": "b200-policy_a10040-simulator",
        "lane_id_prefixes": ("c6m", "g7c6p"),
    },
}


def resolve_placement_policy(spec: Mapping[str, Any]) -> dict[str, Any] | None:
    """Resolve an optional placement policy id from a lane render spec."""
    raw = spec.get("placement_policy")
    if raw is None:
        return None
    policy_id = str(raw)
    if policy_id not in PLACEMENT_POLICIES:
        raise GpuSchedulingError(f"unknown placement_policy: {policy_id!r}")
    return dict(PLACEMENT_POLICIES[policy_id])


def load_protect_list(path: Any | None = None) -> dict[str, Any]:
    import json

    protect_path = path or DEFAULT_PROTECT_LIST
    if not protect_path.is_file():
        return {"productive_c7_lane_pairs": [], "c7_retry_shards_in_flight_do_not_preempt": []}
    return json.loads(protect_path.read_text(encoding="utf-8"))


def protected_c7_lane_ids(protect_list: Mapping[str, Any]) -> list[str]:
    lanes: set[str] = set()
    for row in protect_list.get("productive_c7_lane_pairs") or []:
        if isinstance(row, dict) and row.get("lane_id"):
            lanes.add(str(row["lane_id"]))
    for row in protect_list.get("c7_retry_shards_in_flight_do_not_preempt") or []:
        if isinstance(row, dict) and row.get("lane_id"):
            lanes.add(str(row["lane_id"]))
    return sorted(lanes)


def protected_lane_attempt_keys(protect_list: Mapping[str, Any]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for section in (
        "productive_c7_lane_pairs",
        "c7_retry_shards_in_flight_do_not_preempt",
    ):
        for row in protect_list.get(section) or []:
            if isinstance(row, dict) and row.get("lane_id") and row.get("attempt_id"):
                keys.add((str(row["lane_id"]), str(row["attempt_id"])))
    return keys


def is_lane_protected(
    *,
    lane_id: str,
    attempt_id: str,
    protect_list: Mapping[str, Any],
) -> bool:
    return (lane_id, attempt_id) in protected_lane_attempt_keys(protect_list)


def role_gets_spread(placement_policy: Mapping[str, Any], role: str) -> bool:
    spread_roles = placement_policy.get("spread_roles")
    if spread_roles is None:
        return role in placement_policy.get("roles", ())
    return role in spread_roles


def role_is_deferred_until_partner_running(placement_policy: Mapping[str, Any], role: str) -> bool:
    return role in placement_policy.get("defer_roles_until_partner_running", ())


def render_pod_placement_yaml(
    *,
    placement_policy: Mapping[str, Any],
    role: str,
    protected_c7_lanes: Sequence[str] | None = None,
    indent: str = "      ",
) -> tuple[list[str], dict[str, str]]:
    """Emit topologySpreadConstraints and soft anti-affinity YAML lines."""
    if role not in placement_policy.get("roles", ()):
        return [], {}
    if not role_gets_spread(placement_policy, role):
        return [], {}
    spread_label = str(placement_policy["spread_family_label"])
    extra_labels = {"v4-gpu-spread-family": spread_label}
    rows = [
        f"{indent}topologySpreadConstraints:",
        f"{indent}  - maxSkew: {int(placement_policy['max_skew'])}",
        f"{indent}    topologyKey: {yaml_scalar(str(placement_policy['topology_key']))}",
        f"{indent}    whenUnsatisfiable: {placement_policy['when_unsatisfiable']}",
        f"{indent}    labelSelector:",
        f"{indent}      matchLabels:",
        f"{indent}        v4-gpu-spread-family: {yaml_scalar(spread_label)}",
    ]
    protected = list(protected_c7_lanes or [])
    if protected:
        rows += [
            f"{indent}affinity:",
            f"{indent}  podAntiAffinity:",
            f"{indent}    preferredDuringSchedulingIgnoredDuringExecution:",
            f"{indent}      - weight: 80",
            f"{indent}        podAffinityTerm:",
            f"{indent}          topologyKey: kubernetes.io/hostname",
            f"{indent}          labelSelector:",
            f"{indent}            matchExpressions:",
            f"{indent}              - key: v4-lane-id",
            f"{indent}                operator: In",
            f"{indent}                values:",
        ]
        for lane_id in protected:
            rows.append(f"{indent}                  - {yaml_scalar(lane_id)}")
    return rows, extra_labels


def inject_placement_into_pod_spec(
    pod_spec: dict[str, Any],
    *,
    placement_policy: Mapping[str, Any],
    role: str,
    protected_c7_lanes: Sequence[str] | None = None,
) -> dict[str, str]:
    """Mutate a Job pod spec dict in place; return extra labels to merge."""
    if role not in placement_policy.get("roles", ()):
        return {}
    if not role_gets_spread(placement_policy, role):
        return {}
    spread_label = str(placement_policy["spread_family_label"])
    extra_labels = {"v4-gpu-spread-family": spread_label}
    pod_spec["topologySpreadConstraints"] = [
        {
            "maxSkew": int(placement_policy["max_skew"]),
            "topologyKey": str(placement_policy["topology_key"]),
            "whenUnsatisfiable": placement_policy["when_unsatisfiable"],
            "labelSelector": {
                "matchLabels": {"v4-gpu-spread-family": spread_label},
            },
        }
    ]
    protected = list(protected_c7_lanes or [])
    if protected:
        affinity = pod_spec.setdefault("affinity", {})
        anti = affinity.setdefault("podAntiAffinity", {})
        preferred = anti.setdefault("preferredDuringSchedulingIgnoredDuringExecution", [])
        preferred.append(
            {
                "weight": 80,
                "podAffinityTerm": {
                    "topologyKey": "kubernetes.io/hostname",
                    "labelSelector": {
                        "matchExpressions": [
                            {
                                "key": "v4-lane-id",
                                "operator": "In",
                                "values": protected,
                            }
                        ]
                    },
                },
            }
        )
    return extra_labels


def validate_pod_placement(
    pod_spec: Mapping[str, Any],
    *,
    placement_policy: Mapping[str, Any],
    role: str,
) -> None:
    if role not in placement_policy.get("roles", ()):
        return
    spread = pod_spec.get("topologySpreadConstraints")
    if not isinstance(spread, list) or not spread:
        raise GpuSchedulingError("pod missing topologySpreadConstraints for placement policy")
    entry = spread[0]
    if entry.get("topologyKey") != placement_policy["topology_key"]:
        raise GpuSchedulingError("topologySpreadConstraints topologyKey mismatch")


def analyze_lane_pair_capacity_binding(*, family: str = "c8") -> dict[str, Any]:
    """Document whether the 2-GPU lane pair is a scheduler or planning constraint."""
    if family == "c8":
        return {
            "family": "C8_second_stack",
            "hardware_stratum": "a40-single-container-groot-bridge",
            "scheduler_gpu_request_per_pod": 1,
            "planning_gpus_per_lane_pair": 2,
            "policy_sim_co_location_required": False,
            "binding_factor": "planning_and_pool_occupancy_not_single_pod_binpack",
            "finding": (
                "C8 policy and simulator are independent 1-GPU pods on NVIDIA-A40 "
                "connected via in-cluster Service HTTP. Kubernetes does not require "
                "co-scheduling on one node or one 2-GPU request. The 2-GPU lane pair "
                "is the throughput planning unit because both pods must be Running for "
                "episodes, not because the scheduler atomically allocates two GPUs. "
                "Pending C8 pods with pool-wide free A40 capacity therefore indicate "
                "per-node fragmentation and competing C7 A40 occupancy, not a rigid "
                "2-GPU bin-packing constraint."
            ),
            "paper_note": (
                "Report lane-pair concurrency as a study-design unit (2×A40 per lane), "
                "but attribute scheduling failures to fragmented single-GPU nodes rather "
                "than an unsatisfiable 2-GPU pod request."
            ),
        }
    if family == "c6":
        return {
            "family": "C6_containment",
            "hardware_stratum": "b200-policy_a10040-simulator",
            "scheduler_gpu_request_per_pod": 1,
            "planning_gpus_per_lane_pair": 2,
            "policy_sim_co_location_required": False,
            "binding_factor": "cross_pool_planning_unit",
            "finding": (
                "C6 binds B200 policy and A100-40GB simulator as separate 1-GPU pods. "
                "The pair is a cross-pool planning unit; spread policies apply per pool."
            ),
        }
    raise GpuSchedulingError(f"unknown family for capacity binding analysis: {family!r}")
