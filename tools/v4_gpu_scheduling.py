#!/usr/bin/env python3
"""Shared V4 GPU product scheduling helpers for Kubernetes lane renderers.

Model-blind G2/G3 simulator jobs may run on any product in
``MODEL_BLIND_SIMULATOR_POOL`` after per-product determinism attestation.
Policy episode lanes remain bound to the qualified hardware stratum recorded
in each fixture's released runtime lock and G4 receipt.
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
