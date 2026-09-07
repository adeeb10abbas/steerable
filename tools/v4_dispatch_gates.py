#!/usr/bin/env python3
"""Fail-closed dispatch gates for V4 model-blind Kubernetes bundles."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = (
    ROOT / "artifacts/online_correction_v4/setup/gpu_smoke_registry.json"
)
DEFAULT_NATURAL_GRASP_REGISTRY = (
    ROOT / "artifacts/online_correction_v4/setup/natural_grasp_live_control_registry.json"
)

FIXTURES_REQUIRING_NATURAL_GRASP_LIVE_CONTROL = frozenset(
    {
        "object_pair",
        "horizontal",
        "vertical",
        "containment",
        "reference_binding",
        "second_stack",
    }
)
NATURAL_GRASP_PLATFORMS: dict[str, str] = {
    "object_pair": "isaac",
    "horizontal": "isaac",
    "vertical": "isaac",
    "containment": "isaac",
    "reference_binding": "isaac",
    "second_stack": "simplerenv",
}
POLICY_TRIGGER_GATES = frozenset({"G4", "G5", "G6", "G7", "G8", "policy", "confirmatory"})

JOB_OUTPUT_PARENT_RE = re.compile(
    r'name: OUTPUT_PARENT\n\s+value: "([^"]+)"'
)
CONFIGMAP_LAUNCH_RE = re.compile(
    r"simulator-launch\.json: \|\n    (.+?)\n  image\.digest:",
    re.DOTALL,
)


class DispatchGateError(RuntimeError):
    """Dispatch preconditions failed; do not create Kubernetes objects."""


def smoke_key(*, fixture_id: str, gate: str, gpu_product: str) -> str:
    return f"{fixture_id}:{gate}:{gpu_product}"


def load_smoke_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": "v4-gpu-smoke-registry-v1", "passed_smokes": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_smoke_registry(payload: dict[str, Any], path: Path = DEFAULT_REGISTRY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_natural_grasp_registry(
    path: Path = DEFAULT_NATURAL_GRASP_REGISTRY,
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": "v4-natural-grasp-live-control-registry-v1",
            "passed_controls": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_natural_grasp_registry(
    payload: dict[str, Any],
    path: Path = DEFAULT_NATURAL_GRASP_REGISTRY,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record_passed_natural_grasp_live_control(
    *,
    fixture_id: str,
    control_mode: str,
    attempt_id: str,
    receipt_path: str,
    receipt_sha256: str,
    study_commit: str,
    platform: str | None = None,
    registry_path: Path = DEFAULT_NATURAL_GRASP_REGISTRY,
) -> None:
    if control_mode not in {"scripted_grasp", "hold_only"}:
        raise ValueError(f"unsupported control_mode: {control_mode}")
    resolved_platform = platform or NATURAL_GRASP_PLATFORMS.get(fixture_id, "isaac")
    registry = load_natural_grasp_registry(registry_path)
    entry = {
        "fixture_id": fixture_id,
        "platform": resolved_platform,
        "control_mode": control_mode,
        "attempt_id": attempt_id,
        "receipt_path": receipt_path,
        "receipt_sha256": receipt_sha256,
        "study_commit": study_commit.lower(),
        "status": "passed",
    }
    passed = [
        item
        for item in registry.get("passed_controls", [])
        if not (
            item.get("fixture_id") == fixture_id
            and item.get("control_mode") == control_mode
        )
    ]
    passed.append(entry)
    registry["passed_controls"] = passed
    save_natural_grasp_registry(registry, registry_path)


def require_natural_grasp_live_control(
    *,
    fixture_id: str,
    study_commit: str | None = None,
    registry_path: Path = DEFAULT_NATURAL_GRASP_REGISTRY,
) -> dict[str, Any]:
    if fixture_id not in FIXTURES_REQUIRING_NATURAL_GRASP_LIVE_CONTROL:
        return {"required": False, "fixture_id": fixture_id}
    platform = NATURAL_GRASP_PLATFORMS.get(fixture_id, "isaac")
    registry = load_natural_grasp_registry(registry_path)
    by_mode = {
        str(item.get("control_mode")): item
        for item in registry.get("passed_controls", [])
        if item.get("fixture_id") == fixture_id and item.get("status") == "passed"
    }
    missing = [
        mode
        for mode in ("scripted_grasp", "hold_only")
        if mode not in by_mode
    ]
    if missing:
        raise DispatchGateError(
            "natural-grasp live-control gate blocked dispatch: fixture "
            f"{fixture_id} ({platform}) missing passed controls {missing}. "
            "Each policy family requires its own live positive (scripted_grasp) and "
            "negative (hold_only) controls on the repaired trigger path; object_pair "
            "registry entries do not cover other fixtures."
        )
    wrong_platform = [
        item.get("control_mode")
        for item in by_mode.values()
        if str(item.get("platform", platform)) != platform
    ]
    if wrong_platform:
        raise DispatchGateError(
            f"natural-grasp live-control registry platform mismatch for {fixture_id}: "
            f"expected {platform}, found mismatched controls {wrong_platform}"
        )
    if study_commit is not None:
        expected = study_commit.lower()
        drifted = [
            item.get("control_mode")
            for item in by_mode.values()
            if str(item.get("study_commit", "")).lower() != expected
        ]
        if drifted:
            raise DispatchGateError(
                f"natural-grasp live-control registry stale for {fixture_id}: "
                f"controls {drifted} recorded under a different study commit than {expected}"
            )
    return {
        "required": True,
        "fixture_id": fixture_id,
        "platform": platform,
        "positive_control": by_mode["scripted_grasp"],
        "negative_control": by_mode["hold_only"],
    }


def record_passed_smoke(
    *,
    fixture_id: str,
    gate: str,
    gpu_product: str,
    attempt_id: str,
    receipt_path: str,
    receipt_sha256: str,
    registry_path: Path = DEFAULT_REGISTRY,
) -> None:
    registry = load_smoke_registry(registry_path)
    entry = {
        "key": smoke_key(fixture_id=fixture_id, gate=gate, gpu_product=gpu_product),
        "fixture_id": fixture_id,
        "gate": gate,
        "gpu_product": gpu_product,
        "attempt_id": attempt_id,
        "receipt_path": receipt_path,
        "receipt_sha256": receipt_sha256,
        "status": "passed",
    }
    passed = [
        item
        for item in registry.get("passed_smokes", [])
        if item.get("key") != entry["key"]
    ]
    passed.append(entry)
    registry["passed_smokes"] = passed
    save_smoke_registry(registry, registry_path)


def require_passed_smoke(
    *,
    fixture_id: str,
    gate: str,
    gpu_product: str,
    registry_path: Path = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    key = smoke_key(fixture_id=fixture_id, gate=gate, gpu_product=gpu_product)
    for item in load_smoke_registry(registry_path).get("passed_smokes", []):
        if item.get("key") == key and item.get("status") == "passed":
            return item
    raise DispatchGateError(
        f"smoke-before-wave gate blocked dispatch: no passed smoke for {key}. "
        f"Run a 1-seed smoke on {gpu_product} for fixture {fixture_id} gate {gate} first."
    )


def collect_output_parents(bundle_root: Path) -> list[str]:
    parents: set[str] = set()
    for job_path in sorted(bundle_root.glob("s*-job.yaml")):
        if not re.fullmatch(r"s\d{3}-job\.yaml", job_path.name):
            continue
        text = job_path.read_text(encoding="utf-8")
        match = JOB_OUTPUT_PARENT_RE.search(text)
        if match is None:
            raise DispatchGateError(f"OUTPUT_PARENT missing in {job_path}")
        parents.add(match.group(1))
    return sorted(parents)


def extract_launch_configs(bundle_root: Path) -> list[dict[str, Any]]:
    launches: list[dict[str, Any]] = []
    for config_path in sorted(bundle_root.glob("s*-configmap.yaml")):
        if not re.fullmatch(r"s\d{3}-configmap\.yaml", config_path.name):
            continue
        text = config_path.read_text(encoding="utf-8")
        match = CONFIGMAP_LAUNCH_RE.search(text)
        if match is None:
            continue
        launch = json.loads(match.group(1))
        if isinstance(launch, dict):
            launches.append(launch)
    if not launches:
        raise DispatchGateError(f"no simulator-launch.json blocks under {bundle_root}")
    return launches


def _kubectl_exec(
    *,
    kube_context: str,
    namespace: str,
    publisher_pod: str,
    script: str,
) -> str:
    completed = subprocess.run(
        [
            "kubectl",
            "--context",
            kube_context,
            "-n",
            namespace,
            "exec",
            publisher_pod,
            "--",
            "bash",
            "-lc",
            script,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise DispatchGateError(completed.stderr or completed.stdout or "kubectl exec failed")
    return completed.stdout


def verify_study_checkout_on_cluster(
    *,
    study_root: str,
    expected_commit: str,
    kube_context: str,
    namespace: str,
    publisher_pod: str,
) -> dict[str, Any]:
    script = (
        f'ROOT="{study_root}"; '
        f'[ -d "$ROOT/.git" ] || {{ echo "missing git dir: $ROOT" >&2; exit 3; }}; '
        f'HEAD=$(git -C "$ROOT" rev-parse HEAD); '
        f'DIRTY=$(git -C "$ROOT" status --porcelain | wc -l | tr -d " "); '
        f'echo "{{\\"head\\":\\"$HEAD\\",\\"dirty_count\\":\\"$DIRTY\\"}}"'
    )
    payload = json.loads(
        _kubectl_exec(
            kube_context=kube_context,
            namespace=namespace,
            publisher_pod=publisher_pod,
            script=script,
        ).strip()
    )
    head = str(payload["head"])
    dirty_count = int(payload["dirty_count"])
    if head != expected_commit.lower():
        raise DispatchGateError(
            f"cluster checkout drift at {study_root}: HEAD={head} expected {expected_commit.lower()}"
        )
    if dirty_count != 0:
        raise DispatchGateError(
            f"cluster checkout at {study_root} is dirty ({dirty_count} paths); sync and clean before dispatch"
        )
    return {"study_root": study_root, "commit": head, "dirty_count": dirty_count}


def verify_file_bindings_on_cluster(
    bindings: Sequence[Mapping[str, Any]],
    *,
    kube_context: str,
    namespace: str,
    publisher_pod: str,
) -> list[dict[str, Any]]:
    checked: list[dict[str, Any]] = []
    for binding in bindings:
        path = str(binding["path"])
        # Lane scripts ship in the immutable bundle ConfigMap and mount at
        # /opt/v4-lane inside the pod; they are not PVC-resident study files.
        if path.startswith("/opt/v4-lane/"):
            continue
        expected_sha = str(binding["sha256"]).lower()
        expected_bytes = int(binding["bytes"])
        script = (
            f'P="{path}"; '
            f'[ -f "$P" ] || {{ echo "missing: $P" >&2; exit 4; }}; '
            f'B=$(stat -c%s "$P"); '
            f'S=$(sha256sum "$P" | awk "{{print \\$1}}"); '
            f'echo "{{\\"bytes\\":\\"$B\\",\\"sha256\\":\\"$S\\"}}"'
        )
        payload = json.loads(
            _kubectl_exec(
                kube_context=kube_context,
                namespace=namespace,
                publisher_pod=publisher_pod,
                script=script,
            ).strip()
        )
        actual_sha = str(payload["sha256"]).lower()
        actual_bytes = int(payload["bytes"])
        if actual_sha != expected_sha or actual_bytes != expected_bytes:
            raise DispatchGateError(
                f"file_bindings drift for {path}: expected {expected_bytes}/{expected_sha}, "
                f"cluster has {actual_bytes}/{actual_sha}. Re-render bundle after checkout sync."
            )
        checked.append({"path": path, "bytes": actual_bytes, "sha256": actual_sha})
    return checked


def verify_bundle_cluster_consistency(
    bundle_root: Path,
    *,
    kube_context: str,
    namespace: str,
    publisher_pod: str,
    expected_study_commit: str | None = None,
) -> dict[str, Any]:
    launches = extract_launch_configs(bundle_root)
    study_roots: set[str] = set()
    all_bindings: list[dict[str, Any]] = []
    for launch in launches:
        argv = launch.get("experiment_argv") or []
        for index, token in enumerate(argv):
            if token == "--study-root" and index + 1 < len(argv):
                study_roots.add(str(argv[index + 1]))
            if token == "--expected-study-commit" and index + 1 < len(argv):
                if expected_study_commit is None:
                    expected_study_commit = str(argv[index + 1])
        for binding in launch.get("file_bindings") or []:
            if isinstance(binding, dict):
                all_bindings.append(binding)
    if expected_study_commit is None:
        raise DispatchGateError("bundle launch config lacks --expected-study-commit")
    checkout_reports = [
        verify_study_checkout_on_cluster(
            study_root=root,
            expected_commit=expected_study_commit,
            kube_context=kube_context,
            namespace=namespace,
            publisher_pod=publisher_pod,
        )
        for root in sorted(study_roots)
    ]
    binding_report = verify_file_bindings_on_cluster(
        all_bindings,
        kube_context=kube_context,
        namespace=namespace,
        publisher_pod=publisher_pod,
    )
    return {
        "expected_study_commit": expected_study_commit,
        "study_roots": checkout_reports,
        "file_bindings_checked": len(binding_report),
    }


def prepare_output_parents_on_cluster(
    bundle_root: Path,
    *,
    kube_context: str,
    namespace: str,
    publisher_pod: str,
) -> int:
    parents = collect_output_parents(bundle_root)
    quoted = " ".join(f'"{parent}"' for parent in parents)
    script = f"mkdir -p {quoted} && echo prepared={len(parents)}"
    output = _kubectl_exec(
        kube_context=kube_context,
        namespace=namespace,
        publisher_pod=publisher_pod,
        script=script,
    ).strip()
    if not output.endswith(f"prepared={len(parents)}"):
        raise DispatchGateError(f"unexpected OUTPUT_PARENT preparation result: {output}")
    return len(parents)


def enforce_dispatch_gates(
    *,
    bundle_root: Path,
    fixture_id: str,
    gate: str,
    gpu_product: str,
    kube_context: str,
    namespace: str,
    publisher_pod: str,
    expected_study_commit: str | None,
    is_smoke: bool,
    registry_path: Path = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    if not is_smoke:
        require_passed_smoke(
            fixture_id=fixture_id,
            gate=gate,
            gpu_product=gpu_product,
            registry_path=registry_path,
        )
    natural_grasp_report: dict[str, Any] | None = None
    if (
        not is_smoke
        and fixture_id in FIXTURES_REQUIRING_NATURAL_GRASP_LIVE_CONTROL
        and gate in POLICY_TRIGGER_GATES
    ):
        natural_grasp_report = require_natural_grasp_live_control(
            fixture_id=fixture_id,
            study_commit=expected_study_commit,
        )
    consistency = verify_bundle_cluster_consistency(
        bundle_root,
        kube_context=kube_context,
        namespace=namespace,
        publisher_pod=publisher_pod,
        expected_study_commit=expected_study_commit,
    )
    prepared = prepare_output_parents_on_cluster(
        bundle_root,
        kube_context=kube_context,
        namespace=namespace,
        publisher_pod=publisher_pod,
    )
    return {
        "consistency": consistency,
        "output_parents_prepared": prepared,
        "is_smoke": is_smoke,
        "natural_grasp_live_control": natural_grasp_report,
    }
