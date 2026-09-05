#!/usr/bin/env python3
"""Render one immutable simulator-only Kubernetes Job per horizontal G2 seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import render_v4_k8s_lane_bundle as lane  # noqa: E402

DEFAULT_SPEC = ROOT / "deploy/k8s/v4_lane_bundle/g2-horizontal-spec.example.json"
DEFAULT_OUTPUT = ROOT / "deploy/k8s/v4_lane_bundle/rendered-g2"
SCHEMA = "vla-wam-v4-horizontal-g2-k8s-render-spec-v1"
RESET_SCHEMA = "v4-droid-horizontal-reset-registry-v1"
TOP_LEVEL_KEYS = {
    "schema_version",
    "kube_context",
    "namespace",
    "attempt_id",
    "image_repository",
    "image_sha256",
    "image_pull_secret",
    "pvc",
    "output_parent",
    "prestop_wait_seconds",
    "expected_driver_version",
    "gpu_product",
    "expected_gpu_name",
    "runtime",
    "study_root",
    "expected_study_commit",
    "robolab_root",
    "expected_robolab_commit",
    "native_control_dt_s",
    "runner_source",
    "runner_path",
    "gate_core_source",
    "gate_core_path",
    "reset_registry_source",
    "reset_registry_path",
    "reset_registry_sha256",
    "render_probe_argv",
    "python_imports",
    "max_seed_jobs",
}


class G2RenderError(ValueError):
    """Raised when a model-blind G2 bundle is not immutable and complete."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G2RenderError(message)


def _absolute(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and Path(value).is_absolute(),
        f"{label} must be an absolute path",
    )
    return str(value)


def _source(path: Any, *, spec_dir: Path, label: str) -> Path:
    require(isinstance(path, str) and path, f"{label} is required")
    value = Path(path)
    resolved = value.resolve() if value.is_absolute() else (spec_dir / value).resolve()
    require(resolved.is_file(), f"{label} does not exist: {resolved}")
    return resolved


def _read_spec(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise G2RenderError(f"invalid G2 render spec JSON: {path}") from exc
    require(isinstance(spec, dict), "G2 render spec must be an object")
    unknown = sorted(set(spec) - TOP_LEVEL_KEYS)
    require(not unknown, f"G2 render spec contains unknown keys: {unknown}")
    require(spec.get("schema_version") == SCHEMA, "G2 render spec schema differs")
    return spec, hashlib.sha256(raw).hexdigest()


def _binding(source: Path, path: str) -> dict[str, Any]:
    return {
        "path": _absolute(path, "binding path"),
        "bytes": source.stat().st_size,
        "sha256": lane.sha256_file(source),
    }


def _render_launch_configmap(
    *,
    name: str,
    namespace: str,
    labels: Mapping[str, str],
    launch_json: str,
    image_digest: str,
    spec_sha256: str,
    renderer_sha256: str,
) -> str:
    rows = ["apiVersion: v1", "kind: ConfigMap", "metadata:"]
    rows += lane.metadata_lines(name, namespace, labels, "  ")
    rows += [
        "immutable: true",
        "data:",
        "  simulator-launch.json: |",
        lane.block(launch_json, 4),
        f"  image.digest: {lane.yaml_scalar(image_digest)}",
        f"  render-spec.sha256: {lane.yaml_scalar(spec_sha256)}",
        f"  renderer.sha256: {lane.yaml_scalar(renderer_sha256)}",
    ]
    return "\n".join(rows) + "\n"


def render(spec_path: Path, output_root: Path) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    spec, spec_sha256 = _read_spec(spec_path)
    namespace = lane.token(spec.get("namespace"), "namespace")
    attempt = lane.token(spec.get("attempt_id"), "attempt_id")
    kube_context = str(spec.get("kube_context") or "")
    require(kube_context, "kube_context is required")
    image_sha = lane.digest(spec.get("image_sha256"), "image_sha256")
    image_repository = spec.get("image_repository")
    require(
        isinstance(image_repository, str)
        and image_repository
        and "@" not in image_repository,
        "image_repository is invalid",
    )
    image_digest = f"sha256:{image_sha}"
    image = f"{image_repository}@{image_digest}"
    image_pull_secret = lane.token(
        spec.get("image_pull_secret"), "image_pull_secret"
    )
    pvc = lane.token(spec.get("pvc"), "pvc")
    output_parent = _absolute(spec.get("output_parent"), "output_parent")
    study_root = _absolute(spec.get("study_root"), "study_root")
    robolab_root = _absolute(spec.get("robolab_root"), "robolab_root")
    runner_path = _absolute(spec.get("runner_path"), "runner_path")
    gate_core_path = _absolute(spec.get("gate_core_path"), "gate_core_path")
    reset_registry_path = _absolute(
        spec.get("reset_registry_path"), "reset_registry_path"
    )
    expected_study_commit = str(spec.get("expected_study_commit") or "")
    require(
        len(expected_study_commit) == 40
        and all(char in "0123456789abcdef" for char in expected_study_commit),
        "expected_study_commit must be a lowercase Git SHA",
    )
    expected_robolab_commit = str(spec.get("expected_robolab_commit") or "")
    require(
        len(expected_robolab_commit) == 40
        and all(char in "0123456789abcdef" for char in expected_robolab_commit),
        "expected_robolab_commit must be a lowercase Git SHA",
    )
    expected_driver = str(spec.get("expected_driver_version") or "")
    require(
        expected_driver and expected_driver.count(".") == 2,
        "expected_driver_version is invalid",
    )
    gpu_product = lane.gpu_product(spec.get("gpu_product"), "gpu_product")
    expected_gpu_name = lane.gpu_display_name(
        spec.get("expected_gpu_name"), "expected_gpu_name"
    )
    native_control_dt_s = spec.get("native_control_dt_s")
    require(
        isinstance(native_control_dt_s, (int, float))
        and not isinstance(native_control_dt_s, bool)
        and float(native_control_dt_s) > 0,
        "native_control_dt_s must be positive",
    )
    runtime = spec.get("runtime")
    require(isinstance(runtime, dict), "runtime must be an object")
    unknown_runtime = sorted(set(runtime) - lane.RUNTIME_KEYS)
    require(not unknown_runtime, f"runtime contains unknown keys: {unknown_runtime}")
    for key in lane.RUNTIME_KEYS:
        _absolute(runtime.get(key), f"runtime.{key}") if key != "pythonpath" else require(
            isinstance(runtime.get(key), str) and runtime[key],
            "runtime.pythonpath is required",
        )
    render_probe = spec.get("render_probe_argv")
    require(
        isinstance(render_probe, list)
        and render_probe
        and all(isinstance(item, str) and item for item in render_probe)
        and any("{rendered_frame}" in item for item in render_probe),
        "render_probe_argv is invalid",
    )
    python_imports = spec.get("python_imports")
    require(
        isinstance(python_imports, list)
        and python_imports
        and all(isinstance(item, str) and item for item in python_imports)
        and "curobo" in python_imports,
        "python_imports must include curobo",
    )

    runner_source = _source(
        spec.get("runner_source"), spec_dir=spec_path.parent, label="runner_source"
    )
    gate_core_source = _source(
        spec.get("gate_core_source"),
        spec_dir=spec_path.parent,
        label="gate_core_source",
    )
    reset_registry_source = _source(
        spec.get("reset_registry_source"),
        spec_dir=spec_path.parent,
        label="reset_registry_source",
    )
    reset_sha = lane.digest(
        spec.get("reset_registry_sha256"), "reset_registry_sha256"
    )
    require(
        lane.sha256_file(reset_registry_source) == reset_sha,
        "reset registry source SHA-256 differs from spec",
    )
    reset_payload = json.loads(reset_registry_source.read_text(encoding="utf-8"))
    require(
        reset_payload.get("schema_version") == RESET_SCHEMA,
        "reset registry schema differs",
    )
    require(
        reset_payload.get("status")
        == "model_blind_candidate_not_released_for_inference",
        "reset registry must remain an unreleased model-blind candidate",
    )
    require(
        reset_payload.get("model_request_count") == 0
        and reset_payload.get("behavioral_episode_count") == 0,
        "reset registry candidate contains model or behavioral activity",
    )
    resets = reset_payload.get("resets_by_env_seed")
    require(isinstance(resets, dict) and resets, "reset registry has no seeds")
    seeds = sorted(int(seed) for seed in resets)
    max_seed_jobs = int(spec.get("max_seed_jobs", len(seeds)))
    require(
        max_seed_jobs == len(seeds),
        "max_seed_jobs must equal complete registered reset coverage",
    )

    scripts = lane.load_runtime_scripts(
        ROOT / "deploy/k8s/v4_lane_bundle/scripts"
    )
    renderer_sha = lane.sha256_file(Path(__file__).resolve())
    scripts_identity = hashlib.sha256(
        b"".join(
            name.encode("utf-8") + b"\0" + scripts[name]
            for name in sorted(scripts)
        )
    ).hexdigest()
    stem = f"v4-g2-horizontal-{attempt}-{spec_sha256[:10]}"
    scripts_name = f"{stem}-scripts"
    bundle_root = output_root.resolve() / stem
    require(not bundle_root.exists(), f"refusing to overwrite bundle: {bundle_root}")
    bundle_root.mkdir(parents=True)

    common_script_labels = {
        "app.kubernetes.io/name": "v4-horizontal-g2",
        "app.kubernetes.io/part-of": "vla-wam-v4",
        "v4-gate": "g2-horizontal",
        "v4-attempt-id": attempt,
        "v4-config-sha": spec_sha256[:16],
    }
    files: dict[str, str] = {
        "scripts-configmap.yaml": lane.render_scripts_configmap(
            name=scripts_name,
            namespace=namespace,
            common_labels=common_script_labels,
            scripts=scripts,
        )
    }
    resources = ["scripts-configmap.yaml"]
    seed_identities: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        lane_id = f"g2h-s{index:03d}"
        config_name = f"{stem}-s{index:03d}-config"
        job_name = f"{stem}-s{index:03d}"
        labels = {
            **common_script_labels,
            "v4-lane-id": lane_id,
            "v4-lane-role": "simulator",
            "v4-seed-index": f"{index:03d}",
        }
        bindings = [
            _binding(runner_source, runner_path),
            _binding(gate_core_source, gate_core_path),
            _binding(reset_registry_source, reset_registry_path),
            _binding(
                ROOT / "deploy/k8s/v4_lane_bundle/scripts/lane_entrypoint.py",
                "/opt/v4-lane/scripts/lane_entrypoint.py",
            ),
            _binding(
                ROOT / "deploy/k8s/v4_lane_bundle/scripts/startup_preflight.py",
                "/opt/v4-lane/scripts/startup_preflight.py",
            ),
            _binding(
                ROOT / "deploy/k8s/v4_lane_bundle/scripts/isaac_render_probe.py",
                "/opt/v4-lane/scripts/isaac_render_probe.py",
            ),
        ]
        argv = [
            runtime["python_bin"],
            runner_path,
            "--study-root",
            study_root,
            "--robolab-root",
            robolab_root,
            "--reset-registry",
            reset_registry_path,
            "--reset-registry-sha256",
            reset_sha,
            "--environment-seed",
            str(seed),
            "--expected-study-commit",
            expected_study_commit,
            "--expected-robolab-commit",
            expected_robolab_commit,
            "--expected-driver-version",
            expected_driver,
            "--native-control-dt-s",
            str(float(native_control_dt_s)),
        ]
        launch = {
            "schema_version": "vla-wam-v4-k8s-lane-launch-v1",
            "role": "simulator",
            "execution_scope": "model_blind_g2_no_policy",
            "experiment_argv": argv,
            "file_bindings": bindings,
            "gpu_product": gpu_product,
            "expected_gpu_name": expected_gpu_name,
            "expected_driver_version": expected_driver,
            "checkpoint_path": reset_registry_path,
            "checkpoint_sha256": reset_sha,
            "checkpoint_semantics": "model_blind_reset_registry_candidate",
            "nvidia_smi_bin": "/usr/bin/nvidia-smi",
            "python_imports": python_imports,
            "vulkan_contract": "isaac_app_launcher_rtx_frame_under_bound_vk_icd",
            "render_probe_argv": render_probe,
            "render_probe_timeout_seconds": 300,
            "policy_wait": None,
        }
        launch_json = lane.canonical_json(launch)
        launch_sha = hashlib.sha256(launch_json.encode("utf-8")).hexdigest()
        config_file = f"s{index:03d}-configmap.yaml"
        job_file = f"s{index:03d}-job.yaml"
        files[config_file] = _render_launch_configmap(
            name=config_name,
            namespace=namespace,
            labels=labels,
            launch_json=launch_json,
            image_digest=image_digest,
            spec_sha256=spec_sha256,
            renderer_sha256=renderer_sha,
        )
        files[job_file] = lane.render_job(
            role="simulator",
            name=job_name,
            namespace=namespace,
            common_labels=labels,
            configmap=config_name,
            scripts_configmap=scripts_name,
            image=image,
            image_digest=image_digest,
            gpu_product_value=gpu_product,
            lane=lane_id,
            attempt=attempt,
            output_parent=output_parent,
            launch_sha=launch_sha,
            runtime=runtime,
            policy_port=1,
            pvc=pvc,
            image_pull_secret=image_pull_secret,
            entrypoint="/opt/v4-lane/scripts/lane_entrypoint.py",
            prestop_wait_seconds=int(spec.get("prestop_wait_seconds", 120)),
            kube_context=kube_context,
        )
        resources.extend([config_file, job_file])
        seed_identities.append(
            {
                "environment_seed": seed,
                "lane_id": lane_id,
                "job_name": job_name,
                "configmap_name": config_name,
                "launch_sha256": launch_sha,
            }
        )

    files["kustomization.yaml"] = (
        "apiVersion: kustomize.config.k8s.io/v1beta1\n"
        "kind: Kustomization\n"
        "resources:\n"
        + "".join(f"  - {name}\n" for name in resources)
    )
    for name, body in files.items():
        (bundle_root / name).write_text(body, encoding="utf-8")
    file_hashes = {
        name: lane.sha256_file(bundle_root / name) for name in sorted(files)
    }
    manifest = {
        "schema_version": "vla-wam-v4-horizontal-g2-k8s-bundle-v1",
        "status": "rendered_not_created",
        "campaign_id": "online_correction_v4",
        "fixture_id": "horizontal",
        "execution_scope": "model_blind_g2_no_policy",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "kube_context": kube_context,
        "namespace": namespace,
        "attempt_id": attempt,
        "gpu_product": gpu_product,
        "expected_study_commit": expected_study_commit,
        "expected_robolab_commit": expected_robolab_commit,
        "native_control_dt_s": float(native_control_dt_s),
        "seed_count": len(seeds),
        "render_spec_sha256": spec_sha256,
        "renderer_sha256": renderer_sha,
        "runtime_scripts_sha256": scripts_identity,
        "reset_registry_sha256": reset_sha,
        "files_sha256": file_hashes,
        "seed_identities": seed_identities,
        "release_boundary": (
            "Creating this bundle runs only zero-inference reset/camera G2 seed "
            "checks. It does not complete rendered-axis review or authorize policy inference."
        ),
    }
    manifest_path = bundle_root / "bundle-manifest.json"
    manifest_path.write_text(lane.canonical_json(manifest), encoding="utf-8")
    return {
        "bundle_root": str(bundle_root),
        "bundle_manifest": str(manifest_path),
        "bundle_manifest_sha256": lane.sha256_file(manifest_path),
        "seed_count": len(seeds),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(render(args.spec, args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
