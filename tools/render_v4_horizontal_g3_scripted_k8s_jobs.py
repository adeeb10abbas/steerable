#!/usr/bin/env python3
"""Render immutable simulator-only Kubernetes Jobs for horizontal G3 scripted seeds."""

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

DEFAULT_SPEC = (
    ROOT / "deploy/k8s/v4_lane_bundle/g3-scripted-horizontal-spec.example.json"
)
DEFAULT_OUTPUT = ROOT / "deploy/k8s/v4_lane_bundle/rendered-g3-scripted"
SCHEMA = "vla-wam-v4-horizontal-g3-scripted-k8s-render-spec-v1"
PLAN_SCHEMA = "v4-horizontal-g3-plan-v1"
RESET_SCHEMA = "v4-droid-horizontal-reset-registry-v1"
SCRIPTED_MODES = ("stationary", "moving")
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
    "scale",
    "marker_wrapper_source",
    "marker_wrapper_path",
    "runner_source",
    "runner_path",
    "gate_core_source",
    "gate_core_path",
    "campaign_source",
    "campaign_path",
    "campaign_sha256",
    "plan_source",
    "plan_path",
    "plan_sha256",
    "reset_registry_source",
    "reset_registry_path",
    "reset_registry_sha256",
    "render_probe_argv",
    "python_imports",
}


class G3ScriptedRenderError(ValueError):
    """Raised when a model-blind G3 scripted bundle is not immutable and complete."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G3ScriptedRenderError(message)


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
        raise G3ScriptedRenderError(f"invalid G3 scripted render spec JSON: {path}") from exc
    require(isinstance(spec, dict), "G3 scripted render spec must be an object")
    unknown = sorted(set(spec) - TOP_LEVEL_KEYS)
    require(not unknown, f"G3 scripted render spec contains unknown keys: {unknown}")
    require(spec.get("schema_version") == SCHEMA, "G3 scripted render spec schema differs")
    return spec, hashlib.sha256(raw).hexdigest()


def _binding(source: Path, path: str) -> dict[str, Any]:
    return {
        "path": _absolute(path, "binding path"),
        "bytes": source.stat().st_size,
        "sha256": lane.sha256_file(source),
    }


def _scale_label(scale: float) -> str:
    value = float(scale)
    if value.is_integer():
        return f"{int(value)}.0"
    return str(value)


def _job_output_parent(
    *, output_parent: str, attempt: str, scale: float, mode: str, seed: int
) -> str:
    return (
        f"{output_parent}/attempt-{attempt}/"
        f"scale-{_scale_label(scale)}/{mode}/seed-{seed}"
    )


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


def _scripted_jobs(plan_payload: Mapping[str, Any]) -> list[tuple[int, str, str]]:
    scripted = plan_payload.get("scripted_controller")
    require(isinstance(scripted, Mapping), "G3 plan lacks scripted_controller")
    reset_seeds_raw = scripted.get("reset_env_seeds")
    require(
        isinstance(reset_seeds_raw, list) and reset_seeds_raw,
        "G3 plan lacks scripted reset_env_seeds",
    )
    reset_seeds = [int(seed) for seed in reset_seeds_raw]
    require(len(reset_seeds) == 9, "G3 scripted plan must register nine reset env seeds")
    moving = scripted.get("moving")
    require(isinstance(moving, Mapping), "G3 plan lacks moving scripted checks")
    canonical_seed = moving.get("canonical_env_seed")
    require(type(canonical_seed) is int, "G3 plan lacks canonical_env_seed")
    jobs: list[tuple[int, str, str]] = []
    for index, seed in enumerate(reset_seeds):
        jobs.append((seed, "stationary", f"g3hs-st{index:03d}"))
    jobs.append((canonical_seed, "moving", "g3hs-mv000"))
    require(len(jobs) == 10, "G3 scripted bundle must render exactly ten jobs")
    return jobs


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
    marker_wrapper_path = _absolute(
        spec.get("marker_wrapper_path"), "marker_wrapper_path"
    )
    runner_path = _absolute(spec.get("runner_path"), "runner_path")
    gate_core_path = _absolute(spec.get("gate_core_path"), "gate_core_path")
    campaign_path = _absolute(spec.get("campaign_path"), "campaign_path")
    plan_path = _absolute(spec.get("plan_path"), "plan_path")
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
    scale = spec.get("scale")
    require(
        isinstance(scale, (int, float))
        and not isinstance(scale, bool)
        and float(scale) > 0,
        "scale must be positive",
    )
    scale_value = float(scale)
    runtime = spec.get("runtime")
    require(isinstance(runtime, dict), "runtime must be an object")
    unknown_runtime = sorted(set(runtime) - lane.RUNTIME_KEYS)
    require(not unknown_runtime, f"runtime contains unknown keys: {unknown_runtime}")
    for key in lane.RUNTIME_KEYS:
        if key != "pythonpath":
            _absolute(runtime.get(key), f"runtime.{key}")
        else:
            require(
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

    marker_wrapper_source = _source(
        spec.get("marker_wrapper_source"),
        spec_dir=spec_path.parent,
        label="marker_wrapper_source",
    )
    runner_source = _source(
        spec.get("runner_source"), spec_dir=spec_path.parent, label="runner_source"
    )
    gate_core_source = _source(
        spec.get("gate_core_source"),
        spec_dir=spec_path.parent,
        label="gate_core_source",
    )
    campaign_source = _source(
        spec.get("campaign_source"),
        spec_dir=spec_path.parent,
        label="campaign_source",
    )
    plan_source = _source(
        spec.get("plan_source"), spec_dir=spec_path.parent, label="plan_source"
    )
    reset_registry_source = _source(
        spec.get("reset_registry_source"),
        spec_dir=spec_path.parent,
        label="reset_registry_source",
    )
    campaign_sha = lane.digest(spec.get("campaign_sha256"), "campaign_sha256")
    require(
        lane.sha256_file(campaign_source) == campaign_sha,
        "campaign source SHA-256 differs from spec",
    )
    plan_sha = lane.digest(spec.get("plan_sha256"), "plan_sha256")
    require(
        lane.sha256_file(plan_source) == plan_sha,
        "plan source SHA-256 differs from spec",
    )
    reset_sha = lane.digest(
        spec.get("reset_registry_sha256"), "reset_registry_sha256"
    )
    require(
        lane.sha256_file(reset_registry_source) == reset_sha,
        "reset registry source SHA-256 differs from spec",
    )

    campaign_payload = json.loads(campaign_source.read_text(encoding="utf-8"))
    require(
        campaign_payload.get("campaign_id") == "online_correction_v4",
        "campaign identity differs",
    )
    plan_payload = json.loads(plan_source.read_text(encoding="utf-8"))
    require(
        plan_payload.get("schema_version") == PLAN_SCHEMA,
        "G3 plan schema differs",
    )
    require(
        plan_payload.get("status")
        == "model_blind_candidate_not_released_for_inference",
        "G3 plan must remain an unreleased model-blind candidate",
    )
    require(
        plan_payload.get("plan_status") == "ready_for_live_g3_execution",
        "G3 plan is not ready for live execution",
    )
    require(
        plan_payload.get("model_request_count") == 0
        and plan_payload.get("behavioral_episode_count") == 0,
        "G3 plan candidate contains model or behavioral activity",
    )
    plan_source_identity = plan_payload.get("source_identity") or {}
    require(
        isinstance(plan_source_identity, dict),
        "G3 plan lacks source identity bindings",
    )
    for label, expected_sha, actual_sha in (
        ("campaign", campaign_sha, (plan_source_identity.get("campaign") or {}).get("sha256")),
        (
            "reset registry",
            reset_sha,
            (plan_source_identity.get("reset_registry") or {}).get("sha256"),
        ),
    ):
        require(
            actual_sha == expected_sha,
            f"G3 plan {label} SHA-256 differs from spec",
        )
    scale_candidates = (
        (plan_payload.get("scale_selection") or {}).get("candidate_scales_descending")
    )
    require(
        isinstance(scale_candidates, list) and scale_value in {
            float(item) for item in scale_candidates
        },
        "selected scale is not registered in the G3 plan",
    )
    scripted_jobs = _scripted_jobs(plan_payload)

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
    stem = f"v4-g3-horizontal-scripted-{attempt}-{spec_sha256[:10]}"
    scripts_name = f"{stem}-scripts"
    bundle_root = output_root.resolve() / stem
    require(not bundle_root.exists(), f"refusing to overwrite bundle: {bundle_root}")
    bundle_root.mkdir(parents=True)

    common_script_labels = {
        "app.kubernetes.io/name": "v4-horizontal-g3-scripted",
        "app.kubernetes.io/part-of": "vla-wam-v4",
        "v4-gate": "g3-horizontal-scripted",
        "v4-attempt-id": attempt,
        "v4-config-sha": spec_sha256[:16],
        "v4-scale": _scale_label(scale_value),
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
    job_identities: list[dict[str, Any]] = []
    job_output_parents: list[str] = []
    for index, (seed, mode, lane_id) in enumerate(scripted_jobs):
        config_name = f"{stem}-j{index:03d}-config"
        job_name = f"{stem}-j{index:03d}"
        job_output = _job_output_parent(
            output_parent=output_parent,
            attempt=attempt,
            scale=scale_value,
            mode=mode,
            seed=seed,
        )
        job_output_parents.append(job_output)
        labels = {
            **common_script_labels,
            "v4-lane-id": lane_id,
            "v4-lane-role": "simulator",
            "v4-job-index": f"{index:03d}",
            "v4-scripted-mode": mode,
            "v4-environment-seed": str(seed),
        }
        bindings = [
            _binding(marker_wrapper_source, marker_wrapper_path),
            _binding(runner_source, runner_path),
            _binding(gate_core_source, gate_core_path),
            _binding(campaign_source, campaign_path),
            _binding(plan_source, plan_path),
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
            marker_wrapper_path,
            "--expected-environment-seed",
            str(seed),
            "--expected-scale",
            _scale_label(scale_value),
            "--expected-mode",
            mode,
            "--",
            runtime["python_bin"],
            runner_path,
            "--study-root",
            study_root,
            "--robolab-root",
            robolab_root,
            "--campaign",
            campaign_path,
            "--campaign-sha256",
            campaign_sha,
            "--plan",
            plan_path,
            "--plan-sha256",
            plan_sha,
            "--reset-registry",
            reset_registry_path,
            "--reset-registry-sha256",
            reset_sha,
            "--environment-seed",
            str(seed),
            "--scale",
            _scale_label(scale_value),
            "--mode",
            mode,
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
            "execution_scope": "model_blind_g3_no_policy",
            "experiment_argv": argv,
            "file_bindings": bindings,
            "gpu_product": gpu_product,
            "expected_gpu_name": expected_gpu_name,
            "expected_driver_version": expected_driver,
            "checkpoint_path": plan_path,
            "checkpoint_sha256": plan_sha,
            "checkpoint_semantics": "model_blind_g3_plan_candidate",
            "nvidia_smi_bin": "/usr/bin/nvidia-smi",
            "python_imports": python_imports,
            "vulkan_contract": "isaac_app_launcher_rtx_frame_under_bound_vk_icd",
            "render_probe_argv": render_probe,
            "render_probe_timeout_seconds": 300,
            "policy_wait": None,
        }
        launch_json = lane.canonical_json(launch)
        launch_sha = hashlib.sha256(launch_json.encode("utf-8")).hexdigest()
        config_file = f"j{index:03d}-configmap.yaml"
        job_file = f"j{index:03d}-job.yaml"
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
            output_parent=job_output,
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
        job_identities.append(
            {
                "environment_seed": seed,
                "mode": mode,
                "lane_id": lane_id,
                "job_name": job_name,
                "configmap_name": config_name,
                "output_parent": job_output,
                "launch_sha256": launch_sha,
            }
        )

    require(
        len(set(job_output_parents)) == len(scripted_jobs),
        "G3 scripted job output parents must be unique",
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
        "schema_version": "vla-wam-v4-horizontal-g3-scripted-k8s-bundle-v1",
        "status": "rendered_not_created",
        "campaign_id": "online_correction_v4",
        "fixture_id": "horizontal",
        "execution_scope": "model_blind_g3_no_policy",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "kube_context": kube_context,
        "namespace": namespace,
        "attempt_id": attempt,
        "scale": scale_value,
        "gpu_product": gpu_product,
        "expected_study_commit": expected_study_commit,
        "expected_robolab_commit": expected_robolab_commit,
        "native_control_dt_s": float(native_control_dt_s),
        "campaign_sha256": campaign_sha,
        "plan_sha256": plan_sha,
        "reset_registry_sha256": reset_sha,
        "job_count": len(scripted_jobs),
        "scripted_jobs": [
            {"environment_seed": seed, "mode": mode, "lane_id": lane_id}
            for seed, mode, lane_id in scripted_jobs
        ],
        "render_spec_sha256": spec_sha256,
        "renderer_sha256": renderer_sha,
        "runtime_scripts_sha256": scripts_identity,
        "marker_wrapper_sha256": lane.sha256_file(marker_wrapper_source),
        "runner_sha256": lane.sha256_file(runner_source),
        "gate_core_sha256": lane.sha256_file(gate_core_source),
        "files_sha256": file_hashes,
        "job_identities": job_identities,
        "release_boundary": (
            "Creating this bundle runs only zero-inference G3 scripted-seed checks at "
            "one registered scale. It does not authorize policy inference."
        ),
    }
    manifest_path = bundle_root / "bundle-manifest.json"
    manifest_path.write_text(lane.canonical_json(manifest), encoding="utf-8")
    return {
        "bundle_root": str(bundle_root),
        "bundle_manifest": str(manifest_path),
        "bundle_manifest_sha256": lane.sha256_file(manifest_path),
        "job_count": len(scripted_jobs),
        "scale": scale_value,
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
