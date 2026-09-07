#!/usr/bin/env python3
"""Render and optionally kubectl-create one natural-grasp live-control Isaac job."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import render_v4_k8s_lane_bundle as lane  # noqa: E402
import v4_gpu_scheduling as gpu_scheduling  # noqa: E402
import v4_study_checkout_isolation as checkout_isolation  # noqa: E402
import v4_wave_admission as wave_admission  # noqa: E402

DEFAULT_STUDY_ROOT = "/data/users/ali/vla_wam/src/steerable-v4-g3-c5c6-424a91b"
CAMPAIGN_SOURCE = ROOT / "artifacts/online_correction_v4/setup/c7_confirmatory/campaign.frozen.json"
CAMPAIGN_SHA256 = "7b207c38353a2194859a01130318237cc55231e3504b1a2632b12b28f93c22b9"
ROBOLAB_ROOT = "/data/users/ali/vla_wam/external/RoboLab-11142d4"
EXPECTED_ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
IMAGE_SHA256 = "03f5ce7d090fbd378070a8216d0aedfc6e473c52da99b40b0cf53918612a297c"
IMAGE = f"artifactory-ci.gm.com/docker-approved/devcontainers/base@sha256:{IMAGE_SHA256}"
KUBE_CONTEXT = "prod-dcwi-warrenq1-vmkub007"
NAMESPACE = "211247-prod"
PVC = "211247-prod-pvc"
PUBLISHER_POD = "211247-sz5vjy-vla4-b200-4gpu"
EXPECTED_DRIVER = "580.95.05"
NATIVE_CONTROL_DT_S = 0.06666666666666667
SCALE = 0.5

FIXTURE_CONFIG: dict[str, dict[str, Any]] = {
    "vertical": {
        "environment_seed": 2100020000,
        "goal": "above",
        "plan_source": ROOT / "artifacts/online_correction_v4/setup/vertical_g3_plan.candidate.json",
        "plan_sha256": "ad4cd7efc5d49790a51d76137fef6d631328a9f1b09ae65b5d52470f8976e37d",
        "reset_registry_source": ROOT
        / "artifacts/online_correction_v4/setup/vertical_reset_registry.candidate.json",
        "reset_registry_sha256": "c1c3541ffebf1e9d43c4f75f6a2ee784e6065e99ae8cb7bcd88efca0c03b394c",
        "g2_aggregate_source": ROOT
        / "artifacts/online_correction_v4/qualification/20260906_vertical_g2_aggregate_g2c5q20260906c.json",
        "g2_aggregate_sha256": "2a35cc05b1e5f419a06f61afe64efc0f92cf9e8b89d5c307f502aa3ea2d401b2",
        "runner_source": ROOT / "tools/run_v4_vertical_natural_grasp_live_positive_control.py",
        "output_qual_dir": "vertical-natural-grasp-positive-control",
        "k8s_name_token": "v-ngpc",
    },
    "containment": {
        "environment_seed": 2100030000,
        "goal": "inside",
        "plan_source": ROOT / "artifacts/online_correction_v4/setup/containment_g3_plan.candidate.json",
        "plan_sha256": "e9f6bbff2676ba80a39bbdef6ea52f48212240d6df8fc063c8ebd9c5d3cdc7fc",
        "reset_registry_source": ROOT
        / "artifacts/online_correction_v4/setup/containment_reset_registry.candidate.json",
        "reset_registry_sha256": "e1b4b7d43dfdb4b12d847aaefada9658c0858ca8696101e86112bf6dd9f69c1a",
        "g2_aggregate_source": ROOT
        / "artifacts/online_correction_v4/qualification/20260906_containment_g2_aggregate_g2c6q20260906c.json",
        "g2_aggregate_sha256": "f639a143e87a2afb343a34ebc88ca6174921b63b960aad415331b34984d6a54c",
        "runner_source": ROOT / "tools/run_v4_containment_natural_grasp_live_positive_control.py",
        "output_qual_dir": "containment-natural-grasp-positive-control",
        "k8s_name_token": "c-ngpc",
    },
}


def _binding(source: Path, path: str) -> dict[str, Any]:
    return {
        "path": path,
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
) -> str:
    rows = ["apiVersion: v1", "kind: ConfigMap", "metadata:"]
    rows += lane.metadata_lines(name, namespace, labels, "  ")
    rows += ["immutable: true", "data:", "  simulator-launch.json: |"]
    for line in launch_json.splitlines():
        rows.append(f"    {line}")
    rows.append(f'  image.digest: "{image_digest}"')
    return "\n".join(rows) + "\n"


def render_bundle(
    *,
    fixture_id: str,
    attempt_id: str,
    control_mode: str,
    study_root: str,
    expected_study_commit: str,
    gpu_product: str,
    output_root: Path,
) -> dict[str, Any]:
    cfg = FIXTURE_CONFIG[fixture_id]
    gpu_sched = gpu_scheduling.resolve_model_blind_scheduling(
        {"gpu_product": gpu_product, "expected_gpu_name": gpu_scheduling.display_name_for_product(gpu_product)}
    )
    expected_gpu_name = gpu_sched["expected_gpu_name"]
    plan_path = f"{study_root}/artifacts/online_correction_v4/setup/{fixture_id}_g3_plan.candidate.json"
    if fixture_id == "vertical":
        plan_path = f"{study_root}/artifacts/online_correction_v4/setup/vertical_g3_plan.candidate.json"
    reset_path = f"{study_root}/artifacts/online_correction_v4/setup/{fixture_id}_reset_registry.candidate.json"
    campaign_path = f"{study_root}/artifacts/online_correction_v4/setup/c7_confirmatory/campaign.frozen.json"
    runner_path = f"{study_root}/tools/{cfg['runner_source'].name}"
    g2_path = f"{study_root}/{cfg['g2_aggregate_source'].relative_to(ROOT)}"
    qual_parent = f"/data/users/ali/vla_wam/raw/v4/qualification/{cfg['output_qual_dir']}"
    job_output = f"{qual_parent}/{attempt_id}"
    lane_id = "ngpc-st000"
    spec_sha = hashlib.sha256(
        json.dumps(
            {
                "attempt_id": attempt_id,
                "control_mode": control_mode,
                "fixture_id": fixture_id,
                "study_commit": expected_study_commit,
                "gpu_product": gpu_product,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    stem = f"v4-{cfg['k8s_name_token']}-{attempt_id}-{spec_sha[:10]}"
    bundle_root = output_root / f"dispatch_{attempt_id}"
    if bundle_root.exists():
        raise FileExistsError(f"refusing to overwrite bundle: {bundle_root}")
    bundle_root.mkdir(parents=True)

    runtime = {
        "ffmpeg_bin": "/data/users/ali/vla_wam/envs/lingbot-va-b200/bin/ffmpeg",
        "ld_library_path": (
            "/data/users/ali/vla_wam/envs/robolab-native-libs-ubuntu2204/usr/lib/x86_64-linux-gnu:"
            "/data/users/ali/glvnd/lib:/data/users/ali/vla_wam/envs/fastwam-native-libs/lib:/usr/lib/x86_64-linux-gnu"
        ),
        "python_bin": "/data/users/ali/vla_wam/envs/robolab-v2-isaac50/bin/python",
        "pythonpath": (
            f"{study_root}:{ROBOLAB_ROOT}:/data/users/ali/RoboTwin/envs/curobo/src:/opt/v4-lane/scripts"
        ),
        "vk_icd_filenames": "/etc/vulkan/icd.d/nvidia_icd.json",
    }
    labels = {
        "app.kubernetes.io/name": cfg["k8s_name_token"],
        "app.kubernetes.io/part-of": "vla-wam-v4",
        "v4-gate": f"{fixture_id}-natural-grasp-pc",
        "v4-fixture-id": fixture_id,
        "v4-attempt-id": attempt_id,
        "v4-config-sha": spec_sha[:16],
        "v4-scale": "0.5",
        "v4-lane-id": lane_id,
        "v4-lane-role": "simulator",
        "v4-job-index": "000",
        "v4-scripted-mode": "stationary",
        "v4-environment-seed": str(cfg["environment_seed"]),
    }
    bindings = [
        _binding(cfg["runner_source"], runner_path),
        _binding(ROOT / "tools/v4_natural_grasp_live_control_runner.py", f"{study_root}/tools/v4_natural_grasp_live_control_runner.py"),
        _binding(ROOT / "tools/run_v4_horizontal_g3_path_seed.py", f"{study_root}/tools/run_v4_horizontal_g3_path_seed.py"),
        _binding(ROOT / "tools/run_v4_horizontal_g3_scripted_seed.py", f"{study_root}/tools/run_v4_horizontal_g3_scripted_seed.py"),
        _binding(ROOT / "experiments/online_correction_v4/detectors.py", f"{study_root}/experiments/online_correction_v4/detectors.py"),
        _binding(ROOT / "experiments/online_correction_v4/droid_robolab.py", f"{study_root}/experiments/online_correction_v4/droid_robolab.py"),
        _binding(ROOT / "experiments/online_correction_v4/droid_g3_scripted.py", f"{study_root}/experiments/online_correction_v4/droid_g3_scripted.py"),
        _binding(CAMPAIGN_SOURCE, campaign_path),
        _binding(cfg["plan_source"], plan_path),
        _binding(cfg["reset_registry_source"], reset_path),
        _binding(cfg["g2_aggregate_source"], g2_path),
        _binding(ROOT / "deploy/k8s/v4_lane_bundle/scripts/lane_entrypoint.py", "/opt/v4-lane/scripts/lane_entrypoint.py"),
        _binding(ROOT / "deploy/k8s/v4_lane_bundle/scripts/startup_preflight.py", "/opt/v4-lane/scripts/startup_preflight.py"),
        _binding(ROOT / "deploy/k8s/v4_lane_bundle/scripts/isaac_render_probe.py", "/opt/v4-lane/scripts/isaac_render_probe.py"),
    ]
    argv = [
        runtime["python_bin"],
        runner_path,
        "--study-root",
        study_root,
        "--robolab-root",
        ROBOLAB_ROOT,
        "--campaign",
        campaign_path,
        "--campaign-sha256",
        CAMPAIGN_SHA256,
        "--plan",
        plan_path,
        "--plan-sha256",
        cfg["plan_sha256"],
        "--reset-registry",
        reset_path,
        "--reset-registry-sha256",
        cfg["reset_registry_sha256"],
        "--environment-seed",
        str(cfg["environment_seed"]),
        "--scale",
        "0.5",
        "--goal",
        cfg["goal"],
        "--expected-study-commit",
        expected_study_commit,
        "--expected-robolab-commit",
        EXPECTED_ROBOLAB_COMMIT,
        "--expected-driver-version",
        EXPECTED_DRIVER,
        "--native-control-dt-s",
        str(NATIVE_CONTROL_DT_S),
        "--attempt-id",
        attempt_id,
        "--control-mode",
        control_mode,
        "--output-dir",
        job_output,
    ]
    launch = {
        "schema_version": "vla-wam-v4-k8s-lane-launch-v1",
        "role": "simulator",
        "execution_scope": "model_blind_natural_grasp_positive_control",
        "experiment_argv": argv,
        "file_bindings": bindings,
        "gpu_product": gpu_product,
        "expected_gpu_name": expected_gpu_name,
        "allowed_gpu_names": gpu_sched["allowed_gpu_names"],
        "expected_driver_version": EXPECTED_DRIVER,
        "checkpoint_path": plan_path,
        "checkpoint_sha256": cfg["plan_sha256"],
        "checkpoint_semantics": "model_blind_g3_plan_candidate",
        "authorization_status": "authorized_by_passing_path_scale_receipt",
        "path_scale_receipt_path": g2_path,
        "path_scale_receipt_sha256": cfg["g2_aggregate_sha256"],
        "nvidia_smi_bin": "/usr/bin/nvidia-smi",
        "python_imports": ["json", "torch", "curobo", "experiments.online_correction_v4.detectors"],
        "vulkan_contract": "isaac_app_launcher_rtx_frame_under_bound_vk_icd",
        "render_probe_argv": [
            runtime["python_bin"],
            "/opt/v4-lane/scripts/isaac_render_probe.py",
            "--output",
            "{rendered_frame}",
            "--num-envs",
            "1",
            "--headless",
            "--rendering_mode",
            "balanced",
            "--device",
            "cuda:0",
            "--kit_args=--/rtx/verifyDriverVersion/enabled=false",
        ],
        "render_probe_timeout_seconds": 300,
        "policy_wait": None,
    }
    launch_json = lane.canonical_json(launch)
    launch_sha = hashlib.sha256(launch_json.encode("utf-8")).hexdigest()
    config_name = f"{stem}-job-config"
    job_name = f"{stem}-job"
    scripts_name = f"{stem}-scripts"
    scripts = lane.load_runtime_scripts(ROOT / "deploy/k8s/v4_lane_bundle/scripts")
    files = {
        "configmap.yaml": _render_launch_configmap(
            name=config_name,
            namespace=NAMESPACE,
            labels=labels,
            launch_json=launch_json,
            image_digest=f"sha256:{IMAGE_SHA256}",
        ),
        "scripts-configmap.yaml": lane.render_scripts_configmap(
            name=scripts_name,
            namespace=NAMESPACE,
            common_labels=labels,
            scripts=scripts,
        ),
        "job.yaml": lane.render_job(
            role="simulator",
            name=job_name,
            namespace=NAMESPACE,
            common_labels=labels,
            configmap=config_name,
            scripts_configmap=scripts_name,
            image=IMAGE,
            image_digest=f"sha256:{IMAGE_SHA256}",
            gpu_product_value=gpu_product,
            gpu_product_allowlist=None,
            lane=lane_id,
            attempt=attempt_id,
            output_parent=qual_parent,
            launch_sha=launch_sha,
            runtime=runtime,
            policy_port=1,
            pvc=PVC,
            image_pull_secret="artifactory-ci-pull-secret",
            entrypoint="/opt/v4-lane/scripts/lane_entrypoint.py",
            prestop_wait_seconds=120,
            kube_context=KUBE_CONTEXT,
        ),
    }
    for name, content in files.items():
        (bundle_root / name).write_text(content, encoding="utf-8")
    receipt = {
        "schema_version": "v4-natural-grasp-live-control-dispatch-v1",
        "attempt_id": attempt_id,
        "bundle_path": str(bundle_root.relative_to(ROOT)),
        "control_mode": control_mode,
        "fixture_id": fixture_id,
        "gpu_product": gpu_product,
        "job_name": job_name,
        "output_dir": job_output,
        "study_commit": expected_study_commit,
    }
    (bundle_root / "dispatch_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=sorted(FIXTURE_CONFIG), required=True)
    parser.add_argument(
        "--control-mode",
        choices=("scripted_grasp", "hold_only"),
        required=True,
    )
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--gpu-product", default="NVIDIA-A40")
    parser.add_argument("--study-root", default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--expected-study-commit", default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/online_correction_v4/qualification",
    )
    parser.add_argument("--skip-checkout-isolation", action="store_true")
    parser.add_argument("--skip-admission-gate", action="store_true")
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args(argv)

    expected_commit = (args.expected_study_commit or _git_head()).lower()
    study_root = args.study_root
    if not args.skip_checkout_isolation:
        spec_stub = {
            "attempt_id": args.attempt_id,
            "study_root": study_root,
            "expected_study_commit": expected_commit,
            "kube_context": KUBE_CONTEXT,
            "namespace": NAMESPACE,
        }
        report = checkout_isolation.ensure_isolated_study_root_for_dispatch(
            spec_stub,
            kube_context=KUBE_CONTEXT,
            namespace=NAMESPACE,
            publisher_pod=PUBLISHER_POD,
            apply=True,
        )
        legacy = str(report.get("legacy_study_root") or study_root)
        isolated = str(report.get("study_root") or study_root)
        if isolated and legacy and isolated != legacy:
            study_root = isolated
        elif isolated:
            study_root = isolated
        print(json.dumps({"checkout_isolation": report}, indent=2, sort_keys=True))

    if args.create and not args.skip_admission_gate:
        jobs = wave_admission.fetch_v4_jobs(kube_context=KUBE_CONTEXT, namespace=NAMESPACE)
        wave_admission.require_dispatch_admission(
            attempt_id=args.attempt_id,
            attempt_summary=wave_admission.summarize_attempts(jobs),
        )

    receipt = render_bundle(
        fixture_id=args.fixture,
        attempt_id=args.attempt_id,
        control_mode=args.control_mode,
        study_root=study_root,
        expected_study_commit=expected_commit,
        gpu_product=args.gpu_product,
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not args.create:
        return 0

    qual_dirs = sorted(
        {
            FIXTURE_CONFIG[args.fixture]["output_qual_dir"],
            FIXTURE_CONFIG["vertical"]["output_qual_dir"],
            FIXTURE_CONFIG["containment"]["output_qual_dir"],
        }
    )
    mkdir_cmd = " && ".join(
        f"mkdir -p /data/users/ali/vla_wam/raw/v4/qualification/{name}"
        for name in qual_dirs
    )
    subprocess.run(
        [
            "kubectl",
            "exec",
            "-n",
            NAMESPACE,
            PUBLISHER_POD,
            "--",
            "bash",
            "-c",
            mkdir_cmd,
        ],
        check=True,
    )

    bundle_root = ROOT / receipt["bundle_path"]
    for name in ("configmap.yaml", "scripts-configmap.yaml", "job.yaml"):
        subprocess.run(
            [
                "kubectl",
                "--context",
                KUBE_CONTEXT,
                "create",
                "-f",
                str(bundle_root / name),
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
