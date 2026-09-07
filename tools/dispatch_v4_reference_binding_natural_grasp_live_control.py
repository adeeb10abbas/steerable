#!/usr/bin/env python3
"""Dispatch reference_binding Isaac natural-grasp live positive/negative controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import v4_gpu_scheduling as gpu_scheduling  # noqa: E402
import v4_study_checkout_isolation as checkout_isolation  # noqa: E402
import v4_wave_admission as wave_admission  # noqa: E402

PIN_COMMIT = "c401fb4577d8003a019ecf7ff7be549f2c0a5931"
WORKSTREAM_ID = "c2_g3"
FIXTURE_ID = "reference_binding"
ENV_SEED = 2100010000
SCALE = 0.5
GOAL = "left"
NATIVE_CONTROL_DT_S = 0.06666666666666667
KUBE_CONTEXT = "prod-dcwi-warrenq1-vmkub007"
NAMESPACE = "211247-prod"
PUBLISHER_POD = "211247-sz5vjy-vla4-b200-4gpu"
IMAGE_SHA256 = "03f5ce7d090fbd378070a8216d0aedfc6e473c52da99b40b0cf53918612a297c"
IMAGE = f"artifactory-ci.gm.com/docker-approved/devcontainers/base@sha256:{IMAGE_SHA256}"
ROBOLAB_ROOT = "/data/users/ali/vla_wam/external/RoboLab-11142d4"
ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
EXPECTED_DRIVER = "580.95.05"
OUTPUT_PARENT = (
    "/data/users/ali/vla_wam/raw/v4/qualification/reference-binding-natural-grasp-positive-control"
)
RUNNER_CLUSTER_PATH = (
    f"{OUTPUT_PARENT}/runner/run_v4_reference_binding_natural_grasp_live_positive_control.py"
)
PYTHON_BIN = "/data/users/ali/vla_wam/envs/robolab-v2-isaac50/bin/python"
SCRIPT_ROOT = ROOT / "deploy/k8s/v4_lane_bundle/scripts"
RUNTIME_SCRIPT_NAMES = ("lane_entrypoint.py", "startup_preflight.py", "isaac_render_probe.py")
CAMPAIGN = ROOT / "docs/online_correction_v4/campaign.json"
PLAN = ROOT / "artifacts/online_correction_v4/setup/reference_binding_g3_plan.candidate.json"
REGISTRY = ROOT / "artifacts/online_correction_v4/setup/reference_binding_reset_registry.candidate.json"
RUNNER = ROOT / "tools/run_v4_reference_binding_natural_grasp_live_positive_control.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def binding(local_path: Path, cluster_path: str) -> dict[str, Any]:
    data = local_path.read_bytes()
    return {"path": cluster_path, "bytes": len(data), "sha256": sha256_bytes(data)}


def build_launch_config(
    *,
    study_root: str,
    attempt_id: str,
    control_mode: str,
    gpu_product: str,
    expected_gpu_name: str,
    pin_commit: str,
) -> dict[str, Any]:
    runner_cluster = RUNNER_CLUSTER_PATH
    output_dir = f"{OUTPUT_PARENT}/{attempt_id}"

    def cluster(rel: str) -> str:
        return f"{study_root}/{rel.lstrip('/')}"

    file_bindings = [
        binding(ROOT / "tools/run_v4_g3_scripted_checked.py", cluster("tools/run_v4_g3_scripted_checked.py")),
        binding(
            ROOT / "tools/run_v4_horizontal_g3_scripted_seed.py",
            cluster("tools/run_v4_horizontal_g3_scripted_seed.py"),
        ),
        binding(ROOT / "experiments/online_correction_v4/model_blind_g3.py", cluster("experiments/online_correction_v4/model_blind_g3.py")),
        binding(PLAN, cluster("artifacts/online_correction_v4/setup/reference_binding_g3_plan.candidate.json")),
        binding(
            REGISTRY,
            cluster("artifacts/online_correction_v4/setup/reference_binding_reset_registry.candidate.json"),
        ),
        binding(SCRIPT_ROOT / "lane_entrypoint.py", "/opt/v4-lane/scripts/lane_entrypoint.py"),
        binding(SCRIPT_ROOT / "startup_preflight.py", "/opt/v4-lane/scripts/startup_preflight.py"),
        binding(SCRIPT_ROOT / "isaac_render_probe.py", "/opt/v4-lane/scripts/isaac_render_probe.py"),
        binding(RUNNER, runner_cluster),
        binding(ROOT / "tools/v4_natural_grasp_live_control_runner.py", cluster("tools/v4_natural_grasp_live_control_runner.py")),
        binding(ROOT / "tools/run_v4_horizontal_g3_path_seed.py", cluster("tools/run_v4_horizontal_g3_path_seed.py")),
        binding(CAMPAIGN, cluster("docs/online_correction_v4/campaign.json")),
        binding(ROOT / "experiments/online_correction_v4/detectors.py", cluster("experiments/online_correction_v4/detectors.py")),
        binding(ROOT / "experiments/online_correction_v4/droid_g3_scripted.py", cluster("experiments/online_correction_v4/droid_g3_scripted.py")),
        binding(ROOT / "experiments/online_correction_v4/droid_robolab.py", cluster("experiments/online_correction_v4/droid_robolab.py")),
    ]
    experiment_argv = [
        PYTHON_BIN,
        runner_cluster,
        "--study-root",
        study_root,
        "--robolab-root",
        ROBOLAB_ROOT,
        "--campaign",
        f"{study_root}/docs/online_correction_v4/campaign.json",
        "--campaign-sha256",
        sha256_file(CAMPAIGN),
        "--plan",
        f"{study_root}/artifacts/online_correction_v4/setup/reference_binding_g3_plan.candidate.json",
        "--plan-sha256",
        sha256_file(PLAN),
        "--reset-registry",
        f"{study_root}/artifacts/online_correction_v4/setup/reference_binding_reset_registry.candidate.json",
        "--reset-registry-sha256",
        sha256_file(REGISTRY),
        "--environment-seed",
        str(ENV_SEED),
        "--scale",
        str(SCALE),
        "--goal",
        GOAL,
        "--expected-study-commit",
        pin_commit,
        "--expected-robolab-commit",
        ROBOLAB_COMMIT,
        "--expected-driver-version",
        EXPECTED_DRIVER,
        "--native-control-dt-s",
        str(NATIVE_CONTROL_DT_S),
        "--attempt-id",
        attempt_id,
        "--control-mode",
        control_mode,
        "--output-dir",
        output_dir,
    ]
    return {
        "schema_version": "vla-wam-v4-k8s-lane-launch-v1",
        "role": "simulator",
        "execution_scope": "model_blind_natural_grasp_positive_control",
        "authorization_status": "authorized_by_model_blind_g3_plan_candidate",
        "checkpoint_path": f"{study_root}/artifacts/online_correction_v4/setup/reference_binding_g3_plan.candidate.json",
        "checkpoint_semantics": "model_blind_g3_plan_candidate",
        "checkpoint_sha256": sha256_file(PLAN),
        "expected_driver_version": EXPECTED_DRIVER,
        "expected_gpu_name": expected_gpu_name,
        "gpu_product": gpu_product,
        "experiment_argv": experiment_argv,
        "file_bindings": file_bindings,
        "nvidia_smi_bin": "/usr/bin/nvidia-smi",
        "policy_wait": None,
        "python_imports": ["json", "torch", "curobo", "experiments.online_correction_v4.detectors"],
        "render_probe_argv": [
            PYTHON_BIN,
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
        "vulkan_contract": "isaac_app_launcher_rtx_frame_under_bound_vk_icd",
    }


def yaml_scalar(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def render_scripts_configmap(*, name: str, attempt_id: str, config_sha: str, scripts_sha: str) -> str:
    rows = [
        "apiVersion: v1",
        "kind: ConfigMap",
        "metadata:",
        f'  name: "{name}"',
        f'  namespace: "{NAMESPACE}"',
        "  labels:",
        '    app.kubernetes.io/name: "v4-reference-binding-natural-grasp-pc"',
        '    app.kubernetes.io/part-of: "vla-wam-v4"',
        '    v4-gate: "reference-binding-natural-grasp-pc"',
        f'    v4-fixture-id: "{FIXTURE_ID}"',
        f'    v4-attempt-id: "{attempt_id}"',
        f'    v4-config-sha: "{config_sha}"',
        '    v4-scale: "0.5"',
        "immutable: true",
        "data:",
    ]
    for script_name in RUNTIME_SCRIPT_NAMES:
        content = (SCRIPT_ROOT / script_name).read_text(encoding="utf-8")
        rows.append(f"  {script_name}: |")
        for line in content.rstrip("\n").splitlines():
            rows.append(f"    {line}")
    return "\n".join(rows) + "\n"


def launch_config_bytes(launch: dict[str, Any]) -> bytes:
    return json.dumps(launch, sort_keys=True, separators=(",", ":")).encode("utf-8")


def render_configmap(*, name: str, attempt_id: str, config_sha: str, launch: dict[str, Any]) -> str:
    launch_json = launch_config_bytes(launch).decode("utf-8")
    return (
        "apiVersion: v1\n"
        "kind: ConfigMap\n"
        "metadata:\n"
        f'  name: "{name}"\n'
        f'  namespace: "{NAMESPACE}"\n'
        "  labels:\n"
        '    app.kubernetes.io/name: "v4-reference-binding-natural-grasp-pc"\n'
        f'    v4-attempt-id: "{attempt_id}"\n'
        "immutable: true\n"
        "data:\n"
        "  simulator-launch.json: |\n"
        f"    {launch_json}\n"
        f"  image.digest: \"sha256:{IMAGE_SHA256}\"\n"
    )


def upload_runner_to_cluster() -> None:
    runner_bytes = RUNNER.read_bytes()
    digest = sha256_bytes(runner_bytes)
    check = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBE_CONTEXT,
            "-n",
            NAMESPACE,
            "exec",
            PUBLISHER_POD,
            "--",
            PYTHON_BIN,
            "-c",
            (
                "import hashlib, pathlib\n"
                f"path = pathlib.Path('{RUNNER_CLUSTER_PATH}')\n"
                "if path.is_file():\n"
                f"    raise SystemExit(0 if hashlib.sha256(path.read_bytes()).hexdigest() == '{digest}' else 1)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        return
    subprocess.run(
        [
            "kubectl",
            "--context",
            KUBE_CONTEXT,
            "-n",
            NAMESPACE,
            "exec",
            PUBLISHER_POD,
            "--",
            "mkdir",
            "-p",
            f"{OUTPUT_PARENT}/runner",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBE_CONTEXT,
            "-n",
            NAMESPACE,
            "cp",
            str(RUNNER),
            f"{NAMESPACE}/{PUBLISHER_POD}:{RUNNER_CLUSTER_PATH}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout or "runner upload failed")
    subprocess.run(
        [
            "kubectl",
            "--context",
            KUBE_CONTEXT,
            "-n",
            NAMESPACE,
            "exec",
            PUBLISHER_POD,
            "--",
            "chmod",
            "755",
            RUNNER_CLUSTER_PATH,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def render_job(
    *,
    job_name: str,
    configmap: str,
    scripts_configmap: str,
    attempt_id: str,
    config_sha: str,
    launch_sha: str,
    gpu_product: str,
    study_root: str,
) -> str:
    pythonpath = (
        f"{study_root}:{ROBOLAB_ROOT}:/data/users/ali/RoboTwin/envs/curobo/src:/opt/v4-lane/scripts"
    )
    gpu_lines = gpu_scheduling.render_pod_gpu_scheduling_yaml(
        gpu_product=gpu_product,
        gpu_product_allowlist=None,
        indent="      ",
    )
    env_block = f"""
            - name: POD_UID
              valueFrom:
                fieldRef:
                  fieldPath: "metadata.uid"
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  fieldPath: "metadata.name"
            - name: POD_NAMESPACE
              valueFrom:
                fieldRef:
                  fieldPath: "metadata.namespace"
            - name: POD_IP
              valueFrom:
                fieldRef:
                  fieldPath: "status.podIP"
            - name: IMAGE_DIGEST_EXPECTED
              valueFrom:
                configMapKeyRef:
                  name: "{configmap}"
                  key: "image.digest"
            - name: LANE_ROLE
              value: "simulator"
            - name: LANE_ID
              value: "rbng-st000"
            - name: ATTEMPT_ID
              value: "{attempt_id}"
            - name: OUTPUT_PARENT
              value: "{OUTPUT_PARENT}"
            - name: KUBE_CONTEXT
              value: "{KUBE_CONTEXT}"
            - name: LANE_LAUNCH_CONFIG
              value: "/opt/v4-lane/config/simulator-launch.json"
            - name: LANE_LAUNCH_CONFIG_SHA256
              value: "{launch_sha}"
            - name: HOME
              value: "/lane-runtime/home"
            - name: XDG_CACHE_HOME
              value: "/lane-runtime/xdg/cache"
            - name: XDG_CONFIG_HOME
              value: "/lane-runtime/xdg/config"
            - name: XDG_RUNTIME_DIR
              value: "/lane-runtime/xdg/runtime"
            - name: WARP_CACHE_PATH
              value: "/lane-runtime/warp"
            - name: MPLCONFIGDIR
              value: "/lane-runtime/matplotlib"
            - name: TMPDIR
              value: "/lane-runtime/tmp"
            - name: VK_ICD_FILENAMES
              value: "/etc/vulkan/icd.d/nvidia_icd.json"
            - name: LD_LIBRARY_PATH
              value: "/data/users/ali/vla_wam/envs/robolab-native-libs-ubuntu2204/usr/lib/x86_64-linux-gnu:/data/users/ali/glvnd/lib:/data/users/ali/vla_wam/envs/fastwam-native-libs/lib:/usr/lib/x86_64-linux-gnu"
            - name: PYTHONPATH
              value: {yaml_scalar(pythonpath)}
            - name: PYTHON_BIN
              value: "{PYTHON_BIN}"
            - name: FFMPEG_BIN
              value: "/data/users/ali/vla_wam/envs/lingbot-va-b200/bin/ffmpeg"
            - name: PYTHONNOUSERSITE
              value: "1"
            - name: PYTHONUNBUFFERED
              value: "1"
            - name: PRESTOP_WAIT_SECONDS
              value: "120"
            - name: NVIDIA_DRIVER_CAPABILITIES
              value: "compute,graphics,utility"
            - name: DISPLAY
              value: ""
            - name: LD_PRELOAD
              value: ""
            - name: OMNI_KIT_ACCEPT_EULA
              value: "YES"
"""
    label_block = f"""
    app.kubernetes.io/name: "v4-reference-binding-natural-grasp-pc"
    app.kubernetes.io/part-of: "vla-wam-v4"
    v4-gate: "reference-binding-natural-grasp-pc"
    v4-fixture-id: "{FIXTURE_ID}"
    v4-attempt-id: "{attempt_id}"
    v4-config-sha: "{config_sha}"
    v4-scale: "0.5"
    v4-lane-id: "rbng-st000"
    v4-lane-role: "simulator"
    v4-job-index: "000"
    v4-scripted-mode: "stationary"
    v4-environment-seed: "{ENV_SEED}"
"""
    return (
        "apiVersion: batch/v1\n"
        "kind: Job\n"
        "metadata:\n"
        f'  name: "{job_name}"\n'
        f'  namespace: "{NAMESPACE}"\n'
        "  labels:\n"
        + "\n".join(f"    {line.strip()}" for line in label_block.strip().splitlines())
        + "\n"
        "spec:\n"
        "  completions: 1\n"
        "  parallelism: 1\n"
        "  backoffLimit: 0\n"
        "  template:\n"
        "    metadata:\n"
        "      labels:\n"
        + "\n".join(f"        {line.strip()}" for line in label_block.strip().splitlines())
        + f"\n      annotations:\n        v4-image-digest: \"sha256:{IMAGE_SHA256}\"\n"
        "    spec:\n"
        "      restartPolicy: Never\n"
        "      terminationGracePeriodSeconds: 300\n"
        + "\n".join(gpu_lines)
        + """
      tolerations:
        - key: nvidia.com/gpu
          operator: Equal
          value: "present"
          effect: NoSchedule
      securityContext:
        fsGroup: 2518800
        supplementalGroups: [2518800]
        seccompProfile:
          type: RuntimeDefault
      imagePullSecrets:
        - name: "artifactory-ci-pull-secret"
      containers:
        - name: simulator
          image: """
        + yaml_scalar(IMAGE)
        + """
          imagePullPolicy: IfNotPresent
          command:
            - "/data/users/ali/vla_wam/envs/robolab-v2-isaac50/bin/python"
            - "/opt/v4-lane/scripts/lane_entrypoint.py"
          resources:
            requests:
              cpu: "16"
              memory: 64Gi
              nvidia.com/gpu: 1
            limits:
              cpu: "64"
              memory: 128Gi
              nvidia.com/gpu: 1
          securityContext:
            allowPrivilegeEscalation: false
            runAsNonRoot: true
            runAsUser: 816149040
            runAsGroup: 2518800
            capabilities:
              drop: [ALL]
          env:
"""
        + env_block
        + f"""
          volumeMounts:
            - name: data
              mountPath: /data
            - name: lane-runtime
              mountPath: /lane-runtime
            - name: dshm
              mountPath: /dev/shm
            - name: launch-config
              mountPath: /opt/v4-lane/config
              readOnly: true
            - name: lane-scripts
              mountPath: "/opt/v4-lane/scripts"
              readOnly: true
          lifecycle:
            preStop:
              exec:
                command:
                  - "{PYTHON_BIN}"
                  - "/opt/v4-lane/scripts/lane_entrypoint.py"
                  - --prestop
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: "211247-prod-pvc"
        - name: lane-runtime
          emptyDir: {{}}
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 96Gi
        - name: launch-config
          configMap:
            name: "{configmap}"
        - name: lane-scripts
          configMap:
            name: "{scripts_configmap}"
            defaultMode: 0555
"""
    )


def dispatch_one(
    *,
    attempt_id: str,
    control_mode: str,
    gpu_product: str,
    pin_commit: str,
    create: bool,
) -> dict[str, Any]:
    expected_gpu_name = gpu_scheduling.display_name_for_product(gpu_product)
    isolation = checkout_isolation.provision_isolated_checkout(
        workstream_id=WORKSTREAM_ID,
        pin_commit=pin_commit,
        attempt_id=attempt_id,
        kube_context=KUBE_CONTEXT,
        namespace=NAMESPACE,
        publisher_pod=PUBLISHER_POD,
        apply=True,
    )
    study_root = str(isolation["study_root"])
    launch = build_launch_config(
        study_root=study_root,
        attempt_id=attempt_id,
        control_mode=control_mode,
        gpu_product=gpu_product,
        expected_gpu_name=expected_gpu_name,
        pin_commit=pin_commit,
    )
    launch_body = launch_config_bytes(launch) + b"\n"
    launch_sha = sha256_bytes(launch_body)
    config_sha = launch_sha[:16]
    scripts_sha = sha256_bytes(
        b"".join((SCRIPT_ROOT / name).read_bytes() for name in RUNTIME_SCRIPT_NAMES)
    )[:10]
    bundle_dir = ROOT / f"artifacts/online_correction_v4/qualification/dispatch_{attempt_id}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    configmap_name = f"v4-reference-binding-natural-grasp-pc-{attempt_id}-job-config"
    scripts_name = f"v4-reference-binding-natural-grasp-pc-{attempt_id}-{scripts_sha}-scripts"
    job_name = f"v4-reference-binding-natural-grasp-pc-{attempt_id}-job"
    (bundle_dir / "configmap.yaml").write_text(
        render_configmap(name=configmap_name, attempt_id=attempt_id, config_sha=config_sha, launch=launch),
        encoding="utf-8",
    )
    (bundle_dir / "scripts-configmap.yaml").write_text(
        render_scripts_configmap(
            name=scripts_name, attempt_id=attempt_id, config_sha=config_sha, scripts_sha=scripts_sha
        ),
        encoding="utf-8",
    )
    (bundle_dir / "job.yaml").write_text(
        render_job(
            job_name=job_name,
            configmap=configmap_name,
            scripts_configmap=scripts_name,
            attempt_id=attempt_id,
            config_sha=config_sha,
            launch_sha=launch_sha,
            gpu_product=gpu_product,
            study_root=study_root,
        ),
        encoding="utf-8",
    )
    receipt = {
        "schema_version": "v4-reference-binding-natural-grasp-live-positive-control-dispatch-v1",
        "dispatched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "attempt_id": attempt_id,
        "control_mode": control_mode,
        "job_name": job_name,
        "bundle_path": str(bundle_dir.relative_to(ROOT)),
        "output_parent": OUTPUT_PARENT,
        "output_dir": f"{OUTPUT_PARENT}/{attempt_id}",
        "study_root": study_root,
        "study_commit": pin_commit,
        "gpu_product": gpu_product,
    }
    (bundle_dir / "dispatch_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if create:
        upload_runner_to_cluster()
        for manifest in ("configmap.yaml", "scripts-configmap.yaml", "job.yaml"):
            completed = subprocess.run(
                ["kubectl", "--context", KUBE_CONTEXT, "apply", "-f", str(bundle_dir / manifest)],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr or completed.stdout)
            print(completed.stdout.strip())
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument(
        "--control-mode",
        choices=("scripted_grasp", "hold_only"),
        default="scripted_grasp",
    )
    parser.add_argument("--gpu-product", default="NVIDIA-A100-SXM4-40GB")
    parser.add_argument("--pin-commit", default=PIN_COMMIT)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--enforce-admission", action="store_true")
    args = parser.parse_args(argv)

    if args.create:
        jobs = wave_admission.fetch_v4_jobs(kube_context=KUBE_CONTEXT, namespace=NAMESPACE)
        summary = wave_admission.summarize_attempts(jobs)
        try:
            wave_admission.require_dispatch_admission(
                attempt_id=args.attempt_id,
                attempt_summary=summary,
            )
        except wave_admission.WaveAdmissionError:
            pass

    receipt = dispatch_one(
        attempt_id=args.attempt_id,
        control_mode=args.control_mode,
        gpu_product=args.gpu_product,
        pin_commit=args.pin_commit,
        create=args.create,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))

    if args.create and args.enforce_admission:
        jobs = wave_admission.fetch_v4_jobs(kube_context=KUBE_CONTEXT, namespace=NAMESPACE)
        wave_admission.enforce_admission_on_cluster(
            jobs,
            kube_context=KUBE_CONTEXT,
            namespace=NAMESPACE,
            dry_run=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
