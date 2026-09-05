from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import shutil
import textwrap

import pytest


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "tools/validate_v4_k8s_lane_bundle.py"
SPEC = importlib.util.spec_from_file_location("validate_v4_k8s_lane_bundle", SOURCE)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


IMAGE_DIGEST = "03f5ce7d090fbd378070a8216d0aedfc6e473c52da99b40b0cf53918612a297c"
CHECKPOINT_DIGEST = "a" * 64
CONFIG_DIGEST = "b" * 64


ENTRYPOINT = "/opt/v4-lane/scripts/lane_entrypoint.py"


def _job(role: str) -> str:
    gpu = "NVIDIA-A40" if role == "simulator" else "NVIDIA-A100-SXM4-80GB"
    body = textwrap.dedent(
        f"""
        apiVersion: batch/v1
        kind: Job
        metadata:
          name: v4-lane-{role}-lane00-attempt01
          labels:
            v4-lane-id: lane00
            v4-attempt-id: attempt01
            v4-config-sha: {CONFIG_DIGEST}
            v4-lane-role: {role}
        spec:
          completions: 1
          parallelism: 1
          backoffLimit: 0
          template:
            metadata:
              labels:
                v4-lane-id: lane00
                v4-attempt-id: attempt01
                v4-config-sha: {CONFIG_DIGEST}
                v4-lane-role: {role}
              annotations:
                v4-image-digest: sha256:{IMAGE_DIGEST}
            spec:
              restartPolicy: Never
              terminationGracePeriodSeconds: 300
              nodeSelector:
                node-role.kubernetes.io/worker-gpu: ""
                nvidia.com/gpu.product: {gpu}
              containers:
                - name: {role}
                  image: example.invalid/v4-lane@sha256:{IMAGE_DIGEST}
                  command:
                    - /usr/bin/python3
                    - {ENTRYPOINT}
                  resources:
                    requests:
                      nvidia.com/gpu: 1
                    limits:
                      nvidia.com/gpu: 1
                  env:
                    - name: LANE_ROLE
                      value: {role}
                    - name: POD_UID
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.uid
                    - name: POD_NAME
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.name
                    - name: POD_NAMESPACE
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.namespace
                    - name: POD_IP
                      valueFrom:
                        fieldRef:
                          fieldPath: status.podIP
                    - name: LANE_ID
                      value: lane00
                    - name: ATTEMPT_ID
                      value: attempt01
                    - name: KUBE_CONTEXT
                      value: test-context
                    - name: OUTPUT_PARENT
                      value: /data/users/ali/vla_wam/raw/v4_lane_jobs
                    - name: LANE_LAUNCH_CONFIG
                      value: /opt/v4-lane/config/{role}-launch.json
                    - name: LANE_LAUNCH_CONFIG_SHA256
                      value: deadbeef
                    - name: HOME
                      value: /lane-runtime/home
                    - name: XDG_CACHE_HOME
                      value: /lane-runtime/xdg/cache
                    - name: XDG_CONFIG_HOME
                      value: /lane-runtime/xdg/config
                    - name: XDG_RUNTIME_DIR
                      value: /lane-runtime/xdg/runtime
                    - name: WARP_CACHE_PATH
                      value: /lane-runtime/warp
                    - name: MPLCONFIGDIR
                      value: /lane-runtime/matplotlib
                    - name: TMPDIR
                      value: /lane-runtime/tmp
                    - name: VK_ICD_FILENAMES
                      value: /etc/vulkan/icd.d/nvidia_icd.json
                    - name: LD_LIBRARY_PATH
                      value: /data/native/lib:/usr/lib/x86_64-linux-gnu
                    - name: PYTHONPATH
                      value: /data/curobo/src:/opt/v4-lane
                    - name: FFMPEG_BIN
                      value: /usr/bin/ffmpeg
                    - name: PYTHON_BIN
                      value: /usr/bin/python3
                    - name: PYTHONNOUSERSITE
                      value: "1"
                    - name: PYTHONUNBUFFERED
                      value: "1"
                    - name: PRESTOP_WAIT_SECONDS
                      value: "120"
                    - name: NVIDIA_DRIVER_CAPABILITIES
                      value: compute,graphics,utility
                    - name: DISPLAY
                      value: ""
                    - name: LD_PRELOAD
                      value: ""
                    - name: IMAGE_DIGEST_EXPECTED
                      valueFrom:
                        configMapKeyRef:
                          name: v4-lane-launch
                          key: image.digest
                    # POLICY_PORT_ENV
                  volumeMounts:
                    - name: lane-runtime
                      mountPath: /lane-runtime
                    - name: launch-config
                      mountPath: /opt/v4-lane/config
                      readOnly: true
                    - name: lane-scripts
                      mountPath: /opt/v4-lane/scripts
                      readOnly: true
                  lifecycle:
                    preStop:
                      exec:
                        command: [/usr/bin/python3, {ENTRYPOINT}, --prestop]
                  # POLICY_READINESS
              volumes:
                - name: lane-runtime
                  emptyDir: {{}}
                - name: launch-config
                  configMap:
                    name: v4-lane-launch
                - name: lane-scripts
                  configMap:
                    name: v4-lane-scripts
                    defaultMode: 0555
        """
    ).lstrip()
    port_env = "" if role == "simulator" else '            - name: POLICY_PORT\n              value: "8100"'
    readiness = "" if role == "simulator" else textwrap.indent(
        textwrap.dedent(
            """
            readinessProbe:
              exec:
                command:
                  - /usr/bin/python3
                  - /opt/v4-lane/scripts/check_policy_ready.py
                  - --checkpoint-loaded
                  - --mode
                  - http_healthz
                  - --launch-config
                  - /opt/v4-lane/config/policy-launch.json
                  - --port
                  - "8100"
              periodSeconds: 2
            """
        ).lstrip().rstrip(),
        " " * 10,
    )
    return body.replace("            # POLICY_PORT_ENV", port_env).replace(
        "          # POLICY_READINESS", readiness
    )


def _binding(path: str, payload: bytes) -> dict[str, object]:
    return {"path": path, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _write_valid_bundle(root: Path) -> None:
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    script_sources = {
        "startup_preflight.py": textwrap.dedent(
            '''
            import os
            import signal
            import time

            # Concrete proof names retained in runtime-preflight.json:
            # torch.cuda kernel on cuda:0; vulkaninfo; rendered_frame.png;
            # import curobo; ffmpeg_encode encoded.mp4; ffmpeg_decode decoded.raw;
            # checkpoint_sha256; writable_parent; O_EXCL port_lock.
            # Runtime-startup fields: effective_environment, gpu_uuid,
            # image_digest, argv, checkpoint_sha256.
            def acquire_attempt_lock():
                os.open("attempt-lock", os.O_CREAT | os.O_EXCL)

            def reserve_policy_port(port):
                os.open("port_lock", os.O_CREAT | os.O_EXCL)
                return None, "port_lock"

            def run_preflight():
                required = ("POD_UID", "LANE_ID", "ATTEMPT_ID")
                attempt = f"{os.environ['POD_UID']}/lane-{os.environ['LANE_ID']}/attempt-{os.environ['ATTEMPT_ID']}"
                # simulator policy_wait: wait_for_policy readiness and policy_metadata runtime-startup.json
                assert required and attempt
            '''
        ).lstrip(),
        "lane_entrypoint.py": textwrap.dedent(
            '''
            import os
            import signal
            import time
            from startup_preflight import run_preflight

            IMAGE_DIGEST_EXPECTED = os.environ.get("IMAGE_DIGEST_EXPECTED")

            def pre_stop():
                wait = float(os.environ.get("PRESTOP_WAIT_SECONDS", "120"))
                os.kill(1, signal.SIGINT)
                deadline = time.monotonic() + wait
                while time.monotonic() < deadline:
                    try:
                        os.kill(1, 0)
                    except ProcessLookupError:
                        return

            def main():
                run_preflight()
                os.execvp("experiment", ["experiment"])

            if __name__ == "__main__":
                main()
            '''
        ).lstrip(),
        "check_policy_ready.py": textwrap.dedent(
            '''
            # checkpoint-loaded policy-ready marker must exist
            # launch_config sha256 verification against preflight evidence
            report.get("launch_config", {}).get("sha256")
            '''
        ).lstrip(),
        "isaac_render_probe.py": "# render probe stub\n",
    }
    script_payloads: dict[str, bytes] = {}
    for name, body in script_sources.items():
        payload = body.encode("utf-8")
        script_payloads[name] = payload
        (scripts / name).write_bytes(payload)

    scripts_cm_lines = [
        "apiVersion: v1",
        "kind: ConfigMap",
        "metadata:",
        "  name: v4-lane-scripts",
        "  labels:",
        f"    v4-lane-id: lane00",
        f"    v4-attempt-id: attempt01",
        f"    v4-config-sha: {CONFIG_DIGEST}",
        "immutable: true",
        "data:",
    ]
    for name in sorted(script_payloads):
        scripts_cm_lines.append(f"  {name}: |")
        for line in script_payloads[name].decode("utf-8").splitlines():
            scripts_cm_lines.append(f"    {line}")
    (root / "scripts-configmap.yaml").write_text("\n".join(scripts_cm_lines) + "\n", encoding="utf-8")

    checkpoint_binding = _binding("/data/checkpoint", b"x")
    policy_bindings = [
        _binding(ENTRYPOINT, script_payloads["lane_entrypoint.py"]),
        _binding("/opt/v4-lane/scripts/startup_preflight.py", script_payloads["startup_preflight.py"]),
        _binding("/opt/v4-lane/scripts/check_policy_ready.py", script_payloads["check_policy_ready.py"]),
        {
            "path": "/data/openpi/src/openpi/serving/websocket_policy_server.py",
            "bytes": 3051,
            "sha256": "1370d345e6c3c5b8f15573050e485e60a5b423d1df33e24b237805e6b442b026",
        },
        checkpoint_binding,
    ]
    simulator_bindings = [
        _binding(ENTRYPOINT, script_payloads["lane_entrypoint.py"]),
        _binding("/opt/v4-lane/scripts/startup_preflight.py", script_payloads["startup_preflight.py"]),
        _binding("/opt/v4-lane/scripts/isaac_render_probe.py", script_payloads["isaac_render_probe.py"]),
        checkpoint_binding,
    ]
    import json

    policy_launch = {
        "role": "policy",
        "experiment_argv": [
            "/usr/bin/python3",
            "/data/bin/policy-server",
            "--openpi-root",
            "/data/openpi",
            "--checkpoint",
            "/data/checkpoint",
        ],
        "checkpoint_path": "/data/checkpoint",
        "gpu_product": "NVIDIA-A100-SXM4-80GB",
        "expected_gpu_name": "NVIDIA A100-SXM4-80GB",
        "expected_driver_version": "580.95.05",
        "policy_port": 8100,
        "readiness_contract": "http_healthz_after_checkpoint_load",
        "launch_scope": "infrastructure_qualification_only_no_scientific_behavior",
        "file_bindings": policy_bindings,
    }
    simulator_launch = {
        "role": "simulator",
        "experiment_argv": ["/usr/bin/true"],
        "checkpoint_path": "/data/checkpoint",
        "gpu_product": "NVIDIA-A40",
        "expected_gpu_name": "NVIDIA A40",
        "expected_driver_version": "580.95.05",
        "launch_scope": "infrastructure_qualification_only_no_scientific_behavior",
        "vulkan_contract": "isaac_app_launcher_rtx_frame_under_bound_vk_icd",
        "render_probe_argv": ["/data/bin/render-probe", "{rendered_frame}"],
        "policy_wait": {
            "mode": "http_healthz",
            "host": "v4-lane-policy-lane00-attempt01",
            "port": 8100,
            "service_identity": {
                "v4-lane-id": "lane00",
                "v4-attempt-id": "attempt01",
                "v4-config-sha": CONFIG_DIGEST,
                "v4-lane-role": "policy",
                "service_name": "v4-lane-policy-lane00-attempt01",
            },
        },
        "file_bindings": simulator_bindings,
    }
    policy_json = json.dumps(policy_launch, indent=2, sort_keys=True) + "\n"
    simulator_json = json.dumps(simulator_launch, indent=2, sort_keys=True) + "\n"
    configmap_lines = [
        "apiVersion: v1",
        "kind: ConfigMap",
        "metadata:",
        "  name: v4-lane-launch",
        "  labels:",
        "    v4-lane-id: lane00",
        "    v4-attempt-id: attempt01",
        f"    v4-config-sha: {CONFIG_DIGEST}",
        "immutable: true",
        "data:",
        "  policy-launch.json: |",
    ]
    configmap_lines.extend(f"    {line}" for line in policy_json.splitlines())
    configmap_lines.append("  simulator-launch.json: |")
    configmap_lines.extend(f"    {line}" for line in simulator_json.splitlines())
    configmap_lines.extend(
        [
            '  kube.context: "test-context"',
            '  policy.argv.json: \'["policy-server", "--checkpoint", "/data/checkpoint"]\'',
            '  simulator.argv.json: \'["simulator-runner", "--queue", "/data/queue.jsonl"]\'',
            f'  checkpoint.sha256: "{CHECKPOINT_DIGEST}"',
            f'  launch-config.sha256: "{CONFIG_DIGEST}"',
            f'  image.digest: "sha256:{IMAGE_DIGEST}"',
        ]
    )
    (root / "configmap.yaml").write_text("\n".join(configmap_lines) + "\n", encoding="utf-8")
    (root / "policy-job.yaml").write_text(_job("policy"), encoding="utf-8")
    (root / "simulator-job.yaml").write_text(_job("simulator"), encoding="utf-8")
    (root / "policy-service.yaml").write_text(
        textwrap.dedent(
            f"""
            apiVersion: v1
            kind: Service
            metadata:
              name: v4-lane-policy-lane00-attempt01
              labels:
                v4-lane-id: lane00
                v4-attempt-id: attempt01
                v4-config-sha: {CONFIG_DIGEST}
                v4-lane-role: policy
            spec:
              publishNotReadyAddresses: false
              selector:
                v4-lane-id: lane00
                v4-attempt-id: attempt01
                v4-config-sha: {CONFIG_DIGEST}
                v4-lane-role: policy
              ports:
                - name: policy
                  port: 8100
                  targetPort: 8100
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (root / "kustomization.yaml").write_text(
        textwrap.dedent(
            """
            apiVersion: kustomize.config.k8s.io/v1beta1
            kind: Kustomization
            resources:
              - configmap.yaml
              - scripts-configmap.yaml
              - policy-job.yaml
              - policy-service.yaml
              - simulator-job.yaml
            """
        ).lstrip(),
        encoding="utf-8",
    )


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    _write_valid_bundle(root)
    return root


@pytest.mark.skipif(shutil.which("kubectl") is None, reason="validator deliberately uses kubectl's YAML decoder")
def test_valid_lane_bundle_passes(bundle: Path) -> None:
    report = VALIDATOR.validate(bundle, verify_generated=False)
    assert report["status"] == "valid_v4_k8s_lane_bundle"
    assert report["jobs"]["simulator"]["gpu_product"] == "NVIDIA-A40"
    assert report["jobs"]["policy"]["gpu_product"] == "NVIDIA-A100-SXM4-80GB"


@pytest.mark.skipif(shutil.which("kubectl") is None, reason="validator deliberately uses kubectl's YAML decoder")
@pytest.mark.parametrize(
    ("relative", "old", "new", "message"),
    [
        ("policy-job.yaml", "kind: Job", "kind: Pod", "standalone Pods"),
        ("simulator-job.yaml", "NVIDIA-A40", "NVIDIA-A100-SXM4-80GB", "exact GPU product"),
        ("simulator-job.yaml", "nvidia.com/gpu: 1", "nvidia.com/gpu: 2", "exactly one GPU"),
        ("policy-job.yaml", "backoffLimit: 0", "backoffLimit: 1", "backoffLimit"),
        ("policy-job.yaml", "restartPolicy: Never", "restartPolicy: OnFailure", "restartPolicy"),
        ("configmap.yaml", "immutable: true", "immutable: false", "immutable"),
        ("policy-job.yaml", "readinessProbe:", "notReadinessProbe:", "readinessProbe"),
        ("policy-job.yaml", "terminationGracePeriodSeconds: 300", "terminationGracePeriodSeconds: 30", "termination grace"),
        ("policy-job.yaml", "fieldPath: metadata.uid", "fieldPath: metadata.name", "metadata.uid"),
        ("policy-service.yaml", "    v4-lane-role: policy", "    v4-lane-role: simulator", "exact policy role"),
        (
            "configmap.yaml",
            "/opt/v4-lane/scripts/check_policy_ready.py",
            "/opt/v4-lane/scripts/missing_ready.py",
            "scripts ConfigMap lacks embedded runtime script missing_ready.py",
        ),
        (
            "configmap.yaml",
            "/data/openpi/src/openpi/serving/websocket_policy_server.py",
            "/data/openpi/src/openpi/serving/unbound_server.py",
            "omit HTTP health server semantics",
        ),
    ],
)
def test_bundle_fails_closed_on_manifest_regression(
    bundle: Path, relative: str, old: str, new: str, message: str
) -> None:
    path = bundle / relative
    body = path.read_text(encoding="utf-8")
    assert old in body
    path.write_text(body.replace(old, new, 1), encoding="utf-8")
    if relative == "policy-service.yaml" and old.strip().startswith("v4-lane-role"):
        path.write_text(path.read_text(encoding="utf-8").replace("v4-lane-role: policy", "v4-lane-role: simulator"), encoding="utf-8")
    with pytest.raises(VALIDATOR.LaneBundleValidationError, match=message):
        VALIDATOR.validate(bundle, verify_generated=False)


@pytest.mark.skipif(shutil.which("kubectl") is None, reason="validator deliberately uses kubectl's YAML decoder")
def test_bundle_rejects_episode_directory_precreation(bundle: Path) -> None:
    path = bundle / "scripts/lane_entrypoint.py"
    path.write_text(path.read_text(encoding="utf-8") + '\nos.mkdir("episode_dir")\n', encoding="utf-8")
    with pytest.raises(VALIDATOR.LaneBundleValidationError, match="pre-creates"):
        VALIDATOR.validate(bundle, verify_generated=False)


@pytest.mark.skipif(shutil.which("kubectl") is None, reason="validator deliberately uses kubectl's YAML decoder")
def test_bundle_rejects_preflight_after_exec(bundle: Path) -> None:
    path = bundle / "scripts/lane_entrypoint.py"
    body = path.read_text(encoding="utf-8")
    body = body.replace("run_preflight()\n    os.execvp", "os.execvp").replace(
        '["experiment"])', '["experiment"])\n    run_preflight()'
    )
    path.write_text(body, encoding="utf-8")
    with pytest.raises(VALIDATOR.LaneBundleValidationError, match="execs before"):
        VALIDATOR.validate(bundle, verify_generated=False)


@pytest.mark.skipif(shutil.which("kubectl") is None, reason="validator deliberately uses kubectl's YAML decoder")
def test_bundle_rejects_background_exec_pattern(bundle: Path) -> None:
    path = bundle / "scripts/startup_preflight.py"
    path.write_text(path.read_text(encoding="utf-8") + "\n# forbidden launch\nkubectl exec pod -- server &\n", encoding="utf-8")
    with pytest.raises(VALIDATOR.LaneBundleValidationError, match="kubectl exec|background"):
        VALIDATOR.validate(bundle, verify_generated=False)


@pytest.mark.skipif(shutil.which("kubectl") is None, reason="validator deliberately uses kubectl's YAML decoder")
def test_valid_lane_bundle_reports_unique_identity(bundle: Path) -> None:
    report = VALIDATOR.validate(bundle, verify_generated=False)
    assert report["lane_identity"]["v4-lane-id"] == "lane00"
    assert report["lane_identity"]["v4-attempt-id"] == "attempt01"


@pytest.mark.skipif(shutil.which("kubectl") is None, reason="validator deliberately uses kubectl's YAML decoder")
def test_bundle_rejects_missing_readiness_launch_config_binding(bundle: Path) -> None:
    path = bundle / "policy-job.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("--launch-config", "--missing-launch-config", 1), encoding="utf-8")
    with pytest.raises(VALIDATOR.LaneBundleValidationError, match="launch config"):
        VALIDATOR.validate(bundle, verify_generated=False)


@pytest.mark.skipif(shutil.which("kubectl") is None, reason="validator deliberately uses kubectl's YAML decoder")
def test_bundle_rejects_qualification_scope_with_behavioral_argv(bundle: Path) -> None:
    path = bundle / "configmap.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            '      "/usr/bin/true"',
            '      "/usr/bin/python3",\n      "/data/run_online_correction_v4_episodes.py"',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(VALIDATOR.LaneBundleValidationError, match="qualification-only simulator"):
        VALIDATOR.validate(bundle, verify_generated=False)


def test_generated_manifest_check_rejects_mutation(tmp_path: Path) -> None:
    renderer = ROOT / "tools/render_v4_k8s_lane_bundle.py"
    output = tmp_path / "rendered"
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, str(renderer), "--spec", str(ROOT / "deploy/k8s/v4_lane_bundle/spec.example.json"), "--output-root", str(output)],
        check=True,
    )
    VALIDATOR._validate_generated_from_spec(
        output, ROOT / "deploy/k8s/v4_lane_bundle/spec.example.json"
    )
    manifest = output / "policy-service.yaml"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "# mutation\n", encoding="utf-8")
    with pytest.raises(VALIDATOR.LaneBundleValidationError, match="stale or hand-edited"):
        VALIDATOR._validate_generated_from_spec(
            output, ROOT / "deploy/k8s/v4_lane_bundle/spec.example.json"
        )


@pytest.mark.skipif(shutil.which("kubectl") is None, reason="validator deliberately uses kubectl's YAML decoder")
def test_fresh_portable_render_passes_end_to_end_validator(tmp_path: Path) -> None:
    import subprocess
    import sys

    spec = ROOT / "deploy/k8s/v4_lane_bundle/spec.example.json"
    output = tmp_path / "portable-render"
    subprocess.run(
        [sys.executable, str(ROOT / "tools/render_v4_k8s_lane_bundle.py"), "--spec", str(spec), "--output-root", str(output)],
        check=True,
    )
    report = VALIDATOR.validate(output, spec_path=spec)
    assert report["generated_from_spec"] is True
    assert report["policy_service"]["selector"]["v4-lane-role"] == "policy"
