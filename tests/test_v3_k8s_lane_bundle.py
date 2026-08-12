from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import textwrap

import pytest


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "tools/validate_v3_k8s_lane_bundle.py"
SPEC = importlib.util.spec_from_file_location("validate_v3_k8s_lane_bundle", SOURCE)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


IMAGE_DIGEST = "03f5ce7d090fbd378070a8216d0aedfc6e473c52da99b40b0cf53918612a297c"
CHECKPOINT_DIGEST = "a" * 64
CONFIG_DIGEST = "b" * 64


def _job(role: str) -> str:
    gpu = "NVIDIA-A40" if role == "simulator" else "NVIDIA-A100-SXM4-80GB"
    body = textwrap.dedent(
        f"""
        apiVersion: batch/v1
        kind: Job
        metadata:
          name: v3-lane-{role}-lane00-attempt01
          labels:
            v3-lane-role: {role}
        spec:
          completions: 1
          parallelism: 1
          backoffLimit: 0
          template:
            metadata:
              labels:
                vla-wam/lane-id: lane00
                vla-wam/attempt-id: attempt01
                vla-wam/config-sha256: {CONFIG_DIGEST}
                v3-lane-role: {role}
              annotations:
                vla-wam/image-digest: sha256:{IMAGE_DIGEST}
            spec:
              restartPolicy: Never
              terminationGracePeriodSeconds: 180
              nodeSelector:
                node-role.kubernetes.io/worker-gpu: ""
                nvidia.com/gpu.product: {gpu}
              containers:
                - name: {role}
                  image: example.invalid/v3-lane@sha256:{IMAGE_DIGEST}
                  command:
                    - /usr/bin/python3
                    - /opt/v3-lane/scripts/lane_entrypoint.py
                  args:
                    - --role
                    - {role}
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
                    - name: OUTPUT_PARENT
                      value: /data/users/ali/vla_wam/raw/v3_lane_jobs
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
                      value: /data/curobo/src:/opt/v3-lane
                    - name: FFMPEG_BIN
                      value: /usr/bin/ffmpeg
                    - name: PYTHON_BIN
                      value: /usr/bin/python3
                    - name: NVIDIA_DRIVER_CAPABILITIES
                      value: compute,graphics,utility
                    - name: DISPLAY
                      value: ""
                    - name: LD_PRELOAD
                      value: ""
                    - name: IMAGE_DIGEST_EXPECTED
                      valueFrom:
                        configMapKeyRef:
                          name: v3-lane-launch
                          key: image.digest
                    - name: CONFIG_SHA256
                      valueFrom:
                        configMapKeyRef:
                          name: v3-lane-launch
                          key: launch-config.sha256
                    - name: CHECKPOINT_SHA256
                      valueFrom:
                        configMapKeyRef:
                          name: v3-lane-launch
                          key: checkpoint.sha256
                    - name: EXPERIMENT_ARGV_JSON
                      valueFrom:
                        configMapKeyRef:
                          name: v3-lane-launch
                          key: {role}.argv.json
                    # POLICY_PORT_ENV
                  volumeMounts:
                    - name: lane-runtime
                      mountPath: /lane-runtime
                  lifecycle:
                    preStop:
                      exec:
                        command: [/usr/bin/python3, /opt/v3-lane/scripts/lane_entrypoint.py, --prestop]
                  # POLICY_READINESS
              volumes:
                - name: lane-runtime
                  emptyDir: {{}}
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
                  - /opt/v3-lane/scripts/check_policy_ready.py
                  - --checkpoint-loaded
              periodSeconds: 2
            """
        ).lstrip().rstrip(),
        " " * 10,
    )
    return body.replace("            # POLICY_PORT_ENV", port_env).replace(
        "          # POLICY_READINESS", readiness
    )


def _write_valid_bundle(root: Path) -> None:
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (root / "configmap.yaml").write_text(
        textwrap.dedent(
            f"""
            apiVersion: v1
            kind: ConfigMap
            metadata:
              name: v3-lane-launch
            immutable: true
            data:
              policy.launch.json: '{{"role":"policy","experiment_argv":["/data/bin/policy-server","--checkpoint","/data/checkpoint"],"checkpoint_path":"/data/checkpoint","expected_gpu_name":"NVIDIA A100-SXM4-80GB","expected_driver_version":"580.95.05","readiness_contract":"tcp_bind_after_checkpoint_load","file_bindings":[{{"path":"/opt/v3-lane/scripts/lane_entrypoint.py","bytes":1,"sha256":"{CHECKPOINT_DIGEST}"}},{{"path":"/opt/v3-lane/scripts/startup_preflight.py","bytes":1,"sha256":"{CHECKPOINT_DIGEST}"}},{{"path":"/opt/v3-lane/scripts/check_policy_ready.py","bytes":1,"sha256":"{CHECKPOINT_DIGEST}"}},{{"path":"/data/checkpoint","bytes":1,"sha256":"{CHECKPOINT_DIGEST}"}}]}}'
              simulator.launch.json: '{{"role":"simulator","experiment_argv":["/data/bin/simulator-runner","--queue","/data/queue.jsonl"],"checkpoint_path":"/data/checkpoint","expected_gpu_name":"NVIDIA A40","expected_driver_version":"580.95.05","vulkan_contract":"isaac_app_launcher_rtx_frame_under_bound_vk_icd","render_probe_argv":["/data/bin/render-probe","{{rendered_frame}}"],"policy_wait":{{"mode":"tcp","service_identity":{{"vla-wam/lane-id":"lane00","vla-wam/attempt-id":"attempt01","vla-wam/config-sha256":"{CONFIG_DIGEST}","v3-lane-role":"policy","service_name":"v3-lane-policy-lane00-attempt01"}}}},"file_bindings":[{{"path":"/opt/v3-lane/scripts/lane_entrypoint.py","bytes":1,"sha256":"{CHECKPOINT_DIGEST}"}},{{"path":"/opt/v3-lane/scripts/startup_preflight.py","bytes":1,"sha256":"{CHECKPOINT_DIGEST}"}},{{"path":"/opt/v3-lane/scripts/isaac_render_probe.py","bytes":1,"sha256":"{CHECKPOINT_DIGEST}"}},{{"path":"/data/checkpoint","bytes":1,"sha256":"{CHECKPOINT_DIGEST}"}}]}}'
              policy.argv.json: '["policy-server", "--checkpoint", "/data/checkpoint"]'
              simulator.argv.json: '["simulator-runner", "--queue", "/data/queue.jsonl"]'
              checkpoint.sha256: "{CHECKPOINT_DIGEST}"
              launch-config.sha256: "{CONFIG_DIGEST}"
              image.digest: "sha256:{IMAGE_DIGEST}"
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (root / "policy-job.yaml").write_text(_job("policy"), encoding="utf-8")
    (root / "simulator-job.yaml").write_text(_job("simulator"), encoding="utf-8")
    (root / "policy-service.yaml").write_text(
        textwrap.dedent(
            f"""
            apiVersion: v1
            kind: Service
            metadata:
              name: v3-lane-policy-lane00-attempt01
            spec:
              publishNotReadyAddresses: false
              selector:
                vla-wam/lane-id: lane00
                vla-wam/attempt-id: attempt01
                vla-wam/config-sha256: {CONFIG_DIGEST}
                v3-lane-role: policy
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
              - policy-job.yaml
              - policy-service.yaml
              - simulator-job.yaml
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (scripts / "startup_preflight.py").write_text(
        textwrap.dedent(
            '''
            import os

            # Concrete proof names retained in runtime-preflight.json:
            # torch.cuda kernel on cuda:0; vulkaninfo; rendered_frame.png;
            # import curobo; ffmpeg_encode encoded.mp4; ffmpeg_decode decoded.raw;
            # checkpoint_sha256; writable_parent; O_EXCL port_lock.
            # Runtime-startup fields: effective_environment, gpu_uuid,
            # image_digest, argv, checkpoint_sha256.
            def acquire_attempt_lock():
                os.open("attempt-lock", os.O_CREAT | os.O_EXCL)

            def run_preflight():
                required = ("POD_UID", "LANE_ID", "ATTEMPT_ID")
                attempt = f"{os.environ['POD_UID']}/lane-{os.environ['LANE_ID']}/attempt-{os.environ['ATTEMPT_ID']}"
                # simulator policy_wait: wait_for_policy readiness and policy_metadata runtime-startup.json
                assert required and attempt
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    (scripts / "lane_entrypoint.py").write_text(
        textwrap.dedent(
            '''
            import os
            from startup_preflight import run_preflight

            IMAGE_DIGEST_EXPECTED = os.environ.get("IMAGE_DIGEST_EXPECTED")

            def main():
                run_preflight()
                os.execvp("experiment", ["experiment"])

            if __name__ == "__main__":
                main()
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    (scripts / "check_policy_ready.py").write_text(
        "# checkpoint-loaded policy-ready marker must exist\n",
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
    assert report["status"] == "valid_v3_k8s_lane_bundle"
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
        ("policy-job.yaml", "terminationGracePeriodSeconds: 180", "terminationGracePeriodSeconds: 30", "termination grace"),
        ("policy-job.yaml", "fieldPath: metadata.uid", "fieldPath: metadata.name", "metadata.uid"),
        ("policy-service.yaml", "v3-lane-role: policy", "v3-lane-role: simulator", "exact policy role"),
        (
            "configmap.yaml",
            "/opt/v3-lane/scripts/check_policy_ready.py",
            "/opt/v3-lane/scripts/missing_ready.py",
            "omit required runtime inputs",
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


def test_generated_manifest_check_rejects_mutation(tmp_path: Path) -> None:
    renderer = ROOT / "tools/render_v3_k8s_lane_bundle.py"
    output = tmp_path / "rendered"
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, str(renderer), "--spec", str(ROOT / "deploy/k8s/v3_lane_bundle/spec.example.json"), "--output-root", str(output)],
        check=True,
    )
    VALIDATOR._validate_generated_from_spec(
        output, ROOT / "deploy/k8s/v3_lane_bundle/spec.example.json"
    )
    manifest = output / "policy-service.yaml"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "# mutation\n", encoding="utf-8")
    with pytest.raises(VALIDATOR.LaneBundleValidationError, match="stale or hand-edited"):
        VALIDATOR._validate_generated_from_spec(
            output, ROOT / "deploy/k8s/v3_lane_bundle/spec.example.json"
        )


@pytest.mark.skipif(shutil.which("kubectl") is None, reason="validator deliberately uses kubectl's YAML decoder")
def test_fresh_portable_render_passes_end_to_end_validator(tmp_path: Path) -> None:
    import subprocess
    import sys

    spec = ROOT / "deploy/k8s/v3_lane_bundle/spec.example.json"
    output = tmp_path / "portable-render"
    subprocess.run(
        [sys.executable, str(ROOT / "tools/render_v3_k8s_lane_bundle.py"), "--spec", str(spec), "--output-root", str(output)],
        check=True,
    )
    report = VALIDATOR.validate(output, spec_path=spec)
    assert report["generated_from_spec"] is True
    assert report["policy_service"]["selector"]["v3-lane-role"] == "policy"
