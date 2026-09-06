#!/usr/bin/env python3
"""Render one immutable V4 simulator/policy Kubernetes lane from a strict JSON spec."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "deploy/k8s/v4_lane_bundle"
DEFAULT_SPEC = DEFAULT_ROOT / "spec.example.json"
SHA_RE = re.compile(r"[0-9a-f]{64}")
TOKEN_RE = re.compile(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?")
TOP_LEVEL_KEYS = {
    "schema_version", "qualification_only", "qualification_kind", "kube_context", "namespace", "lane_id", "attempt_id",
    "policy_port", "expected_driver_version", "image_repository", "image_sha256",
    "image_pull_secret", "pvc", "output_parent", "entrypoint", "prestop_wait_seconds",
    "runtime", "policy", "simulator",
}
RUNTIME_KEYS = {"python_bin", "ffmpeg_bin", "vk_icd_filenames", "ld_library_path", "pythonpath"}
ROLE_KEYS = {
    "gpu_product", "expected_gpu_name", "experiment_argv", "checkpoint_path", "checkpoint_sha256",
    "nvidia_smi_bin", "python_imports", "file_bindings", "vulkan_contract", "render_probe_argv",
    "render_probe_timeout_seconds", "cuda_probe_argv", "readiness_interface",
}
BINDING_KEYS = {"source", "path", "bytes", "sha256"}
QUALIFICATION_SCOPE = "infrastructure_qualification_only_no_scientific_behavior"
G4_POLICY_SESSION_SCOPE = "g4_policy_session_only_no_behavioral_episode"
QUALIFICATION_KINDS = frozenset({"g1_infrastructure", "g4_policy_session"})
NOOP_RUNNERS = frozenset({"/usr/bin/true", "/bin/true"})
V4_RUNNER_MARKER = "online_correction_v4"
REQUIRED_IDENTITY_LABELS = ("v4-lane-id", "v4-attempt-id", "v4-config-sha")
SCRIPTS_MOUNT = "/opt/v4-lane/scripts"
DEFAULT_ENTRYPOINT = f"{SCRIPTS_MOUNT}/lane_entrypoint.py"
RUNTIME_SCRIPT_NAMES = (
    "lane_entrypoint.py",
    "startup_preflight.py",
    "check_policy_ready.py",
    "isaac_render_probe.py",
)


class RenderError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RenderError(message)


def require_exact_keys(mapping: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    require(not unknown, f"{label} contains unknown keys: {unknown}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"


def yaml_scalar(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def read_spec(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RenderError(f"invalid lane spec JSON: {path}") from exc
    require(isinstance(spec, dict), "lane spec must be an object")
    require_exact_keys(spec, TOP_LEVEL_KEYS, "lane spec")
    require(spec.get("schema_version") == "vla-wam-v4-k8s-lane-render-spec-v1", "lane spec schema differs")
    return spec, sha256_bytes(raw)


def token(value: Any, label: str) -> str:
    require(isinstance(value, str) and TOKEN_RE.fullmatch(value) is not None, f"unsafe {label}")
    return value


def absolute(value: Any, label: str) -> str:
    require(isinstance(value, str) and Path(value).is_absolute(), f"{label} must be absolute")
    return value


def digest(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA_RE.fullmatch(value) is not None, f"{label} must be SHA-256")
    return value


GPU_PRODUCT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}[A-Za-z0-9]$|^[A-Za-z0-9]$")


def gpu_product(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and GPU_PRODUCT_RE.fullmatch(value) is not None,
        f"unsafe {label}",
    )
    return value


def gpu_display_name(value: Any, label: str) -> str:
    require(isinstance(value, str) and value.strip(), f"{label} must be a nonempty GPU display name")
    return value.strip()


def binding_rows(rows: Any, *, spec_dir: Path) -> list[dict[str, Any]]:
    require(isinstance(rows, list) and rows, "file_bindings must be nonempty")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        require(isinstance(row, dict), f"file_bindings[{index}] must be an object")
        require_exact_keys(row, BINDING_KEYS, f"file_bindings[{index}]")
        mounted = absolute(row.get("path"), f"file_bindings[{index}].path")
        require(mounted not in seen, f"repeated mounted file binding: {mounted}")
        seen.add(mounted)
        source_raw = row.get("source")
        if source_raw is not None:
            require(isinstance(source_raw, str) and source_raw, f"file_bindings[{index}].source is invalid")
            source = (spec_dir / source_raw).resolve() if not Path(source_raw).is_absolute() else Path(source_raw)
            require(source.is_file(), f"bound local source does not exist: {source}")
            observed_bytes = source.stat().st_size
            observed_sha256 = sha256_file(source)
            if "bytes" in row:
                require(row["bytes"] == observed_bytes, f"file_bindings[{index}].bytes differs from source")
            if "sha256" in row:
                require(row["sha256"] == observed_sha256, f"file_bindings[{index}].sha256 differs from source")
            result.append({"path": mounted, "bytes": observed_bytes, "sha256": observed_sha256})
            continue
        external_bytes = row.get("bytes")
        require(
            type(external_bytes) is int and external_bytes >= 0,
            f"file_bindings[{index}] external bytes must be a nonnegative integer",
        )
        external_sha256 = digest(
            row.get("sha256"),
            f"file_bindings[{index}] external sha256",
        )
        result.append({"path": mounted, "bytes": external_bytes, "sha256": external_sha256})
    return result


def _argv_paths(argv: list[str]) -> list[str]:
    return [item for item in argv if isinstance(item, str) and Path(item).is_absolute()]


def _binding_paths(document: Mapping[str, Any]) -> set[str]:
    return {str(binding["path"]) for binding in document.get("file_bindings", [])}


def _contains_v4_runner(*parts: str) -> bool:
    lowered = "/".join(parts).lower()
    return V4_RUNNER_MARKER in lowered and "online_correction_v4.py" not in lowered


def validate_qualification_contract(
    *,
    qualification_only: bool,
    qualification_kind: str,
    policy_doc: Mapping[str, Any],
    simulator_doc: Mapping[str, Any],
) -> None:
    require(isinstance(qualification_only, bool), "qualification_only must be a boolean")
    require(
        qualification_kind in QUALIFICATION_KINDS,
        f"qualification_kind must be one of {sorted(QUALIFICATION_KINDS)}",
    )
    sim_argv = list(simulator_doc.get("experiment_argv") or [])
    sim_bindings = _binding_paths(simulator_doc)
    sim_paths = set(_argv_paths(sim_argv))

    if qualification_only:
        if qualification_kind == "g1_infrastructure":
            require(
                sim_argv == ["/usr/bin/true"] or sim_argv == ["/bin/true"],
                "G1 qualification simulator must execute /usr/bin/true only",
            )
            require(
                not any(_contains_v4_runner(path) for path in sim_paths | sim_bindings),
                "G1 qualification forbids online_correction_v4 behavioral runner bindings",
            )
            require(
                not any(str(item).endswith(".py") for item in sim_argv),
                "G1 qualification forbids a Python runner in simulator argv",
            )
        else:
            require(
                any(str(item).endswith("run_v4_g4_nano_policy_session.py") for item in sim_argv),
                "G4 policy-session qualification must invoke the registered G4 probe",
            )
            require(
                any(str(item).endswith("run_v4_g4_nano_policy_session.py") for item in sim_bindings),
                "G4 policy-session qualification must bind the registered G4 probe",
            )
            require(
                not any(str(item).endswith("run_online_correction_v4.py") for item in sim_paths | sim_bindings),
                "G4 policy-session qualification forbids the behavioral episode runner",
            )
        return
    require(
        qualification_kind == "g1_infrastructure",
        "behavioral execution cannot declare a qualification-only kind",
    )

    require(sim_argv not in (["/usr/bin/true"], ["/bin/true"]), "behavioral simulator must not use /usr/bin/true")
    require(len(sim_argv) >= 2, "behavioral simulator experiment_argv must be nonempty")
    runner_hits = [
        path
        for path in sim_paths | sim_bindings
        if _contains_v4_runner(path)
    ]
    require(runner_hits, "behavioral simulator must bind an online_correction_v4 runner entrypoint")
    argv_runner = [path for path in sim_paths if _contains_v4_runner(path)]
    require(argv_runner, "behavioral simulator argv must invoke the bound online_correction_v4 runner")
    binding_runner = [path for path in sim_bindings if _contains_v4_runner(path)]
    require(binding_runner, "behavioral simulator file_bindings must include the online_correction_v4 runner")
    require(
        set(argv_runner) & set(binding_runner),
        "behavioral simulator argv and file_bindings must bind the same online_correction_v4 runner path",
    )
    # Policy server argv remains allowed for behavioral lanes; only the simulator owns episodes.
    _ = policy_doc


def launch_document(
    raw: Any,
    *,
    role: str,
    spec_dir: Path,
    policy_service: str,
    policy_port: int,
    service_identity: Mapping[str, str],
    expected_driver_version: str,
    entrypoint: str,
) -> dict[str, Any]:
    require(isinstance(raw, dict), f"{role} launch must be an object")
    require_exact_keys(raw, ROLE_KEYS, f"{role} launch")
    document = dict(raw)
    role_gpu_product = gpu_product(document.pop("gpu_product"), f"{role} gpu_product")
    role_gpu_name = gpu_display_name(document.pop("expected_gpu_name"), f"{role} expected_gpu_name")
    document.update(
        {
            "schema_version": "vla-wam-v4-k8s-lane-launch-v1",
            "role": role,
            "file_bindings": binding_rows(raw.get("file_bindings"), spec_dir=spec_dir),
            "gpu_product": role_gpu_product,
            "expected_gpu_name": role_gpu_name,
            "expected_driver_version": expected_driver_version,
        }
    )
    argv = document.get("experiment_argv")
    require(isinstance(argv, list) and argv and all(isinstance(item, str) and item for item in argv), f"{role} argv invalid")
    absolute(argv[0], f"{role} experiment_argv[0]")
    for key in ("checkpoint_path", "nvidia_smi_bin"):
        absolute(document.get(key), f"{role} {key}")
    digest(document.get("checkpoint_sha256"), f"{role} checkpoint_sha256")
    imports = document.get("python_imports")
    require(isinstance(imports, list) and imports and all(isinstance(item, str) and item for item in imports), f"{role} imports invalid")
    render = document.get("render_probe_argv")
    if role == "simulator":
        require(
            document.get("vulkan_contract") == "isaac_app_launcher_rtx_frame_under_bound_vk_icd",
            "simulator vulkan_contract differs",
        )
        require(any(str(item).split(".")[0] == "curobo" for item in imports), "simulator imports lack CuRobo")
        require(isinstance(render, list) and render and absolute(render[0], "simulator render argv[0]"), "simulator render argv invalid")
        require(any("{rendered_frame}" in str(item) for item in render), "simulator render argv lacks output placeholder")
    else:
        require(render is None and "vulkan_contract" not in document, "policy must not declare simulator render/Vulkan probes")
    cuda_probe = document.get("cuda_probe_argv")
    if cuda_probe is not None:
        require(
            isinstance(cuda_probe, list) and cuda_probe and all(isinstance(item, str) and item for item in cuda_probe),
            f"{role} cuda_probe_argv invalid",
        )
        absolute(cuda_probe[0], f"{role} cuda_probe_argv[0]")
    if role == "policy":
        port_flags = [index for index, item in enumerate(argv[:-1]) if item == "--port"]
        require(
            len(port_flags) == 1 and str(argv[port_flags[0] + 1]) == str(policy_port),
            "policy experiment argv port differs from policy_port",
        )
        document["policy_port"] = policy_port
        document["readiness_contract"] = "http_healthz_after_checkpoint_load"
        readiness_interface = document.pop("readiness_interface", "openpi_http_healthz")
        require(
            readiness_interface in {"openpi_http_healthz", "cosmos_http_healthz"},
            "unsupported policy readiness_interface",
        )
        document["readiness_interface"] = readiness_interface
        document.pop("policy_wait", None)
    else:
        document["policy_wait"] = {
            "mode": "http_healthz",
            "host": policy_service,
            "port": policy_port,
            "timeout_seconds": 900,
            "poll_seconds": 2,
            "service_identity": dict(service_identity),
        }
        document.pop("readiness_contract", None)
    binding_paths = {str(binding["path"]) for binding in document["file_bindings"]}
    entrypoint_path = Path(entrypoint)
    required_bindings = {
        entrypoint,
        str(entrypoint_path.with_name("startup_preflight.py")),
        str(document["checkpoint_path"]),
        str(
            entrypoint_path.with_name(
                "check_policy_ready.py" if role == "policy" else "isaac_render_probe.py"
            )
        ),
    }
    for item in list(argv) + (list(render) if isinstance(render, list) else []):
        if Path(item).is_absolute() and Path(item).suffix == ".py":
            required_bindings.add(item)
    missing = sorted(required_bindings - binding_paths)
    require(not missing, f"{role} file_bindings omit required runtime inputs: {missing}")
    if (
        role == "policy"
        and document.get("readiness_contract") == "http_healthz_after_checkpoint_load"
        and document.get("readiness_interface") == "openpi_http_healthz"
    ):
        openpi_flags = [index for index, item in enumerate(argv[:-1]) if item == "--openpi-root"]
        require(len(openpi_flags) == 1, "HTTP health readiness requires one exact --openpi-root")
        openpi_root = absolute(argv[openpi_flags[0] + 1], "policy --openpi-root")
        health_server = str(Path(openpi_root) / "src/openpi/serving/websocket_policy_server.py")
        require(
            health_server in binding_paths,
            f"policy file_bindings omit HTTP health server semantics: {health_server}",
        )
    if role == "policy" and document.get("readiness_interface") == "cosmos_http_healthz":
        checkpoint_flags = [
            index for index, item in enumerate(argv[:-1]) if item == "--checkpoint-path"
        ]
        require(
            len(checkpoint_flags) == 1
            and Path(str(argv[checkpoint_flags[0] + 1])).is_absolute(),
            "Cosmos HTTP readiness requires one absolute --checkpoint-path",
        )
        require("--decode-video" in argv, "Cosmos policy server must expose decoded futures")
        require(
            "--action-space" in argv
            and "joint_pos" in argv
            and "--action-chunk-size" in argv
            and "32" in argv,
            "Cosmos policy server action contract differs",
        )
    return document


def labels(lane: str, attempt: str, config_label: str, role: str | None = None) -> dict[str, str]:
    result = {"v4-lane-id": lane, "v4-attempt-id": attempt, "v4-config-sha": config_label}
    if role is not None:
        result["v4-lane-role"] = role
    return result


def metadata_lines(name: str, namespace: str, common_labels: Mapping[str, str], indent: str = "") -> list[str]:
    rows = [f"{indent}name: {yaml_scalar(name)}", f"{indent}namespace: {yaml_scalar(namespace)}", f"{indent}labels:"]
    rows.extend(f"{indent}  {key}: {yaml_scalar(value)}" for key, value in common_labels.items())
    return rows


def block(value: str, indent: int) -> str:
    prefix = " " * indent
    return "\n".join(
        prefix + line if line else ""
        for line in value.rstrip("\n").splitlines()
    )


def render_configmap(
    *, name: str, namespace: str, common_labels: Mapping[str, str], policy_json: str,
    simulator_json: str, image_digest: str, spec_sha: str, renderer_sha: str,
    immutable_identity_sha: str, bundle_sha: str, kube_context: str,
) -> str:
    rows = ["apiVersion: v1", "kind: ConfigMap", "metadata:"]
    rows += metadata_lines(name, namespace, common_labels, "  ")
    rows += ["immutable: true", "data:", "  policy-launch.json: |", block(policy_json, 4), "  simulator-launch.json: |", block(simulator_json, 4)]
    rows += [
        f"  kube.context: {yaml_scalar(kube_context)}",
        f"  image.digest: {yaml_scalar(image_digest)}",
        f"  render-spec.sha256: {yaml_scalar(spec_sha)}",
        f"  renderer.sha256: {yaml_scalar(renderer_sha)}",
        f"  immutable-identity.sha256: {yaml_scalar(immutable_identity_sha)}",
        f"  bundle-config.sha256: {yaml_scalar(bundle_sha)}",
    ]
    return "\n".join(rows) + "\n"


def load_runtime_scripts(script_root: Path) -> dict[str, bytes]:
    scripts: dict[str, bytes] = {}
    for name in RUNTIME_SCRIPT_NAMES:
        source = script_root / name
        require(source.is_file(), f"canonical runtime script is missing: {source}")
        scripts[name] = source.read_bytes()
    return scripts


def render_scripts_configmap(
    *,
    name: str,
    namespace: str,
    common_labels: Mapping[str, str],
    scripts: Mapping[str, bytes],
) -> str:
    rows = ["apiVersion: v1", "kind: ConfigMap", "metadata:"]
    rows += metadata_lines(name, namespace, common_labels, "  ")
    rows += ["immutable: true", "data:"]
    for filename in sorted(scripts):
        rows.append(f"  {filename}: |")
        rows.append(block(scripts[filename].decode("utf-8"), 4))
    return "\n".join(rows) + "\n"


def common_env(
    *, role: str, lane: str, attempt: str, output_parent: str, launch_path: str,
    launch_sha: str, image_digest: str, runtime: Mapping[str, Any], configmap: str,
    prestop_wait_seconds: int, kube_context: str,
) -> list[dict[str, Any]]:
    values = {
        "LANE_ROLE": role,
        "LANE_ID": lane,
        "ATTEMPT_ID": attempt,
        "OUTPUT_PARENT": output_parent,
        "KUBE_CONTEXT": kube_context,
        "LANE_LAUNCH_CONFIG": launch_path,
        "LANE_LAUNCH_CONFIG_SHA256": launch_sha,
        "HOME": "/lane-runtime/home",
        "XDG_CACHE_HOME": "/lane-runtime/xdg/cache",
        "XDG_CONFIG_HOME": "/lane-runtime/xdg/config",
        "XDG_RUNTIME_DIR": "/lane-runtime/xdg/runtime",
        "WARP_CACHE_PATH": "/lane-runtime/warp",
        "MPLCONFIGDIR": "/lane-runtime/matplotlib",
        "TMPDIR": "/lane-runtime/tmp",
        "TORCHINDUCTOR_CACHE_DIR": "/lane-runtime/torchinductor",
        "VK_ICD_FILENAMES": runtime["vk_icd_filenames"],
        "LD_LIBRARY_PATH": runtime["ld_library_path"],
        "PYTHONPATH": runtime["pythonpath"],
        "PYTHON_BIN": runtime["python_bin"],
        "FFMPEG_BIN": runtime["ffmpeg_bin"],
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "USER": "ali",
        "LOGNAME": "ali",
        "PRESTOP_WAIT_SECONDS": str(prestop_wait_seconds),
        "NVIDIA_DRIVER_CAPABILITIES": "compute,graphics,utility",
        "DISPLAY": "",
        "LD_PRELOAD": "",
        "OMNI_KIT_ACCEPT_EULA": "YES",
    }
    rows: list[dict[str, Any]] = [
        {"name": "POD_UID", "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}}},
        {"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
        {"name": "POD_NAMESPACE", "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}}},
        {"name": "POD_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}},
        {"name": "IMAGE_DIGEST_EXPECTED", "valueFrom": {"configMapKeyRef": {"name": configmap, "key": "image.digest"}}},
    ]
    rows += [{"name": key, "value": str(value)} for key, value in values.items()]
    return rows


def yaml_env(rows: list[dict[str, Any]], indent: int) -> list[str]:
    prefix = " " * indent
    output: list[str] = []
    for row in rows:
        output.append(f"{prefix}- name: {row['name']}")
        if "value" in row:
            output.append(f"{prefix}  value: {yaml_scalar(row['value'])}")
        else:
            output.append(f"{prefix}  valueFrom:")
            ref_type, ref = next(iter(row["valueFrom"].items()))
            output.append(f"{prefix}    {ref_type}:")
            for key, value in ref.items():
                output.append(f"{prefix}      {key}: {yaml_scalar(value)}")
    return output


def render_job(
    *, role: str, name: str, namespace: str, common_labels: Mapping[str, str],
    configmap: str, scripts_configmap: str, image: str, image_digest: str, gpu_product_value: str,
    lane: str, attempt: str, output_parent: str, launch_sha: str,
    runtime: Mapping[str, Any], policy_port: int, pvc: str, image_pull_secret: str,
    entrypoint: str,
    prestop_wait_seconds: int,
    kube_context: str,
) -> str:
    env = common_env(
        role=role, lane=lane, attempt=attempt, output_parent=output_parent,
        launch_path=f"/opt/v4-lane/config/{role}-launch.json", launch_sha=launch_sha,
        image_digest=image_digest, runtime=runtime, configmap=configmap,
        prestop_wait_seconds=prestop_wait_seconds, kube_context=kube_context,
    )
    if role == "policy":
        env.append({"name": "POLICY_PORT", "value": str(policy_port)})
    role_labels = {**common_labels, "v4-lane-role": role}
    rows = ["apiVersion: batch/v1", "kind: Job", "metadata:"]
    rows += metadata_lines(name, namespace, role_labels, "  ")
    rows += [
        "spec:", "  completions: 1", "  parallelism: 1", "  backoffLimit: 0", "  template:", "    metadata:", "      labels:",
    ]
    rows += [f"        {key}: {yaml_scalar(value)}" for key, value in role_labels.items()]
    rows += ["      annotations:", f"        v4-image-digest: {yaml_scalar(image_digest)}", "    spec:", "      restartPolicy: Never", "      terminationGracePeriodSeconds: 300"]
    rows += ["      nodeSelector:", '        node-role.kubernetes.io/worker-gpu: ""', f"        nvidia.com/gpu.product: {yaml_scalar(gpu_product_value)}"]
    rows += ["      tolerations:", "        - key: nvidia.com/gpu", "          operator: Equal", '          value: "present"', "          effect: NoSchedule"]
    rows += ["      securityContext:", "        fsGroup: 2518800", "        supplementalGroups: [2518800]", "        seccompProfile:", "          type: RuntimeDefault"]
    rows += ["      imagePullSecrets:", f"        - name: {yaml_scalar(image_pull_secret)}", "      containers:", f"        - name: {role}", f"          image: {yaml_scalar(image)}", "          imagePullPolicy: IfNotPresent"]
    rows += ["          command:", f"            - {yaml_scalar(runtime['python_bin'])}", f"            - {yaml_scalar(entrypoint)}", "          resources:", "            requests:", '              cpu: "16"', "              memory: 64Gi", "              nvidia.com/gpu: 1", "            limits:", '              cpu: "64"', "              memory: 128Gi", "              nvidia.com/gpu: 1"]
    rows += ["          securityContext:", "            allowPrivilegeEscalation: false", "            runAsNonRoot: true", "            runAsUser: 816149040", "            runAsGroup: 2518800", "            capabilities:", "              drop: [ALL]", "          env:"]
    rows += yaml_env(env, 12)
    rows += [
        "          volumeMounts:",
        "            - name: data",
        "              mountPath: /data",
        "            - name: lane-runtime",
        "              mountPath: /lane-runtime",
        "            - name: dshm",
        "              mountPath: /dev/shm",
        "            - name: launch-config",
        "              mountPath: /opt/v4-lane/config",
        "              readOnly: true",
        "            - name: lane-scripts",
        f"              mountPath: {yaml_scalar(SCRIPTS_MOUNT)}",
        "              readOnly: true",
    ]
    rows += ["          lifecycle:", "            preStop:", "              exec:", "                command:", f"                  - {yaml_scalar(runtime['python_bin'])}", f"                  - {yaml_scalar(entrypoint)}", "                  - --prestop"]
    if role == "policy":
        probe = runtime["python_bin"]
        ready = str(Path(entrypoint).with_name("check_policy_ready.py"))
        rows += ["          readinessProbe:", "            exec:", "              command:", f"                - {yaml_scalar(probe)}", f"                - {yaml_scalar(ready)}", "                - --checkpoint-loaded", "                - --mode", "                - http_healthz", "                - --launch-config", "                - /opt/v4-lane/config/policy-launch.json", "                - --port", f"                - {yaml_scalar(str(policy_port))}", "            initialDelaySeconds: 5", "            periodSeconds: 5", "            timeoutSeconds: 4", "            failureThreshold: 180"]
    rows += [
        "      volumes:",
        "        - name: data",
        "          persistentVolumeClaim:",
        f"            claimName: {yaml_scalar(pvc)}",
        "        - name: lane-runtime",
        "          emptyDir: {}",
        "        - name: dshm",
        "          emptyDir:",
        "            medium: Memory",
        "            sizeLimit: 96Gi",
        "        - name: launch-config",
        "          configMap:",
        f"            name: {yaml_scalar(configmap)}",
        "        - name: lane-scripts",
        "          configMap:",
        f"            name: {yaml_scalar(scripts_configmap)}",
        "            defaultMode: 0555",
    ]
    return "\n".join(rows) + "\n"


def render_service(*, name: str, namespace: str, common_labels: Mapping[str, str], port: int) -> str:
    rows = ["apiVersion: v1", "kind: Service", "metadata:"]
    rows += metadata_lines(name, namespace, common_labels, "  ")
    rows += ["spec:", "  publishNotReadyAddresses: false", "  selector:"]
    rows += [f"    {key}: {yaml_scalar(value)}" for key, value in {**common_labels, "v4-lane-role": "policy"}.items()]
    rows += ["  ports:", "    - name: policy", f"      port: {port}", f"      targetPort: {port}", "      protocol: TCP"]
    return "\n".join(rows) + "\n"


def render(spec_path: Path, output_root: Path) -> dict[str, str]:
    spec_path = spec_path.resolve()
    spec, spec_sha = read_spec(spec_path)
    namespace = token(spec.get("namespace"), "namespace")
    lane = token(spec.get("lane_id"), "lane_id")
    attempt = token(spec.get("attempt_id"), "attempt_id")
    policy_port = int(spec.get("policy_port", 0))
    require(1 <= policy_port <= 65535, "invalid policy_port")
    prestop_wait_seconds = int(spec.get("prestop_wait_seconds", 120))
    require(1 <= prestop_wait_seconds <= 240, "prestop_wait_seconds must be in [1, 240]")
    image_digest_hex = digest(spec.get("image_sha256"), "image_sha256")
    image_repository = spec.get("image_repository")
    require(isinstance(image_repository, str) and image_repository and "@" not in image_repository, "invalid image_repository")
    image_digest = f"sha256:{image_digest_hex}"
    image = f"{image_repository}@{image_digest}"
    output_parent = absolute(spec.get("output_parent"), "output_parent")
    pvc = token(spec.get("pvc"), "pvc")
    image_pull_secret = token(spec.get("image_pull_secret"), "image_pull_secret")
    entrypoint = absolute(spec.get("entrypoint"), "entrypoint")
    require(entrypoint == DEFAULT_ENTRYPOINT, f"entrypoint must be the mounted runtime script {DEFAULT_ENTRYPOINT}")
    runtime = spec.get("runtime")
    require(isinstance(runtime, dict), "runtime must be an object")
    for role in ("policy", "simulator"):
        require(isinstance(runtime.get(role), dict), f"runtime.{role} missing")
        require_exact_keys(runtime[role], RUNTIME_KEYS, f"runtime.{role}")
        for key in ("python_bin", "ffmpeg_bin", "vk_icd_filenames"):
            absolute(runtime[role].get(key), f"runtime.{role}.{key}")
        for key in ("ld_library_path", "pythonpath"):
            value = runtime[role].get(key)
            require(isinstance(value, str) and value and all(Path(item).is_absolute() for item in value.split(":")), f"runtime.{role}.{key} invalid")

    kube_context = token(spec.get("kube_context"), "kube_context")
    driver_version = spec.get("expected_driver_version")
    require(isinstance(driver_version, str) and re.fullmatch(r"\d+\.\d+\.\d+", driver_version), "invalid expected_driver_version")
    policy_doc = launch_document(
        spec.get("policy"), role="policy", spec_dir=spec_path.parent,
        policy_service="identity-pending", policy_port=policy_port, service_identity={"pending": "true"},
        expected_driver_version=driver_version,
        entrypoint=entrypoint,
    )
    simulator_doc = launch_document(
        spec.get("simulator"), role="simulator", spec_dir=spec_path.parent,
        policy_service="identity-pending", policy_port=policy_port, service_identity={"pending": "true"},
        expected_driver_version=driver_version,
        entrypoint=entrypoint,
    )
    qualification_only = spec.get("qualification_only") is True
    require(spec.get("qualification_only") in (True, False), "qualification_only must be explicit true or false")
    qualification_kind = spec.get("qualification_kind", "g1_infrastructure")
    validate_qualification_contract(
        qualification_only=qualification_only,
        qualification_kind=qualification_kind,
        policy_doc=policy_doc,
        simulator_doc=simulator_doc,
    )
    if qualification_only:
        launch_scope = (
            G4_POLICY_SESSION_SCOPE
            if qualification_kind == "g4_policy_session"
            else QUALIFICATION_SCOPE
        )
        policy_doc["launch_scope"] = launch_scope
        simulator_doc["launch_scope"] = launch_scope

    renderer_sha = sha256_file(Path(__file__).resolve())
    runtime_scripts = load_runtime_scripts(DEFAULT_ROOT / "scripts")
    identity_input = {
        "render_spec_sha256": spec_sha,
        "renderer_sha256": renderer_sha,
        "runtime_scripts": sorted(
            (name, sha256_bytes(content)) for name, content in runtime_scripts.items()
        ),
        "bindings": sorted(
            (
                role,
                str(binding["path"]),
                int(binding["bytes"]),
                str(binding["sha256"]),
            )
            for role, document in (("policy", policy_doc), ("simulator", simulator_doc))
            for binding in document["file_bindings"]
        ),
    }
    immutable_identity_sha = sha256_bytes(canonical_json(identity_input).encode())
    config_label = immutable_identity_sha[:32]
    common_labels = labels(lane, attempt, config_label)
    policy_labels = labels(lane, attempt, config_label, "policy")
    simulator_labels = labels(lane, attempt, config_label, "simulator")
    stem = f"v4-{lane}-{attempt}-{immutable_identity_sha[:10]}"
    require(len(stem) <= 48, "lane/attempt make immutable names too long")
    configmap = f"{stem}-launch"
    scripts_configmap = f"{stem}-scripts"
    policy_job = f"{stem}-policy"
    simulator_job = f"{stem}-sim"
    policy_service = f"{stem}-policy"
    service_identity = {**common_labels, "v4-lane-role": "policy", "service_name": policy_service, "namespace": namespace, "spec_sha256": spec_sha, "immutable_identity_sha256": immutable_identity_sha}
    simulator_doc["policy_wait"]["host"] = policy_service
    simulator_doc["policy_wait"]["service_identity"] = service_identity
    simulator_doc["experiment_argv"] = [
        policy_service if item == "{policy_service}" else item
        for item in simulator_doc["experiment_argv"]
    ]
    policy_json, simulator_json = canonical_json(policy_doc), canonical_json(simulator_doc)
    policy_sha, simulator_sha = sha256_bytes(policy_json.encode()), sha256_bytes(simulator_json.encode())
    bundle_sha = sha256_bytes((policy_sha + simulator_sha + immutable_identity_sha).encode())
    files = {
        "configmap.yaml": render_configmap(name=configmap, namespace=namespace, common_labels=common_labels, policy_json=policy_json, simulator_json=simulator_json, image_digest=image_digest, spec_sha=spec_sha, renderer_sha=renderer_sha, immutable_identity_sha=immutable_identity_sha, bundle_sha=bundle_sha, kube_context=kube_context),
        "scripts-configmap.yaml": render_scripts_configmap(
            name=scripts_configmap,
            namespace=namespace,
            common_labels=common_labels,
            scripts=runtime_scripts,
        ),
        "policy-job.yaml": render_job(role="policy", name=policy_job, namespace=namespace, common_labels=policy_labels, configmap=configmap, scripts_configmap=scripts_configmap, image=image, image_digest=image_digest, gpu_product_value=policy_doc["gpu_product"], lane=lane, attempt=attempt, output_parent=output_parent, launch_sha=policy_sha, runtime=runtime["policy"], policy_port=policy_port, pvc=pvc, image_pull_secret=image_pull_secret, entrypoint=entrypoint, prestop_wait_seconds=prestop_wait_seconds, kube_context=kube_context),
        "policy-service.yaml": render_service(name=policy_service, namespace=namespace, common_labels=policy_labels, port=policy_port),
        "simulator-job.yaml": render_job(role="simulator", name=simulator_job, namespace=namespace, common_labels=simulator_labels, configmap=configmap, scripts_configmap=scripts_configmap, image=image, image_digest=image_digest, gpu_product_value=simulator_doc["gpu_product"], lane=lane, attempt=attempt, output_parent=output_parent, launch_sha=simulator_sha, runtime=runtime["simulator"], policy_port=policy_port, pvc=pvc, image_pull_secret=image_pull_secret, entrypoint=entrypoint, prestop_wait_seconds=prestop_wait_seconds, kube_context=kube_context),
        "kustomization.yaml": "apiVersion: kustomize.config.k8s.io/v1beta1\nkind: Kustomization\nresources:\n  - configmap.yaml\n  - scripts-configmap.yaml\n  - policy-service.yaml\n  - policy-job.yaml\n  - simulator-job.yaml\n",
    }
    canonical_scripts = sorted((DEFAULT_ROOT / "scripts").glob("*.py"))
    require(canonical_scripts, "canonical lane runtime scripts are missing")
    output_root.mkdir(parents=True, exist_ok=True)
    existing = [name for name in files if (output_root / name).exists()]
    require(not existing, f"refusing to overwrite rendered manifests: {existing}")
    for name, content in files.items():
        path = output_root / name
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content)
    script_output = output_root / "scripts"
    script_output.mkdir(mode=0o755, exist_ok=True)
    hashes = {name: sha256_bytes(content.encode()) for name, content in files.items()}
    for source in canonical_scripts:
        destination = script_output / source.name
        content = source.read_bytes()
        if destination.resolve() == source.resolve():
            hashes[f"scripts/{source.name}"] = sha256_bytes(content)
            continue
        if destination.exists():
            require(
                destination.read_bytes() == content,
                f"refusing to overwrite mismatched canonical script: {source.name}",
            )
            hashes[f"scripts/{source.name}"] = sha256_bytes(content)
            continue
        with destination.open("xb") as handle:
            handle.write(content)
        hashes[f"scripts/{source.name}"] = sha256_bytes(content)
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    hashes = render(args.spec, args.output_root.resolve())
    print(json.dumps({"output_root": str(args.output_root.resolve()), "files": hashes}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
