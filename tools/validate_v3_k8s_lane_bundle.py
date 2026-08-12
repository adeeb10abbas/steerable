#!/usr/bin/env python3
"""Fail-closed validation for the reusable V3 Kubernetes lane bundle.

The bundle deliberately has a small, inspectable surface: one immutable
ConfigMap, one policy Job, one simulator Job, a kustomization, and startup
scripts.  Kubernetes' own client-side decoder is used instead of adding a
second YAML implementation to the study environment.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "deploy/k8s/v3_lane_bundle"
REQUIRED_FILES = (
    "configmap.yaml",
    "policy-job.yaml",
    "policy-service.yaml",
    "simulator-job.yaml",
    "kustomization.yaml",
)
REQUIRED_CACHE_ENV = (
    "HOME",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_RUNTIME_DIR",
    "WARP_CACHE_PATH",
    "MPLCONFIGDIR",
    "TMPDIR",
)
SENSITIVE_ENV_RE = re.compile(r"(?:SECRET|TOKEN|PASSWORD|PASSWD|PRIVATE_KEY|API_KEY|ACCESS_KEY)", re.IGNORECASE)
REQUIRED_RUNTIME_ENV = (
    "VK_ICD_FILENAMES",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "FFMPEG_BIN",
    "PYTHON_BIN",
)
SHA256_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")
UNRESOLVED_RE = re.compile(r"(?:REPLACE[_ -]?ME|TODO|FIXME|<[^>]+>)", re.IGNORECASE)
BACKGROUND_RE = re.compile(
    r"(?:\bkubectl\s+exec\b|\bnohup\b|\bdisown\b|\bsleep\s+infinity\b|"
    r"\btail\s+-f\s+/dev/null\b|(?:^|[;&|]\s*)[^#\n]*&\s*(?:#.*)?$)",
    re.MULTILINE,
)


class LaneBundleValidationError(ValueError):
    """Raised when a bundle cannot prove every required invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LaneBundleValidationError(message)


def _decode_with_kubectl(path: Path) -> dict[str, Any]:
    kubectl = os.environ.get("KUBECTL") or shutil.which("kubectl")
    _require(bool(kubectl), "kubectl is required for fail-closed YAML decoding")
    completed = subprocess.run(
        [
            str(kubectl),
            "create",
            "--dry-run=client",
            "--validate=false",
            "-f",
            str(path),
            "-o",
            "json",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    _require(
        completed.returncode == 0,
        f"kubectl could not decode {path.name}: {completed.stderr.strip()}",
    )
    try:
        decoded = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise LaneBundleValidationError(f"kubectl emitted invalid JSON for {path.name}") from exc
    _require(isinstance(decoded, dict), f"{path.name} did not decode to an object")
    _require(decoded.get("kind") != "List", f"{path.name} must contain exactly one object")
    return decoded


def _mapping(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be a mapping")
    return value


def _list(value: Any, label: str) -> list[Any]:
    _require(isinstance(value, list), f"{label} must be a list")
    return value


def _env_map(container: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    rows = _list(container.get("env"), f"{label} env")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = _mapping(row, f"{label} env row")
        name = entry.get("name")
        _require(isinstance(name, str) and name, f"{label} env row lacks a name")
        _require(name not in result, f"{label} repeats env {name}")
        result[name] = entry
    return result


def _quantity_is_one(value: Any) -> bool:
    return value == 1 or value == "1"


def _is_absolute_path(value: Any) -> bool:
    return isinstance(value, str) and Path(value).is_absolute()


def _configmap_references(pod_spec: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    containers = list(pod_spec.get("containers") or []) + list(pod_spec.get("initContainers") or [])
    for raw_container in containers:
        container = _mapping(raw_container, "container")
        for raw in container.get("envFrom") or []:
            ref = _mapping(raw, "envFrom row").get("configMapRef")
            if isinstance(ref, dict) and isinstance(ref.get("name"), str):
                names.add(ref["name"])
        for raw in container.get("env") or []:
            row = _mapping(raw, "env row")
            value_from = row.get("valueFrom")
            if isinstance(value_from, dict):
                ref = value_from.get("configMapKeyRef")
                if isinstance(ref, dict) and isinstance(ref.get("name"), str):
                    names.add(ref["name"])
    for raw in pod_spec.get("volumes") or []:
        volume = _mapping(raw, "volume")
        ref = volume.get("configMap")
        if isinstance(ref, dict) and isinstance(ref.get("name"), str):
            names.add(ref["name"])
    return names


def _mount_covers(path: str, mounts: dict[str, str], empty_dir_names: set[str]) -> bool:
    candidate = Path(path)
    if not candidate.is_absolute():
        return False
    for name, mount_path in mounts.items():
        if name not in empty_dir_names:
            continue
        base = Path(mount_path)
        try:
            candidate.relative_to(base)
        except ValueError:
            continue
        return True
    return False


def _flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _flatten_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _flatten_strings(child)


def _validate_launch_json(data: dict[str, str]) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        if not key.lower().endswith(".json"):
            continue
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise LaneBundleValidationError(f"ConfigMap JSON {key} is invalid") from exc
        if isinstance(decoded, dict) and decoded.get("role") in {"policy", "simulator"}:
            role = decoded["role"]
            _require(role not in documents, f"ConfigMap repeats {role} launch JSON")
            documents[role] = decoded
    _require(set(documents) == {"policy", "simulator"}, "ConfigMap must contain policy and simulator launch JSON")
    for role, document in documents.items():
        experiment_argv = _list(document.get("experiment_argv"), f"{role} experiment_argv")
        _require(experiment_argv and _is_absolute_path(experiment_argv[0]), f"{role} argv[0] must be an exact absolute path")
        for key in ("checkpoint_path",):
            _require(_is_absolute_path(document.get(key)), f"{role} {key} must be an exact absolute path")
        _require(isinstance(document.get("expected_gpu_name"), str) and document["expected_gpu_name"], f"{role} expected_gpu_name missing")
        _require(re.fullmatch(r"\d+\.\d+\.\d+", str(document.get("expected_driver_version"))) is not None, f"{role} expected_driver_version missing")
        if role == "simulator":
            _require(
                document.get("vulkan_contract") == "isaac_app_launcher_rtx_frame_under_bound_vk_icd",
                "simulator Vulkan proof contract differs",
            )
            render_argv = _list(document.get("render_probe_argv"), "simulator render_probe_argv")
            _require(render_argv and _is_absolute_path(render_argv[0]), "simulator render probe argv[0] must be absolute")
        bindings = _list(document.get("file_bindings"), f"{role} file_bindings")
        _require(bindings, f"{role} file_bindings must be nonempty")
        for index, raw in enumerate(bindings):
            binding = _mapping(raw, f"{role} file_bindings[{index}]")
            _require(_is_absolute_path(binding.get("path")), f"{role} file binding path must be exact and absolute")
            _require(
                isinstance(binding.get("sha256"), str) and SHA256_RE.fullmatch(binding["sha256"]),
                f"{role} file binding lacks an exact SHA-256",
            )
            _require(
                isinstance(binding.get("bytes"), int) and binding["bytes"] >= 0,
                f"{role} file binding lacks an exact byte count",
            )
    readiness_contract = documents["policy"].get("readiness_contract")
    allowed_readiness = {
        "http_healthz_after_checkpoint_load": "http_healthz",
        "metadata_jsonl_after_checkpoint_load": "metadata_jsonl",
    }
    _require(readiness_contract in allowed_readiness, "policy launch JSON has an unsupported readiness contract")
    policy_wait = _mapping(documents["simulator"].get("policy_wait"), "simulator policy_wait")
    _require(
        policy_wait.get("mode") == allowed_readiness[readiness_contract],
        "simulator policy_wait mode differs from the policy readiness contract",
    )
    service_identity = _mapping(policy_wait.get("service_identity"), "simulator policy_wait.service_identity")
    _require(service_identity, "simulator policy_wait must bind unique Service identity")
    identity_keys = "\n".join(service_identity).lower()
    for token in ("lane", "attempt", "config"):
        _require(token in identity_keys, f"simulator policy_wait Service identity lacks {token}")
    _require(service_identity.get("v3-lane-role") == "policy", "simulator policy_wait does not bind policy role")
    return documents


def _validate_configmap(configmap: dict[str, Any]) -> tuple[str, dict[str, str], dict[str, dict[str, Any]]]:
    _require(configmap.get("apiVersion") == "v1", "configmap apiVersion must be v1")
    _require(configmap.get("kind") == "ConfigMap", "configmap.yaml must be a ConfigMap")
    _require(configmap.get("immutable") is True, "launch ConfigMap must set immutable: true")
    metadata = _mapping(configmap.get("metadata"), "ConfigMap metadata")
    name = metadata.get("name")
    _require(isinstance(name, str) and name, "ConfigMap metadata.name is missing")
    raw_data = _mapping(configmap.get("data"), "ConfigMap data")
    data: dict[str, str] = {}
    for key, value in raw_data.items():
        _require(isinstance(key, str) and isinstance(value, str), "ConfigMap data must be strings")
        _require(not UNRESOLVED_RE.search(value), f"ConfigMap key {key} contains an unresolved placeholder")
        data[key] = value

    all_data = "\n".join(f"{key}\n{value}" for key, value in data.items())
    lowered_data = all_data.lower()
    _require("argv" in lowered_data, "immutable ConfigMap must bind complete launch argv")
    _require("policy" in lowered_data and "simulat" in lowered_data, "ConfigMap argv does not cover both lane roles")
    digests = SHA256_RE.findall(all_data)
    _require(digests, "immutable ConfigMap must contain literal SHA-256 launch bindings")
    _require(
        "checkpoint" in lowered_data,
        "immutable ConfigMap must bind the checkpoint",
    )
    return name, data, _validate_launch_json(data)


def _validate_job(
    job: dict[str, Any],
    *,
    role: str,
    configmap_name: str,
    bundle_source: str,
) -> dict[str, Any]:
    expected_gpu = "NVIDIA-A40" if role == "simulator" else "NVIDIA-A100-SXM4-80GB"
    _require(job.get("apiVersion") == "batch/v1", f"{role} apiVersion must be batch/v1")
    _require(job.get("kind") == "Job", f"{role} workload must be a fresh Job")
    spec = _mapping(job.get("spec"), f"{role} Job spec")
    _require(spec.get("backoffLimit") == 0, f"{role} Job must set backoffLimit: 0")
    _require(spec.get("completions") == 1, f"{role} Job must set completions: 1")
    _require(spec.get("parallelism") == 1, f"{role} Job must set parallelism: 1")
    _require("ttlSecondsAfterFinished" not in spec, f"{role} Job must preserve completed/failed evidence")
    template = _mapping(spec.get("template"), f"{role} template")
    pod_spec = _mapping(template.get("spec"), f"{role} pod spec")
    _require(pod_spec.get("restartPolicy") == "Never", f"{role} Job must set restartPolicy: Never")
    _require(
        isinstance(pod_spec.get("terminationGracePeriodSeconds"), int)
        and pod_spec["terminationGracePeriodSeconds"] >= 120,
        f"{role} Job needs at least 120 seconds termination grace",
    )
    selectors = _mapping(pod_spec.get("nodeSelector"), f"{role} nodeSelector")
    _require(
        selectors.get("nvidia.com/gpu.product") == expected_gpu,
        f"{role} must select exact GPU product {expected_gpu}",
    )
    _require(
        "node-role.kubernetes.io/worker-gpu" in selectors,
        f"{role} must select a GPU worker node",
    )

    containers = _list(pod_spec.get("containers"), f"{role} containers")
    _require(len(containers) == 1, f"{role} Job must have exactly one non-sidecar experiment container")
    container = _mapping(containers[0], f"{role} container")
    image = container.get("image")
    _require(
        isinstance(image, str) and re.search(r"@sha256:[0-9a-f]{64}$", image),
        f"{role} image must be pinned by SHA-256 digest",
    )
    image_digest = "sha256:" + str(image).rsplit("@sha256:", 1)[1]
    template_metadata = _mapping(template.get("metadata"), f"{role} template metadata")
    pod_labels = _mapping(template_metadata.get("labels"), f"{role} pod labels")
    _require(pod_labels.get("v3-lane-role") == role, f"{role} pod lacks exact role label")
    job_labels = _mapping(_mapping(job.get("metadata"), f"{role} metadata").get("labels"), f"{role} Job labels")
    _require(job_labels.get("v3-lane-role") == role, f"{role} Job lacks exact role label")
    annotations = _mapping(template_metadata.get("annotations"), f"{role} pod annotations")
    _require(image_digest in annotations.values(), f"{role} pod annotation does not bind the exact image digest")
    resources = _mapping(container.get("resources"), f"{role} resources")
    requests = _mapping(resources.get("requests"), f"{role} resource requests")
    limits = _mapping(resources.get("limits"), f"{role} resource limits")
    _require(
        _quantity_is_one(requests.get("nvidia.com/gpu"))
        and _quantity_is_one(limits.get("nvidia.com/gpu")),
        f"{role} must request and limit exactly one GPU",
    )
    command = _list(container.get("command"), f"{role} command")
    _require(command and isinstance(command[0], str), f"{role} command must be explicit")
    forbidden_pid1 = {"sh", "bash", "/bin/sh", "/bin/bash", "env", "/usr/bin/env", "tini", "/usr/bin/tini"}
    _require(command[0] not in forbidden_pid1, f"{role} experiment must become PID 1, not {command[0]}")
    command_text = " ".join(str(part) for part in command + list(container.get("args") or []))
    _require(not BACKGROUND_RE.search(command_text), f"{role} command launches a background/stale process")
    _require("lane_entrypoint.py" in command_text, f"{role} must start through the audited lane_entrypoint.py")
    entrypoints = [part for part in command if isinstance(part, str) and Path(part).name == "lane_entrypoint.py"]
    _require(
        len(entrypoints) == 1 and _is_absolute_path(entrypoints[0]),
        f"{role} command must contain one exact absolute lane_entrypoint.py path",
    )

    lifecycle = _mapping(container.get("lifecycle"), f"{role} lifecycle")
    pre_stop = _mapping(lifecycle.get("preStop"), f"{role} lifecycle.preStop")
    _require(pre_stop.get("exec") or pre_stop.get("httpGet"), f"{role} lacks a graceful preStop hook")

    env = _env_map(container, role)
    for name, row in env.items():
        if SENSITIVE_ENV_RE.search(name):
            _require("value" not in row, f"{role} embeds literal sensitive env {name}")
    pod_uid = _mapping(env.get("POD_UID"), f"{role} POD_UID")
    value_from = _mapping(pod_uid.get("valueFrom"), f"{role} POD_UID valueFrom")
    field_ref = _mapping(value_from.get("fieldRef"), f"{role} POD_UID fieldRef")
    _require(field_ref.get("fieldPath") == "metadata.uid", f"{role} POD_UID must use metadata.uid")
    for name, field_path in (("POD_NAME", "metadata.name"), ("POD_NAMESPACE", "metadata.namespace")):
        row = _mapping(env.get(name), f"{role} {name}")
        ref = _mapping(_mapping(row.get("valueFrom"), f"{role} {name} valueFrom").get("fieldRef"), f"{role} {name} fieldRef")
        _require(ref.get("fieldPath") == field_path, f"{role} {name} must use {field_path}")
    pod_ip = _mapping(env.get("POD_IP"), f"{role} POD_IP")
    pod_ip_ref = _mapping(_mapping(pod_ip.get("valueFrom"), f"{role} POD_IP valueFrom").get("fieldRef"), f"{role} POD_IP fieldRef")
    _require(pod_ip_ref.get("fieldPath") == "status.podIP", f"{role} POD_IP must use status.podIP")
    for name in ("LANE_ID", "ATTEMPT_ID"):
        _require(name in env and isinstance(env[name].get("value"), str) and env[name]["value"], f"{role} lacks {name}")
    image_env = _mapping(env.get("IMAGE_DIGEST_EXPECTED"), f"{role} IMAGE_DIGEST_EXPECTED")
    image_value = image_env.get("value")
    if image_value is not None:
        _require(image_value == image_digest, f"{role} IMAGE_DIGEST_EXPECTED differs from its image")
    else:
        image_ref = _mapping(_mapping(image_env.get("valueFrom"), f"{role} image valueFrom").get("configMapKeyRef"), f"{role} image ConfigMap ref")
        _require(image_ref.get("name") == configmap_name, f"{role} IMAGE_DIGEST_EXPECTED is not ConfigMap-bound")
    _require(
        "OUTPUT_PARENT" in env and isinstance(env["OUTPUT_PARENT"].get("value"), str),
        f"{role} lacks a fixed writable OUTPUT_PARENT",
    )

    volumes = _list(pod_spec.get("volumes"), f"{role} volumes")
    empty_dir_names = {
        str(volume.get("name"))
        for raw in volumes
        for volume in [_mapping(raw, f"{role} volume")]
        if isinstance(volume.get("emptyDir"), dict)
    }
    _require(empty_dir_names, f"{role} must mount fresh emptyDir caches")
    mounts = {
        str(mount.get("name")): str(mount.get("mountPath"))
        for raw in _list(container.get("volumeMounts"), f"{role} volumeMounts")
        for mount in [_mapping(raw, f"{role} volumeMount")]
    }
    for name in REQUIRED_CACHE_ENV:
        row = env.get(name)
        _require(row is not None and isinstance(row.get("value"), str), f"{role} lacks fixed {name}")
        _require(
            _mount_covers(row["value"], mounts, empty_dir_names),
            f"{role} {name} is not backed by a fresh emptyDir",
        )
    for name in REQUIRED_RUNTIME_ENV:
        row = env.get(name)
        _require(row is not None and isinstance(row.get("value"), str) and row["value"], f"{role} lacks {name}")
    for name in ("PYTHON_BIN", "FFMPEG_BIN", "VK_ICD_FILENAMES"):
        _require(_is_absolute_path(env[name]["value"]), f"{role} {name} must be an exact absolute path")
    for name in ("LD_LIBRARY_PATH", "PYTHONPATH"):
        components = env[name]["value"].split(":")
        _require(
            components and all(_is_absolute_path(component) for component in components),
            f"{role} {name} must contain only exact absolute paths",
        )
    for name in ("DISPLAY", "LD_PRELOAD"):
        _require(name in env and env[name].get("value") == "", f"{role} must explicitly clear {name}")
    for name in ("PYTHONNOUSERSITE", "PYTHONUNBUFFERED"):
        _require(name in env and env[name].get("value") == "1", f"{role} must set {name}=1")
    prestop_wait = env.get("PRESTOP_WAIT_SECONDS", {}).get("value")
    _require(
        isinstance(prestop_wait, str) and prestop_wait.isdigit() and 1 <= int(prestop_wait) <= 240,
        f"{role} must bind PRESTOP_WAIT_SECONDS in [1, 240]",
    )
    capabilities = env.get("NVIDIA_DRIVER_CAPABILITIES", {}).get("value", "")
    _require(
        all(token in capabilities.split(",") for token in ("compute", "graphics", "utility")),
        f"{role} lacks compute/graphics/utility NVIDIA capabilities",
    )

    references = _configmap_references(pod_spec)
    _require(configmap_name in references, f"{role} Job does not reference immutable launch ConfigMap")
    _require(not pod_spec.get("initContainers"), f"{role} preflight must run synchronously in the experiment container")

    if role == "policy":
        readiness = _mapping(container.get("readinessProbe"), "policy readinessProbe")
        readiness_exec = _mapping(readiness.get("exec"), "policy readinessProbe.exec")
        readiness_text = " ".join(_flatten_strings(readiness_exec)).lower()
        _require("ready" in readiness_text, "policy readiness probe is not checkpoint-readiness based")
        _require(
            "checkpoint" in readiness_text or "loaded" in readiness_text or "policy-ready" in readiness_text,
            "policy readiness probe does not prove checkpoint load",
        )
        _require("http_healthz" in readiness_text, "policy readiness probe must perform HTTP /healthz")
        _require(" tcp" not in f" {readiness_text}", "policy readiness probe must not use raw TCP")

    # These are source-wide by design: scripts are mounted into both Jobs from
    # one immutable bundle, so either role must be covered by the same audit.
    _require("preflight" in bundle_source.lower(), "bundle lacks a startup preflight")
    return {
        "name": _mapping(job.get("metadata"), f"{role} metadata").get("name"),
        "gpu_product": expected_gpu,
        "image": image,
        "image_digest": image_digest,
        "pod_labels": pod_labels,
        "policy_port": env.get("POLICY_PORT", {}).get("value"),
        "configmap_referenced": True,
        "entrypoint": entrypoints[0],
    }


def _validate_service(service: dict[str, Any], policy_job: dict[str, Any]) -> dict[str, Any]:
    _require(service.get("apiVersion") == "v1", "policy Service apiVersion must be v1")
    _require(service.get("kind") == "Service", "policy-service.yaml must be a Service")
    spec = _mapping(service.get("spec"), "policy Service spec")
    _require(spec.get("publishNotReadyAddresses") is False, "policy Service must explicitly keep publishNotReadyAddresses false")
    selector = _mapping(spec.get("selector"), "policy Service selector")
    selector_keys = "\n".join(selector).lower()
    _require("lane" in selector_keys, "policy Service selector lacks lane identity")
    _require("attempt" in selector_keys, "policy Service selector lacks attempt identity")
    _require(
        "config" in selector_keys and ("sha" in selector_keys or "hash" in selector_keys),
        "policy Service selector lacks immutable config-hash identity",
    )
    _require(selector.get("v3-lane-role") == "policy", "policy Service selector lacks exact policy role")
    labels = _mapping(policy_job.get("pod_labels"), "policy Job labels")
    for key, value in selector.items():
        _require(labels.get(key) == value, f"policy Service selector {key} does not match the policy Job")
    ports = _list(spec.get("ports"), "policy Service ports")
    _require(len(ports) == 1, "policy Service must expose exactly one unique policy port")
    port = _mapping(ports[0], "policy Service port")
    target = port.get("targetPort")
    policy_port = policy_job.get("policy_port")
    _require(str(target) == str(policy_port), "policy Service targetPort differs from POLICY_PORT")
    _require(port.get("port") == int(str(policy_port)), "policy Service port differs from POLICY_PORT")
    return {
        "name": _mapping(service.get("metadata"), "policy Service metadata").get("name"),
        "selector": selector,
        "port": port.get("port"),
    }


def _validate_scripts(root: Path, source: str) -> dict[str, Any]:
    script_root = root / "scripts"
    _require(script_root.is_dir(), "scripts directory is missing")
    scripts = sorted(path for path in script_root.rglob("*") if path.is_file() and path.suffix in {".py", ".sh"})
    _require(scripts, "scripts directory is empty")
    shell_scripts = [path for path in scripts if path.suffix == ".sh"]
    python_scripts = [path for path in scripts if path.suffix == ".py"]
    _require(shell_scripts or python_scripts, "bundle must include executable entrypoint/preflight scripts")
    for path in shell_scripts:
        body = path.read_text(encoding="utf-8")
        _require("set -euo pipefail" in body, f"{path.name} must use set -euo pipefail")
        _require(not BACKGROUND_RE.search(body), f"{path.name} launches a background/stale process")

    lowered = source.lower()
    checks = {
        "cuda_kernel": ("torch.cuda", "cuda kernel"),
        "vulkan": ("isaac_app_launcher_rtx_frame_under_bound_vk_icd",),
        "rendered_frame": ("rendered_frame", "rendered frame", "frame.png"),
        "curobo_import": ("import curobo", "curobo"),
        "ffmpeg_encode": ("ffmpeg_encode", "encode.mp4", "encoded.mp4"),
        "ffmpeg_decode": ("ffmpeg_decode", "decoded", "decode.raw"),
        "checkpoint_digest": ("checkpoint_sha256", "sha256sum"),
        "writable_parent": ("writable_parent", "test -w", "os.access"),
        "unique_port": ("port_lock", "flock", "o_excl"),
    }
    for label, alternatives in checks.items():
        _require(any(token in lowered for token in alternatives), f"preflight lacks {label} proof")

    for field in ("effective_environment", "gpu_uuid", "image_digest", "argv", "checkpoint_sha256"):
        _require(field in lowered, f"startup evidence does not record {field}")
    _require("cuda:0" in lowered, "preflight does not bind the container-local cuda:0 device")
    _require(not re.search(r"/dev/nvidia[1-9]\d*", source), "bundle assumes a host GPU index")
    _require(
        "dict(os.environ)" not in source and "os.environ.copy()" not in source,
        "startup evidence must not dump the full environment or secrets",
    )
    _require(all(token in source for token in ("POD_UID", "LANE_ID", "ATTEMPT_ID")), "startup scripts do not bind POD_UID/lane/attempt")
    _require(
        ("lane-" in source and "attempt-" in source) or all(token in source for token in ("lane_id", "attempt_id")),
        "output derivation does not include lane and attempt components",
    )
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        _require(
            not (re.search(r"\bmkdir\b", line) and re.search(r"episode", line, re.I)),
            "bundle pre-creates a write-once episode directory",
        )
    has_exec = any(
        re.search(r"^\s*exec\s+", path.read_text(encoding="utf-8"), re.MULTILINE)
        for path in shell_scripts
    ) or any(
        re.search(r"\bos\.(?:execv|execve|execvp|execvpe)\s*\(", path.read_text(encoding="utf-8"))
        for path in python_scripts
    )
    _require(has_exec, "entrypoint scripts never exec the experiment as PID 1")
    _require(
        any(token in lowered for token in ("wait_for_policy", "policy_service", "policy readiness"))
        and any(token in lowered for token in ("policy metadata", "policy_metadata", "runtime-startup.json")),
        "simulator startup does not wait for policy readiness and metadata",
    )
    _require("image_digest_expected" in lowered, "entrypoint does not require IMAGE_DIGEST_EXPECTED")
    for token in (
        "PRESTOP_WAIT_SECONDS",
        "os.kill(1, signal.SIGINT)",
        "os.kill(1, 0)",
        "ProcessLookupError",
        "time.monotonic()",
    ):
        _require(token in source, f"policy preStop lacks bounded PID-1 shutdown logic: {token}")
    _require(
        "def acquire_attempt_lock" in lowered and "os.o_excl" in lowered,
        "attempt lock is not created exclusively",
    )

    entrypoint = script_root / "lane_entrypoint.py"
    _require(entrypoint.is_file(), "scripts/lane_entrypoint.py is missing")
    try:
        tree = ast.parse(entrypoint.read_text(encoding="utf-8"), filename=str(entrypoint))
    except SyntaxError as exc:
        raise LaneBundleValidationError("lane_entrypoint.py is not valid Python") from exc
    preflight_calls: list[int] = []
    exec_calls: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            continue
        if "preflight" in name.lower():
            preflight_calls.append(node.lineno)
        if name in {"execv", "execve", "execvp", "execvpe"}:
            exec_calls.append(node.lineno)
    _require(preflight_calls and exec_calls, "lane_entrypoint must call preflight then os.exec*")
    _require(max(preflight_calls) < min(exec_calls), "lane_entrypoint execs before its synchronous preflight")
    return {
        "files": len(scripts),
        "shell_scripts": len(shell_scripts),
        "python_scripts": len(python_scripts),
    }


def _validate_bound_runtime_inputs(
    documents: dict[str, dict[str, Any]], jobs: dict[str, dict[str, Any]]
) -> None:
    entrypoints = {str(summary["entrypoint"]) for summary in jobs.values()}
    _require(len(entrypoints) == 1, "policy and simulator Jobs must use one common exact entrypoint")
    entrypoint = next(iter(entrypoints))
    entrypoint_path = Path(entrypoint)
    for role, document in documents.items():
        binding_paths = {
            str(_mapping(raw, f"{role} file binding")["path"])
            for raw in _list(document.get("file_bindings"), f"{role} file_bindings")
        }
        required = {
            entrypoint,
            str(entrypoint_path.with_name("startup_preflight.py")),
            str(document["checkpoint_path"]),
            str(
                entrypoint_path.with_name(
                    "check_policy_ready.py" if role == "policy" else "isaac_render_probe.py"
                )
            ),
        }
        argv_items = list(_list(document.get("experiment_argv"), f"{role} experiment_argv"))
        if role == "simulator":
            argv_items += list(_list(document.get("render_probe_argv"), "simulator render_probe_argv"))
        required.update(
            item
            for item in argv_items
            if isinstance(item, str) and Path(item).is_absolute() and Path(item).suffix == ".py"
        )
        missing = sorted(required - binding_paths)
        _require(not missing, f"{role} file_bindings omit required runtime inputs: {missing}")
        if role == "policy" and document.get("readiness_contract") == "http_healthz_after_checkpoint_load":
            argv = _list(document.get("experiment_argv"), "policy experiment_argv")
            root_flags = [index for index, item in enumerate(argv[:-1]) if item == "--openpi-root"]
            _require(len(root_flags) == 1, "HTTP health readiness requires one exact --openpi-root")
            openpi_root = argv[root_flags[0] + 1]
            _require(_is_absolute_path(openpi_root), "policy --openpi-root must be exact and absolute")
            health_server = str(Path(openpi_root) / "src/openpi/serving/websocket_policy_server.py")
            _require(
                health_server in binding_paths,
                f"policy file_bindings omit HTTP health server semantics: {health_server}",
            )


def _validate_kustomization(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    _require(re.search(r"^kind:\s*Kustomization\s*$", text, re.MULTILINE) is not None, "invalid kustomization kind")
    for required in REQUIRED_FILES[:-1]:
        _require(required in text, f"kustomization does not include {required}")


def _validate_generated_from_spec(root: Path, spec_path: Path | None = None) -> None:
    spec_path = (spec_path or (root / "spec.example.json")).resolve()
    _require(spec_path.is_file(), "bundle is missing spec.example.json for reproducibility validation")
    renderer = REPO_ROOT / "tools/render_v3_k8s_lane_bundle.py"
    _require(renderer.is_file(), "lane-bundle renderer is missing")
    with tempfile.TemporaryDirectory(prefix="v3-lane-render-") as temporary:
        generated = Path(temporary) / "bundle"
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, str(renderer), "--spec", str(spec_path), "--output-root", str(generated)],
            check=False,
            text=True,
            capture_output=True,
            env=environment,
        )
        _require(completed.returncode == 0, f"fresh lane render failed: {completed.stderr.strip()}")
        for relative in REQUIRED_FILES:
            expected = generated / relative
            actual = root / relative
            _require(
                actual.read_bytes() == expected.read_bytes(),
                f"{relative} is stale or hand-edited; it differs from a fresh spec render",
            )
        expected_scripts = sorted(path.name for path in (generated / "scripts").glob("*.py"))
        actual_scripts = sorted(path.name for path in (root / "scripts").glob("*.py"))
        _require(actual_scripts == expected_scripts, "runtime script set differs from a fresh spec render")
        for name in expected_scripts:
            _require(
                (root / "scripts" / name).read_bytes() == (generated / "scripts" / name).read_bytes(),
                f"scripts/{name} is stale or hand-edited; it differs from a fresh spec render",
            )


def validate(
    root: Path = DEFAULT_ROOT,
    *,
    verify_generated: bool = True,
    spec_path: Path | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    _require(root.is_dir(), f"lane bundle directory is missing: {root}")
    for relative in REQUIRED_FILES:
        _require((root / relative).is_file(), f"lane bundle is missing {relative}")
    if verify_generated:
        _validate_generated_from_spec(root, spec_path)
    _validate_kustomization(root / "kustomization.yaml")

    source_files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix in {".yaml", ".yml", ".sh", ".py", ".json"}
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_files)
    _require(not re.search(r"^\s*kind:\s*Pod\s*$", source, re.MULTILINE), "standalone Pods are forbidden; use fresh Jobs")
    _require(not BACKGROUND_RE.search(source), "bundle contains kubectl exec or a background/stale-process launcher")

    configmap = _decode_with_kubectl(root / "configmap.yaml")
    policy = _decode_with_kubectl(root / "policy-job.yaml")
    service = _decode_with_kubectl(root / "policy-service.yaml")
    simulator = _decode_with_kubectl(root / "simulator-job.yaml")
    configmap_name, config_data, launch_documents = _validate_configmap(configmap)
    jobs = {
        "policy": _validate_job(policy, role="policy", configmap_name=configmap_name, bundle_source=source),
        "simulator": _validate_job(simulator, role="simulator", configmap_name=configmap_name, bundle_source=source),
    }
    _require(jobs["policy"]["name"] != jobs["simulator"]["name"], "policy and simulator Job names collide")
    image_config_values = [value for key, value in config_data.items() if "image" in key.lower() and "digest" in key.lower()]
    _require(image_config_values, "ConfigMap lacks an exact image digest binding")
    for role, summary in jobs.items():
        _require(summary["image_digest"] in image_config_values, f"ConfigMap image digest differs from {role} Job")
    service_summary = _validate_service(service, jobs["policy"])
    _require(
        any(jobs["simulator"]["pod_labels"].get(key) != value for key, value in service_summary["selector"].items()),
        "policy Service selector also matches the simulator Job",
    )
    service_identity = _mapping(
        _mapping(launch_documents["simulator"].get("policy_wait"), "simulator policy_wait").get("service_identity"),
        "simulator policy_wait.service_identity",
    )
    for key, value in service_summary["selector"].items():
        _require(service_identity.get(key) == value, f"simulator policy_wait does not bind Service selector {key}")
    _require(service_identity.get("service_name") == service_summary["name"], "simulator policy_wait Service name differs")
    _validate_bound_runtime_inputs(launch_documents, jobs)
    script_summary = _validate_scripts(root, source)
    return {
        "status": "valid_v3_k8s_lane_bundle",
        "root": str(root),
        "configmap": configmap_name,
        "config_keys": sorted(config_data),
        "launch_roles": sorted(launch_documents),
        "jobs": jobs,
        "policy_service": service_summary,
        "scripts": script_summary,
        "generated_from_spec": verify_generated,
        "render_spec": str((spec_path or (root / "spec.example.json")).resolve()) if verify_generated else None,
        "checks": 32,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--spec", type=Path, help="original render spec (required for a relocated rendered bundle)")
    args = parser.parse_args()
    try:
        print(json.dumps(validate(args.root, spec_path=args.spec), indent=2, sort_keys=True))
    except LaneBundleValidationError as exc:
        raise SystemExit(f"V3 Kubernetes lane-bundle validation failed: {exc}") from exc


if __name__ == "__main__":
    main()
