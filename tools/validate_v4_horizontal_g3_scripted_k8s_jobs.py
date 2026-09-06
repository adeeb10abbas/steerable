#!/usr/bin/env python3
"""Validate a rendered simulator-only V4 horizontal G3 scripted Kubernetes bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SCRIPTED_MODES = ("stationary", "moving")
AUTHORIZATION_AUTHORIZED = "authorized_by_passing_path_scale_receipt"


class G3ScriptedBundleValidationError(ValueError):
    """Raised when a G3 scripted bundle could run policy inference or lacks evidence binding."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G3ScriptedBundleValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scale_label(scale: float) -> str:
    value = float(scale)
    if value.is_integer():
        return f"{int(value)}.0"
    return str(value)


def _decode_bundle(root: Path) -> list[dict[str, Any]]:
    kubectl = os.environ.get("KUBECTL") or shutil.which("kubectl")
    require(bool(kubectl), "kubectl is required for fail-closed YAML decoding")
    completed = subprocess.run(
        [
            str(kubectl),
            "create",
            "--dry-run=client",
            "--validate=false",
            "-k",
            str(root),
            "-o",
            "json",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    require(
        completed.returncode == 0,
        f"kubectl could not decode G3 scripted bundle: {completed.stderr.strip()}",
    )
    decoder = json.JSONDecoder()
    items: list[dict[str, Any]] = []
    offset = 0
    while offset < len(completed.stdout):
        while offset < len(completed.stdout) and completed.stdout[offset].isspace():
            offset += 1
        if offset >= len(completed.stdout):
            break
        payload, offset = decoder.raw_decode(completed.stdout, offset)
        if isinstance(payload, dict) and payload.get("kind") == "List":
            nested = payload.get("items")
            require(isinstance(nested, list), "kubectl G3 scripted List lacks items")
            items.extend(nested)
        else:
            require(
                isinstance(payload, dict),
                "kubectl G3 scripted decode returned a non-object",
            )
            items.append(payload)
    require(items, "kubectl G3 scripted decode returned no objects")
    return items


def _env(container: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in container.get("env") or []:
        require(isinstance(row, dict) and isinstance(row.get("name"), str), "invalid env row")
        result[row["name"]] = row
    return result


def validate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "bundle-manifest.json"
    require(manifest_path.is_file(), "G3 scripted bundle manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(
        manifest.get("schema_version")
        == "vla-wam-v4-horizontal-g3-scripted-k8s-bundle-v1",
        "G3 scripted bundle manifest schema differs",
    )
    require(
        manifest.get("execution_scope") == "model_blind_g3_no_policy",
        "G3 scripted bundle execution scope differs",
    )
    require(
        manifest.get("model_request_count") == 0
        and manifest.get("behavioral_episode_count") == 0,
        "G3 scripted bundle manifest contains policy or behavioral work",
    )
    file_hashes = manifest.get("files_sha256")
    require(
        isinstance(file_hashes, dict) and file_hashes,
        "G3 scripted file hash index is missing",
    )
    for name, expected in file_hashes.items():
        path = root / name
        require(path.is_file(), f"G3 scripted indexed file is missing: {name}")
        require(sha256_file(path) == expected, f"G3 scripted indexed file hash changed: {name}")

    items = _decode_bundle(root)
    jobs = [item for item in items if item.get("kind") == "Job"]
    configmaps = [item for item in items if item.get("kind") == "ConfigMap"]
    job_count = int(manifest.get("job_count", 0))
    require(job_count == 10, "G3 scripted manifest job_count must be ten")
    scale = manifest.get("scale")
    require(
        isinstance(scale, (int, float))
        and not isinstance(scale, bool)
        and float(scale) > 0,
        "G3 scripted manifest lacks scale",
    )
    scale_value = float(scale)
    scale_text = _scale_label(scale_value)
    expected_jobs = manifest.get("scripted_jobs")
    require(
        isinstance(expected_jobs, list) and len(expected_jobs) == job_count,
        "G3 scripted manifest lacks scripted_jobs",
    )
    expected_seed_mode = {
        (int(row["environment_seed"]), str(row["mode"]))
        for row in expected_jobs
        if isinstance(row, dict)
        and type(row.get("environment_seed")) is int
        and row.get("mode") in SCRIPTED_MODES
    }
    require(len(expected_seed_mode) == job_count, "G3 scripted manifest job set is invalid")
    expected_study_commit = manifest.get("expected_study_commit")
    expected_robolab_commit = manifest.get("expected_robolab_commit")
    native_control_dt_s = manifest.get("native_control_dt_s")
    campaign_sha256 = manifest.get("campaign_sha256")
    plan_sha256 = manifest.get("plan_sha256")
    reset_registry_sha256 = manifest.get("reset_registry_sha256")
    path_scale_receipt_sha256 = manifest.get("path_scale_receipt_sha256")
    path_scale_receipt_path = manifest.get("path_scale_receipt_path")
    authorization_status = manifest.get("authorization_status")
    marker_wrapper_sha256 = manifest.get("marker_wrapper_sha256")
    runner_sha256 = manifest.get("runner_sha256")
    gate_core_sha256 = manifest.get("gate_core_sha256")
    attempt_id = manifest.get("attempt_id")
    require(isinstance(attempt_id, str) and attempt_id, "G3 scripted manifest lacks attempt_id")
    require(
        authorization_status == AUTHORIZATION_AUTHORIZED,
        "G3 scripted manifest lacks passing path-scale authorization",
    )
    require(
        isinstance(path_scale_receipt_path, str)
        and Path(path_scale_receipt_path).is_absolute(),
        "G3 scripted manifest lacks path-scale receipt path",
    )
    require(
        isinstance(path_scale_receipt_sha256, str)
        and len(path_scale_receipt_sha256) == 64
        and all(char in "0123456789abcdef" for char in path_scale_receipt_sha256),
        "G3 scripted manifest lacks path-scale receipt SHA-256",
    )
    require(
        isinstance(expected_study_commit, str) and len(expected_study_commit) == 40,
        "G3 scripted manifest lacks expected study commit",
    )
    require(
        isinstance(expected_robolab_commit, str)
        and len(expected_robolab_commit) == 40,
        "G3 scripted manifest lacks expected RoboLab commit",
    )
    require(
        isinstance(native_control_dt_s, (int, float))
        and not isinstance(native_control_dt_s, bool)
        and float(native_control_dt_s) > 0,
        "G3 scripted manifest lacks native control dt",
    )
    for label, value in (
        ("campaign", campaign_sha256),
        ("plan", plan_sha256),
        ("reset registry", reset_registry_sha256),
        ("marker wrapper", marker_wrapper_sha256),
        ("runner", runner_sha256),
        ("gate core", gate_core_sha256),
    ):
        require(
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value),
            f"G3 scripted manifest lacks the {label} SHA-256",
        )
    require(len(jobs) == job_count, "G3 scripted Job count differs from manifest")
    require(
        len(configmaps) == job_count + 1,
        "G3 scripted ConfigMap count must equal per-job launches plus runtime scripts",
    )
    require(
        not any(item.get("kind") == "Service" for item in items),
        "G3 scripted bundle must not create a policy Service",
    )

    launch_by_name: dict[str, dict[str, Any]] = {}
    scripts_maps = []
    for configmap in configmaps:
        metadata = configmap.get("metadata") or {}
        data = configmap.get("data") or {}
        require(configmap.get("immutable") is True, "all G3 scripted ConfigMaps must be immutable")
        name = metadata.get("name")
        if "simulator-launch.json" in data:
            launch = json.loads(data["simulator-launch.json"])
            require(
                launch.get("role") == "simulator"
                and launch.get("execution_scope") == "model_blind_g3_no_policy",
                "G3 scripted launch config is not simulator-only model-blind scope",
            )
            require(
                launch.get("policy_wait") is None,
                "G3 scripted launch must not wait for a policy",
            )
            argv = launch.get("experiment_argv")
            require(
                isinstance(argv, list)
                and len(argv) >= 12
                and str(argv[1]).endswith("run_v4_g3_scripted_checked.py")
                and argv[2] == "--expected-environment-seed"
                and argv[4] == "--expected-scale"
                and argv[6] == "--expected-mode"
                and argv[8] == "--"
                and argv[0] == argv[9]
                and str(argv[10]).endswith("run_v4_horizontal_g3_scripted_seed.py"),
                "G3 scripted launch does not invoke the scripted-seed runner through the marker-check wrapper",
            )
            file_bindings = launch.get("file_bindings") or []
            wrapper_bindings = [
                binding
                for binding in file_bindings
                if isinstance(binding, dict) and binding.get("path") == argv[1]
            ]
            require(
                len(wrapper_bindings) == 1
                and wrapper_bindings[0].get("sha256") == marker_wrapper_sha256,
                "G3 scripted marker-check wrapper binding differs from the manifest",
            )
            runner_bindings = [
                binding
                for binding in file_bindings
                if isinstance(binding, dict) and binding.get("path") == argv[10]
            ]
            require(
                len(runner_bindings) == 1
                and runner_bindings[0].get("sha256") == runner_sha256,
                "G3 scripted runner binding differs from the manifest",
            )
            gate_bindings = [
                binding
                for binding in file_bindings
                if isinstance(binding, dict)
                and str(binding.get("path", "")).endswith(
                    "experiments/online_correction_v4/model_blind_g3.py"
                )
            ]
            require(
                len(gate_bindings) == 1
                and gate_bindings[0].get("sha256") == gate_core_sha256,
                "G3 scripted gate-core binding differs from the manifest",
            )
            path_scale_bindings = [
                binding
                for binding in file_bindings
                if isinstance(binding, dict)
                and binding.get("path") == path_scale_receipt_path
            ]
            require(
                len(path_scale_bindings) == 1
                and path_scale_bindings[0].get("sha256") == path_scale_receipt_sha256,
                "G3 scripted path-scale receipt binding differs from the manifest",
            )
            require(
                launch.get("authorization_status") == AUTHORIZATION_AUTHORIZED,
                "G3 scripted launch lacks path-scale authorization",
            )
            require(
                launch.get("path_scale_receipt_path") == path_scale_receipt_path,
                "G3 scripted launch path-scale receipt path differs",
            )
            require(
                launch.get("path_scale_receipt_sha256") == path_scale_receipt_sha256,
                "G3 scripted launch path-scale receipt SHA-256 differs",
            )
            lowered = " ".join(str(item).lower() for item in argv)
            require(
                "--policy-host" not in lowered
                and "--policy-port" not in lowered
                and "serve_policy" not in lowered,
                "G3 scripted launch contains a policy endpoint or server",
            )
            require(
                launch.get("checkpoint_semantics") == "model_blind_g3_plan_candidate",
                "G3 scripted hash-bound artifact semantics differ",
            )
            require(
                launch.get("checkpoint_sha256") == plan_sha256,
                "G3 scripted launch plan SHA-256 differs from manifest",
            )
            option_values = {
                str(argv[index]): str(argv[index + 1])
                for index in range(len(argv) - 1)
                if str(argv[index]).startswith("--")
            }
            require(
                option_values.get("--expected-environment-seed")
                == option_values.get("--environment-seed"),
                "G3 scripted wrapper and child environment-seed bindings differ",
            )
            require(
                option_values.get("--expected-scale") == option_values.get("--scale")
                == scale_text,
                "G3 scripted wrapper and child scale bindings differ",
            )
            require(
                option_values.get("--expected-mode") == option_values.get("--mode"),
                "G3 scripted wrapper and child mode bindings differ",
            )
            require(
                option_values.get("--expected-study-commit")
                == expected_study_commit,
                "G3 scripted launch study commit binding differs",
            )
            require(
                option_values.get("--expected-robolab-commit")
                == expected_robolab_commit,
                "G3 scripted launch RoboLab commit binding differs",
            )
            require(
                option_values.get("--native-control-dt-s")
                == str(float(native_control_dt_s)),
                "G3 scripted launch native control dt binding differs",
            )
            require(
                option_values.get("--campaign-sha256") == campaign_sha256,
                "G3 scripted launch campaign SHA-256 differs",
            )
            require(
                option_values.get("--plan-sha256") == plan_sha256,
                "G3 scripted launch plan SHA-256 differs",
            )
            require(
                option_values.get("--reset-registry-sha256") == reset_registry_sha256,
                "G3 scripted launch reset registry SHA-256 differs",
            )
            digest_key = data.get("image.digest")
            require(
                isinstance(digest_key, str) and digest_key.startswith("sha256:"),
                "G3 scripted launch ConfigMap lacks image digest binding",
            )
            launch_by_name[str(name)] = launch
        else:
            scripts_maps.append(configmap)
    require(
        len(launch_by_name) == job_count,
        "G3 scripted per-job launch ConfigMaps are incomplete",
    )
    require(len(scripts_maps) == 1, "G3 scripted bundle must contain one scripts ConfigMap")
    scripts = scripts_maps[0].get("data") or {}
    for filename in (
        "lane_entrypoint.py",
        "startup_preflight.py",
        "check_policy_ready.py",
        "isaac_render_probe.py",
    ):
        require(filename in scripts, f"G3 scripted scripts ConfigMap lacks {filename}")
        source = ROOT / "deploy/k8s/v4_lane_bundle/scripts" / filename
        require(
            hashlib.sha256(scripts[filename].encode("utf-8")).hexdigest()
            == sha256_file(source),
            f"G3 scripted embedded runtime script differs: {filename}",
        )

    observed_jobs: set[tuple[int, str]] = set()
    observed_output_parents: set[str] = set()
    for job in jobs:
        metadata = job.get("metadata") or {}
        labels = metadata.get("labels") or {}
        require(
            labels.get("v4-gate") == "g3-horizontal-scripted",
            "G3 scripted Job gate label differs",
        )
        require(labels.get("v4-scale") == scale_text, "G3 scripted Job scale label differs")
        mode_label = labels.get("v4-scripted-mode")
        require(mode_label in SCRIPTED_MODES, "G3 scripted Job mode label differs")
        spec = job.get("spec") or {}
        require(spec.get("backoffLimit") == 0, "G3 scripted Jobs must not retry implicitly")
        pod_spec = ((spec.get("template") or {}).get("spec") or {})
        require(
            pod_spec.get("restartPolicy") == "Never",
            "G3 scripted restartPolicy must be Never",
        )
        node_selector = pod_spec.get("nodeSelector") or {}
        require(
            node_selector.get("nvidia.com/gpu.product") == manifest.get("gpu_product"),
            "G3 scripted Job GPU product affinity differs",
        )
        containers = pod_spec.get("containers") or []
        require(
            len(containers) == 1,
            "G3 scripted Job must contain exactly one simulator container",
        )
        container = containers[0]
        config_refs = {
            ((volume.get("configMap") or {}).get("name"))
            for volume in pod_spec.get("volumes") or []
            if isinstance(volume, dict)
        }
        matching = set(launch_by_name) & config_refs
        require(
            len(matching) == 1,
            "G3 scripted Job must reference one per-job launch ConfigMap",
        )
        launch_ref_name = matching.pop()
        launch = launch_by_name[str(launch_ref_name)]
        launch_digest = None
        for configmap in configmaps:
            if (configmap.get("metadata") or {}).get("name") == launch_ref_name:
                launch_digest = (configmap.get("data") or {}).get("image.digest")
                break
        require(
            isinstance(launch_digest, str) and launch_digest.startswith("sha256:"),
            "G3 scripted launch ConfigMap lacks image digest binding",
        )
        image = str(container.get("image") or "")
        require("@sha256:" in image, "G3 scripted Job image must be pinned by digest")
        require(
            image.endswith(launch_digest),
            "G3 scripted Job image digest differs from launch binding",
        )
        requests = (container.get("resources") or {}).get("requests") or {}
        limits = (container.get("resources") or {}).get("limits") or {}
        require(
            str(requests.get("nvidia.com/gpu")) == "1"
            and str(limits.get("nvidia.com/gpu")) == "1",
            "G3 scripted Job must request and limit exactly one GPU",
        )
        env = _env(container)
        require(
            (env.get("LANE_ROLE") or {}).get("value") == "simulator",
            "G3 scripted Job role must be simulator",
        )
        output_parent = (env.get("OUTPUT_PARENT") or {}).get("value")
        require(
            isinstance(output_parent, str) and output_parent.startswith("/data/"),
            "G3 scripted Job lacks absolute OUTPUT_PARENT",
        )
        require(
            f"/attempt-{attempt_id}/scale-{scale_text}/{mode_label}/seed-" in output_parent,
            "G3 scripted Job OUTPUT_PARENT lacks attempt/scale/mode/seed isolation",
        )
        require(
            output_parent not in observed_output_parents,
            "G3 scripted Jobs repeat an OUTPUT_PARENT",
        )
        observed_output_parents.add(output_parent)
        for pod_field in ("POD_UID", "POD_NAME", "POD_NAMESPACE", "POD_IP"):
            row = env.get(pod_field) or {}
            require("valueFrom" in row, f"G3 scripted Job lacks downward API env {pod_field}")
        mounts = container.get("volumeMounts") or []
        mount_names = {
            mount.get("name")
            for mount in mounts
            if isinstance(mount, dict) and isinstance(mount.get("name"), str)
        }
        require("data" in mount_names, "G3 scripted Job lacks PVC data mount")
        launch_ref = ((env.get("LANE_LAUNCH_CONFIG_SHA256") or {}).get("value"))
        require(
            isinstance(launch_ref, str) and len(launch_ref) == 64,
            "G3 scripted Job lacks launch SHA binding",
        )
        argv = launch["experiment_argv"]
        seed_positions = [
            index for index, value in enumerate(argv[:-1]) if value == "--environment-seed"
        ]
        require(len(seed_positions) == 1, "G3 scripted launch must bind one environment seed")
        seed = int(argv[seed_positions[0] + 1])
        mode_positions = [
            index for index, value in enumerate(argv[:-1]) if value == "--mode"
        ]
        require(len(mode_positions) == 1, "G3 scripted launch must bind one mode")
        mode = str(argv[mode_positions[0] + 1])
        require(mode == mode_label, "G3 scripted Job mode label differs from launch argv")
        job_key = (seed, mode)
        require(job_key not in observed_jobs, "G3 scripted Jobs repeat a seed/mode pair")
        observed_jobs.add(job_key)
        require(
            output_parent.endswith(f"/{mode}/seed-{seed}"),
            "G3 scripted Job OUTPUT_PARENT seed/mode suffix differs from launch",
        )

    require(len(observed_jobs) == job_count, "G3 scripted decoded job coverage is incomplete")
    require(observed_jobs == expected_seed_mode, "G3 scripted decoded job set differs from manifest")
    return {
        "ok": True,
        "bundle_root": str(root),
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "job_count": len(jobs),
        "configmap_count": len(configmaps),
        "scripted_job_count": len(observed_jobs),
        "scale": scale_value,
        "execution_scope": "model_blind_g3_no_policy",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = validate(args.root)
    except (G3ScriptedBundleValidationError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
