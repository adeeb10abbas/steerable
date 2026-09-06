#!/usr/bin/env python3
"""Validate a rendered simulator-only V4 horizontal G3 Kubernetes bundle."""

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


class G3BundleValidationError(ValueError):
    """Raised when a G3 bundle could run policy inference or lacks evidence binding."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G3BundleValidationError(message)


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
        f"kubectl could not decode G3 bundle: {completed.stderr.strip()}",
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
            require(isinstance(nested, list), "kubectl G3 List lacks items")
            items.extend(nested)
        else:
            require(isinstance(payload, dict), "kubectl G3 decode returned a non-object")
            items.append(payload)
    require(items, "kubectl G3 decode returned no objects")
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
    require(manifest_path.is_file(), "G3 bundle manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixture_id = str(manifest.get("fixture_id", "horizontal"))
    fixture_token = fixture_id.replace("_", "-")
    require(
        manifest.get("schema_version")
        == f"vla-wam-v4-{fixture_token}-g3-k8s-bundle-v1",
        "G3 bundle manifest schema differs",
    )
    require(
        manifest.get("execution_scope") == "model_blind_g3_no_policy",
        "G3 bundle execution scope differs",
    )
    require(
        manifest.get("model_request_count") == 0
        and manifest.get("behavioral_episode_count") == 0,
        "G3 bundle manifest contains policy or behavioral work",
    )
    file_hashes = manifest.get("files_sha256")
    require(isinstance(file_hashes, dict) and file_hashes, "G3 file hash index is missing")
    for name, expected in file_hashes.items():
        path = root / name
        require(path.is_file(), f"G3 indexed file is missing: {name}")
        require(sha256_file(path) == expected, f"G3 indexed file hash changed: {name}")

    items = _decode_bundle(root)
    jobs = [item for item in items if item.get("kind") == "Job"]
    configmaps = [item for item in items if item.get("kind") == "ConfigMap"]
    seed_count = int(manifest.get("seed_count", 0))
    scale = manifest.get("scale")
    require(
        isinstance(scale, (int, float))
        and not isinstance(scale, bool)
        and float(scale) > 0,
        "G3 manifest lacks scale",
    )
    scale_value = float(scale)
    scale_text = _scale_label(scale_value)
    expected_seeds = manifest.get("registered_env_seeds")
    require(
        isinstance(expected_seeds, list)
        and len(expected_seeds) == seed_count
        and all(isinstance(seed, int) for seed in expected_seeds),
        "G3 manifest lacks registered environment seeds",
    )
    expected_seed_set = set(expected_seeds)
    expected_study_commit = manifest.get("expected_study_commit")
    expected_robolab_commit = manifest.get("expected_robolab_commit")
    native_control_dt_s = manifest.get("native_control_dt_s")
    campaign_sha256 = manifest.get("campaign_sha256")
    plan_sha256 = manifest.get("plan_sha256")
    reset_registry_sha256 = manifest.get("reset_registry_sha256")
    marker_wrapper_sha256 = manifest.get("marker_wrapper_sha256")
    runner_sha256 = manifest.get("runner_sha256")
    gate_core_sha256 = manifest.get("gate_core_sha256")
    attempt_id = manifest.get("attempt_id")
    require(isinstance(attempt_id, str) and attempt_id, "G3 manifest lacks attempt_id")
    require(
        isinstance(expected_study_commit, str) and len(expected_study_commit) == 40,
        "G3 manifest lacks expected study commit",
    )
    require(
        isinstance(expected_robolab_commit, str)
        and len(expected_robolab_commit) == 40,
        "G3 manifest lacks expected RoboLab commit",
    )
    require(
        isinstance(native_control_dt_s, (int, float))
        and not isinstance(native_control_dt_s, bool)
        and float(native_control_dt_s) > 0,
        "G3 manifest lacks native control dt",
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
            f"G3 manifest lacks the {label} SHA-256",
        )
    require(len(jobs) == seed_count, "G3 Job count differs from registered seed count")
    require(
        len(configmaps) == seed_count + 1,
        "G3 ConfigMap count must equal per-seed launches plus runtime scripts",
    )
    require(
        not any(item.get("kind") == "Service" for item in items),
        "G3 bundle must not create a policy Service",
    )

    launch_by_name: dict[str, dict[str, Any]] = {}
    scripts_maps = []
    for configmap in configmaps:
        metadata = configmap.get("metadata") or {}
        data = configmap.get("data") or {}
        require(configmap.get("immutable") is True, "all G3 ConfigMaps must be immutable")
        name = metadata.get("name")
        if "simulator-launch.json" in data:
            launch = json.loads(data["simulator-launch.json"])
            require(
                launch.get("role") == "simulator"
                and launch.get("execution_scope") == "model_blind_g3_no_policy",
                "G3 launch config is not simulator-only model-blind scope",
            )
            require(launch.get("policy_wait") is None, "G3 launch must not wait for a policy")
            argv = launch.get("experiment_argv")
            require(
                isinstance(argv, list)
                and len(argv) >= 12
                and str(argv[1]).endswith("run_v4_g3_checked.py")
                and argv[2] == "--expected-fixture"
                and argv[4] == "--expected-environment-seed"
                and argv[6] == "--expected-scale"
                and argv[8] == "--"
                and argv[0] == argv[9]
                and str(argv[10]).endswith("run_v4_horizontal_g3_path_seed.py"),
                "G3 launch does not invoke the path-seed runner through the marker-check wrapper",
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
                "G3 marker-check wrapper binding differs from the manifest",
            )
            runner_bindings = [
                binding
                for binding in file_bindings
                if isinstance(binding, dict) and binding.get("path") == argv[10]
            ]
            require(
                len(runner_bindings) == 1
                and runner_bindings[0].get("sha256") == runner_sha256,
                "G3 path-seed runner binding differs from the manifest",
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
                "G3 gate-core binding differs from the manifest",
            )
            lowered = " ".join(str(item).lower() for item in argv)
            require(
                "--policy-host" not in lowered
                and "--policy-port" not in lowered
                and "serve_policy" not in lowered,
                "G3 launch contains a policy endpoint or server",
            )
            require(
                launch.get("checkpoint_semantics") == "model_blind_g3_plan_candidate",
                "G3 hash-bound artifact semantics differ",
            )
            require(
                launch.get("checkpoint_sha256") == plan_sha256,
                "G3 launch plan SHA-256 differs from manifest",
            )
            option_values = {
                str(argv[index]): str(argv[index + 1])
                for index in range(len(argv) - 1)
                if str(argv[index]).startswith("--")
            }
            require(
                option_values.get("--expected-fixture") == fixture_id
                and option_values.get("--fixture-id") == fixture_id,
                "G3 wrapper and child fixture bindings differ",
            )
            require(
                option_values.get("--expected-environment-seed")
                == option_values.get("--environment-seed"),
                "G3 wrapper and child environment-seed bindings differ",
            )
            require(
                option_values.get("--expected-scale") == option_values.get("--scale")
                == scale_text,
                "G3 wrapper and child scale bindings differ",
            )
            require(
                option_values.get("--expected-study-commit")
                == expected_study_commit,
                "G3 launch study commit binding differs",
            )
            require(
                option_values.get("--expected-robolab-commit")
                == expected_robolab_commit,
                "G3 launch RoboLab commit binding differs",
            )
            require(
                option_values.get("--native-control-dt-s")
                == str(float(native_control_dt_s)),
                "G3 launch native control dt binding differs",
            )
            require(
                option_values.get("--campaign-sha256") == campaign_sha256,
                "G3 launch campaign SHA-256 differs",
            )
            require(
                option_values.get("--plan-sha256") == plan_sha256,
                "G3 launch plan SHA-256 differs",
            )
            require(
                option_values.get("--reset-registry-sha256") == reset_registry_sha256,
                "G3 launch reset registry SHA-256 differs",
            )
            digest_key = data.get("image.digest")
            require(
                isinstance(digest_key, str) and digest_key.startswith("sha256:"),
                "G3 launch ConfigMap lacks image digest binding",
            )
            launch_by_name[str(name)] = launch
        else:
            scripts_maps.append(configmap)
    require(len(launch_by_name) == seed_count, "G3 per-seed launch ConfigMaps are incomplete")
    require(len(scripts_maps) == 1, "G3 bundle must contain one scripts ConfigMap")
    scripts = scripts_maps[0].get("data") or {}
    for filename in (
        "lane_entrypoint.py",
        "startup_preflight.py",
        "check_policy_ready.py",
        "isaac_render_probe.py",
    ):
        require(filename in scripts, f"G3 scripts ConfigMap lacks {filename}")
        source = ROOT / "deploy/k8s/v4_lane_bundle/scripts" / filename
        require(
            hashlib.sha256(scripts[filename].encode("utf-8")).hexdigest()
            == sha256_file(source),
            f"G3 embedded runtime script differs: {filename}",
        )

    observed_seeds: set[int] = set()
    observed_output_parents: set[str] = set()
    for job in jobs:
        metadata = job.get("metadata") or {}
        labels = metadata.get("labels") or {}
        require(
            labels.get("v4-gate") == f"g3-{fixture_token}",
            "G3 Job gate label differs",
        )
        require(labels.get("v4-scale") == scale_text, "G3 Job scale label differs")
        spec = job.get("spec") or {}
        require(spec.get("backoffLimit") == 0, "G3 Jobs must not retry implicitly")
        pod_spec = ((spec.get("template") or {}).get("spec") or {})
        require(pod_spec.get("restartPolicy") == "Never", "G3 restartPolicy must be Never")
        node_selector = pod_spec.get("nodeSelector") or {}
        require(
            node_selector.get("nvidia.com/gpu.product") == manifest.get("gpu_product"),
            "G3 Job GPU product affinity differs",
        )
        containers = pod_spec.get("containers") or []
        require(len(containers) == 1, "G3 Job must contain exactly one simulator container")
        container = containers[0]
        config_refs = {
            ((volume.get("configMap") or {}).get("name"))
            for volume in pod_spec.get("volumes") or []
            if isinstance(volume, dict)
        }
        matching = set(launch_by_name) & config_refs
        require(len(matching) == 1, "G3 Job must reference one per-seed launch ConfigMap")
        launch_ref_name = matching.pop()
        launch = launch_by_name[str(launch_ref_name)]
        launch_digest = None
        for configmap in configmaps:
            if (configmap.get("metadata") or {}).get("name") == launch_ref_name:
                launch_digest = (configmap.get("data") or {}).get("image.digest")
                break
        require(
            isinstance(launch_digest, str) and launch_digest.startswith("sha256:"),
            "G3 launch ConfigMap lacks image digest binding",
        )
        image = str(container.get("image") or "")
        require("@sha256:" in image, "G3 Job image must be pinned by digest")
        require(
            image.endswith(launch_digest),
            "G3 Job image digest differs from launch binding",
        )
        requests = (container.get("resources") or {}).get("requests") or {}
        limits = (container.get("resources") or {}).get("limits") or {}
        require(
            str(requests.get("nvidia.com/gpu")) == "1"
            and str(limits.get("nvidia.com/gpu")) == "1",
            "G3 Job must request and limit exactly one GPU",
        )
        env = _env(container)
        require(
            (env.get("LANE_ROLE") or {}).get("value") == "simulator",
            "G3 Job role must be simulator",
        )
        output_parent = (env.get("OUTPUT_PARENT") or {}).get("value")
        require(
            isinstance(output_parent, str) and output_parent.startswith("/data/"),
            "G3 Job lacks absolute OUTPUT_PARENT",
        )
        require(
            f"/attempt-{attempt_id}/scale-{scale_text}/seed-" in output_parent,
            "G3 Job OUTPUT_PARENT lacks attempt/scale/seed isolation",
        )
        require(
            output_parent not in observed_output_parents,
            "G3 Jobs repeat an OUTPUT_PARENT",
        )
        observed_output_parents.add(output_parent)
        for pod_field in ("POD_UID", "POD_NAME", "POD_NAMESPACE", "POD_IP"):
            row = env.get(pod_field) or {}
            require("valueFrom" in row, f"G3 Job lacks downward API env {pod_field}")
        mounts = container.get("volumeMounts") or []
        mount_names = {
            mount.get("name")
            for mount in mounts
            if isinstance(mount, dict) and isinstance(mount.get("name"), str)
        }
        require("data" in mount_names, "G3 Job lacks PVC data mount")
        launch_ref = ((env.get("LANE_LAUNCH_CONFIG_SHA256") or {}).get("value"))
        require(
            isinstance(launch_ref, str) and len(launch_ref) == 64,
            "G3 Job lacks launch SHA binding",
        )
        argv = launch["experiment_argv"]
        seed_positions = [
            index for index, value in enumerate(argv[:-1]) if value == "--environment-seed"
        ]
        require(len(seed_positions) == 1, "G3 launch must bind one environment seed")
        seed = int(argv[seed_positions[0] + 1])
        require(seed not in observed_seeds, "G3 Jobs repeat an environment seed")
        observed_seeds.add(seed)
        require(
            output_parent.endswith(f"/seed-{seed}"),
            "G3 Job OUTPUT_PARENT seed suffix differs from launch seed",
        )

    require(len(observed_seeds) == seed_count, "G3 decoded seed coverage is incomplete")
    require(observed_seeds == expected_seed_set, "G3 decoded seed set differs from manifest")
    return {
        "ok": True,
        "bundle_root": str(root),
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "job_count": len(jobs),
        "configmap_count": len(configmaps),
        "environment_seed_count": len(observed_seeds),
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
    except (G3BundleValidationError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
