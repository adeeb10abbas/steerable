#!/usr/bin/env python3
"""Validate a rendered simulator-only V4 horizontal G2 Kubernetes bundle."""

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


class G2BundleValidationError(ValueError):
    """Raised when a G2 bundle could run policy inference or lacks evidence binding."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G2BundleValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        f"kubectl could not decode G2 bundle: {completed.stderr.strip()}",
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
            require(isinstance(nested, list), "kubectl G2 List lacks items")
            items.extend(nested)
        else:
            require(isinstance(payload, dict), "kubectl G2 decode returned a non-object")
            items.append(payload)
    require(items, "kubectl G2 decode returned no objects")
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
    require(manifest_path.is_file(), "G2 bundle manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(
        manifest.get("schema_version")
        == "vla-wam-v4-horizontal-g2-k8s-bundle-v1",
        "G2 bundle manifest schema differs",
    )
    require(
        manifest.get("execution_scope") == "model_blind_g2_no_policy",
        "G2 bundle execution scope differs",
    )
    require(
        manifest.get("model_request_count") == 0
        and manifest.get("behavioral_episode_count") == 0,
        "G2 bundle manifest contains policy or behavioral work",
    )
    file_hashes = manifest.get("files_sha256")
    require(isinstance(file_hashes, dict) and file_hashes, "G2 file hash index is missing")
    for name, expected in file_hashes.items():
        path = root / name
        require(path.is_file(), f"G2 indexed file is missing: {name}")
        require(sha256_file(path) == expected, f"G2 indexed file hash changed: {name}")

    items = _decode_bundle(root)
    jobs = [item for item in items if item.get("kind") == "Job"]
    configmaps = [item for item in items if item.get("kind") == "ConfigMap"]
    seed_count = int(manifest.get("seed_count", 0))
    expected_study_commit = manifest.get("expected_study_commit")
    expected_robolab_commit = manifest.get("expected_robolab_commit")
    native_control_dt_s = manifest.get("native_control_dt_s")
    require(
        isinstance(expected_study_commit, str) and len(expected_study_commit) == 40,
        "G2 manifest lacks expected study commit",
    )
    require(
        isinstance(expected_robolab_commit, str)
        and len(expected_robolab_commit) == 40,
        "G2 manifest lacks expected RoboLab commit",
    )
    require(
        isinstance(native_control_dt_s, (int, float))
        and not isinstance(native_control_dt_s, bool)
        and float(native_control_dt_s) > 0,
        "G2 manifest lacks native control dt",
    )
    require(len(jobs) == seed_count, "G2 Job count differs from registered seed count")
    require(
        len(configmaps) == seed_count + 1,
        "G2 ConfigMap count must equal per-seed launches plus runtime scripts",
    )
    require(
        not any(item.get("kind") == "Service" for item in items),
        "G2 bundle must not create a policy Service",
    )

    launch_by_name: dict[str, dict[str, Any]] = {}
    scripts_maps = []
    for configmap in configmaps:
        metadata = configmap.get("metadata") or {}
        data = configmap.get("data") or {}
        require(configmap.get("immutable") is True, "all G2 ConfigMaps must be immutable")
        name = metadata.get("name")
        if "simulator-launch.json" in data:
            launch = json.loads(data["simulator-launch.json"])
            require(
                launch.get("role") == "simulator"
                and launch.get("execution_scope") == "model_blind_g2_no_policy",
                "G2 launch config is not simulator-only model-blind scope",
            )
            require(launch.get("policy_wait") is None, "G2 launch must not wait for a policy")
            argv = launch.get("experiment_argv")
            require(
                isinstance(argv, list)
                and any(str(item).endswith("run_v4_horizontal_g2_seed.py") for item in argv),
                "G2 launch does not invoke the frozen G2 seed runner",
            )
            lowered = " ".join(str(item).lower() for item in argv)
            require(
                "--policy-host" not in lowered
                and "--policy-port" not in lowered
                and "serve_policy" not in lowered,
                "G2 launch contains a policy endpoint or server",
            )
            require(
                launch.get("checkpoint_semantics")
                == "model_blind_reset_registry_candidate",
                "G2 hash-bound artifact semantics differ",
            )
            option_values = {
                str(argv[index]): str(argv[index + 1])
                for index in range(len(argv) - 1)
                if str(argv[index]).startswith("--")
            }
            require(
                option_values.get("--expected-study-commit")
                == expected_study_commit,
                "G2 launch study commit binding differs",
            )
            require(
                option_values.get("--expected-robolab-commit")
                == expected_robolab_commit,
                "G2 launch RoboLab commit binding differs",
            )
            require(
                option_values.get("--native-control-dt-s")
                == str(float(native_control_dt_s)),
                "G2 launch native control dt binding differs",
            )
            launch_by_name[str(name)] = launch
        else:
            scripts_maps.append(configmap)
    require(len(launch_by_name) == seed_count, "G2 per-seed launch ConfigMaps are incomplete")
    require(len(scripts_maps) == 1, "G2 bundle must contain one scripts ConfigMap")
    scripts = scripts_maps[0].get("data") or {}
    for filename in (
        "lane_entrypoint.py",
        "startup_preflight.py",
        "check_policy_ready.py",
        "isaac_render_probe.py",
    ):
        require(filename in scripts, f"G2 scripts ConfigMap lacks {filename}")
        source = ROOT / "deploy/k8s/v4_lane_bundle/scripts" / filename
        require(
            hashlib.sha256(scripts[filename].encode("utf-8")).hexdigest()
            == sha256_file(source),
            f"G2 embedded runtime script differs: {filename}",
        )

    observed_seeds: set[int] = set()
    for job in jobs:
        metadata = job.get("metadata") or {}
        labels = metadata.get("labels") or {}
        require(labels.get("v4-gate") == "g2-horizontal", "G2 Job gate label differs")
        spec = job.get("spec") or {}
        require(spec.get("backoffLimit") == 0, "G2 Jobs must not retry implicitly")
        pod_spec = ((spec.get("template") or {}).get("spec") or {})
        require(pod_spec.get("restartPolicy") == "Never", "G2 restartPolicy must be Never")
        containers = pod_spec.get("containers") or []
        require(len(containers) == 1, "G2 Job must contain exactly one simulator container")
        container = containers[0]
        requests = (container.get("resources") or {}).get("requests") or {}
        limits = (container.get("resources") or {}).get("limits") or {}
        require(
            str(requests.get("nvidia.com/gpu")) == "1"
            and str(limits.get("nvidia.com/gpu")) == "1",
            "G2 Job must request and limit exactly one GPU",
        )
        env = _env(container)
        require(
            (env.get("LANE_ROLE") or {}).get("value") == "simulator",
            "G2 Job role must be simulator",
        )
        launch_ref = (
            ((env.get("LANE_LAUNCH_CONFIG_SHA256") or {}).get("value"))
        )
        require(
            isinstance(launch_ref, str) and len(launch_ref) == 64,
            "G2 Job lacks launch SHA binding",
        )
        config_refs = {
            ((volume.get("configMap") or {}).get("name"))
            for volume in pod_spec.get("volumes") or []
            if isinstance(volume, dict)
        }
        matching = set(launch_by_name) & config_refs
        require(len(matching) == 1, "G2 Job must reference one per-seed launch ConfigMap")
        launch = launch_by_name[matching.pop()]
        argv = launch["experiment_argv"]
        seed_positions = [
            index for index, value in enumerate(argv[:-1]) if value == "--environment-seed"
        ]
        require(len(seed_positions) == 1, "G2 launch must bind one environment seed")
        seed = int(argv[seed_positions[0] + 1])
        require(seed not in observed_seeds, "G2 Jobs repeat an environment seed")
        observed_seeds.add(seed)

    require(len(observed_seeds) == seed_count, "G2 decoded seed coverage is incomplete")
    return {
        "ok": True,
        "bundle_root": str(root),
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "job_count": len(jobs),
        "configmap_count": len(configmaps),
        "environment_seed_count": len(observed_seeds),
        "execution_scope": "model_blind_g2_no_policy",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = validate(args.root)
    except (G2BundleValidationError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
