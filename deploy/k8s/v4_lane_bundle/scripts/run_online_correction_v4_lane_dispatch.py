#!/usr/bin/env python3
"""Run the immutable episode list bound to one V4 lane dispatch manifest.

The coordinator renders this script plus a dispatch manifest into the simulator
launch ConfigMap. It validates the released runner binding, then invokes that
runner once per assigned episode. It never substitutes /usr/bin/true or a
qualification-only scope for behavioral work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


DISPATCH_SCHEMA = "v4-lane-dispatch-v1"
SHA256_RE = __import__("re").compile(r"^[0-9a-f]{64}$")


class DispatchError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise DispatchError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dispatch_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"dispatch manifest missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot read dispatch manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        _fail("dispatch manifest must be a JSON object")
    if payload.get("schema_version") != DISPATCH_SCHEMA:
        _fail("dispatch manifest schema_version mismatch")
    for key in (
        "manifest_sha256",
        "queue_path",
        "runtime_lock_path",
        "campaign_config_path",
        "runner_entrypoint",
        "runner_sha256",
        "lane_id",
        "attempt_id",
        "policy_id",
        "group_ids",
        "episode_ids",
    ):
        if key not in payload:
            _fail(f"dispatch manifest missing required field: {key}")
    manifest_sha = str(payload["manifest_sha256"])
    if not SHA256_RE.fullmatch(manifest_sha):
        _fail("dispatch manifest manifest_sha256 must be lowercase SHA-256")
    runner_sha = str(payload["runner_sha256"])
    if not SHA256_RE.fullmatch(runner_sha):
        _fail("dispatch manifest runner_sha256 must be lowercase SHA-256")
    episode_ids = payload.get("episode_ids")
    group_ids = payload.get("group_ids")
    if not isinstance(episode_ids, list) or not all(isinstance(item, str) and item for item in episode_ids):
        _fail("dispatch manifest episode_ids must be a nonempty string array")
    if not isinstance(group_ids, list) or not all(isinstance(item, str) and item for item in group_ids):
        _fail("dispatch manifest group_ids must be a nonempty string array")
    if payload.get("qualification_only") is True:
        _fail("dispatch manifest must not be qualification_only for behavioral execution")
    if payload.get("one_episode_per_process") is False:
        _fail("dispatch manifest must enforce one_episode_per_process")
    return payload


def validate_runner_binding(manifest: Mapping[str, Any]) -> Path:
    runner_path = Path(str(manifest["runner_entrypoint"]))
    if not runner_path.is_absolute() or not runner_path.is_file():
        _fail(f"released runner entrypoint missing on lane filesystem: {runner_path}")
    observed = sha256_file(runner_path)
    expected = str(manifest["runner_sha256"])
    if observed != expected:
        _fail(
            f"released runner sha256 mismatch: expected {expected}, observed {observed}"
        )
    lowered = str(runner_path).lower()
    if "online_correction_v4" not in lowered:
        _fail("released runner entrypoint must belong to online_correction_v4")
    return runner_path


def load_policy_wait_endpoint(launch_config_path: Path) -> tuple[str, int]:
    if not launch_config_path.is_file():
        _fail(f"simulator launch config missing: {launch_config_path}")
    try:
        config = json.loads(launch_config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot read simulator launch config {launch_config_path}: {exc}") from exc
    if not isinstance(config, dict):
        _fail("simulator launch config must be a JSON object")
    wait = config.get("policy_wait")
    if not isinstance(wait, dict):
        _fail("simulator launch config policy_wait must be an object")
    host = wait.get("host")
    port = wait.get("port")
    if not isinstance(host, str) or not host.strip():
        _fail("simulator launch config policy_wait.host is missing")
    if not isinstance(port, int) or not (1 <= port <= 65535):
        _fail("simulator launch config policy_wait.port is invalid")
    return host.strip(), port


def resolve_policy_endpoint(
    *,
    args: argparse.Namespace,
    launch_config_path: Path | None,
) -> tuple[str, int]:
    if args.policy_host is not None and args.policy_port is not None:
        return str(args.policy_host), int(args.policy_port)
    if launch_config_path is not None and launch_config_path.is_file():
        return load_policy_wait_endpoint(launch_config_path)
    launch_from_env = os.environ.get("LANE_LAUNCH_CONFIG")
    if launch_from_env:
        return load_policy_wait_endpoint(Path(launch_from_env))
    policy_host = args.policy_host or os.environ.get("POLICY_WAIT_HOST")
    policy_port = args.policy_port
    if policy_port is None:
        raw_port = os.environ.get("POLICY_WAIT_PORT")
        policy_port = int(raw_port) if raw_port else None
    if policy_host is None or policy_port is None:
        _fail(
            "policy endpoint requires --policy-host/--policy-port, --launch-config, "
            "LANE_LAUNCH_CONFIG, or POLICY_WAIT_HOST/POLICY_WAIT_PORT"
        )
    return str(policy_host), int(policy_port)


def build_episode_command(
    *,
    manifest: Mapping[str, Any],
    runner_path: Path,
    episode_id: str,
    output_dir: Path,
    policy_host: str | None,
    policy_port: int | None,
) -> list[str]:
    python_bin = os.environ.get("PYTHON_BIN")
    if not python_bin:
        _fail("PYTHON_BIN is required for lane dispatch")
    cmd = [
        python_bin,
        str(runner_path),
        "--manifest",
        str(manifest["queue_path"]),
        "--runtime-lock",
        str(manifest["runtime_lock_path"]),
        "--episode-id",
        episode_id,
        "--attempt-id",
        str(manifest["attempt_id"]),
        "--output",
        str(output_dir / episode_id.replace("/", "_")),
        "--campaign-config",
        str(manifest["campaign_config_path"]),
    ]
    if policy_host is not None:
        cmd.extend(["--policy-host", policy_host])
    if policy_port is not None:
        cmd.extend(["--policy-port", str(policy_port)])
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dispatch-manifest",
        required=True,
        help="Absolute path to the immutable lane dispatch manifest JSON.",
    )
    parser.add_argument("--policy-host", default=None)
    parser.add_argument("--policy-port", type=int, default=None)
    parser.add_argument(
        "--launch-config",
        default=None,
        help="Mounted simulator launch JSON used to resolve policy_wait host/port.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate bindings and print planned runner invocations without executing.",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.dispatch_manifest).resolve()
    manifest = load_dispatch_manifest(manifest_path)
    runner_path = validate_runner_binding(manifest)
    launch_config_path = Path(args.launch_config).resolve() if args.launch_config else None

    output_parent = os.environ.get("EPISODE_OUTPUT_DIR") or os.environ.get("OUTPUT_PARENT")
    if not output_parent:
        _fail("EPISODE_OUTPUT_DIR or OUTPUT_PARENT must be set by lane entrypoint")
    output_root = Path(output_parent).resolve()
    if not output_root.is_absolute():
        _fail("lane output directory must be absolute")

    policy_host, policy_port = resolve_policy_endpoint(args=args, launch_config_path=launch_config_path)

    planned: list[dict[str, Any]] = []
    for episode_id in manifest["episode_ids"]:
        cmd = build_episode_command(
            manifest=manifest,
            runner_path=runner_path,
            episode_id=episode_id,
            output_dir=output_root,
            policy_host=policy_host,
            policy_port=policy_port,
        )
        planned.append({"episode_id": episode_id, "command": cmd})

    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": DISPATCH_SCHEMA,
                    "lane_id": manifest["lane_id"],
                    "attempt_id": manifest["attempt_id"],
                    "group_ids": list(manifest["group_ids"]),
                    "behavioral_episode_count": len(manifest["episode_ids"]),
                    "policy_host": policy_host,
                    "policy_port": policy_port,
                    "planned_invocations": planned,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    for episode_id in manifest["episode_ids"]:
        cmd = build_episode_command(
            manifest=manifest,
            runner_path=runner_path,
            episode_id=episode_id,
            output_dir=output_root,
            policy_host=policy_host,
            policy_port=policy_port,
        )
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(
                f"[V4 lane dispatch] episode {episode_id} failed with exit {result.returncode}",
                file=sys.stderr,
            )
            return result.returncode
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DispatchError as exc:
        print(f"[V4 lane dispatch] blocked: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
