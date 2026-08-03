#!/usr/bin/env python3
"""Verify byte-identical v2 prompts and evidence hooks across local WAM adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ADAPTERS = (
    {
        "id": "efficient_wam_rt_robotwin",
        "python": Path("/home/ali/projects/Efficient-WAM/.venv/bin/python"),
        "repo": Path("/home/ali/projects/Efficient-WAM"),
        "runner": Path("/home/ali/projects/Efficient-WAM/experiments/robotwin_language_gate/closed_loop_language_gate.py"),
        "robotwin_root": Path("/home/ali/projects/EfficientWAM-RoboTwin"),
    },
    {
        "id": "fastwam_robotwin",
        "python": Path("/home/ali/projects/FastWAM/.venv/bin/python"),
        "repo": Path("/home/ali/projects/FastWAM"),
        "runner": Path("/home/ali/projects/FastWAM/experiments/robotwin_language_gate/closed_loop_language_gate.py"),
        "robotwin_root": Path("/home/ali/projects/FastWAM/third_party/RoboTwin"),
    },
    {
        "id": "lingbot_va_robotwin",
        "python": Path("/home/ali/projects/lerobot-lingbot/.venv/bin/python"),
        "repo": Path("/home/ali/projects/lerobot-lingbot"),
        "runner": Path("/home/ali/projects/lerobot-lingbot/experiments/lingbot_language_gate/closed_loop_language_gate.py"),
        "robotwin_root": Path("/home/ali/projects/lerobot-lingbot/third_party/RoboTwin"),
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_probe(
    adapter: dict[str, Any], workspace: Path, protocol_path: Path
) -> dict[str, Any]:
    for key in ("python", "repo", "runner", "robotwin_root"):
        if not adapter[key].exists():
            raise FileNotFoundError(adapter[key])
    probe = workspace / "tools/probe_vla_wam_v2_robotwin_adapter.py"
    env = os.environ.copy()
    env["VLA_WAM_V2_STUDY_ROOT"] = str(workspace)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(workspace / "tools"),
            str(adapter["repo"]),
            env.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)
    process = subprocess.run(
        [
            str(adapter["python"]),
            str(probe),
            "--adapter-id",
            adapter["id"],
            "--runner",
            str(adapter["runner"]),
            "--robotwin-root",
            str(adapter["robotwin_root"]),
            "--protocol",
            str(protocol_path),
        ],
        cwd=adapter["repo"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode:
        raise RuntimeError(
            f"{adapter['id']} contract probe failed ({process.returncode}):\n{process.stderr}"
        )
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{adapter['id']} contract probe returned no JSON")
    result = json.loads(lines[-1])
    result["runner_sha256"] = sha256(adapter["runner"])
    result["probe_stderr"] = process.stderr.strip()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/vla_wam_shared_v2/pilot/adapter_contracts.json"),
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    protocol_path = workspace / "artifacts/vla_wam_shared_v2/protocol.json"
    results = [run_probe(adapter, workspace, protocol_path) for adapter in ADAPTERS]
    reference = results[0]["rendered_prompts"]
    mismatches = []
    for result in results[1:]:
        if result["rendered_prompts"] != reference:
            mismatches.append(result["adapter_id"])
    if mismatches:
        raise RuntimeError(f"Prompt bytes differ from Efficient-WAM adapter: {mismatches}")
    unique_prompts = {
        prompt
        for scene in reference.values()
        for family in scene.values()
        for prompt in family.values()
    }
    if len(unique_prompts) != 16:
        raise RuntimeError(f"Expected 16 unique scene/form/direction prompts, got {len(unique_prompts)}")
    output = args.output
    if not output.is_absolute():
        output = workspace / output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "1.0.0",
        "status": "contract_pass",
        "scope": "Static import and prompt/evidence-hook contract only; no checkpoint was loaded and no simulator episode was executed.",
        "protocol_path": str(protocol_path.relative_to(workspace)),
        "protocol_sha256": sha256(protocol_path),
        "adapter_count": len(results),
        "scene_fixture_count": len(reference),
        "unique_prompt_count": len(unique_prompts),
        "byte_identical_across_adapters": True,
        "adapters": results,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "adapter_count": report["adapter_count"],
                "scene_fixture_count": report["scene_fixture_count"],
                "unique_prompt_count": report["unique_prompt_count"],
                "output": str(output.relative_to(workspace)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
