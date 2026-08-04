#!/usr/bin/env python3
"""Build the frozen six-cell V2-A010 pi0.5 current-stack media registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from vla_wam_v2_protocol import load_protocol, render_prompt


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(workspace: Path, path: Path) -> dict[str, Any]:
    return {"path":str(path.relative_to(workspace)),"bytes":path.stat().st_size,"sha256":sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_v2a010_registry.json"))
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    protocol_path = workspace / "artifacts/vla_wam_shared_v2/protocol.json"
    amendment_path = workspace / "artifacts/vla_wam_shared_v2/pilot/post_result_pi05_current_stack_media_gate_amendment.json"
    checkpoint_path = workspace / "artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_checkpoint_manifest.json"
    protocol = load_protocol(protocol_path)
    amendment = json.loads(amendment_path.read_text())
    checkpoint = json.loads(checkpoint_path.read_text())
    if amendment["amendment_id"] != "V2-A010" or checkpoint["amendment_id"] != "V2-A010":
        raise ValueError("V2-A010 sources are inconsistent")
    cells = []
    for seed in amendment["behavioral_grid"]["environment_seeds"]:
        for relation in amendment["behavioral_grid"]["requested_relations"]:
            prompt = render_prompt(
                protocol, family_id="direct_command", direction=relation,
                movable="Rubik's cube", movable_short="cube", reference="bowl",
                arena="droid_robolab",
            )
            pair_id = f"droid_pair_seed_{seed}"
            cells.append({
                "cell_id":f"pi05_droid_current_stack__{pair_id}__direct_command__{relation}",
                "experiment_id":amendment["experiment_identity"]["experiment_id"],
                "model_id":"pi05_droid_current_stack",
                "arena":"droid_robolab",
                "pair_id":pair_id,
                "anchor_task":f"RubiksCube{relation.title()}OfBowlMatchedTask",
                "environment_seed":seed,
                "sampling_seed_base":seed,
                "first_policy_request_sampling_seed":seed*1000,
                "prompt_family":"direct_command",
                "requested_relation":relation,
                "rendered_prompt":prompt,
                "instruction_controller":"static",
                "oracle_or_subtask_coach":False,
                "dynamic_prompt_switches":0,
                "open_loop_horizon":15,
                "video_mode":"viewport",
                "executed_action_trace_required":True,
                "valid_failure_retained":True,
                "output_folder_name":f"v2a010_pi05_current_seed{seed}_direct_command_{relation}",
                "action_trace_stem":f"seed{seed}_direct_command_{relation}",
            })
    if len(cells) != 6 or len({row["cell_id"] for row in cells}) != 6:
        raise ValueError("V2-A010 registry must have six unique cells")
    sources = [
        workspace / "tools/build_v2a010_pi05_registry.py",
        workspace / "tools/compile_v2a010_pi05_gate.py",
        workspace / "experiments/pi05_current_stack/v2a010_serve_policy.py",
        workspace / "experiments/pi05_current_stack/v2a010_robolab_client.py",
        workspace / "experiments/pi05_current_stack/v2a010_robolab_gate.py",
        workspace / "experiments/pi05_current_stack/v2a010_capture_fixed_observation.py",
        workspace / "experiments/pi05_current_stack/v2a010_fixed_observation_probe.py",
        workspace / "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_left_of_bowl_matched.py",
        workspace / "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_right_of_bowl_matched.py",
    ]
    payload = {
        "schema_version":"vla-wam-v2a010-pi05-current-stack-registry-v1",
        "amendment_id":"V2-A010",
        "status":"frozen_before_current_stack_model_load_or_behavioral_inference",
        "protocol":{"path":str(protocol_path.relative_to(workspace)),"sha256":sha256(protocol_path)},
        "amendment":{"path":str(amendment_path.relative_to(workspace)),"sha256":sha256(amendment_path)},
        "checkpoint_manifest":{"path":str(checkpoint_path.relative_to(workspace)),"sha256":sha256(checkpoint_path)},
        "experiment_identity":amendment["experiment_identity"],
        "claim_boundary":amendment["claim_boundary"],
        "summary":{"episode_count":6,"left_right_pair_count":3,"environment_seeds":[8300,8301,8302],"left_cells":3,"right_cells":3},
        "adapter_sources":[record(workspace,path) for path in sources],
        "cells":cells,
    }
    output = args.output if args.output.is_absolute() else workspace / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(output),**payload["summary"]},indent=2))


if __name__ == "__main__":
    main()
