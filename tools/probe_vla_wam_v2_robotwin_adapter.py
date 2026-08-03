#!/usr/bin/env python3
"""Probe one RoboTwin adapter's prompt renderer without loading its checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


SCENES = {
    "soap_tea_box": {
        "object_name": "107_soap",
        "object_id": 2,
        "target_name": "112_tea-box",
        "target_id": 2,
    },
    "cards_coffee_box": {
        "object_name": "081_playingcards",
        "object_id": 2,
        "target_name": "113_coffee-box",
        "target_id": 5,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-id", required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    runner_path = args.runner.resolve()
    sys.path.insert(0, str(runner_path.parent))
    spec = importlib.util.spec_from_file_location(
        f"v2_adapter_probe_{args.adapter_id}", runner_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompt_fn = getattr(module, "make_prompt", None) or getattr(
        module, "make_seen_prompt", None
    )
    if prompt_fn is None:
        raise RuntimeError(f"No v2 prompt renderer in {runner_path}")
    protocol = module.load_protocol(args.protocol.resolve())
    rendered = {}
    for scene_id, scene in SCENES.items():
        env = SimpleNamespace(
            selected_modelname_A=scene["object_name"],
            selected_model_id_A=scene["object_id"],
            selected_modelname_B=scene["target_name"],
            selected_model_id_B=scene["target_id"],
        )
        rendered[scene_id] = {}
        for family_id in module.PROMPT_IDS:
            rendered[scene_id][family_id] = {}
            for direction in ("left", "right"):
                rendered[scene_id][family_id][direction] = prompt_fn(
                    env,
                    args.robotwin_root.resolve(),
                    direction,
                    family_id,
                    protocol,
                )
    source = runner_path.read_text()
    contract_markers = {
        "accepts_study_protocol": "--study-protocol" in source,
        "accepts_prompt_family": "--prompt-family" in source,
        "records_prompt_family": '"prompt_family"' in source,
        "records_relation_region": '"relation_region"' in source,
        "records_trajectory": '"trajectory_path"' in source,
        "records_simulator_video": '"simulator_video"' in source,
    }
    if not all(contract_markers.values()):
        raise RuntimeError(f"Incomplete adapter contract: {contract_markers}")
    print(
        json.dumps(
            {
                "adapter_id": args.adapter_id,
                "runner": str(runner_path),
                "robotwin_root": str(args.robotwin_root.resolve()),
                "contract_markers": contract_markers,
                "rendered_prompts": rendered,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
