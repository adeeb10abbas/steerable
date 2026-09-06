#!/usr/bin/env python3
"""Run the C8 GR00T Bridge/WidowX policy-session qualification gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.second_stack import (  # noqa: E402
    ENV_NAME,
    apply_registered_reset,
    ensure_registered_support,
    unwrap_simpler_env,
)
from tools.run_v4_second_stack_g2 import (  # noqa: E402
    canonical_json_bytes,
    load_json,
    sha256_file,
    verify_external_stack,
)


class SecondStackG4Error(RuntimeError):
    pass


HORIZONTAL_SUFFIX = (
    " Use the robot's fixed viewpoint for left, right, front, and behind."
)


def _sha256_array(value: Any) -> str:
    return hashlib.sha256(value.tobytes()).hexdigest()


def _checkpoint_manifest(checkpoint_path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(checkpoint_path.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        relative = str(path.relative_to(checkpoint_path))
        records[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return records


def _verify_backbone_cache(
    *,
    backbone_path: Path,
    backbone_revision: str,
    hf_home: Path,
) -> dict[str, Any]:
    if len(backbone_revision) != 40 or any(
        character not in "0123456789abcdef" for character in backbone_revision
    ):
        raise SecondStackG4Error("C8 backbone revision must be a lowercase commit")
    if not backbone_path.is_dir():
        raise SecondStackG4Error(f"C8 backbone is missing: {backbone_path}")
    metadata_root = backbone_path / ".cache" / "huggingface" / "download"
    for filename in ("config.json", "model.safetensors"):
        metadata = metadata_root / f"{filename}.metadata"
        if not metadata.is_file():
            raise SecondStackG4Error(
                f"C8 backbone provenance metadata is missing: {metadata}"
            )
        observed_revision = metadata.read_text(encoding="utf-8").splitlines()[0]
        if observed_revision != backbone_revision:
            raise SecondStackG4Error(
                f"C8 backbone revision differs for {filename}: "
                f"{observed_revision}"
            )
    cache_repo = hf_home / "hub" / "models--nvidia--Cosmos-Reason2-2B"
    main_ref = cache_repo / "refs" / "main"
    snapshot = cache_repo / "snapshots" / backbone_revision
    if (
        not main_ref.is_file()
        or main_ref.read_text(encoding="utf-8").strip() != backbone_revision
    ):
        raise SecondStackG4Error("C8 backbone cache main ref differs")
    if not snapshot.exists() or snapshot.resolve() != backbone_path.resolve():
        raise SecondStackG4Error("C8 backbone cache snapshot differs")
    return {
        "repository": "nvidia/Cosmos-Reason2-2B",
        "revision": backbone_revision,
        "path": str(backbone_path),
        "hf_home": str(hf_home),
        "content_manifest": _checkpoint_manifest(backbone_path),
    }


def _processed_observation(env: Any) -> dict[str, Any]:
    import cv2
    import numpy as np
    from transforms3d import euler as te, quaternions as tq

    raw = unwrap_simpler_env(env)
    get_obs = getattr(raw, "get_obs", None)
    if not callable(get_obs):
        raise SecondStackG4Error("C8 raw environment does not expose get_obs")
    raw_observation = get_obs()
    color = np.asarray(
        raw_observation["image"]["3rd_view_camera"]["Color"]
    )[..., :3]
    if color.dtype != np.uint8:
        color = np.asarray(color, dtype=np.float32)
        if color.size and float(np.nanmax(color)) <= 1.0:
            color = color * 255.0
        color = np.clip(color, 0.0, 255.0).astype(np.uint8)
    proprio = np.asarray(raw_observation["agent"]["eef_pos"], dtype=np.float64)
    if proprio.shape != (8,):
        raise SecondStackG4Error("C8 Bridge eef_pos must have shape (8,)")
    rotation = tq.quat2mat(proprio[3:7])
    bridge_default_rotation = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    roll, pitch, yaw = te.mat2euler(
        rotation @ bridge_default_rotation.T
    )
    return {
        "video.image_0": cv2.resize(color, (256, 256)),
        "state.x": [float(proprio[0])],
        "state.y": [float(proprio[1])],
        "state.z": [float(proprio[2])],
        "state.roll": [float(roll)],
        "state.pitch": [float(pitch)],
        "state.yaw": [float(yaw)],
        "state.pad": [0.0],
        "state.gripper": [float(proprio[7])],
    }


def _batched_observation(
    *,
    observation: Mapping[str, Any],
    modality_configs: Mapping[str, Any],
    prompt: str,
    np: Any,
) -> dict[str, Any]:
    batched: dict[str, Any] = {}
    for key in modality_configs["video"].modality_keys:
        flat_key = f"video.{key}"
        frame = np.asarray(observation[flat_key], dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise SecondStackG4Error(f"{flat_key} is not an HxWx3 RGB frame")
        horizon = len(modality_configs["video"].delta_indices)
        batched[flat_key] = np.repeat(
            frame[None, None, ...],
            horizon,
            axis=1,
        )
    for key in modality_configs["state"].modality_keys:
        flat_key = f"state.{key}"
        state = np.asarray(observation[flat_key], dtype=np.float32).reshape(
            1, 1, -1
        )
        horizon = len(modality_configs["state"].delta_indices)
        batched[flat_key] = np.repeat(state, horizon, axis=1)
    language_keys = list(modality_configs["language"].modality_keys)
    if not language_keys:
        raise SecondStackG4Error("C8 checkpoint has no language modality")
    batched[language_keys[0]] = (prompt,)
    return batched


def _action_record(actions: Mapping[str, Any], np: Any) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in sorted(actions.items()):
        array = np.asarray(value)
        if (
            array.dtype != np.float32
            or array.ndim != 3
            or array.shape[0] != 1
            or not np.isfinite(array).all()
        ):
            raise SecondStackG4Error(
                f"C8 action {key} is not finite float32 BxTxD output"
            )
        record[str(key)] = {
            "shape": [int(size) for size in array.shape],
            "dtype": str(array.dtype),
            "sha256": _sha256_array(array),
            "minimum": float(array.min()),
            "maximum": float(array.max()),
        }
    if not record:
        raise SecondStackG4Error("C8 policy returned no action arrays")
    return record


def _actions_equal(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    np: Any,
) -> bool:
    return set(first) == set(second) and all(
        np.array_equal(first[key], second[key]) for key in first
    )


def run_g4(
    *,
    registry_path: Path,
    integration_root: Path,
    checkpoint_path: Path,
    backbone_path: Path,
    backbone_revision: str,
    hf_home: Path,
) -> dict[str, Any]:
    registry = load_json(registry_path)
    if (
        registry.get("fixture_id") != "second_stack"
        or registry.get("environment_name") != ENV_NAME
    ):
        raise SecondStackG4Error("C8 reset registry binding differs")
    stack_receipt = verify_external_stack(
        integration_root=integration_root,
        registry=registry,
    )
    if not checkpoint_path.is_dir():
        raise SecondStackG4Error(f"C8 checkpoint is missing: {checkpoint_path}")
    backbone_receipt = _verify_backbone_cache(
        backbone_path=backbone_path,
        backbone_revision=backbone_revision,
        hf_home=hf_home,
    )
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["GROOT_HF_LOCAL_FIRST"] = "1"
    os.environ["GROOT_PATCH_MISTRAL"] = "1"

    sys.path.insert(0, str(integration_root))
    import gymnasium as gym
    import numpy as np
    import torch
    from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.utils.determinism import seed_everything

    register_simpler_envs()
    resets = registry.get("resets_by_env_seed")
    if not isinstance(resets, Mapping):
        raise SecondStackG4Error("C8 reset registry lacks reset rows")
    env_seed = min(int(seed) for seed in resets)
    reset_row = resets[str(env_seed)]
    env = gym.make(ENV_NAME)
    try:
        env.reset(seed=env_seed)
        ensure_registered_support(env)
        apply_registered_reset(env, reset_row, settle_steps=250)
        observation = _processed_observation(env)
        image = np.asarray(observation["video.image_0"], dtype=np.uint8)
    finally:
        env.close()

    policy = Gr00tPolicy(
        embodiment_tag="SIMPLER_ENV_WIDOWX",
        model_path=str(checkpoint_path),
        device="cuda",
        strict=True,
    )
    wrapped = Gr00tSimPolicyWrapper(policy)
    target_name = str(registry["object_roles"]["target"]["prompt_name"])
    reference_name = str(registry["object_roles"]["reference"]["prompt_name"])
    prompts = {
        relation: (
            f"Place the {target_name} so that the {target_name} is "
            f"{relation} of the {reference_name}.{HORIZONTAL_SUFFIX}"
        )
        for relation in ("left", "right")
    }
    outputs: dict[str, Mapping[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for label, relation in (
        ("left", "left"),
        ("left_fresh_session_exact_repeat", "left"),
        ("right", "right"),
    ):
        prompt = prompts[relation]
        batch = _batched_observation(
            observation=observation,
            modality_configs=policy.modality_configs,
            prompt=prompt,
            np=np,
        )
        seed_everything(2100050000)
        wrapped.reset()
        started = time.monotonic()
        actions, _info = wrapped.get_action(batch)
        wall_seconds = time.monotonic() - started
        outputs[label] = actions
        records.append(
            {
                "condition": label,
                "relation": relation,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "observation_image_sha256": _sha256_array(image),
                "action": _action_record(actions, np),
                "fresh_policy_reset": True,
                "wall_seconds": wall_seconds,
            }
        )
    exact_repeat = _actions_equal(
        outputs["left"],
        outputs["left_fresh_session_exact_repeat"],
        np,
    )
    action_shapes = {
        tuple(record["shape"])
        for record in records[0]["action"].values()
    }
    passed = exact_repeat and len(records) == 3 and bool(action_shapes)
    del wrapped
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "schema_version": "v4-second-stack-g4-policy-session-receipt-v1",
        "campaign_id": "online_correction_v4",
        "family_id": "C8",
        "fixture_id": "second_stack",
        "gate": "G4",
        "policy_id": "groot_n1_7_simplerenv_bridge",
        "checkpoint_revision": registry["external_stack_identity"][
            "checkpoint_revision"
        ],
        "checkpoint_content_manifest": _checkpoint_manifest(checkpoint_path),
        "backbone": backbone_receipt,
        "reset_registry": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
        },
        "stack_receipt": stack_receipt,
        "env_seed": env_seed,
        "conditioning_input": {
            "image_shape": [int(size) for size in image.shape],
            "image_dtype": str(image.dtype),
            "image_sha256": _sha256_array(image),
        },
        "checks": {
            "three_model_requests_completed": len(records) == 3,
            "action_arrays_finite_float32": bool(action_shapes),
            "fresh_reset_exact_repeat_actions_equal": exact_repeat,
            "static_prompt_bytes_bound": True,
        },
        "reported_not_gated_language_diagnostics": {
            "left_right_actions_equal": _actions_equal(
                outputs["left"],
                outputs["right"],
                np,
            ),
            "note": "G4 does not require a positive prompt effect.",
        },
        "records": records,
        "model_request_count": 3,
        "behavioral_episode_count": 0,
        "qualification_scope": "policy_session_only_no_behavioral_episode",
        "passed": passed,
        "status": "passed" if passed else "failed",
        "release_boundary": (
            "A pass establishes the C8 GR00T Bridge policy-session interface "
            "only. G5-G8 and a released runtime lock remain required."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--integration-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--backbone-revision", required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = run_g4(
        registry_path=args.registry.resolve(),
        integration_root=args.integration_root.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        backbone_path=args.backbone.resolve(),
        backbone_revision=args.backbone_revision,
        hf_home=args.hf_home.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "path": str(args.output),
                "status": payload["status"],
                "model_request_count": payload["model_request_count"],
                "sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
