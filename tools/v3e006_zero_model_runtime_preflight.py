#!/usr/bin/env python3
"""Exercise the E004 Isaac runtime without constructing a state or loading a policy."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from isaaclab.app import AppLauncher


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": _sha256(resolved)}


parser = argparse.ArgumentParser()
parser.add_argument("--study-root", type=Path, required=True)
parser.add_argument("--robolab-root", type=Path, required=True)
parser.add_argument("--expected-study-commit", required=True)
parser.add_argument("--expected-robolab-commit", required=True)
parser.add_argument("--e004-candidate", type=Path, required=True)
parser.add_argument("--e004-candidate-sha256", required=True)
parser.add_argument("--ood-freeze", type=Path, required=True)
parser.add_argument("--ood-freeze-sha256", required=True)
parser.add_argument("--e004-reset-reference", type=Path, required=True)
parser.add_argument("--e004-reset-reference-sha256", required=True)
parser.add_argument("--runtime-contract", type=Path, required=True)
parser.add_argument("--runtime-contract-sha256", required=True)
parser.add_argument("--diagnostic-input", action="append", type=Path, default=[])
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--pod", required=True)
parser.add_argument("--pod-uid", required=True)
parser.add_argument("--gpu-uuid", required=True)
parser.add_argument("--container-image", required=True)
parser.add_argument("--container-id", required=True)
parser.add_argument("--driver-version", required=True)
parser.add_argument("--num-envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

if args.output.exists():
    parser.error(f"refusing to overwrite preflight evidence: {args.output}")
if args.num_envs != 1 or not args.headless:
    parser.error("preflight requires one headless environment")
if args.rendering_mode != "balanced" or args.device != "cuda:0":
    parser.error("preflight requires the E004 realtime/balanced/cuda:0 runtime")

study_root = args.study_root.resolve()
robolab_root = args.robolab_root.resolve()
sys.path.insert(0, str(study_root))
from experiments.v3.phase_e.canonical_stage_localization_v3e006.runtime_contract import (  # noqa: E402
    expected_observation,
    load_runtime_contract,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006 import preflight_pose  # noqa: E402
from experiments.v3.phase_e.canonical_stage_localization_v3e006.preflight_pose import (  # noqa: E402
    scalar_z_from_world_pose,
)

study_commit = subprocess.check_output(["git", "-C", str(study_root), "rev-parse", "HEAD"], text=True).strip()
robolab_commit = subprocess.check_output(["git", "-C", str(robolab_root), "rev-parse", "HEAD"], text=True).strip()
if study_commit != args.expected_study_commit or robolab_commit != args.expected_robolab_commit:
    parser.error("source checkout differs from the expected immutable commit")
if subprocess.check_output(["git", "-C", str(study_root), "status", "--porcelain"], text=True):
    parser.error("study checkout is not clean")
bound_inputs: dict[str, dict[str, object]] = {}
for label, path, expected_sha256 in (
    ("e004_candidate", args.e004_candidate, args.e004_candidate_sha256),
    ("ood_freeze", args.ood_freeze, args.ood_freeze_sha256),
    ("e004_reset_reference", args.e004_reset_reference, args.e004_reset_reference_sha256),
    ("runtime_contract", args.runtime_contract, args.runtime_contract_sha256),
):
    if not path.is_file() or _sha256(path) != expected_sha256:
        parser.error(f"hash-bound {label} input is missing or changed")
    bound_inputs[label] = _binding(path)
runtime_contract = load_runtime_contract(
    args.runtime_contract,
    args.runtime_contract_sha256,
    study_root=study_root,
    external_roots=(robolab_root,),
)
runtime_observation = expected_observation(runtime_contract)
for path in args.diagnostic_input:
    if not path.is_file():
        parser.error(f"diagnostic input is missing: {path}")

expected_ld_library_path = (
    "/data/users/ali/vla_wam/envs/robolab-native-libs-ubuntu2204/usr/lib/x86_64-linux-gnu:"
    "/data/users/ali/glvnd/lib:"
    "/data/users/ali/vla_wam/envs/fastwam-native-libs/lib:"
    "/usr/lib/x86_64-linux-gnu"
)
if os.environ.get("VK_ICD_FILENAMES") != "/etc/vulkan/icd.d/nvidia_icd.json":
    parser.error("Vulkan ICD differs from the proven E004 runtime")
if os.environ.get("LD_LIBRARY_PATH") != expected_ld_library_path:
    parser.error("native-library path differs from the proven E004 runtime")
for key in ("HOME", "XDG_CACHE_HOME", "WARP_CACHE_PATH", "MPLCONFIGDIR", "TMPDIR"):
    value = os.environ.get(key)
    if not value or not Path(value).is_dir() or not os.access(value, os.W_OK):
        parser.error(f"isolated writable runtime directory is invalid: {key}")

simulation_app = AppLauncher(args).app
source = Path(__file__).resolve()
environment = {
    key: os.environ.get(key)
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "VK_ICD_FILENAMES",
        "LD_LIBRARY_PATH",
        "HOME",
        "XDG_CACHE_HOME",
        "WARP_CACHE_PATH",
        "MPLCONFIGDIR",
        "TMPDIR",
    )
}
base_report = {
    "schema_version": "vla-wam-shared-v3e006-zero-model-runtime-preflight-v1",
    "scope": "generic runtime health only; not a state candidate, E004 scene validation, or behavioral release",
    "model_request_count": 0,
    "behavioral_episode_count": 0,
    "state_candidate_count": 0,
    "study_commit": study_commit,
    "robolab_commit": robolab_commit,
    "source": _binding(source),
    "pose_helper_source": _binding(Path(preflight_pose.__file__).resolve()),
    "bound_inputs": bound_inputs,
    "runtime_contract_observation": runtime_observation,
    "app_launcher_runtime": {
        "renderer": "realtime",
        "rendering_mode": args.rendering_mode,
        "device": args.device,
        "headless": args.headless,
    },
    "diagnostic_inputs": [_binding(path) for path in args.diagnostic_input],
    "invocation": sys.argv,
    "environment": environment,
    "lane": {
        "pod": args.pod,
        "pod_uid": args.pod_uid,
        "gpu_uuid": args.gpu_uuid,
        "container_image": args.container_image,
        "container_id": args.container_id,
        "driver_version": args.driver_version,
        "python": sys.executable,
    },
}
health_failure: BaseException | None = None
pose_api_source: dict[str, object] | None = None

try:
    import numpy as np
    import omni.replicator.core as rep
    import torch
    import warp as wp
    import isaacsim.core.utils.prims as prim_utils
    from isaacsim.core.api.world import World
    from isaacsim.core.prims import SingleGeometryPrim, SingleRigidPrim
    from pxr import UsdGeom

    pose_api_path = Path(inspect.getsourcefile(SingleRigidPrim.get_world_pose) or "").resolve()
    pose_api_source = _binding(pose_api_path)
    if pose_api_source != {
        "path": str(pose_api_path),
        "bytes": 16979,
        "sha256": "30cad6463a0ffa514ec357640f3fcff2fe076d241179c809f5f05e2b85b41333",
    }:
        raise RuntimeError("installed Isaac single-prim pose API source differs")

    if not torch.cuda.is_available():
        raise RuntimeError("torch CUDA unavailable")
    tensor = torch.arange(16, device="cuda:0", dtype=torch.float32)
    torch_sum = float(tensor.sum().cpu())
    if torch_sum != 120.0:
        raise RuntimeError("CUDA tensor result changed")
    wp.init()
    warp_devices = [str(device) for device in wp.get_devices()]

    world = World(physics_dt=0.01, rendering_dt=0.01, backend="torch", device="cuda")
    world.scene.add_default_ground_plane()
    prim_utils.create_prim("/World/Light", "SphereLight", translation=(2.0, 2.0, 4.0))
    prim_utils.create_prim("/World/Cube", "Cube", translation=(0.0, 0.0, 1.0), scale=(0.2, 0.2, 0.2))
    SingleGeometryPrim("/World/Cube", collision=True)
    cube = world.scene.add(SingleRigidPrim("/World/Cube", name="cube", mass=1.0))
    camera_prim = prim_utils.create_prim(
        "/World/Camera",
        "Camera",
        translation=(2.5, 2.5, 2.5),
        orientation=(0.33985113, 0.17591988, 0.42470818, 0.82047324),
    )
    UsdGeom.Camera(camera_prim)
    render_product = rep.create.render_product("/World/Camera", resolution=(160, 120))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    rgb.attach(render_product)

    world.reset()
    z_initial = scalar_z_from_world_pose(cube.get_world_pose())
    for _ in range(60):
        world.step(render=True)
    frame = np.asarray(rgb.get_data())
    z_final = scalar_z_from_world_pose(cube.get_world_pose())
    if z_final >= z_initial - 0.05:
        raise RuntimeError("minimal rigid-body physics did not advance")
    if frame.shape[:2] != (120, 160) or frame.size == 0 or int(np.count_nonzero(frame)) == 0:
        raise RuntimeError(f"minimal RTX camera capture invalid: {frame.shape}")

    report = {
        **base_report,
        "status": "passed_generic_zero_model_cuda_vulkan_isaac_physics_render_health_preflight",
        "passed": True,
        "installed_pose_api_source": pose_api_source,
        "cuda": {
            "available": True,
            "device_name": torch.cuda.get_device_name(0),
            "tensor_sum": torch_sum,
            "warp_devices": warp_devices,
        },
        "physics": {"cube_z_initial": z_initial, "cube_z_final": z_final, "fell": True},
        "render": {
            "rgb_shape": list(frame.shape),
            "rgb_nonzero": int(np.count_nonzero(frame)),
            "rgb_mean": float(frame.astype(np.float64).mean()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "sha256": _sha256(args.output)}, sort_keys=True))
except BaseException as exc:
    health_failure = exc
    failure_report = {
        **base_report,
        "status": "infrastructure_invalid_zero_model_runtime_health_preflight",
        "passed": False,
        "installed_pose_api_source": pose_api_source,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(failure_report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    traceback.print_exc()
finally:
    try:
        simulation_app.close()
    except BaseException as close_error:
        if health_failure is None:
            raise
        print(f"SimulationApp.close raised after retained health failure: {type(close_error).__name__}: {close_error}", file=sys.stderr)

if health_failure is not None:
    raise health_failure.with_traceback(health_failure.__traceback__)
