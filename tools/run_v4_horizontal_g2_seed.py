#!/usr/bin/env python3
"""Run one zero-inference V4 DROID G2 reset/camera seed on RoboLab."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
FIXTURE_PROMPTS = {
    "horizontal": (
        "Place the cube so that the cube is left of the bowl. "
        "Use the robot's fixed viewpoint for left, right, front, and behind."
    ),
    "object_pair": (
        "Place the sponge so that the sponge is left of the tray. "
        "Use the robot's fixed viewpoint for left, right, front, and behind."
    ),
    "vertical": "Place the cube so that the cube is above the bowl.",
    "containment": "Place the cube so that the cube is inside the bowl.",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--robolab-root", type=Path, required=True)
    parser.add_argument("--reset-registry", type=Path, required=True)
    parser.add_argument("--reset-registry-sha256", required=True)
    parser.add_argument(
        "--fixture-id",
        choices=tuple(FIXTURE_PROMPTS),
        default="horizontal",
    )
    parser.add_argument("--environment-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-study-commit", required=True)
    parser.add_argument("--expected-robolab-commit", default=ROBOLAB_COMMIT)
    parser.add_argument("--expected-driver-version", required=True)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--pod")
    parser.add_argument("--pod-uid")
    parser.add_argument("--native-control-dt-s", type=float, required=True)
    parser.add_argument(
        "--reset-trace-only",
        action="store_true",
        help=(
            "Run an explicitly non-qualifying model-blind settle trace instead "
            "of producing a G2 receipt."
        ),
    )
    return parser


def _git_identity(path: Path, *, expected_commit: str | None = None) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    tracked_diff = subprocess.check_output(
        [
            "git",
            "-C",
            str(path),
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
        ],
        text=True,
    )
    if tracked_diff:
        raise RuntimeError(f"gate requires a clean tracked checkout: {path}")
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(
            f"checkout commit {commit} differs from required {expected_commit}: {path}"
        )
    return {"path": str(path.resolve()), "commit": commit, "tracked_diff_empty": True}


def _gpu_identity(*, expected_driver: str, gpu_uuid: str | None) -> dict[str, str]:
    lines = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    matching = [line for line in lines if gpu_uuid in line] if gpu_uuid else lines
    if len(matching) != 1:
        raise RuntimeError("exactly one assigned GPU must be visible")
    fields = [field.strip() for field in matching[0].split(",")]
    if len(fields) != 4 or fields[3] != expected_driver:
        raise RuntimeError(
            f"GPU driver differs from required {expected_driver}: {matching[0]}"
        )
    return {
        "index": fields[0],
        "uuid": fields[1],
        "name": fields[2],
        "driver_version": fields[3],
    }


def _raw_camera_arrays(raw_observation: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    from experiments.online_correction_v4.model_blind_g2 import (
        REQUIRED_POLICY_CAMERAS,
    )

    image_obs = raw_observation["image_obs"]
    arrays: dict[str, Any] = {}
    for camera in REQUIRED_POLICY_CAMERAS:
        value = image_obs[camera]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        arrays[camera] = np.ascontiguousarray(array)
    return arrays


def _write_camera_artifacts(
    *,
    raw_observation: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    from experiments.online_correction_v4.model_blind_g2 import (
        sha256_bytes,
        sha256_file,
    )

    arrays = _raw_camera_arrays(raw_observation)
    records: dict[str, Any] = {}
    panels: list[Any] = []
    target_height = min(int(array.shape[0]) for array in arrays.values())
    for camera, array in arrays.items():
        path = output_dir / f"{camera}.png"
        if array.dtype != np.uint8:
            raise RuntimeError(f"camera {camera} is not uint8")
        if not cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"failed to write camera artifact: {path}")
        decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            raise RuntimeError(f"failed to decode camera artifact: {path}")
        decoded_rgb = np.ascontiguousarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
        if decoded_rgb.shape != array.shape or not np.array_equal(decoded_rgb, array):
            raise RuntimeError(f"lossless camera artifact differs from raw pixels: {path}")
        width = round(array.shape[1] * target_height / array.shape[0])
        panel = cv2.resize(
            cv2.cvtColor(array, cv2.COLOR_RGB2BGR),
            (width, target_height),
        )
        cv2.putText(
            panel,
            camera,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
        records[camera] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "decoded_raw_array_sha256": sha256_bytes(decoded_rgb.tobytes()),
        }
    montage = np.concatenate(panels, axis=1)
    montage_path = output_dir / "policy_camera_montage.png"
    if not cv2.imwrite(str(montage_path), montage):
        raise RuntimeError("failed to write policy camera montage")
    records["montage"] = {
        "path": str(montage_path.resolve()),
        "sha256": sha256_file(montage_path),
        "bytes": montage_path.stat().st_size,
    }
    return records


def _write_axis_overlay_artifacts(
    *,
    raw_observation: Mapping[str, Any],
    axis_projection: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    from experiments.online_correction_v4.model_blind_g2 import sha256_file

    arrays = _raw_camera_arrays(raw_observation)
    camera_rows = axis_projection.get("camera_rows")
    if not isinstance(camera_rows, Mapping):
        raise RuntimeError("axis projection lacks camera rows")
    colors = {
        "left": (255, 128, 0),
        "front": (0, 165, 255),
        "up": (0, 255, 0),
    }
    records: dict[str, Any] = {}
    panels: list[Any] = []
    drawn: set[str] = set()
    target_height = min(int(array.shape[0]) for array in arrays.values())
    for camera, array in arrays.items():
        row = camera_rows.get(camera)
        if not isinstance(row, Mapping):
            raise RuntimeError(f"axis projection lacks camera row {camera}")
        image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        origin_raw = row.get("axis_origin_pixel_uv")
        endpoints = row.get("axis_endpoint_pixel_uv")
        in_frame = row.get("axis_fully_in_frame")
        if (
            not isinstance(origin_raw, list)
            or len(origin_raw) != 2
            or not isinstance(endpoints, Mapping)
            or not isinstance(in_frame, Mapping)
        ):
            raise RuntimeError(f"axis projection row is malformed for {camera}")
        origin = tuple(int(round(float(value))) for value in origin_raw)
        for axis, color in colors.items():
            endpoint_raw = endpoints.get(axis)
            if in_frame.get(axis) is not True:
                continue
            if not isinstance(endpoint_raw, list) or len(endpoint_raw) != 2:
                raise RuntimeError(f"axis endpoint is malformed for {camera}/{axis}")
            endpoint = tuple(int(round(float(value))) for value in endpoint_raw)
            cv2.arrowedLine(
                image,
                origin,
                endpoint,
                color,
                3,
                cv2.LINE_AA,
                tipLength=0.18,
            )
            cv2.putText(
                image,
                axis.upper(),
                endpoint,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            drawn.add(axis)
        cv2.putText(
            image,
            camera,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        path = output_dir / f"axis_overlay_{camera}.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"failed to write axis overlay: {path}")
        decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            raise RuntimeError(f"failed to decode axis overlay: {path}")
        width = round(image.shape[1] * target_height / image.shape[0])
        panels.append(cv2.resize(image, (width, target_height)))
        records[camera] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "axes_drawn": sorted(
                axis for axis in colors if in_frame.get(axis) is True
            ),
        }
    if drawn != set(colors):
        raise RuntimeError("axis overlays do not jointly contain left/front/up")
    montage = np.concatenate(panels, axis=1)
    montage_path = output_dir / "axis_overlay_montage.png"
    if not cv2.imwrite(str(montage_path), montage):
        raise RuntimeError("failed to write axis overlay montage")
    records["montage"] = {
        "path": str(montage_path.resolve()),
        "sha256": sha256_file(montage_path),
        "bytes": montage_path.stat().st_size,
        "axes_drawn": sorted(drawn),
    }
    return records


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    from experiments.online_correction_v4.model_blind_g2 import (
        canonical_json_bytes,
        sha256_bytes,
    )

    body = canonical_json_bytes(dict(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": str(path.resolve()),
        "sha256": sha256_bytes(body),
        "bytes": len(body),
    }


def _host_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else list(value)


def _reset_trace_sample(backend: Any, *, control_tick: int) -> dict[str, Any]:
    world = backend.modules["get_world"](backend.env)
    objects: dict[str, Any] = {}
    for name in (*backend.settle_objects, "table"):
        try:
            position, quaternion = world.get_pose(name, env_id=0)
            objects[name] = {
                "position_world_xyz_m": _host_list(position),
                "quaternion_world_wxyz": _host_list(quaternion),
                "velocity_world_xyz_rad_s": _host_list(
                    world.get_velocity(name, env_id=0)
                ),
            }
        except Exception as exc:
            objects[name] = {
                "sample_error": f"{type(exc).__name__}: {exc}"
            }
    sample: dict[str, Any] = {
        "control_tick": control_tick,
        "simulation_time_s": control_tick * backend.control_dt_s,
        "objects": objects,
    }
    try:
        sample["contact_force_n_by_sensor"] = backend.g3_contact_force_evidence()
    except Exception as exc:
        sample["contact_probe_error"] = f"{type(exc).__name__}: {exc}"
    return sample


def _run_reset_trace(
    *,
    env: Any,
    environment_seed: int,
    output_dir: Path,
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    from experiments.online_correction_v4.droid_reset import (
        SETTLE_STEPS,
        STABILITY_WINDOW_STEPS,
    )

    backend = env.backend
    backend.reset(seed=environment_seed)
    robot = backend.env.scene["robot"]
    samples = [_reset_trace_sample(backend, control_tick=0)]
    hold = backend.hold_action_tensor()
    total_steps = SETTLE_STEPS + STABILITY_WINDOW_STEPS
    for control_tick in range(1, total_steps + 1):
        backend.step(hold)
        samples.append(_reset_trace_sample(backend, control_tick=control_tick))

    def _bounds_or_error(name: str) -> dict[str, Any]:
        try:
            return backend.g3_world_aabb(name)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    payload = {
        "schema_version": "v4-object-pair-g2-reset-settle-trace-v1",
        "campaign_id": "online_correction_v4",
        "fixture_id": backend.config.fixture.fixture_id,
        "environment_seed": environment_seed,
        "status": "diagnostic_only_not_a_g2_receipt",
        "authorizes_behavioral_inference": False,
        "authorizes_g3_execution": False,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "settle_steps": SETTLE_STEPS,
        "stability_window_steps": STABILITY_WINDOW_STEPS,
        "runtime_identity": dict(runtime_identity),
        "robot": {
            "body_names": list(getattr(robot, "body_names", ())),
            "body_positions_world_m_at_end": _host_list(robot.data.body_pos_w),
            "world_aabb_m_at_end": _bounds_or_error("robot"),
        },
        "table_world_aabb_m_at_end": _bounds_or_error("table"),
        "object_world_aabb_m_at_end": {
            name: _bounds_or_error(name) for name in backend.settle_objects
        },
        "samples": samples,
    }
    return _write_json(output_dir / "reset_settle_trace.json", payload)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    prompt = FIXTURE_PROMPTS[args.fixture_id]
    study_root = args.study_root.resolve()
    robolab_root = args.robolab_root.resolve()
    registry_path = args.reset_registry.resolve()
    if str(study_root) not in sys.path:
        sys.path.insert(0, str(study_root))
    output_raw = args.output_dir or (
        Path(os.environ["EPISODE_OUTPUT_DIR"])
        if os.environ.get("EPISODE_OUTPUT_DIR")
        else None
    )
    if output_raw is None:
        raise RuntimeError("--output-dir or EPISODE_OUTPUT_DIR is required")
    output_dir = output_raw.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite gate output: {output_dir}")
    output_dir.mkdir(parents=True)

    env = None
    try:
        from experiments.online_correction_v4.droid_contract import sha256_bytes
        from experiments.online_correction_v4.droid_robolab import (
            ResetFixtureBinding,
            RoboLabSession,
            build_live_robolab_env,
            close_live_droid_stack,
            write_queue_row,
        )
        from experiments.online_correction_v4.droid_task_files.binding import (
            sha256_file,
        )
        from experiments.online_correction_v4.droid_task_files.reset_registry import (
            MODEL_BLIND_CANDIDATE_STATUS,
            load_reset_registry,
        )
        from experiments.online_correction_v4.model_blind_g2 import (
            REQUIRED_POLICY_CAMERAS,
            axis_projection_evidence,
            camera_view_evidence,
            canonical_json_bytes,
            compile_seed_receipt,
        )

        if sha256_file(registry_path) != args.reset_registry_sha256:
            raise RuntimeError("reset registry SHA-256 mismatch")
        study_identity = _git_identity(
            study_root, expected_commit=args.expected_study_commit
        )
        robolab_identity = _git_identity(
            robolab_root, expected_commit=args.expected_robolab_commit
        )
        gpu_identity = _gpu_identity(
            expected_driver=args.expected_driver_version,
            gpu_uuid=args.gpu_uuid,
        )
        pod_name = args.pod or os.environ.get("POD_NAME")
        pod_uid = args.pod_uid or os.environ.get("POD_UID")
        if not pod_name or not pod_uid:
            raise RuntimeError("--pod/--pod-uid or POD_NAME/POD_UID are required")
        registry = load_reset_registry(
            registry_path=str(registry_path),
            registry_sha256=args.reset_registry_sha256,
            required_status=MODEL_BLIND_CANDIDATE_STATUS,
            expected_fixture_id=args.fixture_id,
        )
        if args.environment_seed not in registry.positions_by_env_seed:
            raise RuntimeError("environment seed is absent from reset registry")

        episode_id = (
            f"online-correction-v4-g2-{args.fixture_id.replace('_', '-')}-"
            f"{args.environment_seed}"
        )
        prompt_sha256 = sha256_bytes(prompt.encode("utf-8"))
        queue_row, queue_row_sha256 = write_queue_row(
            output_dir=output_dir,
            episode_id=episode_id,
            fixture_id=args.fixture_id,
            prompt_text=prompt,
            prompt_sha256=prompt_sha256,
            env_seed=args.environment_seed,
            goal="left",
        )
        runtime_identity = {
            "study_checkout": study_identity,
            "robolab_checkout": robolab_identity,
            "gpu": gpu_identity,
            "pod": pod_name,
            "pod_uid": pod_uid,
            "gate_entrypoint_sha256": sha256_file(Path(__file__).resolve()),
            "gate_core_sha256": sha256_file(
                study_root
                / "experiments/online_correction_v4/model_blind_g2.py"
            ),
            "droid_robolab_sha256": sha256_file(
                study_root
                / "experiments/online_correction_v4/droid_robolab.py"
            ),
            "reset_registry_sha256": args.reset_registry_sha256,
        }
        runtime_identity_sha256 = sha256_bytes(
            canonical_json_bytes(runtime_identity)
        )
        fixture = ResetFixtureBinding(
            fixture_id=args.fixture_id,
            reset_registry_sha256=args.reset_registry_sha256,
            reset_registry_uri=f"file://{registry_path}",
        )
        os.environ["V4_DROID_RENDERER"] = "realtime"
        os.environ["V4_DROID_RENDERING_MODE"] = "balanced"
        os.environ["ONLINE_CORRECTION_V4_OUTPUT_DIR"] = str(
            (output_dir / "robolab_native").resolve()
        )
        RoboLabSession.begin_episode(episode_id)
        env = build_live_robolab_env(
            fixture=fixture,
            env_seed=args.environment_seed,
            episode_id=episode_id,
            goal="left",
            prompt_text=prompt,
            prompt_sha256=prompt_sha256,
            policy_id="model_blind_no_policy",
            queue_row_path=queue_row,
            queue_row_sha256=queue_row_sha256,
            output_dir=output_dir / "robolab_native",
            locked_native_control_dt_s=args.native_control_dt_s,
            g3_contact_probe=args.reset_trace_only,
        )
        if args.reset_trace_only:
            trace_record = _run_reset_trace(
                env=env,
                environment_seed=args.environment_seed,
                output_dir=output_dir,
                runtime_identity=runtime_identity,
            )
            print(json.dumps({"diagnostic_trace": trace_record}, indent=2))
            return 0
        env.reset(seed=args.environment_seed)
        env.reset(seed=args.environment_seed)
        initial_state_sha256 = sha256_bytes(env.capture_observation_bytes())
        attestation = env.reset_proxy.finalize_attestation(
            prompt_sha256=prompt_sha256,
            runtime_identity_sha256=runtime_identity_sha256,
            initial_state_sha256=initial_state_sha256,
        )
        physical = env.backend.physical_reset_payload()
        raw_observation = env.backend.latest_raw_observation()
        camera_views = camera_view_evidence(raw_observation)
        camera_geometry = env.backend.camera_geometry_payload(
            REQUIRED_POLICY_CAMERAS
        )
        axis_projection = axis_projection_evidence(
            physical_reset=physical,
            camera_geometry=camera_geometry,
            camera_views=camera_views,
            reference_object=registry.object_roles["reference"].scene_object,
            fixture_id=args.fixture_id,
        )

        artifacts = {
            "queue_row": {
                "path": str(queue_row.resolve()),
                "sha256": queue_row_sha256,
                "bytes": queue_row.stat().st_size,
            },
            "reset_attestation": _write_json(
                output_dir / "reset_attestation.json", attestation
            ),
            "physical_reset": _write_json(
                output_dir / "physical_reset.json", physical
            ),
            "policy_camera_images": _write_camera_artifacts(
                raw_observation=raw_observation,
                output_dir=output_dir,
            ),
            "axis_overlay_images": _write_axis_overlay_artifacts(
                raw_observation=raw_observation,
                axis_projection=axis_projection,
                output_dir=output_dir,
            ),
        }
        receipt = compile_seed_receipt(
            env_seed=args.environment_seed,
            episode_id=episode_id,
            registry=registry,
            reset_attestation=attestation,
            physical_reset=physical,
            camera_views=camera_views,
            camera_geometry=camera_geometry,
            expected_native_control_dt_s=args.native_control_dt_s,
            runtime_identity=runtime_identity,
            artifacts=artifacts,
            fixture_id=args.fixture_id,
        )
        receipt_record = _write_json(
            output_dir / "g2_seed_receipt.json",
            receipt,
        )
        print(json.dumps({"passed": True, "receipt": receipt_record}, indent=2))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": (
                f"v4-{args.fixture_id.replace('_', '-')}-g2-"
                "infrastructure-failure-v1"
            ),
            "campaign_id": "online_correction_v4",
            "fixture_id": args.fixture_id,
            "environment_seed": args.environment_seed,
            "status": "infrastructure_invalid",
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(output_dir / "infrastructure_failure.json", failure)
        print(
            f"[V4 {args.fixture_id} G2] infrastructure failure: {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            from experiments.online_correction_v4.droid_robolab import (
                close_live_droid_stack,
            )

            close_live_droid_stack()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
