"""GR00T Bridge observation processing for C8 second_stack episodes."""

from __future__ import annotations

from typing import Any, Mapping


class GrootObservationError(RuntimeError):
    """Raised when C8 observations cannot be processed for GR00T Bridge."""


def processed_observation_from_env(env: Any) -> dict[str, Any]:
    """Build the G4-verified GR00T observation dict from a live SimplerEnv handle."""
    from experiments.online_correction_v4.second_stack import unwrap_simpler_env

    import cv2
    import numpy as np
    from transforms3d import euler as te, quaternions as tq

    raw = unwrap_simpler_env(env)
    get_obs = getattr(raw, "get_obs", None)
    if not callable(get_obs):
        raise GrootObservationError("C8 raw environment does not expose get_obs")
    raw_observation = get_obs()
    color = np.asarray(raw_observation["image"]["3rd_view_camera"]["Color"])[..., :3]
    if color.dtype != np.uint8:
        color = np.asarray(color, dtype=np.float32)
        if color.size and float(np.nanmax(color)) <= 1.0:
            color = color * 255.0
        color = np.clip(color, 0.0, 255.0).astype(np.uint8)
    proprio = np.asarray(raw_observation["agent"]["eef_pos"], dtype=np.float64)
    if proprio.shape != (8,):
        raise GrootObservationError("C8 Bridge eef_pos must have shape (8,)")
    rotation = tq.quat2mat(proprio[3:7])
    bridge_default_rotation = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    roll, pitch, yaw = te.mat2euler(rotation @ bridge_default_rotation.T)
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


def batched_observation(
    *,
    observation: Mapping[str, Any],
    modality_configs: Mapping[str, Any],
    prompt: str,
    np: Any,
) -> dict[str, Any]:
    """Expand one processed observation into the GR00T batched request layout."""
    batched: dict[str, Any] = {}
    for key in modality_configs["video"].modality_keys:
        flat_key = f"video.{key}"
        frame = np.asarray(observation[flat_key], dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise GrootObservationError(f"{flat_key} is not an HxWx3 RGB frame")
        horizon = len(modality_configs["video"].delta_indices)
        batched[flat_key] = np.repeat(frame[None, None, ...], horizon, axis=1)
    for key in modality_configs["state"].modality_keys:
        flat_key = f"state.{key}"
        state = np.asarray(observation[flat_key], dtype=np.float32).reshape(1, 1, -1)
        horizon = len(modality_configs["state"].delta_indices)
        batched[flat_key] = np.repeat(state, horizon, axis=1)
    language_keys = list(modality_configs["language"].modality_keys)
    if not language_keys:
        raise GrootObservationError("C8 checkpoint has no language modality")
    batched[language_keys[0]] = (prompt,)
    return batched


GROOT_ACTION_COMPONENT_KEYS = (
    "action.x",
    "action.y",
    "action.z",
    "action.roll",
    "action.pitch",
    "action.yaw",
    "action.gripper",
)


def tuple_to_simpler_env_action(action: tuple[float, ...]) -> dict[str, Any]:
    """Convert one V4 action tuple into the SimplerEnv gym step dict."""
    import numpy as np

    if len(action) < len(GROOT_ACTION_COMPONENT_KEYS):
        raise GrootObservationError(
            f"action tuple must have at least {len(GROOT_ACTION_COMPONENT_KEYS)} dims"
        )
    return {
        key: np.asarray([float(action[index])], dtype=np.float32)
        for index, key in enumerate(GROOT_ACTION_COMPONENT_KEYS)
    }


def normalize_groot_action_chunk(raw: Any, expected_shape: tuple[int, int]) -> tuple[tuple[float, ...], ...]:
    """Convert GR00T Bridge action dict output into V4 (horizon, 8) tuples."""
    import math

    tolist = getattr(raw, "tolist", None)
    if callable(tolist):
        raw = tolist()
    rows, cols = expected_shape
    if isinstance(raw, Mapping):
        component_order = GROOT_ACTION_COMPONENT_KEYS
        import numpy as np

        arrays = []
        for key in component_order:
            if key not in raw:
                raise GrootObservationError(f"GR00T action missing required key: {key}")
            value = np.asarray(raw[key], dtype=np.float32)
            if value.ndim == 3 and value.shape[0] == 1 and value.shape[2] == 1:
                arrays.append(value[0, :, 0])
            elif value.ndim == 2 and value.shape[0] == rows:
                arrays.append(value[:, 0] if value.shape[1] == 1 else value[0])
            else:
                raise GrootObservationError(f"GR00T action {key} has unexpected shape {value.shape}")
        if len(arrays[0]) != rows:
            raise GrootObservationError(f"GR00T action horizon must be {rows}")
        normalized: list[tuple[float, ...]] = []
        for step in range(rows):
            values = [float(arrays[index][step]) for index in range(len(arrays))]
            if cols == 8 and len(values) == 7:
                values.append(0.0)
            if len(values) != cols:
                raise GrootObservationError(f"GR00T action step must have {cols} dims")
            if not all(math.isfinite(value) for value in values):
                raise GrootObservationError("GR00T action values must be finite")
            normalized.append(tuple(values))
        return tuple(normalized)
    if not isinstance(raw, (list, tuple)) or isinstance(raw, (str, bytes)):
        raise GrootObservationError(f"GR00T action chunk must be a mapping or nested sequence {expected_shape}")
    if len(raw) != rows:
        raise GrootObservationError(f"GR00T action chunk must be shape {expected_shape}")
    normalized_seq: list[tuple[float, ...]] = []
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) != cols:
            raise GrootObservationError(f"GR00T action chunk must be shape {expected_shape}")
        values = [float(value) for value in row]
        if not all(math.isfinite(value) for value in values):
            raise GrootObservationError("GR00T action values must be finite")
        normalized_seq.append(tuple(values))
    return tuple(normalized_seq)
