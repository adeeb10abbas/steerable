#!/usr/bin/env python3
"""Run the frozen DreamZero V2-A015 action-guidance release probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import websockets.sync.client

from policies.dreamzero.client import MsgPackNumpy

from v2_robolab_client import LEFT, OFFICIAL_NOISE_SEED, RIGHT


AMENDMENT_ID = "V2-A015"
OFFICIAL_COMMIT = "ab790c198fbce33503358efbbd4187ce9a89adf3"
V2A015_PATCH_SHA256 = (
    "de2b82a1c9f81ee4751fb384158775865f5e5e8fb622134cae3b2c4e8b2a2cc0"
)
V2A015_PATCHED_ACTION_HEAD_SHA256 = (
    "65dc9873aef37563dedf3787fd7b59e0a6d50e575775e38b70edd9e38489f9b8"
)
BASE_INSTRUMENTED_SERVER_SHA256 = (
    "2b0a83e21ee921527a2d0fc5cb9344d0b1f5950ff88906fd441fac2421c5a95f"
)
DERIVED_INSTRUMENTED_SERVER_SHA256 = (
    "5f9932acc77fe8fc80157622ef3e1db5b552bd31f322768330f2000f9af7d09d"
)
SERVER_SCHEMA = "vla-wam-shared-v2-dreamzero-v2a015-server-contract-v1"
FUTURE_SCHEMA = "vla-wam-shared-v2-dreamzero-v2a015-future-retention-v1"
OFFICIAL_PROBE_SCHEMA = "vla-wam-shared-v2-dreamzero-exact-repeat-probe-v1"
PROBE_SCHEMA = "vla-wam-shared-v2-dreamzero-v2a015-fixed-observation-probe-v1"
CONDITIONS = (("left_a", LEFT), ("left_b", LEFT), ("right", RIGHT))
EXPECTED_ACTION_SHAPE = (24, 8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_data_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _tensor_data_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    byte_view = tensor.view(torch.uint8).numpy()
    return hashlib.sha256(byte_view.tobytes()).hexdigest()


def _rms(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"Cannot compare shapes {left.shape} and {right.shape}")
    delta = left.astype(np.float64) - right.astype(np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"Cannot compare shapes {left.shape} and {right.shape}")
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _tensor_float_array(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().float().numpy()


def _tensor_bit_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(
        left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and torch.equal(left.detach().cpu(), right.detach().cpu())
    )


def _resolve_record_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    candidates = (Path.cwd() / path, manifest_path.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _connect(uri: str, packer: MsgPackNumpy):
    connection = websockets.sync.client.connect(
        uri,
        compression=None,
        max_size=None,
        open_timeout=300,
        ping_interval=60,
        ping_timeout=600,
    )
    metadata = packer.unpack(connection.recv(timeout=300))
    return connection, metadata


def _recv_action(connection: Any, packer: MsgPackNumpy) -> np.ndarray:
    raw = connection.recv(timeout=600)
    if isinstance(raw, str):
        raise RuntimeError(f"DreamZero server error:\n{raw}")
    response = packer.unpack(raw)
    if isinstance(response, dict):
        response = response.get("actions", response)
    action = np.asarray(response, dtype=np.float32)
    if action.shape != EXPECTED_ACTION_SHAPE:
        raise ValueError(
            f"DreamZero returned {action.shape}; expected {EXPECTED_ACTION_SHAPE}"
        )
    if not np.isfinite(action).all():
        raise ValueError("DreamZero returned a non-finite fixed-observation action")
    return action


def _check_reset_reply(raw: Any, packer: MsgPackNumpy) -> None:
    if isinstance(raw, str):
        if raw.lower().startswith("error"):
            raise RuntimeError(f"DreamZero reset failed: {raw}")
        return
    reply = packer.unpack(raw)
    if isinstance(reply, dict) and reply.get("error"):
        raise RuntimeError(f"DreamZero reset failed: {reply['error']}")


def _wait_for_v2a015_manifest(
    path: Path,
    *,
    action_cfg_scale: float,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    """Wait past the base manifest write until the derived rewrite is complete."""
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                manifest = json.loads(path.read_text())
                if (
                    manifest.get("schema_version") == FUTURE_SCHEMA
                    and manifest.get("amendment_id") == AMENDMENT_ID
                    and float(manifest.get("action_cfg_style_scale", -1.0))
                    == action_cfg_scale
                    and manifest.get("source_patch_sha256")
                    == V2A015_PATCH_SHA256
                    and manifest.get("patched_action_head_sha256")
                    == V2A015_PATCHED_ACTION_HEAD_SHA256
                ):
                    return manifest
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
                last_error = exc
        time.sleep(0.25)
    detail = f"; last read error: {last_error}" if last_error else ""
    raise TimeoutError(f"Timed out waiting for finalized V2-A015 manifest: {path}{detail}")


def _load_server_record(
    manifest: dict[str, Any],
    *,
    prompt: str,
    action_cfg_scale: float,
) -> tuple[np.ndarray, torch.Tensor, dict[str, Any]]:
    if manifest.get("request_count") != 1 or len(manifest.get("requests", [])) != 1:
        raise ValueError("Each fixed-observation session must retain exactly one request")
    record = manifest["requests"][0]
    if record.get("prompt") != prompt:
        raise ValueError("V2-A015 server retained the wrong prompt")
    if float(record.get("action_cfg_style_scale", -1.0)) != action_cfg_scale:
        raise ValueError("V2-A015 request record contains the wrong action scale")

    action_entry = record.get("returned_action")
    if not isinstance(action_entry, dict):
        raise ValueError("V2-A015 manifest is missing its returned_action record")
    action_path = Path(action_entry["path"])
    if _sha256(action_path) != action_entry["sha256"]:
        raise ValueError(f"Returned-action hash mismatch: {action_path}")
    action = np.load(action_path, allow_pickle=False)
    if (
        action.shape != EXPECTED_ACTION_SHAPE
        or list(action.shape) != action_entry["shape"]
    ):
        raise ValueError(f"Returned-action shape mismatch: {action_path}")
    if not np.isfinite(action).all():
        raise ValueError(f"Returned action contains non-finite values: {action_path}")

    latent_entry = record.get("latent_video")
    if not isinstance(latent_entry, dict):
        raise ValueError("V2-A015 manifest is missing its latent_video record")
    latent_path = Path(latent_entry["path"])
    if _sha256(latent_path) != latent_entry["sha256"]:
        raise ValueError(f"Latent-future hash mismatch: {latent_path}")
    latent = torch.load(latent_path, map_location="cpu", weights_only=True)
    if list(latent.shape) != latent_entry["shape"]:
        raise ValueError(f"Latent-future shape mismatch: {latent_path}")
    if not bool(torch.isfinite(latent.float()).all()):
        raise ValueError(f"Latent future contains non-finite values: {latent_path}")
    return action, latent, record


def _load_comparison_condition(
    manifest_path: Path,
    record: dict[str, Any],
) -> tuple[np.ndarray, torch.Tensor]:
    action_path = _resolve_record_path(record["action_path"], manifest_path)
    if _sha256(action_path) != record["action_sha256"]:
        raise ValueError(f"Comparison action hash mismatch: {action_path}")
    action = np.load(action_path, allow_pickle=False)
    if action.shape != EXPECTED_ACTION_SHAPE or not np.isfinite(action).all():
        raise ValueError(
            "Comparison action violates the [24,8] finite contract: "
            f"{action_path}"
        )

    latent_path = _resolve_record_path(record["latent_path"], manifest_path)
    if _sha256(latent_path) != record["latent_sha256"]:
        raise ValueError(f"Comparison latent hash mismatch: {latent_path}")
    latent = torch.load(latent_path, map_location="cpu", weights_only=True)
    if not bool(torch.isfinite(latent.float()).all()):
        raise ValueError(f"Comparison latent contains non-finite values: {latent_path}")
    return action, latent


def _compare_probe(
    comparison_path: Path,
    *,
    action_cfg_scale: float,
    arrays: dict[str, np.ndarray],
    latents: dict[str, torch.Tensor],
) -> dict[str, Any]:
    comparison = json.loads(comparison_path.read_text())
    schema = comparison.get("schema_version")
    if action_cfg_scale == 1.0:
        if schema != OFFICIAL_PROBE_SCHEMA or comparison.get("passed") is not True:
            raise ValueError(
                "Scale 1 must compare with the passed archived official V2-A007 probe"
            )
        reference_kind = "archived_official_v2_a007"
        reference_scale = 1.0
        equality_required = True
    else:
        if schema == OFFICIAL_PROBE_SCHEMA:
            reference_kind = "archived_official_v2_a007_scale_1_equivalent"
            reference_scale = 1.0
        elif schema == PROBE_SCHEMA and float(
            comparison.get("action_cfg_style_scale", -1.0)
        ) == 1.0:
            if comparison.get("release_gate_passed") is not True:
                raise ValueError(
                    "Scale-2 comparison requires a passed V2-A015 scale-1 probe"
                )
            reference_kind = "v2_a015_scale_1_overlay"
            reference_scale = 1.0
        else:
            raise ValueError("Scale 2 must compare with an action-scale-1 DreamZero probe")
        equality_required = False

    reference_records = comparison.get("records", {})
    by_condition: dict[str, dict[str, Any]] = {}
    all_action_equal = True
    all_latent_equal = True
    for label, prompt in CONDITIONS:
        record = reference_records.get(label)
        if not isinstance(record, dict) or record.get("prompt") != prompt:
            raise ValueError(f"Comparison probe is missing the exact {label} prompt record")
        reference_action, reference_latent = _load_comparison_condition(
            comparison_path, record
        )
        action_equal = bool(
            arrays[label].dtype == reference_action.dtype
            and np.array_equal(arrays[label], reference_action)
        )
        latent_equal = _tensor_bit_equal(latents[label], reference_latent)
        all_action_equal = all_action_equal and action_equal
        all_latent_equal = all_latent_equal and latent_equal
        by_condition[label] = {
            "prompt": prompt,
            "action_array_equal": action_equal,
            "action_rms": _rms(arrays[label], reference_action),
            "action_max_abs": _max_abs(arrays[label], reference_action),
            "latent_tensor_equal": latent_equal,
            "latent_rms": _rms(
                _tensor_float_array(latents[label]),
                _tensor_float_array(reference_latent),
            ),
            "latent_max_abs": _max_abs(
                _tensor_float_array(latents[label]),
                _tensor_float_array(reference_latent),
            ),
        }

    return {
        "status": "passed" if not equality_required or (
            all_action_equal and all_latent_equal
        ) else "failed",
        "path": str(comparison_path),
        "sha256": _sha256(comparison_path),
        "schema_version": schema,
        "reference_kind": reference_kind,
        "reference_action_cfg_style_scale": reference_scale,
        "equality_required": equality_required,
        "all_actions_bit_exact": all_action_equal,
        "all_latents_bit_exact": all_latent_equal,
        "by_condition": by_condition,
    }


def _validate_server_contract(
    future_root: Path,
    *,
    action_cfg_scale: float,
    remote_port: int,
) -> tuple[Path, dict[str, Any]]:
    path = future_root / "server_contract.json"
    contract = json.loads(path.read_text())
    required = {
        "schema_version": SERVER_SCHEMA,
        "amendment_id": AMENDMENT_ID,
        "official_repository_commit": OFFICIAL_COMMIT,
        "source_patch_sha256": V2A015_PATCH_SHA256,
        "patched_action_head_sha256": V2A015_PATCHED_ACTION_HEAD_SHA256,
        "base_instrumented_server_sha256": BASE_INSTRUMENTED_SERVER_SHA256,
        "derived_instrumented_server_sha256": DERIVED_INSTRUMENTED_SERVER_SHA256,
        "runtime_num_inference_steps": 16,
        "evaluated_dit_steps_with_cache": 8,
        "video_cfg_scale": 5.0,
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise ValueError(
                f"V2-A015 server contract mismatch for {key}: "
                f"{contract.get(key)!r} != {expected!r}"
            )
    if float(contract.get("action_cfg_style_scale", -1.0)) != action_cfg_scale:
        raise ValueError("Connected V2-A015 server uses the wrong action scale")
    if contract.get("port") != remote_port:
        raise ValueError("Connected port does not match the V2-A015 server contract")
    if Path(contract["future_root"]).resolve() != future_root.resolve():
        raise ValueError("V2-A015 server contract names a different future root")

    patch_path = Path(contract["source_patch"])
    target_path = Path(contract["patched_action_head"])
    if _sha256(patch_path) != V2A015_PATCH_SHA256:
        raise ValueError("V2-A015 source patch bytes do not match the frozen hash")
    if _sha256(target_path) != V2A015_PATCHED_ACTION_HEAD_SHA256:
        raise ValueError("V2-A015 patched action-head bytes do not match the frozen hash")
    return path, contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--future-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, required=True)
    parser.add_argument(
        "--action-cfg-scale", type=float, choices=(1.0, 2.0), required=True
    )
    parser.add_argument("--comparison-probe", type=Path)
    args = parser.parse_args()
    if args.remote_port == 5000:
        parser.error("V2-A015 prohibits requests to the pre-existing port 5000")

    fixture_manifest = json.loads(args.fixture_manifest.read_text())
    fixture_entry = fixture_manifest.get("fixture", {})
    if fixture_entry.get("sha256") != _sha256(args.fixture):
        raise ValueError("Fixed-observation fixture hash does not match its manifest")
    if fixture_manifest.get("prompt") != LEFT:
        raise ValueError("Fixed-observation manifest does not contain the frozen LEFT prompt")
    if fixture_manifest.get("status") != "passed":
        raise ValueError("Renderer/reset gate did not pass")

    with np.load(args.fixture, allow_pickle=False) as archive:
        base_request = {key: archive[key] for key in archive.files}
    required_request_keys = {
        "observation/exterior_image_0_left",
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
        "observation/joint_position",
        "observation/cartesian_position",
        "observation/gripper_position",
    }
    if set(base_request) != required_request_keys:
        raise ValueError(
            "Fixed-observation keys changed: "
            f"missing={sorted(required_request_keys - set(base_request))}, "
            f"extra={sorted(set(base_request) - required_request_keys)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "v2a015_fixed_observation_probe.json"
    expected_outputs = [
        report_path,
        *(args.output_dir / f"{label}_returned_action.npy" for label, _ in CONDITIONS),
    ]
    existing_outputs = [path for path in expected_outputs if path.exists()]
    if existing_outputs:
        raise ValueError(
            "Refusing to overwrite existing probe outputs: "
            f"{[str(path) for path in existing_outputs]}"
        )
    if any(args.future_root.glob("episode_*")):
        raise ValueError(
            "V2-A015 future root must have no episode directories before this probe"
        )
    server_contract_path, server_contract = _validate_server_contract(
        args.future_root,
        action_cfg_scale=args.action_cfg_scale,
        remote_port=args.remote_port,
    )

    packer = MsgPackNumpy()
    uri = f"ws://{args.remote_host}:{args.remote_port}"
    connection, server_metadata = _connect(uri, packer)
    if isinstance(server_metadata, dict):
        metadata_scale = server_metadata.get("action_cfg_style_scale")
        if metadata_scale is not None and float(metadata_scale) != args.action_cfg_scale:
            raise ValueError("Websocket metadata reports the wrong V2-A015 action scale")

    arrays: dict[str, np.ndarray] = {}
    latents: dict[str, torch.Tensor] = {}
    records: dict[str, dict[str, Any]] = {}
    scale_tag = f"s{int(args.action_cfg_scale)}"
    try:
        for episode_index, (label, prompt) in enumerate(CONDITIONS):
            session_id = f"dreamzero-v2-a015-{scale_tag}-fixed-{label}"
            request = dict(base_request)
            request.update(
                {
                    "prompt": prompt,
                    "session_id": session_id,
                    "endpoint": "infer",
                }
            )
            connection.send(packer.pack(request))
            returned_action = _recv_action(connection, packer)
            action_path = args.output_dir / f"{label}_returned_action.npy"
            np.save(action_path, returned_action, allow_pickle=False)

            connection.send(
                packer.pack({"endpoint": "reset", "session_ids": [session_id]})
            )
            _check_reset_reply(connection.recv(timeout=600), packer)
            future_manifest_path = (
                args.future_root
                / f"episode_{episode_index:03d}"
                / "future_manifest.json"
            )
            future_manifest = _wait_for_v2a015_manifest(
                future_manifest_path,
                action_cfg_scale=args.action_cfg_scale,
            )
            server_action, latent, server_record = _load_server_record(
                future_manifest,
                prompt=prompt,
                action_cfg_scale=args.action_cfg_scale,
            )
            if not np.array_equal(returned_action, server_action):
                raise ValueError(
                    f"V2-A015 instrumentation changed the returned action for {label}"
                )

            arrays[label] = returned_action
            latents[label] = latent
            records[label] = {
                "prompt": prompt,
                "requested_relation": "left" if prompt == LEFT else "right",
                "condition_role": (
                    "exact_repeat" if label == "left_b" else "primary"
                ),
                "sampling_seed_label": 8300,
                "effective_official_model_noise_seed": OFFICIAL_NOISE_SEED,
                "session_id": session_id,
                "clean_session": True,
                "action_path": str(action_path),
                "action_sha256": _sha256(action_path),
                "action_data_sha256": _array_data_sha256(returned_action),
                "shape": list(returned_action.shape),
                "dtype": str(returned_action.dtype),
                "future_manifest": str(future_manifest_path),
                "future_manifest_sha256": _sha256(future_manifest_path),
                "returned_action": server_record["returned_action"],
                "latent_path": server_record["latent_video"]["path"],
                "latent_sha256": server_record["latent_video"]["sha256"],
                "latent_data_sha256": _tensor_data_sha256(latent),
                "latent_shape": list(latent.shape),
                "latent_dtype": str(latent.dtype),
                "official_decode_count": len(
                    future_manifest.get("official_reset_decode", [])
                ),
            }
    finally:
        connection.close()

    action_repeat_equal = bool(np.array_equal(arrays["left_a"], arrays["left_b"]))
    latent_repeat_equal = _tensor_bit_equal(latents["left_a"], latents["left_b"])
    metrics = {
        "all_actions_finite_shape_24x8": all(
            value.shape == EXPECTED_ACTION_SHAPE and np.isfinite(value).all()
            for value in arrays.values()
        ),
        "all_latents_finite": all(
            bool(torch.isfinite(value.float()).all()) for value in latents.values()
        ),
        "left_exact_repeat_action_array_equal": action_repeat_equal,
        "left_exact_repeat_action_rms": _rms(arrays["left_a"], arrays["left_b"]),
        "left_exact_repeat_action_max_abs": _max_abs(
            arrays["left_a"], arrays["left_b"]
        ),
        "left_vs_right_action_rms": _rms(arrays["left_a"], arrays["right"]),
        "left_vs_right_action_max_abs": _max_abs(
            arrays["left_a"], arrays["right"]
        ),
        "left_exact_repeat_latent_tensor_equal": latent_repeat_equal,
        "left_exact_repeat_latent_rms": _rms(
            _tensor_float_array(latents["left_a"]),
            _tensor_float_array(latents["left_b"]),
        ),
        "left_exact_repeat_latent_max_abs": _max_abs(
            _tensor_float_array(latents["left_a"]),
            _tensor_float_array(latents["left_b"]),
        ),
        "left_vs_right_latent_rms": _rms(
            _tensor_float_array(latents["left_a"]),
            _tensor_float_array(latents["right"]),
        ),
        "left_vs_right_latent_max_abs": _max_abs(
            _tensor_float_array(latents["left_a"]),
            _tensor_float_array(latents["right"]),
        ),
    }
    internal_gate_passed = bool(
        metrics["all_actions_finite_shape_24x8"]
        and metrics["all_latents_finite"]
        and action_repeat_equal
        and latent_repeat_equal
        and metrics["left_vs_right_action_rms"] > 0.0
    )

    comparison: dict[str, Any]
    if args.comparison_probe is None:
        comparison = {
            "status": "not_provided",
            "equality_required": args.action_cfg_scale == 1.0,
        }
    else:
        comparison = _compare_probe(
            args.comparison_probe,
            action_cfg_scale=args.action_cfg_scale,
            arrays=arrays,
            latents=latents,
        )
    comparison_gate_passed = bool(
        args.action_cfg_scale == 2.0 or comparison.get("status") == "passed"
    )
    release_gate_passed = internal_gate_passed and comparison_gate_passed

    report = {
        "schema_version": PROBE_SCHEMA,
        "amendment_id": AMENDMENT_ID,
        "status": "passed" if release_gate_passed else "failed",
        "action_cfg_style_scale": args.action_cfg_scale,
        "video_cfg_scale": 5.0,
        "fixture_manifest": str(args.fixture_manifest),
        "fixture_sha256": _sha256(args.fixture),
        "future_root": str(args.future_root),
        "remote_host": args.remote_host,
        "remote_port": args.remote_port,
        "server_metadata": server_metadata,
        "server_contract": {
            "path": str(server_contract_path),
            "sha256": _sha256(server_contract_path),
            "schema_version": server_contract["schema_version"],
            "action_cfg_style_scale": server_contract["action_cfg_style_scale"],
            "source_patch_sha256": server_contract["source_patch_sha256"],
            "patched_action_head_sha256": server_contract[
                "patched_action_head_sha256"
            ],
            "base_instrumented_server_sha256": server_contract[
                "base_instrumented_server_sha256"
            ],
            "derived_instrumented_server_sha256": server_contract[
                "derived_instrumented_server_sha256"
            ],
        },
        "sampling_seed_label": 8300,
        "effective_official_model_noise_seed": OFFICIAL_NOISE_SEED,
        "records": records,
        "metrics": metrics,
        "comparison": comparison,
        "contract": {
            "action_shape": list(EXPECTED_ACTION_SHAPE),
            "clean_unique_session_per_request": True,
            "reset_after_every_request": True,
            "repeat_requires_bit_exact_action_and_latent": True,
            "prompt_difference_requires_nonzero_action_rms": True,
            "prompt_difference_latent_rms_is_descriptive": True,
            "scale_1_requires_archived_official_action_and_latent_equivalence": True,
            "scale_2_comparison_equality_required": False,
            "returned_action_preservation_verified": True,
            "negative_branch_caveat": (
                "This is CFG-style negative-branch action guidance, not strict empty-text "
                "classifier-free guidance and not an official DreamZero action-CFG feature."
            ),
        },
        "internal_gate_passed": internal_gate_passed,
        "comparison_gate_passed": comparison_gate_passed,
        "release_gate_passed": release_gate_passed,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not release_gate_passed:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
