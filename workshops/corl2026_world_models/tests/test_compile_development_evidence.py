from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import struct
import tempfile
import unittest
from unittest import mock
import zipfile


WORKSHOP = Path(__file__).resolve().parents[1]
MODULE = WORKSHOP / "analysis/compile_development_evidence.py"
SPEC = importlib.util.spec_from_file_location("compile_development_evidence", MODULE)
compiler = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(compiler)

ANNOTATION_MODULE = WORKSHOP / "analysis/forecast_annotation_workflow.py"
ANNOTATION_SPEC = importlib.util.spec_from_file_location(
    "compiler_annotation_compat", ANNOTATION_MODULE
)
annotation = importlib.util.module_from_spec(ANNOTATION_SPEC)
assert ANNOTATION_SPEC.loader is not None
ANNOTATION_SPEC.loader.exec_module(annotation)


def write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def descriptor(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": compiler.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def npy_bytes(dtype: str, shape: tuple[int, ...], raw: bytes) -> bytes:
    header = repr({"descr": dtype, "fortran_order": False, "shape": shape})
    padding = (-((10 + len(header) + 1) % 16)) % 16
    encoded = (header + " " * padding + "\n").encode("latin1")
    return b"\x93NUMPY" + bytes((1, 0)) + struct.pack("<H", len(encoded)) + encoded + raw


def write_npz(path: Path, members: dict[str, bytes]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


TORCH_FLOAT32_VECTOR2_METADATA = bytes.fromhex(
    "800263746f7263682e5f7574696c730a5f72656275696c645f74656e736f72"
    "5f76320a71002828580700000073746f72616765710163746f7263680a466c6f"
    "617453746f726167650a71025801000000307103580300000063707571044b02"
    "747105514b004b028571064b018571078963636f6c6c656374696f6e730a4f72"
    "6465726564446963740a71082952710974710a52710b2e"
)

TORCH_FLOAT32_MATRIX1X2_METADATA = bytes.fromhex(
    "800263746f7263682e5f7574696c730a5f72656275696c645f74656e736f72"
    "5f76320a71002828580700000073746f72616765710163746f7263680a466c6f"
    "617453746f726167650a71025801000000307103580300000063707571044b02"
    "747105514b004b014b028671064b024b018671078963636f6c6c656374696f6e"
    "730a4f726465726564446963740a71082952710974710a52710b2e"
)

TORCH_FLOAT32_DECODE1X3X3X1X1_METADATA = bytes.fromhex(
    "800263746f7263682e5f7574696c730a5f72656275696c645f74656e736f72"
    "5f76320a71002828580700000073746f72616765710163746f7263680a466c6f"
    "617453746f726167650a71025801000000307103580300000063707571044b09"
    "747105514b00284b014b034b034b014b01747106284b094b034b014b014b0174"
    "71078963636f6c6c656374696f6e730a4f726465726564446963740a71082952"
    "710974710a52710b2e"
)


def write_torch_archive(
    path: Path, raw: bytes, *, metadata: bytes = TORCH_FLOAT32_VECTOR2_METADATA
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", metadata)
        archive.writestr("archive/data/0", raw)
        archive.writestr("archive/byteorder", b"little")
        archive.writestr("archive/version", b"3\n")
    return path


def d1_exact_mapping(
    root: Path, name: str, prefix: str, *, kind: str
) -> dict:
    directory = root / name
    directory.mkdir(parents=True)
    if kind == "numpy_array":
        raw = b"\x01\x02"
        path = directory / f"{prefix}_000.npy"
        path.write_bytes(npy_bytes("|u1", (2,), raw))
        entry = {
            "key": "value",
            "kind": kind,
            "path": str(path.resolve()),
            "file_sha256": compiler.sha256_file(path),
            "bytes": path.stat().st_size,
            "shape": [2],
            "dtype": "uint8",
            "data_sha256": compiler.sha256_bytes(raw),
        }
    elif kind == "torch_tensor":
        raw = struct.pack("<2f", 1.0, 2.0)
        path = write_torch_archive(directory / f"{prefix}_000.pt", raw)
        entry = {
            "key": "value",
            "kind": kind,
            "path": str(path.resolve()),
            "file_sha256": compiler.sha256_file(path),
            "bytes": path.stat().st_size,
            "shape": [2],
            "dtype": "torch.float32",
            "data_sha256": compiler.sha256_bytes(raw),
        }
    else:
        value = {"finite": True, "value": 3}
        path = write_json(directory / f"{prefix}_000.json", value)
        entry = {
            "key": "value",
            "kind": "json_value",
            "path": str(path.resolve()),
            "file_sha256": compiler.sha256_file(path),
            "bytes": path.stat().st_size,
            "json_sha256": compiler.sha256_bytes(compiler.canonical_bytes(value)),
        }
    identity_fields = ("key", "kind", "shape", "dtype", "data_sha256", "json_sha256")
    identity = {field: entry[field] for field in identity_fields if field in entry}
    return {
        "entries": [entry],
        "entry_count": 1,
        "content_sha256": compiler.sha256_bytes(compiler.canonical_bytes([identity])),
        "content_hash_definition": compiler.D1_MAPPING_CONTENT_HASH_DEFINITION,
    }


def freeze_scalar(value: object) -> object:
    if isinstance(value, dict):
        return {"__type__": "mapping", "items": {key: freeze_scalar(item) for key, item in value.items()}}
    if isinstance(value, list):
        return {"__type__": "list", "items": [freeze_scalar(item) for item in value]}
    return value


def recorder_payload(
    root: Path,
    name: str,
    role: str,
    structure: dict,
    arrays: dict[str, tuple[str, tuple[int, ...], bytes]],
) -> dict:
    result = {"role": role, "structure": structure, "array_count": len(arrays)}
    if arrays:
        path = write_npz(
            root / f"{name}.npz",
            {
                f"{key}.npy": npy_bytes(dtype, shape, raw)
                for key, (dtype, shape, raw) in arrays.items()
            },
        )
        result["artifact"] = descriptor(path)
    result["payload_sha256"] = compiler.sha256_bytes(
        compiler.canonical_bytes(result, ensure_ascii=True)
    )
    return result


def n3_nested_payload(
    root: Path,
    role: str,
    raw: bytes = b"\x01\x02",
    *,
    dtype: str = "|u1",
    shape: tuple[int, ...] | None = None,
    field: str | None = None,
) -> dict:
    shape = (len(raw),) if shape is None else shape
    payload_root = root / role
    payload_root.mkdir(parents=True)
    artifact_path = payload_root / "0000_value.npy"
    artifact_path.write_bytes(npy_bytes(dtype, shape, raw))
    identity_header = {"kind": "numpy", "dtype": dtype, "shape": list(shape)}
    node = {
        "__type__": "numpy",
        "artifact": {
            "path": artifact_path.name,
            "bytes": artifact_path.stat().st_size,
            "sha256": compiler.sha256_file(artifact_path),
        },
        **identity_header,
        "value_sha256": compiler.sha256_bytes(
            compiler.canonical_bytes(identity_header, ensure_ascii=True) + raw
        ),
    }
    structure = node if field is None else {"__type__": "mapping", "items": {field: node}}
    logical = compiler.sha256_bytes(
        compiler.canonical_bytes(compiler._n3_logical_structure(structure), ensure_ascii=True)
    )
    manifest = {
        "schema_version": "wmf-lossless-nested-payload-v1",
        "role": role,
        "structure": structure,
        "logical_sha256": logical,
    }
    manifest_path = write_json(payload_root / f"{role}.manifest.json", manifest)
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": compiler.sha256_file(manifest_path),
        "logical_sha256": logical,
    }


def n3_official_response_payload(
    root: Path, action_raw: bytes, video_raw: bytes
) -> dict:
    role = "official_returned_response"
    payload_root = root / role
    payload_root.mkdir(parents=True)
    fields = {
        "action": ("<f4", (2, 8), action_raw),
        "video": ("|u1", (3, 1, 1, 3), video_raw),
    }
    nodes = {}
    for index, (field, (dtype, shape, raw)) in enumerate(fields.items()):
        path = payload_root / f"{index:04d}_{field}.npy"
        path.write_bytes(npy_bytes(dtype, shape, raw))
        identity_header = {"kind": "numpy", "dtype": dtype, "shape": list(shape)}
        nodes[field] = {
            "__type__": "numpy",
            "artifact": {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": compiler.sha256_file(path),
            },
            **identity_header,
            "value_sha256": compiler.sha256_bytes(
                compiler.canonical_bytes(identity_header, ensure_ascii=True) + raw
            ),
        }
    structure = {"__type__": "mapping", "items": nodes}
    logical = compiler.sha256_bytes(
        compiler.canonical_bytes(compiler._n3_logical_structure(structure), ensure_ascii=True)
    )
    manifest = {
        "schema_version": "wmf-lossless-nested-payload-v1",
        "role": role,
        "structure": structure,
        "logical_sha256": logical,
    }
    path = write_json(payload_root / f"{role}.manifest.json", manifest)
    return {
        "manifest_path": str(path.resolve()),
        "manifest_sha256": compiler.sha256_file(path),
        "logical_sha256": logical,
    }


def journal_row(sequence: int, previous: str | None, kind: str, payload: dict) -> dict:
    base = {
        "sequence": sequence,
        "kind": kind,
        "wall_time_ns": 2_000_000_000 + sequence,
        "monotonic_ns": 3_000_000_000 + sequence,
        "previous_event_sha256": previous,
        "payload": payload,
    }
    return {
        **base,
        "event_sha256": compiler.sha256_bytes(compiler.freeze.recorder_canonical_bytes(base)),
    }


TINY_LIMITS = {
    "N3": {
        **compiler.MODEL_LIMITS["N3"],
        "action_cap": 2,
        "observation_count": 3,
        "request_count": 1,
        "executed_prefix": 2,
        "returned_actions": 2,
        "decoded_frames": 3,
    },
    "D1": {
        **compiler.MODEL_LIMITS["D1"],
        "action_cap": 2,
        "observation_count": 3,
        "request_count": 1,
        "executed_prefix": 2,
        "returned_actions": 2,
        "decoded_frames": 3,
        "development_decode_shapes": {
            "full_conditioning_origin": {
                "latent": [2],
                "tensor": [1, 3, 3, 1, 1],
                "rgb": [3, 1, 1, 3],
            },
            "incremental_standalone": {
                "latent": [2],
                "tensor": [1, 3, 2, 1, 1],
                "rgb": [2, 1, 1, 3],
            },
        },
    },
}


class TinyEvidence:
    def __init__(self, root: Path, layouts: tuple[str, ...] = ("D01",)) -> None:
        self.root = root
        self.layouts = layouts
        self.planned_path = self._planned_csv()
        self.aggregates: dict[tuple[str, str], dict] = {}
        for model in compiler.MODELS:
            for layout in layouts:
                self.aggregates[(model, layout)] = self._cell_and_aggregate(model, layout)

    def _planned_csv(self) -> Path:
        path = self.root / "planned_cells.csv"
        lines = [
            "phase,selected,model_config,layout_pair_id,layout_arm,command,cell_id,"
            "candidate_effective_policy_seed"
        ]
        for model in compiler.MODELS:
            for layout in self.layouts:
                cell = f"wmf1__development__{layout}__{model}__original__left"
                seed = 2026091100 + int(layout[1:]) if model == "N3" else 1140
                lines.append(
                    f"development,true,{model},{layout},original,left,{cell},{seed}"
                )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _observation_payload(self, cell_root: Path, step: int) -> tuple[dict, dict]:
        clock = {
            "physics_step": step * 8,
            "physics_time_s": step / 15,
            "control_step": step,
            "cameras": {
                "over_shoulder_left_camera": {
                    "frame_id": 100 + step,
                    "capture_time_ns": 1_000_000_000 + step * 66_666_667,
                    "timestamp_source": "native-test-clock",
                }
            },
        }
        node = {"__type__": "ndarray", "key": "array_0000", "shape": [1, 1, 3], "dtype": "|u1"}
        structure = freeze_scalar({"clock": clock, "image_obs": {}})
        structure["items"]["image_obs"]["items"]["over_shoulder_left_camera"] = node
        artifact = recorder_payload(
            cell_root / "recording/payloads",
            f"observation-{step}",
            "observation",
            structure,
            {"array_0000": ("|u1", (1, 1, 3), bytes((step, step, step)))},
        )
        return clock, artifact

    def _official_request(
        self, model: str, cell_root: Path, cell_id: str,
        action_raw: bytes, video_raw: bytes, contract_sha: str,
    ) -> tuple[dict, dict]:
        episode_id = f"episode-{cell_id}"
        request_root = (
            cell_root / "official/request-00"
            if model == "N3"
            else cell_root / "future" / "episodes" / episode_id / "request_0000"
        )
        if model == "N3":
            layout = cell_id.split("__")[2]
            effective_seed = 2026091100 + int(layout[1:])
            roles = {
                "wire_request": "exact_wire_request",
                "exact_transformed_model_input": "exact_transformed_model_input",
                "raw_generated_action": "raw_generated_action",
                "retained_vision_latent": "retained_vision_latent",
                "exact_decoder_input_latent": "exact_decoder_input_latent",
                "raw_decoder_output": "raw_decoder_output",
                "official_returned_response": "official_returned_response",
            }
            receipt = {
                "schema_version": TINY_LIMITS[model]["request_schema"],
                "status": "passed",
                "study_id": compiler.STUDY_ID,
                "block_id": f"wmf_ablation_001_20260912__development__{layout}__N3",
                "cell_id": cell_id,
                "condition_index": 0,
                "request_index": 0,
                "action_step_start": 0,
                "behavioral_model_request": True,
                "generation_qualification_request": False,
                "sampling_seed": effective_seed,
                "server_context_id": f"context-{cell_id}",
                "returned_action_shape": [2, 8],
                "decoded_future_shape": [3, 1, 1, 3],
                "joint_generation_calls": 1,
                "decode_calls": 1,
                "started_wall_time_ns": 1,
                "completed_wall_time_ns": 2,
                "started_monotonic_ns": 3,
                "completed_monotonic_ns": 4,
                **{
                    key: (
                        n3_official_response_payload(request_root, action_raw, video_raw)
                        if key == "official_returned_response"
                        else n3_nested_payload(request_root, role)
                    )
                    for key, role in roles.items()
                },
            }
        else:
            request_root.mkdir(parents=True)

            def direct(name: str, payload: bytes = b"artifact") -> dict:
                path = request_root / name
                path.write_bytes(payload)
                return {
                    "path": str(path.resolve()),
                    "file_sha256": compiler.sha256_file(path),
                    "bytes": path.stat().st_size,
                }

            latent_raw = struct.pack("<2f", 0.25, 0.5)
            latent_path = write_torch_archive(request_root / "official_video_pred.pt", latent_raw)
            latent = {
                "path": str(latent_path.resolve()),
                "file_sha256": compiler.sha256_file(latent_path),
                "bytes": latent_path.stat().st_size,
                "data_sha256": compiler.sha256_bytes(latent_raw),
                "shape": [2],
                "dtype": "torch.float32",
            }
            decoded_tensor_raw = struct.pack("<9f", *([-1.0, 0.0, 1.0] * 3))
            decoded_tensor_path = write_torch_archive(
                request_root / "offline_decoded_tensor.pt",
                decoded_tensor_raw,
                metadata=TORCH_FLOAT32_DECODE1X3X3X1X1_METADATA,
            )
            decoded_rgb_raw = bytes(range(9))
            decoded_rgb_path = request_root / "offline_decoded_rgb.npy"
            decoded_rgb_path.write_bytes(
                npy_bytes("|u1", (3, 1, 1, 3), decoded_rgb_raw)
            )
            receipt = {
                "schema_version": TINY_LIMITS[model]["request_schema"],
                "configuration_id": "D1",
                "episode_id": episode_id,
                "request_index": 0,
                "probe_id": None,
                "prompt": "Put the Rubik's cube to the left of the bowl.",
                "session_id": f"session-{cell_id}",
                "official_action_path": "GrootSimPolicy.lazy_joint_forward_causal",
                "custom_s2_used": False,
                "patched_s1_used": False,
                "effective_official_model_noise_seed": 1140,
                "noise_semantics": "fixed; this request is not an independent noise draw",
                "measurement_control": {
                    "probe_id": None,
                    "offline_decode": True,
                    "probe_plan_sha256": contract_sha,
                },
                "raw_inputs": d1_exact_mapping(
                    request_root, "raw_inputs", "raw", kind="numpy_array"
                ),
                "converted_inputs": d1_exact_mapping(
                    request_root, "converted_inputs", "converted", kind="torch_tensor"
                ),
                "normalized_model_inputs": d1_exact_mapping(
                    request_root, "normalized_model_inputs", "normalized", kind="json_value"
                ),
                "official_returned_action": {
                    **direct("action.npy", npy_bytes("<f4", (2, 8), action_raw)),
                    "shape": [2, 8],
                    "dtype": "float32",
                    "data_sha256": compiler.sha256_bytes(action_raw),
                },
                "official_forward_call_count": 1,
                "latent_video": latent,
                "offline_decode": {
                    "requested": True,
                    "performed": True,
                    "latent_data_sha256_before": latent["data_sha256"],
                    "latent_data_sha256_after": latent["data_sha256"],
                    "decoded_tensor": {
                        "path": str(decoded_tensor_path.resolve()),
                        "file_sha256": compiler.sha256_file(decoded_tensor_path),
                        "bytes": decoded_tensor_path.stat().st_size,
                        "data_sha256": compiler.sha256_bytes(decoded_tensor_raw),
                        "shape": [1, 3, 3, 1, 1],
                        "dtype": "torch.float32",
                    },
                    "decoded_rgb": {
                        "path": str(decoded_rgb_path.resolve()),
                        "file_sha256": compiler.sha256_file(decoded_rgb_path),
                        "bytes": decoded_rgb_path.stat().st_size,
                        "data_sha256": compiler.sha256_bytes(decoded_rgb_raw),
                        "shape": [3, 1, 1, 3],
                        "dtype": "uint8",
                    },
                },
                "temporal_and_cache_rank_metrics": [
                    {
                        "rank": rank,
                        "temporal_before": {
                            "current_start_frame": 0,
                            "fields": {
                                field: {"is_none": True}
                                for field in compiler.D1_CACHE_FIELDS
                            },
                        },
                        "temporal_after": {
                            "current_start_frame": 1,
                            "fields": {
                                field: {"is_none": False}
                                for field in compiler.D1_CACHE_FIELDS
                            },
                        },
                        "cache_reinitialization": {
                            "_create_kv_caches": [{"called": True}],
                            "_create_crossattn_caches": [{"called": True}],
                        },
                    }
                    for rank in (0, 1)
                ],
            }
        request_path = write_json(request_root / "request_receipt.json", receipt)
        return receipt, descriptor(request_path)

    def _model_chain(
        self,
        model: str,
        cell_root: Path,
        cell_id: str,
        layout: str,
        contract_sha: str,
        request: dict,
    ) -> dict:
        study_commit = "1" * 40
        pose_sha = compiler.sha256_bytes(f"pose-{model}-{layout}".encode())
        pins = compiler.MODEL_PINS[model]
        if model == "N3":
            context_id = f"context-{cell_id}"
            effective_seed = 2026091100 + int(layout[1:])
            begin = {
                "passed": True,
                "reset_scope": compiler.CONTEXT_RESET_SCOPE,
                "server_context_id": context_id,
                "cache_reset_evidence": {
                    "exclusive_active_episode": cell_id,
                    "wrapper_request_index_reset_to_zero": True,
                    "official_history_length": 1,
                    "request_bound_seed": effective_seed,
                    "model_process_reused_but_episode_state_not_reused": True,
                    "passed": True,
                    "episode_context_id": context_id,
                    "unresolved_mutable_temporal_fields": [],
                },
                "cell_id": cell_id,
                "condition_index": 0,
                "effective_seed": effective_seed,
            }
            end = {
                "passed": True,
                "status": "completed",
                "server_context_id": context_id,
                "cell_id": cell_id,
                "condition_index": 0,
                "server_request_count": 1,
                "client_request_count": 1,
                "actions_executed": 2,
            }
            return {
                "study_commit": study_commit,
                "pose_sha": pose_sha,
                "source_pins": {
                    "study_commit": study_commit,
                    "robolab_commit": pins["robolab_commit"],
                    "cosmos_commit": pins["cosmos_commit"],
                },
                "checkpoint_pin": {
                    "revision": pins["checkpoint_revision"],
                    "aggregate_sha256": pins["checkpoint_aggregate_sha256"],
                },
                "source_identity": (
                    f"study:{study_commit};robolab:{pins['robolab_commit']};"
                    f"cosmos:{pins['cosmos_commit']};pose:{pose_sha}"
                ),
                "begin": begin,
                "cell_fields": {
                    "block_id": f"wmf_ablation_001_20260912__development__{layout}__N3",
                    "server_begin_receipt": begin,
                    "server_end_receipt": end,
                },
            }

        future_root = cell_root / "future"
        identity_receipt = {
            "status": "passed",
            "source": {
                "commit": pins["dreamzero_commit"],
                "git_tree": pins["dreamzero_tree"],
                "aggregate_sha256": pins["dreamzero_aggregate_sha256"],
            },
            "checkpoint": {
                "revision": pins["checkpoint_revision"],
                "aggregate_sha256": pins["checkpoint_aggregate_sha256"],
            },
            "tokenizer": {
                "revision": pins["tokenizer_revision"],
                "aggregate_sha256": pins["tokenizer_aggregate_sha256"],
            },
        }
        identity_path = write_json(future_root / "identity_receipt.json", identity_receipt)
        contract = {
            "schema_version": compiler.D1_SERVER_CONTRACT_SCHEMA,
            "status": "passed",
            "configuration_id": "D1",
            "official_repository_commit": pins["dreamzero_commit"],
            "official_repository_tree": pins["dreamzero_tree"],
            "official_action_path": "GrootSimPolicy.lazy_joint_forward_causal",
            "custom_s2_used": False,
            "patched_s1_used": False,
            "world_size": 2,
            "port": 18021,
            "future_root": str(future_root.resolve()),
            "returned_action_shape": [2, 8],
            "executed_action_prefix": 2,
            "effective_official_model_noise_seed": 1140,
            "noise_semantics": "fixed; no request is an independent noise draw",
            "dynamic_cache_schedule": False,
            "tensorrt_engine_active": False,
            "instrumentation_overlay": {"returned_action_modified": False},
            "topology": [
                {"rank": 0, "cuda_device_index": 0, "cuda_device_name": "NVIDIA B200"},
                {"rank": 1, "cuda_device_index": 1, "cuda_device_name": "NVIDIA B200"},
            ],
            "head_contracts": [{"rank": 0, "status": "passed"}, {"rank": 1, "status": "passed"}],
            "bounded_loader_receipts": [
                {"rank": 0, "receipt": {"passed": True, "forward_path_modified": False}},
                {"rank": 1, "receipt": {"passed": True, "forward_path_modified": False}},
            ],
            "identity_receipt": str(identity_path.resolve()),
            "identity_receipt_sha256": compiler.sha256_file(identity_path),
        }
        server_contract_path = write_json(future_root / "server_contract.json", contract)
        ready = {
            "schema_version": compiler.D1_READY_SCHEMA,
            "status": "ready",
            "run_id": f"run-{layout}",
            "server_job_id": f"server-{layout}",
            "paired_simulator_job_id": f"simulator-{layout}",
            "study_commit": study_commit,
            "study_id": compiler.STUDY_ID,
            "block_id": f"wmf_ablation_001_20260912__development__{layout}__D1",
            "model_config": "D1",
            "service_host": "wmf-forecast-0912-d1",
            "service_port": 18021,
            "future_root": str(future_root.resolve()),
            "server_contract": descriptor(server_contract_path),
            "server_contract_sha256": compiler.sha256_file(server_contract_path),
            "runtime_identity": descriptor(identity_path),
            "pilot_contract_sha256": contract_sha,
            "expected_cell_ids": [cell_id],
            "returned_action_shape": [2, 8],
            "executed_prefix_horizon": 2,
            "effective_model_noise_seed": 1140,
            "global_state_noninterleaving": True,
        }
        ready_path = write_json(cell_root / "coordination/server_ready.json", ready)
        ready_descriptor = descriptor(ready_path)
        claim = {
            "schema_version": compiler.D1_CLAIM_SCHEMA,
            "status": "claimed",
            "run_id": ready["run_id"],
            "simulator_job_id": ready["paired_simulator_job_id"],
            "server_job_id": ready["server_job_id"],
            "server_ready_sha256": ready_descriptor["sha256"],
            "study_commit": study_commit,
            "block_id": ready["block_id"],
            "worker_role": "wmf-forecast-0912-worker-00",
            "pilot_contract_sha256": contract_sha,
            "lease_token": f"lease-{layout}",
            "start_cell_index": 0,
        }
        claim_path = write_json(cell_root / "coordination/simulator_claim.json", claim)
        claim_descriptor = descriptor(claim_path)
        episode_id = request["episode_id"]
        session_id = request["session_id"]
        control = {
            "episode_id": episode_id,
            "expected_session_id": session_id,
            "purpose": "d1_behavioral_development",
            "study_id": compiler.STUDY_ID,
            "block_id": ready["block_id"],
            "cell_id": cell_id,
            "condition_index": 0,
            "layout_arm": "original",
            "command": "left",
            "server_ready_sha256": ready_descriptor["sha256"],
            "simulator_claim_sha256": claim_descriptor["sha256"],
            "simulator_lease_token": claim["lease_token"],
            "pilot_contract_sha256": contract_sha,
        }
        reset_id = f"reset-{layout}"
        fields = {
            field: {"is_none": True} for field in compiler.D1_RESET_FIELDS_TO_NONE
        }
        ranks = []
        for rank in (0, 1):
            row = {
                "schema_version": compiler.D1_RESET_SCHEMA,
                "reset_id": reset_id,
                "rank": rank,
                "status": "passed",
                "before": {"rank": rank},
                "after": {"current_start_frame": 0, "fields": fields},
                "fields_cleared": ["current_start_frame", *compiler.D1_RESET_FIELDS_TO_NONE],
                "failures": [],
            }
            if rank == 0:
                row["wrapper_after"] = {
                    "frame_buffer_lengths": {"cam": 0},
                    "call_count": 0,
                    "is_first_call": True,
                    "video_across_time_count": 0,
                    "current_session_id": None,
                }
            ranks.append(row)
        reset = {
            "schema_version": compiler.D1_RESET_SCHEMA,
            "reset_id": reset_id,
            "status": "passed",
            "world_size": 2,
            "rank_receipts": ranks,
            "control": control,
        }
        reset_path = write_json(
            future_root / "episodes" / episode_id / "reset_receipt.json", reset
        )
        reset_descriptor = descriptor(reset_path)
        reset_scan = {
            "passed": True,
            "reset_id": reset_id,
            "world_size": 2,
            "rank_temporal_state_scan": [
                {
                    "rank": rank,
                    "before": ranks[rank]["before"],
                    "after": ranks[rank]["after"],
                    "fields_cleared": ranks[rank]["fields_cleared"],
                }
                for rank in (0, 1)
            ],
            "unresolved_mutable_temporal_fields": [],
        }
        cache = {
            "source": "validated_official_d1_two_rank_reset_receipt",
            "server_reset_receipt": reset_descriptor,
            "reset_id": reset_id,
            "world_size": 2,
            "rank_temporal_state_scan": reset_scan["rank_temporal_state_scan"],
            "unresolved_mutable_temporal_fields": [],
        }
        begin = {
            "passed": True,
            "reset_scope": compiler.CONTEXT_RESET_SCOPE,
            "server_context_id": episode_id,
            "cache_reset_evidence": cache,
            "service_route_proved_by_reset_artifact": True,
            "service_host": "wmf-forecast-0912-d1",
            "service_port": 18021,
            "server_ready_sha256": ready_descriptor["sha256"],
            "simulator_claim_sha256": claim_descriptor["sha256"],
            "episode_context_id": episode_id,
            "client_session_id": session_id,
            "server_reset_receipt": reset_descriptor,
            "temporal_state_scan": reset_scan,
            "server_metadata": {"test": True},
        }
        manifest = {
            "schema_version": compiler.D1_EPISODE_SCHEMA,
            "configuration_id": "D1",
            "episode_id": episode_id,
            "status": "complete",
            "official_repository_commit": pins["dreamzero_commit"],
            "official_action_path": "GrootSimPolicy.lazy_joint_forward_causal",
            "custom_s2_used": False,
            "patched_s1_used": False,
            "effective_official_model_noise_seed": 1140,
            "noise_semantics": "fixed; not an independent draw",
            "request_count": 1,
            "server_contract_sha256": compiler.sha256_file(server_contract_path),
            "two_rank_reset": reset,
            "requests": [request],
        }
        manifest_path = write_json(
            future_root / "episodes" / episode_id / "episode_manifest.json", manifest
        )
        return {
            "study_commit": study_commit,
            "pose_sha": pose_sha,
            "source_pins": {
                "study_commit": study_commit,
                "robolab_commit": pins["robolab_commit"],
                "dreamzero_commit": pins["dreamzero_commit"],
                "dreamzero_tree": pins["dreamzero_tree"],
            },
            "checkpoint_pin": {
                "revision": pins["checkpoint_revision"],
                "aggregate_sha256": pins["checkpoint_aggregate_sha256"],
            },
            "source_identity": (
                f"study:{study_commit};robolab:{pins['robolab_commit']};"
                f"dreamzero:{pins['dreamzero_commit']};pose:{pose_sha}"
            ),
            "begin": begin,
            "cell_fields": {
                "block_id": ready["block_id"],
                "condition_index": 0,
                "development_contract_sha256": contract_sha,
                "server_ready": ready_descriptor,
                "simulator_claim": claim_descriptor,
                "server_begin_receipt": begin,
                "server_reset_receipt": reset_descriptor,
                "server_temporal_reset_scan": reset_scan,
                "server_episode_manifest": descriptor(manifest_path),
            },
        }

    def _cell_and_aggregate(self, model: str, layout: str) -> dict:
        cell_id = f"wmf1__development__{layout}__{model}__original__left"
        cell_root = self.root / "behavioral" / model / layout / cell_id
        completion_path = cell_root / "recording/completion.json"
        action_raw = struct.pack("<16f", *[float(index) for index in range(16)])
        video_raw = bytes(range(9))
        contract_sha = compiler.sha256_bytes(f"contract-{model}-{layout}".encode())
        request, request_descriptor = self._official_request(
            model, cell_root, cell_id, action_raw, video_raw, contract_sha
        )
        chain = self._model_chain(
            model, cell_root, cell_id, layout, contract_sha, request
        )
        clocks_and_artifacts = [self._observation_payload(cell_root, step) for step in range(3)]
        action_node_returned = {
            "__type__": "ndarray", "key": "array_0000", "shape": [2, 8], "dtype": "<f4"
        }
        action_node_executable = {
            "__type__": "ndarray", "key": "array_0001", "shape": [2, 8], "dtype": "<f4"
        }
        chunks = recorder_payload(
            cell_root / "recording/payloads",
            "action-chunks",
            "action_chunks",
            freeze_scalar({
                "returned_action_chunk": action_node_returned,
                "executable_action_chunk": action_node_executable,
            }),
            {
                "array_0000": ("<f4", (2, 8), action_raw),
                "array_0001": ("<f4", (2, 8), action_raw),
            },
        )
        # freeze_scalar treats our array-node dict as an ordinary mapping; put
        # the exact stored nodes back where the real recorder writes them.
        chunks["structure"]["items"]["returned_action_chunk"] = action_node_returned
        chunks["structure"]["items"]["executable_action_chunk"] = action_node_executable
        unsigned = dict(chunks)
        unsigned.pop("payload_sha256")
        chunks["payload_sha256"] = compiler.sha256_bytes(
            compiler.canonical_bytes(unsigned, ensure_ascii=True)
        )
        response_raw = {
            "wmf_server_request_receipt": request_descriptor,
            "wmf_request_index": 0,
        }
        if model == "N3":
            response_raw.update({"wmf_cell_id": cell_id, "wmf_action_step_start": 0})
            action_node = {
                "__type__": "ndarray", "key": "array_0000", "shape": [2, 8], "dtype": "<f4"
            }
            video_node = {
                "__type__": "ndarray", "key": "array_0001", "shape": [3, 1, 1, 3], "dtype": "|u1"
            }
            decoded_node = {
                "__type__": "ndarray", "key": "array_0002", "shape": [3, 1, 1, 3], "dtype": "|u1"
            }
            raw_structure = freeze_scalar(response_raw)
            raw_structure["items"]["action"] = action_node
            raw_structure["items"]["video"] = video_node
            response_structure = {
                "__type__": "mapping",
                "items": {
                    "raw_response": raw_structure,
                    "future_evidence": {
                        "__type__": "mapping", "items": {"decoded": decoded_node}
                    },
                },
            }
            response = recorder_payload(
                cell_root / "recording/payloads",
                "response",
                "model_response",
                response_structure,
                {
                    "array_0000": ("<f4", (2, 8), action_raw),
                    "array_0001": ("|u1", (3, 1, 1, 3), video_raw),
                    "array_0002": ("|u1", (3, 1, 1, 3), video_raw),
                },
            )
        else:
            response_raw["wmf_episode_context_id"] = request["episode_id"]
            def recorder_ref(value: dict) -> dict:
                return {
                    "path": value["path"],
                    "sha256": value["file_sha256"],
                    "bytes": value["bytes"],
                }
            response_future = {
                "latent": recorder_ref(request["latent_video"]),
                "decoded": {
                    "tensor": recorder_ref(request["offline_decode"]["decoded_tensor"]),
                    "rgb": recorder_ref(request["offline_decode"]["decoded_rgb"]),
                },
            }
            response_raw["future_evidence"] = response_future
            response = recorder_payload(
                cell_root / "recording/payloads",
                "response",
                "model_response",
                freeze_scalar({"raw_response": response_raw, "future_evidence": response_future}),
                {},
            )
        execution = [{
            "request_index": 0,
            "action_step_start": 0,
            "returned_actions": 2,
            "eligible_executable_prefix_actions": 2,
            "executed_actions": 2,
            "unused_executable_prefix_actions": 0,
            "returned_actions_outside_executable_prefix": 0,
            "current_observation_id": "obs_000000",
            "preceding_observation_id": None,
            "future_kinds": TINY_LIMITS[model]["future_kinds"],
            "action_chunks_artifact": chunks,
        }]
        identity = {
            "attempt_id": f"attempt-{model}-{layout}",
            "cell_id": cell_id,
            "stage": "development",
            "layout_pair_id": layout,
            "layout_arm": "original",
            "command": "left",
            "prompt": "Put the Rubik's cube to the left of the bowl.",
            "model_config": model,
            "effective_seed": (
                2026091100 + int(layout[1:]) if model == "N3" else 1140
            ),
            "source_identity": chain["source_identity"],
            "checkpoint_identity": (
                f"revision:{chain['checkpoint_pin']['revision']};"
                f"aggregate:{chain['checkpoint_pin']['aggregate_sha256']}"
            ),
        }
        context_artifact = recorder_payload(
            cell_root / "recording/payloads",
            "context-reset",
            "context_reset",
            freeze_scalar({
                **chain["begin"],
                "client_state_before": {
                    "chunk_env_ids": [],
                    "counter_env_ids": [],
                    "session_ids": [],
                },
                "client_state_after": {
                    "chunk_env_ids": [],
                    "counter_env_ids": [],
                    "session_ids": [],
                },
            }),
            {},
        )
        model_request_artifact = recorder_payload(
            cell_root / "recording/payloads",
            "model-request",
            "model_request",
            freeze_scalar({
                "extracted_preprocessing_output": {"test": model},
                "wire_request": {"cell_id": cell_id},
            }),
            {},
        )
        completion = {
            "schema_version": "wmf-forecast-recording-attempt-v1",
            "study_id": compiler.STUDY_ID,
            "identity": identity,
            "stop_reason": "action_cap",
            "behavioral_result_valid": True,
            "technical_invalid": False,
            "right_censored": False,
            "actions_executed": 2,
            "observation_count": 3,
            "request_count": 1,
            "request_execution": execution,
            "context_reset_artifact": context_artifact,
            "validation_errors": [],
        }
        rows: list[dict] = []
        previous = None

        def append(kind: str, payload: dict) -> None:
            nonlocal previous
            row = journal_row(len(rows), previous, kind, payload)
            rows.append(row)
            previous = row["event_sha256"]

        append("attempt_started", {"identity": identity})
        append("model_context_reset", {"artifact": context_artifact, "model_attached": True})
        clock, artifact = clocks_and_artifacts[0]
        append("observation_captured", {"observation_id": "obs_000000", "control_step": 0, "clock": clock, "artifact": artifact})
        append("model_request_packed", {
            "request_index": 0,
            "action_step_start": 0,
            "current_observation_id": "obs_000000",
            "preceding_observation_id": None,
            "constant_velocity_baseline": "reduces_to_persistence_no_preceding_observation",
            "returned_action_horizon": 2,
            "executed_prefix_horizon": 2,
            "required_future_evidence": TINY_LIMITS[model]["required_future_evidence"],
            "model_request_artifact": model_request_artifact,
            "pack_monotonic_ns": 3_000_000_001,
            "send_monotonic_ns": None,
            "receive_monotonic_ns": None,
            "future_kinds": [],
            "executed_offsets": [],
        })
        append("model_request_sent", {"request_index": 0, "send_monotonic_ns": 3_000_000_002})
        append("model_response_received", {"request_index": 0, "receive_monotonic_ns": 3_000_000_003, "response_artifact": response, "future_kinds": TINY_LIMITS[model]["future_kinds"]})
        append("model_request_completed", {"request_index": 0, "action_chunks_artifact": chunks, "returned_action_shape": [2, 8], "executable_action_shape": [2, 8], "missing_future_evidence": []})
        action_prefix = compiler.canonical_bytes({"dtype": "<f4", "shape": [8], "order": "C"}, ensure_ascii=True)
        for step in (1, 2):
            offset = step - 1
            raw_row = action_raw[offset * 32:(offset + 1) * 32]
            action_identity = {
                "dtype": "<f4", "shape": [8], "order": "C",
                "value_sha256": compiler.sha256_bytes(action_prefix + raw_row),
            }
            append("policy_action_returned", {"action_step": step, "request_index": 0, "chunk_offset": offset, "action_identity": action_identity})
            start_ns = 4_000_000_000 + step
            append("environment_step_started", {"action_step": step, "request_index": 0, "chunk_offset": offset, "env_step_start_monotonic_ns": start_ns, "executed_action_identity": action_identity})
            append("environment_step_completed", {"action_step": step, "request_index": 0, "chunk_offset": offset, "env_step_start_monotonic_ns": start_ns, "env_step_end_monotonic_ns": start_ns + 1, "terminated": False, "truncated": False})
            clock, artifact = clocks_and_artifacts[step]
            append("observation_captured", {"observation_id": f"obs_{step:06d}", "control_step": step, "clock": clock, "artifact": artifact})
        completion["event_count_before_final"] = len(rows)
        completion["journal_tail_sha256_before_final"] = previous
        append("attempt_finalized", copy.deepcopy(completion))
        completion["event_count"] = len(rows)
        completion["journal_tail_sha256"] = previous
        journal_path = cell_root / "recording/events.jsonl"
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        journal_path.write_bytes(b"".join(compiler.freeze.recorder_canonical_bytes(row) + b"\n" for row in rows))
        completion["journal_path"] = str(journal_path.resolve())
        write_json(completion_path, completion)
        viewport = cell_root / "viewport.mp4"
        viewport.write_bytes(f"video-{model}-{layout}".encode())
        journal_descriptor = descriptor(journal_path)
        journal_descriptor.update(event_count=len(rows), tail_sha256=previous)
        cell = {
            "schema_version": TINY_LIMITS[model]["cell_schema"],
            "status": "passed",
            "study_id": compiler.STUDY_ID,
            "cell_id": cell_id,
            "model_config": model,
            "layout_pair_id": layout,
            "layout_arm": "original",
            "command": "left",
            "prompt": identity["prompt"],
            "condition_index": 0,
            **(
                {"effective_seed": 2026091100 + int(layout[1:])}
                if model == "N3"
                else {
                    "effective_model_noise_seed": 1140,
                    "environment_seed": 2026091100 + int(layout[1:]),
                    "noise_semantics": "fixed; this cell is not an independent noise draw",
                }
            ),
            "actions_executed": 2,
            "observation_count": 3,
            "behavioral_model_request_count": 1,
            "behavioral_episode_count": 1,
            "generation_qualification_request_count": 0,
            "source_pins": chain["source_pins"],
            "checkpoint_pin": chain["checkpoint_pin"],
            "adapter_completion": descriptor(completion_path),
            "adapter_journal": journal_descriptor,
            "viewport_video": descriptor(viewport),
            **chain["cell_fields"],
        }
        if model == "D1":
            cell["server_request_receipts"] = [request_descriptor]
        cell_path = write_json(cell_root / "cell_receipt.json", cell)
        aggregate = {
            "schema_version": TINY_LIMITS[model]["aggregate_schema"],
            "status": "passed",
            "exit_code": 0,
            "study_id": compiler.STUDY_ID,
            "phase": "development",
            "layout_pair_id": layout,
            "model_config": model,
            "source_commit": chain["study_commit"],
            "counts": {
                "planned_behavioral_cells": 1,
                "launched_behavioral_cells": 1,
                "completed_valid_behavioral_cells": 1,
                "technically_invalid_behavioral_cells": 0,
                "right_censored_behavioral_cells": 0,
                "unrun_behavioral_cells": 0,
                "actual_behavioral_actions": 2,
                "actual_behavioral_model_requests": 1,
                "new_generation_qualification_requests": 0,
            },
            "failure": None,
            "raw_attempt_root": str(cell_root.resolve()),
            "cell_ids": [cell_id],
            "cell_receipts": [descriptor(cell_path)],
        }
        aggregate["all_simulator_children_reaped" if model == "D1" else "raw_attempt_recoverable_on_gm_pvc"] = True
        aggregate_path = write_json(self.root / f"aggregate-{model}-{layout}.json", aggregate)
        return descriptor(aggregate_path)

    def manifest(self, mode: str, included: set[tuple[str, str]] | None = None) -> Path:
        keys = set(self.aggregates) if included is None else included
        return write_json(self.root / f"manifest-{mode}.json", {
            "schema_version": compiler.INPUT_SCHEMA,
            "study_id": compiler.STUDY_ID,
            "mode": mode,
            "raw_root": str(self.root.resolve()),
            "camera_id": "over_shoulder_left_camera",
            "planned_cells": descriptor(self.planned_path),
            "aggregate_receipts": [
                {"model_id": model, "layout_pair_id": layout, "receipt": self.aggregates[(model, layout)]}
                for model, layout in sorted(keys)
            ],
        })

    def mutate_cell(self, model: str, layout: str, callback) -> None:
        aggregate_path = Path(self.aggregates[(model, layout)]["path"])
        aggregate = json.loads(aggregate_path.read_text())
        cell_path = Path(aggregate["cell_receipts"][0]["path"])
        cell = json.loads(cell_path.read_text())
        callback(cell)
        write_json(cell_path, cell)
        aggregate["cell_receipts"] = [descriptor(cell_path)]
        write_json(aggregate_path, aggregate)
        self.aggregates[(model, layout)] = descriptor(aggregate_path)

    def mutate_journal(self, model: str, layout: str, callback) -> None:
        aggregate_path = Path(self.aggregates[(model, layout)]["path"])
        aggregate = json.loads(aggregate_path.read_text())
        cell_path = Path(aggregate["cell_receipts"][0]["path"])
        cell = json.loads(cell_path.read_text())
        journal_path = Path(cell["adapter_journal"]["path"])
        completion_path = Path(cell["adapter_completion"]["path"])
        rows = [json.loads(line) for line in journal_path.read_text().splitlines()]
        callback(rows)

        def resign(values: list[dict]) -> str:
            previous = None
            for sequence, row in enumerate(values):
                base = {
                    "sequence": sequence,
                    "kind": row["kind"],
                    "wall_time_ns": row["wall_time_ns"],
                    "monotonic_ns": row["monotonic_ns"],
                    "previous_event_sha256": previous,
                    "payload": row["payload"],
                }
                digest = compiler.sha256_bytes(compiler.freeze.recorder_canonical_bytes(base))
                row.clear()
                row.update(base, event_sha256=digest)
                previous = digest
            assert previous is not None
            return previous

        before_final = resign(rows[:-1])
        completion = json.loads(completion_path.read_text())
        completion["journal_tail_sha256_before_final"] = before_final
        rows[-1]["payload"]["journal_tail_sha256_before_final"] = before_final
        tail = resign(rows)
        completion["journal_tail_sha256"] = tail
        journal_path.write_bytes(
            b"".join(compiler.freeze.recorder_canonical_bytes(row) + b"\n" for row in rows)
        )
        write_json(completion_path, completion)
        journal_identity = descriptor(journal_path)
        journal_identity.update(event_count=len(rows), tail_sha256=tail)
        cell["adapter_journal"] = journal_identity
        cell["adapter_completion"] = descriptor(completion_path)
        write_json(cell_path, cell)
        aggregate["cell_receipts"] = [descriptor(cell_path)]
        write_json(aggregate_path, aggregate)
        self.aggregates[(model, layout)] = descriptor(aggregate_path)


def fake_release_validate(entry: dict, **_: object) -> dict:
    return {
        "request_receipt_sha256s": [item["sha256"] for item in entry["server_request_receipts"]]
    }


class NestedPayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_n3_numpy_manifest_is_deeply_verified(self) -> None:
        reference = n3_nested_payload(self.root, "exact_wire_request", b"\x03\x04")
        verified = compiler._verify_n3_payload_reference(
            reference,
            base=self.root,
            raw_root=self.root,
            label="test N3 payload",
            required_role="exact_wire_request",
        )
        self.assertEqual(verified, reference)

        manifest = Path(reference["manifest_path"])
        value = json.loads(manifest.read_text())
        value["structure"]["value_sha256"] = "0" * 64
        write_json(manifest, value)
        tampered = dict(reference, manifest_sha256=compiler.sha256_file(manifest))
        with self.assertRaisesRegex(compiler.CompilerError, "logical identity changed"):
            compiler._verify_n3_payload_reference(
                tampered,
                base=self.root,
                raw_root=self.root,
                label="tampered N3 payload",
                required_role="exact_wire_request",
            )

    def test_tensor_archive_value_identity_and_metadata_are_verified_safely(self) -> None:
        raw = struct.pack("<2f", 1.0, 2.0)
        path = write_torch_archive(self.root / "tensor.pt", raw)
        header = {"kind": "torch", "dtype": "torch.float32", "shape": [2]}
        node = {
            "dtype": header["dtype"],
            "shape": header["shape"],
            "value_sha256": compiler.sha256_bytes(
                compiler.canonical_bytes(header, ensure_ascii=True) + raw
            ),
        }
        compiler._verify_torch_archive(path, node=node, label="test tensor")
        node["value_sha256"] = "f" * 64
        with self.assertRaisesRegex(compiler.CompilerError, "value identity changed"):
            compiler._verify_torch_archive(path, node=node, label="test tensor")

        wrong_metadata = write_torch_archive(
            self.root / "tensor-wrong-metadata.pt",
            raw,
            metadata=TORCH_FLOAT32_MATRIX1X2_METADATA,
        )
        with self.assertRaisesRegex(
            compiler.CompilerError, "serialized shape/dtype layout changed"
        ):
            compiler._verify_torch_archive(
                wrong_metadata,
                node={**node, "value_sha256": compiler.sha256_bytes(
                    compiler.canonical_bytes(header, ensure_ascii=True) + raw
                )},
                label="metadata-tampered tensor",
            )

    def test_each_d1_input_mapping_requires_content_and_artifact_identity(self) -> None:
        raw = d1_exact_mapping(self.root, "raw_inputs", "raw", kind="numpy_array")
        converted = d1_exact_mapping(
            self.root, "converted_inputs", "converted", kind="torch_tensor"
        )
        normalized = d1_exact_mapping(
            self.root, "normalized_model_inputs", "normalized", kind="json_value"
        )
        for label, value, prefix in (
            ("raw", raw, "raw"),
            ("converted", converted, "converted"),
            ("normalized", normalized, "normalized"),
        ):
            compiler._verify_d1_exact_mapping(
                value,
                base=self.root,
                raw_root=self.root,
                label=label,
                prefix=prefix,
            )

        raw["content_sha256"] = "0" * 64
        with self.assertRaisesRegex(compiler.CompilerError, "mapping content hash changed"):
            compiler._verify_d1_exact_mapping(
                raw, base=self.root, raw_root=self.root, label="raw", prefix="raw"
            )

        converted["entries"][0]["data_sha256"] = "1" * 64
        identity = {
            key: converted["entries"][0][key]
            for key in ("key", "kind", "shape", "dtype", "data_sha256")
        }
        converted["content_sha256"] = compiler.sha256_bytes(
            compiler.canonical_bytes([identity])
        )
        with self.assertRaisesRegex(compiler.CompilerError, "data hash changed"):
            compiler._verify_d1_exact_mapping(
                converted,
                base=self.root,
                raw_root=self.root,
                label="converted",
                prefix="converted",
            )

        converted_metadata = d1_exact_mapping(
            self.root, "converted_inputs_metadata", "converted", kind="torch_tensor"
        )
        converted_path = Path(converted_metadata["entries"][0]["path"])
        write_torch_archive(
            converted_path,
            struct.pack("<2f", 1.0, 2.0),
            metadata=TORCH_FLOAT32_MATRIX1X2_METADATA,
        )
        converted_metadata["entries"][0]["file_sha256"] = compiler.sha256_file(
            converted_path
        )
        converted_metadata["entries"][0]["bytes"] = converted_path.stat().st_size
        with self.assertRaisesRegex(
            compiler.CompilerError, "serialized shape/dtype layout changed"
        ):
            compiler._verify_d1_exact_mapping(
                converted_metadata,
                base=self.root,
                raw_root=self.root,
                label="converted metadata tamper",
                prefix="converted",
            )

        normalized_path = Path(normalized["entries"][0]["path"])
        write_json(normalized_path, {"finite": False, "value": 4})
        normalized["entries"][0]["file_sha256"] = compiler.sha256_file(normalized_path)
        normalized["entries"][0]["bytes"] = normalized_path.stat().st_size
        with self.assertRaisesRegex(compiler.CompilerError, "JSON content hash changed"):
            compiler._verify_d1_exact_mapping(
                normalized,
                base=self.root,
                raw_root=self.root,
                label="normalized",
                prefix="normalized",
            )

    def test_descriptor_rejects_ancestor_symlink_before_resolve(self) -> None:
        actual = self.root / "actual"
        actual.mkdir()
        artifact = actual / "artifact.bin"
        artifact.write_bytes(b"bound")
        link = self.root / "linked"
        link.symlink_to(actual, target_is_directory=True)
        value = {
            "path": str(link / artifact.name),
            "sha256": compiler.sha256_file(artifact),
            "bytes": artifact.stat().st_size,
        }
        with self.assertRaisesRegex(compiler.CompilerError, "contains a symlink"):
            compiler._descriptor(
                value,
                base=self.root,
                raw_root=self.root,
                label="symlinked artifact",
            )

        safe = self.root / "safe"
        safe.mkdir()
        safe_artifact = safe / "artifact.bin"
        safe_artifact.write_bytes(b"bound")
        erased_by_abspath = link / ".." / safe.name / safe_artifact.name
        with self.assertRaisesRegex(compiler.CompilerError, "contains a symlink"):
            compiler._descriptor(
                {
                    "path": str(erased_by_abspath),
                    "sha256": compiler.sha256_file(safe_artifact),
                    "bytes": safe_artifact.stat().st_size,
                },
                base=self.root,
                raw_root=self.root,
                label="symlink before dot-dot",
            )

class CompilerEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def patches(self, layouts: tuple[str, ...]):
        return (
            mock.patch.object(compiler, "LAYOUT_IDS", layouts),
            mock.patch.object(compiler, "CONDITIONS", ("original_left",)),
            mock.patch.object(
                compiler,
                "DEVELOPMENT_CONDITION_ORDERS",
                {layout: ("original-left",) for layout in layouts},
            ),
            mock.patch.object(compiler, "MODEL_LIMITS", copy.deepcopy(TINY_LIMITS)),
            mock.patch.object(compiler.freeze, "_validate_cell", side_effect=fake_release_validate),
        )

    def compile(self, fixture: TinyEvidence, manifest: Path, output: Path) -> dict:
        patches = self.patches(fixture.layouts)
        with patches[0], patches[1], patches[2], patches[3], patches[4], mock.patch.object(
            compiler, "PLANNED_CELLS_SHA256", compiler.sha256_file(fixture.planned_path)
        ):
            return compiler.compile_manifest(
                manifest, compiler.sha256_file(manifest), output
            )

    def test_formal_compile_is_deterministic_and_annotation_compatible(self) -> None:
        fixture = TinyEvidence(self.root)
        manifest = fixture.manifest("formal_full")
        first = self.root / "compiled-one"
        second = self.root / "compiled-two"
        receipt = self.compile(fixture, manifest, first)
        self.compile(fixture, manifest, second)
        self.assertTrue(receipt["formal_cohort_complete"])
        self.assertTrue(receipt["safe_for_timing_binding"])
        self.assertEqual(receipt["compiler_science_activity"]["model_requests_issued"], 0)
        dependency = receipt["freeze_validator_dependency"]
        self.assertEqual(
            dependency["sha256"],
            compiler.sha256_file(compiler.FREEZE_PATH),
        )
        self.assertEqual(dependency["bytes"], compiler.FREEZE_PATH.stat().st_size)
        first_files = {
            str(path.relative_to(first)): path.read_bytes()
            for path in sorted(first.rglob("*")) if path.is_file()
        }
        second_files = {
            str(path.relative_to(second)): path.read_bytes()
            for path in sorted(second.rglob("*")) if path.is_file()
        }
        self.assertEqual(first_files, second_files)
        for model in compiler.MODELS:
            timing = json.loads(
                (first / f"{model.lower()}_development_timing_request_inventory.json").read_text()
            )
            self.assertEqual(timing["schema_version"], compiler.TIMING_INVENTORY_SCHEMA)
            self.assertEqual(len(timing["request_receipts"]), 1)
            provenance = json.loads(
                (first / f"{model.lower()}_development_request_provenance.json").read_text()
            )
            packed = provenance["requests"][0]["recorder_model_request"]
            self.assertEqual(packed["role"], "model_request")
            self.assertRegex(packed["payload_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(packed["structure_sha256"], r"^[0-9a-f]{64}$")
            roster = provenance["episode_roster"][0]
            action_path = first / roster["action_manifest_path"]
            action = annotation._validate_action_manifest(
                action_path,
                expected_sha256=roster["action_manifest_sha256"],
                roster_row=roster,
            )
            self.assertEqual(len(action["actions"]), 2)
            _, _, recording_action = annotation._validate_recording_artifacts(
                roster, base=first, stage="development"
            )
            self.assertEqual(recording_action, action)

    def test_formal_mode_fails_closed_without_d1_and_writes_nothing(self) -> None:
        fixture = TinyEvidence(self.root)
        manifest = fixture.manifest("formal_full", {("N3", "D01")})
        output = self.root / "must-not-exist"
        with self.assertRaisesRegex(compiler.CompilerError, "requires all four passed D1"):
            self.compile(fixture, manifest, output)
        self.assertFalse(output.exists())

    def test_partial_d1_is_only_accepted_as_nonrelease_diagnostic(self) -> None:
        fixture = TinyEvidence(self.root, ("D01", "D02"))
        included = {("N3", "D01"), ("N3", "D02"), ("D1", "D01")}
        diagnostic_manifest = fixture.manifest("diagnostic_partial", included)
        output = self.root / "diagnostic"
        receipt = self.compile(fixture, diagnostic_manifest, output)
        self.assertFalse(receipt["formal_cohort_complete"])
        self.assertFalse(receipt["safe_for_timing_binding"])
        self.assertEqual(receipt["status"], "diagnostic_partial_only")
        timing = json.loads(
            (output / "d1_development_timing_request_inventory.diagnostic.json").read_text()
        )
        self.assertEqual(timing["schema_version"], compiler.DIAGNOSTIC_TIMING_INVENTORY_SCHEMA)
        self.assertFalse(timing["safe_for_timing_binding"])
        self.assertEqual(timing["missing_layout_pair_ids"], ["D02"])

    def test_action_chunk_tamper_breaks_retained_action_chain(self) -> None:
        fixture = TinyEvidence(self.root)
        manifest = fixture.manifest("formal_full")
        chunks = next(self.root.rglob("action-chunks.npz"))
        chunks.write_bytes(chunks.read_bytes() + b"tamper")
        output = self.root / "tampered-output"
        with self.assertRaisesRegex(compiler.CompilerError, "(?:file hash|byte count) changed"):
            self.compile(fixture, manifest, output)
        self.assertFalse(output.exists())

    def test_d1_cell_list_must_equal_transported_request_descriptor(self) -> None:
        fixture = TinyEvidence(self.root)
        aggregate_path = Path(fixture.aggregates[("D1", "D01")]["path"])
        aggregate = json.loads(aggregate_path.read_text())
        cell_path = Path(aggregate["cell_receipts"][0]["path"])
        cell = json.loads(cell_path.read_text())
        source_request = Path(cell["server_request_receipts"][0]["path"])
        alternate = source_request.with_name("alternate_receipt.json")
        alternate.write_bytes(source_request.read_bytes())
        cell["server_request_receipts"] = [descriptor(alternate)]
        write_json(cell_path, cell)
        aggregate["cell_receipts"] = [descriptor(cell_path)]
        write_json(aggregate_path, aggregate)
        fixture.aggregates[("D1", "D01")] = descriptor(aggregate_path)
        manifest = fixture.manifest("formal_full")
        output = self.root / "mismatched-output"
        with self.assertRaisesRegex(
            compiler.CompilerError, "differ from transported recorder receipts"
        ):
            self.compile(fixture, manifest, output)
        self.assertFalse(output.exists())

    def test_packed_observation_ids_must_equal_completion_execution(self) -> None:
        fixture = TinyEvidence(self.root)

        def mutate(rows: list[dict]) -> None:
            packed = next(row for row in rows if row["kind"] == "model_request_packed")
            packed["payload"]["current_observation_id"] = "obs_000001"

        fixture.mutate_journal("N3", "D01", mutate)
        manifest = fixture.manifest("formal_full")
        output = self.root / "packed-observation-mismatch"
        with self.assertRaisesRegex(compiler.CompilerError, "packed/completion observation"):
            self.compile(fixture, manifest, output)
        self.assertFalse(output.exists())

    def test_packed_model_request_payload_must_authenticate(self) -> None:
        fixture = TinyEvidence(self.root)

        def mutate(rows: list[dict]) -> None:
            packed = next(row for row in rows if row["kind"] == "model_request_packed")
            packed["payload"]["model_request_artifact"]["payload_sha256"] = "0" * 64

        fixture.mutate_journal("N3", "D01", mutate)
        manifest = fixture.manifest("formal_full")
        output = self.root / "packed-artifact-mismatch"
        with self.assertRaisesRegex(compiler.CompilerError, "payload descriptor hash changed"):
            self.compile(fixture, manifest, output)
        self.assertFalse(output.exists())

    def test_request_completion_must_precede_first_owned_action(self) -> None:
        fixture = TinyEvidence(self.root)

        def mutate(rows: list[dict]) -> None:
            complete = next(i for i, row in enumerate(rows) if row["kind"] == "model_request_completed")
            proposal = next(i for i, row in enumerate(rows) if row["kind"] == "policy_action_returned")
            rows[complete]["kind"], rows[proposal]["kind"] = rows[proposal]["kind"], rows[complete]["kind"]
            rows[complete]["payload"], rows[proposal]["payload"] = rows[proposal]["payload"], rows[complete]["payload"]

        fixture.mutate_journal("N3", "D01", mutate)
        manifest = fixture.manifest("formal_full")
        output = self.root / "late-completion"
        with self.assertRaisesRegex(compiler.CompilerError, "completed after its first owned action"):
            self.compile(fixture, manifest, output)
        self.assertFalse(output.exists())

    def test_recorded_context_reset_binds_server_receipt_and_empty_client_state(self) -> None:
        expected = {"passed": True, "server_context_id": "context-1"}

        def validate(structure: dict) -> dict:
            artifact = {
                "role": "context_reset",
                "payload_sha256": "a" * 64,
                "array_count": 0,
                "structure": freeze_scalar(structure),
                "artifact": None,
            }
            rows = [
                {
                    "sequence": 0,
                    "kind": "model_context_reset",
                    "payload": {"artifact": artifact},
                },
                {"sequence": 1, "kind": "model_request_packed", "payload": {}},
            ]
            with mock.patch.object(
                compiler, "_verify_payload_descriptor", return_value=artifact
            ):
                return compiler._validate_context_payload(
                    rows=rows,
                    completion={"context_reset_artifact": artifact},
                    completion_path=self.root / "completion.json",
                    raw_root=self.root,
                    expected_receipt=expected,
                    cell_id="test-cell",
                )

        empty = {
            "chunk_env_ids": [],
            "counter_env_ids": [],
            "session_ids": [],
        }
        accepted = validate(
            {
                **expected,
                "client_state_before": empty,
                "client_state_after": empty,
            }
        )
        self.assertEqual(accepted["role"], "context_reset")
        with self.assertRaisesRegex(
            compiler.CompilerError, "recorder client reset state changed"
        ):
            validate(
                {
                    **expected,
                    "client_state_before": empty,
                    "client_state_after": {**empty, "chunk_env_ids": [0]},
                }
            )

    def test_n3_source_pin_and_end_attestation_fail_closed(self) -> None:
        for mutation, expected in (
            (
                lambda cell: cell["source_pins"].__setitem__("cosmos_commit", "2" * 40),
                "model source pins changed",
            ),
            (
                lambda cell: cell["server_end_receipt"].__setitem__("passed", False),
                "N3 end attestation changed",
            ),
            (
                lambda cell: cell["server_begin_receipt"]["cache_reset_evidence"].__setitem__(
                    "wrapper_request_index_reset_to_zero", False
                ),
                "N3 temporal/cache reset attestation changed",
            ),
        ):
            with self.subTest(expected=expected):
                nested = self.root / re.sub(r"[^A-Za-z0-9]+", "-", expected)
                nested.mkdir()
                fixture = TinyEvidence(nested)
                fixture.mutate_cell("N3", "D01", mutation)
                manifest = fixture.manifest("formal_full")
                with self.assertRaisesRegex(compiler.CompilerError, expected):
                    self.compile(fixture, manifest, nested / "output")

    def test_d1_runtime_identity_and_episode_manifest_fail_closed(self) -> None:
        identity_root = self.root / "identity"
        identity_root.mkdir()
        identity_fixture = TinyEvidence(identity_root)

        def mutate_identity(cell: dict) -> None:
            ready_path = Path(cell["server_ready"]["path"])
            ready = json.loads(ready_path.read_text())
            identity_path = Path(ready["runtime_identity"]["path"])
            identity = json.loads(identity_path.read_text())
            identity["checkpoint"]["aggregate_sha256"] = "3" * 64
            write_json(identity_path, identity)
            ready["runtime_identity"] = descriptor(identity_path)
            write_json(ready_path, ready)
            cell["server_ready"] = descriptor(ready_path)

        identity_fixture.mutate_cell("D1", "D01", mutate_identity)
        identity_manifest = identity_fixture.manifest("formal_full")
        with self.assertRaisesRegex(compiler.CompilerError, "checkpoint identity changed"):
            self.compile(identity_fixture, identity_manifest, identity_root / "output")

        manifest_root = self.root / "episode"
        manifest_root.mkdir()
        episode_fixture = TinyEvidence(manifest_root)

        def mutate_episode(cell: dict) -> None:
            path = Path(cell["server_episode_manifest"]["path"])
            value = json.loads(path.read_text())
            value["request_count"] = 0
            write_json(path, value)
            cell["server_episode_manifest"] = descriptor(path)

        episode_fixture.mutate_cell("D1", "D01", mutate_episode)
        episode_manifest = episode_fixture.manifest("formal_full")
        with self.assertRaisesRegex(compiler.CompilerError, "episode manifest changed"):
            self.compile(episode_fixture, episode_manifest, manifest_root / "output")

    def test_d1_claim_and_two_rank_reset_fail_closed(self) -> None:
        claim_root = self.root / "claim"
        claim_root.mkdir()
        claim_fixture = TinyEvidence(claim_root)

        def mutate_claim(cell: dict) -> None:
            path = Path(cell["simulator_claim"]["path"])
            value = json.loads(path.read_text())
            value["status"] = "released"
            write_json(path, value)
            cell["simulator_claim"] = descriptor(path)

        claim_fixture.mutate_cell("D1", "D01", mutate_claim)
        claim_manifest = claim_fixture.manifest("formal_full")
        with self.assertRaisesRegex(compiler.CompilerError, "simulator claim changed"):
            self.compile(claim_fixture, claim_manifest, claim_root / "output")

        reset_root = self.root / "reset"
        reset_root.mkdir()
        reset_fixture = TinyEvidence(reset_root)

        def mutate_reset(cell: dict) -> None:
            path = Path(cell["server_reset_receipt"]["path"])
            value = json.loads(path.read_text())
            value["rank_receipts"][1]["after"]["fields"]["kv_cache1"]["is_none"] = False
            write_json(path, value)
            cell["server_reset_receipt"] = descriptor(path)

        reset_fixture.mutate_cell("D1", "D01", mutate_reset)
        reset_manifest = reset_fixture.manifest("formal_full")
        with self.assertRaisesRegex(compiler.CompilerError, "temporal fields were not cleared"):
            self.compile(reset_fixture, reset_manifest, reset_root / "output")

    def test_d1_first_request_cache_reinitialization_fails_closed(self) -> None:
        fixture = TinyEvidence(self.root)
        aggregate = json.loads(
            Path(fixture.aggregates[("D1", "D01")]["path"]).read_text()
        )
        cell = json.loads(Path(aggregate["cell_receipts"][0]["path"]).read_text())
        request = json.loads(Path(cell["server_request_receipts"][0]["path"]).read_text())
        metrics = request["temporal_and_cache_rank_metrics"]
        compiler._verify_d1_first_request_temporal_metrics(metrics, label="D1 request 0")
        metrics[0]["temporal_before"]["fields"]["kv_cache1"]["is_none"] = False
        with self.assertRaisesRegex(compiler.CompilerError, "empty to initialized"):
            compiler._verify_d1_first_request_temporal_metrics(
                metrics, label="D1 request 0"
            )

    def test_d1_exact_nine_five_decode_schedule_is_structural_only(self) -> None:
        schedule = compiler.MODEL_LIMITS["D1"]["development_decode_shapes"]

        def request_for(kind: str) -> dict:
            shapes = schedule[kind]
            return {
                "latent_video": {"shape": list(shapes["latent"])},
                "offline_decode": {
                    "decoded_tensor": {"shape": list(shapes["tensor"])},
                    "decoded_rgb": {"shape": list(shapes["rgb"])},
                },
            }

        # The compiler accepts both exact retained artifact forms, but this
        # helper deliberately returns no timing/mapping assertion.
        self.assertIsNone(compiler._verify_d1_development_decode_shapes(
            request_for("full_conditioning_origin"),
            request_index=0,
            limits=compiler.MODEL_LIMITS["D1"],
            label="full",
        ))
        self.assertIsNone(compiler._verify_d1_development_decode_shapes(
            request_for("incremental_standalone"),
            request_index=1,
            limits=compiler.MODEL_LIMITS["D1"],
            label="incremental",
        ))

        for request_index, wrong_kind, message in (
            (0, "incremental_standalone", "retained latent shape/schedule changed"),
            (1, "full_conditioning_origin", "retained latent shape/schedule changed"),
        ):
            with self.subTest(request_index=request_index), self.assertRaisesRegex(
                compiler.CompilerError, message
            ):
                compiler._verify_d1_development_decode_shapes(
                    request_for(wrong_kind),
                    request_index=request_index,
                    limits=compiler.MODEL_LIMITS["D1"],
                    label="wrong modulo",
                )

        malformed = request_for("incremental_standalone")
        malformed["offline_decode"]["decoded_rgb"]["shape"] = [7, 352, 640, 3]
        with self.assertRaisesRegex(compiler.CompilerError, "RGB shape/schedule changed"):
            compiler._verify_d1_development_decode_shapes(
                malformed,
                request_index=1,
                limits=compiler.MODEL_LIMITS["D1"],
                label="arbitrary",
            )

    def test_d1_ready_requires_exact_cohort_and_nonempty_process_ids(self) -> None:
        for suffix, mutation, expected in (
            (
                "cohort",
                lambda ready: ready.__setitem__("expected_cell_ids", []),
                "server-ready cohort changed",
            ),
            (
                "run-id",
                lambda ready: ready.__setitem__("run_id", None),
                "server-ready run_id is invalid",
            ),
        ):
            with self.subTest(suffix=suffix):
                nested = self.root / suffix
                nested.mkdir()
                fixture = TinyEvidence(nested)

                def mutate(cell: dict) -> None:
                    ready_path = Path(cell["server_ready"]["path"])
                    ready = json.loads(ready_path.read_text())
                    mutation(ready)
                    write_json(ready_path, ready)
                    cell["server_ready"] = descriptor(ready_path)

                fixture.mutate_cell("D1", "D01", mutate)
                manifest = fixture.manifest("formal_full")
                with self.assertRaisesRegex(compiler.CompilerError, expected):
                    self.compile(fixture, manifest, nested / "output")

    def test_block_context_ids_cannot_be_reused_or_aliased(self) -> None:
        duplicate_n3 = [
            {"model_context": {"server_context_id": "context"}},
            {"model_context": {"server_context_id": "context"}},
        ]
        with self.assertRaisesRegex(compiler.CompilerError, "context ID was reused"):
            compiler._verify_block_context_uniqueness(
                duplicate_n3, model="N3", layout="D01"
            )

        aliased_d1 = [
            {
                "cell": {
                    "server_begin_receipt": {
                        "episode_context_id": "episode-a",
                        "client_session_id": "session-a",
                    }
                }
            },
            {
                "cell": {
                    "server_begin_receipt": {
                        "episode_context_id": "session-a",
                        "client_session_id": "session-b",
                    }
                }
            },
        ]
        with self.assertRaisesRegex(compiler.CompilerError, "reused or aliased"):
            compiler._verify_block_context_uniqueness(
                aliased_d1, model="D1", layout="D01"
            )

    def test_planned_model_and_environment_seeds_are_authoritative(self) -> None:
        planned_root = self.root / "planned"
        planned_root.mkdir()
        planned_fixture = TinyEvidence(planned_root)
        lines = planned_fixture.planned_path.read_text().splitlines()
        lines = [
            line.rsplit(",", 1)[0] + ",999" if ",D1," in line else line
            for line in lines
        ]
        planned_fixture.planned_path.write_text("\n".join(lines) + "\n")
        planned_manifest = planned_fixture.manifest("formal_full")
        with self.assertRaisesRegex(compiler.CompilerError, "planned effective policy seed changed"):
            self.compile(planned_fixture, planned_manifest, planned_root / "output")

        n3_root = self.root / "n3-seed"
        n3_root.mkdir()
        n3_fixture = TinyEvidence(n3_root)
        n3_fixture.mutate_cell(
            "N3", "D01", lambda cell: cell.__setitem__("effective_seed", 7)
        )
        n3_manifest = n3_fixture.manifest("formal_full")
        with self.assertRaisesRegex(compiler.CompilerError, "N3 effective seed changed"):
            self.compile(n3_fixture, n3_manifest, n3_root / "output")

        d1_root = self.root / "d1-environment-seed"
        d1_root.mkdir()
        d1_fixture = TinyEvidence(d1_root)
        d1_fixture.mutate_cell(
            "D1", "D01", lambda cell: cell.__setitem__("environment_seed", 2026091000)
        )
        d1_manifest = d1_fixture.manifest("formal_full")
        with self.assertRaisesRegex(
            compiler.CompilerError, "D1 model/environment seed contract changed"
        ):
            self.compile(d1_fixture, d1_manifest, d1_root / "output")


if __name__ == "__main__":
    unittest.main()
