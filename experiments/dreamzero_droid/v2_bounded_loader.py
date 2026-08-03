#!/usr/bin/env python3
"""Bounded-host-memory loader overlay for the exact DreamZero release."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file


OFFICIAL_BASE_VLA_SHA256 = (
    "6e926096f20b4c1bbba98c3e270bde1d5260609e4155de27c91c911f2f8f8e20"
)
OFFICIAL_ACTION_HEAD_SHA256 = (
    "7193cd73423472aa252bee73bd80e0d673c89d773ec852e90f50154729b50845"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_signature(state: dict[str, Any]) -> str:
    """Hash names, shapes, and dtypes without reading tensor payload pages."""
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        digest.update(key.encode())
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\n")
    return digest.hexdigest()


class _ConstructionTorchProxy:
    """Delegate to torch while making only constructor ``torch.load`` inert."""

    def __init__(self, delegate: Any, load_calls: list[dict[str, Any]]) -> None:
        self._delegate = delegate
        self._load_calls = load_calls

    def load(self, path: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._load_calls.append(
            {
                "path": str(path),
                "positional_argument_count": len(args),
                "keyword_arguments": sorted(kwargs),
            }
        )
        return {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


@contextmanager
def _suppress_redundant_component_loads() -> Any:
    """Skip exactly the released T5, CLIP, and VAE constructor loads.

    The old audited DreamZero patch moved these three load calls under the
    existing ``skip_component_loading`` condition. This runtime overlay has the
    identical construction-only effect without modifying the checked-out
    external source. All monkeypatches are restored before checkpoint shards
    are assigned, and strict complete-key/no-meta checks remain authoritative.
    """
    import groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf as action_module

    action_path = Path(action_module.__file__).resolve()
    action_hash = _sha256(action_path)
    if action_hash != OFFICIAL_ACTION_HEAD_SHA256:
        raise RuntimeError(
            "DreamZero component suppression refuses a non-official action head: "
            f"{action_hash}"
        )

    ensure_calls: list[dict[str, Any]] = []
    torch_load_calls: list[dict[str, Any]] = []
    state_load_calls: list[dict[str, Any]] = []
    original_ensure_file = action_module.ensure_file
    original_action_torch = action_module.torch
    original_load_state_dict = torch.nn.Module.load_state_dict

    def suppressed_ensure_file(path: Any, filename: str, **kwargs: Any) -> str:
        ensure_calls.append(
            {
                "configured_path": None if path is None else str(path),
                "filename": filename,
                "keyword_arguments": sorted(kwargs),
            }
        )
        return os.devnull

    def suppressed_load_state_dict(
        module: torch.nn.Module,
        state_dict: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> SimpleNamespace:
        state_load_calls.append(
            {
                "target_type": f"{type(module).__module__}.{type(module).__qualname__}",
                "state_key_count": len(state_dict),
                "positional_argument_count": len(args),
                "keyword_arguments": sorted(kwargs),
            }
        )
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    action_module.ensure_file = suppressed_ensure_file
    action_module.torch = _ConstructionTorchProxy(torch, torch_load_calls)
    torch.nn.Module.load_state_dict = suppressed_load_state_dict
    record = {
        "official_action_head_path": str(action_path),
        "official_action_head_sha256": action_hash,
        "ensure_file_calls": ensure_calls,
        "torch_load_calls": torch_load_calls,
        "load_state_dict_calls": state_load_calls,
    }
    try:
        yield record
    finally:
        torch.nn.Module.load_state_dict = original_load_state_dict
        action_module.torch = original_action_torch
        action_module.ensure_file = original_ensure_file


def install_bounded_loader(*, contract_path: Path) -> None:
    """Replace only ``VLA.from_pretrained`` with an assign-only shard loader.

    The exact released forward path remains untouched. ``assign=True`` installs
    the original safetensors storage, shape, and dtype into meta-initialized
    parameters without a second host copy. Strict post-load checks reject any
    incomplete or structurally different model before inference can start.
    """
    import groot.vla.model.dreamzero.base_vla as official_base

    official_path = Path(official_base.__file__).resolve()
    official_hash = _sha256(official_path)
    if official_hash != OFFICIAL_BASE_VLA_SHA256:
        raise RuntimeError(
            "DreamZero bounded loader refuses a non-official base_vla.py: "
            f"{official_hash}"
        )

    VLA = official_base.VLA
    VLAConfig = official_base.VLAConfig
    contract_path = Path(contract_path).resolve()
    contract_path.parent.mkdir(parents=True, exist_ok=True)

    def bounded_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config: Any = None,
    ):
        del config  # Preserve the released loader's checkpoint-config behavior.
        checkpoint_root = Path(pretrained_model_name_or_path).resolve()
        single_path = checkpoint_root / "model.safetensors"
        index_path = checkpoint_root / "model.safetensors.index.json"
        config_path = checkpoint_root / "config.json"

        config_dict = json.loads(config_path.read_text())
        action_inner = config_dict.get("action_head_cfg", {}).get("config", {})
        original_skip_present = "skip_component_loading" in action_inner
        original_skip_value = bool(action_inner.get("skip_component_loading", False))
        # The complete DreamZero checkpoint supplies these tensors. Skipping
        # redundant base-component reads is required for bounded construction;
        # complete key coverage below proves that no component value is omitted.
        action_inner["skip_component_loading"] = True
        loaded_config = VLAConfig(**config_dict)

        if (
            "config" in loaded_config.action_head_cfg
            and isinstance(loaded_config.action_head_cfg["config"], dict)
        ):
            loaded_config.action_head_cfg["config"]["defer_lora_injection"] = False
        else:
            loaded_config.action_head_cfg["defer_lora_injection"] = False

        with _suppress_redundant_component_loads() as component_suppression:
            with init_empty_weights(include_buffers=False):
                model = cls(loaded_config)

        suppression_counts = {
            "ensure_file": len(component_suppression["ensure_file_calls"]),
            "torch_load": len(component_suppression["torch_load_calls"]),
            "load_state_dict": len(component_suppression["load_state_dict_calls"]),
        }
        if suppression_counts != {
            "ensure_file": 3,
            "torch_load": 3,
            "load_state_dict": 3,
        }:
            raise RuntimeError(
                "DreamZero constructor suppression did not match the exact "
                f"three-component contract: {suppression_counts}"
            )

        if index_path.is_file():
            index = json.loads(index_path.read_text())
            shard_files = sorted(set(index["weight_map"].values()))
        elif single_path.is_file():
            shard_files = [single_path.name]
        else:
            raise FileNotFoundError(
                f"No model.safetensors or index under {checkpoint_root}"
            )

        loaded_keys: set[str] = set()
        unexpected_keys: set[str] = set()
        shard_records: list[dict[str, Any]] = []
        transformed_base_layer_keys = 0
        for shard_index, shard_file in enumerate(shard_files, start=1):
            shard_path = checkpoint_root / shard_file
            state = load_file(str(shard_path), device="cpu")
            normalized: dict[str, Any] = {}
            for key, value in state.items():
                normalized_key = key.replace(".base_layer.", ".")
                transformed_base_layer_keys += int(normalized_key != key)
                if normalized_key in normalized or normalized_key in loaded_keys:
                    raise RuntimeError(
                        f"Duplicate normalized checkpoint key: {normalized_key}"
                    )
                normalized[normalized_key] = value
            result = model.load_state_dict(normalized, strict=False, assign=True)
            unexpected_keys.update(result.unexpected_keys)
            loaded_keys.update(normalized)
            dtype_counts: dict[str, int] = {}
            tensor_nbytes = 0
            for tensor in normalized.values():
                dtype = str(tensor.dtype)
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                tensor_nbytes += tensor.numel() * tensor.element_size()
            shard_records.append(
                {
                    "index": shard_index,
                    "file": shard_file,
                    "bytes": shard_path.stat().st_size,
                    "tensor_count": len(normalized),
                    "tensor_nbytes": tensor_nbytes,
                    "dtype_counts": dtype_counts,
                    "state_structure_sha256": _state_signature(normalized),
                }
            )
            del state, normalized, result
            gc.collect()

        model_state = model.state_dict()
        expected_keys = set(model_state)
        missing_keys = sorted(expected_keys - loaded_keys)
        unexpected = sorted(unexpected_keys | (loaded_keys - expected_keys))
        meta_parameters = sorted(
            name for name, value in model.named_parameters() if value.is_meta
        )
        meta_buffers = sorted(
            name for name, value in model.named_buffers() if value.is_meta
        )

        # Restore the released config value after its one-time constructor role.
        model_inner = model.config.action_head_cfg.get("config", {})
        if original_skip_present:
            model_inner["skip_component_loading"] = original_skip_value
        else:
            model_inner.pop("skip_component_loading", None)
        if hasattr(model.action_head, "config"):
            model.action_head.config.skip_component_loading = original_skip_value

        passed = not (missing_keys or unexpected or meta_parameters or meta_buffers)
        contract = {
            "schema_version": "vla-wam-shared-v2-dreamzero-bounded-loader-v1",
            "status": "passed" if passed else "failed",
            "role": "load_time_compatibility_only",
            "official_base_vla_path": str(official_path),
            "official_base_vla_sha256": official_hash,
            "overlay_path": str(Path(__file__).resolve()),
            "overlay_sha256": _sha256(Path(__file__).resolve()),
            "checkpoint_root": str(checkpoint_root),
            "checkpoint_config_sha256": _sha256(config_path),
            "checkpoint_index_sha256": _sha256(index_path) if index_path.is_file() else None,
            "shards": shard_records,
            "loaded_key_count": len(loaded_keys),
            "expected_model_key_count": len(expected_keys),
            "loaded_state_structure_sha256": _state_signature(model_state),
            "transformed_base_layer_key_count": transformed_base_layer_keys,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected,
            "meta_parameters": meta_parameters,
            "meta_buffers": meta_buffers,
            "assign_preserves_checkpoint_tensor_dtype_and_storage": True,
            "dtype_or_value_conversion_performed": False,
            "temporary_skip_component_loading": {
                "used_during_empty_construction": True,
                "original_present": original_skip_present,
                "original_value": original_skip_value,
                "restored_after_complete_checkpoint_assignment": True,
            },
            "construction_only_component_suppression": {
                **component_suppression,
                "call_counts": suppression_counts,
                "expected_components": ["T5", "CLIP", "VAE"],
                "official_source_modified": False,
                "all_monkeypatches_restored_before_checkpoint_assignment": True,
            },
            "forward_path_modified": False,
            "passed": passed,
        }
        contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
        del model_state
        gc.collect()
        if not passed:
            raise RuntimeError(
                "Bounded DreamZero loader contract failed; see " f"{contract_path}"
            )
        return model

    VLA.from_pretrained = classmethod(bounded_from_pretrained)
