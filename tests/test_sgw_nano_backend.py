from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.workshops.spatial_grounding_v1 import nano_backend
from experiments.workshops.spatial_grounding_v1.adapters import AdapterError


def test_concrete_nano_factory_constructs_audited_native_service(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class NativeArgs:
        def __init__(self, **kwargs: object) -> None:
            captured["args"] = kwargs

    @dataclass(frozen=True)
    class NativeConfig:
        seed: int
        deterministic_seed: bool
        guidance: float = 3.0
        num_steps: int = 4
        shift: float = 5.0
        conditioning_fps: float = 15.0
        action_chunk_size: int = 32
        action_dim: int = 8
        resolution: str = "480"
        action_space: str = "joint_pos"
        use_state: bool = True
        history_length: int = 1
        decode_video: bool = True
        domain_name: str = "droid_lerobot"
        image_height: int = 540
        image_width: int = 640

    class NativeService:
        def __init__(self, args: NativeArgs) -> None:
            captured["service_args"] = args
            self.cfg = NativeConfig(seed=1140, deterministic_seed=True)

        def infer(self, observation: dict) -> dict:
            captured["observation"] = observation
            return {
                "action": np.zeros((32, 8), dtype=np.float32),
                "video": np.ones((33, 2, 2, 3), dtype=np.uint8),
            }

    monkeypatch.setattr(
        nano_backend,
        "_native_module",
        lambda: SimpleNamespace(
            __file__="/pinned/source/cosmos_framework/scripts/action_policy_server_robolab.py",
            RobolabServerArgs=NativeArgs,
            RobolabPolicyService=NativeService,
        ),
    )
    monkeypatch.setattr(nano_backend, "_git_revision", lambda _: nano_backend.NANO_CONFIG["source_commit"])
    monkeypatch.setattr(nano_backend, "_checkpoint_identity", lambda _: (nano_backend.NANO_CONFIG["revision"], nano_backend.NANO_CONFIG["asset"]))
    monkeypatch.setenv("SGW01_NANO_CHECKPOINT_PATH", "/pinned/checkpoint")
    monkeypatch.setenv("SGW01_NANO_SOURCE_ROOT", "/pinned/source")

    backend = nano_backend.build_pinned_nano_backend()
    result = backend.predict({"observation/image": [[1]]}, "static", 8300)

    args = captured["args"]
    assert isinstance(args, dict)
    assert args["guidance"] == 3.0
    assert args["num_steps"] == 4
    assert args["shift"] == 5.0
    assert args["history_length"] == 1
    assert args["action_chunk_size"] == 32
    assert args["decode_video"] is True
    assert "image_height" not in args and "image_width" not in args
    assert captured["observation"]["prompt"] == "static"  # type: ignore[index]
    assert backend.service.cfg.seed == 8300
    assert np.asarray(result["action"]).shape == (32, 8)
    assert np.asarray(result["future"]).shape[0] == 33

    class DriftedService(NativeService):
        def __init__(self, args: NativeArgs) -> None:
            super().__init__(args)
            self.cfg = NativeConfig(seed=1140, deterministic_seed=True, guidance=2.0)

    monkeypatch.setattr(
        nano_backend,
        "_native_module",
        lambda: SimpleNamespace(
            __file__="/pinned/source/cosmos_framework/scripts/action_policy_server_robolab.py",
            RobolabServerArgs=NativeArgs,
            RobolabPolicyService=DriftedService,
        ),
    )
    with pytest.raises(AdapterError, match="resolved config"):
        nano_backend.build_pinned_nano_backend()


def test_source_identity_rejected_before_native_import(monkeypatch) -> None:
    monkeypatch.setenv("SGW01_NANO_CHECKPOINT_PATH", "/pinned/checkpoint")
    monkeypatch.setenv("SGW01_NANO_SOURCE_ROOT", "/pinned/source")
    monkeypatch.setattr(nano_backend, "_git_revision", lambda _: "wrong-source")
    monkeypatch.setattr(nano_backend, "_native_module", lambda: pytest.fail("native code imported"))
    monkeypatch.setattr(nano_backend, "_checkpoint_identity", lambda _: pytest.fail("checkpoint read"))
    with pytest.raises(AdapterError, match="source commit"):
        nano_backend.build_pinned_nano_backend()
