"""V2-A015 identity checks over the frozen Cosmos3 RoboLab client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from v2_robolab_client import V2Cosmos3Client, _sha256

MODEL_ID = "cosmos3_nano_policy_droid"
CHECKPOINT_ID = "nvidia/Cosmos3-Nano-Policy-DROID"
CHECKPOINT_REVISION = "6706d7680581c255ff61e0f3bb49d90eac55c79e"
OFFICIAL_SOURCE_COMMIT = "411d25b2e35bc441126f48c44a4b93e1c0564274"
AMENDMENT_ID = "V2-A015"
ARM_ID = "cosmos3_nano_no_cfg_g1"
GUIDANCE = 1.0
BASELINE_GUIDANCE = 3.0
BASELINE_RESULT_ARTIFACT = (
    "artifacts/vla_wam_shared_v2/pilot/expansion/"
    "cosmos3_nano_policy_droid_direct_gate.json"
)
BASELINE_RESULT_SHA256 = "4a6cc1d61593c7ba5272e1707f6bbe51261f7d23438070992bd75fd9e95fdb93"


class V2A015NanoCosmos3Client(V2Cosmos3Client):
    """Retain unchanged actions/futures and bind them to the g=1 arm."""

    def __init__(
        self,
        *,
        environment_seed: int,
        amendment_path: Path,
        fixed_observation_gate_path: Path,
        **kwargs: Any,
    ) -> None:
        sampling_seed_base = int(kwargs["sampling_seed_base"])
        if int(environment_seed) != sampling_seed_base:
            raise ValueError("V2-A015 environment and sampling seed labels must match")
        self.environment_seed = int(environment_seed)
        self.amendment_path = Path(amendment_path).resolve()
        self.fixed_observation_gate_path = Path(fixed_observation_gate_path).resolve()
        for path in (self.amendment_path, self.fixed_observation_gate_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        super().__init__(**kwargs)

    def _unpack_response(self, response: dict) -> np.ndarray:
        expected = {
            "sampling_seed": self.sampling_seed_base,
            "amendment_id": AMENDMENT_ID,
            "arm_id": ARM_ID,
            "guidance": GUIDANCE,
            "baseline_guidance": BASELINE_GUIDANCE,
            "baseline_result_artifact": BASELINE_RESULT_ARTIFACT,
            "baseline_result_sha256": BASELINE_RESULT_SHA256,
        }
        for key, value in expected.items():
            if response.get(key) != value:
                raise ValueError(
                    f"V2-A015 server metadata mismatch for {key}: "
                    f"expected={value!r}, observed={response.get(key)!r}"
                )
        action = super()._unpack_response(response)
        if not np.isfinite(action).all():
            raise ValueError("V2-A015 g=1 server returned a non-finite action")
        self.request_records[-1].update(
            model_id=MODEL_ID,
            checkpoint=CHECKPOINT_ID,
            checkpoint_revision=CHECKPOINT_REVISION,
            official_source_commit=OFFICIAL_SOURCE_COMMIT,
            environment_seed=self.environment_seed,
            prompt=self.prompt,
            requested_relation=self._relation(),
            server_sampling_seed=int(response["sampling_seed"]),
            amendment_id=AMENDMENT_ID,
            arm_id=ARM_ID,
            guidance=GUIDANCE,
            baseline_guidance=BASELINE_GUIDANCE,
            baseline_result_artifact=BASELINE_RESULT_ARTIFACT,
            baseline_result_sha256=BASELINE_RESULT_SHA256,
            amendment_sha256=_sha256(self.amendment_path),
            fixed_observation_gate_sha256=_sha256(
                self.fixed_observation_gate_path
            ),
        )
        return action

    def _write_trace(self) -> None:
        already_written = self._trace_written
        super()._write_trace()
        if already_written or not self._trace_written:
            return
        relation = self._relation()
        metadata_path = Path(self.action_trace_dir) / (
            f"seed{self.sampling_seed_base}_{relation}_executed_actions.json"
        )
        metadata = json.loads(metadata_path.read_text())
        metadata.update(
            schema_version=(
                "vla-wam-shared-v2-cosmos3-nano-v2a015-g1-"
                "action-future-trace-v1"
            ),
            model_id=MODEL_ID,
            checkpoint=CHECKPOINT_ID,
            checkpoint_revision=CHECKPOINT_REVISION,
            official_source_commit=OFFICIAL_SOURCE_COMMIT,
            environment_seed=self.environment_seed,
            requested_relation=relation,
            amendment_id=AMENDMENT_ID,
            arm_id=ARM_ID,
            guidance=GUIDANCE,
            baseline_guidance=BASELINE_GUIDANCE,
            baseline_result_artifact=BASELINE_RESULT_ARTIFACT,
            baseline_result_sha256=BASELINE_RESULT_SHA256,
            amendment={
                "path": str(self.amendment_path),
                "sha256": _sha256(self.amendment_path),
            },
            fixed_observation_release_gate={
                "path": str(self.fixed_observation_gate_path),
                "sha256": _sha256(self.fixed_observation_gate_path),
            },
            cfg_intervention=(
                "Joint action-video guidance changed from the archived g=3 baseline "
                "to g=1; all other frozen sampling and controller settings are unchanged."
            ),
        )
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
