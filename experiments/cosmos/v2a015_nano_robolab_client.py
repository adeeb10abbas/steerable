"""V2-A015 identity checks over the frozen Cosmos3 RoboLab client."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from v2_robolab_client import V2Cosmos3Client

MODEL_ID = "cosmos3_nano_policy_droid"
CHECKPOINT_REVISION = "6706d7680581c255ff61e0f3bb49d90eac55c79e"
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
            checkpoint_revision=CHECKPOINT_REVISION,
            server_sampling_seed=int(response["sampling_seed"]),
            amendment_id=AMENDMENT_ID,
            arm_id=ARM_ID,
            guidance=GUIDANCE,
            baseline_guidance=BASELINE_GUIDANCE,
            baseline_result_artifact=BASELINE_RESULT_ARTIFACT,
            baseline_result_sha256=BASELINE_RESULT_SHA256,
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
            model_id=MODEL_ID,
            checkpoint_revision=CHECKPOINT_REVISION,
            amendment_id=AMENDMENT_ID,
            arm_id=ARM_ID,
            guidance=GUIDANCE,
            baseline_guidance=BASELINE_GUIDANCE,
            baseline_result_artifact=BASELINE_RESULT_ARTIFACT,
            baseline_result_sha256=BASELINE_RESULT_SHA256,
            cfg_intervention=(
                "Joint action-video guidance changed from the archived g=3 baseline "
                "to g=1; all other frozen sampling and controller settings are unchanged."
            ),
        )
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
