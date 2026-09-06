from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.online_correction_v4.coordinator import ClusterBinding
from tools.render_v4_episode_subset import ROOT, render_subset


class RenderV4EpisodeSubsetTests(unittest.TestCase):
    def test_renders_one_frozen_pilot_retry(self) -> None:
        queue_path = (
            ROOT
            / "artifacts/online_correction_v4/setup/"
            "object_pair_g7_pilot_queue.jsonl"
        )
        episode_id = json.loads(
            queue_path.read_text(encoding="utf-8").splitlines()[0]
        )["episode_id"]
        with tempfile.TemporaryDirectory() as tmp:
            render_root = Path(tmp) / "rendered"
            receipt = render_subset(
                runtime_lock_path=ROOT
                / "artifacts/online_correction_v4/setup/"
                "object_pair_g7_runtime_lock.released.json",
                queue_path=queue_path,
                campaign_config_path=ROOT
                / "docs/online_correction_v4/campaign.json",
                lane_template_path=ROOT
                / "deploy/k8s/v4_lane_bundle/"
                "g7-object-pair-pilot-spec.example.json",
                episode_ids=[episode_id],
                render_root=render_root,
                cluster=ClusterBinding(
                    kube_context="test",
                    namespace="test",
                    pvc="test",
                    output_parent="/tmp/v4-test",
                ),
                attempt_index=100,
            )
            self.assertEqual(receipt["behavioral_episode_count"], 1)
            bundle = Path(receipt["rows"][0]["bundle_root"])
            dispatch = json.loads(
                (bundle / ".bindings/lane_dispatch_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(dispatch["episode_ids"], [episode_id])


if __name__ == "__main__":
    unittest.main()
