from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.compile_v3c_phase_c_results import MODELS, PROMPT_FAMILIES, SEEDS, canonical_json_bytes, summarize
from tools.render_v3c_phase_c_results import render


def test_renderer_emits_both_scientific_diagnostics(tmp_path: Path) -> None:
    left_actions = tmp_path / "left.npy"
    right_actions = tmp_path / "right.npy"
    np.save(left_actions, np.zeros((10, 8), dtype=np.float32))
    np.save(right_actions, np.ones((10, 8), dtype=np.float32))
    summaries = []
    episodes = []
    for model_index, model_id in enumerate(MODELS):
        rows = []
        for family_index, family in enumerate(PROMPT_FAMILIES):
            for seed in SEEDS:
                for relation in ("left", "right"):
                    success = (seed + family_index + model_index + (relation == "right")) % 3 != 0
                    rows.append({
                        "model_id": model_id,
                        "seed": seed,
                        "prompt_family": family,
                        "relation": relation,
                        "requested_success": success,
                        "failure_taxonomy": "correct" if success else "transport_failed",
                        "model_request_count": 1 if "cosmos" in model_id else None,
                        "measurements": {
                            "signed_final_lateral_offset_m": (
                                0.04 + 0.002 * (seed - 8500)
                                if relation == "left"
                                else -0.06 + 0.002 * (seed - 8500)
                            ),
                        },
                        "artifacts": {
                            "executed_actions": {
                                "path": str(left_actions if relation == "left" else right_actions),
                            },
                        },
                    })
        summary_path = tmp_path / f"{model_id}_summary.json"
        summary_path.write_bytes(canonical_json_bytes(summarize(rows, model_id=model_id)))
        episode_path = tmp_path / f"{model_id}_episodes.jsonl"
        episode_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        summaries.append(summary_path)
        episodes.append(episode_path)
    output = tmp_path / "figures"
    manifest = render(summaries=summaries, episodes=episodes, output_dir=output)
    assert len(manifest["outputs"]) == 4
    for record in manifest["outputs"]:
        assert Path(record["path"]).stat().st_size == record["bytes"]
        assert record["bytes"] > 1000
    assert (output / "phase_c_figure_manifest.json").is_file()
