"""Static tests for the DreamZero V3-B003 runtime contract."""

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.v3.dreamzero_droid.future_retention import (
    FUTURE_SCHEMA,
    identify_and_partition_session,
    partition_session,
)
from experiments.v3.dreamzero_phase_b.compile_cell import (
    _normalize_frozen_failure_stage,
)
from experiments.v3.dreamzero_phase_b.contract import load_cells


ROOT = Path(__file__).resolve().parents[1]


def test_registered_cells_are_complete_matched_seed_blocks() -> None:
    cells = load_cells(ROOT)
    assert len(cells) == 108
    assert {cell.seed for cell in cells} == set(range(9400, 9427))
    for seed in range(9400, 9427):
        block = [cell for cell in cells if cell.seed == seed]
        assert len(block) == 4
        assert {(cell.arm, cell.relation) for cell in block} == {
            ("control", "left"),
            ("control", "right"),
            ("position_mirrored", "left"),
            ("position_mirrored", "right"),
        }
        assert {cell.row["execution_order_index_within_seed"] for cell in block} == {
            1, 2, 3, 4
        }


def test_registered_noise_seed_is_constant_while_pair_labels_match() -> None:
    for cell in load_cells(ROOT):
        assert cell.row["effective_model_noise_seed"] == 1140
        assert cell.row["registered_sampling_seed_label"] == cell.seed


def test_initial_reset_sample_does_not_mark_behavior_as_executed() -> None:
    source = (
        ROOT / "experiments" / "v3" / "dreamzero_phase_b" / "robolab_bridge.py"
    ).read_text()
    assert "def _sample(self, action_step: int, *, persist: bool = True)" in source
    assert "self._sample(0, persist=False)" in source


def test_queue_restores_the_validated_isaac_and_glvnd_library_path() -> None:
    source = (
        ROOT / "experiments" / "v3" / "dreamzero_phase_b" / "queue.py"
    ).read_text()
    assert "/data/users/ali/glvnd/lib" in source
    assert '"LD_LIBRARY_PATH": FROZEN_LD_LIBRARY_PATH' in source


def test_queue_can_stop_after_one_new_cell_for_concurrency_proof() -> None:
    source = (
        ROOT / "experiments" / "v3" / "dreamzero_phase_b" / "queue.py"
    ).read_text()
    assert 'parser.add_argument("--max-new-cells", type=int)' in source
    assert "new_cells >= args.max_new_cells" in source


def test_phase_b_bridge_keeps_legacy_stage_separate_from_v3_taxonomy() -> None:
    source = (
        ROOT / "experiments" / "v3" / "dreamzero_phase_b" / "robolab_bridge.py"
    ).read_text()
    assert 'failure_stage = "success"' in source
    assert 'failure_category = "correct"' in source
    assert '"frozen_failure_stage": failure_stage' in source
    assert '"failure_taxonomy": failure_category' in source


def test_compiler_recovers_known_taxonomy_stage_mixup_from_retained_samples() -> None:
    capture = {
        "frozen_failure_stage": "correct",
        "requested_success": True,
        "requested_relation": "right",
        "samples": [
            {"object_xyz": [0.3, -0.2, 0.1], "reference_xyz": [0.4, 0.0, 0.1]}
        ],
    }
    normalized = _normalize_frozen_failure_stage(capture)
    assert normalized["frozen_failure_stage"] == "success"
    assert capture["frozen_failure_stage"] == "correct"


def test_compiler_preserves_already_valid_legacy_stage() -> None:
    capture = {"frozen_failure_stage": "no_object_interaction"}
    assert _normalize_frozen_failure_stage(capture) is capture


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def test_concurrent_future_manifests_partition_by_exact_session(tmp_path: Path) -> None:
    future_root = tmp_path / "futures"
    prompt_a = "Put the Rubik's cube to the left of the bowl."
    prompt_b = "Put the Rubik's cube to the right of the bowl."
    chunks_a = np.arange(2 * 24 * 8, dtype=np.float32).reshape(2, 24, 8)
    chunks_b = np.full((1, 24, 8), 7.0, dtype=np.float32)

    def request(episode: int, index: int, session: str, prompt: str, chunk: np.ndarray):
        directory = future_root / f"episode_{episode:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        action = directory / f"request_{index:04d}_official_action.npy"
        latent = directory / f"request_{index:04d}_latent.pt"
        np.save(action, chunk, allow_pickle=False)
        latent.write_bytes(f"latent-{episode}-{index}".encode())
        return {
            "session_id": session,
            "prompt": prompt,
            "action_cfg_style_scale": 2.0,
            "returned_action": _record(action),
            "latent_video": _record(latent),
        }

    def manifest(episode: int, requests: list[dict[str, object]], reset_session: str):
        directory = future_root / f"episode_{episode:03d}"
        decode = future_root / "decoded" / f"{episode:03d}.mp4"
        decode.parent.mkdir(parents=True, exist_ok=True)
        decode.write_bytes(f"decode-{reset_session}".encode())
        value = {
            "schema_version": FUTURE_SCHEMA,
            "amendment_id": "V2-A015",
            "episode_index": episode,
            "action_cfg_style_scale": 2.0,
            "video_cfg_scale": 5.0,
            "requests": requests,
            "request_count": len(requests),
            "reset_info": {"session_ids": [reset_session]},
            "official_reset_decode": [_record(decode)],
        }
        (directory / "future_manifest.json").write_text(json.dumps(value))

    # Session A is interleaved across both global batches. Session B closes the
    # first batch; session A closes the second.
    manifest(0, [
        request(0, 0, "session-a", prompt_a, chunks_a[0]),
        request(0, 1, "session-b", prompt_b, chunks_b[0]),
    ], "session-b")
    manifest(1, [request(1, 0, "session-a", prompt_a, chunks_a[1])], "session-a")

    selected = partition_session(
        future_root,
        session_id="session-a",
        prompt=prompt_a,
        returned_raw_chunks=chunks_a,
    )
    assert selected["session_id"] == "session-a"
    assert selected["request_count"] == 2
    assert [row["session_id"] for row in selected["requests"]] == [
        "session-a", "session-a"
    ]
    assert selected["reset_info"] == {"session_ids": ["session-a"]}
    assert len(
        selected["concurrent_session_partition"]["source_future_manifests"]
    ) == 2
    assert identify_and_partition_session(
        future_root,
        prompt=prompt_a,
        returned_raw_chunks=chunks_a,
    )["session_id"] == "session-a"
