#!/usr/bin/env python3
"""Fail-closed validator for the V3-E007 computation and compact result."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/zero_model_reachability_v3e007"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path)
    args = parser.parse_args()
    registration_path = BASE / "registration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    assert registration["status"] == "frozen_before_reachability_computation"
    assert registration["learned_model_request_count"] == 0
    assert registration["behavioral_episode_count"] == 0
    assert registration["gpu_required"] is False
    assert len(registration["layouts"]) == 14
    assert len({row["layout_id"] for row in registration["layouts"]}) == 14
    assert registration["ik"]["position_error_m_inclusive"] == 0.001
    assert registration["ik"]["orientation_error_deg_inclusive"] == 1.0
    for item in registration["source_bindings"]:
        path = ROOT / item["path"]
        assert path.is_file(), path
        assert path.stat().st_size == item["bytes"], path
        assert sha256(path) == item["sha256"], path

    if args.raw_root is None:
        print(json.dumps({"status": "valid_registration", "registration_sha256": sha256(registration_path)}, indent=2))
        return
    summary_path = args.raw_root / "workspace_summary.json"
    points_path = args.raw_root / "workspace_points.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["registration_sha256"] == sha256(registration_path)
    assert summary["learned_model_request_count"] == 0
    assert summary["behavioral_episode_count"] == 0
    assert summary["runtime"]["torch_imported"] is False
    assert summary["runtime"]["isaacsim_imported"] is False
    assert summary["runtime"]["isaaclab_imported"] is False
    assert len(summary["fk_validation"]) == 4
    assert all(row["passed"] for row in summary["fk_validation"])
    assert summary["raw_points"]["bytes"] == points_path.stat().st_size
    assert summary["raw_points"]["sha256"] == sha256(points_path)
    rows = [json.loads(line) for line in points_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == summary["point_count"]
    expected_layouts = {row["layout_id"] for row in registration["layouts"]}
    assert {row["layout_id"] for row in rows} == expected_layouts
    for layout in summary["layouts"]:
        groups = {side: [row for row in rows if row["layout_id"] == layout["layout_id"] and row["side"] == side] for side in ("left", "right")}
        assert len(groups["left"]) == len(groups["right"]) > 0
        for left, right in zip(groups["left"], groups["right"], strict=True):
            assert left["voxel_index"] == right["voxel_index"]
            assert left["relative_center_m"][0] == right["relative_center_m"][0]
            assert left["relative_center_m"][1] == -right["relative_center_m"][1]
            assert left["relative_center_m"][2] == right["relative_center_m"][2]
        for side in ("left", "right"):
            observed = sum(row["feasible"] for row in groups[side])
            assert observed == layout["sides"][side]["feasible_voxel_count"]
            assert layout["sides"][side]["feasible_volume_m3"] == observed * summary["voxel_volume_m3"]

    results_path = BASE / "results/results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        assert results["status"] == "complete"
        assert results["registration_sha256"] == sha256(registration_path)
        assert results["learned_model_request_count"] == 0
        assert results["behavioral_episode_count"] == 0
        manifest = json.loads((BASE / "results/evidence_manifest.json").read_text(encoding="utf-8"))
        for item in manifest["files"]:
            path = ROOT / item["path"]
            assert path.stat().st_size == item["bytes"]
            assert sha256(path) == item["sha256"]
    print(json.dumps({"status": "valid", "registration_sha256": sha256(registration_path), "points": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

