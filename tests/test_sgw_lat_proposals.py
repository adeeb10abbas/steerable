import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.workshops.spatial_grounding_v1.fixtures import FixtureCandidate, candidate_order
from experiments.workshops.spatial_grounding_v1.lat_candidate_generator import workspace_digest
from experiments.workshops.spatial_grounding_v1.lat_proposals import propose_lat_layouts


WORKSPACE = Path(__file__).parents[1] / "artifacts/workshops/spatial_grounding_v1/infrastructure/a40-20260922r-workspace.json"


def test_real_geometry_proposals_are_bounded_neutral_distinct_and_unqualified():
    workspace = json.loads(WORKSPACE.read_text())
    rows = propose_lat_layouts(workspace, seed=20260922)
    assert len(rows) == 100
    assert rows[:1] == propose_lat_layouts(workspace, seed=20260922, count=1)
    candidates = [FixtureCandidate.from_json(row) for row in rows]
    assert len(candidate_order(20260922, candidates)) == 100
    assert {row["metadata"]["geometric_screen_status"] for row in rows} == {"passed", "rejected"}
    for row, candidate in zip(rows, candidates, strict=True):
        assert row["metadata"]["status"] == "proposed_unqualified_requires_physical_validation"
        assert bool(row["metadata"]["geometric_rejection_reasons"]) == (row["metadata"]["geometric_screen_status"] == "rejected")
        assert abs(candidate.relation_m()) < 1e-8
        centers = candidate.scoring_poses()
        for sign, name in ((1, "positive"), (-1, "negative")):
            points = row["metadata"]["abs_ik_waypoints"][name]
            assert sum(point["hold_steps"] for point in points) == 450
            target = points[-1]["position_world_xyz_m"]
            assert sign * (target[1] - centers["bowl"].position_m[1]) > 0.12
            assert abs(target[0] - centers["bowl"].position_m[0]) > 0.12


def test_proposal_cli_runs_and_refuses_overwrite(tmp_path):
    output = tmp_path / "proposals.json"
    command = [sys.executable, "-m", "experiments.workshops.spatial_grounding_v1.lat_proposals",
               "--workspace-receipt", str(WORKSPACE), "--output", str(output), "--count", "1"]
    subprocess.run(command, check=True, capture_output=True)
    value = json.loads(output.read_text())
    assert len(value["candidates"]) == 1
    assert value["model_request_count"] == value["behavioral_episode_count"] == 0
    assert value["candidate_cap_includes_geometric_rejections"] is True
    assert value["qualification_order"] == [value["candidates"][0]["candidate_id"]]
    rerun = subprocess.run(command, capture_output=True, text=True)
    assert rerun.returncode != 0 and "refusing to overwrite" in rerun.stderr


def test_proposals_verify_receipt_and_transform_environment_origin():
    workspace = json.loads(WORKSPACE.read_text())
    with pytest.raises(ValueError, match="digest"):
        propose_lat_layouts({**workspace, "receipt_sha256": "bad"}, seed=1)
    for count in (0, 101, True):
        with pytest.raises(ValueError, match="1..100"):
            propose_lat_layouts(workspace, seed=1, count=count)
    original = propose_lat_layouts(workspace, seed=1, count=1)[0]
    workspace["environment_origin_world_xyz_m"] = [1.0, 2.0, 3.0]
    workspace["receipt_sha256"] = workspace_digest(workspace)
    moved = propose_lat_layouts(workspace, seed=1, count=1)[0]
    assert original["object_poses"] == moved["object_poses"]
    for name in ("positive", "negative"):
        before = original["metadata"]["abs_ik_waypoints"][name][0]["position_world_xyz_m"]
        after = moved["metadata"]["abs_ik_waypoints"][name][0]["position_world_xyz_m"]
        np.testing.assert_allclose(np.subtract(after, before), [1, 2, 3])
