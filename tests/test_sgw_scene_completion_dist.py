"""Geometric contracts for prospective DIST scenes; no simulator success claims."""
import importlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import pytest

from experiments.workshops.spatial_grounding_v1.fixtures import FixtureCandidate, select_qualified_layouts
from experiments.workshops.spatial_grounding_v1.lat_workspace_capture import _rotate_wxyz


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / 'artifacts/workshops/spatial_grounding_v1/infrastructure/a40-20260922z-workspace.json'


def api():
    spec = importlib.util.find_spec('experiments.workshops.spatial_grounding_v1.scene_completion_dist')
    assert spec is not None, 'The DIST completion geometry generator is missing'
    return importlib.import_module(spec.name)


@pytest.fixture
def workspace():
    return json.loads(WORKSPACE.read_text())


def center_from_root(row, obj, workspace):
    offset = _rotate_wxyz(row['orientations'][obj], workspace['objects'][obj]['geometric_center_offset_root_local_xyz_m'])
    return [a + b for a, b in zip(row['roots'][obj], offset)]


@pytest.mark.parametrize('side', ['left', 'right'])
def test_short_transfers_are_neutral_in_three_dimensions_and_clear_both_anchors(workspace, side):
    row = api().design(side, 0, 0, workspace)
    cube, bowl, plate = [row['centers'][name] for name in ('rubiks_cube', 'bowl', 'plate')]
    assert abs(math.dist(cube, plate) - math.dist(cube, bowl)) <= .005
    assert (bowl[1] > plate[1]) == (side == 'left')
    for name in ('rubiks_cube', 'bowl', 'banana'):
        assert center_from_root(row, name, workspace) == pytest.approx(row['centers'][name], abs=1e-9)
    for sign, target in row['targets'].items():
        assert int(sign) * (math.dist(target, plate) - math.dist(target, bowl)) >= .03
        assert math.dist(cube, target) <= .18
        assert min(bowl[0], plate[0]) - target[0] >= .20
        assert math.dist(target, [0, 0, 0]) < .65
    assert row['anchor_object_names'] == ['bowl', 'plate']
    assert row['qualification_requirements']['reference_motion_limit_m'] == .005
    assert row['qualification_requirements']['action_cap'] == 450
    assert row['qualification_requirements']['reset_indices'] == [0, 1, 2]
    assert row['status'] == 'authored_unqualified'


@pytest.mark.parametrize('variant', ['short-transfer-v1', 'native-smoke-v1'])
@pytest.mark.parametrize('side', ['left', 'right'])
def test_supports_touch_table_and_hold_cube_at_neutral_and_both_goals(workspace, variant, side):
    row = api().design(side, 0, 0, workspace, variant=variant)
    specs = {spec['name']: spec for spec in row['supports']}
    table_top = workspace['objects']['table']['bbox_env_local_max_xyz_m'][2]
    for name in row['support_names']:
        support = specs[name]
        assert support['rigid_body'] and support['kinematic_body']
        assert support['center_m'][2] - support['size_m'][2] / 2 == pytest.approx(table_top)
    cube = workspace['objects']['rubiks_cube']
    half_height = cube['geometric_center_env_local_xyz_m'][2] - cube['bbox_env_local_min_xyz_m'][2]
    for name, center in (
        ('dist_neutral_cube_support', row['centers']['rubiks_cube']),
        ('dist_bowl_landing_support', row['targets']['1']),
        ('dist_plate_landing_support', row['targets']['-1']),
    ):
        support = specs[name]
        assert support['center_m'][:2] == pytest.approx(center[:2])
        assert support['center_m'][2] + support['size_m'][2] / 2 + half_height == pytest.approx(center[2])
    plate = specs['plate']
    assert plate['shape'] == 'cylinder' and plate['rigid_body']
    assert plate['radius_m'] == .11 and plate['thickness_m'] == .012
    assert plate['center_m'] == row['centers']['plate']
    assert set(row['object_names']) == {'rubiks_cube', 'bowl', 'banana', 'table', 'plate', *row['support_names']}
    assert row['goal_support_names'] == {'near_bowl': 'dist_bowl_landing_support', 'near_plate': 'dist_plate_landing_support'}


def test_campaign_covers_frozen_29_layout_balance_without_seed_replicas(workspace):
    module = api()
    for seed in (20260922, 20260923):
        rows = module.campaign(workspace, seed=seed, count=29)
        assert rows == module.campaign(workspace, seed=seed, count=29)
        assert len({row['design_id'] for row in rows}) == 29
        candidates = [FixtureCandidate.from_json({
            'candidate_id': row['design_id'], 'family': 'DIST', 'seed': seed,
            'asset_manifest_sha256': 'a' * 64, 'task_asset': 'not-yet-authored.usda',
            'object_poses': {name: {'position_m': row['centers'][name], 'quaternion_wxyz': [1, 0, 0, 0]}
                             for name in ('rubiks_cube', 'bowl', 'plate')},
            'metadata': {'bowl_side': row['side'],
                         'scoring_center_offsets_root_local_m': {name: [0, 0, 0] for name in ('rubiks_cube', 'bowl', 'plate')}},
        }) for row in rows]
        # This exercises allocation only. Every input is explicitly hypothetical,
        # not an accepted native fixture or qualification result.
        selected = select_qualified_layouts('DIST', seed, candidates, {c.candidate_id for c in candidates})
        side_by_id = {row['design_id']: row['side'] for row in rows}
        assert len(selected) == 29
        for stage, expected in (('D', 2), ('C', 12)):
            assert Counter(side_by_id[cid] for label, cid in selected.items() if label.startswith('DIST-' + stage)) == {'left': expected, 'right': expected}
        assert side_by_id[selected['DIST-P01']] == ('left', 'right')[seed % 2]
        for first, second in itertools.combinations(rows, 2):
            assert max(math.dist(first['centers'][n], second['centers'][n]) for n in ('rubiks_cube', 'bowl', 'plate')) > .006


def test_all_bounded_campaign_supports_fit_table_and_clear_banana(workspace):
    rows = api().campaign(workspace, count=100)
    assert len(rows) == 100
    table = workspace['objects']['table']
    banana = workspace['objects']['banana']
    for row in rows:
        specs = [spec for spec in row['supports'] if spec['name'] != 'plate']
        shift = [row['roots']['banana'][i] - banana['root_position_env_local_xyz_m'][i] for i in range(3)]
        banana_min = [banana['bbox_env_local_min_xyz_m'][i] + shift[i] for i in range(3)]
        banana_max = [banana['bbox_env_local_max_xyz_m'][i] + shift[i] for i in range(3)]
        for spec in specs:
            lo = [spec['center_m'][i] - spec['size_m'][i] / 2 for i in range(2)]
            hi = [spec['center_m'][i] + spec['size_m'][i] / 2 for i in range(2)]
            assert all(table['bbox_env_local_min_xyz_m'][i] <= lo[i] < hi[i] <= table['bbox_env_local_max_xyz_m'][i] for i in range(2))
            assert math.hypot(*[max(lo[i] - banana_max[i], banana_min[i] - hi[i], 0) for i in range(2)]) >= .02
        for first, second in itertools.combinations(specs, 2):
            assert any(abs(first['center_m'][i] - second['center_m'][i]) >= (first['size_m'][i] + second['size_m'][i]) / 2 - 1e-9 for i in range(2))


def test_source_provenance_preserves_historical_right_failure(workspace):
    module = api()
    for side, index, passed in (('left', '2', 6), ('right', '3', 5)):
        row = module.design(side, 0, 0, workspace, variant='native-smoke-v1')
        source = row['source_provenance']
        assert source['historical_passed_checks'] == passed
        assert source['evidence_directory'].endswith('/' + index + '/evidence')
        capture = json.loads((ROOT / source['evidence_directory'] / 'candidate_capture.json').read_text())
        for name in ('rubiks_cube', 'bowl', 'plate'):
            assert row['centers'][name] == pytest.approx(capture['objects'][name]['geometric_center_env_local_xyz_m'], abs=1e-6)
        assert row['status'] == 'authored_unqualified'


@pytest.mark.parametrize('side,dx,dy,variant', [('none', 0, 0, 'short-transfer-v1'), ('left', float('nan'), 0, 'short-transfer-v1'), ('left', .05, 0, 'short-transfer-v1'), ('left', 0, 0, 'unknown')])
def test_invalid_design_inputs_fail_before_authoring(workspace, side, dx, dy, variant):
    with pytest.raises(ValueError):
        api().design(side, dx, dy, workspace, variant=variant)


def test_campaign_rejects_unbounded_counts(workspace):
    for count in (0, 101):
        with pytest.raises(ValueError):
            api().campaign(workspace, count=count)
