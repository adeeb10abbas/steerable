"""LAT approach balance keeps goal semantics and real actor-center offsets."""
import copy
import importlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOTYPE = ROOT / 'artifacts/workshops/spatial_grounding_v1/scene_design_rtx_20260923/prototype-06.json'


def api():
    spec = importlib.util.find_spec('experiments.workshops.spatial_grounding_v1.scene_completion_lat')
    assert spec is not None, 'LAT approach-side completion authoring is missing'
    return importlib.import_module(spec.name)


def prototype():
    return json.loads(PROTOTYPE.read_text())


def test_reflection_preserves_real_cube_offset_and_positive_negative_goal_semantics():
    m = api()
    source = prototype()
    before = copy.deepcopy(source)
    reflected = m.reflect_lat_design(source)
    assert source == before
    assert reflected['centers']['rubiks_cube'] == pytest.approx(
        [0.4602469445911803, 0.17882823465527428, 0.07869696617126465])
    # Reflecting the root itself would move the center by nearly 58 mm because
    # orientations stay fixed. The root must follow the center displacement.
    assert reflected['roots']['rubiks_cube'] == pytest.approx(
        [0.4704165847746291, 0.14990558741749656, 0.08113233000040054])
    assert reflected['targets']['1'] == pytest.approx(
        [0.46024691974545473, 0.3083477567669411, 0.07869696617126465])
    assert reflected['targets']['-1'] == pytest.approx(
        [0.46024691974545473, 0.04930871894326405, 0.07869696617126465])
    assert reflected['orientations'] == source['orientations']
    assert reflected['supports'] == []
    assert reflected['side'] == 'none'  # Existing LAT runner convention.
    assert reflected['approach_side'] == 'left'
    assert reflected['status'] == 'authored_unqualified'
    provenance = reflected['approach_balance']
    assert provenance['source_design_id'] == 'SGW-RTX-CLEAN-PROTOTYPE-06'
    assert provenance['source_approach_side'] == 'right'
    assert provenance['target_source_sign'] == {'1': '-1', '-1': '1'}
    assert provenance['intervention'] == 'movable-object-center-position-reflection'
    assert reflected['model_requests'] == reflected['behavioral_episodes'] == 0
    assert len(provenance['source_design_sha256']) == 64


def test_every_object_retains_its_orientation_and_root_to_center_offset():
    m = api()
    source = prototype()
    reflected = m.reflect_lat_design(source)
    for name in ('rubiks_cube', 'bowl', 'banana'):
        center, moved = source['centers'][name], reflected['centers'][name]
        assert moved == pytest.approx([center[0], -center[1], center[2]])
        for axis in range(3):
            assert moved[axis] - reflected['roots'][name][axis] == pytest.approx(
                center[axis] - source['roots'][name][axis])


def test_completion_pool_has_exact_first_prototype_and_distinct_10mm_shifts():
    m = api()
    source = prototype()
    rows = m.make_lat_completion_pool(source, seed=20260924, count=24)
    assert rows == m.make_lat_completion_pool(source, seed=20260924, count=24)
    assert rows[:3] == m.make_lat_completion_pool(source, seed=20260924, count=3)
    assert len(rows) == len({r['design_id'] for r in rows}) == 24
    assert rows[0]['centers'] == m.reflect_lat_design(source)['centers']
    assert rows[0]['translation_xy_m'] == [0.0, 0.0]
    for index, row in enumerate(rows):
        assert m.classify_approach_side(row) == row['approach_side'] == 'left'
        assert row['centers']['rubiks_cube'][1] == row['centers']['bowl'][1]
        assert all(abs(v) <= .020001 and v * 100 == pytest.approx(round(v * 100))
                   for v in row['translation_xy_m'])
        for other in rows[:index]:
            assert math.dist(row['centers']['rubiks_cube'], other['centers']['rubiks_cube']) >= .009999
        for sign, target in row['targets'].items():
            assert int(sign) * (target[1] - row['centers']['bowl'][1]) > .12
        assert row['orientations'] == source['orientations']


def test_classification_uses_absolute_robot_side_not_goal_sign_and_reuses_right_pool():
    m = api()
    source = prototype()
    assert m.classify_approach_side(source) == 'right'
    mixed = copy.deepcopy(source)
    mixed['centers']['bowl'][1] = .18
    assert m.classify_approach_side(mixed) == 'mixed'
    centered = copy.deepcopy(source)
    centered['centers']['rubiks_cube'][1] = centered['centers']['bowl'][1] = 0.0
    assert m.classify_approach_side(centered) == 'centered'
    # Existing right-side pool designs remain reusable with no mutation.
    from experiments.workshops.spatial_grounding_v1.scene_design_batch import make_pool
    existing = make_pool('LAT', {}, source, 20260923)
    assert {m.classify_approach_side(row) for row in existing} == {'right'}


@pytest.mark.parametrize('count', [0, 26, True, 1.5])
def test_pool_is_finite(count):
    with pytest.raises(ValueError, match='1..25'):
        api().make_lat_completion_pool(prototype(), count=count)


def test_rejects_non_neutral_or_reversed_goal_sources():
    m = api()
    non_neutral = prototype()
    non_neutral['centers']['bowl'][1] += .02
    with pytest.raises(ValueError, match='neutral'):
        m.reflect_lat_design(non_neutral)
    wrong_goals = prototype()
    wrong_goals['targets']['1'], wrong_goals['targets']['-1'] = (
        wrong_goals['targets']['-1'], wrong_goals['targets']['1'])
    with pytest.raises(ValueError, match='goal sign'):
        m.reflect_lat_design(wrong_goals)
