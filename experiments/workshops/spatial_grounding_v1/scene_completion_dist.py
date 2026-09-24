"""Prospective clean DIST rows for the existing calibrated scene runner.

The visible plate and all five pedestal types reuse prospective_family_scene.
``short-transfer-v1`` changes only authored scene geometry and support colors:
both 16 cm transfers remain in front of the two anchors. ``native-smoke-v1``
reconstructs the historical two-side geometry for comparison, including the
right-side failure. Neither recipe is qualified by this module.

Runner integration: pass ``supports`` to the existing scene writer (it includes
the plate); measure all ``object_names``; include every ``scored_object_names``
entry in candidate poses and center offsets. Derive goal_supports from measured
``goal_support_names`` surfaces plus measured cube half-height, with contact IDs
``rubiks_cube__<surface>``. Use ``support_names`` for geometry guards and copy the
measured bowl_side and banana pose. The existing family controller, 450-action
cap, scorer, three resets per goal, robot, cameras and static prompts are reused.

Fixed anchors means the existing 5 mm measured motion gate for BOTH bowl and
plate, not new kinematic pins: their inherited dynamic physics is retained.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Mapping

from .prospective_family_scene import (
    _banana_root_on_table, _bottom_z, _disc, _pedestal, _root_for_center,
    _validate_authored_banana_geometry,
)
from .lat_workspace_capture import _rotate_wxyz


CAMPAIGN = 'SGW-RTX-COMPLETION-20260923'
VARIANTS = ('short-transfer-v1', 'native-smoke-v1')
_SMOKE_ROOT = 'artifacts/workshops/spatial_grounding_v1/infrastructure/native-smoke-20260923br'
_SOURCES = {
    'left': {
        'evidence_directory': _SMOKE_ROOT + '/2/evidence',
        'candidate_capture_sha256': '063a3cf5ed19796505a058582db04aac7297e951817b986b299eecc8e03f97a7',
        'qualification_sha256': 'ec9df6517e7680fc6d090ed0221bd3ac5aa19857e77583f6b9a4db96356cf4ae',
        'historical_passed_checks': 6,
        'historical_status': 'accepted_model_blind_fixture_candidate',
        'translation_xy_m': [-.038745, -.031843],
    },
    'right': {
        'evidence_directory': _SMOKE_ROOT + '/3/evidence',
        'candidate_capture_sha256': 'bce9b92932088836903e20e5adb4c28a8306e7136daa7b9078b7fa23e56a6ccd',
        'qualification_sha256': 'd9ac67b9abbc42d50789bc0229dfaf41341fa1411d2b6be787ce86c208d957b3',
        'historical_passed_checks': 5,
        'historical_status': 'rejected_model_blind_fixture_candidate',
        'historical_failure': 'goal +1 reset 0: detached but off landing support at action 450',
        'translation_xy_m': [-.009882, -.014275],
    },
}


def design(side: str, dx: float, dy: float, workspace: Mapping[str, Any], *,
           variant: str = 'short-transfer-v1') -> dict[str, Any]:
    """Author one bounded, unqualified row; dx/dy are whole-layout offsets."""
    if side not in ('left', 'right') or variant not in VARIANTS:
        raise ValueError('Expected left/right DIST side and a supported variant')
    if not all(math.isfinite(v) and abs(v) <= .04 for v in (dx, dy)):
        raise ValueError('DIST translation must be finite and within 40 mm per axis')
    objects = workspace['objects']
    sign = 1 if side == 'left' else -1
    source = dict(_SOURCES[side])
    if variant == 'short-transfer-v1':
        cube = [.46 + dx, -.10 + dy, .12]
        bowl = [.68 + dx, cube[1] + sign * .20, .12]
        plate = [.68 + dx, cube[1] - sign * .20, .12]
        targets = {str(s): [cube[0], cube[1] + s * sign * .16, .12] for s in (1, -1)}
        neutral_width, landing_width = .10, .16
        source['reuse_scope'] = 'existing visible plate and pedestal implementation; newly authored short transfers'
    else:
        x, y = source['translation_xy_m']
        x, y = x + dx, y + dy
        cube, bowl, plate = [.55 + x, y, .12], [.44 + x, sign * .16 + y, .12], [.66 + x, -sign * .16 + y, .12]
        targets = {'1': [.30 + x, sign * .16 + y, .12], '-1': [.66 + x, -sign * .34 + y, .12]}
        neutral_width = landing_width = .08
        source['reuse_scope'] = 'historical geometry reconstructed to submicrometer rounding; clean support colors'
    source['geometry_module'] = 'experiments/workshops/spatial_grounding_v1/prospective_family_scene.py'
    source['qualification_inheritance'] = False
    table_top = objects['table']['bbox_env_local_max_xyz_m'][2]
    color = (.55, .60, .65)
    specs = [_disc('plate', tuple(plate), radius_m=.11, thickness_m=.012,
                   color=(.94, .94, .90), rigid=True)]
    for name, obj, center, width in (
        ('dist_bowl_support', 'bowl', bowl, .18),
        ('dist_neutral_cube_support', 'rubiks_cube', cube, neutral_width),
        ('dist_bowl_landing_support', 'rubiks_cube', targets['1'], landing_width),
        ('dist_plate_landing_support', 'rubiks_cube', targets['-1'], landing_width),
    ):
        bottom = _bottom_z(objects[obj], _root_for_center(objects[obj], center))
        specs.append(_pedestal(name, center[:2], bottom, (width, width), color, table_top))
    specs.append(_pedestal('dist_plate_support', plate[:2], plate[2] - .006,
                           (.24, .24), color, table_top))
    roots = {name: _root_for_center(objects[name], center) for name, center in (('rubiks_cube', cube), ('bowl', bowl))}
    roots['banana'] = _banana_root_on_table(workspace)
    orientations = {name: list(objects[name]['root_quaternion_world_wxyz']) for name in roots}
    banana_offset = _rotate_wxyz(orientations['banana'], objects['banana']['geometric_center_offset_root_local_xyz_m'])
    banana = [roots['banana'][i] + banana_offset[i] for i in range(3)]
    _validate_authored_banana_geometry(workspace, roots['banana'], specs)
    for spec in specs:
        if spec['name'] == 'plate':
            continue
        if any(spec['center_m'][i] - spec['size_m'][i] / 2 < objects['table']['bbox_env_local_min_xyz_m'][i]
               or spec['center_m'][i] + spec['size_m'][i] / 2 > objects['table']['bbox_env_local_max_xyz_m'][i]
               for i in range(2)):
            raise ValueError(f"DIST support leaves measured table: {spec['name']}")
    support_names = [spec['name'] for spec in specs if spec['name'] != 'plate']
    identity = hashlib.sha256(json.dumps([variant, side, dx, dy], separators=(',', ':')).encode()).hexdigest()[:12]
    return {
        'design_id': f'{CAMPAIGN}-DIST-{identity}', 'family': 'DIST', 'side': side,
        'variant': variant, 'visual_style': 'clean-studio-v1', 'translation_xy_m': [dx, dy],
        'status': 'authored_unqualified', 'model_requests': 0,
        'centers': {'rubiks_cube': cube, 'bowl': bowl, 'plate': plate, 'banana': banana},
        'targets': targets, 'roots': roots, 'orientations': orientations, 'supports': specs,
        'object_names': ['rubiks_cube', 'bowl', 'banana', 'table', *[spec['name'] for spec in specs]],
        'scored_object_names': ['rubiks_cube', 'bowl', 'plate'],
        'anchor_object_names': ['bowl', 'plate'], 'support_names': support_names,
        'goal_support_names': {'near_bowl': 'dist_bowl_landing_support', 'near_plate': 'dist_plate_landing_support'},
        'source_provenance': source,
        'plate_category_caveat': 'Existing simplified off-white disc; policy-sized visual recognition remains to be checked.',
        'qualification_requirements': {
            'initial_neutral_tolerance_m': .005, 'relation_margin_m': .03,
            'reference_motion_limit_m': .005, 'action_cap': 450,
            'goal_signs': [1, -1], 'reset_indices': [0, 1, 2],
            'released_supported_stable_cube': True,
        },
    }


def campaign(workspace: Mapping[str, Any], seed: int = 20260923, count: int = 100) -> list[dict[str, Any]]:
    """Bounded short-transfer candidates, with the seeded pilot side first.

Every same-side layout changes scored-object positions by at least 10 mm.
A 29-row prefix contains 15 of the pilot side and 14 of the other, allowing
the frozen selector to allocate P01, 2+2 development and 12+12 confirmation
only AFTER all selected candidates independently pass native qualification.
"""
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 100:
        raise ValueError('One DIST completion campaign has 1..100 candidate slots')
    grid = [(x / 1000, y / 1000) for x in range(-40, 41, 10) for y in range(-40, 41, 10)]
    random.Random(seed).shuffle(grid)
    rows = []
    for index in range(count):
        side = ('left', 'right')[(seed + index) % 2]
        row = design(side, *grid[index // 2], workspace)
        row.update(design_id=f'{CAMPAIGN}-DIST-{index:03d}', ordinal=index, seed=seed)
        rows.append(row)
    return rows
