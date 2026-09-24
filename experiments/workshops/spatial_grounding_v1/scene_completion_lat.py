"""Bounded LAT approach-side completion using the clean historical geometry.

The intervention reflects movable-object *centers* across robot y=0 and keeps
their orientations fixed. This is not a full-scene mirror: the robot, cameras,
controller, static prompts, scoring, and table remain unchanged. Every new
design still requires the existing two-goal, three-reset physical qualification.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random


def classify_approach_side(row):
    """Classify both starting scored-object centers in the identity robot frame.

    +y is robot-left. This is independent of requested goal sign, which remains
    the cube-minus-bowl relation. Historical right-side rows need no rewriting.
    """
    values = [row['centers'][name][1] for name in ('rubiks_cube', 'bowl')]
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Approach-side centers must be finite')
    if all(value > 0 for value in values):
        return 'left'
    if all(value < 0 for value in values):
        return 'right'
    if all(value == 0 for value in values):
        return 'centered'
    return 'mixed'


def reflect_lat_design(source, *, design_id=None):
    """Return an unqualified center reflection with correct LEFT/RIGHT goals.

    Actor roots follow each center displacement, rather than being reflected
    directly: unchanged orientations require unchanged root-to-center offsets.
    The banana's asymmetric mesh is not geometrically mirrored.
    """
    if source.get('family') != 'LAT' or source.get('supports') != []:
        raise ValueError('Expected a LAT design without support geometry')
    if not math.isclose(source['centers']['rubiks_cube'][1], source['centers']['bowl'][1], abs_tol=1e-8):
        raise ValueError('LAT completion requires neutral initial lateral geometry')
    side = classify_approach_side(source)
    if side not in ('left', 'right'):
        raise ValueError('LAT completion requires both starting objects on one robot side')
    for sign in ('1', '-1'):
        if int(sign) * (source['targets'][sign][1] - source['centers']['bowl'][1]) <= 0:
            raise ValueError('LAT target violates the robot-relative goal sign')

    row = copy.deepcopy(source)
    for name, center in source['centers'].items():
        row['centers'][name][1] = -center[1]
        row['roots'][name][1] = source['roots'][name][1] - 2 * center[1]
    # Reflection reverses the physical lateral relation, so exchange goal labels.
    row['targets'] = {
        sign: [point[0], -point[1], point[2]]
        for sign, point in ((str(-int(key)), value) for key, value in source['targets'].items())
    }
    row.update(
        design_id=design_id or f"{source['design_id']}-APPROACH-REFLECTED",
        geometry_revision='historical-LAT-center-reflection-fixed-orientations',
        side='none', approach_side=classify_approach_side(row),
        translation_xy_m=[0.0, 0.0], status='authored_unqualified',
        model_requests=0, behavioral_episodes=0,
    )
    row['approach_balance'] = {
        'source_design_id': source['design_id'],
        'source_design_sha256': hashlib.sha256(
            (json.dumps(source, sort_keys=True, separators=(',', ':')) + '\n').encode()
        ).hexdigest(),
        'source_digest_encoding': 'sorted compact JSON with trailing newline',
        'source_approach_side': side,
        'intervention': 'movable-object-center-position-reflection',
        'reflection_plane': 'robot y=0; robot root identity at world origin',
        'target_source_sign': {'1': '-1', '-1': '1'},
        'orientations': 'unchanged; roots translated by center displacement',
        'qualification_required': 'both goal signs, three fresh resets each, 450 actions per trial',
        'claim_boundary': 'Design inputs only; no physical success or learned-policy evidence.',
    }
    return row


def make_lat_completion_pool(lat_template, *, seed=20260924, count=24):
    """Exact reflected prototype first, followed by a finite 10 mm local lattice.

    At most 25 candidates cover +/-20 mm in x/y. These robot-left proposals
    complement already qualified robot-right layouts, which can be reused.
    Candidate order is fixed before physical outcomes and prefix-stable.
    """
    if type(count) is not int or not 1 <= count <= 25:
        raise ValueError('LAT completion pool count must be 1..25')
    if lat_template.get('design_id') != 'SGW-RTX-CLEAN-PROTOTYPE-06':
        raise ValueError('LAT completion pool requires historical clean prototype-06')
    if classify_approach_side(lat_template) != 'right':
        raise ValueError('Historical LAT completion template must be robot-right')
    reflected = reflect_lat_design(lat_template)
    shifts = [(x / 1000, y / 1000) for x in range(-20, 21, 10)
              for y in range(-20, 21, 10) if (x, y) != (0, 0)]
    random.Random(seed).shuffle(shifts)
    rows = []
    for index, (dx, dy) in enumerate(([(0.0, 0.0)] + shifts)[:count]):
        row = copy.deepcopy(reflected)
        for group in ('centers', 'roots', 'targets'):
            for position in row[group].values():
                position[0] += dx
                position[1] += dy
        row.update(design_id=f'SGW-COMPLETE-{seed}-LAT-LEFT-{index:03d}',
                   ordinal=index, seed=seed, translation_xy_m=[dx, dy])
        rows.append(row)
    return rows
