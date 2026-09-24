"""Small measured-geometry contracts shared by the three-family scene runner."""
from __future__ import annotations


def scored_names(family):
    if family not in ('LAT', 'HEIGHT', 'DIST'):
        raise ValueError(f'Unknown family: {family}')
    return ('rubiks_cube', 'bowl', 'plate') if family == 'DIST' else ('rubiks_cube', 'bowl')


def restore_scored_order(value):
    names = scored_names(value['family'])
    missing = set(names) - set(value['object_poses'])
    if missing:
        raise ValueError(f'Missing scored object(s): {sorted(missing)}')
    return {**value, 'object_poses': {name: value['object_poses'][name] for name in names}}


def measured_goal_supports(family, objects):
    mappings = {
        'HEIGHT': {'higher': 'height_upper_support', 'lower': 'height_lower_support'},
        'DIST': {'near_bowl': 'dist_bowl_landing_support', 'near_plate': 'dist_plate_landing_support'},
    }
    if family not in mappings:
        raise ValueError('Measured raised supports apply to HEIGHT/DIST')
    cube = objects['rubiks_cube']
    half = cube['geometric_center_env_local_xyz_m'][2] - cube['bbox_env_local_min_xyz_m'][2]
    return {
        key: {'contact_sensor_id': f'rubiks_cube__{name}',
              'cube_center_env_local_xyz_m': [*objects[name]['geometric_center_env_local_xyz_m'][:2],
                                              objects[name]['bbox_env_local_max_xyz_m'][2] + half]}
        for key, name in mappings[family].items()
    }


def measured_side(family, objects):
    def y(name): return objects[name]['geometric_center_env_local_xyz_m'][1]
    if family == 'LAT':
        delta = y('rubiks_cube')  # fixed native robot root at world y=0, identity orientation
    elif family == 'HEIGHT':
        delta = y('height_upper_support') - y('height_lower_support')
    elif family == 'DIST':
        delta = y('bowl') - y('plate')
    else:
        raise ValueError(f'Unknown family: {family}')
    if abs(delta) < .005:
        raise ValueError('Counterbalance side is ambiguous within 5mm')
    return 'left' if delta > 0 else 'right'
