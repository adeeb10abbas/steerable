"""Compact revised LAT/HEIGHT scene authoring, separate from frozen campaigns.

All values are design inputs. Physical outcomes come only from the native
runner using the existing controller, measurements and SGW scorer.
"""
from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path

from .prospective_family_scene import _root_for_center, _bottom_z, _pedestal, _usda

CAMPAIGN = 'SGW-RTX-SCENES-20260923'


def design(family, side, dx, dy, workspace):
    if family not in ('LAT', 'HEIGHT') or (family == 'HEIGHT' and side not in ('left', 'right')):
        raise ValueError('Expected LAT or counterbalanced HEIGHT')
    objects = workspace['objects']
    if family == 'LAT':
        cube = [.46 + dx, -.18 + dy, objects['rubiks_cube']['geometric_center_env_local_xyz_m'][2]]
        bowl = [.66 + dx, cube[1], objects['bowl']['geometric_center_env_local_xyz_m'][2]]
        targets = {str(s): [cube[0], cube[1] + s*.13, cube[2]] for s in (1,-1)}
        supports = []
    else:
        cube = [.46 + dx, -.10 + dy, .16]
        bowl = [.68 + dx, cube[1], .16]
        sign = 1 if side == 'left' else -1
        targets = {str(s): [cube[0], cube[1] + s*sign*.18, cube[2] + s*.055] for s in (1,-1)}
        top = objects['table']['bbox_env_local_max_xyz_m'][2]
        supports = []
        for name, obj, center, width in (
            ('height_reference_support', 'bowl', bowl, .18),
            ('height_neutral_cube_support', 'rubiks_cube', cube, .10),
            ('height_upper_support', 'rubiks_cube', targets['1'], .16),
            ('height_lower_support', 'rubiks_cube', targets['-1'], .16),
        ):
            # Equal neutral colors avoid adding a new color cue for higher/lower.
            supports.append(_pedestal(name, center[:2], _bottom_z(objects[obj], _root_for_center(objects[obj], center)),
                                     (width,width), (.55,.60,.65), top))
    banana = [.79, .36, objects['banana']['geometric_center_env_local_xyz_m'][2]]
    centers = {'rubiks_cube':cube, 'bowl':bowl, 'banana':banana}
    return {'family':family, 'side':side, 'visual_style':'clean-studio-v1',
            'translation_xy_m':[dx,dy], 'centers':centers,
            'targets':targets, 'supports':supports,
            'roots':{n:_root_for_center(objects[n],c) for n,c in centers.items()},
            'orientations':{n:objects[n]['root_quaternion_world_wxyz'] for n in centers}}


def campaign(family, workspace, seed=20260923, count=100):
    if not 1 <= count <= 100:
        raise ValueError('One revised campaign has at most 100 candidates per family')
    # 10mm lattice: independently distinct scored-object positions, not seed replicas.
    # Separate halves of the lattice for mirrored HEIGHT scenes prevent duplicate
    # cube/bowl configurations even when support-side labels are ignored.
    grid=[(x/1000,y/1000) for x in range(-40,41,10) for y in range(-60,61,10)]
    random.Random(seed).shuffle(grid)
    rows=[]
    for i,(x,y) in enumerate(grid[:count]):
        side=('left' if i%2==0 else 'right') if family=='HEIGHT' else 'none'
        row=design(family,side,x,y,workspace)
        row.update(design_id=f'{CAMPAIGN}-{family}-{i:03d}', ordinal=i, seed=seed,
                   status='authored_unqualified', model_requests=0)
        rows.append(row)
    return rows


def write_scene(row, workspace, robolab_root, output):
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    if row.get('visual_style') not in (None,'original-office','clean-studio-v1'):
        raise ValueError('Unknown scene appearance')
    base=Path(robolab_root)/'assets/scenes/rubiks_cube_banana_bowl.usda'
    output.parent.mkdir(parents=True,exist_ok=True)
    text=_usda(base.resolve(),row['supports'],row['roots'],row['orientations'])
    if row.get('visual_style') == 'clean-studio-v1':
        text = text.rstrip()[:-1] + clean_studio_materials().removeprefix('\nover "World" {')
    output.write_text(text)
    return {'path':str(output.resolve()), 'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'object_names':['rubiks_cube','bowl','banana','table',*[s['name'] for s in row['supports']]]}


def clean_studio_materials():
    """Appearance-only opinions: keep every transform and collision shape intact."""
    return '''
over "World" {
    def Scope "SGWMaterials" {
        def Material "MatteTable" {
            token outputs:surface.connect = </World/SGWMaterials/MatteTable/Surface.outputs:surface>
            def Shader "Surface" {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.24, 0.28, 0.32)
                float inputs:roughness = 0.9
                float inputs:metallic = 0
                token outputs:surface
            }
        }
        def Material "MatteFloor" {
            token outputs:surface.connect = </World/SGWMaterials/MatteFloor/Surface.outputs:surface>
            def Shader "Surface" {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.62, 0.64, 0.67)
                float inputs:roughness = 1
                float inputs:metallic = 0
                token outputs:surface
            }
        }
    }
    over "table" (prepend apiSchemas = ["MaterialBindingAPI"]) {
        rel material:binding = </World/SGWMaterials/MatteTable> (
            bindMaterialAs = "strongerThanDescendants"
        )
    }
    over "GroundPlane" (prepend apiSchemas = ["MaterialBindingAPI"]) {
        token visibility = "inherited"
        rel material:binding = </World/SGWMaterials/MatteFloor> (
            bindMaterialAs = "strongerThanDescendants"
        )
    }
}
'''
