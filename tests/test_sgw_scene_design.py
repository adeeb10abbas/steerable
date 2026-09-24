"""Scene geometry contracts: keep reference obstacles away from transfer lanes."""
import importlib
import json
import math
from pathlib import Path

import pytest

BASE = Path('artifacts/workshops/spatial_grounding_v1/infrastructure/a40-20260922z-workspace.json')


def api():
    spec = importlib.util.find_spec('experiments.workshops.spatial_grounding_v1.scene_design')
    assert spec is not None, 'The revised deterministic scene generator is missing'
    return importlib.import_module(spec.name)


@pytest.mark.parametrize('family,side', [('LAT','none'),('HEIGHT','left'),('HEIGHT','right')])
def test_neutral_scene_keeps_both_transfer_paths_clear_of_bowl(family, side):
    m=api(); w=json.loads(BASE.read_text()); d=m.design(family,side,0,0,w)
    c=d['centers']['rubiks_cube']; b=d['centers']['bowl']
    assert abs(c[1 if family=='LAT' else 2]-b[1 if family=='LAT' else 2]) <= .005
    for sign,t in d['targets'].items():
        axis=1 if family=='LAT' else 2
        assert int(sign)*(t[axis]-b[axis]) >= .05
        assert math.dist(c[:2],t[:2]) <= .20
        # Keep the reference on the far side of the entire swept transfer lane.
        assert b[0]-max(c[0],t[0]) >= .19
        assert .35 < math.dist(t,[0,0,0]) < .66
    if family=='HEIGHT':
        assert (d['targets']['1'][1] > d['targets']['-1'][1]) == (side=='left')
        assert all(s['size_m'][0]>=.14 and s['size_m'][1]>=.14 for s in d['supports'] if 'upper' in s['name'] or 'lower' in s['name'])


def test_seeded_campaign_has_balanced_distinct_layouts():
    m=api(); w=json.loads(BASE.read_text())
    rows=m.campaign('HEIGHT',w,seed=20260923,count=100)
    assert rows == m.campaign('HEIGHT',w,seed=20260923,count=100)
    assert len(rows)==100
    assert sum(r['side']=='left' for r in rows)==50
    # Every two scored cube roots differ by more than the 3mm reset tolerance,
    # except mirrored layouts which have different measured support geometry.
    for i,a in enumerate(rows):
        for b in rows[:i]:
            assert a['side']!=b['side'] or math.dist(a['centers']['rubiks_cube'],b['centers']['rubiks_cube']) > .006
    with pytest.raises(ValueError): m.campaign('LAT',w,seed=1,count=101)


def test_clean_scene_changes_appearance_without_changing_collision_or_poses(tmp_path):
    from pxr import Usd, UsdGeom
    m=api(); w=json.loads(BASE.read_text()); row=m.design('HEIGHT','left',0,0,w)
    base=tmp_path/'RoboLab/assets/scenes/rubiks_cube_banana_bowl.usda'; base.parent.mkdir(parents=True)
    base.write_text('''#usda 1.0
(defaultPrim = "World")
def Xform "World" {
 def Xform "table" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
  bool physics:kinematicEnabled = true
  double3 xformOp:translate = (0.2,0,0.05)
  uniform token[] xformOpOrder = ["xformOp:translate"]
  def Cube "surface" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
  double size = 0.7
  }
 }
 def Xform "GroundPlane" {
  token visibility = "invisible"
 }
}
''')
    row['visual_style']='clean-studio-v1'
    scene=m.write_scene(row,w,tmp_path/'RoboLab',tmp_path/'clean.usda')
    stage=Usd.Stage.Open(scene['path']); table=stage.GetPrimAtPath('/World/table')
    assert table.GetRelationship('material:binding').GetTargets(), 'Clean table must have a matte material'
    assert table.GetAttribute('physics:kinematicEnabled').Get() is True
    assert tuple(table.GetAttribute('xformOp:translate').Get()) == (.2,0,.05)
    assert stage.GetPrimAtPath('/World/table/surface').HasAPI(__import__('pxr.UsdPhysics',fromlist=['CollisionAPI']).CollisionAPI)
    assert UsdGeom.Imageable(stage.GetPrimAtPath('/World/GroundPlane')).ComputeVisibility() == 'inherited'
    assert not stage.GetPrimAtPath('/World/robot')
