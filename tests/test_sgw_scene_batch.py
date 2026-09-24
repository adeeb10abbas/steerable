import json
from pathlib import Path
import pytest


def test_prospective_pool_is_clean_distinct_balanced_and_excludes_reused_pilot():
    from experiments.workshops.spatial_grounding_v1.scene_design_batch import make_pool, enough
    base=Path('artifacts/workshops/spatial_grounding_v1')
    w=json.loads((base/'infrastructure/a40-20260922z-workspace.json').read_text())
    template=json.loads((base/'scene_design_rtx_20260923/prototype-06.json').read_text())
    a=make_pool('LAT',w,template,20260924)
    h=make_pool('HEIGHT',w,template,20260924)
    assert a==make_pool('LAT',w,template,20260924)
    assert len(a)==len(h)==100
    assert all(r['visual_style']=='clean-studio-v1' for r in a+h)
    assert len({tuple(r['roots']['rubiks_cube']) for r in a})==100
    assert all(max(abs(r['roots']['rubiks_cube'][k]-template['roots']['rubiks_cube'][k]) for k in (0,1))>=.0099 for r in a)
    assert sum(r['side']=='left' for r in h)==50
    assert not enough('LAT',[{'side':'none'}]*28,20260924)
    assert enough('LAT',[{'side':'none'}]*29,20260924)
    assert not enough('HEIGHT',[{'side':'left'}]*14+[{'side':'right'}]*14,20260924)
    assert enough('HEIGHT',[{'side':'left'}]*15+[{'side':'right'}]*14,20260924)


def test_assignments_have_one_pilot_two_per_side_development_twelve_confirmation():
    from experiments.workshops.spatial_grounding_v1.scene_design_batch import assign
    rows=[{'design_id':str(i),'side':'left' if i%2==0 else 'right'} for i in range(30)]
    result=assign('HEIGHT',rows,20260924)
    lookup={r['design_id']:r['side'] for r in rows}
    assert len(result)==len(set(result.values()))==29
    assert lookup[result['HEIGHT-P01']]=='left'
    assert sum(lookup[v]=='left' for k,v in result.items() if '-D' in k)==2
    assert sum(lookup[v]=='left' for k,v in result.items() if '-C' in k)==12
    with pytest.raises(ValueError): assign('LAT',rows[:28],20260924)
