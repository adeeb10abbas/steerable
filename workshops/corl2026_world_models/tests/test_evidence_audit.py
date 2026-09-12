import copy
from collections import Counter
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'analysis/evidence_audit.py'
spec = importlib.util.spec_from_file_location('evidence_audit', MODULE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

class AuditTests(unittest.TestCase):
    def test_missing_and_duplicate_cohort_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            audit.validate_ids(['a', 'a'], ['a', 'b'])
        with self.assertRaisesRegex(ValueError, 'missing'):
            audit.validate_ids(['a'], ['a', 'b'])
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            audit.validate_ids(['a', 'b', 'c'], ['a', 'b'])

    def test_temporal_pool_can_disagree_with_final_frame(self):
        frames = [{'frame_index': i, 'semantics': {'reliable': True, 'relation': r}}
                  for i, r in zip((8,16,24,32), ('left','left','left','right'))]
        self.assertIs(audit.legacy_label(frames, 'left'), True)
        self.assertIs(audit.endpoint_label(frames, 'left'), False)

    def test_abstention_is_not_negative(self):
        frames = [{'frame_index': i, 'semantics': {'reliable': i == 8, 'relation': 'left' if i == 8 else None}}
                  for i in (8,16,24,32)]
        self.assertIsNone(audit.legacy_label(frames, 'left'))
        self.assertIsNone(audit.endpoint_label(frames, 'left'))
        self.assertEqual(audit.quadrant(None, True), 'uncertain_future')

    def test_symmetric_planar_predicate_has_no_hidden_height_test(self):
        self.assertEqual(audit.planar_direction([0,1],[0,0]), 'left')
        self.assertEqual(audit.planar_direction([0,-1],[0,0]), 'right')
        self.assertEqual(audit.planar_direction([1,0],[0,0]), 'neutral')
        self.assertEqual(audit.planar_direction([0,0],[0,0]), 'neutral')
        with self.assertRaises(ValueError):
            audit.planar_direction([0,1,2],[0,0,0])

    def test_adapter_rejects_temporal_and_geometry_mismatches(self):
        record = {'id':'a', 'prediction':{'time_index':32,'coordinate_frame':'robot',
                  'geometry':'visual_centroid_xy','cube':[0,1],'bowl':[0,0], 'reliable':True},
                  'execution':{'time_index':31,'coordinate_frame':'robot',
                  'geometry':'visual_centroid_xy','cube':[0,1],'bowl':[0,0], 'reliable':True}, 'provenance':[]}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'time'):
                audit.validate_pair(record, Path(d))
            record['execution']['time_index'] = 32
            record['execution']['geometry'] = 'root_pose_xy'
            with self.assertRaisesRegex(ValueError, 'geometry'):
                audit.validate_pair(record, Path(d))
            record['execution']['geometry'] = 'visual_centroid_xy'
            with self.assertRaisesRegex(ValueError, 'provenance'):
                audit.validate_pair(record, Path(d))

    def test_constant_negative_baseline_uses_same_covered_subset(self):
        rows=[{'actual':False,'legacy':False},{'actual':True,'legacy':True},
              {'actual':True,'legacy':None}]
        stats=audit.metrics(rows,'legacy')['historical_label_arithmetic']
        self.assertEqual(stats['denominator'],2)
        self.assertEqual(stats['observed_agreement_numerator'],2)
        self.assertEqual(stats['constant_negative_agreement_numerator'],1)

    def test_adapter_verifies_hash_and_retains_abstention(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/'source.json').write_text('{}')
            item = {'time_index':32,'coordinate_frame':'robot','geometry':'visual_centroid_xy',
                    'cube':[0,1],'bowl':[0,0],'reliable':False}
            record = {'id':'a', 'prediction':item, 'execution':copy.deepcopy(item),
                      'provenance':[{'path':'source.json','sha256':hashlib.sha256(b'{}').hexdigest(),
                                     'role':'prediction'},{'path':'source.json','sha256':hashlib.sha256(b'{}').hexdigest(),'role':'execution'},
                                    {'path':'source.json','sha256':hashlib.sha256(b'{}').hexdigest(),'role':'alignment'}]}
            record['execution']['reliable'] = True
            audit.validate_pair(record, root)
            record['provenance'][0]['sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'hash'):
                audit.validate_pair(record, root)

    def valid_pair(self, root):
        (root/'source.json').write_text('{}')
        item = {'time_index':32,'coordinate_frame':'robot','geometry':'visual_centroid_xy',
                'cube':[0,1],'bowl':[0,0],'reliable':True}
        return {'prediction':copy.deepcopy(item),'execution':copy.deepcopy(item),
                'provenance':[{'path':'source.json','sha256':audit.sha256(root/'source.json'),
                               'role':role} for role in ('prediction','execution','alignment')]}

    def test_adapter_rejects_invalid_time_on_either_side(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for side in ('prediction','execution'):
                for value in (True, False, -1, 32.0, '32', None):
                    with self.subTest(side=side,value=value):
                        record=self.valid_pair(root)
                        # True==1 and 32.0==32: equality cannot validate either side.
                        other=1 if value is True else 0 if value is False else 32
                        record['prediction']['time_index']=other
                        record['execution']['time_index']=other
                        record[side]['time_index']=value
                        with self.assertRaisesRegex(ValueError,'time_index'):
                            audit.validate_pair(record,root)
            record=self.valid_pair(root)
            record['prediction']['time_index']=record['execution']['time_index']=0
            audit.validate_pair(record,root)

    def test_adapter_requires_boolean_reliability_on_both_sides(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for side in ('prediction','execution'):
                for value in (None, 0, 1, 'false', 'missing'):
                    with self.subTest(side=side,value=value):
                        record=self.valid_pair(root)
                        if value=='missing': del record[side]['reliable']
                        else: record[side]['reliable']=value
                        with self.assertRaisesRegex(ValueError,'reliability'):
                            audit.validate_pair(record,root)

    def test_adapter_rejects_unknown_execution_but_preserves_forecast_abstention(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); record=self.valid_pair(root)
            record['prediction']['reliable']=False
            audit.validate_pair(record,root)
            self.assertIs(record['prediction']['reliable'],False)
            record['execution']['reliable']=False
            with self.assertRaisesRegex(ValueError,'execution.*ineligible'):
                audit.validate_pair(record,root)

    def test_adapter_retains_unreadable_forecast_without_inventing_coordinates(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); record=self.valid_pair(root)
            record['prediction'].update(reliable=False,cube=None,bowl=None)
            try:
                audit.validate_pair(record,root)
            except (TypeError,ValueError) as error:
                self.fail(f'unreadable forecast must remain an abstention: {error}')
            self.assertIsNone(record['prediction']['cube'])
            self.assertIsNone(record['prediction']['bowl'])
            record['prediction']['reliable']=True
            with self.assertRaisesRegex(ValueError,'geometry'):
                audit.validate_pair(record,root)

    def seed_block_rows(self):
        rows=[]
        for tier,wordings in ((6100,('canonical','short')),(7200,('declarative','contrastive'))):
            for ep in range(2):
                for wi,wording in enumerate(wordings):
                    for direction in ('left','right'):
                        for chunk in range(2):
                            rows.append({'id':f'{tier}/{ep}/{wording}/{direction}/{chunk}',
                                         'wording':wording,'requested':direction,'episode':ep,
                                         'replan':chunk,'sampling_seed':(tier+ep)*1000+chunk,
                                         'actual':True,'legacy':True if wi==ep else None})
        return rows

    def test_bootstrap_preserves_joint_seed_blocks_and_every_chunk(self):
        rows=self.seed_block_rows(); original_metrics=audit.metrics; draws=[]
        def inspect_and_score(selected,key):
            draws.append(list(selected))
            return original_metrics(selected,key)
        with patch.object(audit,'metrics',side_effect=inspect_and_score):
            result=audit.cluster_intervals(rows,'legacy',repeats=40)
        self.assertEqual(result['intervals']['coverage'],[.5,.5])
        for selected in draws:
            counts=Counter(r['id'] for r in selected)
            for tier in (6100,7200):
                multiplicities=[]
                for ep in range(2):
                    block=[r['id'] for r in rows if r['sampling_seed']//1000==tier+ep]
                    self.assertEqual(len({counts[key] for key in block}),1)
                    multiplicities.append(counts[block[0]])
                self.assertEqual(sum(multiplicities),2)

    def test_bootstrap_rejects_unrecoverable_or_incomplete_seed_blocks(self):
        rows=self.seed_block_rows()
        rows[0]['sampling_seed']+=100
        with self.assertRaisesRegex(ValueError,'seed'):
            audit.cluster_intervals(rows,'legacy',repeats=2)
        rows=[r for r in self.seed_block_rows()
              if not (r['wording']=='short' and r['requested']=='right' and r['episode']==0)]
        with self.assertRaisesRegex(ValueError,'incomplete.*block'):
            audit.cluster_intervals(rows,'legacy',repeats=2)

if __name__ == '__main__': unittest.main()
