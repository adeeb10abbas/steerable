import importlib.util
import math
from pathlib import Path
import unittest

MODULE=Path(__file__).resolve().parents[1]/'analysis/prepare_annotation_sample.py'
spec=importlib.util.spec_from_file_location('annotation_sample',MODULE)
sample=importlib.util.module_from_spec(spec); spec.loader.exec_module(sample)


def population():
    rows=[]
    definitions=[(22,True,True,'imagines_requested_executes_requested'),
                 (5,True,False,'imagines_requested_executes_not_requested'),
                 (3,False,True,'does_not_imagine_requested_executes_requested'),
                 (391,False,False,'neither_imagines_nor_executes_requested'),
                 (72,None,True,'uncertain_future'),(259,None,False,'uncertain_future')]
    for group,(n,pred,actual,quadrant) in enumerate(definitions):
        for i in range(n):
            rows.append({'id':f'g{group}/chunk_{i:03d}','legacy':pred,'actual':actual,
                         'legacy_quadrant':quadrant,'wording':'synthetic','requested':'left',
                         'episode':i//15,'replan':i%15,'sampling_seed':100+i,
                         'chunk_dir':f'/synthetic/g{group}/chunk_{i:03d}'})
    return rows

class SamplingTests(unittest.TestCase):
    def test_exact_counts_unique_and_population_weight(self):
        records=sample.draw(population())
        self.assertEqual(len(records),160)
        self.assertEqual(len({r['source_id'] for r in records}),160)
        self.assertEqual([sum(r['stratum']==s for r in records) for s in sample.STRATA], [22,5,3,50,40,40])
        self.assertEqual(math.fsum(r['population_weight'] for r in records),752)
        for row in records:
            self.assertAlmostEqual(row['inclusion_probability'],row['n_h']/row['N_h'])
            self.assertAlmostEqual(row['population_weight'],row['N_h']/row['n_h'])

    def test_reproducibility_and_input_order_invariance(self):
        rows=population()
        self.assertEqual(sample.draw(rows), sample.draw(list(reversed(rows))))
        self.assertEqual(sample.draw(rows,20260912), sample.draw(rows,20260912))
        self.assertNotEqual(sample.draw(rows,20260912), sample.draw(rows,20260913))

    def test_missing_duplicate_and_malformed_strata_fail_closed(self):
        rows=population()
        with self.assertRaisesRegex(ValueError,'count'): sample.draw(rows[:-1])
        with self.assertRaisesRegex(ValueError,'duplicate'): sample.draw(rows+[rows[0]])
        broken=[dict(r) for r in rows]; broken[0]['actual']='True'
        with self.assertRaisesRegex(ValueError,'boolean'): sample.draw(broken)
        broken=[dict(r) for r in rows]; broken[0]['legacy_quadrant']='uncertain_future'
        with self.assertRaisesRegex(ValueError,'stratum'): sample.draw(broken)
        broken=[dict(r) for r in rows]; broken[0]['legacy']=1
        with self.assertRaisesRegex(ValueError,'boolean'): sample.draw(broken)

    def test_census_strata_have_probability_one(self):
        records=sample.draw(population())
        census=[r for r in records if r['stratum'] in list(sample.STRATA)[:3]]
        self.assertEqual(len(census),30)
        self.assertTrue(all(r['inclusion_probability']==1 and r['population_weight']==1 for r in census))

if __name__=='__main__': unittest.main()
