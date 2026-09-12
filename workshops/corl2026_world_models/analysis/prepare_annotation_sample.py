#!/usr/bin/env python3
"""Private historical-stratum probability draw; no assets, alignment or labels verified."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys

STATUS='PROVISIONAL PRE-ANNOTATION DESIGN'
# Ordered exactly as the retrospective protocol table. N_h is the immutable
# historical population size, not a claim about independently validated truth.
STRATA={
    'both_positive':(22,22),
    'future_only_positive':(5,5),
    'execution_only_positive':(3,3),
    'both_negative':(391,50),
    'forecast_abstention_execution_positive':(72,40),
    'forecast_abstention_execution_negative':(259,40),
}
QUADRANTS={
    (True,True):'imagines_requested_executes_requested',
    (True,False):'imagines_requested_executes_not_requested',
    (False,True):'does_not_imagine_requested_executes_requested',
    (False,False):'neither_imagines_nor_executes_requested',
    (None,True):'uncertain_future',
    (None,False):'uncertain_future',
}
STRATUM_BY_LABEL={key:stratum for key,stratum in zip(QUADRANTS,STRATA)}

def digest(data):
    return hashlib.sha256(data).hexdigest()

def canonical_bytes(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')

def historical_stratum(row):
    prediction,execution=row['legacy'],row['actual']
    if (prediction is not None and type(prediction) is not bool) or type(execution) is not bool:
        raise ValueError('historical labels must be boolean or prediction null')
    key=(prediction,execution)
    if row['legacy_quadrant']!=QUADRANTS[key]:
        raise ValueError('malformed historical stratum: quadrant and labels disagree')
    return STRATUM_BY_LABEL[key]

def draw(rows,seed=20260912):
    buckets={s:[] for s in STRATA}; ids=[]
    for row in rows:
        required={'id','legacy','actual','legacy_quadrant','wording','requested','episode','replan','sampling_seed','chunk_dir'}
        if not required <= row.keys(): raise ValueError('missing required population fields')
        if not isinstance(row['id'],str) or not row['id']: raise ValueError('invalid population ID')
        ids.append(row['id']); buckets[historical_stratum(row)].append(row)
    if len(ids)!=len(set(ids)): raise ValueError('duplicate population ID')
    for stratum,(N_h,n_h) in STRATA.items():
        if len(buckets[stratum])!=N_h: raise ValueError(f'population count mismatch: {stratum}: {len(buckets[stratum])} != {N_h}')
    rng=random.Random(seed); selected=[]
    for stratum,(N_h,n_h) in STRATA.items():
        population=sorted(buckets[stratum],key=lambda r:r['id'])
        chosen=sorted(rng.sample(population,n_h),key=lambda r:r['id'])
        for row in chosen:
            selected.append({'source_id':row['id'],'stratum':stratum,'N_h':N_h,'n_h':n_h,
                'inclusion_probability':n_h/N_h,'population_weight':N_h/n_h,
                'inclusion_probability_exact':f'{n_h}/{N_h}', 'population_weight_exact':f'{N_h}/{n_h}',
                'wording':row['wording'],'requested_relation':row['requested'],'episode_index':row['episode'],
                'replan_index':row['replan'],'sampling_seed':row['sampling_seed'],
                'source_chunk_dir':row['chunk_dir'],'status':STATUS,
                'alignment_status':'pending_recovery_and_verification',
                'media_status':'pending_recovery_and_verification', 'human_label_status':'not_collected'})
    if len(selected)!=160 or len({r['source_id'] for r in selected})!=160:
        raise ValueError('invalid sample size or uniqueness')
    if math.fsum(r['population_weight'] for r in selected)!=752:
        raise ValueError('population weights do not sum to 752')
    return selected

def main():
    directory=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--population',type=Path,default=directory/'results/reconciled_chunks.jsonl')
    parser.add_argument('--protocol',type=Path,default=directory/'docs/ANALYSIS_PROTOCOL.md')
    parser.add_argument('--output',type=Path,default=directory/'results')
    args=parser.parse_args()
    raw=args.population.read_bytes()
    rows=[json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    selected=draw(rows)
    csv_stream=io.StringIO(newline=''); writer=csv.DictWriter(csv_stream,fieldnames=list(selected[0]),lineterminator='\n')
    writer.writeheader(); writer.writerows(selected); csv_bytes=csv_stream.getvalue().encode('utf-8')
    document={
        'status':STATUS,'visibility':'PRIVATE SAMPLING MANIFEST; NEVER DISTRIBUTE TO RATERS',
        'interpretation':'Historical strata determine sampling only. They are not human truth. No images, pairing, alignment or annotations have been verified.',
        'sampling_design':{'seed':20260912,'algorithm':'Python random.Random(seed).sample without replacement; six strata in declared order; population and selected rows sorted by canonical source ID within stratum',
            'python_version':sys.version.split()[0], 'population_size':752,'sample_size':160,
            'strata':[{ 'stratum':s,'N_h':N,'n_h':n,'inclusion_probability':n/N,'population_weight':N/n} for s,(N,n) in STRATA.items()],
            'population_weight_sum':math.fsum(r['population_weight'] for r in selected),
            'weights_scope':'All sampled chunks, before any recovery/annotation nonresponse; never transfer missing-case weights to replacements silently.'},
        'provenance':{'full_population_file':args.population.name,'full_population_file_sha256':digest(raw),
            'canonical_full_population_sha256':digest(canonical_bytes(sorted(rows,key=lambda r:r['id']))),
            'sampling_script_sha256':digest(Path(__file__).read_bytes()),
            'protocol_file_sha256':digest(args.protocol.read_bytes()),
            'draw_sha256':digest(canonical_bytes(selected)),
            'draw_hash_definition':'SHA256 of UTF-8 JSON sample array with sorted keys, comma/colon separators, no final newline; see canonical_bytes',
            'csv_sha256':digest(csv_bytes)},
        'release_gates':['Recover original media and verify hashes','Establish endpoint alignment and documented entity/camera identity',
                         'Freeze rubric, scorer configuration, private draw and rendering/blinding plan before new labels',
                         'Create isolated images with randomized opaque IDs and a separate restricted mapping'],
        'sample':selected}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'annotation_sample.json').write_text(json.dumps(document,indent=2,sort_keys=True,allow_nan=False)+'\n')
    (args.output/'annotation_sample.csv').write_bytes(csv_bytes)
    print(json.dumps({'status':STATUS,'n':len(selected),'counts':dict(Counter(r['stratum'] for r in selected)),
                      'population_weight_sum':document['sampling_design']['population_weight_sum'],
                      'draw_sha256':document['provenance']['draw_sha256']},indent=2))

if __name__=='__main__': main()
