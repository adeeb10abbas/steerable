#!/usr/bin/env python3
"""Read-only upstream audit. Cache replay is not a corrected fidelity estimate."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess

QUADRANTS = ('imagines_requested_executes_requested',
             'imagines_requested_executes_not_requested',
             'does_not_imagine_requested_executes_requested',
             'neither_imagines_nor_executes_requested', 'uncertain_future')
FRAMES = [8, 16, 24, 32]

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def validate_ids(actual, expected):
    counts = Counter(actual)
    if any(v > 1 for v in counts.values()): raise ValueError('duplicate cohort IDs')
    missing, extra = set(expected)-set(actual), set(actual)-set(expected)
    if missing: raise ValueError(f'missing cohort IDs: {sorted(missing)[:5]}')
    if extra: raise ValueError(f'unexpected cohort IDs: {sorted(extra)[:5]}')

def planar_direction(cube, bowl):
    if len(cube) != 2 or len(bowl) != 2: raise ValueError('planar predicate requires exactly XY')
    if not all(math.isfinite(float(x)) for x in list(cube)+list(bowl)): raise ValueError('nonfinite geometry')
    dx, dy = cube[0]-bowl[0], cube[1]-bowl[1]
    norm = math.hypot(dx, dy)
    if norm <= 1e-6: return 'neutral'
    cosine = dy/norm
    if cosine >= math.cos(math.radians(45)): return 'left'
    if cosine <= -math.cos(math.radians(45)): return 'right'
    return 'neutral'

def legacy_label(frames, requested):
    reliable = [f for f in frames if f['semantics']['reliable']]
    if len(reliable) < 2: return None
    fraction = sum(f['semantics']['relation'] == requested for f in reliable)/len(reliable)
    return True if fraction >= .75 else False if fraction <= .25 else None

def endpoint_label(frames, requested):
    selected = [f for f in frames if f['frame_index'] == 32]
    if len(selected) != 1: raise ValueError('missing or duplicate endpoint frame')
    s = selected[0]['semantics']
    return s['relation'] == requested if s['reliable'] else None

def quadrant(predicted, actual):
    if predicted is None: return QUADRANTS[4]
    return QUADRANTS[(0 if actual else 1) if predicted else (2 if actual else 3)]

def validate_pair(record, root):
    """Fail-closed normalized adapter validation, not a raw-data extractor.

    time_index is a common aligned physical sample ID, not a video frame number.
    Provenance must document the mapping, prediction and execution coordinates.
    Valid hashes establish byte identity, not truth of alignment/annotations.
    """
    p, e = record['prediction'], record['execution']
    for name,side in (('prediction',p),('execution',e)):
        time=side.get('time_index')
        if type(time) is not int or time < 0:
            raise ValueError(f'{name} time_index must be a nonnegative integer')
        if type(side.get('reliable')) is not bool:
            raise ValueError(f'{name} reliability must be boolean')
    if not e['reliable']:
        raise ValueError('execution unknown: ineligible for fidelity; retain in missingness ledger')
    for field in ('time_index','coordinate_frame','geometry'):
        if p[field] != e[field]: raise ValueError(f'{field} mismatch')
    if p['geometry'] != 'visual_centroid_xy': raise ValueError('unsupported geometry')
    if not p['coordinate_frame']: raise ValueError('invalid alignment')
    for name,side in (('prediction',p),('execution',e)):
        for entity in ('cube','bowl'):
            xy=side.get(entity)
            if xy is None and not side['reliable']:
                continue  # No coordinates are invented for an unreadable forecast.
            if (not isinstance(xy,(list,tuple)) or len(xy)!=2
                    or not all(type(x) in (int,float) and math.isfinite(x) for x in xy)):
                raise ValueError(f'{name} {entity} geometry requires finite numeric XY')
    provenance = record.get('provenance', [])
    if not {'prediction','execution','alignment'} <= {x.get('role') for x in provenance}:
        raise ValueError('missing provenance roles')
    for source in provenance:
        path = root/source['path']
        if not path.is_file() or sha256(path) != source['sha256']: raise ValueError('provenance file/hash mismatch')

def replay_frame(frame, threshold):
    s = frame['semantics']; coords = s['world_xy_by_camera']
    relations = {}
    reasons = []
    for camera in ('left_camera','right_camera'):
        c,b = coords[camera]['cube'], coords[camera]['bowl']
        relations[camera] = planar_direction(c,b) if c is not None and b is not None else None
    if None in relations.values(): reasons.append('missing_localization')
    if len(set(relations.values())) != 1: reasons.append('camera_relation_disagreement')
    for obj in ('cube','bowl'):
        a,b = coords['left_camera'][obj], coords['right_camera'][obj]
        distance = math.dist(a,b) if a is not None and b is not None else None
        stored = s['cross_camera_disagreement_m'][obj]
        if (distance is None) != (stored is None) or (distance is not None and abs(distance-stored)>1e-9):
            raise ValueError('cached distance mismatch')
        if distance is not None and distance > threshold: reasons.append(f'{obj}_cross_camera_disagreement')
    return {'frame_index':frame['frame_index'], 'semantics':{
        'reliable':not reasons, 'relation':relations['left_camera'] if not reasons else None, 'reasons':reasons}}

def bool_csv(s):
    if s == '': return None
    if s not in ('True','False'): raise ValueError('invalid CSV boolean')
    return s == 'True'

def metrics(rows, label_key):
    counts = Counter(quadrant(r[label_key],r['actual']) for r in rows)
    certain = sum(r[label_key] is not None for r in rows)
    positives = sum(r['actual'] for r in rows)
    covered_positive = sum(r['actual'] and r[label_key] is not None for r in rows)
    agreement = sum(r[label_key] is not None and r[label_key] == r['actual'] for r in rows)
    negative_baseline = sum(r[label_key] is not None and not r['actual'] for r in rows)
    return {'n':len(rows),'certain':certain,'executed_positive':positives,
            'historical_label_arithmetic':{'denominator':certain,'observed_agreement_numerator':agreement,
            'constant_negative_agreement_numerator':negative_baseline,
            'observed_agreement':agreement/certain if certain else None,
            'constant_negative_agreement':negative_baseline/certain if certain else None,
            'difference_percentage_points':100*(agreement-negative_baseline)/certain if certain else None,
            'interpretation':'same covered subset only; not validated fidelity or utility'},
            'covered_executed_positive':covered_positive,'quadrants':{k:counts[k] for k in QUADRANTS},
            'coverage':certain/len(rows), 'positive_execution_coverage':covered_positive/positives if positives else None}

def cluster_intervals(rows, label_key, repeats=2000):
    strata = defaultdict(lambda: defaultdict(list))
    tier_wordings={6100:('canonical','short'),7200:('declarative','contrastive')}
    wording_tier={wording:tier for tier,wordings in tier_wordings.items() for wording in wordings}
    for r in rows:
        tier=wording_tier.get(r['wording'])
        ep,chunk,sampling_seed=r.get('episode'),r.get('replan'),r.get('sampling_seed')
        if (tier is None or type(ep) is not int or not 0 <= ep < 10
                or type(chunk) is not int or chunk < 0 or type(sampling_seed) is not int
                or sampling_seed != (tier+ep)*1000+chunk):
            raise ValueError('unrecoverable or inconsistent shared episode seed')
        strata[tier][tier+ep].append(r)
    for tier,blocks in strata.items():
        expected={(w,d) for w in tier_wordings[tier] for d in ('left','right')}
        for seed,block in blocks.items():
            if {(r['wording'],r['requested']) for r in block} != expected:
                raise ValueError(f'incomplete four-episode seed block: {seed}')
    rng=random.Random(20260912); samples=defaultdict(list)
    for _ in range(repeats):
        selected=[]
        for tier in sorted(strata):
            values=[strata[tier][seed] for seed in sorted(strata[tier])]
            for _ in values: selected.extend(rng.choice(values))
        m=metrics(selected,label_key)
        for key in ('coverage','positive_execution_coverage'):
            if m[key] is not None: samples[key].append(m[key])
    def interval(v):
        v=sorted(v)
        return [v[math.floor(.025*(len(v)-1))],v[math.ceil(.975*(len(v)-1))]]
    return {'method':'percentile matched-seed-block bootstrap within 6100 and 7200 tiers; four episodes per block, all chunks; pooled chunk ratio',
            'independent_seed_blocks':sum(len(blocks) for blocks in strata.values()),
            'seed_blocks_by_tier':{str(tier):len(blocks) for tier,blocks in strata.items()},
            'episodes_per_block':4,
            'seed_identity_validation':'sampling_seed == (tier + episode_index) * 1000 + replan_index for every row; both wordings and both directions required per block',
            'replicates':repeats,'seed':20260912,'confidence':.95,
            'limitations':'Descriptive sampling uncertainty only; does not include evaluator error or selection bias.',
            'intervals':{k:interval(v) for k,v in samples.items()}}

def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(); root=args.root.resolve()
    out=args.output or root/'workshops/corl2026_world_models/results'; out.mkdir(parents=True,exist_ok=True)
    base=root/'artifacts/vla_wam_shared_v1'; confirmation=base/'semantic_confirmation'
    calibration=base/'semantic_future_calibration.json'; calibration_hash=sha256(calibration)
    threshold=json.loads(calibration.read_text())['cross_camera_disagreement_threshold_m']
    expected_cells={'cosmos_canonical':{'left':99,'right':86},'cosmos_vague':{'left':125,'right':64},
                    'cosmos_declarative':{'left':66,'right':87},'cosmos_contrastive':{'left':146,'right':79}}
    inputs=[calibration,base/'final_evidence/raw_evidence_manifest.csv',base/'final_evidence/supporting_evidence_manifest.csv',
            root/'tools/score_cosmos_semantic_futures.py',base/'semantic_confirmation_audit.md']
    rows=[]; expected_cache=[]; counts=Counter(); reasons=Counter(); reliable_frames=0
    for condition,cells in expected_cells.items():
        source=confirmation/condition/'semantic_quadrants.csv'; inputs.append(source)
        for row in csv.DictReader(source.open()):
            task=Path(row['task_dir']); ep=int(row['episode_index']); chunk=int(row['replan_index'])
            key=f'{condition}/{task.name}/episode_{ep:03d}/chunk_{chunk:03d}'
            cache=confirmation/condition/'localization_cache'/f'{task.parent.name}__{task.name}'/f'episode_{ep:03d}_chunk_{chunk:03d}.json'
            expected_cache.append(str(cache.relative_to(confirmation))); inputs.append(cache)
            raw=json.loads(cache.read_text())
            if raw['calibration_sha256'] != calibration_hash: raise ValueError('calibration hash mismatch')
            if raw['frame_indices'] != FRAMES or [f['frame_index'] for f in raw['frames']] != FRAMES:
                raise ValueError('missing/duplicate/misordered scored frames')
            replay=[replay_frame(f,threshold) for f in raw['frames']]
            for original,rebuilt in zip(raw['frames'],replay):
                for field in ('reliable','relation','reasons'):
                    if original['semantics'][field] != rebuilt['semantics'][field]: raise ValueError('frame semantics mismatch')
                reliable_frames += rebuilt['semantics']['reliable']; reasons.update(rebuilt['semantics']['reasons'])
            requested=row['requested_relation']; label=legacy_label(replay,requested); actual=bool_csv(row['executed_requested'])
            if actual is None or actual != (row['executed_relation']==requested): raise ValueError('execution CSV inconsistency')
            nrel=sum(f['semantics']['reliable'] for f in replay)
            nreq=sum(f['semantics']['reliable'] and f['semantics']['relation']==requested for f in replay)
            if label != bool_csv(row['imagined_requested']) or quadrant(label,actual) != row['quadrant'] or nrel != int(row['reliable_future_frames']) or nreq != int(row['requested_future_frames']):
                raise ValueError('legacy CSV replay mismatch')
            counts[(condition,requested)]+=1
            rows.append({'id':key,'wording':'short' if condition=='cosmos_vague' else condition.removeprefix('cosmos_'),
                         'requested':requested,'episode':ep,'replan':chunk,'sampling_seed':int(row['sampling_seed']),
                         'legacy':label,'endpoint_sensitivity':endpoint_label(replay,requested),'actual':actual,
                         'legacy_quadrant':quadrant(label,actual),'execution_end_step':int(row['execution_end_step']),
                         'task_dir':str(task),'chunk_dir':row['chunk_dir'],
                         'threshold_predictions':{str(t):legacy_label([replay_frame(f,t) for f in raw['frames']],requested) for t in (.1,.15,.2)}})
    validate_ids([r['id'] for r in rows],set(r['id'] for r in rows))
    for c,cells in expected_cells.items():
        for direction,n in cells.items():
            if counts[(c,direction)] != n: raise ValueError('registered cohort cell count mismatch')
    validate_ids([str(p.relative_to(confirmation)) for p in confirmation.glob('*/localization_cache/*/*.json')],expected_cache)
    episodes={(r['wording'],r['requested'],r['episode']) for r in rows}
    expected_episodes={(w,d,e) for w in ('canonical','short','declarative','contrastive') for d in ('left','right') for e in range(10)}
    validate_ids(list(episodes),expected_episodes)
    if len(episodes)!=80 or len(rows)!=752: raise ValueError('cohort total mismatch')
    # Exact required raw paths are derived from row identity, never guessed from seeds.
    required={str(Path(r['task_dir'])/f"run_{r['episode']}.hdf5"):'execution_trajectory' for r in rows}
    required.update({str(Path(r['chunk_dir'])/'metadata.json'):'chunk_alignment_metadata' for r in rows})
    required.update({str(Path(r['task_dir'])/'env_cfg.json'):'camera_and_coordinate_configuration' for r in rows})
    raw_manifest={r['absolute_path']:r for r in csv.DictReader((base/'final_evidence/raw_evidence_manifest.csv').open())}
    missing=[]
    for path,role in sorted(required.items()):
        receipt=raw_manifest.get(path)
        missing.append({'absolute_source_path':path,'role':role,'manifest_sha256':receipt['sha256'] if receipt else None,
                        'manifest_bytes':int(receipt['bytes']) if receipt else None,
                        'available_in_workshop_checkout':False,'receipt_status':'hash_in_upstream_manifest' if receipt else 'no_hash_receipt_in_manifest'})
    media=[]
    for row in rows:
        for name,role in [('action.npy','predicted_action'),('conditioning.png','conditioning_image'),('future.mp4','generated_video')]:
            path=str(Path(row['chunk_dir'])/name); receipt=raw_manifest.get(path)
            media.append({'absolute_source_path':path,'role':role,'manifest_sha256':receipt['sha256'] if receipt else None,
                          'manifest_bytes':int(receipt['bytes']) if receipt else None,'available_in_workshop_checkout':False})
    source_hashes=[{'path':str(p.relative_to(root)),'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted(set(inputs))]
    baseline_commit='ce561e66'
    tree=subprocess.check_output(['git','ls-tree','-r',baseline_commit],cwd=root,text=True)
    blobs={line.split('\t',1)[1]:line.split()[2] for line in tree.splitlines()}
    for receipt in source_hashes:
        data=(root/receipt['path']).read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        receipt['upstream_git_blob_sha1']=blobs.get(receipt['path'])
        receipt['matches_upstream_commit']=blob==receipt['upstream_git_blob_sha1']
        if not receipt['matches_upstream_commit']: raise ValueError('upstream source changed: '+receipt['path'])
    historical={Path(r['absolute_path']).as_posix():r for r in csv.DictReader((base/'final_evidence/supporting_evidence_manifest.csv').open())}
    verified=0; mismatches=[]
    for receipt in source_hashes:
        suffix='/'+receipt['path']
        matches=[v for k,v in historical.items() if k.endswith(suffix)]
        if matches:
            if any(v['sha256']!=receipt['sha256'] for v in matches): mismatches.append(receipt['path'])
            else: verified+=1
    if mismatches: raise ValueError(f'historical source hash mismatch: {mismatches}')
    legacy=metrics(rows,'legacy'); endpoint=metrics(rows,'endpoint_sensitivity')
    thresholds={}
    for threshold in (.1,.15,.2):
        replaced=[dict(r,threshold_label=r['threshold_predictions'][str(threshold)]) for r in rows]
        thresholds[str(threshold)]=metrics(replaced,'threshold_label')
    summary={'cohort':{'chunks':len(rows),'episodes':len(episodes),'frames':len(rows)*4,'reliable_frames':reliable_frames,
                       'frame_rejection_reasons_overlapping':dict(reasons)},
             'legacy_replay':legacy,'legacy_coverage_uncertainty':cluster_intervals(rows,'legacy'),
             'threshold_replay':thresholds,
             'endpoint_sensitivity':{'status':'posthoc aggregation sensitivity, NOT corrected fidelity',
                                    'limitations':['video/action time correspondence unverified without metadata and interface evidence',
                                                  'visual planar world-coordinate prediction versus robot-frame root-pose plus height execution',
                                                  'fallible localizer and reliability selection persist'],
                                    'metrics':endpoint,'coverage_uncertainty':cluster_intervals(rows,'endpoint_sensitivity')},
             'source_validation':{'inputs_hashed':len(source_hashes),'historical_receipts_verified':verified,'upstream_commit':baseline_commit,
                                  'upstream_git_blobs_verified':len(source_hashes),
                                  'historical_hash_mismatches':mismatches,'cache_calibration_hash':calibration_hash,
                                  'csv_label_mismatches':0,'cache_semantics_mismatches':0},
             'corrected_fidelity_readiness':{'ready':False,'reason':'raw trajectories and alignment metadata absent; prediction geometry is planar visual XY',
                                            'required_paths':len(missing),'missing_by_role':dict(Counter(x['role'] for x in missing)),
                                            'manifest_hashes_available':sum(x['manifest_sha256'] is not None for x in missing)}}
    dump(out/'audit_summary.json',summary); dump(out/'source_hashes.json',source_hashes)
    dump(out/'missing_data_inventory.json',missing)
    dump(out/'missing_media_inventory.json',media)
    with (out/'reconciled_chunks.jsonl').open('w') as f:
        for row in rows: f.write(json.dumps(row,sort_keys=True)+'\n')
    dump(out/'legacy_mismatches.json',[r for r in rows if r['legacy'] is not None and r['legacy']!=r['actual']])
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
