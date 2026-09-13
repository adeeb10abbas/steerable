#!/usr/bin/env python3
"""Publish a small GPU-worker diagnostic. Never inspect credentials or run a model."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time

NAMESPACE='wmf_ablation_001_20260912'


def numeric(text):
    try:return int(text.strip())
    except ValueError:return text.strip()


def query_gpu():
    executable=shutil.which('nvidia-smi')
    report={'available':executable is not None,'devices':[],'compute_processes':[], 'errors':[]}
    if executable is None:
        report.update(count=0);report['errors'].append({'query':'nvidia-smi','reason':'unavailable'});return report
    for name,columns in [('devices',['index','uuid','name','driver_version','memory.total','memory.free','utilization.gpu']),
                         ('compute_processes',['gpu_uuid','pid','process_name','used_memory'])]:
        query='--query-gpu=' if name=='devices' else '--query-compute-apps='
        try:
            result=subprocess.run([executable,query+','.join(columns),'--format=csv,noheader,nounits'],
                                  capture_output=True,text=True,timeout=15)
            if result.returncode:
                report['errors'].append({'query':name,'reason':'nonzero_exit','returncode':result.returncode});continue
            if len(result.stdout)>65536:
                report['errors'].append({'query':name,'reason':'unexpected_oversize_output'});continue
            for values in csv.reader(io.StringIO(result.stdout)):
                if not values:continue
                if len(values)!=len(columns):
                    report['errors'].append({'query':name,'reason':'unexpected_csv_shape'});continue
                row={key:value.strip() for key,value in zip(columns,values)}
                for key in set(columns)&{'index','pid','memory.total','memory.free','utilization.gpu','used_memory'}:
                    row[key]=numeric(row[key])
                report[name].append(row)
        except (OSError,subprocess.TimeoutExpired) as error:
            report['errors'].append({'query':name,'reason':type(error).__name__})
    report['count']=len(report['devices']);return report


def idle_worker_errors(gpu,expected_count):
    errors=[]
    if expected_count is None:
        errors.append('expected_gpu_count_missing')
        return errors
    if gpu.get('available') is not True:errors.append('nvidia_smi_unavailable')
    if gpu.get('errors') != []:errors.append('gpu_query_errors')
    devices=gpu.get('devices')
    if gpu.get('count') != expected_count or not isinstance(devices,list) or len(devices) != expected_count:
        errors.append('unexpected_gpu_count')
    else:
        if [device.get('index') for device in devices] != list(range(expected_count)):
            errors.append('visible_gpu_index_mismatch')
        if any(device.get('name') != 'NVIDIA B200' for device in devices):
            errors.append('visible_gpu_not_b200')
        if any(device.get('utilization.gpu') != 0 for device in devices):
            errors.append('visible_gpu_not_idle')
    if gpu.get('compute_processes') != []:errors.append('preexisting_compute_processes')
    return errors


def write_diagnostic(job_dir,source_root=None,expected_worker_id=None,expected_gpu_count=None):
    job_dir=Path(job_dir).resolve();job_dir.mkdir(parents=True,exist_ok=True)
    publish=job_dir/'publish'
    if publish.is_symlink():raise ValueError('publish directory must not be a symlink')
    publish.mkdir(exist_ok=True)
    start=time.monotonic();disk=shutil.disk_usage(job_dir)
    hostname=socket.gethostname();pod_uid=os.environ.get('POD_UID')
    worker_identity_errors=[]
    if expected_worker_id is not None:
        if not hostname.startswith(expected_worker_id+'-'):
            worker_identity_errors.append('hostname_does_not_match_expected_worker')
        if not pod_uid:
            worker_identity_errors.append('pod_uid_environment_missing')
    gpu=query_gpu();idle_errors=(idle_worker_errors(gpu,expected_gpu_count)
                                 if expected_worker_id is not None else [])
    diagnostic_passed=(not worker_identity_errors and not idle_errors
                       if expected_worker_id is not None
                       else not gpu['errors'] and
                       (expected_gpu_count is None or gpu['count']==expected_gpu_count))
    report={'schema_version':'wmf-worker-diagnostic-v1','namespace':NAMESPACE,
            'observed_at_utc':datetime.now(timezone.utc).isoformat(),
            'hostname':hostname,'pod_uid':pod_uid,'expected_worker_id':expected_worker_id,
            'worker_identity_errors':worker_identity_errors,
            'expected_gpu_count':expected_gpu_count,
            'idle_worker_checks_applied':expected_worker_id is not None,
            'idle_worker_errors':idle_errors,'diagnostic_passed':diagnostic_passed,
            'pid':os.getpid(),'uid':os.getuid(),'gid':os.getgid(),
            'job_dir':str(job_dir),'job_dir_writable':os.access(job_dir,os.W_OK),
            'filesystem_space':{'total_bytes':disk.total,'free_bytes':disk.free,
                                'interpretation':'filesystem capacity, not a verified per-user quota'},
            'gpu':gpu,'scientific_qualification':False,
            'cross_pod_lock_qualification':False,
            'scope':'Harmless host/GPU/logging diagnostic only; no model import, generation, simulator or credential inspection.'}
    if source_root is not None:
        source=Path(source_root).resolve();report['source_root']=str(source)
        try:
            result=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],capture_output=True,text=True,timeout=10)
            report['source_commit']=result.stdout.strip() if result.returncode==0 else None
        except (OSError,subprocess.TimeoutExpired):report['source_commit']=None
    report['elapsed_seconds']=time.monotonic()-start
    target=publish/'diagnostic.json'
    if target.exists() or target.is_symlink():raise FileExistsError('diagnostic already exists; preserve the prior attempt')
    fd,temporary=tempfile.mkstemp(prefix='.diagnostic-',dir=publish)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.link(temporary,target)
    finally:os.unlink(temporary)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-dir',type=Path,required=True)
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--expect-gpus',type=int)
    parser.add_argument('--expected-worker-id')
    args=parser.parse_args()
    if args.expect_gpus is not None and args.expect_gpus<0:parser.error('expect-gpus must be nonnegative')
    if args.expected_worker_id is not None and (
        not args.expected_worker_id or len(args.expected_worker_id)>96
        or any(not (char.isalnum() or char in '._-') for char in args.expected_worker_id)
    ):parser.error('expected-worker-id is not a safe queue role')
    report=write_diagnostic(args.job_dir,args.source_root,args.expected_worker_id,args.expect_gpus)
    passed=report['diagnostic_passed']
    print(json.dumps({'diagnostic_published':True,'gpu_count':report['gpu']['count'],'diagnostic_passed':passed,'scientific_qualification':False}))
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
