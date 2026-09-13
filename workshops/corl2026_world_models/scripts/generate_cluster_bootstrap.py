#!/usr/bin/env python3
"""Generate, never apply, the Ali-owned cluster queue bootstrap Jobs.

The coordinator is CPU-only. Each of the 32 isolated B200 workers uses the
primary pod's verified 24 CPU/128Gi request and 48 CPU/256Gi limit. Keeping this
headroom avoids an unmeasured reduction in simulator/model capacity. Scheduler
admission, cross-pod NFS flock and an empty initial queue must be verified before
any job is released. Controllers admit work for seven days and retain up to
48 additional hours for an already-started job to drain. A generated manifest
is not scientific qualification.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

NAMESPACE='211247-prod'
STUDY='wmf_ablation_001_20260912'
SOURCE='/data/users/ali/vla_wam/src/'+STUDY
STATE='/data/users/ali/vla_wam/raw/'+STUDY+'/control'
BOOTSTRAP=STATE+'/bootstrap/cluster_queue.py'
IMAGE='artifactory-ci.gm.com/docker-approved/devcontainers/base@sha256:03f5ce7d090fbd378070a8216d0aedfc6e473c52da99b40b0cf53918612a297c'
CONTROLLER_WALL_SECONDS=604800
HASH_CHECK='''import hashlib, os, sys
path, expected = sys.argv[1:3]
with open(path, "rb") as stream:
    actual = hashlib.sha256(stream.read()).hexdigest()
if actual != expected:
    raise SystemExit("bootstrap digest mismatch; no queue code executed")
os.execv(sys.executable, [sys.executable, path, *sys.argv[3:]])
'''


def build_manifest(bootstrap_sha256, worker_count=32):
    if not isinstance(bootstrap_sha256,str) or not re.fullmatch('[0-9a-f]{64}',bootstrap_sha256):
        raise ValueError('a verified lowercase SHA-256 for the final bootstrap code is required')
    if type(worker_count) is not int or not 0<=worker_count<=32:
        raise ValueError('worker_count must be an integer from zero through 32')
    items=[]
    for role,index in [('coordinator',None)]+[('worker',i) for i in range(worker_count)]:
        name='wmf-forecast-0912-'+('coordinator' if index is None else f'worker-{index:02d}')
        labels={'app.kubernetes.io/name':'wmf-forecast-queue','app.kubernetes.io/part-of':STUDY,
                'user':'ali','wmf-role':role}
        args=[BOOTSTRAP,bootstrap_sha256,role,'--repo',SOURCE,'--state-dir',STATE,
              '--max-wall-seconds',str(CONTROLLER_WALL_SECONDS),'--poll-seconds','30']
        if role=='worker':args+=['--worker-id',name,'--role','any']
        else:args+=['--control-ref','refs/remotes/origin/codex/forecast-layout-gm-20260912',
                    '--queue-path','workshops/corl2026_world_models/execution/20260912/autonomy/cluster_queue.json',
                    '--results-branch','codex/forecast-layout-gm-20260912-results']
        resources=({'requests':{'cpu':'24','memory':'128Gi','nvidia.com/gpu':1},
                    'limits':{'cpu':'48','memory':'256Gi','nvidia.com/gpu':1}}
                   if role=='worker' else
                   {'requests':{'cpu':'2','memory':'8Gi'},'limits':{'cpu':'4','memory':'16Gi'}})
        environment={'USER':'ali','LOGNAME':'ali','HOME':'/home/ali',
                     'XDG_CACHE_HOME':'/home/ali/.cache','TMPDIR':'/tmp',
                     'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1',
                     'GIT_TERMINAL_PROMPT':'0','GCM_INTERACTIVE':'never'}
        if role=='worker':environment.update({'NVIDIA_VISIBLE_DEVICES':'all',
            'NVIDIA_DRIVER_CAPABILITIES':'compute,utility,graphics,display,video',
            'CUDA_DEVICE_ORDER':'PCI_BUS_ID','VK_ICD_FILENAMES':'/etc/vulkan/icd.d/nvidia_icd.json',
            'VK_DRIVER_FILES':'/etc/vulkan/icd.d/nvidia_icd.json',
            'LD_LIBRARY_PATH':'/data/users/jsalfity/glvnd/lib'})
        pod={'restartPolicy':'OnFailure','terminationGracePeriodSeconds':120,
             'automountServiceAccountToken':False,
             'securityContext':{'fsGroup':2518800,'supplementalGroups':[2518800],
                                'seccompProfile':{'type':'RuntimeDefault'}},
             'imagePullSecrets':[{'name':'artifactory-ci-pull-secret'}],
             'containers':[{'name':role,'image':IMAGE,'imagePullPolicy':'IfNotPresent',
                'command':['/usr/bin/python3','-c',HASH_CHECK],'args':args,'workingDir':SOURCE,
                'resources':resources,'env':[{'name':key,'value':value} for key,value in environment.items()],
                'securityContext':{'allowPrivilegeEscalation':False,'capabilities':{'drop':['ALL']},
                    'readOnlyRootFilesystem':False,'runAsGroup':2518800,'runAsUser':816149040,'runAsNonRoot':True},
                'volumeMounts':[{'name':'workspace','mountPath':'/data'},
                    {'name':'dshm','mountPath':'/dev/shm'}, {'name':'tmp','mountPath':'/tmp'},
                    {'name':'vartmp','mountPath':'/var/tmp'}, {'name':'home','mountPath':'/home/ali'},
                    {'name':'userdb','mountPath':'/etc/passwd','subPath':'passwd','readOnly':True},
                    {'name':'userdb','mountPath':'/etc/group','subPath':'group','readOnly':True}]}],
             'volumes':[{'name':'workspace','persistentVolumeClaim':{'claimName':'211247-prod-pvc'}},
                 {'name':'dshm','emptyDir':{'medium':'Memory'}}, {'name':'tmp','emptyDir':{}},
                 {'name':'vartmp','emptyDir':{}}, {'name':'home','emptyDir':{}},
                 {'name':'userdb','configMap':{'name':'211247-ali-b200-1gpu-userdb'}}]}
        if role=='worker':
            pod['nodeSelector']={'node-role.kubernetes.io/worker-gpu':'','nvidia.com/gpu.product':'NVIDIA-B200'}
            pod['tolerations']=[{'effect':'NoSchedule','key':'nvidia.com/gpu','operator':'Equal','value':'present'}]
        items.append({'apiVersion':'batch/v1','kind':'Job',
            'metadata':{'name':name,'namespace':NAMESPACE,'labels':labels,
                        'annotations':{'wmf-bootstrap-sha256':bootstrap_sha256}},
            'spec':{'completions':1,'parallelism':1,'backoffLimit':3,
                    'activeDeadlineSeconds':CONTROLLER_WALL_SECONDS+172800+300,
                    'template':{'metadata':{'labels':labels,'annotations':{'wmf-bootstrap-sha256':bootstrap_sha256}},'spec':pod}}})
    return {'apiVersion':'v1','kind':'List','items':items}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap-sha256',required=True)
    parser.add_argument('--worker-count',type=int,default=32)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=build_manifest(args.bootstrap_sha256,args.worker_count)
    data=json.dumps(manifest,indent=2,sort_keys=True)+'\n'
    if args.output.exists() and args.output.read_text()!=data:
        raise FileExistsError('refusing to overwrite a different bootstrap manifest')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if not args.output.exists():args.output.write_text(data)
    print(json.dumps({'output':str(args.output),'coordinators':1,'workers':args.worker_count,
                      'applied':False,'requires_cross_pod_flock_preflight':True}))

if __name__=='__main__':main()
