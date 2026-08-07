#!/usr/bin/env python3
"""Run registered E003 cells sequentially on one ali-owned simulator lane."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

def sha(p:Path)->str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main()->None:
 ap=argparse.ArgumentParser(); ap.add_argument('--repo-root',type=Path,required=True); ap.add_argument('--registration',type=Path,required=True); ap.add_argument('--runtime',type=Path,required=True); ap.add_argument('--release-gate',type=Path,required=True); ap.add_argument('--candidate',type=Path,required=True); ap.add_argument('--raw-root',type=Path,required=True); ap.add_argument('--remote-host',required=True); ap.add_argument('--remote-port',type=int,default=8001); ap.add_argument('--lane-pod-uid',required=True); ap.add_argument('--lane-gpu-uuid',required=True); ap.add_argument('--gpu-index',type=int,default=0); ap.add_argument('--limit',type=int); a=ap.parse_args()
 reg=json.loads(a.registration.read_text()); cells=reg['queue']; cells=sorted(cells,key=lambda x:(x['environment_seed'],x['execution_order_index_within_seed']))
 if a.limit: cells=cells[:a.limit]
 base=a.raw_root.resolve()/'V3-E003_pi05_symmetric'; results=[]
 for row in cells:
  cid=row['cell_id']; attempt=base/cid.replace(':','__')/'attempt01'
  if attempt.exists():
   if (attempt/'raw_episode.jsonl').is_file(): results.append({'cell_id':cid,'status':'already_compiled'}); continue
   # Retain an infrastructure-invalid partial and advance to a fresh attempt.
   # A partial is never relabeled as a behavioral failure or overwritten.
   n=2
   while (base/cid.replace(':','__')/f'attempt{n:02d}').exists(): n += 1
   attempt=base/cid.replace(':','__')/f'attempt{n:02d}'
  attempt.mkdir(parents=True)
  env=dict(os.environ); env.update({'OMNI_KIT_ACCEPT_EULA':'YES','NVIDIA_DRIVER_CAPABILITIES':'all','VK_ICD_FILENAMES':'/etc/vulkan/icd.d/nvidia_icd.json','TMPDIR':f'/tmp/vla_wam_v3e003/{cid.replace(":","__")}'})
  for key in ('TMPDIR',): Path(env[key]).mkdir(parents=True,exist_ok=False)
  common=['--study-root',str(a.repo_root.resolve()),'--release-manifest',str(a.registration.resolve()),'--release-manifest-sha256',sha(a.registration),'--runtime-manifest',str(a.runtime.resolve()),'--release-gate',str(a.release_gate.resolve()),'--cell-id',cid,'--lane-pod-uid',a.lane_pod_uid,'--lane-gpu-uuid',a.lane_gpu_uuid,'--fixture-candidate',str(a.candidate.resolve()),'--fixture-candidate-sha256',sha(a.candidate),'--state-capture-dir',str(attempt/'state_capture'),'--action-trace-dir',str(attempt/'action_traces'),'--reset-attestation',str(attempt/'reset_attestation.json'),'--simulator-export',str(attempt/'simulator_export.json'),'--output-dir',str(attempt/'simulator'),'--remote-host',a.remote_host,'--remote-port',str(a.remote_port),'--open-loop-horizon','15','--instruction-controller','static','--output-folder-name',str(attempt/'simulator'),'--num-envs','1','--num-runs','1','--headless','--renderer','realtime','--rendering-type','balanced','--device','cuda:0','--kit_args=--/rtx/verifyDriverVersion/enabled=false','--video-mode','viewport','--instruction-type','default','--disable-subtask']
  cmd=[sys.executable,'-m','experiments.v3.phase_e.bilateral_symmetry_null_control_v3e003.robolab_bridge',*common]
  try:
   subprocess.run(cmd,cwd=a.repo_root,env=env,check=True)
   subprocess.run([sys.executable,'-m','experiments.v3.phase_e.bilateral_symmetry_null_control_v3e003.compile_cell','--registration',str(a.registration),'--registration-sha256',sha(a.registration),'--runtime',str(a.runtime),'--export',str(attempt/'simulator_export.json'),'--output',str(attempt/'raw_episode.jsonl')],cwd=a.repo_root,env=env,check=True)
   results.append({'cell_id':cid,'status':'compiled_valid_behavioral_cell'})
  except Exception as exc:
   (attempt/'infrastructure_failure.json').write_text(json.dumps({'cell_id':cid,'denominator_eligible':False,'error':f'{type(exc).__name__}: {exc}'},indent=2)+'\n')
   results.append({'cell_id':cid,'status':'infrastructure_failed_excluded_from_denominator'})
   raise
 (base/'queue_results.json').parent.mkdir(parents=True,exist_ok=True); (base/'queue_results.json').write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps({'completed':sum(x['status']=='compiled_valid_behavioral_cell' for x in results),'results':results},indent=2))
if __name__=='__main__': main()
