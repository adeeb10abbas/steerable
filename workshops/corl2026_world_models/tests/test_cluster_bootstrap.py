"""Bootstrap manifests and a harmless diagnostic, without cluster mutations."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'

def load(name):
    path=SCRIPTS/name
    if not path.exists(): return None
    spec=importlib.util.spec_from_file_location(name[:-3],path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.module=load('generate_cluster_bootstrap.py')
        self.assertIsNotNone(self.module,'bootstrap generator must be implemented')
    def test_exact_owned_pool_security_and_persistent_bootstrap(self):
        result=self.module.build_manifest('a'*64,admission_deadline_unix=1800000000)
        self.assertEqual(result['kind'],'List');self.assertEqual(len(result['items']),33)
        workers=[]
        for job in result['items']:
            spec=job['spec']['template']['spec'];container=spec['containers'][0]
            self.assertEqual(job['metadata']['namespace'],'211247-prod')
            self.assertEqual(job['metadata']['labels']['user'],'ali')
            self.assertFalse(spec['automountServiceAccountToken'])
            self.assertEqual(spec['restartPolicy'],'OnFailure')
            self.assertEqual(job['spec']['backoffLimit'],3)
            expected_deadline=777900 if job['metadata']['labels']['wmf-role']=='worker' else 778500
            self.assertEqual(job['spec']['activeDeadlineSeconds'],expected_deadline)
            self.assertEqual(spec['securityContext']['fsGroup'],2518800)
            self.assertEqual(container['securityContext']['runAsUser'],816149040)
            self.assertFalse(container['securityContext']['allowPrivilegeEscalation'])
            self.assertEqual(container['securityContext']['capabilities']['drop'],['ALL'])
            self.assertEqual(container['image'].split('@sha256:')[1],'03f5ce7d090fbd378070a8216d0aedfc6e473c52da99b40b0cf53918612a297c')
            self.assertIn('/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/control/bootstrap/cluster_queue.py',container['args'])
            cutoff_index=container['args'].index('--admission-deadline-unix')
            self.assertEqual(container['args'][cutoff_index+1],'1800000000')
            self.assertEqual([v['persistentVolumeClaim']['claimName'] for v in spec['volumes'] if 'persistentVolumeClaim' in v],['211247-prod-pvc'])
            if job['metadata']['labels']['wmf-role']=='worker':
                workers.append(job['metadata']['name'])
                self.assertEqual(container['resources']['requests']['nvidia.com/gpu'],1)
                self.assertEqual(container['resources']['limits']['nvidia.com/gpu'],1)
                self.assertEqual(container['resources']['requests']['cpu'],'24')
                self.assertEqual(container['resources']['requests']['memory'],'128Gi')
                self.assertEqual(spec['nodeSelector']['nvidia.com/gpu.product'],'NVIDIA-B200')
                self.assertIn('604800',container['args'])
                self.assertEqual(container['args'][container['args'].index('--role')+1],job['metadata']['name'])
                environment={item['name']:item['value'] for item in container['env']}
                self.assertEqual(environment['LD_LIBRARY_PATH'],'/data/users/jsalfity/glvnd/lib')
            else:
                self.assertIn('778200',container['args'])
                self.assertNotIn('nvidia.com/gpu',container['resources']['requests'])
        self.assertEqual(workers,[f'wmf-forecast-0912-worker-{i:02d}' for i in range(32)])
    def test_wrong_bootstrap_digest_stops_before_code_executes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'queue.py';marker=Path(tmp)/'marker'
            path.write_text(f'from pathlib import Path\nPath({str(marker)!r}).write_text("ran")\n')
            result=subprocess.run([sys.executable,'-c',self.module.HASH_CHECK,str(path),'0'*64],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertFalse(marker.exists())
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            result=subprocess.run([sys.executable,'-c',self.module.HASH_CHECK,str(path),digest],capture_output=True,text=True)
            self.assertEqual(result.returncode,0);self.assertEqual(marker.read_text(),'ran')
    def test_invalid_counts_and_hashes_rejected_and_output_is_deterministic(self):
        for digest in ['short','A'*64,'g'*64]:
            with self.assertRaises(ValueError):self.module.build_manifest(digest)
        with self.assertRaises(ValueError):self.module.build_manifest('a'*64,worker_count=33)
        self.assertEqual(self.module.build_manifest('a'*64,admission_deadline_unix=1800000000),
                         self.module.build_manifest('a'*64,admission_deadline_unix=1800000000))
        with patch.object(self.module.time,'time',return_value=1800000000):
            result=self.module.build_manifest('a'*64)
        deadlines={job['spec']['template']['spec']['containers'][0]['args'][job['spec']['template']['spec']['containers'][0]['args'].index('--admission-deadline-unix')+1] for job in result['items']}
        self.assertEqual(deadlines,{'1800604800'})

class DiagnosticTests(unittest.TestCase):
    def test_diagnostic_publishes_only_curated_observations(self):
        module=load('cluster_worker_diagnostic.py')
        self.assertIsNotNone(module,'cluster diagnostic must be implemented')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);binary=root/'nvidia-smi'
            binary.write_text('#!/bin/sh\ncase "$1" in\n--query-gpu=*) echo "0, GPU-test, NVIDIA B200, 580.95.05, 183359, 182632, 0";;\n--query-compute-apps=*) exit 0;;\nesac\n')
            binary.chmod(0o755)
            pod_uid='11111111-2222-3333-4444-555555555555'
            with (patch.dict(os.environ,{'PATH':str(root)+os.pathsep+os.environ['PATH'],
                                        'POD_UID':pod_uid,'GH_TOKEN':'do-not-publish-this'}),
                  patch.object(module.socket,'gethostname',return_value='fixture-worker-00-testpod')):
                result=module.write_diagnostic(root/'job',source_root=root,
                                               expected_worker_id='fixture-worker-00',
                                               expected_gpu_count=1)
            text=(root/'job/publish/diagnostic.json').read_text()
            self.assertNotIn('do-not-publish-this',text)
            self.assertFalse(result['scientific_qualification'])
            self.assertEqual(result['gpu']['count'],1)
            self.assertEqual(result['gpu']['devices'][0]['name'],'NVIDIA B200')
            self.assertEqual(result['gpu']['compute_processes'],[])
            self.assertEqual(result['pod_uid'],pod_uid)
            self.assertEqual(result['expected_worker_id'],'fixture-worker-00')
            self.assertEqual(result['worker_identity_errors'],[])
            self.assertTrue(result['idle_worker_checks_applied'])
            self.assertEqual(result['idle_worker_errors'],[])
            self.assertTrue(result['diagnostic_passed'])
            self.assertLess(len(text),16000)

    def test_worker_bound_diagnostic_exits_nonzero_when_not_idle_or_wrong_pod(self):
        module=load('cluster_worker_diagnostic.py')
        base_gpu={
            'available':True,
            'count':1,
            'devices':[{'index':0,'uuid':'GPU-test','name':'NVIDIA B200',
                        'driver_version':'580.95.05','memory.total':183359,
                        'memory.free':182632,'utilization.gpu':0}],
            'compute_processes':[],
            'errors':[],
        }
        cases={
            'busy':('visible_gpu_not_idle','fixture-worker-00-testpod','uid-test'),
            'process':('preexisting_compute_processes','fixture-worker-00-testpod','uid-test'),
            'wrong_index':('visible_gpu_index_mismatch','fixture-worker-00-testpod','uid-test'),
            'wrong_gpu':('visible_gpu_not_b200','fixture-worker-00-testpod','uid-test'),
            'wrong_host':('hostname_does_not_match_expected_worker','wrong-worker-testpod','uid-test'),
            'missing_pod_uid':('pod_uid_environment_missing','fixture-worker-00-testpod',''),
        }
        for case,(expected_error,hostname,pod_uid) in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                gpu=json.loads(json.dumps(base_gpu))
                if case=='busy':gpu['devices'][0]['utilization.gpu']=9
                if case=='process':gpu['compute_processes']=[{
                    'gpu_uuid':'GPU-test','pid':123,'process_name':'other','used_memory':1024}]
                if case=='wrong_index':gpu['devices'][0]['index']=1
                if case=='wrong_gpu':gpu['devices'][0]['name']='NVIDIA H100'
                argv=['cluster_worker_diagnostic.py','--job-dir',str(Path(tmp)/'job'),
                      '--expect-gpus','1','--expected-worker-id','fixture-worker-00']
                with (patch.object(module,'query_gpu',return_value=gpu),
                      patch.object(module.socket,'gethostname',return_value=hostname),
                      patch.dict(os.environ,{'POD_UID':pod_uid}),
                      patch.object(sys,'argv',argv),patch('builtins.print')):
                    exit_code=module.main()
                report=json.loads((Path(tmp)/'job/publish/diagnostic.json').read_text())
                self.assertEqual(exit_code,1)
                self.assertFalse(report['diagnostic_passed'])
                self.assertIn(expected_error,
                              report['worker_identity_errors']+report['idle_worker_errors'])

if __name__=='__main__':unittest.main()
