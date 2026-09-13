"""Real filesystem/Git/subprocess checks for the cluster-only transport."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'execution/20260912/autonomy/cluster_queue.py'

class ClusterQueueTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.exists(), 'cluster queue backend must be implemented')
        spec = importlib.util.spec_from_file_location('cluster_queue', MODULE)
        self.q = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.q)
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.repo = self.root/'repo'; self.repo.mkdir()
        self.state = self.root/'state'
        self.git('init', '-q'); self.git('config','user.name','Queue test'); self.git('config','user.email','test@example.invalid')
        (self.repo/'run.py').write_text('import sys\nprint("diagnostic complete")\nsys.exit(int(sys.argv[1]))\n')
        self.git('add','.'); self.git('commit','-qm','source')
        self.commit = self.git('rev-parse','HEAD').strip()
        self.ref = 'refs/remotes/origin/codex/forecast-layout-gm-20260912'
        self.git('update-ref',self.ref,self.commit)
    def git(self,*args):
        return subprocess.check_output(['git','-C',str(self.repo),*args],text=True,stderr=subprocess.PIPE)
    def manifest(self, jobs=None, shutdown=False):
        return {'schema_version':'wmf-cluster-queue-v1','namespace':self.q.NAMESPACE,'shutdown':shutdown,'jobs':jobs or []}
    def job(self, ident='diagnostic-001', code=0):
        return {'job_id':ident,'released':True,'source_commit':self.commit,'role':'any','argv':[sys.executable,'{source_root}/run.py',str(code)],'max_wall_seconds':10}
    def stage(self, manifest):
        return self.q.stage_queue(self.repo,self.state,self.ref,manifest)
    def test_released_only_and_exact_trusted_commit(self):
        self.stage(self.manifest([{'job_id':'future','released':False},self.job()]))
        self.assertFalse((self.state/'jobs/future').exists())
        descriptor=json.loads((self.state/'jobs/diagnostic-001/descriptor.json').read_text())
        self.assertEqual(descriptor['source_commit'],self.commit)
        self.assertEqual(subprocess.check_output(['git','-C',str(self.state/'sources'/self.commit),'rev-parse','HEAD'],text=True).strip(),self.commit)
    def test_unreachable_commit_is_rejected(self):
        self.git('checkout','--orphan','unrelated'); self.git('commit','-qm','other root')
        job=self.job(); job['source_commit']=self.git('rev-parse','HEAD').strip()
        with self.assertRaisesRegex(ValueError,'reachable'):
            self.stage(self.manifest([job]))
        self.assertFalse((self.state/'jobs/diagnostic-001/descriptor.json').exists())
    def test_descriptor_cannot_change_after_staging(self):
        self.stage(self.manifest([self.job()])); changed=self.job(code=7)
        with self.assertRaisesRegex(ValueError,'immutable'):
            self.stage(self.manifest([changed]))
        self.assertEqual(json.loads((self.state/'jobs/diagnostic-001/descriptor.json').read_text())['argv'][-1],'0')
    def test_atomic_claim_excludes_second_worker_even_without_receipt(self):
        self.stage(self.manifest([self.job()]))
        path=self.state/'jobs/diagnostic-001'
        self.assertTrue(self.q.claim_job(path,'worker-1'))
        self.assertFalse(self.q.claim_job(path,'worker-2'))
        self.assertTrue((path/'claim').is_dir())
        status=self.q.snapshot(self.state,stale_after_seconds=-1)
        self.assertEqual(status['jobs'][0]['status'],'stale_claim_requires_decision')
    def test_failed_receipt_and_full_logs_are_retained_without_retry(self):
        self.stage(self.manifest([self.job(code=7)]))
        self.assertEqual(self.q.worker_once(self.repo,self.state,'worker-1','any',max_wall_seconds=10,poll_seconds=.02),1)
        path=self.state/'jobs/diagnostic-001'; result=json.loads((path/'result.json').read_text())
        self.assertEqual(result['status'],'failed'); self.assertEqual(result['returncode'],7)
        self.assertIn('diagnostic complete',(path/'stdout.log').read_text())
        self.assertEqual(self.q.worker_once(self.repo,self.state,'worker-2','any',max_wall_seconds=10),0)
        self.assertEqual(json.loads((path/'result.json').read_text()),result)
    def test_timeout_terminates_only_owned_child_and_retains_partial_output(self):
        job=self.job(); job['argv']=[sys.executable,'-u','-c','import time; print("partial"); time.sleep(20)']
        self.stage(self.manifest([job]))
        self.q.worker_once(self.repo,self.state,'worker-1','any',max_wall_seconds=.15,poll_seconds=.02,terminate_grace_seconds=.1)
        path=self.state/'jobs/diagnostic-001'; result=json.loads((path/'result.json').read_text())
        self.assertEqual(result['status'],'timed_out'); self.assertIn('partial',(path/'stdout.log').read_text())
        self.assertTrue(result['child_reaped'])
    def test_shutdown_prevents_new_claims_and_wrong_namespace_fails(self):
        self.stage(self.manifest([self.job()],shutdown=True))
        self.assertEqual(self.q.worker_once(self.repo,self.state,'worker-1','any',max_wall_seconds=10),0)
        wrong=self.manifest(); wrong['namespace']='other'
        with self.assertRaises(ValueError): self.stage(wrong)
    def test_results_are_compact_and_push_without_touching_control_tree(self):
        bare=self.root/'remote.git'; subprocess.run(['git','init','--bare','-q',str(bare)],check=True)
        self.git('remote','add','origin',str(bare)); self.git('push','-q','origin','HEAD:refs/heads/control')
        self.stage(self.manifest([self.job()])); self.q.worker_once(self.repo,self.state,'worker-1','any',max_wall_seconds=10,poll_seconds=.02)
        path=self.state/'jobs/diagnostic-001'; (path/'stdout.log').write_text('Bearer private-token-value\n'+'x'*100000)
        before=self.git('status','--porcelain')
        self.q.publish_results(self.repo,self.state,self.ref,'codex/forecast-layout-gm-20260912-results')
        self.assertEqual(self.git('status','--porcelain'),before)
        data=subprocess.check_output(['git','--git-dir',str(bare),'show','refs/heads/codex/forecast-layout-gm-20260912-results:results/wmf_ablation_001_20260912/status.json'],text=True)
        self.assertNotIn('private-token-value',data); self.assertLess(len(data),15000)
        self.assertIn('succeeded',data)
        self.assertNotIn('log_tails',data)
        self.q.publish_results(self.repo,self.state,self.ref,'codex/forecast-layout-gm-20260912-results')
    def test_log_tails_require_explicit_opt_in_and_share_one_bounded_budget(self):
        job=self.job(); job['publish_log_tail_bytes']=12
        self.stage(self.manifest([job])); self.q.worker_once(self.repo,self.state,'worker-1','any',max_wall_seconds=10,poll_seconds=.02)
        directory=self.state/'jobs/diagnostic-001'
        (directory/'stdout.log').write_text('stdout-prefix-123456789')
        (directory/'stderr.log').write_text('stderr-prefix-ABCDEFGHI')
        tails=self.q.snapshot(self.state)['jobs'][0]['log_tails']
        self.assertLessEqual(sum(len(v.encode()) for v in tails.values()),12)
        self.assertEqual(tails['stdout'],'456789'); self.assertEqual(tails['stderr'],'DEFGHI')
        invalid=self.job('invalid'); invalid['publish_log_tail_bytes']=8193
        with self.assertRaises(ValueError): self.stage(self.manifest([invalid]))

    def test_selected_artifacts_are_bounded_and_symlinks_never_copied(self):
        source=self.root/'job'; publish=source/'publish'; publish.mkdir(parents=True)
        (publish/'selected.json').write_text('{"frame": 1}\n')
        (publish/'escape.png').symlink_to(self.repo/'run.py')
        with (publish/'oversize.mp4').open('wb') as f: f.truncate(16*1024*1024+1)
        destination=self.root/'export'
        report=self.q.copy_publish_artifacts(source,destination)
        self.assertEqual([x['path'] for x in report['files']],['selected.json'])
        self.assertEqual({x['reason'] for x in report['errors']},{'symlink_rejected','per_file_limit_exceeded'})
        self.assertEqual((destination/'selected.json').read_text(),'{"frame": 1}\n')
        self.assertFalse((destination/'escape.png').exists())
        self.assertFalse((destination/'oversize.mp4').exists())

    def test_unsafe_job_ids_and_source_mutation_fail_closed(self):
        job=self.job('../escape')
        with self.assertRaises(ValueError): self.stage(self.manifest([job]))
        self.stage(self.manifest([self.job()]))
        (self.state/'sources'/self.commit/'run.py').write_text('print("tampered")')
        self.q.worker_once(self.repo,self.state,'worker-1','any',max_wall_seconds=10,poll_seconds=.02)
        result=json.loads((self.state/'jobs/diagnostic-001/result.json').read_text())
        self.assertEqual(result['status'],'failed'); self.assertEqual(result['error_type'],'ValueError')

if __name__=='__main__': unittest.main()
