"""Real filesystem/Git/subprocess checks for the cluster-only transport."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

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
    def test_withdrawal_after_worker_snapshot_prevents_unclaimed_launch(self):
        self.stage(self.manifest([self.job()]))
        original=self.q.claim_job
        def withdraw_then_claim(directory, worker_id):
            self.stage(self.manifest())
            return original(directory,worker_id)
        self.q.claim_job=withdraw_then_claim
        self.assertEqual(self.q.worker_once(self.repo,self.state,'worker-1',poll_seconds=.02),0)
        path=self.state/'jobs/diagnostic-001'
        self.assertFalse((path/'claim').exists())
        self.assertFalse((path/'stdout.log').exists())

    def test_claim_before_withdrawal_keeps_explicit_claim_authority(self):
        staged=self.stage(self.manifest([self.job()]))
        original=self.q.claim_job
        def claim_then_withdraw(directory,worker_id):
            claimed=original(directory,worker_id)
            self.stage(self.manifest())
            return claimed
        self.q.claim_job=claim_then_withdraw
        self.assertEqual(self.q.worker_once(self.repo,self.state,'worker-1',poll_seconds=.02),1)
        path=self.state/'jobs/diagnostic-001'
        owner=json.loads((path/'claim/owner.json').read_text())
        self.assertEqual(owner['control_commit'],self.commit)
        self.assertEqual(owner['control_generation'],staged['control_generation'])
        self.assertEqual(owner['descriptor_sha256'],self.q.file_identity(path/'descriptor.json')['sha256'])
        self.assertEqual(json.loads((path/'result.json').read_text())['status'],'succeeded')

    def test_shared_release_lock_serializes_control_change_and_claim(self):
        self.stage(self.manifest([self.job()]))
        ready=self.root/'claim-ready'
        code='''import importlib.util, pathlib, sys
spec=importlib.util.spec_from_file_location('queue',sys.argv[1]); q=importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
pathlib.Path(sys.argv[3]).write_text('ready')
print(q.claim_job(pathlib.Path(sys.argv[2])/'jobs/diagnostic-001','other-process'),flush=True)
'''
        with self.q.shared_release_lock(self.state):
            proc=subprocess.Popen([sys.executable,'-c',code,str(MODULE),str(self.state),str(ready)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            self.addCleanup(lambda: proc.kill() if proc.poll() is None else None)
            deadline=time.monotonic()+3
            while not ready.exists() and time.monotonic()<deadline: time.sleep(.01)
            self.assertTrue(ready.exists()); self.assertIsNone(proc.poll())
            control=self.q.control_state(self.state); control['active_job_ids']=[]; control['control_generation']+=1
            self.q.atomic_json(self.state/'control.json',control)
        stdout,stderr=proc.communicate(timeout=3)
        self.assertEqual(proc.returncode,0,stderr); self.assertEqual(stdout.strip(),'False')
        self.assertFalse((self.state/'jobs/diagnostic-001/claim').exists())

    def test_daemon_lock_releases_after_process_death_but_job_claim_does_not(self):
        self.stage(self.manifest([self.job()]))
        self.assertTrue(self.q.claim_job(self.state/'jobs/diagnostic-001','original-worker'))
        ready=self.root/'lock-ready'
        code='''import importlib.util, pathlib, sys, time
spec=importlib.util.spec_from_file_location('queue',sys.argv[1]); q=importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
with q.exclusive_process(sys.argv[2],'coordinator.lock'):
 pathlib.Path(sys.argv[3]).write_text('ready')
 time.sleep(30)
'''
        proc=subprocess.Popen([sys.executable,'-c',code,str(MODULE),str(self.state),str(ready)])
        self.addCleanup(lambda: proc.kill() if proc.poll() is None else None)
        deadline=time.monotonic()+3
        while not ready.exists() and time.monotonic()<deadline: time.sleep(.01)
        self.assertTrue(ready.exists())
        with self.assertRaises(RuntimeError):
            with self.q.exclusive_process(self.state,'coordinator.lock'): self.fail('duplicate coordinator')
        proc.kill(); proc.wait(timeout=3)
        with self.q.exclusive_process(self.state,'coordinator.lock'):
            self.assertTrue((self.state/'coordinator.lock').is_file())
            self.assertFalse(self.q.claim_job(self.state/'jobs/diagnostic-001','new-worker'))
        self.assertTrue((self.state/'coordinator.lock').is_file())

    def test_controller_accepts_seven_days_but_jobs_remain_at_most_two_days(self):
        command=[sys.executable,str(MODULE),'worker','--repo',str(self.repo),'--state-dir',str(self.state),'--once']
        result=subprocess.run(command+['--max-wall-seconds','604800'],capture_output=True,text=True,timeout=3)
        self.assertEqual(result.returncode,0,result.stderr)
        result=subprocess.run(command+['--max-wall-seconds','604801'],capture_output=True,text=True,timeout=3)
        self.assertEqual(result.returncode,2)
        job=self.job(); job['max_wall_seconds']=604800
        with self.assertRaisesRegex(ValueError,'172800'): self.q.normalize_job(job)

    def test_claimed_job_finishes_own_budget_after_controller_poll_window_expires(self):
        job=self.job(); job['max_wall_seconds']=1
        job['argv']=[sys.executable,'-c','import time; time.sleep(.2); print("finished within job budget")']
        self.stage(self.manifest([job]))
        result=subprocess.run([sys.executable,str(MODULE),'worker','--repo',str(self.repo),'--state-dir',str(self.state),
                               '--max-wall-seconds','.1','--poll-seconds','.01','--once'],capture_output=True,text=True,timeout=4)
        self.assertEqual(result.returncode,0,result.stderr)
        path=self.state/'jobs/diagnostic-001'
        self.assertEqual(json.loads((path/'result.json').read_text())['status'],'succeeded')

    def test_shared_absolute_admission_cutoff_blocks_late_claim(self):
        control=self.q.stage_queue(self.repo,self.state,self.ref,self.manifest([self.job()]),admission_deadline_unix=time.time()-.1)
        self.assertLess(control['admission_deadline_unix'],time.time())
        self.assertFalse(self.q.claim_job(self.state/'jobs/diagnostic-001','late-worker'))
        self.assertFalse((self.state/'jobs/diagnostic-001/claim').exists())

    def test_coordinator_keeps_publishing_when_admission_snapshot_crosses_cutoff(self):
        job=self.job(); job['max_wall_seconds']=1
        job['argv']=[sys.executable,'-c','import time; time.sleep(.25); print("late receipt")']
        cutoff=time.time()+.15; observed=[]; threads=[]
        def publish(*_args):
            report=self.q.snapshot(self.state)
            if not threads:
                thread=threading.Thread(target=self.q.worker_once,args=(self.repo,self.state,'drain-worker'),kwargs={'poll_seconds':.01})
                threads.append(thread); thread.start()
                deadline=time.monotonic()+2
                while not (self.state/'jobs/diagnostic-001/claim').exists() and time.monotonic()<deadline: time.sleep(.005)
                time.sleep(max(0,cutoff-time.time())+.02)
            observed.append((time.time(),report['jobs'][0]['status']))
            return report
        argv=[str(MODULE),'coordinator','--repo',str(self.repo),'--state-dir',str(self.state),
              '--max-wall-seconds','.8','--admission-deadline-unix',str(cutoff),'--poll-seconds','.01']
        with mock.patch.object(sys,'argv',argv), mock.patch.object(self.q,'fetch_control',return_value=self.manifest([job])) as fetch, mock.patch.object(self.q,'publish_results',side_effect=publish):
            self.assertEqual(self.q.main(),0)
            self.assertEqual(fetch.call_count,1)
        for thread in threads: thread.join(timeout=2)
        self.assertEqual(observed[0][1],'staged')
        self.assertEqual(observed[-1][1],'succeeded')
        self.assertGreater(observed[-1][0],cutoff)
        self.assertEqual(json.loads((self.state/'coordinator_status.json').read_text())['phase'],'draining')
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
