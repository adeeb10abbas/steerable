#!/usr/bin/env python3
"""Cluster-only Git/PVC transport. No Kubernetes or model API is used here.

The trusted Git control branch authorizes exact-commit argv jobs. Scientific
qualification belongs to the coordinator producing that manifest. Claims are
never reclaimed automatically. Full logs stay on PVC; published receipts contain
metadata and hashes by default. Log tails require an explicit bounded opt-in;
command arguments are never published automatically.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import tempfile
import threading
import time

NAMESPACE = 'wmf_ablation_001_20260912'
CONTROL_REF = 'refs/remotes/origin/codex/forecast-layout-gm-20260912'
QUEUE_PATH = 'workshops/corl2026_world_models/execution/20260912/autonomy/cluster_queue.json'
RESULTS_BRANCH = 'codex/forecast-layout-gm-20260912-results'
SAFE_ID = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}\Z')
COMMIT = re.compile(r'[0-9a-f]{40}\Z')


def now():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value, immutable=False):
    """Complete fsynced content is installed atomically, including on shared NFS."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = encode(value)
    fd, temporary = tempfile.mkstemp(prefix='.queue-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ValueError(f'immutable descriptor differs: {path.name}') from None
        else:
            os.replace(temporary, path)
    finally:
        try: os.unlink(temporary)
        except FileNotFoundError: pass


def git(repo, *args, accepted=(0,), timeout=60):
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='never')
    result = subprocess.run(['git', '-C', str(repo), *args], capture_output=True,
                            text=True, env=env, timeout=timeout)
    if result.returncode not in accepted:
        # Git diagnostics can contain URLs/credentials; never print them.
        raise RuntimeError(f'Git {args[0]} failed with exit {result.returncode}')
    return result


def safe_id(value):
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value) or value in {'.', '..'}:
        raise ValueError('unsafe job or worker ID')
    return value


def verified_source(repo, commit, control_ref):
    if not isinstance(commit, str) or not COMMIT.fullmatch(commit):
        raise ValueError('source_commit must be exactly 40 lowercase hexadecimal characters')
    exists = git(repo, 'cat-file', '-t', commit, accepted=(0, 1, 128))
    if exists.returncode or exists.stdout.strip() != 'commit':
        raise ValueError('source_commit is not a commit')
    check = git(repo, 'merge-base', '--is-ancestor', commit, control_ref, accepted=(0, 1))
    if check.returncode:
        raise ValueError('source_commit is not reachable from trusted fetched control ref')


def normalize_job(job):
    allowed = {'job_id', 'released', 'source_commit', 'role', 'argv', 'max_wall_seconds', 'publish_log_tail_bytes'}
    if set(job) - allowed:
        raise ValueError('unknown executable descriptor fields')
    ident = safe_id(job.get('job_id'))
    argv = job.get('argv')
    if (not isinstance(argv, list) or not argv or len(argv) > 256
            or any(not isinstance(v, str) or '\0' in v or len(v) > 65536 for v in argv)):
        raise ValueError('argv must be a bounded array of literal strings')
    wall = job.get('max_wall_seconds', 172800)
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not 0 < wall <= 172800:
        raise ValueError('job max_wall_seconds must be positive and at most 172800')
    tail = job.get('publish_log_tail_bytes', 0)
    if type(tail) is not int or not 0 <= tail <= 8192:
        raise ValueError('publish_log_tail_bytes must be an integer from 0 through 8192')
    return {'schema_version': 'wmf-cluster-job-v1', 'namespace': NAMESPACE,
            'job_id': ident, 'released': True, 'source_commit': job.get('source_commit'),
            'role': safe_id(job.get('role', 'any')), 'argv': argv, 'max_wall_seconds': wall,
            'publish_log_tail_bytes': tail}


def verify_worktree(path, commit):
    if git(path, 'rev-parse', 'HEAD').stdout.strip() != commit:
        raise ValueError('immutable source worktree HEAD changed')
    if git(path, 'diff', '--no-ext-diff', '--quiet', 'HEAD', accepted=(0, 1)).returncode:
        raise ValueError('immutable source worktree tracked content changed')


def stage_queue(repo, state_dir, control_ref, manifest):
    state = Path(state_dir).resolve(); repo = Path(repo).resolve()
    if (manifest.get('schema_version') != 'wmf-cluster-queue-v1'
            or manifest.get('namespace') != NAMESPACE
            or type(manifest.get('shutdown')) is not bool
            or not isinstance(manifest.get('jobs'), list)):
        raise ValueError('invalid queue schema, namespace or shutdown flag')
    if set(manifest) - {'schema_version', 'namespace', 'shutdown', 'jobs'}:
        raise ValueError('unknown queue fields')
    jobs = []; seen = set()
    for raw in manifest['jobs']:
        if not isinstance(raw, dict): raise ValueError('job must be an object')
        ident = safe_id(raw.get('job_id'))
        if ident in seen: raise ValueError('duplicate job ID')
        seen.add(ident)
        if type(raw.get('released')) is not bool: raise ValueError('released must be boolean')
        old = state/'jobs'/ident/'descriptor.json'
        if not raw['released']:
            if old.exists(): raise ValueError('immutable staged descriptor cannot be rewritten as unreleased; remove it from active queue instead')
            continue
        job = normalize_job(raw)
        verified_source(repo, job['source_commit'], control_ref)
        if old.exists() and old.read_bytes() != encode(job):
            raise ValueError(f'immutable descriptor differs: {ident}')
        jobs.append(job)
    control_commit = git(repo, 'rev-parse', control_ref+'^{commit}').stdout.strip()
    # Validate the complete update before staging any executable descriptor.
    for job in jobs:
        source = state/'sources'/job['source_commit']
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            git(repo, 'worktree', 'add', '--detach', str(source), job['source_commit'])
        verify_worktree(source, job['source_commit'])
        atomic_json(state/'jobs'/job['job_id']/'descriptor.json', job, immutable=True)
    control = {'namespace': NAMESPACE, 'control_commit': control_commit,
               'shutdown': manifest['shutdown'], 'active_job_ids': [j['job_id'] for j in jobs]}
    atomic_json(state/'control.json', control)
    return control


def claim_job(job_dir, worker_id):
    worker_id = safe_id(worker_id); claim = Path(job_dir)/'claim'
    try: claim.mkdir()
    except FileExistsError: return False
    atomic_json(claim/'owner.json', {'worker_id': worker_id, 'claimed_at': now(),
                                  'claimed_unix': time.time(), 'worker_pid': os.getpid()})
    return True


def control_state(state):
    path = Path(state)/'control.json'
    return read_json(path) if path.exists() else {'shutdown': False, 'active_job_ids': []}


def file_identity(path):
    digest = hashlib.sha256(); count = 0
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024*1024):
            digest.update(chunk); count += len(chunk)
    return {'bytes': count, 'sha256': digest.hexdigest()}


def terminate_owned(proc, grace):
    # start_new_session below makes this PGID exclusively this job's descendants.
    try: os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError: pass
    try: proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        proc.wait()
    # Reap any background descendants left behind when the direct child exited.
    try: os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError: pass


def worker_once(repo, state_dir, worker_id, role='any', *, max_wall_seconds=172800,
                poll_seconds=2, terminate_grace_seconds=10, stop_event=None):
    state = Path(state_dir).resolve(); safe_id(worker_id); safe_id(role)
    stop_event = stop_event or threading.Event()
    control = control_state(state)
    if control['shutdown'] or stop_event.is_set() or max_wall_seconds <= 0: return 0
    for ident in control['active_job_ids']:
        directory = state/'jobs'/safe_id(ident); job = read_json(directory/'descriptor.json')
        if role != 'any' and job['role'] not in {'any', role}: continue
        if not claim_job(directory, worker_id): continue
        started = time.monotonic(); proc = None; status = 'failed'; error_type = None
        argv = [v.replace('{source_root}', str(state/'sources'/job['source_commit']))
                .replace('{job_dir}', str(directory)).replace('{state_dir}', str(state)) for v in job['argv']]
        result = {'schema_version': 'wmf-cluster-result-v1', 'namespace': NAMESPACE,
                  'job_id': ident, 'worker_id': worker_id, 'source_commit': job['source_commit'],
                  'descriptor_sha256': hashlib.sha256(encode(job)).hexdigest(),
                  'started_at': now(), 'argv': argv, 'job_dir': str(directory)}
        try:
            with (directory/'stdout.log').open('wb') as stdout, (directory/'stderr.log').open('wb') as stderr:
                verify_worktree(state/'sources'/job['source_commit'], job['source_commit'])
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
                proc = subprocess.Popen(argv, cwd=directory, stdout=stdout, stderr=stderr,
                                        env=env, start_new_session=True)
                deadline = started + min(max_wall_seconds, job['max_wall_seconds'])
                while True:
                    atomic_json(directory/'heartbeat.json', {'worker_id': worker_id,
                                'worker_pid': os.getpid(), 'child_pid': proc.pid,
                                'at': now(), 'unix': time.time()})
                    if proc.poll() is not None:
                        status = 'succeeded' if proc.returncode == 0 else 'failed'; break
                    if stop_event.is_set() or control_state(state)['shutdown']:
                        status = 'interrupted'; terminate_owned(proc, terminate_grace_seconds); break
                    if time.monotonic() >= deadline:
                        status = 'timed_out'; terminate_owned(proc, terminate_grace_seconds); break
                    stop_event.wait(min(poll_seconds, max(.001, deadline-time.monotonic())))
                terminate_owned(proc, terminate_grace_seconds)
                stdout.flush(); stderr.flush(); os.fsync(stdout.fileno()); os.fsync(stderr.fileno())
        except Exception as error:
            error_type = type(error).__name__
            if proc is not None: terminate_owned(proc, terminate_grace_seconds)
        finally:
            result.update(status=status, returncode=proc.returncode if proc else None,
                          error_type=error_type, ended_at=now(), wall_seconds=time.monotonic()-started,
                          child_pid=proc.pid if proc else None, child_reaped=proc is None or proc.returncode is not None,
                          stdout=file_identity(directory/'stdout.log'), stderr=file_identity(directory/'stderr.log'))
            atomic_json(directory/'result.json', result, immutable=True)
        return 1
    return 0


def snapshot(state_dir, stale_after_seconds=180):
    state = Path(state_dir); rows = []
    for directory in sorted((state/'jobs').glob('*')) if (state/'jobs').exists() else []:
        if not (directory/'descriptor.json').exists(): continue
        job = read_json(directory/'descriptor.json')
        row = {'job_id': job['job_id'], 'source_commit': job['source_commit'],
               'descriptor_sha256': hashlib.sha256(encode(job)).hexdigest(), 'status': 'staged'}
        if (directory/'result.json').exists():
            result = read_json(directory/'result.json')
            for key in ('status','returncode','worker_id','started_at','ended_at','wall_seconds',
                        'error_type','child_pid','child_reaped','stdout','stderr'):
                row[key] = result.get(key)
            budget = job['publish_log_tail_bytes']
            if budget:
                row['log_tails'] = {}
                for name, count in [('stderr', budget//2), ('stdout', budget-budget//2)]:
                    with (directory/(name+'.log')).open('rb') as stream:
                        stream.seek(max(0, os.fstat(stream.fileno()).st_size-count))
                        row['log_tails'][name] = stream.read(count).decode('utf-8', errors='ignore')
        elif (directory/'claim').exists():
            heartbeat = directory/'heartbeat.json'; owner = directory/'claim/owner.json'
            claim = read_json(owner) if owner.exists() else {}
            beat = read_json(heartbeat) if heartbeat.exists() else {}
            timestamp = beat.get('unix', claim.get('claimed_unix', (directory/'claim').stat().st_mtime))
            row.update(status='stale_claim_requires_decision' if time.time()-timestamp > stale_after_seconds else 'claimed',
                       worker_id=claim.get('worker_id'), heartbeat=beat)
        rows.append(row)
    return {'schema_version': 'wmf-cluster-status-v1', 'namespace': NAMESPACE,
            'control': control_state(state), 'jobs': rows,
            'automatic_claim_recovery': False, 'default_published_log_tail_bytes': 0,
            'maximum_combined_log_tail_bytes_per_job': 8192}


def copy_publish_artifacts(job_dir, destination, per_file_limit=16*1024*1024, total_limit=64*1024*1024):
    """Copy only intentional bounded regular files; retain every rejection."""
    source = Path(job_dir)/'publish'; destination = Path(destination)
    report = {'files': [], 'errors': [], 'per_file_limit_bytes': per_file_limit,
              'total_limit_bytes': total_limit, 'copied_bytes': 0}
    if not source.exists() and not source.is_symlink(): return report
    if source.is_symlink() or not source.is_dir():
        report['errors'].append({'path': '.', 'reason': 'symlink_rejected' if source.is_symlink() else 'not_directory'})
        return report
    if destination.is_symlink():
        report['errors'].append({'path': '.', 'reason': 'destination_escape_rejected'}); return report
    destination.mkdir(parents=True, exist_ok=True)
    for folder, directories, names in os.walk(source, followlinks=False):
        directories.sort(); names.sort()
        for name in list(directories):
            candidate = Path(folder)/name
            if candidate.is_symlink():
                directories.remove(name)
                report['errors'].append({'path': str(candidate.relative_to(source)), 'reason': 'symlink_rejected'})
        for name in names:
            candidate = Path(folder)/name; relative = candidate.relative_to(source)
            error = None; data = None
            try:
                info = candidate.lstat()
                if stat.S_ISLNK(info.st_mode): error = 'symlink_rejected'
                elif not stat.S_ISREG(info.st_mode): error = 'non_regular_file_rejected'
                elif info.st_size > per_file_limit: error = 'per_file_limit_exceeded'
                elif report['copied_bytes']+info.st_size > total_limit: error = 'total_limit_exceeded'
                else:
                    fd = os.open(candidate, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
                    with os.fdopen(fd, 'rb') as stream:
                        before = os.fstat(stream.fileno()); data = stream.read(per_file_limit+1)
                        after = os.fstat(stream.fileno())
                    if len(data) > per_file_limit: error = 'per_file_limit_exceeded'
                    elif not stat.S_ISREG(before.st_mode): error = 'non_regular_file_rejected'
                    elif (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns) or len(data) != after.st_size:
                        error = 'file_changed_during_copy'
                    elif report['copied_bytes']+len(data) > total_limit: error = 'total_limit_exceeded'
            except OSError:
                error = 'file_read_failed'
            if error:
                report['errors'].append({'path': str(relative), 'reason': error}); continue
            target = destination/relative
            if not target.resolve().is_relative_to(destination.resolve()):
                report['errors'].append({'path': str(relative), 'reason': 'destination_escape_rejected'}); continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() or (target.exists() and target.read_bytes() != data):
                report['errors'].append({'path': str(relative), 'reason': 'immutable_published_file_changed'}); continue
            if not target.exists():
                fd, temporary = tempfile.mkstemp(prefix='.publish-', dir=target.parent)
                try:
                    with os.fdopen(fd, 'wb') as stream:
                        stream.write(data); stream.flush(); os.fsync(stream.fileno())
                    os.link(temporary, target)
                finally: os.unlink(temporary)
            report['files'].append({'path': str(relative), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            report['copied_bytes'] += len(data)
    return report


def publish_results(repo, state_dir, control_ref, results_branch):
    state = Path(state_dir).resolve(); repo = Path(repo).resolve()
    git(repo, 'check-ref-format', '--branch', results_branch)
    tree = state/'results-git'; remote_ref = 'refs/remotes/origin/'+results_branch
    remote = git(repo, 'ls-remote', '--heads', 'origin', 'refs/heads/'+results_branch).stdout.strip()
    if remote:
        git(repo, 'fetch', '--no-tags', 'origin', 'refs/heads/'+results_branch+':'+remote_ref)
    if not tree.exists():
        exists = git(repo, 'show-ref', '--verify', '--quiet', 'refs/heads/'+results_branch, accepted=(0,1))
        args = ['worktree','add'] + ([] if exists.returncode == 0 else ['-b', results_branch])
        args += [str(tree), results_branch if exists.returncode == 0 else (remote_ref if remote else control_ref)]
        git(repo, *args)
    if git(tree,'branch','--show-current').stdout.strip() != results_branch:
        raise ValueError('results worktree branch mismatch')
    if remote:
        ancestor = git(tree,'merge-base','--is-ancestor',remote_ref,'HEAD',accepted=(0,1))
        if ancestor.returncode:
            git(tree,'merge','--ff-only',remote_ref)
    report = snapshot(state)
    base = tree/'results'/NAMESPACE
    atomic_json(base/'status.json', report)
    for row in report['jobs']:
        if row['status'] in {'succeeded','failed','timed_out','interrupted'}:
            atomic_json(base/'jobs'/(row['job_id']+'.json'), row, immutable=True)
            published = tree/'results/jobs'/row['job_id']
            if not published.resolve().is_relative_to(tree.resolve()):
                raise ValueError('artifact destination escapes results worktree')
            manifest_path = published/'publish_manifest.json'
            if not manifest_path.exists():
                artifacts = copy_publish_artifacts(state/'jobs'/row['job_id'], published/'publish')
                atomic_json(manifest_path, artifacts, immutable=True)
    git(tree,'add','--','results/'+NAMESPACE)
    if (tree/'results/jobs').exists(): git(tree,'add','--','results/jobs')
    changed = git(tree,'diff','--cached','--quiet',accepted=(0,1)).returncode
    if changed:
        git(tree,'-c','user.name=WMF cluster coordinator','-c','user.email=wmf-cluster@users.noreply.github.com',
            'commit','-m','Record cluster queue receipts')
    # Respect origin.pushurl/GIT_SSH_COMMAND supplied at bootstrap; never force.
    git(tree,'push','origin','HEAD:refs/heads/'+results_branch)
    return report


@contextmanager
def exclusive_process(state, name):
    lock = Path(state)/name; lock.parent.mkdir(parents=True, exist_ok=True)
    try: lock.mkdir()
    except FileExistsError: raise RuntimeError(f'{name} exists; operator must inspect prior process') from None
    atomic_json(lock/'owner.json', {'pid': os.getpid(), 'started_at': now()})
    try: yield
    finally:
        (lock/'owner.json').unlink(); lock.rmdir()


def fetch_control(repo, control_ref, queue_path):
    prefix = 'refs/remotes/origin/'
    if not control_ref.startswith(prefix): raise ValueError('control-ref must be an origin remote-tracking ref')
    if Path(queue_path).is_absolute() or '..' in Path(queue_path).parts:
        raise ValueError('queue-path must be repository-relative')
    branch = control_ref[len(prefix):]; git(repo,'check-ref-format','--branch',branch)
    git(repo,'fetch','--no-tags','origin','refs/heads/'+branch+':'+control_ref)
    return json.loads(git(repo,'show',control_ref+':'+queue_path).stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['coordinator','worker'])
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--state-dir',type=Path,required=True)
    parser.add_argument('--control-ref',default=CONTROL_REF)
    parser.add_argument('--queue-path',default=QUEUE_PATH)
    parser.add_argument('--results-branch',default=RESULTS_BRANCH)
    parser.add_argument('--worker-id',default=os.environ.get('HOSTNAME','worker'))
    parser.add_argument('--role',default='any')
    parser.add_argument('--max-wall-seconds',type=float,default=172800)
    parser.add_argument('--poll-seconds',type=float,default=30)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if not 0 < args.max_wall_seconds <= 172800 or args.poll_seconds <= 0:
        parser.error('positive polling and wall limit at most 172800 required')
    state=args.state_dir.resolve(); state.mkdir(parents=True,exist_ok=True)
    stop=threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:stop.set())
    deadline=time.monotonic()+args.max_wall_seconds
    lock='coordinator.lock' if args.mode=='coordinator' else 'worker-'+safe_id(args.worker_id)+'.lock'
    with exclusive_process(state,lock):
        while not stop.is_set() and time.monotonic()<deadline:
            if args.mode=='coordinator':
                try:
                    manifest=fetch_control(args.repo,args.control_ref,args.queue_path)
                    stage_queue(args.repo,state,args.control_ref,manifest)
                    report=publish_results(args.repo,state,args.control_ref,args.results_branch)
                    atomic_json(state/'coordinator_status.json',{'at':now(),'status':'ok','jobs':len(report['jobs'])})
                    if manifest['shutdown']: break
                except Exception as error:
                    atomic_json(state/'coordinator_status.json',{'at':now(),'status':'error','error_type':type(error).__name__})
                    # Leave all claims/results intact. A later trusted control update may repair readiness.
            else:
                worker_once(args.repo,state,args.worker_id,args.role,max_wall_seconds=deadline-time.monotonic(),stop_event=stop)
                if control_state(state)['shutdown']: break
            if args.once: break
            stop.wait(min(args.poll_seconds,max(0,deadline-time.monotonic())))
    return 0

if __name__=='__main__': raise SystemExit(main())
