# Durable GM execution

The workstation stays online. The work Mac is needed only to provision this
pool, and can go offline at 7 a.m. Eastern on 13 September 2026 after the
independence receipt passes. The personal laptop is outside the execution path.

## Ownership and transport

- Workstation service: `wmf-forecast-autonomy.service` (systemd user service,
  linger enabled), state `/home/ali/wmf-forecast-autonomy-20260912`.
- Workstation checkout: `/home/ali/projects/steerable-forecast-layout-20260912`.
- Git control branch: `codex/forecast-layout-gm-20260912`.
- Cluster result branch: `codex/forecast-layout-gm-20260912-results`.
- Cluster source: `/data/users/ali/vla_wam/src/wmf_ablation_001_20260912`.
- Cluster queue state: `/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/control`.
- Raw evidence stays under `/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912`.

The cluster coordinator is the only results-branch writer. It fetches normal
GitHub control commits, stages exact source commits on the PVC, and publishes
durable job receipts. Pre-created workers claim from that shared storage; later
Kubernetes access is unnecessary. Source commands, model credentials and raw
logs are not automatically published. The repo-scoped deploy key stays on the
cluster; no GM or Codex credentials are copied.

## Releasing and reading jobs

The executable manifest is `cluster_queue.json`. Each released descriptor has
an immutable unique ID, a full source commit reachable from the control branch,
literal argv, a bounded duration, and a role. Generic scientific jobs use role
`any`; a worker's unique name targets a specific worker for diagnostics. The
58 planned scientific blocks remain subject to the specification's qualification
and freeze gates. Infrastructure diagnostics are not behavioral episodes.

Commands can use `{source_root}`, `{job_dir}`, and `{state_dir}` substitutions.
Working directory is the unique job directory. Write intentional compact
deliverables to `{job_dir}/publish/`; publication retains hashes and rejection
reasons, with limits of 16 MiB per file and 64 MiB per job. Full logs and larger
evidence remain on the PVC. Optional combined log tails are capped at 8192 bytes
and should be enabled only for commands with non-sensitive output.

Read `results/wmf_ablation_001_20260912/status.json` on the results branch.
Completed job receipts are in its `jobs/` directory; intentional artifacts and
their manifests are under `results/jobs/<job_id>/`. Count successful receipts
and actual unique GPU identities, not requested workers or queue length.

Shared-storage claim locks are qualified on the actual PVC before release.
Removing an unclaimed job prevents a later claim; existing claims retain their
recorded authority. Global `shutdown: true` interrupts running work and should
only be used after reconciling that consequence. Failed or stale claims are
never silently reclaimed, and no valid scientific cell is automatically rerun.

All workers share a fixed seven-day admission cutoff. Each admitted job has at
most 48 hours, and the publisher remains available through the drain interval.
The deployment receipt records the actual cutoff and pool status. Shutdown and
completion must retain all valid failures, invalid attempts and push receipts.
