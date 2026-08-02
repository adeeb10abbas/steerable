# Excluded Cosmos canonical confirmation thermal stop

- Status: excluded in full; no episode from this directory enters any estimate.
- Batch started: 2026-08-02T17:10:13Z (approximately, from Isaac log startup).
- Stop recorded: 2026-08-02T17:25Z (approximately); preservation verified at
  2026-08-02T17:26:05Z.
- Raw output:
  `/home/ali/projects/RoboLab/output/v1_cosmos_canonical_interrupted_thermal_gpu1roles`
- Policy role: physical GPU 1, Cosmos3 Edge DROID with decoded futures.
- Simulator role: physical GPU 0, RoboLab Isaac Sim.
- RoboLab commit: `3f55dcf2b7d83fc06cb647a2da7af821038f14c9`.

The physical simulator card reached the preregistered 90 C safety threshold.
The exact container was stopped immediately. `nvidia-smi` reported software
thermal slowdown active and hardware thermal slowdown inactive at the
threshold observation. The card fell to 84 C during container shutdown and 62
C after output preservation.

The interrupted directory contains seven completed left-task episodes with
outcome sequence `SSFSSFS`, followed by three recorded prediction chunks from
an incomplete eighth episode. All files are retained. The definitive canonical
condition must restart from seed 6100 with an absent output directory; it may
not resume this batch or pool its completed episodes.

The thermal event is an operational limitation, not a model failure. It is
hashed as supporting evidence and disclosed in the blog's setup-burden result.
