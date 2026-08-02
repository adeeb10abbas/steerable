# Excluded Cosmos hierarchy oracle pilot

The frozen setup-only pilot used the left matched task, episode seed 5190,
five-action replanning, the Cosmos3 Edge DROID action-only server, and the
`predicate_oracle` controller. Its output is retained at:

`/home/ali/projects/RoboLab/output/pilot_cosmos_h5_oracle_5190`

It is excluded from every confirmation numerator, denominator, interval, and
paired comparison. Its sole purpose was to verify the privileged controller
before seeds 7100--7104.

The pilot ran all 450 environment steps and failed the requested left-side
predicate. The persisted `timing.command_history` contains exactly 90 entries,
one every five steps. It selected:

- `grasp`: 67 requests;
- `spatial_move`: 23 requests;
- `release`: 0 requests, because the requested-side predicate was never true.

The controller switched between grasp and spatial-move phases six times as
the simulator contact predicate changed. There were 23 history entries with
`held=true` and none with `at_requested_side=true`. This verifies that the
oracle is live, state-dependent, direction-aware, and persisted rather than a
static prompt alias. It does not show that Cosmos can reliably execute the
selected commands.

The episode took 243.114 seconds wall time, including 160.338 seconds waiting
for 90 action-only policy requests. The accompanying thermal log
`cosmos_h5_oracle_pilot_5190_thermal.jsonl` has a complete monitor lifecycle,
no cooldown, and no emergency event.
