# Supervisor review fixes

12 September 2026. This report supersedes the original finite-retry behavior
described with commit `8c21c8a`. No real Codex request or service deployment was
used to implement or test these changes.

## Unattended retry behavior

Recognized transient network, service, rate-limit and usage-limit errors now
retry indefinitely. The default delay is 10, 20, 40, 80, 160, then 300 seconds
for subsequent failures. The delay remains capped at 300 seconds; the failure
count is informational and resets after a normally completed turn. There is
no production failure-count cutoff. `next_retry_at` remains durable and is
honored on restart. The obsolete `--max-transient-failures` option was removed.

## Errors remain supervised

Unknown or mismatched session identity, nontransient execution errors,
unrecoverable checkpoints and unresolved prior children enter `needs_attention`.
The supervisor remains alive, keeps its single-instance lock, and polls for
operator resolution and safe stop receipts. `--attention-poll` defaults to ten
seconds. It never silently starts another session after an explicit resume
failure. Signal-driven shutdown remains available and checkpoints `stopped`.

After correcting the cause, an operator may atomically write
`operator_resolution.json` in the state directory:

```json
{"action":"retry","thread_id":"THE-UNCHANGED-RECORDED-UUID"}
```

The UUID must match the checkpoint exactly. `null` is permitted only when the
checkpoint has no observed UUID. Applied and rejected resolution files are
retained with distinct timestamped names. A valid resolution retries the
recorded session; it cannot bypass an unresolved prior child. A safe completion
or needs-input receipt can terminate the watchdog cleanly. Internal errors
themselves no longer produce a terminal needs-input status or clean exit.

## Prior-child exclusion

Every restoration and every new launch checks retained unresolved child
identity. The checkpoint preserves `unresolved_child_pid` and
`unresolved_child_start_identity`; Linux identities combine boot ID and process
start ticks to distinguish PID reuse. Where that evidence is unavailable, the
supervisor conservatively blocks while the PID is alive. Legacy
`previous_child_pid` checkpoints are honored, so a previous park cannot erase
the guard. `--continue-after-input` and operator resolution both remain subject
to that guard. Unknown prior processes are never signaled or killed.

## Receipt contract

Both terminal receipts require `safe_to_stop` to be exactly `true`.

- `completion.json`: `status: "complete"`, nonempty summary, and a nonempty list
  of nonempty evidence strings.
- `needs_input.json`: `status: "needs_input"` and a nonempty reason.

Missing, false, mismatched or malformed receipt fields are rejected and
preserved. Rejection requests corrective continuation during ordinary agent
operation and does not abandon the attention watchdog. Scientific claim gates
and durable monitoring assertions remain the agent's responsibility.

## Validation

All 16 supervisor tests pass using only temporary fake executables. The full
current workshop suite passes all 68 tests. `git diff --check` passes.

The review regressions exercise 14 consecutive transient failures followed by
recovery, capped delays, a future retry deadline surviving restart, a live
watchdog after unknown/mismatched UUID errors, explicit same-UUID operator
resolution, safe marker shutdown from the watchdog, strict needs-input status,
and legacy orphan PID preservation across both `--continue-after-input` and an
explicit retry request. Existing tests continue covering logs, identity recovery,
receipt rejection, lock ownership and SIGTERM forwarding to the owned child.

```bash
python3 -m unittest discover -s workshops/corl2026_world_models/tests -p test_autonomous_supervisor.py
python3 -m unittest discover -s workshops/corl2026_world_models/tests
```
