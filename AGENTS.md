# Agent operating contract

This repository contains multiple research tracks. The active track is the
VLA/WAM language-steerability study. Start every continuation by reading:

1. `docs/VLA_WAM_CONTINUATION.md`
2. `artifacts/vla_wam_shared_v2/continuation_state.json`
3. `docs/VLA_WAM_STEERABILITY_V2_PROTOCOL.md`

Do not infer study state from chat history. Treat the committed artifacts as
the source of truth.

When continuing on the work laptop or a Kubernetes/B200 cluster, also read
`docs/WORK_LAPTOP_B200_HANDOFF.md` before cloning model repositories or
launching a pod. Restore the exact external integration commits from
`handoff/repo_bundles/` and keep raw outputs on persistent cluster storage.

## Non-negotiable study rules

- Never pool raw DROID and RoboTwin success rates.
- Do not edit `artifacts/vla_wam_shared_v2/protocol.json` retroactively.
- Use static episode prompts: no oracle, subtask coach, progress-conditioned
  instruction, or prompt switching.
- Preserve every valid failure and distinguish infrastructure failures from
  model failures.
- Record viewport video for every new pilot episode.
- Use simulator state only for post-action scoring and visualization.
- Run the frozen direct-command gate before any wording sweep.
- A generated future is scored only when the released interface exposes a
  decodable future. Never turn missing future evidence into a zero.
- Keep raw simulator collections and checkpoints outside ordinary Git. Commit
  compact evidence, hashes, manifests, figures, and reproducible renderers.
- Preserve unrelated working-tree changes. External repository dirt is not
  permission to clean or commit it.

## Before running inference

```bash
cd /home/ali/projects/steerable
git status --short
nvidia-smi
python3 tools/validate_vla_wam_v2_protocol.py
```

Confirm that no other policy server or simulator owns the intended GPUs. Then
follow the exact experiment card in `docs/VLA_WAM_CONTINUATION.md`.

## Before stopping or losing model access

1. Stop policy servers, simulators, thermal guards, and containers cleanly.
2. Preserve partial raw outputs; never relabel a partial cell as a failure.
3. Update `artifacts/vla_wam_shared_v2/continuation_state.json` with completed
   cells, invalid attempts, active blockers, and the exact next command.
4. Update `docs/VLA_WAM_CONTINUATION.md` if the queue or decision gate changed.
5. Regenerate derived results and run the validator.
6. Commit one coherent evidence slice. Leave a clean worktree except for files
   explicitly documented as external or intentionally ignored.

If a result changes the authorized next experiment, record that as a disclosed
post-result amendment. Do not rewrite the original freeze.
