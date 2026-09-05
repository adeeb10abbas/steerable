# Online correction V4 continuation

**Status:** `QUALIFYING` / `IMPLEMENTING` — prospective design freeze committed; no V4 policy inference has run.

This document is the machine-readable continuation companion to `docs/online_correction_v4/README.md`. It records the current implementation boundary, blocked families, and exact next commands. Do not infer study state from chat history; use the committed artifacts under `artifacts/online_correction_v4/`.

## Authority

| Source | Role |
| --- | --- |
| `docs/online_correction_v4/` numbered docs | Scientific design, metrics, runbook |
| `docs/online_correction_v4/campaign.json` | Allocation and defaults |
| `artifacts/online_correction_v4/` | Prospective freeze artifacts (this campaign) |
| `artifacts/online_correction_v4/continuation_state.json` | Active status and hashes |
| `artifacts/vla_wam_shared_v2/protocol.json` | Immutable historical V2 protocol |
| `artifacts/vla_wam_shared_v3/protocol.json` | Immutable historical V3 protocol |

Historical V2/V3 evidence remains closed. V4 adds a separately identified online-correction campaign with a narrow simulator-state extension documented in `docs/online_correction_v4/00_EXISTING_EVIDENCE_AND_SCOPE.md`.

## Current status

| Field | Value |
| --- | --- |
| Lifecycle | `QUALIFYING` |
| Implementation | `IMPLEMENTING` |
| Release | `NOT_RELEASED` |
| Policy episodes executed | `0` |
| Confirmatory queue rows | `17664` (frozen in `queue.jsonl`) |
| Engineering pilots | `240` excluded; not in main queue |

### Family disposition (pre-runtime)

| Family | Status | Block reason (summary) |
| --- | --- | --- |
| C1 | `IMPLEMENTING` | Runner, geometry, checkpoint receipts pending |
| C2 | `BLOCKED_SETUP` | Primary H selectivity requires **verified common-prefix replay** |
| C3 | `IMPLEMENTING` | Depends on C1 controls; runtime pending |
| C4 | `IMPLEMENTING` | Depends on C1/C3 controls; runtime pending |
| C5 | `IMPLEMENTING` | Vertical fixture geometry receipt pending |
| C6 | `IMPLEMENTING` | Containment fixture geometry receipt pending |
| C7 | `IMPLEMENTING` | Object-pair fixture receipt pending |
| C8 | `BLOCKED_RUNTIME` | GR00T N1.7 Bridge/WidowX checkpoint, adapter, and SimplerEnv fixture names not verified |

`runtime_lock.template.json` remains intentionally unreleased. Launch-critical runtime manifests must not be treated as released until qualification receipts pass.

## Exact next commands

Run from the repository root:

```bash
python3 tools/online_correction_v4.py validate
python3 tools/build_online_correction_v4_freeze.py --out artifacts/online_correction_v4
python3 tools/validate_online_correction_v4.py
python3 -m unittest discover -s tests -p 'test_online_correction_v4*.py'
python3 tools/validate_vla_wam_v2_protocol.py
python3 tools/validate_vla_wam_v3_protocol.py
```

After runtime modules qualify (not yet):

```bash
python3 tools/online_correction_v4.py manifest --out /persistent/v4/planned_episodes.jsonl
python3 tools/online_correction_v4.py release-check --lock /persistent/v4/runtime_lock.json --manifest /persistent/v4/planned_episodes.jsonl
```

Use the actual persistent mount bound in the qualified runtime lock; do not write large robot traces to ordinary Git.

## Freeze artifact index

| Artifact | Purpose |
| --- | --- |
| `protocol.json` | Frozen estimands, families, stopping rule |
| `prompt_manifest.json` | Exact symbolic prompts and SHA-256; `second_stack` physical names unresolved |
| `motion_manifest.json` | Registered profiles; calibration scales pending geometry gate |
| `scoring_manifest.json` | Thresholds and analysis registry; `D_cap` pending per fixture |
| `seed_manifest.json` | Reserved env/policy seeds and historical collision audit |
| `queue.jsonl` | 17,664 confirmatory episode identities |
| `queue_manifest.json` | Row counts and queue byte hash |
| `frozen_analysis_manifest.json` | Primary contrasts and bootstrap registry |
| `gate_report.json` | Per-family gates; receipts pending except seed audit |
| `historical_protocol_ledger.json` | V2/V3 protocol byte hashes for fail-closed checks |
| `continuation_state.json` | This campaign's machine-readable state |

## Verification boundary

Passing `validate_online_correction_v4.py` certifies:

- deterministic queue and prompt resolution
- historical protocol bytes unchanged vs the ledger
- seed namespace disjoint from scanned repository artifact seeds
- C2/C8 block reasons and no fake-pass runtime receipts
- no release claimed in freeze artifacts

It does **not** certify simulator timing, checkpoint access, physical feasibility, or empirical findings. Those require the qualification gates in `docs/online_correction_v4/03_AGENT_EXECUTION_RUNBOOK.md`.

## What is explicitly not done

- No V4 policy server launch
- No geometry/checkpoint/cluster qualification receipts (except historical seed collision audit)
- No confirmatory results or paper tables
- No modification of V2/V3 protocol files

When the runner, scheduler, fixtures, and receipts qualify, update `continuation_state.json`, bind `runtime_lock.json` on persistent storage, and advance family status through the documented gate graph — not by editing the historical freeze in place without a disclosed amendment.
