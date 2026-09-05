# Online correction V4 continuation

**Status:** `QUALIFYING` / `IMPLEMENTING` — prospective design freeze; no V4 policy inference has run.

This document is the machine-readable continuation companion to `docs/online_correction_v4/README.md`. Do not infer study state from chat history; use the committed artifacts under `artifacts/online_correction_v4/`.

## Authority

| Source | Role |
| --- | --- |
| `docs/online_correction_v4/` numbered docs | Scientific design, metrics, runbook |
| `docs/online_correction_v4/campaign.json` | Allocation and defaults |
| `docs/online_correction_v4/design_validation.json` | Design-only planning validation (unchanged semantics) |
| `artifacts/online_correction_v4/freeze_manifest.json` | Index of all freeze artifact hashes |
| `artifacts/online_correction_v4/continuation_state.json` | Active status, hashes, and next commands |
| `artifacts/online_correction_v4/` | Prospective freeze artifacts (queue, manifests, gate report) |

Historical V2/V3 protocol files remain immutable.

## Hash semantics

| Hash field | Meaning |
| --- | --- |
| `planning_manifest_sha256` | Pre-enrichment inventory from `tools/online_correction_v4.py build_manifest` (matches `design_validation.json` when campaign unchanged) |
| `frozen_queue_sha256` | Enriched `queue.jsonl` bytes including prompt text, hashes, and `queue_row_kind=new_episode` |
| `generation_parent_commit` | Git HEAD when the freeze builder last ran — **not** the commit containing freeze artifacts |

After merge, record a `git_receipt.freeze_commit` binding the commit that contains `artifacts/online_correction_v4/`.

## Prompt identity (C2 counterbalance)

Semantic prompt identity is **`prompt_id`**, not `prompt_sha256`. Under C2 reference binding, identical UTF-8 prompt text can name different physical A/B bowl identities when counterbalance swaps which color is “A”. The same words therefore may share one `prompt_sha256` while mapping to multiple `prompt_id` values.

- `prompt_sha256 = sha256(utf8(prompt_text))` — identical hashes iff byte-identical resolved text; different text must not share a hash.
- Episode-level binding uses `episode_id`, `prompt_id`, `prompt_text`, and `prompt_sha256`.
- Analysis must **never** aggregate, join, or key contrasts on `prompt_sha256` alone.
- `prompt_manifest.json` and `frozen_analysis_manifest.json` document this rule explicitly.

## Current status

| Field | Value |
| --- | --- |
| Lifecycle | `QUALIFYING` |
| Implementation | `IMPLEMENTING` |
| Release | `NOT_RELEASED` |
| Policy episodes executed | `0` |
| Confirmatory queue rows | `17664` |

### Family disposition

| Family | Disposition | Status |
| --- | --- | --- |
| C1, C3–C7 | `pending_qualification` | `NOT_RELEASED` — runtime/geometry receipts pending |
| C2 | `hard_blocked` | `BLOCKED_SETUP` — verified common-prefix replay required |
| C8 | `hard_blocked` | `BLOCKED_RUNTIME` — GR00T Bridge/WidowX stack unverified |

Every queue row is a **registered new episode**. `reuse_episode_ids` are comparison-control links only; C3/C4 rows are not reuse-only aliases. C4 fast-schedule sham/move rows are new episodes because schedule differs from reused C1 controls.

## Exact next commands

```bash
python3 tools/online_correction_v4.py validate
python3 tools/build_online_correction_v4_freeze.py --out artifacts/online_correction_v4
python3 tools/validate_online_correction_v4.py
python3 tools/compile_online_correction_v4_ledger.py \
  --manifest artifacts/online_correction_v4/queue.jsonl \
  --attempts-root "$V4_ATTEMPTS_ROOT" \
  --out artifacts/online_correction_v4/compiled_ledger
python3 tools/analyze_online_correction_v4.py \
  --manifest artifacts/online_correction_v4/queue.jsonl \
  --results artifacts/online_correction_v4/compiled_ledger/accepted_ledger.jsonl \
  --out artifacts/online_correction_v4/analysis_tables
python3 -m unittest discover -s tests -p 'test_online_correction_v4*.py'
python3 tools/validate_vla_wam_v2_protocol.py
python3 tools/validate_vla_wam_v3_protocol.py
```

## Freeze artifact index

| Artifact | Purpose |
| --- | --- |
| `protocol.json` | Frozen estimands; planning vs enriched queue hashes |
| `prompt_manifest.json` | Bare-noun resolved prompts (no duplicate articles) |
| `queue.jsonl` / `queue_manifest.json` | 17,664 new episodes + control link metrics |
| `seed_manifest.json` | Env/policy seed reservation + best-effort collision audit |
| `gate_report.json` | `hard_blocked_families` vs `pending_not_released_families`; historical seed receipt derived from seed audit |
| `freeze_manifest.json` / `continuation_state.json` | Hash index and continuation authority |

| `git_receipt` (in protocol/continuation) | Pending until post-merge commit binding |
| `runtime_manifest.json` | `NOT_RELEASED` stub |
| `setup_manifest.json` | `NOT_RELEASED` stub (fixture keys mirror `campaign.json`) |
| `launch_matrix.json` | `NOT_RELEASED` stub |
| `compiled_ledger/accepted_ledger.jsonl` | One accepted valid row per manifest episode after attempt compilation |
| `compiled_ledger/rejected_attempts.jsonl` | Infra-invalid, superseded, and corrupted attempt inventory |
| `compiled_ledger/accepted_ledger_manifest.json` | Transitive hashes and queue/control reconciliation report |

### Accepted-ledger compiler

`tools/compile_online_correction_v4_ledger.py` consumes write-once attempt directories with `COMPLETE.json` and `evidence_manifest.json`, verifies blob hashes, classifies infra-invalid vs behavioral outcomes, selects at most one verified valid attempt per episode using `latest_verified_valid_by_attempt_id` (no outcome peeking), reconciles control reuse against accepted source episodes, and atomically emits `accepted_ledger.jsonl`, `rejected_attempts.jsonl`, and `accepted_ledger_manifest.json`. C2 prefix/response fields are copied only when the runtime recorded them; confirmatory C2 analysis remains fail-closed until the contract is complete.

### Seed collision audit limitations

The historical seed collision audit in `seed_manifest.json` is **best-effort**: it regex-scans committed JSON/JSONL under the repository (excluding `artifacts/online_correction_v4/`) for env/policy seed fields. It does not scan binary blobs, external cluster storage, or uncommitted files. A passing audit means no collision was found in the scanned scope, not a proof of global uniqueness.

## Verification boundary

Passing `validate_online_correction_v4.py` certifies deterministic freeze structure (all 15 generated artifacts byte-stable or generation-parent-normalized), prompt invariants including C2 `prompt_sha256` byte-identity semantics, control-link semantics, historical protocol integrity, seed-manifest vs queue alignment, seed receipt derivation from collision audit, setup/runtime/launch stub cross-checks vs campaign, family disposition parity across gate report and continuation, continuation/freeze hash cross-checks, and gate-report seed receipt derivation from the seed audit. It does not authorize policy inference or family release.
