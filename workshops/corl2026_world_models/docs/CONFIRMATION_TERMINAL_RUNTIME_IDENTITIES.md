# Confirmation terminal-runtime identity publication

Status: source-only producer and contract frozen to the reviewed N3 and D1
confirmation runtime bytes; no identity receipt may be produced until these
files are committed and visible in the authorized public control history. No
identity receipt, queue entry, model request, action, episode or label is
created by this source slice.

`experiments/forecast_layout/confirmation_terminal_runtime_identities.py`
produces the exact `wmf-confirmation-terminal-runtime-identities-v1` document
consumed by the confirmation release-wave gate. It closes the gap between a
receipt that merely repeats caller-selected hashes and one whose identities are
derived from an authenticated control commit.

## Authority and source boundary

The production command has only three inputs: the immutable staged source root,
its full commit ID, and one output filename. Runtime and prerequisite paths or
hashes are intentionally not command-line inputs. They come only from the
producer's tracked contract.

Before it accepts that contract, the producer:

1. uses a root-owned, non-writable system Git binary with inherited Git config,
   credential helpers, askpass, SSH overrides, extra HTTP headers, replacements
   and redirects disabled;
2. fetches only
   `refs/heads/codex/forecast-layout-gm-20260912` from the literal public HTTPS
   repository into a disposable controlled bare graph;
3. requires the ref to be stable before and after the fetch, requires the
   supplied immutable source commit `S` to exist in that fetched public graph,
   and proves `S` is an ancestor of the stable release/control head `R`; and
4. requires the executing producer, its contract, and all five listed runtime
   files to be regular tracked blobs whose local bytes equal the blobs at that
   exact commit.

The five runtime files are the N3 behavioral runtime, N3 confirmation adapter,
D1 behavioral runtime, D1 confirmation adapter and D1 instrumented server. The
receipt exposes only each canonical repository path and SHA-256, matching the
closed consumer schema. It does not claim that a machine, GPU, policy server or
simulator is currently live.

## Historical prerequisite boundary

The contract also freezes the exact absolute path, byte count, SHA-256 and
minimal passed-state fields of these immutable GM PVC receipts:

- N3 P00 behavioral pilot attempt 004;
- fixed-duration recorder qualification attempt 003;
- official-path D1 six-request qualification attempt 005;
- D1 P00 simulator pilot attempt 003; and
- D1 P00 server pilot attempt 003.

Every path component is opened with no-follow directory descriptors. Each file
is read from one descriptor, checked before and after the complete read, and
required to remain the same inode at its original path. The producer rereads
all source and prerequisite evidence and reobserves the remote ref immediately
before publication and again after publication.

## Immutable output and native validation

The output parent must already exist and may contain no symlink component. A
mode-0444 staging file is created relative to a held parent descriptor, fsynced,
validated byte-for-byte, and renamed with `RENAME_NOREPLACE`. Parent replacement,
source drift, prerequisite drift, remote-ref drift or a post-rename validation
failure removes the just-created inode through the held descriptor. Existing
outputs are never replaced.

The release consumer must authenticate the producer and contract as closed
source dependencies, then call
`validate_terminal_runtime_identities(...)`. That native validator repeats the
same remote, Git-blob, contract, path and prerequisite checks; duplicating only
the outer JSON schema is insufficient.

Once the producer/contract and their pinned confirmation adapters are published
in one immutable source commit `S`, and a release/control commit `R` containing
the descriptor is publicly visible with `S <= R`, run on the clean staged `S`:

```bash
python workshops/corl2026_world_models/experiments/forecast_layout/confirmation_terminal_runtime_identities.py \
  --source-root /data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/control/sources/<commit> \
  --source-commit <immutable-published-source-commit-S> \
  --output /data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/control/confirmation/terminal_runtime_identities.json
```

The production output must be generated only on the GM PVC from that exact
published immutable source commit after its release/control descendant is
visible. A source test fixture is not a runtime identity receipt and
must never be used for release.

The claim boundary is deliberately narrow: the artifact publishes Git blob
identities and immutable prerequisite descriptors only. It releases no job,
mutates no queue, starts no process, performs no inference, and provides no
forecast-accuracy or behavioral evidence.
