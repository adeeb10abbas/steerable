# Forecast publication compiler

`analysis/compile_forecast_publication.py` is the final source-only boundary
between the validated confirmation analysis and paper-facing evidence files.
It creates compact, signed evidence and table/figure inputs. It does **not**
edit or generate a manuscript, create labels, recompute an estimator, start a
model or simulator, choose examples by outcome, or publish to Git by itself.

The command is intentionally unusable before the scientific evidence is
complete. Before staging any output it:

1. verifies the signed publication input and every exact file descriptor;
2. authenticates the executing source commit against the exact authorized
   GitHub repository and control-ref history, as described below;
3. requires the analyzer descriptor to equal the confirmation compiler's
   `final_analyzer_dependency`, checks that it is at the canonical analyzer
   path in a clean checkout whose HEAD is the receipt `study_commit`, and
   compares its bytes with `git show study_commit:<canonical-path>` before
   importing it;
4. runs the final analyzer again from the signed analysis-evidence manifest
   and requires object-for-object equality with the supplied final analysis;
5. applies the same independent canonical-path, clean-checkout and tracked-blob
   checks to the receipt's `compiler_source` before importing it and running
   `compile_confirmation_evidence._validate_compiled_bundle`;
6. applies those checks to `fixture_freeze_dependency` before importing it,
   deep-replays the fixture separately for all 24 base-layout pairs, and
   requires every authenticated model/layout aggregate to bind that freeze;
7. requires the passed human-consensus claim gate, a nonempty annotation
   inventory, separate model reports, the correct full/reduced branch, exact
   96-cell status denominators per included model, and non-null forecast,
   persistence and primary-skill estimates from 10,000 layout bootstraps;
8. reconstructs the declared video choice from status/identity metadata and
   uniquely joins it to the deep-validated private-video inventory; and
9. re-authenticates the remote control ref and re-hashes all source and evidence
   dependencies immediately before one atomic directory rename.

## Committed source boundary

The confirmation `study_commit` identifies the source that produced the
scientific cohort. It can predate this publication compiler. The signed input
therefore carries a separate `publication_source_commit` P, which must equal
the executing checkout's exact `HEAD`. P is not required to equal the current
control-branch head: an immutable queue worker normally runs P while the queue
descriptor that released it is a later commit R. Instead, the compiler pins
the authorized repository to
`https://github.com/adeeb10abbas/steerable.git` and the control ref to
`refs/heads/codex/forecast-layout-gm-20260912`. The executing checkout's
`publish` alias (workstation), or `origin` when `publish` is absent (staged
cluster checkout), is checked only as a deployment-route sanity check. Its raw,
Git-expanded fetch and effective push URLs must be the exact credential-free
HTTPS URL above or an exact GitHub SSH spelling for the same owner/repository.
No alias, checkout config, SSH command or credential helper is used to
authenticate source bytes.

Read authentication always addresses the literal canonical public HTTPS URL
from a new template-free bare repository under `/tmp`. It invokes exactly one
root-owned, non-writable Git binary selected from `/usr/bin/git` and
`/usr/local/bin/git`; a `/usr/local/bin/git` symlink is accepted only when its
fully inspected chain terminates at another pinned, protected binary. `PATH`,
`GIT_EXEC_PATH`, Git/SSH/askpass variables, credential helpers, extra HTTP
headers, URL rewrites and inherited local/global/system config cannot select
the transport. Only bounded proxy and CA environment settings survive. The
new bare repository's generated config is replaced with an exact inert
core-only config. A filtered fetch may add promisor state, so that state is
immediately erased and the inert config is hash-checked before any source or
ancestry read; later config drift is fatal.

The compiler obtains R with exact-ref `ls-remote --refs`, fetches only that ref
with a 1 MiB blob filter into the disposable graph, and repeats the same
`ls-remote`. Each Git process has a 60-second timeout and the complete graph is
capped at 192 MiB. All three R observations must agree, and the remotely
fetched objects—not the executing checkout—must prove `study_commit <= P <= R`.
Required source blobs above the 1 MiB limit are rejected rather than lazily
fetched. An attached checkout must be on the exact control branch; a detached
checkout is accepted for an immutable cluster source stage. Thus a locally
committed but unpublished descendant and divergent history both fail, while a
legitimate P-source/R-descriptor queue release succeeds. The entire isolated
read authentication is repeated in a second fresh graph immediately before
atomic publication.

For both the compiler and its JSON contract, the command reads the tracked blob
from the isolated remote graph with `git show P:<path>` and requires those
bytes to equal the executing file exactly. An untracked, dirty or drifted
compiler/contract is rejected.
Independently, the confirmation `study_commit` must exist in the same trusted
repository and be an ancestor of P. The historical source root must be a clean
Git checkout exactly at that commit. The analyzer, confirmation compiler and
fixture validator must occupy their fixed canonical paths, match their
`study_commit` blobs read from the isolated graph, and also be byte-identical
at authenticated P before any is imported. Thus neither an input-selected
alternate commit nor a
malicious-but-ancestral validator revision can establish its own trust.
Coordinated replacement modules and self-hashed replacement receipts cannot
define their own replay validators. These checks repeat immediately before
atomic publication. The build receipt records the study commit, P, the
separately authenticated R, literal read transport, Git binary identity and
version, filter/graph limits, inert-config digest, exact authorized
repository/ref identity, and canonical tracked-source hashes. Temporary graphs
are removed on success and failure. Errors never include commands, URLs, Git
stderr, proxy credentials or credential-bearing remote material.

## Signed input

All descriptors have exactly `path`, `bytes`, and `sha256`. Paths may point to
private PVC evidence because the input itself is restricted. Absolute private
paths are stripped from every public output.

```json
{
  "schema_version": "wmf-forecast-publication-input-v1",
  "study_id": "WMF-ABLATION-001",
  "cohort_branch": "full_two_model",
  "final_analysis": {"path": "/restricted/final_analysis.json", "bytes": 0, "sha256": "..."},
  "confirmation_fixture_freeze": {"path": "/restricted/confirmation_fixture_freeze.json", "bytes": 0, "sha256": "..."},
  "confirmation_compiler_receipt": {"path": "/restricted/compiled/compiler_receipt.json", "bytes": 0, "sha256": "..."},
  "private_video_inventory": {"path": "/restricted/compiled/private_video_inventory.json", "bytes": 0, "sha256": "..."},
  "analyzer_source": {"path": "/staged/source/workshops/corl2026_world_models/analysis/forecast_evidence_analysis.py", "bytes": 0, "sha256": "..."},
  "publication_source_commit": "REPLACE_WITH_FULL_COMMITTED_OBJECT_ID",
  "copy_selected_videos": false,
  "payload_sha256": "..."
}
```

`payload_sha256` is SHA-256 over canonical UTF-8 JSON (sorted keys, compact
separators, no NaN) after removing `payload_sha256`.

Run only from the exact committed source tree:

```sh
python3 workshops/corl2026_world_models/analysis/compile_forecast_publication.py \
  --input-manifest /restricted/forecast_publication_input.json \
  --output-dir /compact/forecast_publication
```

The output directory must not exist. A failed validation leaves no partial
output directory and never replaces an existing one.

## Outputs

- `paper_evidence.json` preserves the separate per-model baselines/skill,
  coverage, full-design missingness bounds, paired reflection contrasts,
  stopping controls, movement strata/decomposition, earlier-horizon state,
  annotation quality, sample-size table and claim boundaries.
- `model_results.csv` and `coverage_by_condition.csv` are deterministic table
  inputs. `model_results_table.tex` and `sample_size_table.tex` are fragments,
  not manuscript files.
- `forecast_skill_layout_effects.json` contains per-model skill layout values
  and actual-versus-predicted reflection contrasts without pooling models.
- `actual_scene_timing.json` contains the physically gated robot-base scene
  poses plus the exact qualified horizon, frame, action-offset, camera and
  tolerance fields used for each included model.
- `example_videos.json` contains one metadata-hash-selected example per
  included model/condition and no private source path.
- `build_receipt.json` binds every other output by relative path, bytes and
  SHA-256, plus P and the separately authenticated remote control head R; its
  own canonical `payload_sha256` is its self-binding.

When `copy_selected_videos` is false, only the public manifest is emitted and
the private MP4s remain on the PVC. When true, every declared example is copied
byte-for-byte under `videos/`; none may exceed 16 MiB and their combined size
may not exceed 64 MiB, matching the deployed compact publication limits. There
is no transcoding, fallback choice or partial subset. Technical-invalid rows
may legitimately have no video, including nonzero retained action prefixes,
but only when both the compiler roster and the deep-validated source-artifact
row bind the same technical-invalid/null-video state. `valid_complete` and
`valid_censored` rows always require retained hash-verified MP4 evidence and
are the only statuses from which examples can be chosen.

The public compiler never asserts a between-model ranking or a causal policy
benefit. A reduced branch includes the absent primary model only in the
sample-size denominator row as `unqualified_branch_not_run`; it never inserts a
replacement model or pools the remaining one with anything else.
