# Forecast publication figure rendering

`analysis/render_forecast_publication.py` turns the signed, source-only output
of `compile_forecast_publication.py` into deterministic SVG figures. It accepts
an immutable publication directory and refuses to overwrite an existing output
directory. Create the desired output parent yourself; the renderer deliberately
does not create missing parents:

```bash
python3 workshops/corl2026_world_models/analysis/render_forecast_publication.py \
  --publication-dir PUBLICATION_BUNDLE \
  --output-dir FIGURE_OUTPUT
```

Run the command from a committed, published
`codex/forecast-layout-gm-20260912` checkout (or that exact commit detached).
The input directory must contain the exact inventory signed by
`build_receipt.json`. The renderer re-hashes that complete inventory, verifies
the signatures and exact schemas of `forecast_skill_layout_effects.json` and
`actual_scene_timing.json`, and checks their final-analysis and fixture bindings
against the build receipt. It repeats validation before atomically publishing
the output.

Source identity is not self-declared from worktree hashes. If `P` is the
publication compiler's source commit and `S` is the renderer checkout HEAD, the
renderer first requires the canonical compiler to equal both its build-receipt
descriptor and the renderer's pinned trusted compiler SHA-256. It then uses the
compiler's protected system Git, scrubbed environment, fresh controlled bare
repository, literal public HTTPS URL and stable exact control ref to prove
`P <= S <= control`. The compiler must be byte-identical at `P` and `S`; the
renderer and this contract must equal their tracked blobs at `S`.
The authenticated `S`, remote receipt and exact source descriptors are signed
into `render_receipt.json`. Dirty coordinated replacement and an unpublished
local descendant both fail before any output directory is published.

The output boundary is directory-fd based. Every already-existing parent
component is opened with `O_DIRECTORY|O_NOFOLLOW`; the held parent identity is
rechecked before staging, immediately before rename, and after rename. The
staging directory and its exclusive output files are created relative to held
directory fds, and `renameat2(RENAME_NOREPLACE)` publishes it within the same
held parent. If the named parent changes across the rename boundary, publication
is rolled back through that fd before staging cleanup. A symlink ancestor,
missing parent, raced-in target or swapped parent therefore fails without
creating an outside directory or publishing a partial bundle.

The output contains:

- `actual_scene_layouts.svg`: all 24 accepted original/reflected scene pairs on
  one robot-base x-y scale. Marker titles retain each exact x, y, z and
  quaternion input; z is also printed beside each marker.
- `qualified_timing.svg`: separate schematic tracks for generated-frame index,
  physical horizon and executed-action offset, plus the exact context cap,
  camera and timestamp tolerance for each included model.
- `forecast_skill_MODEL.svg`: a separate signed-input skill figure for each
  included model.
- `reflection_layout_effects_MODEL.svg`: a separate reflection figure for each
  included model. Missing contrasts remain visibly “not estimable.”
- `render_receipt.json`: a closed-schema, signed, path-safe SHA-256 inventory of
  every SVG and the authenticated renderer commit, contract and publication
  input bindings. Extra top-level or nested claims are rejected.

The arrows in the timing diagram are deliberately schematic. The publication
inputs do not supply a conversion that equates a generated-frame index with a
physical duration or executed-action offset. The renderer therefore prints the
three recorded identities without deriving such a mapping. It also performs no
labeling, model request, model pooling, estimator recomputation, paper edit or
scientific interpretation.
