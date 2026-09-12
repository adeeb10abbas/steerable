# First-sprint delivery verification

Date: 2026-09-12. Status: **working research package; not submitted**.

## Verified artifact

`paper/main.pdf` has **four pages total including references**, compiled using
the unchanged official CoRL 2026 style in preprint mode. The author line explicitly
identifies it as a working draft; the final author list has not been supplied.

- PDF SHA256: `0fb7d780160c08b0189675abb854bc5e8d09135c650129e9d58ec60895a2d756`.
- Official style SHA256: `62f38cd8df7ad718796617595c8230da136fc83ea1d7e1ba80084113409ec957`.
- Automated checks: four pages; expected evidence text; references resolved;
  no overfull boxes or LaTeX errors.
- All four final pages were rendered at 120 dpi and visually inspected. The
  table, figure labels, equation, captions, citations and body text are readable;
  no clipping, overlap or missing glyphs was observed. The earlier stretched
  checkpoint-name line was corrected before the final render.

The editable manuscript, reference database, official template provenance,
data-derived PDF/PNG figure and build instructions are included. The numerical
figure is generated directly from `results/audit_summary.json`.

## Verified analysis and review

The integrated build ran all **17 tests**, the complete historical replay, the
private annotation draw, figure generation and the PDF build successfully.
See `results/validation.log` and `results/build_report.json` for receipts.

- All 752 historical chunk rows and 3,008 cached frames reconcile.
- All 761 audited source-file Git identities match the pinned upstream evidence.
  Current SHA256 receipts are separate from historical raw-file hashes.
- Historical totals and the same-subset always-negative baseline match the
  manuscript. No table is labeled corrected prediction fidelity.
- Descriptive coverage uncertainty retains all four episodes and their chunks
  within each of 20 shared-seed blocks, separately sampling the two seed tiers.
- The private annotation draw contains 160 unique chunks with the intended
  stratum sizes and weights summing to 752. Its canonical draw SHA256 is
  `ab780ea4e1b91918ca72801ef0641b563725d0a7d935f6de15e1d135b0ed6d47`.

An independent scientific/code review identified and checked fixes for the
bootstrap dependence and normalized adapter's time/reliability validation. It
also independently reproduced the annotation sample and checked manuscript
claim boundaries, model identity and citation fit. No blocking finding remained
within that review's scope. This review is not independent human labeling of
the model's predictions or a substitute for empirical validation.

## Outstanding scientific work

Original raw evidence must be recovered and its physical time and geometry
verified before an aligned comparison can be produced. Human annotations,
persistence comparison, any calibrated decision rule and independent replication
remain pending. The adapter only validates a normalized record and byte identity;
it does not prove temporal or semantic alignment. Historical runtime patches
remain a separate exact-reproduction requirement.

The final recovery check includes an 80-episode execution-imagery readiness
inventory. No separate execution-image/video receipts appear under the selected
run roots; HDF5 RGB contents have not been inspected. A later conditioning image
is a possible source only with verified physical-time alignment. None of these
routes is presently certified, and all unknown endpoints remain explicit.

The original study files were preserved. This workshop package is a disclosed
retrospective amendment in its own directory and local branch. No model inference,
remote compute, new human annotation, submission or public release was performed.
