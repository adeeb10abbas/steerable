# Success Is Not Grounding

This directory contains the six-page, double-anonymous robotics-paper draft and its publication figures.

## Build

1. Upload the project to Overleaf and set `main.tex` as the main file.
2. For an RA-L submission, add the official `ieeeconf.cls` from PaperCept. The source selects it automatically.
3. Compile with pdfLaTeX and BibTeX.

The included `main.pdf` was verified as six pages with the installed `IEEEtran` conference fallback. Recheck pagination with the official class; the source does not alter margins or body font size.

For a local rebuild:

```bash
make
```

`Makefile` accepts a custom plotting runtime through `PYTHON=/path/to/python` when the default Python lacks Matplotlib.

## Anonymous review

`main.tex` defaults to `\anonymoustrue`. Change it to `\anonymousfalse` only for the camera-ready version. Identifying metadata is isolated in `metadata.tex`.

## Paper structure

- The abstract and introduction frame binary LEFT/RIGHT success as a non-identifying grounding measure.
- The evaluation design separates canonical-token response, endpoint orientation, continuous placement, task completion, reference inversion, and scene interventions.
- C002-R001 is reported as a prospective post-gate operational repair, with the semantic-equivalence claim withheld.
- Reflection, the seven-position sweep, and the symmetric-scene cohort establish dependence of directional completion on scene configuration.
- Discussion states the two benchmark errors directly: responsive control can look like failure, and canonical surface-form response can look like semantic generalization.

## Evidence and figures

- `EVIDENCE_MAP.md` traces every headline claim to the verified repository commit and records cohort boundaries.
- `figures/README.md` describes all six vector plots.
- `submission_checklist.md` lists the remaining scientific and formatting gates.
- `notes/abstract_metadata.txt` is a submission-form abstract under 1,200 characters.

The manuscript uses six vector figures generated at final placement size. Figure 1 reports the complete eight-row canonical screen, and Figure 2 reports the independent C002 reference-inversion cohort before the paper narrows to registered scene interventions.

## Remaining submission work

1. Compile with the official `ieeeconf.cls` and recheck six-page pagination.
2. Insert complete public checkpoint and embodiment provenance and add canonical citations where available.
3. Re-run all evidence and figure validators from the pinned commit.
4. Have a coauthor independently trace every headline number.
5. Verify the compact screening and C002 figures against their pinned source artifacts after the final evidence rerun.
