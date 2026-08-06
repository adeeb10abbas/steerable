# Research PDF publication assets

This directory contains the compact, selected figures used by
`tools/build_vla_wam_research_pdf.py`.

The PDF builder reads all result numbers from committed evidence summaries. The
two raster figures in `figures/` are selected publication renderings of the
historical DROID endpoint layer and expanded RoboTwin diagnostics. Historical
wording-success and Nano V3-B001 seed-level plots are regenerated directly from
the committed summary JSON files.

Build:

```bash
tmp/pdfs/.venv/bin/python tools/build_vla_wam_research_pdf.py
```

The final report and its SHA-bearing manifest are written under `output/pdf/`.
DROID/RoboLab and RoboTwin success rates remain separate, compatibility cohorts
remain separate from historical identities, and future-interface evidence is
never scored as behavioral episodes.
