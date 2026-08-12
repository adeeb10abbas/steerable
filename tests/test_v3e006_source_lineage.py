from __future__ import annotations

import json
from pathlib import Path
import subprocess


def test_every_source_lineage_sha_is_an_existing_commit() -> None:
    path = Path("artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006/source_lineage.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    for row in value["commits"]:
        subprocess.run(
            ["git", "cat-file", "-e", f"{row['sha']}^{{commit}}"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
