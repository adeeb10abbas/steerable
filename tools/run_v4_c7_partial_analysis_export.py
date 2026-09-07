#!/usr/bin/env python3
"""Re-run C7 partial analysis export from a PVC-compiled accepted ledger."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "docs/online_correction_v4/campaign.json"
DEFAULT_MANIFEST = ROOT / "artifacts/online_correction_v4/queue.jsonl"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="accepted_ledger.jsonl")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts/online_correction_v4/results/c7_partial",
    )
    parser.add_argument("--tag", type=str, default="latest")
    args = parser.parse_args(argv)
    results = args.results.resolve()
    if not results.is_file():
        raise SystemExit(f"missing accepted ledger: {results}")
    out_root = args.out.resolve() / args.tag
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    analyze = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/analyze_online_correction_v4.py"),
            "--config",
            str(args.config.resolve()),
            "--manifest",
            str(args.manifest.resolve()),
            "--results",
            str(results),
            "--out",
            str(out_root / "tables"),
        ],
        check=True,
        cwd=ROOT,
    )
    export = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/export_v4_paper_bundle.py"),
            "--config",
            str(args.config.resolve()),
            "--manifest",
            str(args.manifest.resolve()),
            "--results",
            str(results),
            "--out",
            str(out_root),
        ],
        check=True,
        cwd=ROOT,
    )
    manifest = {
        "schema_version": "v4-c7-partial-results-manifest-v1",
        "tag": args.tag,
        "accepted_ledger": {
            "path": str(results),
            "sha256": sha256_file(results),
            "bytes": results.stat().st_size,
        },
        "output_root": str(out_root),
        "analyze_exit_code": analyze.returncode,
        "export_exit_code": export.returncode,
    }
    manifest_path = out_root / "results_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
