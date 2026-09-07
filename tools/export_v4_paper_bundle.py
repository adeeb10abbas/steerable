#!/usr/bin/env python3
"""Export registered V4 paper tables, audit bundle, and markdown stubs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.analysis import (  # noqa: E402
    DEFAULT_ANALYSIS_SEED,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    compile_analysis,
    digest_bytes,
    export_analysis_tables,
    load_accepted_ledger,
    load_campaign_config,
    load_manifest,
)


DEFAULT_CONFIG = ROOT / "docs/online_correction_v4/campaign.json"
DEFAULT_MANIFEST = ROOT / "artifacts/online_correction_v4/queue.jsonl"


def _table_markdown(path: Path, title: str) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        return f"## {title}\n\n_not available_\n"
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= 1:
        return f"## {title}\n\n_empty_\n"
    header = lines[0].split(",")
    body = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in lines[1:21]:
        body.append("| " + " | ".join(row.split(",")) + " |")
    suffix = ""
    if len(lines) > 21:
        suffix = f"\n\n_Showing 20 of {len(lines) - 1} rows. Source: `{path}`._\n"
    return f"## {title}\n\n" + "\n".join(body) + suffix


def export_paper_bundle(
    *,
    manifest_path: Path,
    results_path: Path,
    config_path: Path,
    output_dir: Path,
    bootstrap_resamples: int,
    seed: int,
) -> dict:
    config, _ = load_campaign_config(config_path)
    manifest = load_manifest(manifest_path)
    results = load_accepted_ledger(results_path)
    compiled = compile_analysis(
        manifest=manifest,
        results=results,
        config=config,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )
    tables_dir = output_dir / "tables"
    export = export_analysis_tables(
        compiled,
        tables_dir,
        manifest_path=manifest_path,
        results_path=results_path,
        config_path=config_path,
    )
    paper_dir = output_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    memo = {
        "schema_version": "v4-paper-evidence-memo-v1",
        "analysis_seed": seed,
        "bootstrap_resamples": bootstrap_resamples,
        "coverage": compiled["coverage"],
        "primary_results": compiled["primary_results"],
        "blocked_families": {
            row["family"]: row.get("status", "blocked")
            for row in compiled["scope_replications"]
            if row.get("status") != "descriptive"
        },
        "limitations": [
            "Figures require accepted trajectory evidence; this bundle exports tables only.",
            "C2 primary inference remains not estimable until verified common-prefix replay is recorded.",
        ],
    }
    memo_path = paper_dir / "evidence_memo.json"
    memo_path.write_text(json.dumps(memo, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = paper_dir / "RESULTS_STUB.md"
    sections = [
        "# V4 results export stub\n",
        "_Generated from accepted ledger; replace TODO markers after review._\n",
        _table_markdown(tables_dir / "coverage_by_cell.csv", "Coverage by cell"),
        _table_markdown(tables_dir / "primary_results.csv", "Primary results"),
        _table_markdown(tables_dir / "scope_replications.csv", "Scope replications (C5–C8)"),
        _table_markdown(tables_dir / "failure_composition.csv", "Failure composition"),
        _table_markdown(tables_dir / "timing_and_motion.csv", "Timing and motion"),
    ]
    report_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    bundle_manifest = {
        "schema_version": "v4-paper-bundle-manifest-v1",
        "tables_dir": str(tables_dir),
        "paper_dir": str(paper_dir),
        "analysis_tables": export["tables"],
        "evidence_memo": {
            "path": str(memo_path),
            "sha256": digest_bytes(memo_path.read_bytes()),
        },
        "results_stub": {
            "path": str(report_path),
            "sha256": digest_bytes(report_path.read_bytes()),
        },
    }
    bundle_path = output_dir / "paper_bundle_manifest.json"
    bundle_path.write_text(json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_ANALYSIS_SEED)
    args = parser.parse_args(argv)
    manifest = export_paper_bundle(
        manifest_path=args.manifest.resolve(),
        results_path=args.results.resolve(),
        config_path=args.config.resolve(),
        output_dir=args.out.resolve(),
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
