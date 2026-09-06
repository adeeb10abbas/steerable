#!/usr/bin/env python3
"""Compile V4 accepted-ledger evidence into frozen primary tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.analysis import (  # noqa: E402
    AnalysisError,
    DEFAULT_ANALYSIS_SEED,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    compile_analysis,
    export_analysis_tables,
    load_accepted_ledger,
    load_campaign_config,
    load_manifest,
    validate_accepted_ledger,
)

DEFAULT_CONFIG = ROOT / "docs/online_correction_v4/campaign.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=None,
        help="Optionally write the raw ledger validation object as JSON.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_ANALYSIS_SEED)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run ledger validation and coverage reconciliation without confirmatory inference.",
    )
    args = parser.parse_args(argv)
    try:
        config, _ = load_campaign_config(args.config)
        manifest = load_manifest(args.manifest)
        results = load_accepted_ledger(args.results)
        validation = validate_accepted_ledger(manifest, results, config=config)
        if args.validation_report is not None:
            args.validation_report.parent.mkdir(parents=True, exist_ok=True)
            args.validation_report.write_text(
                json.dumps(validation, allow_nan=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        if args.validate_only:
            report = {"ok": validation["ok"], "validation": validation}
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if validation["ok"] else 1
        compiled = compile_analysis(
            manifest=manifest,
            results=results,
            config=config,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )
        export = export_analysis_tables(
            compiled,
            args.out,
            manifest_path=args.manifest,
            results_path=args.results,
            config_path=args.config,
        )
        report = {
            "ok": True,
            "validation": validation,
            "coverage": compiled["coverage"],
            "primary_results": compiled["primary_results"],
            "export": export,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (AnalysisError, ValueError, KeyError, OSError) as exc:
        print(json.dumps({"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
