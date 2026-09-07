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

from experiments.online_correction_v4.analysis import load_manifest  # noqa: E402

DEFAULT_CONFIG = ROOT / "docs/online_correction_v4/campaign.json"
DEFAULT_MANIFEST = ROOT / "artifacts/online_correction_v4/queue.jsonl"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_accepted_rows(results_path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def build_scoped_manifest(
    *,
    queue_path: Path,
    accepted_rows: list[dict],
    family: str = "C7",
) -> tuple[list[dict], dict]:
    queue_by_id = {row["episode_id"]: row for row in load_manifest(queue_path)}
    scoped: list[dict] = []
    missing: list[str] = []
    for row in accepted_rows:
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str):
            continue
        manifest_row = queue_by_id.get(episode_id)
        if manifest_row is None:
            missing.append(episode_id)
            continue
        if manifest_row.get("family") != family:
            continue
        bound = dict(manifest_row)
        config_sha = row.get("config_sha256")
        if isinstance(config_sha, str):
            bound["config_sha256"] = config_sha
        scoped.append(bound)
    if missing:
        raise SystemExit(
            f"accepted ledger references {len(missing)} episode IDs missing from queue manifest"
        )
    if not scoped:
        raise SystemExit("no accepted rows matched the requested family in the queue manifest")
    planned_c7 = sum(1 for row in queue_by_id.values() if row.get("family") == family)
    summary = {
        "family": family,
        "accepted_unique_episodes": len({row["episode_id"] for row in scoped}),
        "accepted_rows": len(accepted_rows),
        "planned_family_episodes": planned_c7,
        "missing_family_episodes": planned_c7 - len({row["episode_id"] for row in scoped}),
        "manifest_binding": "accepted_rows_only_with_frozen_config_sha256_from_ledger",
    }
    return scoped, summary


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


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
    accepted_rows = load_accepted_rows(results)
    scoped_manifest, scope_summary = build_scoped_manifest(
        queue_path=args.manifest.resolve(),
        accepted_rows=accepted_rows,
    )
    out_root = args.out.resolve() / args.tag
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    scoped_manifest_path = out_root / "scoped_c7_manifest.jsonl"
    write_jsonl(scoped_manifest_path, scoped_manifest)
    analyze = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/analyze_online_correction_v4.py"),
            "--config",
            str(args.config.resolve()),
            "--manifest",
            str(scoped_manifest_path),
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
            str(scoped_manifest_path),
            "--results",
            str(results),
            "--out",
            str(out_root),
        ],
        check=True,
        cwd=ROOT,
    )
    blocked_path = out_root / "blocked_scope.json"
    blocked_path.write_text(
        json.dumps(
            {
                "schema_version": "v4-c7-partial-blocked-scope-v1",
                "accepted_c7_episodes": scope_summary["accepted_unique_episodes"],
                "planned_c7_episodes": scope_summary["planned_family_episodes"],
                "missing_c7_episodes": scope_summary["missing_family_episodes"],
                "not_estimable_or_blocked": {
                    "C1": "no accepted ledger rows in this partial export",
                    "C2": "primary blocked until verified common-prefix replay",
                    "C3": "no accepted ledger rows in this partial export",
                    "C4": "no accepted ledger rows in this partial export",
                    "C5": "no accepted ledger rows in this partial export",
                    "C6": "no accepted ledger rows in this partial export",
                    "C8": "no accepted ledger rows in this partial export",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "v4-c7-partial-results-manifest-v1",
        "tag": args.tag,
        "scope_summary": scope_summary,
        "accepted_ledger": {
            "path": str(results),
            "sha256": sha256_file(results),
            "bytes": results.stat().st_size,
        },
        "scoped_manifest": {
            "path": str(scoped_manifest_path),
            "sha256": sha256_file(scoped_manifest_path),
        },
        "blocked_scope": {
            "path": str(blocked_path),
            "sha256": sha256_file(blocked_path),
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
