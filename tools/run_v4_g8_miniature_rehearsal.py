#!/usr/bin/env python3
"""Exercise V4 scheduler evidence, retry, resume, compiler, figure, and closure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.analysis import load_campaign_config  # noqa: E402
from experiments.online_correction_v4.ledger import (  # noqa: E402
    compile_accepted_ledger_from_attempts,
    discover_finalized_attempts,
    write_ledger_outputs,
)
from tools.build_v4_pilot_video_montage import (  # noqa: E402
    discover_attempts,
    load_queue,
    parse_selections,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _link_attempt(root: Path, episode_id: str, source: Path) -> Path:
    destination = root / episode_id / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=True)
    return destination


def _make_deliberate_invalid_attempt(
    *,
    root: Path,
    episode_id: str,
    source: Path,
) -> Path:
    destination = root / episode_id / "attempt-g8-deliberate-infra-failure"
    destination.mkdir()
    for child in source.iterdir():
        if child.name in {"COMPLETE.json", "evidence_manifest.json"}:
            continue
        (destination / child.name).symlink_to(
            child,
            target_is_directory=child.is_dir(),
        )
    evidence = json.loads(
        (source / "evidence_manifest.json").read_text(encoding="utf-8")
    )
    evidence["attempt_id"] = "attempt-g8-deliberate-infra-failure"
    evidence["trajectory_sha256"] = "0" * 64
    evidence_bytes = canonical_json(evidence)
    (destination / "evidence_manifest.json").write_bytes(evidence_bytes)
    complete = json.loads((source / "COMPLETE.json").read_text(encoding="utf-8"))
    complete["attempt_id"] = "attempt-g8-deliberate-infra-failure"
    complete["evidence_manifest_sha256"] = hashlib.sha256(
        evidence_bytes
    ).hexdigest()
    (destination / "COMPLETE.json").write_bytes(canonical_json(complete))
    return destination


def render_outcome_figure(
    rows: list[dict[str, Any]],
    output: Path,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("outcome", {}).get("failure_label", "missing"))
        counts[label] = counts.get(label, 0) + 1
    labels = sorted(counts)
    width = 640
    height = 360
    maximum = max(counts.values(), default=1)
    bar_width = max(40, 480 // max(1, len(labels)))
    bars = []
    for index, label in enumerate(labels):
        x = 90 + index * bar_width
        bar_height = int(220 * counts[label] / maximum)
        y = 290 - bar_height
        bars.append(
            f'<rect x="{x}" y="{y}" width="{bar_width - 16}" '
            f'height="{bar_height}" fill="#315aa6"/>'
        )
        bars.append(
            f'<text x="{x + (bar_width - 16) // 2}" y="{y - 8}" '
            f'text-anchor="middle">{counts[label]}</text>'
        )
        bars.append(
            f'<text x="{x + (bar_width - 16) // 2}" y="315" '
            f'text-anchor="middle">{label}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="320" y="28" text-anchor="middle" font-size="18">'
        "V4 G8 miniature rehearsal outcomes</text>"
        '<line x1="70" y1="290" x2="590" y2="290" stroke="black"/>'
        + "".join(bars)
        + "</svg>\n"
    )
    output.write_text(svg, encoding="utf-8")
    return counts


def run_rehearsal(
    *,
    raw_root: Path,
    pilot_queue_path: Path,
    selections: dict[str, str],
    config_path: Path,
    protocol_sha256: str,
    scorer_sha256: str,
    scheduler_receipts: list[Path],
    invalid_attempt_receipts: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    queue_by_id = load_queue(pilot_queue_path)
    selected_attempts = discover_attempts(
        raw_root=raw_root,
        queue=queue_by_id,
        selections=selections,
    )
    miniature_ids = sorted(selected_attempts)[:3]
    if len(miniature_ids) != 3:
        raise ValueError("G8 rehearsal requires three reusable pilot attempts")
    miniature_rows = [queue_by_id[episode_id] for episode_id in miniature_ids]
    miniature_manifest = output_dir / "miniature_manifest.jsonl"
    write_jsonl(miniature_manifest, miniature_rows)
    config, _config_sha = load_campaign_config(config_path)

    attempts_root = output_dir / "attempts"
    for episode_id in miniature_ids[:2]:
        _link_attempt(attempts_root, episode_id, selected_attempts[episode_id])
    interrupted = compile_accepted_ledger_from_attempts(
        manifest=miniature_rows,
        attempts=discover_finalized_attempts(attempts_root),
        attempts_root=attempts_root,
        protocol_sha256=protocol_sha256,
        scorer_sha256=scorer_sha256,
        config=config,
        queue_episode_ids=set(miniature_ids),
        require_full_coverage=True,
    )
    if interrupted.ok or interrupted.reconciliation.get("missing_valid") != 1:
        raise ValueError("controlled interruption did not fail closed on one row")
    _link_attempt(
        attempts_root,
        miniature_ids[2],
        selected_attempts[miniature_ids[2]],
    )
    resumed = compile_accepted_ledger_from_attempts(
        manifest=miniature_rows,
        attempts=discover_finalized_attempts(attempts_root),
        attempts_root=attempts_root,
        protocol_sha256=protocol_sha256,
        scorer_sha256=scorer_sha256,
        config=config,
        queue_episode_ids=set(miniature_ids),
        require_full_coverage=True,
    )
    if not resumed.ok or len(resumed.accepted_rows) != 3:
        raise ValueError(f"resume compilation failed: {resumed.errors}")
    resumed_outputs = write_ledger_outputs(
        resumed,
        output_dir / "resumed-ledger",
        attempts_root=attempts_root,
        manifest_path=miniature_manifest,
    )
    invalid_path = _make_deliberate_invalid_attempt(
        root=attempts_root,
        episode_id=miniature_ids[0],
        source=selected_attempts[miniature_ids[0]],
    )
    retry = compile_accepted_ledger_from_attempts(
        manifest=miniature_rows,
        attempts=discover_finalized_attempts(attempts_root),
        attempts_root=attempts_root,
        protocol_sha256=protocol_sha256,
        scorer_sha256=scorer_sha256,
        config=config,
        queue_episode_ids=set(miniature_ids),
        require_full_coverage=True,
    )
    if (
        not retry.ok
        or len(retry.accepted_rows) != 3
        or len(retry.rejected_rows) != 1
    ):
        raise ValueError(
            "deliberate infrastructure failure was not rejected while valid "
            f"retry remained accepted: errors={retry.errors}"
        )
    retry_outputs = write_ledger_outputs(
        retry,
        output_dir / "retry-ledger",
        attempts_root=attempts_root,
        manifest_path=miniature_manifest,
    )
    figure_path = output_dir / "miniature_outcomes.svg"
    outcome_counts = render_outcome_figure(retry.accepted_rows, figure_path)
    no_trigger_count = sum(
        not bool(row.get("trigger_eligible"))
        for row in retry.accepted_rows
    )
    valid_behavior_failure_count = sum(
        not bool(row.get("success"))
        for row in retry.accepted_rows
    )
    source_artifacts = [
        pilot_queue_path,
        config_path,
        *scheduler_receipts,
        *invalid_attempt_receipts,
    ]
    receipt = {
        "schema_version": "v4-g8-miniature-campaign-rehearsal-v1",
        "campaign_id": "online_correction_v4",
        "gate": "G8",
        "status": "passed",
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "reused_excluded_engineering_episode_count": 3,
        "miniature_episode_ids": miniature_ids,
        "checks": {
            "scheduler_execution_evidence_present": bool(scheduler_receipts),
            "failed_attempt_retry_exercised": len(retry.rejected_rows) == 1,
            "interruption_failed_closed": (
                not interrupted.ok
                and interrupted.reconciliation.get("missing_valid") == 1
            ),
            "resume_restored_full_coverage": (
                resumed.ok and len(resumed.accepted_rows) == 3
            ),
            "compiler_rejected_deliberate_infra_failure": (
                retry.ok
                and len(retry.accepted_rows) == 3
                and len(retry.rejected_rows) == 1
            ),
            "figure_rendered": figure_path.is_file(),
            "valid_behavior_failure_present": valid_behavior_failure_count >= 1,
            "no_trigger_episode_present": no_trigger_count >= 1,
            "prior_invalid_attempt_inventory_present": bool(
                invalid_attempt_receipts
            ),
        },
        "outcome_counts": outcome_counts,
        "controlled_interruption": {
            "accepted_before_resume": len(interrupted.accepted_rows),
            "missing_before_resume": interrupted.reconciliation.get(
                "missing_valid"
            ),
            "accepted_after_resume": len(resumed.accepted_rows),
        },
        "deliberate_infrastructure_failure": {
            "path": str(invalid_path),
            "mechanism": (
                "A copied manifest binding declared an intentionally incorrect "
                "trajectory digest; the compiler rejected it while accepting "
                "the untouched retry."
            ),
        },
        "outputs": {
            "miniature_manifest": {
                "path": str(miniature_manifest),
                "sha256": sha256_file(miniature_manifest),
            },
            "resumed_ledger": {
                "path": resumed_outputs["accepted_ledger"],
                "sha256": sha256_file(Path(resumed_outputs["accepted_ledger"])),
            },
            "retry_ledger": {
                "path": retry_outputs["accepted_ledger"],
                "sha256": sha256_file(Path(retry_outputs["accepted_ledger"])),
            },
            "rejected_attempts": {
                "path": retry_outputs["rejected_attempts"],
                "sha256": sha256_file(Path(retry_outputs["rejected_attempts"])),
            },
            "figure": {
                "path": str(figure_path),
                "sha256": sha256_file(figure_path),
            },
        },
        "source_artifacts": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in source_artifacts
        ],
        "release_boundary": (
            "This proves the G8 pipeline with reused excluded pilot evidence and "
            "controlled harness failures. It adds no behavioral observations."
        ),
    }
    if not all(receipt["checks"].values()):
        receipt["status"] = "blocked"
        receipt["passed"] = False
    receipt_path = output_dir / "g8_receipt.json"
    receipt_path.write_bytes(canonical_json(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--pilot-queue", type=Path, required=True)
    parser.add_argument("--select", action="append", default=[])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--scorer-sha256", required=True)
    parser.add_argument("--scheduler-receipt", type=Path, action="append", default=[])
    parser.add_argument(
        "--invalid-attempt-receipt",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_rehearsal(
        raw_root=args.raw_root.resolve(),
        pilot_queue_path=args.pilot_queue.resolve(),
        selections=parse_selections(args.select),
        config_path=args.config.resolve(),
        protocol_sha256=args.protocol_sha256,
        scorer_sha256=args.scorer_sha256,
        scheduler_receipts=[
            path.resolve() for path in args.scheduler_receipt
        ],
        invalid_attempt_receipts=[
            path.resolve() for path in args.invalid_attempt_receipt
        ],
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps({"status": receipt["status"], "output": str(args.output_dir)}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
