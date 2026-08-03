#!/usr/bin/env python3
"""Compile the frozen v2 direct-command evidence into an arena-separated table.

The compiler intentionally reads only compact, committed evidence.  It fails
closed when any source byte hash or expected result changes, so updating the
comparison to a later evidence boundary requires an explicit source audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
from pathlib import Path
from typing import Any


EVIDENCE_HEAD = "a7bed6e4381106ce9a59132775953f0e7ba68b67"
EVIDENCE_CUTOFF_UTC = "2026-08-03T19:45:12Z"
RESULT_DIR = Path("artifacts/vla_wam_shared_v2/results")
FIGURE_DIR = Path("artifacts/vla_wam_shared_v2/figures")

SOURCES = {
    "pi0": (
        "artifacts/vla_wam_shared_v2/pilot/results/pi0_fast_direct_confirmation.json",
        "491c74812ed0e4d36c16f8e0ded17a70af3e69740c9bcb87af129bb6d9563073",
    ),
    "groot": (
        "artifacts/vla_wam_shared_v2/pilot/expansion/groot_n17_droid_v2_registry.json",
        "95077a42bb0115bc673ea13ae5acdc6fdef6f476627804662f73c219ebd88bc7",
    ),
    "cosmos": (
        "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_edge_droid_direct_gate.json",
        "1c559ee5667ac9d22d7b66eafa7a65551783eedaf7fb3de29a2faf450c2dd029",
    ),
    "cosmos_invalid": (
        "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_edge_droid_invalid_attempts.json",
        "b3a62c792c82d15143ef6c94b768e2bcf712dd69d9c2f96584c904140a452754",
    ),
    "lingbot_vla": (
        "artifacts/vla_wam_shared_v2/pilot/expansion/lingbot_vla_4b_direct_gate.json",
        "7c0ad19833d6cbb51bb5fbdac8f9546f0e311333e498a15594b55f68dc7b6534",
    ),
    "lingbot_vla_readiness": (
        "artifacts/vla_wam_shared_v2/pilot/expansion/lingbot_vla_4b_robotwin_readiness.json",
        "588699ed912fe900de5a5ca36c350236e15ae0b0a2c55afeea22670759b68c30",
    ),
    "efficient_pair03": (
        "artifacts/vla_wam_shared_v2/pilot/directional_confirmation/efficient_wam_rt_pair03_integration.json",
        "d850f8f2c3d32db705cc65326ca673c150ab7208a203a180bd32d88dfb0e5471",
    ),
    "efficient_04_09": (
        "artifacts/vla_wam_shared_v2/pilot/directional_confirmation/efficient_wam_rt_pairs04_09_slice.json",
        "e7d2d3791323fecedb49e8c1ecc8fda1e0ade91dedd6613b9079f7b290e1fd54",
    ),
    "fastwam": (
        "artifacts/vla_wam_shared_v2/pilot/directional_confirmation/fastwam_pairs03_09_slice.json",
        "9d52920cded17d0f61c997f02624ea38ffc4c9a3536cdacf9f71115b151d14be",
    ),
    "lingbot_va": (
        "artifacts/vla_wam_shared_v2/pilot/directional_confirmation/lingbot_va_pairs03_09_slice.json",
        "8617da77c819ea57d374f463a672bf73414b088bb8188b9b13a6c1c2e1fb9d85",
    ),
    "light_wam": (
        "artifacts/vla_wam_shared_v2/pilot/expansion/light_wam_robotwin_direct_gate.json",
        "f33e1ff8fdc82c4a035f2cc113b91b311d685d9747dbcbb1104453fc455745d6",
    ),
    "light_wam_registry": (
        "artifacts/vla_wam_shared_v2/pilot/expansion/light_wam_robotwin_registry.json",
        "c316bbebe8aa73cfe86b8749cb3cf6d8ebf438389a57b1d9e6c52dbfee67bbb5",
    ),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_sources(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    documents: dict[str, Any] = {}
    ledger: dict[str, dict[str, Any]] = {}
    for key, (relative_path, expected_sha) in SOURCES.items():
        path = root / relative_path
        payload = path.read_bytes()
        actual_sha = sha256_bytes(payload)
        require(actual_sha == expected_sha, f"source hash changed: {relative_path}: {actual_sha}")
        documents[key] = json.loads(payload)
        ledger[key] = {
            "path": relative_path,
            "bytes": len(payload),
            "sha256": actual_sha,
        }
    return documents, ledger


def source_set_sha256(ledger: dict[str, dict[str, Any]]) -> str:
    canonical = "".join(
        f"{item['path']}\0{item['sha256']}\n"
        for item in sorted(ledger.values(), key=lambda value: value["path"])
    )
    return sha256_bytes(canonical.encode("utf-8"))


def metric(count: int | None, trials: int | None, status: str = "measured") -> dict[str, Any]:
    return {"count": count, "trials": trials, "status": status}


def paired_from_summary(summary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    pairs = summary["paired_endpoint_responses"]
    aligned = sum(pair["endpoint_response_direction"] == "aligned" for pair in pairs)
    measured_actions = [
        pair for pair in pairs if pair.get("first_ten_executed_action_rms") is not None
    ]
    distinct = sum(pair["first_ten_executed_action_rms"] > 0 for pair in measured_actions)
    require(aligned == summary["aligned_endpoint_pairs"], "endpoint summary mismatch")
    return metric(aligned, len(pairs)), metric(distinct, len(measured_actions))


def build_rows(d: dict[str, Any], ledger: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    pi0 = d["pi0"]
    s = pi0["summary"]
    require((s["episode_count"], s["by_direction"]["left"]["successes"], s["by_direction"]["right"]["successes"]) == (20, 1, 10), "pi0 result mismatch")
    rows.append({
        "model_id": "pi0_fast_droid_vla", "model": "π0-FAST DROID", "model_class": "VLA",
        "arena": "DROID / RoboLab", "arena_id": "droid_robolab", "valid_n": 20,
        "left_success": metric(1, 10), "right_success": metric(10, 10),
        "paired_endpoint_alignment": metric(s["aligned_endpoint_pairs"], s["pair_count"]),
        "paired_action_distinctness": metric(s["nonzero_first_chunk_pairs"], s["pair_count"]),
        "future_interface": "none", "future_evidence_status": "not_applicable_never_zero",
        "invalid_attempt_count": 0, "invalid_attempt_unit": "cell_attempts",
        "model_revision": None, "source_keys": ["pi0"],
        "claim_boundary": "Frozen ten-seed direct-command directional confirmation.",
    })

    groot = d["groot"]
    gs = groot["direction_summary"]
    gp = groot["paired_directional_evidence"]
    require((groot["design"]["valid_episode_count"], gs["LEFT"]["successes"], gs["RIGHT"]["successes"]) == (6, 0, 0), "GR00T result mismatch")
    rows.append({
        "model_id": "groot_n17_droid", "model": "GR00T N1.7 DROID", "model_class": "VLA",
        "arena": "DROID / RoboLab", "arena_id": "droid_robolab", "valid_n": 6,
        "left_success": metric(0, 3), "right_success": metric(0, 3),
        "paired_endpoint_alignment": metric(gp["endpoint_requested_ordering_aligned_pair_count"], gp["pair_count"]),
        "paired_action_distinctness": metric(gp["action_different_pair_count"], gp["pair_count"]),
        "future_interface": "none", "future_evidence_status": "not_applicable_never_zero",
        "invalid_attempt_count": groot["invalid_attempts"]["ledger_entry_count"],
        "invalid_attempt_unit": "ledger_entries; 2 behavior cells excluded",
        "model_revision": groot["model"]["revision"], "source_keys": ["groot"],
        "claim_boundary": "V2-A005 bounded six-cell direct-command replication.",
    })

    cosmos = d["cosmos"]
    cs = cosmos["summary"]
    require((cs["episode_count"], cs["by_direction"]["left"]["successes"], cs["by_direction"]["right"]["successes"]) == (6, 3, 3), "Cosmos result mismatch")
    cosmos_invalid_count = len(d["cosmos_invalid"]["attempts"])
    rows.append({
        "model_id": "cosmos3_edge_droid_wam", "model": "Cosmos3 Edge DROID", "model_class": "WAM",
        "arena": "DROID / RoboLab", "arena_id": "droid_robolab", "valid_n": 6,
        "left_success": metric(3, 3), "right_success": metric(3, 3),
        "paired_endpoint_alignment": metric(cs["aligned_endpoint_pairs"], cs["pair_count"]),
        "paired_action_distinctness": metric(sum(pair["executed_actions_distinct"] for pair in cosmos["pairs"]), cs["pair_count"]),
        "future_interface": cosmos["measurement"]["future_interface"], "future_evidence_status": "exposed_and_retained",
        "invalid_attempt_count": cosmos_invalid_count, "invalid_attempt_unit": "setup_attempts; all before model request",
        "model_revision": cosmos["checkpoint_revision"], "source_keys": ["cosmos", "cosmos_invalid"],
        "claim_boundary": "V2-A005 six-cell behavioral replication; separate from v1.",
    })

    lingbot = d["lingbot_vla"]
    ls = lingbot["summary"]
    aligned = sum(pair["endpoint_response_direction"] == "aligned" for pair in ls["paired_endpoint_responses"])
    readiness = d["lingbot_vla_readiness"]
    require((ls["episode_count"], ls["by_direction"]["left"]["successes"], ls["by_direction"]["right"]["successes"], aligned) == (6, 1, 0, 2), "LingBot-VLA result mismatch")
    rows.append({
        "model_id": "lingbot_vla_4b_robotwin", "model": "LingBot-VLA 4B", "model_class": "VLA",
        "arena": "RoboTwin place-A-relative-to-B", "arena_id": "robotwin_place_a2b", "valid_n": 6,
        "left_success": metric(1, 3), "right_success": metric(0, 3),
        "paired_endpoint_alignment": metric(aligned, 3),
        "paired_action_distinctness": metric(None, None, "not_reported_by_compiled_gate"),
        "future_interface": readiness["interface_contract"]["future_interface"], "future_evidence_status": "not_applicable_never_zero",
        "invalid_attempt_count": len(readiness["technical_setup_attempts_excluded_from_behavior_denominator"]),
        "invalid_attempt_unit": "technical_setup_attempts",
        "model_revision": readiness["model"]["checkpoint_revision"], "source_keys": ["lingbot_vla", "lingbot_vla_readiness"],
        "claim_boundary": "V2-A005 bounded six-cell direct-command gate.",
    })

    efficient03 = d["efficient_pair03"]
    efficient = d["efficient_04_09"]
    es = efficient["summary"]
    endpoint, actions = paired_from_summary(es)
    endpoint["trials"] += 1
    actions["trials"] += 1
    actions["count"] += int(efficient03["paired_metrics"]["first_ten_executed_action_rms"] > 0)
    require(efficient03["paired_metrics"]["endpoint_ordering"] == "anti_aligned", "Efficient pair03 endpoint mismatch")
    require((es["episode_count"], es["by_direction"]["left"]["successes"], es["by_direction"]["right"]["successes"], endpoint["count"], actions["count"]) == (12, 3, 2, 6, 7), "Efficient result mismatch")
    rows.append({
        "model_id": "efficient_wam_rt_robotwin", "model": "Efficient-WAM-RT", "model_class": "WAM",
        "arena": "RoboTwin place-A-relative-to-B", "arena_id": "robotwin_place_a2b", "valid_n": 14,
        "left_success": metric(3, 7), "right_success": metric(2, 7),
        "paired_endpoint_alignment": endpoint, "paired_action_distinctness": actions,
        "future_interface": "decoded_future_video", "future_evidence_status": "exposed_and_retained",
        "invalid_attempt_count": es["invalid_attempt_count"] + efficient03["pair"]["invalid_attempt_count"], "invalid_attempt_unit": "cell_attempts",
        "model_revision": efficient03["model_repository_commit"], "source_keys": ["efficient_pair03", "efficient_04_09"],
        "claim_boundary": "Prospective pairs03–09 only; historical pairs00–02 are not merged into this row.",
    })

    for key, label, future, future_status in (
        ("fastwam", "FastWAM", "action_only_at_test_time", "not_applicable_never_zero"),
        ("lingbot_va", "LingBot-VA", "latent_only_future_not_decodable", "latent_retained_not_scored_never_zero"),
    ):
        doc = d[key]
        summary = doc["summary"]
        endpoint, actions = paired_from_summary(summary)
        expected = (14, 1, 1, 3, 7) if key == "fastwam" else (14, 3, 4, 6, 7)
        actual = (summary["episode_count"], summary["by_direction"]["left"]["successes"], summary["by_direction"]["right"]["successes"], endpoint["count"], actions["count"])
        require(actual == expected, f"{label} result mismatch: {actual}")
        rows.append({
            "model_id": doc["model_id"], "model": label, "model_class": "WAM",
            "arena": "RoboTwin place-A-relative-to-B", "arena_id": "robotwin_place_a2b", "valid_n": 14,
            "left_success": metric(expected[1], 7), "right_success": metric(expected[2], 7),
            "paired_endpoint_alignment": endpoint, "paired_action_distinctness": actions,
            "future_interface": future, "future_evidence_status": future_status,
            "invalid_attempt_count": summary["invalid_attempt_count"], "invalid_attempt_unit": "cell_attempts",
            "model_revision": None, "source_keys": [key],
            "claim_boundary": "Prospective pairs03–09 only; historical pairs00–02 are not merged into this row.",
        })

    light = d["light_wam"]
    light_registry = d["light_wam_registry"]
    require((light["valid_episode_count"], light["success_by_relation"]["left"]["successes"], light["success_by_relation"]["right"]["successes"]) == (6, 1, 0), "Light-WAM result mismatch")
    rows.append({
        "model_id": "light_wam_robotwin", "model": "Light-WAM", "model_class": "WAM",
        "arena": "RoboTwin place-A-relative-to-B", "arena_id": "robotwin_place_a2b", "valid_n": 6,
        "left_success": metric(1, 3), "right_success": metric(0, 3),
        "paired_endpoint_alignment": metric(sum(pair["endpoint_ordering_aligned"] for pair in light["pairs"]), len(light["pairs"])),
        "paired_action_distinctness": metric(sum(pair["executed_actions_distinct"] for pair in light["pairs"]), len(light["pairs"])),
        "future_interface": light["future_interface"], "future_evidence_status": "not_applicable_never_zero",
        "invalid_attempt_count": light["invalid_attempt_count"], "invalid_attempt_unit": "cell_attempts",
        "model_revision": light_registry["checkpoint"]["revision"], "source_keys": ["light_wam", "light_wam_registry"],
        "claim_boundary": "V2-A006 bounded six-cell direct-command gate.",
    })

    for row in rows:
        row["sources"] = [ledger[key] for key in row.pop("source_keys")]
    return rows


def ratio(metric_value: dict[str, Any]) -> str:
    if metric_value["count"] is None:
        return "NR"
    return f"{metric_value['count']}/{metric_value['trials']}"


def csv_text(rows: list[dict[str, Any]], source_set: str) -> str:
    fields = [
        "source_set_sha256", "arena", "model_class", "model", "model_id", "model_revision",
        "valid_n", "left_successes", "left_trials", "right_successes", "right_trials",
        "endpoint_aligned_pairs", "endpoint_pairs", "action_distinct_pairs", "action_pairs",
        "action_metric_status", "future_interface", "future_evidence_status", "invalid_attempt_count",
        "invalid_attempt_unit", "claim_boundary", "evidence_sources",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "source_set_sha256": source_set, "arena": row["arena"], "model_class": row["model_class"],
            "model": row["model"], "model_id": row["model_id"], "model_revision": row["model_revision"] or "not_recorded_in_selected_result",
            "valid_n": row["valid_n"], "left_successes": row["left_success"]["count"], "left_trials": row["left_success"]["trials"],
            "right_successes": row["right_success"]["count"], "right_trials": row["right_success"]["trials"],
            "endpoint_aligned_pairs": row["paired_endpoint_alignment"]["count"], "endpoint_pairs": row["paired_endpoint_alignment"]["trials"],
            "action_distinct_pairs": row["paired_action_distinctness"]["count"], "action_pairs": row["paired_action_distinctness"]["trials"],
            "action_metric_status": row["paired_action_distinctness"]["status"], "future_interface": row["future_interface"],
            "future_evidence_status": row["future_evidence_status"], "invalid_attempt_count": row["invalid_attempt_count"],
            "invalid_attempt_unit": row["invalid_attempt_unit"], "claim_boundary": row["claim_boundary"],
            "evidence_sources": ";".join(f"{item['path']}@sha256:{item['sha256']}" for item in row["sources"]),
        })
    return output.getvalue()


def markdown_text(rows: list[dict[str, Any]], source_set: str) -> str:
    lines = [
        "# Direct-command evidence across models",
        "",
        f"Evidence cutoff: `{EVIDENCE_HEAD}` ({EVIDENCE_CUTOFF_UTC}). Source-set SHA-256: `{source_set}`.",
        "",
        "This is an arena-separated descriptive comparison. Raw DROID and RoboTwin success rates are never pooled. `NR` means the selected compiled evidence did not report a paired action-distinctness statistic; it is not a zero. Infrastructure-invalid attempts remain outside every valid-model denominator.",
    ]
    for arena_id, title in (("droid_robolab", "DROID / RoboLab"), ("robotwin_place_a2b", "RoboTwin place-A-relative-to-B")):
        lines.extend(["", f"## {title}", "", "| Class | Model | Valid n | LEFT | RIGHT | Endpoint aligned | Actions distinct | Future interface | Invalid attempts |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |"])
        for row in rows:
            if row["arena_id"] != arena_id:
                continue
            invalid = f"{row['invalid_attempt_count']} ({row['invalid_attempt_unit']})"
            lines.append(f"| {row['model_class']} | {row['model']} | {row['valid_n']} | {ratio(row['left_success'])} | {ratio(row['right_success'])} | {ratio(row['paired_endpoint_alignment'])} | {ratio(row['paired_action_distinctness'])} | `{row['future_interface']}` | {invalid} |")
    lines.extend(["", "## Exact evidence sources", ""])
    seen: set[str] = set()
    for row in rows:
        for source in row["sources"]:
            if source["path"] in seen:
                continue
            seen.add(source["path"])
            lines.append(f"- `{source['path']}` — {source['bytes']:,} bytes; SHA-256 `{source['sha256']}`")
    lines.extend(["", "## Interpretation limits", "", "- Success is the frozen arena-specific requested-relation completion predicate.", "- Endpoint alignment and action distinctness are paired sensitivity measures, not task success.", "- Exposed decoded futures are retained; action-only, latent-only, and missing future interfaces are never converted into zero-valued future scores.", "- The three pairs03–09 RoboTwin WAM rows are prospective slices and do not synthesize or merge unavailable historical raw pairs00–02.", ""])
    return "\n".join(lines)


COLORS = {"VLA": "#2563EB", "WAM": "#EA580C", "LEFT": "#0F766E", "RIGHT": "#7C3AED"}


def svg_text(rows: list[dict[str, Any]], source_set: str, width: int, height: int) -> str:
    landscape = width > height
    title_y = 58
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#F8FAFC"/>', '<style>text{font-family:Inter,Arial,sans-serif;fill:#172033}.title{font-size:30px;font-weight:700}.subtitle{font-size:14px;fill:#526076}.panel{font-size:20px;font-weight:700}.model{font-size:15px;font-weight:650}.small{font-size:12px;fill:#526076}.metric{font-size:13px;font-weight:650}.badge{font-size:11px;font-weight:700;fill:white}.foot{font-size:11px;fill:#64748B}</style>', f'<text class="title" x="{width/2}" y="{title_y}" text-anchor="middle">Direct-command physical steerability evidence</text>', f'<text class="subtitle" x="{width/2}" y="{title_y+27}" text-anchor="middle">Arena-specific success; paired endpoint and action sensitivity shown separately</text>']

    def panel(panel_rows: list[dict[str, Any]], x: float, y: float, w: float, h: float, title: str) -> None:
        parts.extend([f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="white" stroke="#D8E0EA"/>', f'<text class="panel" x="{x+24}" y="{y+34}">{html.escape(title)}</text>', f'<text class="small" x="{x+w-24}" y="{y+33}" text-anchor="end">denominators stay within this panel</text>'])
        header_y = y + 62
        parts.append(f'<text class="small" x="{x+w*0.48}" y="{header_y}" text-anchor="middle">REQUESTED SUCCESS</text>')
        parts.append(f'<text class="small" x="{x+w*0.72}" y="{header_y}" text-anchor="middle">ENDPOINT</text>')
        parts.append(f'<text class="small" x="{x+w*0.86}" y="{header_y}" text-anchor="middle">ACTIONS</text>')
        row_h = (h - 82) / len(panel_rows)
        for index, row in enumerate(panel_rows):
            cy = y + 76 + row_h * (index + 0.5)
            if index:
                parts.append(f'<line x1="{x+20}" x2="{x+w-20}" y1="{cy-row_h/2}" y2="{cy-row_h/2}" stroke="#EEF2F6"/>')
            parts.append(f'<rect x="{x+22}" y="{cy-24}" width="40" height="18" rx="9" fill="{COLORS[row["model_class"]]}"/>')
            parts.append(f'<text class="badge" x="{x+42}" y="{cy-11}" text-anchor="middle">{row["model_class"]}</text>')
            parts.append(f'<text class="model" x="{x+22}" y="{cy+7}">{html.escape(row["model"])}</text>')
            parts.append(f'<text class="small" x="{x+22}" y="{cy+24}">valid n={row["valid_n"]}</text>')
            bar_x, bar_w = x + w * 0.34, w * 0.28
            for offset, direction in ((-10, "LEFT"), (10, "RIGHT")):
                m = row[f"{direction.lower()}_success"]
                frac = m["count"] / m["trials"]
                yy = cy + offset
                parts.append(f'<rect x="{bar_x}" y="{yy-6}" width="{bar_w}" height="8" rx="4" fill="#E7ECF2"/>')
                parts.append(f'<rect x="{bar_x}" y="{yy-6}" width="{bar_w*frac:.2f}" height="8" rx="4" fill="{COLORS[direction]}"/>')
                parts.append(f'<text class="small" x="{bar_x-7}" y="{yy+1}" text-anchor="end">{direction[0]}</text>')
                parts.append(f'<text class="metric" x="{bar_x+bar_w+8}" y="{yy+2}">{ratio(m)}</text>')
            for xpos, field in ((x+w*0.72, "paired_endpoint_alignment"), (x+w*0.86, "paired_action_distinctness")):
                m = row[field]
                fill = "#ECFDF5" if m["count"] is not None else "#F1F5F9"
                parts.append(f'<rect x="{xpos-37}" y="{cy-18}" width="74" height="36" rx="10" fill="{fill}" stroke="#D8E0EA"/>')
                parts.append(f'<text class="metric" x="{xpos}" y="{cy+5}" text-anchor="middle">{ratio(m)}</text>')
            parts.append(f'<text class="small" x="{x+w-20}" y="{cy+4}" text-anchor="end">invalid {row["invalid_attempt_count"]}</text>')

    droid = [row for row in rows if row["arena_id"] == "droid_robolab"]
    robotwin = [row for row in rows if row["arena_id"] == "robotwin_place_a2b"]
    if landscape:
        panel( droid, 45, 115, 735, 690, "DROID / RoboLab")
        panel(robotwin, 820, 115, 735, 690, "RoboTwin place-A-relative-to-B")
        foot_y = 840
    else:
        panel(droid, 45, 112, 1110, 392, "DROID / RoboLab")
        panel(robotwin, 45, 530, 1110, 575, "RoboTwin place-A-relative-to-B")
        foot_y = 1140
    parts.append(f'<text class="foot" x="{width/2}" y="{foot_y}" text-anchor="middle">LEFT/RIGHT = requested-relation success · ENDPOINT = aligned paired final ordering · ACTIONS = distinct paired traces · NR = not reported</text>')
    parts.append(f'<text class="foot" x="{width/2}" y="{foot_y+20}" text-anchor="middle">Invalid setup/partial attempts are excluded from valid n · source set {source_set[:16]}… · evidence {EVIDENCE_HEAD[:8]}</text>')
    parts.append("</svg>\n")
    return "".join(parts)


def write_outputs(root: Path, rows: list[dict[str, Any]], ledger: dict[str, dict[str, Any]]) -> None:
    source_set = source_set_sha256(ledger)
    result_dir = root / RESULT_DIR
    figure_dir = root / FIGURE_DIR
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    comparison = {
        "schema_version": "vla-wam-shared-v2-cross-model-direct-comparison-v1",
        "evidence_cutoff_git_commit": EVIDENCE_HEAD,
        "evidence_cutoff_utc": EVIDENCE_CUTOFF_UTC,
        "source_set_sha256": source_set,
        "denominator_policy": "DROID and RoboTwin remain separate arenas; no raw success rate is pooled across them.",
        "missing_evidence_policy": "Missing, action-only, and latent-only futures are never zeros. Unreported paired action metrics are null/NR, never zero.",
        "rows": rows,
        "source_ledger": sorted(ledger.values(), key=lambda value: value["path"]),
    }
    outputs = {
        RESULT_DIR / "direct_command_cross_model_comparison.json": (json.dumps(comparison, indent=2, sort_keys=True) + "\n").encode(),
        RESULT_DIR / "direct_command_cross_model_comparison.csv": csv_text(rows, source_set).encode(),
        RESULT_DIR / "direct_command_cross_model_comparison.md": markdown_text(rows, source_set).encode(),
        FIGURE_DIR / "direct_command_cross_model_comparison_1600x900.svg": svg_text(rows, source_set, 1600, 900).encode(),
        FIGURE_DIR / "direct_command_cross_model_comparison_1200x1200.svg": svg_text(rows, source_set, 1200, 1200).encode(),
    }
    for relative, payload in outputs.items():
        (root / relative).write_bytes(payload)
    manifest = {
        "schema_version": "vla-wam-shared-v2-cross-model-direct-comparison-manifest-v1",
        "evidence_cutoff_git_commit": EVIDENCE_HEAD,
        "source_set_sha256": source_set,
        "outputs": [
            {"path": str(path), "bytes": len(payload), "sha256": sha256_bytes(payload)}
            for path, payload in sorted(outputs.items(), key=lambda item: str(item[0]))
        ],
    }
    (result_dir / "direct_command_cross_model_comparison_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.repo_root.resolve()
    documents, ledger = load_sources(root)
    rows = build_rows(documents, ledger)
    require(len(rows) == 8, f"expected eight measured model rows, got {len(rows)}")
    require(sum(row["valid_n"] for row in rows if row["arena_id"] == "droid_robolab") == 32, "DROID valid-n audit failed")
    require(sum(row["valid_n"] for row in rows if row["arena_id"] == "robotwin_place_a2b") == 54, "RoboTwin valid-n audit failed")
    write_outputs(root, rows, ledger)


if __name__ == "__main__":
    main()
