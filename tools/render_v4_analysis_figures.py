#!/usr/bin/env python3
"""Render compact SVG figures from V4 partial-analysis table exports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bar_svg(*, title: str, labels: list[str], values: list[int], colors: list[str]) -> str:
    width, height = 720, 400
    margin_x, base_y = 90, 320
    plot_w = width - 2 * margin_x
    max_value = max(values) if values else 1
    bar_w = plot_w / max(len(values), 1)
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width // 2}" y="28" text-anchor="middle" font-size="18">{title}</text>',
        f'<line x1="{margin_x}" y1="{base_y}" x2="{width - margin_x}" y2="{base_y}" stroke="black"/>',
    ]
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        bar_h = 0 if max_value == 0 else int(240 * value / max_value)
        x = margin_x + int(index * bar_w + bar_w * 0.15)
        w = int(bar_w * 0.7)
        y = base_y - bar_h
        cx = x + w // 2
        rows.append(f'<rect x="{x}" y="{y}" width="{w}" height="{bar_h}" fill="{color}"/>')
        rows.append(f'<text x="{cx}" y="{y - 8}" text-anchor="middle">{value}</text>')
        rows.append(f'<text x="{cx}" y="{base_y + 25}" text-anchor="middle">{label}</text>')
    rows.append("</svg>")
    return "\n".join(rows) + "\n"


def render_figures(tables_dir: Path, out_dir: Path) -> dict[str, dict[str, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage_rows = _read_csv_rows(tables_dir / "coverage_by_cell.csv")
    failure_rows = _read_csv_rows(tables_dir / "failure_composition.csv")

    if coverage_rows:
        accepted = sum(int(row.get("accepted_valid") or 0) for row in coverage_rows)
        infra = sum(int(row.get("infra_invalid") or 0) for row in coverage_rows)
        missing = sum(int(row.get("missing_valid") or 0) for row in coverage_rows)
        blocked = sum(int(row.get("blocked") or 0) for row in coverage_rows)
        labels = ["accepted_valid", "infra_invalid", "missing_valid", "blocked"]
        values = [accepted, infra, missing, blocked]
        colors = ["#315aa6", "#b45309", "#6b7280", "#9ca3af"]
    else:
        labels = ["behavioral_scope_empty"]
        values = [0]
        colors = ["#6b7280"]

    coverage_svg = _bar_svg(
        title="V4 scoped coverage status",
        labels=labels,
        values=values,
        colors=colors,
    )

    if failure_rows:
        totals: dict[str, int] = {}
        for row in failure_rows:
            label = row.get("failure_label") or row.get("label") or "unknown"
            totals[label] = totals.get(label, 0) + int(row.get("count") or row.get("n") or 1)
        fail_labels = list(totals)
        fail_values = [totals[label] for label in fail_labels]
        fail_colors = ["#dc2626"] * len(fail_labels)
    else:
        fail_labels = ["excluded_from_behavioral_claims"]
        fail_values = [0]
        fail_colors = ["#6b7280"]

    failure_svg = _bar_svg(
        title="V4 failure composition",
        labels=fail_labels,
        values=fail_values,
        colors=fail_colors,
    )

    outputs = {
        "coverage_status.svg": coverage_svg,
        "failure_composition.svg": failure_svg,
    }
    manifest: dict[str, dict[str, str]] = {}
    for name, body in outputs.items():
        path = out_dir / name
        path.write_text(body, encoding="utf-8")
        manifest[name.replace(".svg", "")] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    manifest_path = out_dir / "figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = render_figures(args.tables.resolve(), args.out.resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
