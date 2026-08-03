#!/usr/bin/env python3
"""Render the paired GR00T endpoint-redirection result as a compact SVG."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    pairs = registry["paired_directional_evidence"]["pairs"]
    if len(pairs) != 3 or not all(p["requested_ordering_aligned"] for p in pairs):
        raise ValueError("Expected three aligned frozen endpoint pairs")

    width, height = 820, 480
    left, right, top, bottom = 90, 770, 70, 390
    y_min, y_max = -0.02, 0.08

    def sx(index: int) -> float:
        return left + (right - left) * (index + 1) / 4

    def sy(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="410" y="30" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">GR00T N1.7 DROID: paired endpoint redirection</text>',
        '<text x="410" y="52" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#444">All six episodes failed; all three LEFT endpoints remained above matched RIGHT endpoints</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222"/>',
    ]
    for tick in (-0.02, 0.0, 0.02, 0.04, 0.06, 0.08):
        y = sy(tick)
        lines.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#ddd"/>',
                f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick:+.2f}</text>',
            ]
        )
    lines.append(
        f'<text x="24" y="{(top + bottom) / 2}" transform="rotate(-90 24 {(top + bottom) / 2})" text-anchor="middle" font-family="sans-serif" font-size="13">cube − bowl lateral y (m; LEFT-positive)</text>'
    )
    for i, pair in enumerate(pairs):
        x = sx(i)
        left_y = float(pair["left_endpoint_lateral_y"])
        right_y = float(pair["right_endpoint_lateral_y"])
        lines.extend(
            [
                f'<line x1="{x:.1f}" y1="{sy(left_y):.1f}" x2="{x:.1f}" y2="{sy(right_y):.1f}" stroke="#888" stroke-width="2"/>',
                f'<circle cx="{x:.1f}" cy="{sy(left_y):.1f}" r="7" fill="#1f77b4"/>',
                f'<circle cx="{x:.1f}" cy="{sy(right_y):.1f}" r="7" fill="#d62728"/>',
                f'<text x="{x:.1f}" y="{bottom + 26}" text-anchor="middle" font-family="sans-serif" font-size="13">seed {pair["environment_seed"]}</text>',
            ]
        )
    lines.extend(
        [
            '<circle cx="290" cy="438" r="6" fill="#1f77b4"/><text x="304" y="443" font-family="sans-serif" font-size="13">LEFT prompt</text>',
            '<circle cx="455" cy="438" r="6" fill="#d62728"/><text x="469" y="443" font-family="sans-serif" font-size="13">RIGHT prompt</text>',
            "</svg>",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    manifest = {
        "schema_version": "vla-wam-shared-v2-groot-endpoint-figure-v1",
        "source_registry": str(args.registry),
        "source_registry_sha256": _sha256(args.registry),
        "figure": str(args.output),
        "figure_sha256": _sha256(args.output),
        "pair_count": 3,
        "aligned_pair_count": 3,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
