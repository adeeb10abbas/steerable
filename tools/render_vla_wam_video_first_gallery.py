#!/usr/bin/env python3
"""Render and validate the video-first VLA/WAM evidence gallery.

The committed gallery manifest is the metadata source. DreamZero execution and
imagination entries are optionally loaded from separate canonical manifests
only after their media hashes validate. Model-predicted video is rendered in a
dedicated section and is never presented as simulator execution.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "artifacts/vla_wam_shared_v2/media/video_first_gallery_manifest.json"
DEFAULT_HTML = REPO_ROOT / "docs/VLA_WAM_STEERABILITY_VIDEO_GALLERY.html"
DEFAULT_MARKDOWN = REPO_ROOT / "docs/VLA_WAM_STEERABILITY_VIDEO_GALLERY.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_file(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    path.relative_to(REPO_ROOT.resolve())
    return path


def validate_file(record: dict[str, Any], label: str) -> None:
    path = repo_file(record["path"])
    if not path.is_file():
        raise SystemExit(f"missing {label}: {record['path']}")
    actual_bytes = path.stat().st_size
    if actual_bytes != record["bytes"]:
        raise SystemExit(
            f"byte mismatch for {label}: {record['path']} "
            f"expected={record['bytes']} actual={actual_bytes}"
        )
    actual_sha = sha256(path)
    if actual_sha != record["sha256"]:
        raise SystemExit(
            f"SHA-256 mismatch for {label}: {record['path']} "
            f"expected={record['sha256']} actual={actual_sha}"
        )


def validate_entry(entry: dict[str, Any], required_fields: list[str] | None = None) -> None:
    required = required_fields or [
        "id",
        "arena",
        "arena_label",
        "model_label",
        "category",
        "future_interface",
        "evidence_status",
        "pair_label",
        "seed",
        "video",
        "directions",
        "source_manifest",
    ]
    missing = [field for field in required if field not in entry]
    if missing:
        raise SystemExit(f"gallery entry {entry.get('id', '<unknown>')} missing: {missing}")
    if entry["arena"] not in {"droid", "robotwin"}:
        raise SystemExit(f"invalid arena on {entry['id']}: {entry['arena']}")
    if len(entry["directions"]) != 2:
        raise SystemExit(f"gallery entry {entry['id']} must have exactly two directions")
    for direction in entry["directions"]:
        missing_direction = [field for field in ("relation", "prompt", "outcome") if field not in direction]
        if missing_direction:
            raise SystemExit(
                f"gallery entry {entry['id']} has direction missing: {missing_direction}"
            )
    relations = {direction.get("relation") for direction in entry["directions"]}
    if relations != {"LEFT", "RIGHT"}:
        raise SystemExit(f"gallery entry {entry['id']} must contain LEFT and RIGHT")
    validate_file(entry["video"], f"{entry['id']} video")
    for optional in ("poster", "captions"):
        if optional in entry:
            validate_file(entry[optional], f"{entry['id']} {optional}")
    source = repo_file(entry["source_manifest"])
    if not source.is_file():
        raise SystemExit(f"missing source manifest for {entry['id']}: {entry['source_manifest']}")


def load_entries(
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool, list[dict[str, Any]], list[dict[str, Any]], bool]:
    entries = list(manifest["entries"])
    for entry in entries:
        validate_entry(entry)

    contract = manifest["dreamzero_manifest_contract"]
    dreamzero_path = repo_file(contract["path"])
    dreamzero_present = dreamzero_path.is_file()
    if dreamzero_present:
        dreamzero = json.loads(dreamzero_path.read_text())
        key = contract["gallery_entries_key"]
        dreamzero_entries = dreamzero.get(key)
        if not isinstance(dreamzero_entries, list) or not dreamzero_entries:
            raise SystemExit(
                f"DreamZero manifest exists but has no non-empty {key!r}: {contract['path']}"
            )
        for entry in dreamzero_entries:
            validate_entry(entry, contract["required_entry_fields"])
            if entry["arena"] != "droid" or "dreamzero" not in entry["id"].lower():
                raise SystemExit(f"non-DreamZero entry in canonical DreamZero manifest: {entry['id']}")
        entries = dreamzero_entries + entries
    ids = [entry["id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate gallery entry id")

    imagination_entries: list[dict[str, Any]] = []
    official_decodes: list[dict[str, Any]] = []
    imagination_contract = manifest["dreamzero_imagination_manifest_contract"]
    imagination_path = repo_file(imagination_contract["path"])
    imagination_present = imagination_path.is_file()
    if imagination_present:
        imagination = json.loads(imagination_path.read_text())
        if (
            imagination.get("schema_version")
            != "vla-wam-shared-v2-dreamzero-imagination-media-v1"
            or imagination.get("status") != "complete_all_official_decodes_archived"
            or imagination.get("official_decode_count")
            != imagination_contract["required_official_decode_count"]
            or imagination.get("behavioral_decode_count")
            != imagination_contract["required_behavioral_decode_count"]
            or imagination.get("fixed_observation_probe_decode_count")
            != imagination_contract["required_fixed_observation_probe_decode_count"]
        ):
            raise SystemExit("DreamZero imagination manifest contract mismatch")
        imagination_entries = imagination.get(imagination_contract["gallery_entries_key"], [])
        official_decodes = imagination.get(imagination_contract["archive_entries_key"], [])
        if len(imagination_entries) != 3 or len(official_decodes) != 9:
            raise SystemExit("DreamZero imagination manifest has incomplete media lists")
        for entry in imagination_entries:
            validate_entry(entry)
            if entry.get("media_kind") != "model_prediction_not_execution":
                raise SystemExit(f"DreamZero imagination entry lacks prediction label: {entry['id']}")
        for record in official_decodes:
            if record.get("scope") not in {
                "valid_behavioral_episode",
                "fixed_observation_diagnostic",
            }:
                raise SystemExit(f"Invalid DreamZero decode scope: {record.get('id')}")
            validate_file(record["archived_video"], f"{record['id']} archived official decode")
        all_ids = ids + [entry["id"] for entry in imagination_entries]
        if len(all_ids) != len(set(all_ids)):
            raise SystemExit("duplicate execution/imagination gallery entry id")
    return entries, dreamzero_present, imagination_entries, official_decodes, imagination_present


def rel(path: str) -> str:
    return "../" + path


def direction_card(direction: dict[str, Any]) -> str:
    relation = html.escape(direction["relation"])
    css = direction["relation"].lower()
    return (
        f'<div class="direction {css}"><div class="direction-top">'
        f'<strong>{relation}</strong><span>{html.escape(direction["outcome"])}</span></div>'
        f'<blockquote>“{html.escape(direction["prompt"])}”</blockquote></div>'
    )


def entry_card(entry: dict[str, Any]) -> str:
    video = rel(entry["video"]["path"])
    poster_path = rel(entry["poster"]["path"]) if "poster" in entry else ""
    poster = f' poster="{html.escape(poster_path)}"' if poster_path else ""
    captions = ""
    if "captions" in entry:
        captions_path = html.escape(rel(entry["captions"]["path"]))
        captions = (
            f'<track kind="captions" src="{captions_path}" '
            'srclang="en" label="English">'
        )
    notes = html.escape(entry.get("selection_note", ""))
    paired_control = entry.get("paired_control")
    paired_link = ""
    if paired_control:
        paired_link = (
            f' · <a href="{html.escape(paired_control["href"])}">'
            f'{html.escape(paired_control["label"])}</a>'
        )
    return f"""
      <article class="card" id="{html.escape(entry['id'])}">
        <header>
          <div><p class="overline">{html.escape(entry['category'])} · {html.escape(entry['pair_label'])}</p>
          <h3>{html.escape(entry['model_label'])}</h3></div>
          <span class="status">{html.escape(entry['evidence_status'])}</span>
        </header>
        <video controls preload="metadata"{poster}>
          <source src="{html.escape(video)}" type="video/mp4">{captions}
          <a href="{html.escape(video)}">Open the MP4 directly</a>.
        </video>
        <div class="directions">{''.join(direction_card(d) for d in entry['directions'])}</div>
        <dl class="facts">
          <div><dt>Arena</dt><dd>{html.escape(entry['arena_label'])}</dd></div>
          <div><dt>Future interface</dt><dd>{html.escape(entry['future_interface'])}</dd></div>
          <div><dt>Video SHA-256</dt><dd><code>{html.escape(entry['video']['sha256'])}</code></dd></div>
        </dl>
        <p class="note">{notes} <a href="{html.escape(video)}">Open video</a> · <a href="{html.escape(rel(entry['source_manifest']))}">Evidence manifest</a>{paired_link}</p>
      </article>"""


def missing_card(item: dict[str, Any]) -> str:
    source = item.get("expected_manifest") or item.get("behavioral_manifest")
    source_link = (
        f' <a href="{html.escape(rel(source))}">Expected/source manifest</a>.'
        if source and repo_file(source).exists()
        else f" Expected manifest: <code>{html.escape(source or 'none')}</code>."
    )
    return (
        '<article class="missing"><p class="overline">No substituted media</p>'
        f'<h3>{html.escape(item["model_id"])}</h3>'
        f'<p><strong>{html.escape(item["status"])}</strong> — {html.escape(item["reason"])}'
        f'{source_link}</p></article>'
    )


def imagination_archive(official_decodes: list[dict[str, Any]]) -> str:
    items = []
    for record in official_decodes:
        video = html.escape(rel(record["archived_video"]["path"]))
        scope = (
            "behavioral session"
            if record["scope"] == "valid_behavioral_episode"
            else "fixed-observation probe"
        )
        label = record["id"].replace("_", " ")
        items.append(
            f'<li><a href="{video}">{html.escape(label)}</a> '
            f'<span>({scope}; SHA-256 <code>{html.escape(record["archived_video"]["sha256"])}</code>)</span></li>'
        )
    return "<ul class=\"archive\">" + "".join(items) + "</ul>"


def render_html(
    manifest: dict[str, Any],
    entries: list[dict[str, Any]],
    dreamzero_present: bool,
    imagination_entries: list[dict[str, Any]],
    official_decodes: list[dict[str, Any]],
    imagination_present: bool,
) -> str:
    sections = []
    dreamzero_execution = [
        dict(
            entry,
            category="ACTUAL ROLLOUT / EXECUTED BEHAVIOR",
            paired_control={
                "href": f'#{entry["id"]}_imagined_futures',
                "label": "Open same-seed imagined-future control",
            },
        )
        for entry in entries
        if entry["id"].startswith("dreamzero_droid_seed")
    ]
    world_model_execution = dreamzero_execution + [
        entry
        for entry in entries
        if entry["category"] == "WAM"
        and not entry["id"].startswith("dreamzero_droid_seed")
    ]
    vla_execution = [entry for entry in entries if entry["category"] == "VLA"]
    section_specs = (
        (
            "world-model-execution",
            "WORLD MODELS — actual simulator execution",
            "Every card is an executed rollout. Its future interface states whether decoded, latent-only, or action-only future evidence was exposed.",
            world_model_execution,
        ),
        (
            "vla-execution",
            "VLAs — actual simulator execution",
            "Behavioral rollout evidence from action-producing policies; no imagined video is inferred when the interface exposes none.",
            vla_execution,
        ),
    )
    for section_id, title, intro, section_entries in section_specs:
        cards = "".join(entry_card(entry) for entry in section_entries)
        if section_id == "world-model-execution" and not dreamzero_present:
            contract = manifest["dreamzero_manifest_contract"]
            cards = f"""
      <article class="pending" id="dreamzero-pending">
        <p class="overline">RTX rollout target · evidence pending</p>
        <h3>DreamZero DROID</h3>
        <p>No valid DreamZero behavioral clip is committed yet. This is intentionally not a zero and not a placeholder rollout.</p>
        <p>When the RTX lane produces valid videos, the renderer will ingest hash-validated <code>gallery_entries</code> from <code>{html.escape(contract['path'])}</code>.</p>
      </article>""" + cards
        sections.append(
            f'<section id="{html.escape(section_id)}"><div class="section-head"><h2>{title}</h2><p>{intro}</p></div>'
            f'<div class="grid">{cards}</div></section>'
        )

    if imagination_present:
        imagination_cards = "".join(
            entry_card(dict(
                entry,
                category="IMAGINED FUTURE / MODEL PREDICTION — NOT EXECUTION",
                paired_control={
                    "href": f'#{entry["id"].removesuffix("_imagined_futures")}',
                    "label": "Open same-seed actual rollout control",
                },
            ))
            for entry in imagination_entries
        )
        imagination_section = (
            '<section id="dreamzero-imagination"><div class="section-head">'
            '<h2>DreamZero imagined futures — not execution</h2><p>IMAGINED FUTURES / MODEL PREDICTIONS.</p>'
            f'</div><p class="boundary"><strong>Prediction boundary.</strong> These MP4s decode DreamZero’s retained latent video predictions. '
            'They do not show what the robot actually executed and do not add behavioral trials.</p>'
            f'<div class="grid">{imagination_cards}</div>'
            '<div class="archive-box"><h3>All nine original official decodes</h3>'
            '<p>Six valid behavioral-session decodes and three fixed-observation diagnostic decodes are archived byte-for-byte.</p>'
            f'{imagination_archive(official_decodes)}</div></section>'
        )
    else:
        imagination_contract = manifest["dreamzero_imagination_manifest_contract"]
        imagination_section = (
            '<section id="dreamzero-imagination"><div class="section-head">'
            '<h2>DreamZero imagined futures</h2><p>Official model predictions — not simulator execution.</p></div>'
            '<article class="pending"><p class="overline">Imagination archive pending</p>'
            '<h3>No committed official decodes yet</h3>'
            f'<p>The renderer will ingest the bounded archive from <code>{html.escape(imagination_contract["path"])}</code> '
            'only after all nine files validate.</p></article></section>'
        )

    missing_items = [
        item for item in manifest["missing_publication_media"]
        if not (dreamzero_present and item["model_id"] == "dreamzero_droid")
    ]
    missing = "".join(missing_card(item) for item in missing_items)
    manifest_digest = sha256(DEFAULT_MANIFEST)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(manifest['title'])}</title>
<style>
:root{{--ink:#17202a;--muted:#596775;--paper:#f4f1ea;--card:#fff;--line:#d8d8d2;--left:#fff0d4;--right:#e4f1ff;--accent:#6941c6}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{width:min(1420px,calc(100% - 32px));margin:auto;padding:52px 0 80px}}h1{{max-width:980px;margin:.12em 0;font-size:clamp(42px,7vw,84px);line-height:.98;letter-spacing:-.05em}}h2{{font-size:clamp(31px,4vw,48px);margin:0}}h3{{font-size:25px;margin:2px 0 0}}.lede{{max-width:920px;color:var(--muted);font-size:20px}}.boundary{{padding:16px 20px;border-left:5px solid var(--accent);background:#fff;border-radius:0 12px 12px 0;max-width:1050px}}section{{margin-top:62px}}.section-head{{display:flex;align-items:end;justify-content:space-between;gap:24px;margin-bottom:20px;border-bottom:1px solid var(--line);padding-bottom:14px}}.section-head p{{color:var(--muted);margin:0}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}}article{{background:var(--card);border:1px solid var(--line);border-radius:16px;overflow:hidden}}article header{{padding:20px 22px 14px;display:flex;justify-content:space-between;gap:18px}}.overline{{margin:0;color:var(--accent);text-transform:uppercase;letter-spacing:.06em;font-size:12px;font-weight:800}}.status{{max-width:46%;color:var(--muted);font-size:12px;text-align:right}}video{{display:block;width:100%;max-height:560px;background:#111}}.directions{{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:16px 18px 10px}}.direction{{padding:14px;border-radius:10px}}.direction.left{{background:var(--left)}}.direction.right{{background:var(--right)}}.direction-top{{display:flex;justify-content:space-between;gap:10px;font-size:13px}}blockquote{{margin:10px 0 0;font-weight:650}}.facts{{display:grid;grid-template-columns:1fr 1.4fr;gap:1px;background:var(--line);border-block:1px solid var(--line)}}.facts div{{background:#fff;padding:12px 18px}}.facts div:last-child{{grid-column:1/-1}}dt{{font-size:11px;text-transform:uppercase;color:var(--muted);font-weight:800}}dd{{margin:3px 0 0}}code{{overflow-wrap:anywhere;font-size:12px}}.note{{padding:0 18px 18px;color:var(--muted);font-size:14px}}a{{color:#4a2aa5}}.pending,.missing{{padding:24px;border-style:dashed}}.pending{{border-color:#8c6ddb;background:#faf7ff}}.missing h3{{margin-top:4px}}.missing-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}.archive-box{{margin-top:22px;padding:22px;background:#fff;border:1px solid var(--line);border-radius:16px}}.archive{{columns:2;column-gap:32px;padding-left:22px}}.archive li{{break-inside:avoid;margin:8px 0}}.archive span{{color:var(--muted);font-size:12px}}footer{{margin-top:52px;color:var(--muted);font-size:13px;border-top:1px solid var(--line);padding-top:20px}}@media(max-width:840px){{.grid,.missing-grid{{grid-template-columns:1fr}}.section-head{{display:block}}.directions,.facts{{grid-template-columns:1fr}}.facts div:last-child{{grid-column:auto}}article header{{display:block}}.status{{display:block;max-width:none;text-align:left;margin-top:7px}}.archive{{columns:1}}}}
</style></head><body><main>
<p class="overline">Video-first evidence index · direct static LEFT/RIGHT commands</p><h1>{html.escape(manifest['title'])}</h1>
<p class="lede">Videos are embedded at full card width, separated into WORLD MODEL and VLA sections, and labeled with arena, prompt, direction, outcome, model interface, and evidence status. Missing media stays missing. {html.escape(manifest['display_policy'])}</p>
<p class="boundary"><strong>Claim boundary.</strong> {html.escape(manifest['claim_boundary'])}</p>
{sections[0]}
{imagination_section}
{''.join(sections[1:])}
<section><div class="section-head"><h2>Explicit media gaps</h2><p>No raw or diagnostic artifact is substituted for publication video.</p></div><div class="missing-grid">{missing}</div></section>
<footer>Generated by <code>tools/render_vla_wam_video_first_gallery.py</code> from the hash-bearing gallery manifest (SHA-256 <code>{manifest_digest}</code>). Re-run the generator after any conforming media-manifest update.</footer>
</main></body></html>
"""


def render_markdown(
    manifest: dict[str, Any],
    entries: list[dict[str, Any]],
    dreamzero_present: bool,
    imagination_entries: list[dict[str, Any]],
    official_decodes: list[dict[str, Any]],
    imagination_present: bool,
) -> str:
    lines = [
        f"# {manifest['title']}",
        "",
        "This is the portable index for the embedded [HTML video gallery](VLA_WAM_STEERABILITY_VIDEO_GALLERY.html). "
        "DROID and RoboTwin are listed separately and their success rates are never pooled.",
        "",
        "## WORLD MODELS — actual simulator execution",
        "",
    ]
    if dreamzero_present:
        lines.append(
            "DreamZero's three hash-validated actual simulator pairs are listed in this section and link to "
            "same-seed imagined-future controls later in this document. Every other WAM states whether its released "
            "interface exposes decoded, latent-only, or action-only future evidence."
        )
    else:
        path = manifest["dreamzero_manifest_contract"]["path"]
        lines.extend([
            "**Pending — no behavioral video exists in the committed evidence.** This is not a zero. The generator is wired to "
            f"`{path}` and will ingest its `gallery_entries` only after every referenced clip validates.",
        ])
    lines.append("")
    imagined_by_seed = {entry["seed"]: entry for entry in imagination_entries}
    dreamzero_actual_by_seed = {
        entry["seed"]: entry
        for entry in entries
        if entry["id"].startswith("dreamzero_droid_seed")
    }

    def append_behavior_section(
        title: str,
        section_entries: list[dict[str, Any]],
        heading_already_emitted: bool = False,
    ) -> None:
        if not heading_already_emitted:
            lines.extend(["", f"## {title}", ""])
        for entry in section_entries:
            outcomes = "; ".join(
                f"{direction['relation']}: {direction['outcome']}" for direction in entry["directions"]
            )
            lines.extend([
                f"### {entry['model_label']} — {entry['pair_label']}",
                "",
                f"[▶ Open video]({rel(entry['video']['path'])}) · [Evidence manifest]({rel(entry['source_manifest'])})",
                "",
                f"- Outcome: {outcomes}",
                f"- Future interface: {entry['future_interface']}",
                f"- Evidence status: {entry['evidence_status']}",
                f"- Video SHA-256: `{entry['video']['sha256']}`",
                "",
            ])
            if entry["id"].startswith("dreamzero_droid_seed") and entry["seed"] in imagined_by_seed:
                imagined = imagined_by_seed[entry["seed"]]
                lines.append(f"[Open same-seed imagined-future control]({rel(imagined['video']['path'])})")
                lines.append("")
            for direction in entry["directions"]:
                lines.append(f"> {direction['relation']}: “{direction['prompt']}”")
            lines.append("")

    append_behavior_section(
        "WORLD MODELS — actual simulator execution",
        [entry for entry in entries if entry["category"] == "WAM"],
        heading_already_emitted=True,
    )
    lines.extend(["## DreamZero imagined futures — not execution", ""])
    if imagination_present:
        lines.append(
            "These are official model-predicted video decodes, not simulator executions, task outcomes, or additional episodes."
        )
        lines.append("")
        for entry in imagination_entries:
            lines.extend([
                f"### {entry['model_label']} — {entry['pair_label']}",
                "",
                f"[▶ Open paired imagination video]({rel(entry['video']['path'])}) · "
                f"[Imagination manifest]({rel(entry['source_manifest'])})",
                "",
                f"- Evidence status: {entry['evidence_status']}",
                f"- Video SHA-256: `{entry['video']['sha256']}`",
                "",
            ])
            if entry["seed"] in dreamzero_actual_by_seed:
                actual = dreamzero_actual_by_seed[entry["seed"]]
                lines.extend([
                    f"[Open same-seed actual rollout control]({rel(actual['video']['path'])})",
                    "",
                ])
        lines.extend(["### All nine original official decodes", ""])
        for record in official_decodes:
            lines.append(
                f"- [{record['id']}]({rel(record['archived_video']['path'])}) — "
                f"`{record['archived_video']['sha256']}`"
            )
        lines.append("")
    else:
        lines.extend([
            "**Pending — the bounded nine-file official decode archive is not committed yet.**",
            "",
        ])
    append_behavior_section(
        "VLAs — actual simulator execution",
        [entry for entry in entries if entry["category"] == "VLA"],
    )
    lines.extend(["## Missing publication media", ""])
    for item in manifest["missing_publication_media"]:
        if dreamzero_present and item["model_id"] == "dreamzero_droid":
            continue
        lines.append(f"- **{item['model_id']} — {item['status']}:** {item['reason']}")
    lines.extend([
        "",
        "Regenerate and validate with:",
        "",
        "```bash",
        "python3 tools/render_vla_wam_video_first_gallery.py",
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    html_path = args.html if args.html.is_absolute() else REPO_ROOT / args.html
    markdown_path = args.markdown if args.markdown.is_absolute() else REPO_ROOT / args.markdown

    manifest = json.loads(args.manifest.read_text())
    entries, dreamzero_present, imagination_entries, official_decodes, imagination_present = load_entries(manifest)
    html_path.write_text(
        render_html(
            manifest,
            entries,
            dreamzero_present,
            imagination_entries,
            official_decodes,
            imagination_present,
        )
    )
    markdown_path.write_text(
        render_markdown(
            manifest,
            entries,
            dreamzero_present,
            imagination_entries,
            official_decodes,
            imagination_present,
        )
    )
    print(
        json.dumps(
            {
                "status": "valid",
                "entry_count": len(entries),
                "dreamzero_media_present": dreamzero_present,
                "dreamzero_imagination_media_present": imagination_present,
                "dreamzero_imagination_entry_count": len(imagination_entries),
                "dreamzero_official_decode_count": len(official_decodes),
                "html": str(html_path.relative_to(REPO_ROOT)),
                "markdown": str(markdown_path.relative_to(REPO_ROOT)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
