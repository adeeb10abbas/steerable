#!/usr/bin/env python3
"""Build the publication PDF for the VLA/WAM language-steerability study.

The report is deliberately generated from committed evidence summaries.  It
keeps DROID/RoboLab and RoboTwin results separate and marks compatibility,
historical, prediction-only, and latent-only evidence explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output/pdf/language_sensitivity_directional_control_report.pdf"

BG = colors.HexColor("#F6F1E7")
CARD = colors.HexColor("#FCFAF5")
INK = colors.HexColor("#19242C")
MUTED = colors.HexColor("#667278")
LINE = colors.HexColor("#D6CEC1")
LEFT = colors.HexColor("#2563EB")
RIGHT = colors.HexColor("#D65332")
TEAL = colors.HexColor("#168B7B")
PURPLE = colors.HexColor("#7157A9")
GREEN = colors.HexColor("#2D7A5F")
SOFT_BLUE = colors.HexColor("#E8EFFB")
SOFT_RED = colors.HexColor("#F8E9E2")
SOFT_TEAL = colors.HexColor("#E4F1ED")
SOFT_PURPLE = colors.HexColor("#EEE9F6")

PAGE_W, PAGE_H = A4
MARGIN_X = 42
CONTENT_W = PAGE_W - 2 * MARGIN_X


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_number(mapping: dict[str, Any], *paths: str) -> float:
    for path in paths:
        value: Any = mapping
        try:
            for part in path.split("."):
                value = value[part]
        except (KeyError, TypeError):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    raise KeyError(paths)


def normalize_taxonomy(raw: dict[str, int]) -> dict[str, int]:
    return {
        "correct": int(raw.get("correct", 0)),
        "pick_failed": int(raw.get("pick_failed", 0)),
        "transport_failed": int(raw.get("transport_failed", 0)),
        "wrong_side": int(raw.get("wrong_side", 0)),
        "release_failed": int(raw.get("release_failed", 0)),
    }


def load_droid_summary(filename: str, label: str, kind: str, future: str) -> dict[str, Any]:
    path = ROOT / "artifacts/vla_wam_shared_v3/results" / filename
    data = read_json(path)
    left = data["directional"]["left"]
    right = data["directional"]["right"]
    pairs = len(data["pairs"])
    shift = data.get("right_minus_left_endpoint_shift_m") or data.get("right_minus_left_raw_robot_y_shift_m")
    action_summary = data["action_rms_common_prefix"]
    actions_all_differ = bool(
        action_summary.get("all_pairs_actions_differ")
        or (
            int(action_summary.get("observed_count", 0)) == pairs
            and float(action_summary.get("minimum", 0.0)) > 0.0
        )
    )
    return {
        "label": label,
        "kind": kind,
        "source": path,
        "episodes": int(left.get("valid_denominator", left.get("trials"))) + int(right.get("valid_denominator", right.get("trials"))),
        "pairs": pairs,
        "left": int(left["successes"]),
        "left_n": int(left.get("valid_denominator", left.get("trials"))),
        "right": int(right["successes"]),
        "right_n": int(right.get("valid_denominator", right.get("trials"))),
        "aligned": int(data["endpoint_ordering_counts"]["aligned"]),
        "distinct": pairs if actions_all_differ else 0,
        "shift_cm": -100.0 * float(shift["mean"]),
        "shift_median_cm": -100.0 * float(shift["median"]),
        "taxonomy": normalize_taxonomy(data["overall_failure_taxonomy_counts"]),
        "discordance": data["matched_discordance"],
        "future": future,
    }


def load_robotwin_summary(filename: str, label: str, kind: str, future: str) -> dict[str, Any]:
    path = ROOT / "artifacts/vla_wam_shared_v3/results" / filename
    data = read_json(path)["v3_primary_results"]
    left = data["by_direction"]["left"]
    right = data["by_direction"]["right"]
    endpoint = data["paired_endpoint_response"]
    action = data["paired_action_response"]
    shift = endpoint["left_minus_right_shift_summary_m"]
    return {
        "label": label,
        "kind": kind,
        "source": path,
        "episodes": int(data["valid_episodes"]),
        "pairs": int(data["matched_pairs"]),
        "left": int(left["successes"]),
        "left_n": int(left["valid_episodes"]),
        "right": int(right["successes"]),
        "right_n": int(right["valid_episodes"]),
        "aligned": int(endpoint["aligned"]),
        "distinct": int(action["distinct_trace_pairs"]),
        "shift_cm": 100.0 * float(shift["mean"]),
        "shift_median_cm": 100.0 * float(shift["median"]),
        "taxonomy": normalize_taxonomy(data["failure_taxonomy_counts"]),
        "discordance": data["paired_success_discordance"],
        "future": future,
    }


def load_pi0_compatibility() -> dict[str, Any]:
    path = ROOT / "artifacts/vla_wam_shared_v3/results/pi0_fast_old_name_config_v3a002_summary.json"
    data = read_json(path)
    shift = data["right_minus_left_endpoint_shift_m"]
    return {
        "label": "pi0-FAST public compatibility",
        "kind": "VLA",
        "source": path,
        "episodes": int(data["behavioral_episodes"]),
        "pairs": int(data["behavioral_matched_pairs"]),
        "left": int(data["success_by_direction"]["left"]["successes"]),
        "left_n": int(data["success_by_direction"]["left"]["episodes"]),
        "right": int(data["success_by_direction"]["right"]["successes"]),
        "right_n": int(data["success_by_direction"]["right"]["episodes"]),
        "aligned": int(data["endpoint_ordering"]["aligned"]),
        "distinct": int(data["distinct_executed_action_pairs"]["count"]),
        "shift_cm": -100.0 * float(shift["mean"]),
        "shift_median_cm": -100.0 * float(shift["median"]),
        "taxonomy": normalize_taxonomy(data["failure_taxonomy"]),
        "future": "action-only",
    }


def load_study_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    droid = [
        load_droid_summary("pi05_current_stack_droid_phase_a_summary.json", "pi0.5 current stack", "VLA", "action-only"),
        load_droid_summary("groot_n17_droid_phase_a_summary.json", "GR00T N1.7", "VLA", "action-only"),
        load_droid_summary("cosmos3_edge_policy_droid_phase_a_summary.json", "Cosmos3 Edge Policy", "WAM", "452 decoded local futures"),
        load_droid_summary("cosmos3_nano_policy_droid_phase_a_summary.json", "Cosmos3 Nano Policy", "WAM", "349 decoded local futures"),
        load_droid_summary("dreamzero_droid_action_cfg_phase_a_summary.json", "DreamZero action guidance s=2", "WAM", "54 full-reset decodes + 2,554 latent futures"),
    ]
    robotwin = [
        load_robotwin_summary("efficient_wam_rt_robotwin_phase_a_summary.json", "Efficient-WAM-RT", "WAM", "126 decoded coarse futures"),
        load_robotwin_summary("fastwam_robotwin_phase_a_summary.json", "FastWAM", "WAM", "action-only at test time"),
        load_robotwin_summary("lingbot_va_robotwin_phase_a_summary.json", "LingBot-VA", "WAM", "126 latent-only futures"),
    ]
    compatibility = load_pi0_compatibility()
    nano_path = ROOT / "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/nano_v3b001_summary.json"
    nano = read_json(nano_path)
    nano["_source"] = str(nano_path)
    return droid, robotwin, compatibility, nano


def register_fonts() -> None:
    candidates = {
        "Georgia": "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "Georgia-Bold": "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
        "Georgia-Italic": "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
        "Arial": "/System/Library/Fonts/Supplemental/Arial.ttf",
        "Arial-Bold": "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "Arial-Italic": "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
    }
    for name, path in candidates.items():
        if Path(path).exists():
            pdfmetrics.registerFont(TTFont(name, path))
    if "Georgia" not in pdfmetrics.getRegisteredFontNames():
        for name, fallback in (("Georgia", "Times-Roman"), ("Georgia-Bold", "Times-Bold"), ("Georgia-Italic", "Times-Italic"), ("Arial", "Helvetica"), ("Arial-Bold", "Helvetica-Bold"), ("Arial-Italic", "Helvetica-Oblique")):
            pdfmetrics.registerFont(pdfmetrics.getFont(fallback))


BODY = ParagraphStyle("Body", fontName="Arial", fontSize=10.2, leading=14.7, textColor=INK, spaceAfter=6)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.2, leading=11.4, textColor=MUTED)
TINY = ParagraphStyle("Tiny", parent=BODY, fontSize=6.8, leading=8.7, textColor=MUTED)
TABLE_TINY = ParagraphStyle("TableTiny", parent=BODY, fontSize=5.8, leading=7.2, textColor=INK)
TABLE_TINY_BOLD = ParagraphStyle("TableTinyBold", parent=TABLE_TINY, fontName="Arial-Bold")
TABLE_HEADER = ParagraphStyle("TableHeader", parent=TABLE_TINY_BOLD, textColor=colors.white)
LABEL = ParagraphStyle("Label", fontName="Arial-Bold", fontSize=7.5, leading=9, textColor=MUTED, uppercase=True)
H1 = ParagraphStyle("H1", fontName="Georgia-Bold", fontSize=26, leading=30, textColor=INK)
H2 = ParagraphStyle("H2", fontName="Georgia-Bold", fontSize=17, leading=21, textColor=INK)
H3 = ParagraphStyle("H3", fontName="Georgia-Bold", fontSize=12, leading=15, textColor=INK)
QUOTE = ParagraphStyle("Quote", fontName="Georgia", fontSize=13.2, leading=18, textColor=INK)
CENTER = ParagraphStyle("Center", parent=BODY, alignment=TA_CENTER)


def para(c: canvas.Canvas, text: str, x: float, top: float, width: float, style: ParagraphStyle = BODY, max_height: float = 1000) -> float:
    p = Paragraph(text, style)
    w, h = p.wrap(width, max_height)
    p.drawOn(c, x, top - h)
    return h


def fit_image(c: canvas.Canvas, path: Path, x: float, y: float, w: float, h: float, pad: float = 0, link: str | None = None) -> None:
    with Image.open(path) as image:
        iw, ih = image.size
    scale = min((w - 2 * pad) / iw, (h - 2 * pad) / ih)
    dw, dh = iw * scale, ih * scale
    dx, dy = x + (w - dw) / 2, y + (h - dh) / 2
    c.drawImage(ImageReader(str(path)), dx, dy, width=dw, height=dh, mask="auto")
    if link:
        c.linkURL(link, (dx, dy, dx + dw, dy + dh), relative=0)


def crop_image(c: canvas.Canvas, path: Path, x: float, y: float, w: float, h: float, link: str | None = None) -> None:
    with Image.open(path) as image:
        iw, ih = image.size
    scale = max(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    c.saveState()
    clip = c.beginPath()
    clip.rect(x, y, w, h)
    c.clipPath(clip, stroke=0, fill=0)
    c.drawImage(ImageReader(str(path)), x + (w - dw) / 2, y + (h - dh) / 2, width=dw, height=dh, mask="auto")
    c.restoreState()
    if link:
        c.linkURL(link, (x, y, x + w, y + h), relative=0)


def rounded_card(c: canvas.Canvas, x: float, y: float, w: float, h: float, fill: colors.Color = CARD, stroke: colors.Color = LINE, radius: float = 10) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(0.7)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=1)


class Report:
    def __init__(self, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        self.output = output
        self.c = canvas.Canvas(str(output), pagesize=A4, invariant=1, pageCompression=1)
        self.c.setTitle("Language Sensitivity Is Not Directional Control")
        self.c.setSubject("Matched static-instruction interventions in robot policies and world-action models")
        self.c.setAuthor("Steerable Robotics Study")
        self.page_number = 0

    def new_page(self, title: str | None = None, kicker: str | None = None, subtitle: str | None = None) -> float:
        if self.page_number:
            self.c.showPage()
        self.page_number += 1
        self.c.setPageSize(A4)
        self.c.setFillColor(BG)
        self.c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        if title:
            if kicker:
                self.c.setFont("Arial-Bold", 7.4)
                self.c.setFillColor(MUTED)
                self.c.drawString(MARGIN_X, PAGE_H - 41, kicker.upper())
            top = PAGE_H - 54
            used = para(self.c, title, MARGIN_X, top, CONTENT_W, H1)
            y = top - used - 7
            if subtitle:
                y -= para(self.c, subtitle, MARGIN_X, y, CONTENT_W, SMALL)
            self.c.setStrokeColor(LINE)
            self.c.line(MARGIN_X, y - 5, PAGE_W - MARGIN_X, y - 5)
            return y - 20
        return PAGE_H - 42

    def footer(self, label: str) -> None:
        self.c.setStrokeColor(LINE)
        self.c.line(MARGIN_X, 29, PAGE_W - MARGIN_X, 29)
        self.c.setFillColor(MUTED)
        self.c.setFont("Arial", 6.9)
        self.c.drawString(MARGIN_X, 18, label)
        self.c.drawRightString(PAGE_W - MARGIN_X, 18, str(self.page_number))

    def finish(self) -> None:
        self.c.save()


def metric_bar(c: canvas.Canvas, x: float, y: float, w: float, value: float, color: colors.Color, label: str, detail: str) -> None:
    c.setFillColor(colors.HexColor("#E5E0D7"))
    c.roundRect(x, y, w, 8, 4, fill=1, stroke=0)
    c.setFillColor(color)
    c.roundRect(x, y, max(1.5, w * max(0, min(1, value))), 8, 4, fill=1, stroke=0)
    c.setFillColor(INK)
    c.setFont("Arial-Bold", 7.6)
    c.drawString(x, y + 13, label)
    c.setFillColor(MUTED)
    c.setFont("Arial", 7.2)
    c.drawRightString(x + w, y + 13, detail)


def draw_overview_row(c: canvas.Canvas, item: dict[str, Any], x: float, y: float, w: float) -> None:
    rounded_card(c, x, y, w, 57)
    c.setFillColor(INK)
    c.setFont("Arial-Bold", 8.6)
    c.drawString(x + 10, y + 38, item["label"])
    c.setFillColor(MUTED)
    c.setFont("Arial", 7.0)
    c.drawString(x + 10, y + 25, f'{item["kind"]}  |  {item["episodes"]} episodes')
    bx = x + 158
    bw = (w - 178) / 3
    metric_bar(c, bx, y + 17, bw, item["distinct"] / item["pairs"], PURPLE, "Action response", f'{item["distinct"]}/{item["pairs"]}')
    metric_bar(c, bx + bw + 10, y + 17, bw, item["aligned"] / item["pairs"], TEAL, "Endpoint aligned", f'{item["aligned"]}/{item["pairs"]}')
    sx = bx + 2 * (bw + 10)
    c.setFont("Arial-Bold", 7.4)
    c.setFillColor(LEFT)
    c.drawString(sx, y + 31, f'LEFT {item["left"]}/{item["left_n"]}')
    c.setFillColor(RIGHT)
    c.drawString(sx, y + 18, f'RIGHT {item["right"]}/{item["right_n"]}')


def draw_droid_result_rows(c: canvas.Canvas, items: list[dict[str, Any]], x: float, y_top: float, w: float) -> None:
    label_w = 150
    plot_x = x + label_w
    plot_w = w - label_w - 86
    c.setFont("Arial-Bold", 7.1)
    c.setFillColor(MUTED)
    c.drawString(plot_x, y_top + 9, "TASK SUCCESS RATE")
    c.drawRightString(x + w, y_top + 9, "ALIGNMENT  SHIFT")
    for pct in (0, 0.25, 0.5, 0.75, 1.0):
        px = plot_x + pct * plot_w
        c.setStrokeColor(LINE)
        c.line(px, y_top - len(items) * 48 + 8, px, y_top + 2)
        c.setFont("Arial", 6.5)
        c.setFillColor(MUTED)
        c.drawCentredString(px, y_top - len(items) * 48 - 2, f"{int(pct*100)}%")
    for index, item in enumerate(items):
        y = y_top - index * 48 - 25
        c.setFillColor(INK)
        c.setFont("Arial-Bold", 8.6)
        c.drawString(x, y + 7, item["label"])
        c.setFont("Arial", 6.9)
        c.setFillColor(MUTED)
        c.drawString(x, y - 6, f'{item["kind"]} | n={item["left_n"]} matched pairs')
        left_rate = item["left"] / item["left_n"]
        right_rate = item["right"] / item["right_n"]
        lx, rx = plot_x + left_rate * plot_w, plot_x + right_rate * plot_w
        c.setStrokeColor(colors.HexColor("#AAB2B6"))
        c.setLineWidth(2)
        c.line(lx, y + 2, rx, y + 2)
        c.setFillColor(LEFT)
        c.circle(lx, y + 2, 4.2, fill=1, stroke=0)
        c.setFillColor(RIGHT)
        c.circle(rx, y + 2, 4.2, fill=1, stroke=0)
        c.setFont("Arial-Bold", 6.8)
        c.setFillColor(LEFT)
        c.drawCentredString(lx, y + 11, f'{item["left"]}/{item["left_n"]}')
        c.setFillColor(RIGHT)
        c.drawCentredString(rx, y - 10, f'{item["right"]}/{item["right_n"]}')
        c.setFillColor(INK)
        c.setFont("Arial-Bold", 7.2)
        c.drawRightString(x + w, y + 4, f'{item["aligned"]}/{item["pairs"]}  {item["shift_cm"]:+.1f} cm')


def stacked_taxonomy(c: canvas.Canvas, item: dict[str, Any], x: float, y: float, w: float) -> None:
    palette = [GREEN, colors.HexColor("#A68B67"), colors.HexColor("#D5A145"), colors.HexColor("#9C6DB0"), colors.HexColor("#6C7780")]
    keys = ["correct", "pick_failed", "transport_failed", "wrong_side", "release_failed"]
    total = sum(item["taxonomy"].values())
    cursor = x
    for key, color in zip(keys, palette):
        value = item["taxonomy"][key]
        if value:
            width = w * value / total
            c.setFillColor(color)
            c.rect(cursor, y, width, 10, fill=1, stroke=0)
            cursor += width
    c.setFillColor(INK)
    c.setFont("Arial-Bold", 7.6)
    c.drawString(x, y + 16, item["label"])
    c.setFillColor(MUTED)
    c.setFont("Arial", 6.7)
    counts = "/".join(str(item["taxonomy"][k]) for k in keys)
    c.drawRightString(x + w, y + 16, f"C/P/T/W/R = {counts}")


def seed_strip(c: canvas.Canvas, values: list[float], x: float, y: float, w: float, lo: float, hi: float, color: colors.Color, label: str) -> None:
    c.setStrokeColor(LINE)
    c.line(x, y, x + w, y)
    if lo < 0 < hi:
        zero_x = x + (0 - lo) / (hi - lo) * w
        c.setStrokeColor(INK)
        c.setLineWidth(0.8)
        c.line(zero_x, y - 11, zero_x, y + 11)
    for idx, value in enumerate(values):
        px = x + (value - lo) / (hi - lo) * w
        jitter = ((idx % 5) - 2) * 1.8
        c.setFillColor(color)
        c.setFillAlpha(0.55)
        c.circle(px, y + jitter, 2.4, fill=1, stroke=0)
    c.setFillAlpha(1)
    c.setFillColor(INK)
    c.setFont("Arial-Bold", 8)
    c.drawString(x, y + 17, label)


def draw_cover(report: Report) -> None:
    c = report.c
    report.new_page()
    c.setFillColor(INK)
    c.rect(0, PAGE_H - 18, PAGE_W, 18, fill=1, stroke=0)
    c.setFont("Arial-Bold", 8)
    c.setFillColor(MUTED)
    c.drawString(MARGIN_X, PAGE_H - 68, "STEERABLE ROBOTICS STUDY  |  RESEARCH REPORT  |  AUGUST 2026")
    para(c, "Language Sensitivity Is Not Directional Control", MARGIN_X, PAGE_H - 116, CONTENT_W - 28, ParagraphStyle("Cover", parent=H1, fontSize=34, leading=39))
    para(c, "Matched static-instruction interventions in robot policies and world-action models", MARGIN_X, PAGE_H - 236, CONTENT_W - 50, ParagraphStyle("Sub", parent=QUOTE, fontSize=15, leading=21, textColor=MUTED))
    rounded_card(c, MARGIN_X, 285, CONTENT_W, 165, fill=CARD)
    c.setFillColor(TEAL)
    c.rect(MARGIN_X, 285, 7, 165, fill=1, stroke=0)
    para(c, "Across eight expanded checkpoints, changing the static instruction changed every matched action trace. Endpoint redirection was frequent, but requested task completion remained checkpoint- and direction-dependent.", MARGIN_X + 27, 421, CONTENT_W - 56, ParagraphStyle("CoverClaim", parent=QUOTE, fontSize=16, leading=23))
    c.setFont("Arial-Bold", 8)
    c.setFillColor(MUTED)
    c.drawString(MARGIN_X + 27, 307, "PRIMARY EVIDENCE: 648 PHASE-A EPISODES  |  SEPARATE 40-EPISODE COMPATIBILITY COHORT  |  108-EPISODE ABLATION")
    c.setStrokeColor(LINE)
    c.line(MARGIN_X, 109, PAGE_W - MARGIN_X, 109)
    para(c, "DROID/RoboLab and RoboTwin use different tasks, controllers, and success predicates. Their success rates are reported separately and never pooled.", MARGIN_X, 92, CONTENT_W, SMALL)
    c.setFillColor(MUTED)
    c.setFont("Arial", 7)
    c.drawString(MARGIN_X, 34, "Branch: main")
    c.drawRightString(PAGE_W - MARGIN_X, 34, "Evidence head: fabefebb1b13")


def draw_question(report: Report, droid: list[dict[str, Any]], robotwin: list[dict[str, Any]]) -> None:
    c = report.c
    y = report.new_page("Question, hypothesis, and answer", "1  STUDY LOGIC", "The intervention changes one semantic relation while preserving the matched reset and runtime identity.")
    rounded_card(c, MARGIN_X, y - 132, CONTENT_W, 118, fill=SOFT_BLUE, stroke=colors.HexColor("#C8D6EF"))
    para(c, "<b>Question.</b> From an identical scene and seed, does changing only the static instruction from LEFT to RIGHT (1) change the executed actions, (2) order the final endpoints accordingly, and (3) preserve task competence in both directions?", MARGIN_X + 18, y - 31, CONTENT_W - 36, QUOTE)
    y -= 157
    para(c, "<b>Hypothesis.</b> A controllable policy should be sensitive to the instruction, redirect the object in the requested ordering, and complete the frozen relation-and-release task under both commands.", MARGIN_X, y, CONTENT_W, BODY)
    y -= 66
    c.setFont("Georgia-Bold", 16)
    c.setFillColor(INK)
    c.drawString(MARGIN_X, y, "Answer")
    y -= 24
    col_w = (CONTENT_W - 18) / 2
    rounded_card(c, MARGIN_X, y - 165, col_w, 165)
    para(c, "<b>What generalized</b><br/><br/>All 135 DROID Phase-A pairs and all 189 RoboTwin replicate pairs produced distinct executed action traces. Endpoint ordering aligned in 125/135 DROID pairs and 128/189 RoboTwin pairs.", MARGIN_X + 15, y - 17, col_w - 30, BODY)
    rounded_card(c, MARGIN_X + col_w + 18, y - 165, col_w, 165)
    para(c, "<b>What did not</b><br/><br/>Task completion depended strongly on checkpoint and requested direction. GR00T changed all action traces but completed only 3/54 episodes; Nano completed 51/54 while still exhibiting geometry-sensitive side depth.", MARGIN_X + col_w + 33, y - 17, col_w - 30, BODY)
    y -= 192
    rounded_card(c, MARGIN_X, y - 96, CONTENT_W, 96, fill=SOFT_TEAL, stroke=colors.HexColor("#C4DED5"))
    para(c, "<b>Scientific reading.</b> Language sensitivity is necessary evidence of conditioning, not sufficient evidence of directional control. A model can react to the word, move in the requested ordering, and still fail to pick, transport, release, or robustly satisfy both directions.", MARGIN_X + 17, y - 17, CONTENT_W - 34, BODY)
    report.footer("Question, hypothesis, and answer")


def draw_intervention(report: Report) -> None:
    c = report.c
    y = report.new_page("The exact intervention", "2  EXPERIMENTAL DESIGN", "LEFT and RIGHT are shorthand labels for full static prompts, not isolated tokens.")
    c.setFont("Arial-Bold", 7.5)
    c.setFillColor(LEFT)
    c.drawString(MARGIN_X, y, "DROID / ROBOLAB - LEFT CONDITION")
    rounded_card(c, MARGIN_X, y - 80, CONTENT_W, 64, fill=SOFT_BLUE, stroke=colors.HexColor("#C8D6EF"))
    para(c, "Put the Rubik's cube to the left of the bowl.", MARGIN_X + 18, y - 34, CONTENT_W - 36, QUOTE)
    y -= 100
    c.setFillColor(RIGHT)
    c.setFont("Arial-Bold", 7.5)
    c.drawString(MARGIN_X, y, "DROID / ROBOLAB - RIGHT CONDITION")
    rounded_card(c, MARGIN_X, y - 80, CONTENT_W, 64, fill=SOFT_RED, stroke=colors.HexColor("#E7C7BA"))
    para(c, "Put the Rubik's cube to the right of the bowl.", MARGIN_X + 18, y - 34, CONTENT_W - 36, QUOTE)
    y -= 112
    para(c, "RoboTwin uses the same sentence structure with scene-specific objects. For example, pair03 changes only <b>left</b> to <b>right</b> in: \"Put the small woodenblock to the [left/right] of the red playingcards box.\" The seven expanded scenes use pair03-pair09; all full strings appear in Appendix B.", MARGIN_X, y, CONTENT_W, BODY)
    y -= 88
    c.setFont("Georgia-Bold", 16)
    c.setFillColor(INK)
    c.drawString(MARGIN_X, y, "Matched invariants")
    y -= 22
    invariants = [
        ("Scene and reset", "same initial state hash within each pair"),
        ("Seeds", "same environment and sampling seed"),
        ("Runtime", "same checkpoint, adapter, controller, and horizon"),
        ("Instruction", "static for the full episode; relation word changes"),
        ("Scoring", "frozen arena-specific predicate and failure taxonomy"),
        ("Evidence", "video and raw per-episode record for every valid rollout"),
    ]
    box_w = (CONTENT_W - 12) / 2
    for idx, (name, detail) in enumerate(invariants):
        col, row = idx % 2, idx // 2
        bx = MARGIN_X + col * (box_w + 12)
        by = y - row * 66 - 56
        rounded_card(c, bx, by, box_w, 52)
        c.setFillColor(INK)
        c.setFont("Arial-Bold", 8.3)
        c.drawString(bx + 12, by + 33, name)
        c.setFillColor(MUTED)
        c.setFont("Arial", 7.4)
        c.drawString(bx + 12, by + 17, detail)
    para(c, "No oracle, subtask coach, prompt switching, or progress-conditioned language. Every valid behavioral failure stays in the denominator; infrastructure-invalid and partial attempts remain separate.", MARGIN_X, 112, CONTENT_W, SMALL)
    report.footer("Exact prompts and matched design")


def draw_estimands(report: Report) -> None:
    c = report.c
    y = report.new_page("Three estimands, kept separate", "3  MEASUREMENT", "The analysis asks whether language changes behavior, whether the change has the requested sign, and whether the task is completed.")
    card_w = (CONTENT_W - 24) / 3
    cards = [
        (PURPLE, "1", "Action sensitivity", "LEFT and RIGHT executed action arrays differ over their common prefix. This detects conditioning, not correctness."),
        (TEAL, "2", "Endpoint redirection", "The matched RIGHT endpoint is ordered to the requested side of the LEFT endpoint. This measures semantic direction."),
        (GREEN, "3", "Task competence", "The rollout satisfies the frozen requested relation and detached-release predicate. This is the success denominator."),
    ]
    for idx, (color, number, title, body) in enumerate(cards):
        x = MARGIN_X + idx * (card_w + 12)
        rounded_card(c, x, y - 205, card_w, 190)
        c.setFillColor(color)
        c.circle(x + 28, y - 44, 16, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Arial-Bold", 11)
        c.drawCentredString(x + 28, y - 48, number)
        para(c, title, x + 14, y - 82, card_w - 28, H3)
        para(c, body, x + 14, y - 113, card_w - 28, SMALL)
    y -= 230
    rounded_card(c, MARGIN_X, y - 145, CONTENT_W, 132, fill=SOFT_PURPLE, stroke=colors.HexColor("#D4C8E4"))
    para(c, "<b>Continuous fields retained for every episode</b>", MARGIN_X + 18, y - 20, CONTENT_W - 36, H3)
    para(c, "Signed final lateral offset; requested-side margin; cone-entry time; episode length; sustained versus transient entry; time to first contact where the adapter exposes it; and object path length. Margin is success-conditional when used as a competence measure, whereas signed offset remains defined on failures.", MARGIN_X + 18, y - 50, CONTENT_W - 36, BODY)
    y -= 174
    c.setFont("Georgia-Bold", 15)
    c.setFillColor(INK)
    c.drawString(MARGIN_X, y, "Evidence accounting")
    y -= 25
    accounting = [
        ("648", "completed Phase-A episodes", "270 DROID + 378 RoboTwin"),
        ("40", "separate pi0-FAST compatibility episodes", "20 matched pairs; not historical recovery"),
        ("108", "Nano position-reflection episodes", "27 seeds x 2 layouts x 2 prompts"),
        ("160", "historical wording episodes", "two checkpoints; separate V1 cohort"),
    ]
    for idx, (n, label, note) in enumerate(accounting):
        col, row = idx % 2, idx // 2
        x = MARGIN_X + col * (CONTENT_W / 2)
        yy = y - row * 59
        c.setFillColor(INK)
        c.setFont("Georgia-Bold", 18)
        c.drawString(x, yy, n)
        c.setFont("Arial-Bold", 8)
        c.drawString(x + 55, yy + 2, label)
        c.setFillColor(MUTED)
        c.setFont("Arial", 7)
        c.drawString(x + 55, yy - 12, note)
    report.footer("Estimands and evidence accounting")


def draw_overview(report: Report, droid: list[dict[str, Any]], robotwin: list[dict[str, Any]]) -> None:
    c = report.c
    y = report.new_page("Sensitivity was widespread; control was not", "4  PRINCIPAL RESULT", "Every expanded checkpoint changed its executed trace, but endpoint alignment and success remained checkpoint-specific.")
    c.setFont("Arial-Bold", 7.5)
    c.setFillColor(MUTED)
    c.drawString(MARGIN_X, y, "DROID / ROBOLAB - 27 MATCHED PAIRS PER CHECKPOINT")
    y -= 66
    for item in droid:
        draw_overview_row(c, item, MARGIN_X, y, CONTENT_W)
        y -= 64
    y -= 3
    c.setFont("Arial-Bold", 7.5)
    c.setFillColor(MUTED)
    c.drawString(MARGIN_X, y + 44, "ROBOTWIN - 63 REPLICATE PAIRS PER CHECKPOINT, NESTED WITHIN SEVEN SCENES")
    for item in robotwin:
        draw_overview_row(c, item, MARGIN_X, y - 18, CONTENT_W)
        y -= 64
    report.footer("Expanded Phase A: action response, endpoint alignment, task success")


def draw_droid(report: Report, droid: list[dict[str, Any]], compatibility: dict[str, Any]) -> None:
    c = report.c
    y = report.new_page("DROID: direction-dependent competence", "5  DROID / ROBOLAB", "Task success, endpoint ordering, and mean requested-oriented shift are shown on the same checkpoint row.")
    draw_droid_result_rows(c, droid, MARGIN_X, y - 8, CONTENT_W)
    y -= 285
    rounded_card(c, MARGIN_X, y - 90, CONTENT_W, 81, fill=SOFT_TEAL, stroke=colors.HexColor("#C4DED5"))
    para(c, "<b>Reading the pattern.</b> Nano remained near ceiling in both directions (26/27 LEFT; 25/27 RIGHT), while pi0.5 and DreamZero favored RIGHT and GR00T favored LEFT weakly but failed almost everywhere. Edge completed both directions more often, yet RIGHT still exceeded LEFT by 7/27 episodes. High alignment therefore does not erase checkpoint-specific directional asymmetry.", MARGIN_X + 17, y - 17, CONTENT_W - 34, SMALL)
    y -= 104
    c.setFont("Georgia-Bold", 14)
    c.setFillColor(INK)
    c.drawString(MARGIN_X, y, "Separate public compatibility cohort")
    y -= 18
    rounded_card(c, MARGIN_X, y - 78, CONTENT_W, 68)
    c.setFillColor(INK)
    c.setFont("Arial-Bold", 9)
    c.drawString(MARGIN_X + 15, y - 23, compatibility["label"])
    c.setFillColor(LEFT)
    c.drawString(MARGIN_X + 225, y - 23, f'LEFT {compatibility["left"]}/{compatibility["left_n"]}')
    c.setFillColor(RIGHT)
    c.drawString(MARGIN_X + 315, y - 23, f'RIGHT {compatibility["right"]}/{compatibility["right_n"]}')
    c.setFillColor(INK)
    c.drawString(MARGIN_X + 405, y - 23, f'ALIGNED {compatibility["aligned"]}/{compatibility["pairs"]}')
    para(c, "All 20 action prefixes differed; exact McNemar p=0.000488 and endpoint sign-test p=0.00443. This is a public old-name-config bridge, not recovery of the unavailable historical integration.", MARGIN_X + 15, y - 43, CONTENT_W - 30, TINY)
    y -= 94
    c.setFont("Georgia-Bold", 13)
    c.setFillColor(INK)
    c.drawString(MARGIN_X, y, "Failure decomposition")
    y -= 34
    for item in droid:
        stacked_taxonomy(c, item, MARGIN_X, y, CONTENT_W)
        y -= 27
    report.footer("DROID direct-command evidence; compatibility cohort shown separately")


def draw_robotwin(report: Report, robotwin: list[dict[str, Any]]) -> None:
    c = report.c
    y = report.new_page("RoboTwin: response across seven scenes", "6  ROBOTWIN", "Nine sampling replicates are nested within each of seven scenes; 63 replicate pairs are not 63 independent scenes.")
    figure = ROOT / "artifacts/vla_wam_shared_v3/publication_pdf/figures/robotwin-diagnostics.png"
    fit_image(c, figure, MARGIN_X, y - 420, CONTENT_W, 405)
    y -= 438
    col_w = (CONTENT_W - 14) / 2
    rounded_card(c, MARGIN_X, y - 134, col_w, 126, fill=SOFT_TEAL, stroke=colors.HexColor("#C4DED5"))
    para(c, "<b>What was consistent</b><br/><br/>All 189 replicate pairs changed their first ten executed actions. Endpoint ordering aligned in 42/63 Efficient-WAM-RT, 39/63 FastWAM, and 47/63 LingBot-VA pairs.", MARGIN_X + 14, y - 21, col_w - 28, SMALL)
    rounded_card(c, MARGIN_X + col_w + 14, y - 134, col_w, 126, fill=SOFT_RED, stroke=colors.HexColor("#E7C7BA"))
    para(c, "<b>What remained model- and scene-dependent</b><br/><br/>Requested success was 54/126, 44/126, and 38/126 respectively. These conditional replicate counts are descriptive; arena success is never pooled with DROID.", MARGIN_X + col_w + 28, y - 21, col_w - 28, SMALL)
    report.footer("RoboTwin expanded direct-command evidence")


def draw_wording_panel(c: canvas.Canvas, summaries: list[dict[str, Any]], model_id: str, label: str, x: float, y: float, w: float, h: float) -> None:
    rounded_card(c, x, y, w, h)
    para(c, label, x + 14, y + h - 17, w - 28, H3)
    c.setFillColor(MUTED)
    c.setFont("Arial", 6.5)
    c.drawString(x + 14, y + h - 35, "Task-success rate with Beta(1,1) posterior 95% credible interval")
    forms = [
        ("canonical", "Canonical"),
        ("short_paraphrase", "Shortened"),
        ("declarative_goal", "Goal statement"),
        ("contrastive_goal", "Contrastive"),
    ]
    records = {(r["wording"], r["direction"]): r for r in summaries if r["model_id"] == model_id}
    plot_x = x + 155
    plot_w = w - 230
    top = y + h - 61
    for pct in (0, 0.25, 0.5, 0.75, 1.0):
        px = plot_x + pct * plot_w
        c.setStrokeColor(LINE)
        c.line(px, y + 25, px, top)
        c.setFillColor(MUTED)
        c.setFont("Arial", 5.8)
        c.drawCentredString(px, y + 14, f"{int(pct*100)}%")
    for idx, (wording, form_label) in enumerate(forms):
        row_y = top - idx * 42
        c.setFillColor(INK)
        c.setFont("Arial-Bold", 7.1)
        c.drawString(x + 14, row_y - 4, form_label)
        for direction, color, offset in (("left", LEFT, 4), ("right", RIGHT, -10)):
            record = records[(wording, direction)]
            rate = float(record["success_rate"])
            low, high = record["success_beta11_interval_95"]
            py = row_y + offset
            c.setStrokeColor(color)
            c.setLineWidth(1.2)
            c.line(plot_x + low * plot_w, py, plot_x + high * plot_w, py)
            c.line(plot_x + low * plot_w, py - 3, plot_x + low * plot_w, py + 3)
            c.line(plot_x + high * plot_w, py - 3, plot_x + high * plot_w, py + 3)
            c.setFillColor(color)
            c.circle(plot_x + rate * plot_w, py, 3.1, fill=1, stroke=0)
            c.setFont("Arial-Bold", 5.9)
            c.drawRightString(x + w - 16, py - 2, f'{direction.upper()} {record["successes"]}/{record["episodes"]}')


def draw_wording_success(report: Report) -> None:
    c = report.c
    y = report.new_page("Phrasing interacted with requested direction", "7  HISTORICAL WORDING EXPERIMENT", "A separate 160-episode V1 DROID cohort tested four static prompt forms at ten matched seeds per cell.")
    summary_path = ROOT / "artifacts/vla_wam_shared_v1/final_evidence/closed_loop_summary.json"
    summaries = read_json(summary_path)["group_summaries"]
    draw_wording_panel(c, summaries, "pi05_droid_vla", "pi0.5 DROID", MARGIN_X, y - 223, CONTENT_W, 212)
    draw_wording_panel(c, summaries, "cosmos3_edge_droid_wam", "Cosmos3 Edge DROID", MARGIN_X, y - 451, CONTENT_W, 212)
    y -= 466
    rounded_card(c, MARGIN_X, y - 132, CONTENT_W, 123, fill=SOFT_PURPLE, stroke=colors.HexColor("#D4C8E4"))
    para(c, "<b>The result is an interaction, not a checkpoint-wide score.</b> pi0.5 ranged from 0/10 LEFT versus 8/10 RIGHT under the canonical wording to 2/10 versus 2/10 under the contrastive wording. Cosmos3 Edge ranged from 10/10 versus 9/10 under the goal statement to 1/10 versus 9/10 under the contrastive wording. The contrastive Cosmos direction gap had exact paired p=0.00781; the exploratory wording-pair tests were not multiplicity corrected.", MARGIN_X + 17, y - 20, CONTENT_W - 34, BODY)
    para(c, "These historical V1 prompts were lower-case and unpunctuated; Appendix B prints the byte-level strings. This cohort is not the unreleased 480-episode V3 Phase C grid.", MARGIN_X + 17, y - 98, CONTENT_W - 34, SMALL)
    report.footer("Historical V1 DROID wording layer; separate from Phase A and unreleased Phase C")


def draw_wording_shift(report: Report) -> None:
    c = report.c
    y = report.new_page("Physical redirection exceeded reliable completion", "8  HISTORICAL PAIRED ENDPOINTS", "Each dot is one matched RIGHT-minus-LEFT endpoint shift; the plot measures redirection, not competence.")
    figure = ROOT / "artifacts/vla_wam_shared_v3/publication_pdf/figures/droid-paired-shifts.png"
    fit_image(c, figure, MARGIN_X, y - 435, CONTENT_W, 425)
    y -= 450
    rounded_card(c, MARGIN_X, y - 129, CONTENT_W, 120, fill=SOFT_TEAL, stroke=colors.HexColor("#C4DED5"))
    para(c, "pi0.5 endpoints moved in the requested LEFT-to-RIGHT ordering in 32/40 wording-matched pairs (median shift +9.7 cm); Cosmos3 Edge did so in 38/40 (median +41.6 cm). Endpoint motion can therefore be prompt-conditioned even when pick, transport, release, or the full task predicate fails.", MARGIN_X + 17, y - 22, CONTENT_W - 34, BODY)
    para(c, "The right-hand mini-panels locate median LEFT- and RIGHT-condition endpoints relative to the bowl. Success remains the full relation-and-release predicate, not lateral position alone.", MARGIN_X + 17, y - 88, CONTENT_W - 34, SMALL)
    report.footer("Historical V1 physical-response layer")


def draw_nano_design(report: Report, nano: dict[str, Any]) -> None:
    c = report.c
    y = report.new_page("Nano position-reflection ablation", "9  V3-B001 DESIGN", "The registered intervention reflected movable-object center positions while holding robot, cameras, and nonmovable geometry fixed.")
    rounded_card(c, MARGIN_X, y - 96, CONTENT_W, 82, fill=SOFT_BLUE, stroke=colors.HexColor("#C8D6EF"))
    para(c, "<b>Question.</b> Does Nano's directional margin asymmetry change when only movable-object center positions are reflected about the robot sagittal plane?", MARGIN_X + 18, y - 25, CONTENT_W - 36, QUOTE)
    y -= 122
    cx = PAGE_W / 2
    box_w, box_h = 185, 108
    for idx, (label, fill, mirrored) in enumerate((("CONTROL LAYOUT", SOFT_BLUE, False), ("POSITION-REFLECTED LAYOUT", SOFT_TEAL, True))):
        x = MARGIN_X + idx * (box_w + 56)
        rounded_card(c, x, y - box_h, box_w, box_h, fill=fill)
        c.setFillColor(MUTED)
        c.setFont("Arial-Bold", 7.3)
        c.drawString(x + 12, y - 18, label)
        c.setStrokeColor(INK)
        c.line(x + box_w / 2, y - 92, x + box_w / 2, y - 35)
        left_x, right_x = x + 52, x + box_w - 52
        if mirrored:
            left_x, right_x = right_x, left_x
        c.setFillColor(LEFT)
        c.circle(left_x, y - 65, 8, fill=1, stroke=0)
        c.setFillColor(RIGHT)
        c.circle(right_x, y - 65, 8, fill=1, stroke=0)
        c.setFillColor(INK)
        c.rect(x + box_w / 2 - 9, y - 75, 18, 20, fill=1, stroke=0)
        c.setFillColor(MUTED)
        c.setFont("Arial", 6.6)
        c.drawCentredString(x + box_w / 2, y - 99, "robot and cameras fixed")
    c.setFillColor(PURPLE)
    c.setFont("Arial-Bold", 18)
    c.drawCentredString(cx, y - 70, "<->")
    y -= 140
    para(c, "27 prespecified seeds x 2 layouts x 2 static prompts = <b>108/108 valid episodes</b>. Success was control 26/27 LEFT and 26/27 RIGHT; reflected 27/27 LEFT and 23/27 RIGHT. The intervention is positions-only, not a full-scene mirror.", MARGIN_X, y, CONTENT_W, BODY)
    y -= 84
    c.setFont("Georgia-Bold", 15)
    c.setFillColor(INK)
    c.drawString(MARGIN_X, y, "Two geometric estimands")
    y -= 25
    rounded_card(c, MARGIN_X, y - 116, (CONTENT_W - 14) / 2, 106)
    para(c, "<b>Endpoint redirection D</b><br/><br/>RIGHT-condition final endpoint minus LEFT-condition final endpoint, oriented so positive means the requested LEFT-to-RIGHT ordering.", MARGIN_X + 14, y - 21, (CONTENT_W - 42) / 2, SMALL)
    rounded_card(c, MARGIN_X + (CONTENT_W + 14) / 2, y - 116, (CONTENT_W - 14) / 2, 106)
    para(c, "<b>Requested-depth contrast B</b><br/><br/>RIGHT requested-side margin minus LEFT requested-side margin. Positive means deeper placement under RIGHT than LEFT; negative reverses that contrast.", MARGIN_X + (CONTENT_W + 14) / 2 + 14, y - 21, (CONTENT_W - 42) / 2, SMALL)
    y -= 143
    rounded_card(c, MARGIN_X, y - 92, CONTENT_W, 82, fill=SOFT_PURPLE, stroke=colors.HexColor("#D4C8E4"))
    para(c, "<b>Primary interaction tests.</b> J asks whether position reflection changes endpoint redirection. I asks whether it changes requested-depth contrast. Bootstrap resampling and exact sign tests use the matched seed as the unit.", MARGIN_X + 17, y - 21, CONTENT_W - 34, BODY)
    report.footer("Nano V3-B001 positions-only ablation design")


def draw_nano_results(report: Report, nano: dict[str, Any]) -> None:
    c = report.c
    y = report.new_page("Redirection persisted; side depth reversed", "10  V3-B001 RESULTS", "All seed-level points are shown. Means use matched-seed bootstrap 95% intervals; p-values are exact two-sided sign tests.")
    seeds = nano["seed_level"]
    d_control = [100 * s["full_sample"]["D_control_m"] for s in seeds]
    d_mirror = [100 * s["full_sample"]["D_position_mirrored_m"] for s in seeds]
    j = [100 * s["full_sample"]["J_redirection_interaction_m"] for s in seeds]
    b_control = [100 * s["full_sample"]["B_control_m"] for s in seeds]
    b_mirror = [100 * s["full_sample"]["B_position_mirrored_m"] for s in seeds]
    interaction = [100 * s["full_sample"]["I_position_reflection_interaction_m"] for s in seeds]
    rounded_card(c, MARGIN_X, y - 250, CONTENT_W, 235)
    para(c, "Endpoint redirection", MARGIN_X + 16, y - 24, CONTENT_W - 32, H2)
    para(c, "The prompt still ordered endpoints in all 27 seeds under both layouts.", MARGIN_X + 16, y - 51, CONTENT_W - 32, SMALL)
    seed_strip(c, d_control, MARGIN_X + 25, y - 105, CONTENT_W - 50, -25, 85, LEFT, "Control D: mean +44.8 cm  [40.2, 49.4]")
    seed_strip(c, d_mirror, MARGIN_X + 25, y - 155, CONTENT_W - 50, -25, 85, TEAL, "Reflected D: mean +45.3 cm  [40.5, 50.3]")
    seed_strip(c, j, MARGIN_X + 25, y - 205, CONTENT_W - 50, -25, 85, PURPLE, "Interaction J: mean +0.5 cm  [-5.6, +6.6], p=0.701")
    y -= 274
    rounded_card(c, MARGIN_X, y - 250, CONTENT_W, 235)
    para(c, "Requested-side depth", MARGIN_X + 16, y - 24, CONTENT_W - 32, H2)
    para(c, "The same prompt redirection persisted, but how deeply the model placed the object on each requested side changed with geometry.", MARGIN_X + 16, y - 51, CONTENT_W - 32, SMALL)
    seed_strip(c, b_control, MARGIN_X + 25, y - 105, CONTENT_W - 50, -80, 50, LEFT, "Control B: mean +14.8 cm  [+9.6, +19.7]")
    seed_strip(c, b_mirror, MARGIN_X + 25, y - 155, CONTENT_W - 50, -80, 50, TEAL, "Reflected B: mean -10.0 cm  [-14.8, -5.3]")
    seed_strip(c, interaction, MARGIN_X + 25, y - 205, CONTENT_W - 50, -80, 50, PURPLE, "Interaction I: mean -24.8 cm  [-32.4, -17.3], p=4.92e-05")
    y -= 273
    rounded_card(c, MARGIN_X, y - 88, CONTENT_W, 78, fill=SOFT_TEAL, stroke=colors.HexColor("#C4DED5"))
    para(c, "<b>Interpretation.</b> The model-side language response remained strong, but the quantitative quality of the requested placement was causally altered by the registered object-position intervention. This does not identify training distribution, reachability, or embodiment as the mechanism.", MARGIN_X + 17, y - 20, CONTENT_W - 34, SMALL)
    report.footer("Nano V3-B001 geometric results")


def draw_nano_evidence(report: Report, nano: dict[str, Any]) -> None:
    c = report.c
    y = report.new_page("Failures and future-interface evidence", "11  V3-B001 EPISODE EVIDENCE", "Actual simulator execution and request-local predictions are displayed together but remain analytically distinct.")
    posters_dir = ROOT / "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/publication_media"
    base_url = "https://github.com/adeeb10abbas/steerable/blob/main/artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/publication_media/"
    cells = [
        ("CONTROL - LEFT", "nano_v3b001_seed9400_control_left_poster.png", "nano_v3b001_seed9400_control_left_actual_vs_local_predictions.mp4"),
        ("CONTROL - RIGHT", "nano_v3b001_seed9400_control_right_poster.png", "nano_v3b001_seed9400_control_right_actual_vs_local_predictions.mp4"),
        ("REFLECTED - LEFT", "nano_v3b001_seed9400_position_mirrored_left_poster.png", "nano_v3b001_seed9400_position_mirrored_left_actual_vs_local_predictions.mp4"),
        ("REFLECTED - RIGHT", "nano_v3b001_seed9400_position_mirrored_right_poster.png", "nano_v3b001_seed9400_position_mirrored_right_actual_vs_local_predictions.mp4"),
    ]
    cell_w = (CONTENT_W - 12) / 2
    cell_h = 142
    for idx, (label, poster, video) in enumerate(cells):
        col, row = idx % 2, idx // 2
        x = MARGIN_X + col * (cell_w + 12)
        yy = y - row * (cell_h + 30) - cell_h
        rounded_card(c, x, yy, cell_w, cell_h)
        crop_image(c, posters_dir / poster, x + 6, yy + 25, cell_w - 12, cell_h - 31, link=base_url + video)
        c.setFillColor(INK)
        c.setFont("Arial-Bold", 7.5)
        c.drawString(x + 8, yy + 10, label)
        c.setFillColor(TEAL)
        c.drawRightString(x + cell_w - 8, yy + 10, "CLICK FOR VIDEO")
    y -= 360
    rounded_card(c, MARGIN_X, y - 88, CONTENT_W, 78, fill=SOFT_RED, stroke=colors.HexColor("#E7C7BA"))
    para(c, "<b>Prediction boundary.</b> The right-hand video panels concatenate request-local 33-frame predictions returned during control. They are not one continuous full-task imagination, not robot execution, and not additional episodes.", MARGIN_X + 17, y - 20, CONTENT_W - 34, SMALL)
    y -= 108
    c.setFont("Georgia-Bold", 15)
    c.setFillColor(INK)
    c.drawString(MARGIN_X, y, "Failure taxonomy: 108 episodes")
    y -= 35
    counts = [("correct", 102, GREEN), ("transport failed", 5, colors.HexColor("#D5A145")), ("release failed", 1, colors.HexColor("#6C7780"))]
    bx = MARGIN_X
    for label, count, color in counts:
        width = CONTENT_W * count / 108
        c.setFillColor(color)
        c.rect(bx, y, width, 16, fill=1, stroke=0)
        bx += width
    c.setFillColor(MUTED)
    c.setFont("Arial", 7.3)
    c.drawString(MARGIN_X, y - 16, "102 correct  |  5 transport_failed  |  1 release_failed  |  0 pick_failed  |  0 wrong_side")
    para(c, "The cohort retained 738 decoded local predictions and 21,972 executed actions. Success-conditional interaction analysis uses only the 21 seeds correct in all four cells; no failure is converted to zero.", MARGIN_X, y - 44, CONTENT_W, SMALL)
    report.footer("Nano V3-B001 actual execution and local-prediction evidence")


def draw_interfaces_guidance(report: Report) -> None:
    c = report.c
    y = report.new_page("World-model evidence is not robot execution", "12  FUTURE INTERFACES AND GUIDANCE", "Decoded videos, latent futures, and action-only policies expose different evidence and must not share a success denominator.")
    rows = [
        ["Checkpoint", "Executed rollouts", "Future evidence", "How it is used"],
        ["Cosmos3 Edge Policy", "54", "452 decoded local futures", "interface evidence; not extra trials"],
        ["Cosmos3 Nano Policy", "54", "349 local predictions", "request-local horizons; not full-task imagination"],
        ["DreamZero s=2", "54", "54 official full-reset decodes + 2,554 latent futures", "imagination reported beside, not as, execution"],
        ["Efficient-WAM-RT", "126", "126 decoded coarse futures", "RoboTwin future evidence"],
        ["FastWAM", "126", "action-only at test time", "no missing future encoded as zero"],
        ["LingBot-VA", "126", "126 latent-only futures", "latent-only, never substituted with video"],
        ["pi0.5 / GR00T / pi0-FAST", "54 / 54 / 40", "action-only", "VLA execution evidence"],
    ]
    table = Table(rows, colWidths=[115, 66, 186, 142], rowHeights=[28] + [42] * 7)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Arial-Bold"),
        ("FONTNAME", (1, 1), (-1, -1), "Arial"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("LEADING", (0, 0), (-1, -1), 9.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("BACKGROUND", (0, 1), (-1, -1), CARD),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [CARD, colors.HexColor("#F2EEE6")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ]))
    table.wrapOn(c, CONTENT_W, 360)
    table.drawOn(c, MARGIN_X, y - 335)
    y -= 370
    c.setFont("Georgia-Bold", 15)
    c.setFillColor(INK)
    c.drawString(MARGIN_X, y, "Small guidance ablation (descriptive)")
    y -= 28
    rounded_card(c, MARGIN_X, y - 112, (CONTENT_W - 14) / 2, 102, fill=SOFT_BLUE, stroke=colors.HexColor("#C8D6EF"))
    para(c, "<b>Nano local guidance g=3 to g=1</b><br/><br/>Success 6/6 to 4/6. LEFT 3/3 to 1/3; RIGHT 3/3 to 3/3. The requested-margin gap shrank because RIGHT depth fell more; weak LEFT placement did not improve.", MARGIN_X + 14, y - 20, (CONTENT_W - 42) / 2, SMALL)
    rounded_card(c, MARGIN_X + (CONTENT_W + 14) / 2, y - 112, (CONTENT_W - 14) / 2, 102, fill=SOFT_RED, stroke=colors.HexColor("#E7C7BA"))
    para(c, "<b>DreamZero s=1 to derived action guidance s=2</b><br/><br/>Success 3/6 to 4/6. LEFT 2/3 to 1/3; RIGHT 1/3 to 3/3. Bias moved toward RIGHT; this is redistribution, not a powered improvement claim.", MARGIN_X + (CONTENT_W + 14) / 2 + 14, y - 20, (CONTENT_W - 42) / 2, SMALL)
    para(c, "Both guidance comparisons use only three seeds per direction and remain descriptive. DreamZero s=2 is a derived negative-branch action-guidance intervention, not official action-CFG. Additional Cosmos3 Super 64B and Cosmos3 Edge base artifacts are prediction/action probes without simulator execution; Cosmos-Reason2 has no behavioral denominator.", MARGIN_X, 80, CONTENT_W, TINY)
    report.footer("Future-interface taxonomy and V2-A015 descriptive guidance ablation")


def draw_interpretation(report: Report) -> None:
    c = report.c
    y = report.new_page("What the evidence supports", "13  INTERPRETATION", "The strongest claim is checkpoint-specific: language reliably perturbs behavior, but semantic control and competence can separate.")
    supported = [
        "Within an exact checkpoint and matched reset, changing the static relation word caused different executed action traces.",
        "Endpoint ordering often followed the requested direction, including in many task failures.",
        "Task competence remained checkpoint-, direction-, wording-, and scene-dependent.",
        "Nano's registered positions-only intervention causally changed requested-side depth while preserving endpoint redirection.",
    ]
    unsupported = [
        "A universal VLA-versus-WAM ranking or a pooled cross-arena success rate.",
        "Training-data imbalance as the mechanism behind directional asymmetry.",
        "Isolation of reachability, embodiment, base handedness, or a full-scene mirror.",
        "Decoded or latent futures as additional trials, robot outcomes, or proof that imagination caused execution.",
    ]
    col_w = (CONTENT_W - 16) / 2
    rounded_card(c, MARGIN_X, y - 330, col_w, 315, fill=SOFT_TEAL, stroke=colors.HexColor("#C4DED5"))
    para(c, "Supported", MARGIN_X + 16, y - 25, col_w - 32, H2)
    yy = y - 64
    for item in supported:
        c.setFillColor(TEAL)
        c.circle(MARGIN_X + 23, yy - 4, 3.5, fill=1, stroke=0)
        yy -= para(c, item, MARGIN_X + 36, yy + 4, col_w - 54, SMALL) + 18
    rounded_card(c, MARGIN_X + col_w + 16, y - 330, col_w, 315, fill=SOFT_RED, stroke=colors.HexColor("#E7C7BA"))
    para(c, "Not supported", MARGIN_X + col_w + 32, y - 25, col_w - 32, H2)
    yy = y - 64
    for item in unsupported:
        c.setFillColor(RIGHT)
        c.circle(MARGIN_X + col_w + 39, yy - 4, 3.5, fill=1, stroke=0)
        yy -= para(c, item, MARGIN_X + col_w + 52, yy + 4, col_w - 54, SMALL) + 18
    y -= 360
    rounded_card(c, MARGIN_X, y - 142, CONTENT_W, 132, fill=SOFT_PURPLE, stroke=colors.HexColor("#D4C8E4"))
    para(c, "A more precise story than \"the model succeeded\"", MARGIN_X + 18, y - 24, CONTENT_W - 36, H3)
    para(c, "Success is a binary endpoint of a longer causal chain: recognize the prompt, select a distinct action sequence, contact and pick the object, transport it in the requested semantic direction, enter and sustain the target region, and release. The expanded data show failures at different links. Directional asymmetry can therefore coexist with both high sensitivity and apparently plausible trajectories.", MARGIN_X + 18, y - 54, CONTENT_W - 36, BODY)
    y -= 170
    para(c, "The practical implication is that best-of-N or language-guided sampling should optimize a continuous quality measure such as requested-side margin, while retaining the full task predicate and failure taxonomy. Binary success alone is insensitive at ceiling and opaque about why a rollout failed.", MARGIN_X, y, CONTENT_W, BODY)
    report.footer("Interpretation and causal boundary")


def draw_limitations(report: Report) -> None:
    c = report.c
    y = report.new_page("Limitations and current status", "14  CLAIM BOUNDARY", "The report distinguishes completed evidence from registered or externally blocked work.")
    limitations = [
        ("Checkpoint and simulator scope", "Results identify exact runtimes, controllers, action spaces, and scene distributions. They do not establish a model-family ranking."),
        ("Arena separation", "DROID/RoboLab and RoboTwin use different tasks and success rules. Their rates are never pooled."),
        ("RoboTwin nesting", "Nine stochastic replicates are nested within each of seven scenes. Episode-level intervals do not represent 63-scene generalization."),
        ("Historical pi0-FAST", "Exact historical OpenPI/RoboLab revisions remain unavailable. The 40-episode public compatibility cohort is separate and cannot replace them."),
        ("Measurement coverage", "The committed 982/982 audit covers pre-B001 episodes; the later 108 B001 episodes are separately complete. No new 1,090-episode pooled audit is claimed."),
        ("Future interfaces", "Decoded, local, full-reset, latent-only, and action-only evidence are labeled separately; missing futures are never zeros."),
    ]
    for idx, (title, body) in enumerate(limitations):
        col, row = idx % 2, idx // 2
        w = (CONTENT_W - 14) / 2
        x = MARGIN_X + col * (w + 14)
        yy = y - row * 126 - 110
        rounded_card(c, x, yy, w, 104)
        para(c, title, x + 14, yy + 82, w - 28, H3)
        para(c, body, x + 14, yy + 54, w - 28, SMALL)
    y -= 404
    c.setFont("Georgia-Bold", 15)
    c.setFillColor(INK)
    c.drawString(MARGIN_X, y, "Released and unreleased work")
    y -= 30
    status_rows = [
        ["Layer", "Status", "Accounting"],
        ["V3 Phase A", "complete", "648/648 authorized-new episodes"],
        ["pi0-FAST V3-A002", "complete, separate", "40 episodes; not historical recovery"],
        ["Nano V3-B001", "complete", "108/108 valid episodes"],
        ["Other Phase B ablations", "unreleased", "no result claimed"],
        ["V3 Phase C wording", "registered, unreleased", "0/480 released episodes"],
        ["Eligible Phase D repeats", "unreleased", "no result claimed"],
    ]
    table = Table(status_rows, colWidths=[150, 122, 237], rowHeights=[25] + [31] * 6)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Arial-Bold"),
        ("FONTNAME", (1, 1), (-1, -1), "Arial"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [CARD, colors.HexColor("#F2EEE6")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ]))
    table.wrapOn(c, CONTENT_W, 220)
    table.drawOn(c, MARGIN_X, 54)
    report.footer("Limitations and release status")


def draw_droid_table(report: Report, droid: list[dict[str, Any]], compatibility: dict[str, Any]) -> None:
    c = report.c
    y = report.new_page("Appendix A: DROID checkpoint table", "15  COMPLETE RESULT TABLE", "Expanded Phase A and the public compatibility cohort are separated from bounded historical reference evidence.")
    rows: list[list[Any]] = [["Cohort / checkpoint", "Type", "Episodes", "LEFT", "RIGHT", "Aligned", "Actions", "Shift", "C/P/T/W/R", "Future interface"]]
    for item in droid:
        tax = "/".join(str(item["taxonomy"][k]) for k in ("correct", "pick_failed", "transport_failed", "wrong_side", "release_failed"))
        rows.append([item["label"], item["kind"], str(item["episodes"]), f'{item["left"]}/{item["left_n"]}', f'{item["right"]}/{item["right_n"]}', f'{item["aligned"]}/{item["pairs"]}', f'{item["distinct"]}/{item["pairs"]}', f'{item["shift_cm"]:+.1f} cm', tax, item["future"]])
    tax = "/".join(str(compatibility["taxonomy"][k]) for k in ("correct", "pick_failed", "transport_failed", "wrong_side", "release_failed"))
    rows.append(["SEPARATE: " + compatibility["label"], compatibility["kind"], str(compatibility["episodes"]), f'{compatibility["left"]}/{compatibility["left_n"]}', f'{compatibility["right"]}/{compatibility["right_n"]}', f'{compatibility["aligned"]}/{compatibility["pairs"]}', f'{compatibility["distinct"]}/{compatibility["pairs"]}', f'{compatibility["shift_cm"]:+.1f} cm', tax, compatibility["future"]])
    rows.append(["BOUNDED V2: pi0-FAST frozen reference", "VLA", "20", "1/10", "10/10", "10/10", "10/10", "+27.9 cm", "not harmonized", "action-only"])
    wrapped = [
        [Paragraph(str(value), TABLE_HEADER if row_idx == 0 else (TABLE_TINY_BOLD if col_idx == 0 else TABLE_TINY)) for col_idx, value in enumerate(row)]
        for row_idx, row in enumerate(rows)
    ]
    widths = [102, 28, 35, 35, 35, 40, 40, 44, 55, 96]
    table = Table(wrapped, colWidths=widths, rowHeights=[34] + [55] * (len(rows) - 1))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, 5), [CARD, colors.HexColor("#F2EEE6")]),
        ("BACKGROUND", (0, 6), (-1, 6), SOFT_PURPLE),
        ("BACKGROUND", (0, 7), (-1, 7), SOFT_RED),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    table.wrapOn(c, CONTENT_W, 500)
    table.drawOn(c, MARGIN_X, y - 450)
    y -= 480
    rounded_card(c, MARGIN_X, y - 118, CONTENT_W, 108, fill=SOFT_BLUE, stroke=colors.HexColor("#C8D6EF"))
    para(c, "<b>How to read the table.</b> Shift is the requested-oriented mean endpoint displacement: positive follows the requested LEFT-to-RIGHT ordering. C/P/T/W/R means correct, pick_failed, transport_failed, wrong_side, release_failed. The bounded historical row predates the harmonized V3 taxonomy and is not combined with the compatibility cohort.", MARGIN_X + 17, y - 20, CONTENT_W - 34, SMALL)
    para(c, "All five expanded DROID checkpoints have 27/27 distinct action pairs. Wilson intervals and exact paired tests are retained in the source summaries; the main text reports the named tests and avoids cross-arena pooling.", MARGIN_X + 17, y - 82, CONTENT_W - 34, TINY)
    report.footer("Appendix A1: DROID checkpoint results")


def draw_robotwin_table(report: Report, robotwin: list[dict[str, Any]]) -> None:
    c = report.c
    y = report.new_page("Appendix A: RoboTwin checkpoint table", "16  COMPLETE RESULT TABLE", "Expanded replicate evidence and bounded references are shown in separate rows; no DROID result is pooled here.")
    rows: list[list[Any]] = [["Cohort / checkpoint", "Type", "Episodes", "LEFT", "RIGHT", "Aligned", "Actions", "Mean / median shift", "C/P/T/W/R", "Future interface"]]
    for item in robotwin:
        tax = "/".join(str(item["taxonomy"][k]) for k in ("correct", "pick_failed", "transport_failed", "wrong_side", "release_failed"))
        rows.append([item["label"], item["kind"], str(item["episodes"]), f'{item["left"]}/{item["left_n"]}', f'{item["right"]}/{item["right_n"]}', f'{item["aligned"]}/{item["pairs"]}', f'{item["distinct"]}/{item["pairs"]}', f'{item["shift_cm"]:+.1f} / {item["shift_median_cm"]:+.1f} cm', tax, item["future"]])
    rows.extend([
        ["BOUNDED V2: LingBot-VLA 4B", "VLA", "6", "1/3", "0/3", "2/3", "not reported", "-0.9 cm", "not harmonized", "none"],
        ["BOUNDED V2: Light-WAM", "WAM", "6", "1/3", "0/3", "1/3", "3/3", "-3.5 cm", "not harmonized", "action-only"],
    ])
    wrapped = [
        [Paragraph(str(value), TABLE_HEADER if row_idx == 0 else (TABLE_TINY_BOLD if col_idx == 0 else TABLE_TINY)) for col_idx, value in enumerate(row)]
        for row_idx, row in enumerate(rows)
    ]
    widths = [102, 28, 35, 35, 35, 40, 40, 62, 55, 78]
    table = Table(wrapped, colWidths=widths, rowHeights=[35] + [58] * (len(rows) - 1))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, 3), [CARD, colors.HexColor("#F2EEE6")]),
        ("BACKGROUND", (0, 4), (-1, -1), SOFT_RED),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    table.wrapOn(c, CONTENT_W, 420)
    table.drawOn(c, MARGIN_X, y - 370)
    y -= 403
    rounded_card(c, MARGIN_X, y - 145, CONTENT_W, 135, fill=SOFT_TEAL, stroke=colors.HexColor("#C4DED5"))
    para(c, "<b>Nesting and outliers.</b> The 63 expanded pairs per checkpoint are nine stochastic replicates nested within seven scenes. Efficient-WAM-RT has rare reverse-tail outliers that flip its mean shift negative (-14.4 cm) even though the median remains positive (+4.4 cm); report both. FastWAM is near zero by both summaries, while LingBot-VA shows a positive mean and median.", MARGIN_X + 17, y - 20, CONTENT_W - 34, SMALL)
    para(c, "Historical r00 coverage remains separate: Efficient-WAM-RT 3/7 LEFT and 2/7 RIGHT; FastWAM 1/7 and 1/7; LingBot-VA 3/7 and 4/7. Those rows are not added to the expanded denominators above.", MARGIN_X + 17, y - 98, CONTENT_W - 34, TINY)
    report.footer("Appendix A2: RoboTwin checkpoint results")


def draw_prompt_registry(report: Report) -> None:
    c = report.c
    y = report.new_page("Appendix B: exact DROID prompts", "17  PROMPTS", "Every LEFT and RIGHT string is printed in full. Historical V1 wording is not byte-identical to V3.")
    rounded_card(c, MARGIN_X, y - 105, CONTENT_W, 92, fill=SOFT_BLUE, stroke=colors.HexColor("#C8D6EF"))
    para(c, "<b>Current V3 direct command</b>", MARGIN_X + 16, y - 20, CONTENT_W - 32, H3)
    para(c, "<font color='#2563EB'><b>LEFT</b></font>  Put the Rubik's cube to the left of the bowl.<br/><font color='#D65332'><b>RIGHT</b></font>  Put the Rubik's cube to the right of the bowl.", MARGIN_X + 16, y - 48, CONTENT_W - 32, SMALL)
    y -= 127
    c.setFont("Georgia-Bold", 14)
    c.setFillColor(INK)
    c.drawString(MARGIN_X, y, "Historical V1 wording layer")
    y -= 18
    historical = [
        ("Canonical", "Put the rubiks cube to the left of the bowl", "Put the rubiks cube to the right of the bowl"),
        ("Short", "Put the cube left of the bowl", "Put the cube right of the bowl"),
        ("Declarative", "The rubiks cube should end up to the left of the bowl", "The rubiks cube should end up to the right of the bowl"),
        ("Contrastive", "Put the rubiks cube to the left of the bowl, not to the right of the bowl", "Put the rubiks cube to the right of the bowl, not to the left of the bowl"),
    ]
    for label, left_prompt, right_prompt in historical:
        rounded_card(c, MARGIN_X, y - 88, CONTENT_W, 78)
        c.setFillColor(MUTED)
        c.setFont("Arial-Bold", 7.2)
        c.drawString(MARGIN_X + 13, y - 25, label.upper())
        para(c, f"<font color='#2563EB'><b>LEFT</b></font>  {left_prompt}<br/><font color='#D65332'><b>RIGHT</b></font>  {right_prompt}", MARGIN_X + 90, y - 19, CONTENT_W - 106, SMALL)
        y -= 92
    rounded_card(c, MARGIN_X, y - 88, CONTENT_W, 78, fill=SOFT_PURPLE, stroke=colors.HexColor("#D4C8E4"))
    para(c, "<b>Cohort boundary.</b> These eight V1 strings were lower-case and unpunctuated. They produced the completed 160-episode historical wording layer. They are not the title-cased, punctuated prompts registered for unreleased V3 Phase C.", MARGIN_X + 16, y - 19, CONTENT_W - 32, SMALL)
    report.footer("Appendix B1: exact DROID prompt strings")

    y = report.new_page("Appendix B: exact RoboTwin prompts", "18  PROMPTS", "pair03-pair09 are the expanded V3 scene set; pair00-pair02 are bounded historical scenes.")
    pairs = [
        ("pair00", "bounded", "blue soap", "tea-box"),
        ("pair01", "bounded", "box of playingcards", "rubikscube"),
        ("pair02", "bounded", "box with cards inside", "red coffee-box"),
        ("pair03", "expanded", "small woodenblock", "red playingcards box"),
        ("pair04", "expanded", "plastic mouse", "blue stapler"),
        ("pair05", "expanded", "box of playingcards", "rubikscube"),
        ("pair06", "expanded", "coffee box", "red playingcards box"),
        ("pair07", "expanded", "golden bread", "blue stapler"),
        ("pair08", "expanded", "box with cards inside", "black phone"),
        ("pair09", "expanded", "rubikscube", "brown woodenblock"),
    ]
    for pair_id, cohort, movable, reference in pairs:
        fill = SOFT_TEAL if cohort == "expanded" else CARD
        stroke = colors.HexColor("#C4DED5") if cohort == "expanded" else LINE
        rounded_card(c, MARGIN_X, y - 59, CONTENT_W, 51, fill=fill, stroke=stroke)
        c.setFillColor(TEAL if cohort == "expanded" else MUTED)
        c.setFont("Arial-Bold", 7.1)
        c.drawString(MARGIN_X + 12, y - 23, f"{pair_id}  {cohort}".upper())
        left_prompt = f"Put the {movable} to the left of the {reference}."
        right_prompt = f"Put the {movable} to the right of the {reference}."
        para(c, f"<font color='#2563EB'><b>LEFT</b></font>  {left_prompt}<br/><font color='#D65332'><b>RIGHT</b></font>  {right_prompt}", MARGIN_X + 110, y - 13, CONTENT_W - 125, TINY)
        y -= 62
    para(c, "Each matched pair holds scene, reset, seeds, runtime, controller, and horizon fixed. Only the complete static instruction's requested relation changes.", MARGIN_X, 64, CONTENT_W, TINY)
    report.footer("Appendix B2: exact RoboTwin prompt strings")


def github_media_url(path: str) -> str:
    return "https://github.com/adeeb10abbas/steerable/blob/main/" + path


def media_button(c: canvas.Canvas, label: str, url: str, x: float, y: float, w: float = 54) -> None:
    c.setFillColor(SOFT_TEAL)
    c.setStrokeColor(colors.HexColor("#B9D8CE"))
    c.roundRect(x, y, w, 18, 6, fill=1, stroke=1)
    c.setFillColor(TEAL)
    c.setFont("Arial-Bold", 6.2)
    c.drawCentredString(x + w / 2, y + 6, label.upper())
    c.linkURL(url, (x, y, x + w, y + 18), relative=0)


def draw_media_index(report: Report) -> None:
    c = report.c
    y = report.new_page("Appendix C: publication media index", "19  MEDIA", "PDF buttons open committed repository videos. Actual execution, model prediction, and imagination remain labeled separately.")
    rounded_card(c, MARGIN_X, y - 84, CONTENT_W, 72, fill=SOFT_BLUE, stroke=colors.HexColor("#C8D6EF"))
    para(c, "<b>DROID / RoboLab prompt pair</b><br/><font color='#2563EB'><b>LEFT</b></font> Put the Rubik's cube to the left of the bowl. &nbsp;&nbsp; <font color='#D65332'><b>RIGHT</b></font> Put the Rubik's cube to the right of the bowl.", MARGIN_X + 15, y - 19, CONTENT_W - 30, SMALL)
    y -= 105
    droid_media = [
        ("pi0-FAST compatibility", "actual", "artifacts/vla_wam_shared_v3/media/pi0_fast_old_name_config_v3a002/pi0_fast_v3a002_seed8311_paired_actual.mp4"),
        ("pi0.5 current stack", "actual", "artifacts/vla_wam_shared_v2/media/pi05_current_stack_v2a010/pi05_current_stack_v2a010_seed8300_paired_actual.mp4"),
        ("GR00T N1.7", "actual", "artifacts/vla_wam_shared_v2/media/groot_n17_droid/groot_n17_droid_seed8301_pair.mp4"),
        ("Cosmos3 Edge Policy", "actual", "artifacts/vla_wam_shared_v2/media/cosmos3_edge_droid/cosmos3_edge_seed8302_paired.mp4"),
        ("Cosmos3 Nano Policy", "actual", "artifacts/vla_wam_shared_v2/media/cosmos3_nano_policy_droid_v2a011/cosmos3_nano_v2a011_seed8300_paired_actual.mp4"),
        ("Cosmos3 Nano Policy", "prediction", "artifacts/vla_wam_shared_v2/media/cosmos3_nano_policy_droid_v2a011/cosmos3_nano_v2a011_seed8300_paired_model_prediction.mp4"),
        ("DreamZero", "actual", "artifacts/vla_wam_shared_v2/media/dreamzero_droid/dreamzero_droid_seed8300_paired.mp4"),
        ("DreamZero", "full imagination", "artifacts/vla_wam_shared_v2/media/dreamzero_droid/imagination/paired/dreamzero_seed8300_left_right_imagined_futures.mp4"),
    ]
    rounded_card(c, MARGIN_X, y - 222, CONTENT_W, 212)
    c.setFillColor(INK); c.setFont("Georgia-Bold", 13); c.drawString(MARGIN_X + 14, y - 25, "DROID execution and future media")
    for idx, (model, kind, path) in enumerate(droid_media):
        yy = y - 51 - idx * 19
        c.setFillColor(INK); c.setFont("Arial-Bold", 7.2); c.drawString(MARGIN_X + 16, yy, model)
        media_button(c, kind, github_media_url(path), PAGE_W - MARGIN_X - 82, yy - 7, 66)
    y -= 242
    c.setFillColor(INK); c.setFont("Georgia-Bold", 13); c.drawString(MARGIN_X, y, "RoboTwin execution media")
    y -= 18
    robotwin_media = [
        ("Efficient-WAM-RT - pair05", "Put the box of playingcards to the left of the rubikscube.", "Put the box of playingcards to the right of the rubikscube.", "artifacts/vla_wam_shared_v2/media/robotwin_wam_confirmation/efficient_wam_rt_pair05_left_right_both_success.mp4"),
        ("FastWAM - pair02", "Put the box with cards inside to the left of the red coffee-box.", "Put the box with cards inside to the right of the red coffee-box.", "artifacts/vla_wam_shared_v2/media/robotwin_wam_pairs/fastwam_pair02_left_success_right_failure.mp4"),
        ("LingBot-VA - pair03", "Put the small woodenblock to the left of the red playingcards box.", "Put the small woodenblock to the right of the red playingcards box.", "artifacts/vla_wam_shared_v2/media/robotwin_wam_confirmation/lingbot_va_pair03_left_right_normalized_full_rollouts.mp4"),
        ("LingBot-VLA 4B - pair00", "Put the blue soap to the left of the tea-box.", "Put the blue soap to the right of the tea-box.", "artifacts/vla_wam_shared_v2/pilot/expansion/media/lingbot_vla_4b_pair00_matched.mp4"),
        ("Light-WAM - pair00", "Put the blue soap to the left of the tea-box.", "Put the blue soap to the right of the tea-box.", "artifacts/vla_wam_shared_v2/media/light_wam_robotwin/light_wam_pair00_left_success_right_failure.mp4"),
    ]
    for model, left_prompt, right_prompt, path in robotwin_media:
        rounded_card(c, MARGIN_X, y - 40, CONTENT_W, 35)
        c.setFillColor(INK); c.setFont("Arial-Bold", 6.8); c.drawString(MARGIN_X + 10, y - 22, model)
        para(c, f"<font color='#2563EB'><b>LEFT</b></font> {left_prompt}<br/><font color='#D65332'><b>RIGHT</b></font> {right_prompt}", MARGIN_X + 145, y - 8, CONTENT_W - 225, TINY)
        media_button(c, "actual", github_media_url(path), PAGE_W - MARGIN_X - 67, y - 32, 51)
        y -= 42
    y -= 4
    rounded_card(c, MARGIN_X, y - 90, CONTENT_W, 82, fill=SOFT_PURPLE, stroke=colors.HexColor("#D4C8E4"))
    para(c, "<b>Prediction-only probes.</b> Cosmos3 Super 64B and Cosmos3 Edge base have committed prediction/action videos but no robot-execution score. Cosmos-Reason2 has no behavioral video and is not represented by substituted media.", MARGIN_X + 15, y - 19, CONTENT_W - 150, TINY)
    media_button(c, "Edge base", github_media_url("artifacts/vla_wam_shared_v2/media/cosmos3_edge_base_v2a013/cosmos3_edge_base_v2a013_seed8300_paired_model_prediction.mp4"), PAGE_W - MARGIN_X - 236, y - 54, 104)
    media_button(c, "Super probe", github_media_url("artifacts/vla_wam_shared_v2/media/cosmos3_super_base_v2a014/cosmos3_super_v2a014_paired_prediction_and_actions.mp4"), PAGE_W - MARGIN_X - 124, y - 54, 108)
    report.footer("Appendix C: selected committed publication media")


def draw_methods(report: Report, sources: Iterable[Path]) -> None:
    c = report.c
    y = report.new_page("Appendix D: methods and provenance", "20  REPRODUCIBILITY", "The PDF is generated from committed summaries and selected publication assets; raw rollouts and checkpoints remain outside ordinary Git.")
    c.setFont("Georgia-Bold", 14)
    c.setFillColor(INK)
    c.drawString(MARGIN_X, y, "Metric and statistical definitions")
    y -= 25
    methods = [
        ("Requested success", "Frozen arena-specific relation-and-detached-release predicate. All valid behavioral failures remain in the denominator."),
        ("Endpoint alignment", "Strict requested ordering of the matched LEFT and RIGHT final lateral endpoints. Exact sign tests exclude ties."),
        ("Requested-side margin", "Signed geometric depth into the requested region. Success-conditional analyses omit failures; no missing value becomes zero."),
        ("Intervals", "V3 direction rates use Wilson 95% intervals. Nano means use 20,000 matched-seed bootstrap replicates. V1 wording intervals are Beta(1,1) posterior credible intervals."),
        ("Paired tests", "Exact two-sided McNemar tests use discordant success pairs. Exact two-sided sign tests use non-tied geometric differences."),
        ("Failure taxonomy", "Precedence: correct; pick_failed; wrong_side; release_failed; transport_failed. Original frozen stage remains retained."),
    ]
    for idx, (name, detail) in enumerate(methods):
        col, row = idx % 2, idx // 2
        w = (CONTENT_W - 14) / 2
        x = MARGIN_X + col * (w + 14)
        yy = y - row * 103 - 90
        rounded_card(c, x, yy, w, 84)
        para(c, name, x + 12, yy + 66, w - 24, H3)
        para(c, detail, x + 12, yy + 43, w - 24, TINY)
    y -= 330
    c.setFont("Georgia-Bold", 14)
    c.setFillColor(INK)
    c.drawString(MARGIN_X, y, "Authoritative evidence map")
    y -= 25
    evidence = [
        "docs/VLA_WAM_STEERABILITY_V3_PROTOCOL.md",
        "docs/VLA_WAM_V3_CONTINUATION.md",
        "artifacts/vla_wam_shared_v3/continuation_state.json",
        "artifacts/vla_wam_shared_v3/results/*_phase_a_summary.json",
        "artifacts/vla_wam_shared_v3/results/pi0_fast_old_name_config_v3a002_summary.json",
        "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/nano_v3b001_summary.json",
        "artifacts/vla_wam_shared_v3/measurement_coverage_audit.json",
        "artifacts/vla_wam_shared_v1/run_manifest.json and final_evidence/",
    ]
    for idx, item in enumerate(evidence):
        col, row = idx % 2, idx // 2
        x = MARGIN_X + col * (CONTENT_W / 2)
        yy = y - row * 34
        c.setFillColor(TEAL)
        c.circle(x + 4, yy - 3, 2.4, fill=1, stroke=0)
        para(c, item, x + 13, yy + 3, CONTENT_W / 2 - 20, TINY)
    y -= 158
    rounded_card(c, MARGIN_X, y - 102, CONTENT_W, 92, fill=SOFT_BLUE, stroke=colors.HexColor("#C8D6EF"))
    para(c, "<b>Rebuild contract.</b> Run <font name='Courier'>tmp/pdfs/.venv/bin/python tools/build_vla_wam_research_pdf.py</font>. The builder reads the committed JSON summaries above, selected publication figures under <font name='Courier'>artifacts/vla_wam_shared_v3/publication_pdf/</font>, and committed Nano posters. It writes a deterministic PDF and SHA-bearing manifest under <font name='Courier'>output/pdf/</font>.", MARGIN_X + 17, y - 20, CONTENT_W - 34, SMALL)
    report.footer("Appendix D: methods, evidence provenance, and rebuild contract")


def write_manifest(output: Path, sources: Iterable[Path], page_count: int) -> Path:
    manifest_path = output.with_suffix(".manifest.json")
    manifest = {
        "schema_version": "vla-wam-research-pdf-manifest-v1",
        "report": {
            "path": str(output.relative_to(ROOT)),
            "sha256": sha256(output),
            "bytes": output.stat().st_size,
            "pages": page_count,
        },
        "source_files": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(set(sources))
        ],
        "cohort_boundaries": {
            "droid_and_robotwin_success_never_pooled": True,
            "pi0_fast_compatibility_separate_from_historical": True,
            "v1_wording_separate_from_v3_phase_c": True,
            "futures_never_scored_as_behavioral_episodes": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def build(output: Path) -> tuple[Path, Path]:
    register_fonts()
    droid, robotwin, compatibility, nano = load_study_data()
    report = Report(output)
    draw_cover(report)
    draw_question(report, droid, robotwin)
    draw_intervention(report)
    draw_estimands(report)
    draw_overview(report, droid, robotwin)
    draw_droid(report, droid, compatibility)
    draw_robotwin(report, robotwin)
    draw_wording_success(report)
    draw_wording_shift(report)
    draw_nano_design(report, nano)
    draw_nano_results(report, nano)
    draw_nano_evidence(report, nano)
    draw_interfaces_guidance(report)
    draw_interpretation(report)
    draw_limitations(report)
    draw_droid_table(report, droid, compatibility)
    draw_robotwin_table(report, robotwin)
    draw_prompt_registry(report)
    draw_media_index(report)
    sources = [item["source"] for item in droid + robotwin] + [compatibility["source"]]
    sources.extend([
        Path(nano["_source"]),
        ROOT / "artifacts/vla_wam_shared_v3/continuation_state.json",
        ROOT / "artifacts/vla_wam_shared_v3/measurement_coverage_audit.json",
        ROOT / "artifacts/vla_wam_shared_v1/run_manifest.json",
        ROOT / "artifacts/vla_wam_shared_v1/final_evidence/closed_loop_summary.json",
        ROOT / "artifacts/vla_wam_shared_v3/publication_pdf/figures/droid-paired-shifts.png",
        ROOT / "artifacts/vla_wam_shared_v3/publication_pdf/figures/robotwin-diagnostics.png",
    ])
    draw_methods(report, sources)
    report.finish()
    manifest = write_manifest(output, sources, report.page_number)
    return output, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    pdf, manifest = build(output)
    print(pdf)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
