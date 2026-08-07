#!/usr/bin/env python3
"""Build the compact, figure-led scientific report for the V3 study.

The builder is intentionally fail-closed: it requires the completed three-model
reflection and phrasing figures plus the released Tier-B stochastic and factor
ablations. It never substitutes historical plots or partial experiment outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output/pdf/language_sensitivity_geometry_scientific_report.pdf"
PAGE_W, PAGE_H = landscape(A4)
MARGIN = 31

BG = colors.HexColor("#F6F1E8")
CARD = colors.HexColor("#FCFAF5")
INK = colors.HexColor("#17232B")
MUTED = colors.HexColor("#5E6A70")
LINE = colors.HexColor("#D7D0C5")
LEFT = colors.HexColor("#B85F35")
RIGHT = colors.HexColor("#286EA6")
TEAL = colors.HexColor("#2D7A63")
PURPLE = colors.HexColor("#6B55A5")

FIGURES = {
    1: ROOT / "artifacts/vla_wam_shared_v3/analysis/paper_figures/figure1_nano_instrument_sensitivity.png",
    2: ROOT / "artifacts/vla_wam_shared_v3/analysis/paper_figures/figure2_three_checkpoint_position_reflection.png",
    3: ROOT / "artifacts/vla_wam_shared_v3/phase_b/nano_lateral_sweep_v3b005/results/figures/figure3_nano_lateral_dose_response.png",
    4: ROOT / "artifacts/vla_wam_shared_v3/analysis/mechanism/figures/figure4_gap_vs_competence.png",
    5: ROOT / "artifacts/vla_wam_shared_v3/analysis/paper_figures/figure5_cross_arena_directional_success.png",
    6: ROOT / "artifacts/vla_wam_shared_v3/analysis/mechanism/figures/figure6_failure_taxonomy_by_direction.png",
    7: ROOT / "artifacts/vla_wam_shared_v3/phase_c/four_phrasings_v3c001/results/figures/figure7_phase_c_phrasing_direction.png",
}

SOURCES = {
    "pi05_phase_a": ROOT / "artifacts/vla_wam_shared_v3/results/pi05_current_stack_droid_phase_a_summary.json",
    "groot_phase_a": ROOT / "artifacts/vla_wam_shared_v3/results/groot_n17_droid_phase_a_summary.json",
    "edge_phase_a": ROOT / "artifacts/vla_wam_shared_v3/results/cosmos3_edge_policy_droid_phase_a_summary.json",
    "nano_phase_a": ROOT / "artifacts/vla_wam_shared_v3/results/cosmos3_nano_policy_droid_phase_a_summary.json",
    "dream_phase_a": ROOT / "artifacts/vla_wam_shared_v3/results/dreamzero_droid_action_cfg_phase_a_summary.json",
    "pi0_compatibility": ROOT / "artifacts/vla_wam_shared_v3/results/pi0_fast_old_name_config_v3a002_summary.json",
    "efficient_robotwin": ROOT / "artifacts/vla_wam_shared_v3/results/efficient_wam_rt_robotwin_phase_a_summary.json",
    "fast_robotwin": ROOT / "artifacts/vla_wam_shared_v3/results/fastwam_robotwin_phase_a_summary.json",
    "lingbot_robotwin": ROOT / "artifacts/vla_wam_shared_v3/results/lingbot_va_robotwin_phase_a_summary.json",
    "bounded_checkpoint_table": ROOT / "artifacts/vla_wam_shared_v2/results/direct_command_cross_model_comparison.json",
    "coverage": ROOT / "artifacts/vla_wam_shared_v3/measurement_coverage_audit.json",
    "nano_mirror": ROOT / "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/nano_v3b001_summary.json",
    "pi05_mirror": ROOT / "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/results/pi05_v3b002_report.json",
    "dream_mirror": ROOT / "artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003/results/dreamzero_v3b003_summary.json",
    "dose": ROOT / "artifacts/vla_wam_shared_v3/phase_b/nano_lateral_sweep_v3b005/results/nano_v3b005_dose_response_report.json",
    "failure": ROOT / "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/analysis/failure_mode_split_report.json",
    "competence": ROOT / "artifacts/vla_wam_shared_v3/analysis/mechanism/gap_vs_competence_report.json",
    "provenance": ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/checkpoint_provenance/checkpoint_provenance_table.json",
    "robotwin_mirror": ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/results/v3b007/fastwam_v3b007_summary.json",
    "start_side": ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/results/v3b008/v3b008_summary.json",
    "role_swap": ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/results/v3b009/v3b009_summary.json",
    "stochastic_repeats": ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/results/v3d001/pi05_v3d001_summary.json",
    "base_rotation_gate": ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/gates/v3b006/model_blind_structural_gate_failure.json",
}

REQUIRED_MANIFESTS = (
    ROOT / "artifacts/vla_wam_shared_v3/analysis/paper_figures/figure1_nano_instrument_sensitivity.manifest.json",
    ROOT / "artifacts/vla_wam_shared_v3/analysis/paper_figures/figure2_three_checkpoint_position_reflection.manifest.json",
    ROOT / "artifacts/vla_wam_shared_v3/phase_b/nano_lateral_sweep_v3b005/results/figures/manifest.json",
    ROOT / "artifacts/vla_wam_shared_v3/analysis/mechanism/figures/mechanism_figure_manifest.json",
    ROOT / "artifacts/vla_wam_shared_v3/analysis/paper_figures/figure5_cross_arena_directional_success.manifest.json",
    ROOT / "artifacts/vla_wam_shared_v3/phase_c/four_phrasings_v3c001/results/figures/phase_c_figure_manifest.json",
    ROOT / "artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003/results/evidence_manifest.json",
    ROOT / "artifacts/vla_wam_shared_v3/phase_b/nano_lateral_sweep_v3b005/results/nano_v3b005_results_manifest.json",
    ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/results/v3b007/evidence_hash_manifest.json",
    ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/results/v3b008/evidence_manifest.json",
    ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/results/v3b009/evidence_manifest.json",
    ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/results/v3d001/evidence_manifest.json",
)

PHASE_C_DIR = ROOT / "artifacts/vla_wam_shared_v3/phase_c/four_phrasings_v3c001/results"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required closed evidence is missing: {path}")
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}


def _verify_manifest_records(value: Any, *, manifest: Path, verified: list[Path]) -> None:
    """Verify every repository-local path/bytes/SHA record exposed by a manifest."""

    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            declared = Path(str(value["path"]))
            if declared.is_absolute():
                candidate = manifest.parent / declared.name
                if not candidate.is_file():
                    candidate = None
            else:
                candidate = ROOT / declared
                if not candidate.is_file():
                    candidate = manifest.parent / declared
            if candidate is not None:
                if not candidate.is_file():
                    raise FileNotFoundError(f"manifest-bound file is missing: {declared} ({manifest})")
                if candidate.stat().st_size != int(value["bytes"]) or sha256(candidate) != value["sha256"]:
                    raise ValueError(f"manifest-bound file changed: {declared} ({manifest})")
                verified.append(candidate.resolve())
        for name, nested in value.items():
            if (
                isinstance(nested, dict)
                and "path" not in nested
                and {"bytes", "sha256"}.issubset(nested)
            ):
                candidate = manifest.parent / str(name)
                if candidate.is_file():
                    if candidate.stat().st_size != int(nested["bytes"]) or sha256(candidate) != nested["sha256"]:
                        raise ValueError(f"manifest-bound file changed: {candidate} ({manifest})")
                    verified.append(candidate.resolve())
            _verify_manifest_records(nested, manifest=manifest, verified=verified)
    elif isinstance(value, list):
        for nested in value:
            _verify_manifest_records(nested, manifest=manifest, verified=verified)


def verify_required_manifests() -> list[Path]:
    verified: list[Path] = []
    for manifest in REQUIRED_MANIFESTS:
        if not manifest.is_file():
            raise FileNotFoundError(f"required evidence manifest is missing: {manifest}")
        _verify_manifest_records(read_json(manifest), manifest=manifest, verified=verified)
        verified.append(manifest.resolve())
    return sorted(set(verified))


def validate_closed_evidence(data: dict[str, Any], phase_c: list[dict[str, Any]]) -> None:
    for key in ("pi05_phase_a", "groot_phase_a", "edge_phase_a", "nano_phase_a", "dream_phase_a"):
        summary = data[key]
        if len(summary.get("pairs", [])) != 27:
            raise ValueError(f"{key} is not the closed 27-pair expanded DROID cohort")
    for key in ("efficient_robotwin", "fast_robotwin", "lingbot_robotwin"):
        primary = data[key].get("v3_primary_results", {})
        if primary.get("valid_episodes") != 126 or primary.get("matched_pairs") != 63:
            raise ValueError(f"{key} is not the closed 126-episode RoboTwin cohort")
    if data["robotwin_mirror"].get("status") != "complete_27_matched_seeds_108_valid_episodes":
        raise ValueError("V3-B007 is not closed")
    for key, expected_episodes, expected_pairs in (("start_side", 162, 81), ("role_swap", 108, 54)):
        population = data[key].get("population", {})
        if population.get("behavioral_episode_count") != expected_episodes or population.get("matched_left_right_pair_count") != expected_pairs:
            raise ValueError(f"{key} is not closed")
    stochastic = data["stochastic_repeats"]
    if stochastic.get("status") != "complete_exact_release_analyzed" or stochastic.get("population", {}).get("valid_behavioral_episodes") != 432:
        raise ValueError("V3-D001 is not closed")
    if len(phase_c) != 3 or any(summary.get("status") != "complete_20_seed_160_behavioral_episode_result" for summary in phase_c):
        raise ValueError("Phase C is not closed for all three checkpoints")


def register_fonts() -> None:
    candidates = {
        "Georgia": "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "Georgia-Bold": "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
        "Arial": "/System/Library/Fonts/Supplemental/Arial.ttf",
        "Arial-Bold": "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "Arial-Italic": "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
    }
    fallbacks = {
        "Georgia": "Times-Roman",
        "Georgia-Bold": "Times-Bold",
        "Arial": "Helvetica",
        "Arial-Bold": "Helvetica-Bold",
        "Arial-Italic": "Helvetica-Oblique",
    }
    for name, path in candidates.items():
        if Path(path).is_file():
            pdfmetrics.registerFont(TTFont(name, path))
        elif name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(pdfmetrics.Font(name, fallbacks[name], "WinAnsiEncoding"))


BODY = ParagraphStyle("body", fontName="Arial", fontSize=9.5, leading=13.6, textColor=INK, spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.8, leading=10.8, textColor=MUTED)
NOTE = ParagraphStyle("note", parent=BODY, fontSize=8.4, leading=12.0, textColor=INK)
H1 = ParagraphStyle("h1", fontName="Georgia-Bold", fontSize=23, leading=26, textColor=INK)
H2 = ParagraphStyle("h2", fontName="Georgia-Bold", fontSize=15.5, leading=19, textColor=INK)
LABEL = ParagraphStyle("label", fontName="Arial-Bold", fontSize=7.2, leading=9, textColor=MUTED)
TABLE_TEXT = ParagraphStyle("table_text", parent=BODY, fontSize=6.6, leading=8.2, textColor=INK)
TABLE_BOLD = ParagraphStyle("table_bold", parent=TABLE_TEXT, fontName="Arial-Bold")
TABLE_HEAD = ParagraphStyle("table_head", parent=TABLE_BOLD, textColor=colors.white)


def para(c: canvas.Canvas, text: str, x: float, top: float, width: float, style: ParagraphStyle = BODY, max_height: float = 1000) -> float:
    block = Paragraph(text, style)
    _, height = block.wrap(width, max_height)
    block.drawOn(c, x, top - height)
    return top - height


def rounded_card(c: canvas.Canvas, x: float, y: float, w: float, h: float, fill: colors.Color = CARD) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(LINE)
    c.setLineWidth(0.7)
    c.roundRect(x, y, w, h, 10, fill=1, stroke=1)


def fit_image(c: canvas.Canvas, path: Path, x: float, y: float, w: float, h: float) -> None:
    with Image.open(path) as image:
        iw, ih = image.size
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    c.drawImage(ImageReader(str(path)), x + (w - dw) / 2, y + (h - dh) / 2, width=dw, height=dh, preserveAspectRatio=True, mask="auto")


class Report:
    def __init__(self, output: Path):
        output.parent.mkdir(parents=True, exist_ok=True)
        self.canvas = canvas.Canvas(
            str(output),
            pagesize=(PAGE_W, PAGE_H),
            pageCompression=1,
            invariant=1,
        )
        self.canvas.setTitle("Language Sensitivity Is Not Directional Control")
        self.canvas.setSubject("Matched spatial-language interventions in robot policies")
        self.canvas.setAuthor("Steerable VLA/WAM study")
        self.canvas.setCreator("tools/build_v3_scientific_report.py")
        self.page = 0

    def new_page(self, title: str, kicker: str, subtitle: str = "") -> float:
        if self.page:
            self.canvas.showPage()
        self.page += 1
        c = self.canvas
        c.setFillColor(BG)
        c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        c.setFillColor(MUTED)
        c.setFont("Arial-Bold", 7.2)
        c.drawString(MARGIN, PAGE_H - 27, kicker.upper())
        top = para(c, title, MARGIN, PAGE_H - 40, PAGE_W - 2 * MARGIN, H1)
        if subtitle:
            top = para(c, subtitle, MARGIN, top - 4, PAGE_W - 2 * MARGIN, SMALL)
        c.setStrokeColor(LINE)
        c.line(MARGIN, top - 10, PAGE_W - MARGIN, top - 10)
        c.setFillColor(MUTED)
        c.setFont("Arial", 6.8)
        c.drawRightString(PAGE_W - MARGIN, 17, f"{self.page}")
        return top - 22

    def finish(self) -> None:
        self.canvas.save()


def prompt_line() -> str:
    return "<b>LEFT condition:</b> \"Put the Rubik's cube to the left of the bowl.\"<br/><b>RIGHT condition:</b> \"Put the Rubik's cube to the right of the bowl.\""


def phase_c_prompt_lines() -> str:
    return (
        "<b>Direct:</b> \"Put the Rubik's cube to the {left|right} of the bowl.\"<br/>"
        "<b>Shortened:</b> \"Put the cube {left|right} of the bowl.\"<br/>"
        "<b>Goal:</b> \"The Rubik's cube should end up to the {left|right} of the bowl.\"<br/>"
        "<b>Contrastive:</b> \"Put the Rubik's cube to the {left|right} of the bowl, not to the {right|left} of the bowl.\""
    )


def draw_cover(report: Report, data: dict[str, Any]) -> None:
    y = report.new_page(
        "Language sensitivity is not directional control",
        "Scientific report · VLA/WAM language steerability",
        "Matched language interventions reveal a geometry-dependent competence asymmetry in robot policies.",
    )
    c = report.canvas
    left_w, gap = 485, 22
    right_x = MARGIN + left_w + gap
    right_w = PAGE_W - MARGIN - right_x
    rounded_card(c, MARGIN, 77, left_w, y - 88)
    top = y - 18
    top = para(c, "<b>Research question</b>", MARGIN + 18, top, left_w - 36, H2)
    top = para(c, "Does changing only a static spatial instruction reliably redirect a robot policy, and does that response imply competent task completion?", MARGIN + 18, top - 7, left_w - 36, BODY)
    top = para(c, prompt_line(), MARGIN + 18, top - 11, left_w - 36, NOTE)
    top = para(c, "<b>Answer</b>", MARGIN + 18, top - 17, left_w - 36, H2)
    top = para(
        c,
        "Executed actions and endpoints often changed with the instruction, but binary task success remained direction-, layout-, and checkpoint-dependent; an exploratory block also showed descriptive variation across four prompt forms. Position reflection and a seven-level lateral sweep show that scene geometry changes the directional advantage. The effect therefore cannot be summarized as either language control or a fixed model bias.",
        MARGIN + 18,
        top - 7,
        left_w - 36,
        BODY,
    )
    rounded_card(c, right_x, 77, right_w, y - 88)
    audit = data["coverage"]["nano_phase_a_margin_sensitivity_reproduction"]
    dose = data["dose"]["primary_depth_dose_response"]["slope_m_per_m"]
    failure = {row["model_id"]: row for row in data["failure"]["results"]}
    competence = data["competence"]["descriptive_associations"]["competence_vs_signed_gap"]["spearman"]
    dream_p = failure["dreamzero_droid_action_cfg"]["failure_only_exact_test"]["p_value"]
    bullets = [
        (f"{audit['right_minus_left_mean_margin_gap_m']*100:.1f} cm", f"Nano RIGHT-minus-LEFT requested-depth gap; exact p={audit['exact_two_sided_sign_test_p_excluding_ties']:.3g}."),
        (f"{dose['mean']:+.2f} m/m", f"Dose-response slope across reference-object position; 95% CI [{dose['ci95'][0]:.2f}, {dose['ci95'][1]:.2f}]."),
        (f"p = {dream_p:.3g}", "DreamZero’s failure composition differs by requested direction; a rate-only difficulty account is insufficient there."),
        (f"ρ = {competence['coefficient']:.2f}", f"No monotonic success-gap association is detected across five DROID checkpoints; exact p={competence['two_sided_exact_permutation_p']:.2f}."),
    ]
    top = y - 18
    for value, text_value in bullets:
        c.setFillColor(TEAL)
        c.setFont("Georgia-Bold", 17)
        c.drawString(right_x + 17, top - 14, value)
        top = para(c, text_value, right_x + 17, top - 22, right_w - 34, SMALL) - 16
    para(c, "DROID/RoboLab and RoboTwin use different tasks and success predicates. They are reported separately and never pooled.", right_x + 17, 112, right_w - 34, SMALL)


def draw_figure_page(
    report: Report,
    *,
    number: int,
    title: str,
    subtitle: str,
    finding: str,
    interpretation: str,
    boundary: str,
    prompt: bool = False,
    prompt_text: str | None = None,
) -> None:
    top = report.new_page(title, f"Figure {number}", subtitle)
    c = report.canvas
    figure_x, figure_w = MARGIN, 574
    side_x = figure_x + figure_w + 17
    side_w = PAGE_W - MARGIN - side_x
    bottom = 48
    figure_h = top - bottom
    rounded_card(c, figure_x, bottom, figure_w, figure_h)
    fit_image(c, FIGURES[number], figure_x + 8, bottom + 8, figure_w - 16, figure_h - 16)
    rounded_card(c, side_x, bottom, side_w, figure_h)
    y = top - 17
    if prompt or prompt_text:
        y = para(c, prompt_text or prompt_line(), side_x + 14, y, side_w - 28, SMALL) - 12
    for label, text_value, color in (
        ("FINDING", finding, TEAL),
        ("INTERPRETATION", interpretation, PURPLE),
        ("CLAIM BOUNDARY", boundary, MUTED),
    ):
        c.setFillColor(color)
        c.setFont("Arial-Bold", 7.1)
        c.drawString(side_x + 14, y, label)
        y = para(c, text_value, side_x + 14, y - 7, side_w - 28, NOTE) - 13


def draw_phase_c_page(report: Report, phase_c: list[dict[str, Any]]) -> None:
    finding, interpretation = phase_c_findings(phase_c)
    top = report.new_page(
        "Directional outcomes vary across four prompt forms",
        "Figure 7",
        "Exploratory: four frozen static prompt forms are crossed with direction at 20 matched seeds on three checkpoints.",
    )
    c = report.canvas
    figure_bottom = 178
    rounded_card(c, MARGIN, figure_bottom, PAGE_W - 2 * MARGIN, top - figure_bottom)
    fit_image(
        c,
        FIGURES[7],
        MARGIN + 8,
        figure_bottom + 8,
        PAGE_W - 2 * MARGIN - 16,
        top - figure_bottom - 16,
    )
    gap = 12
    prompt_w = 468
    card_h = 116
    rounded_card(c, MARGIN, 49, prompt_w, card_h)
    c.setFillColor(PURPLE)
    c.setFont("Arial-Bold", 7.1)
    c.drawString(MARGIN + 14, 147, "THE FOUR STATIC PROMPT TEMPLATES")
    para(c, phase_c_prompt_lines(), MARGIN + 14, 137, prompt_w - 28, SMALL)
    right_x = MARGIN + prompt_w + gap
    right_w = PAGE_W - MARGIN - right_x
    rounded_card(c, right_x, 49, right_w, card_h)
    c.setFillColor(TEAL)
    c.setFont("Arial-Bold", 7.1)
    c.drawString(right_x + 14, 147, "READING")
    y = para(c, f"<b>Finding:</b> {finding}", right_x + 14, 137, right_w - 28, SMALL) - 3
    y = para(c, "<b>Interpretation:</b> Outcomes differ descriptively across prompt forms; this is compatible with a language contribution layered over geometry, not a confirmatory phrasing effect.", right_x + 14, y, right_w - 28, SMALL) - 3
    para(c, "<b>Boundary:</b> The same 20 seeds are reused; redirection and completion remain separate.", right_x + 14, y, right_w - 28, SMALL)


def _dream_reflection_text(dream: dict[str, Any]) -> tuple[str, str]:
    primary = dream["full_sample_primary"]
    depth = primary["I_position_reflection_interaction"]
    binary = primary["binary_success_DiD"]
    cells = binary["cell_success_table_2x2"]
    return (
        f"DreamZero depth interaction {depth['mean_m']*100:+.1f} cm; exact sign-test p={depth['paired_sign_test']['p_value']:.3g}.",
        f"Control L/R {cells['control']['left']['successes']}/27 and {cells['control']['right']['successes']}/27; reflected L/R {cells['position_mirrored']['left']['successes']}/27 and {cells['position_mirrored']['right']['successes']}/27.",
    )


def phase_c_findings(summaries: list[dict[str, Any]]) -> tuple[str, str]:
    labels = {
        "groot_n17_droid_vla": "GR00T",
        "cosmos3_edge_policy_droid": "Edge",
        "cosmos3_nano_policy_droid": "Nano",
    }
    family_names = {
        "direct_command": "direct",
        "short_command": "shortened",
        "goal_as_outcome": "goal",
        "desired_plus_negated_opposite": "contrastive",
    }
    gaps = []
    for summary in summaries:
        for family in family_names:
            left = summary["success_by_condition"][f"{family}:left"]["successes"]
            right = summary["success_by_condition"][f"{family}:right"]["successes"]
            gaps.append((abs(right - left), right - left, labels[summary["model_id"]], family_names[family], left, right))
    _, signed, label, family, left, right = max(gaps)
    finding = f"The largest descriptive direction gap is {label} under the {family} prompt: LEFT {left}/20 versus RIGHT {right}/20 ({signed:+d} episodes)."
    interpretation = "Observed success and endpoint-response patterns differ across the four prompt forms. This exploratory result is compatible with a language contribution layered over geometry, but it is not a registered confirmatory phrasing effect."
    return finding, interpretation


def _taxonomy(raw: dict[str, Any]) -> dict[str, int]:
    return {key: int(raw.get(key, 0)) for key in ("correct", "pick_failed", "transport_failed", "wrong_side", "release_failed")}


def _droid_table_row(summary: dict[str, Any], *, label: str, family: str, future: str) -> dict[str, Any]:
    left, right = summary["directional"]["left"], summary["directional"]["right"]
    pairs = len(summary["pairs"])
    shift = summary.get("right_minus_left_endpoint_shift_m") or summary["right_minus_left_raw_robot_y_shift_m"]
    action = summary["action_rms_common_prefix"]
    distinct = pairs if action.get("all_pairs_actions_differ") or (action.get("observed_count") == pairs and action.get("minimum", 0) > 0) else 0
    return {
        "label": label,
        "family": family,
        "n": int(left.get("valid_denominator", left.get("trials"))) + int(right.get("valid_denominator", right.get("trials"))),
        "left": f"{left['successes']}/{left.get('valid_denominator', left.get('trials'))}",
        "right": f"{right['successes']}/{right.get('valid_denominator', right.get('trials'))}",
        "aligned": f"{summary['endpoint_ordering_counts']['aligned']}/{pairs}",
        "actions": f"{distinct}/{pairs}",
        "shift": f"{-100*shift['mean']:+.1f} / {-100*shift['median']:+.1f} cm",
        "taxonomy": _taxonomy(summary["overall_failure_taxonomy_counts"]),
        "future": future,
    }


def _pi0_table_row(summary: dict[str, Any]) -> dict[str, Any]:
    shift = summary["right_minus_left_endpoint_shift_m"]
    return {
        "label": "π0-FAST public compatibility",
        "family": "VLA",
        "n": summary["behavioral_episodes"],
        "left": f"{summary['success_by_direction']['left']['successes']}/{summary['success_by_direction']['left']['episodes']}",
        "right": f"{summary['success_by_direction']['right']['successes']}/{summary['success_by_direction']['right']['episodes']}",
        "aligned": f"{summary['endpoint_ordering']['aligned']}/{summary['behavioral_matched_pairs']}",
        "actions": f"{summary['distinct_executed_action_pairs']['count']}/{summary['behavioral_matched_pairs']}",
        "shift": f"{-100*shift['mean']:+.1f} / {-100*shift['median']:+.1f} cm",
        "taxonomy": _taxonomy(summary["failure_taxonomy"]),
        "future": "action-only",
    }


def _robotwin_table_row(summary: dict[str, Any], *, label: str, future: str) -> dict[str, Any]:
    primary = summary["v3_primary_results"]
    left, right = primary["by_direction"]["left"], primary["by_direction"]["right"]
    endpoint, action = primary["paired_endpoint_response"], primary["paired_action_response"]
    shift = endpoint["left_minus_right_shift_summary_m"]
    return {
        "label": label,
        "family": "WAM",
        "n": primary["valid_episodes"],
        "left": f"{left['successes']}/{left['valid_episodes']}",
        "right": f"{right['successes']}/{right['valid_episodes']}",
        "aligned": f"{endpoint['aligned']}/{primary['matched_pairs']}",
        "actions": f"{action['distinct_trace_pairs']}/{primary['matched_pairs']}",
        "shift": f"{100*shift['mean']:+.1f} / {100*shift['median']:+.1f} cm",
        "taxonomy": _taxonomy(primary["failure_taxonomy_counts"]),
        "future": future,
    }


def _bounded_robotwin_row(comparison: dict[str, Any], model_id: str) -> dict[str, Any]:
    source = next(row for row in comparison["rows"] if row["model_id"] == model_id)
    pair_n = source["paired_endpoint_alignment"]["trials"]
    action = source["paired_action_distinctness"]
    action_text = "NR" if action["status"] != "measured" else f"{action['count']}/{action['trials']}"
    return {
        "label": "BOUNDED V2: " + source["model"],
        "family": source["model_class"],
        "n": source["valid_n"],
        "left": f"{source['left_success']['count']}/{source['left_success']['trials']}",
        "right": f"{source['right_success']['count']}/{source['right_success']['trials']}",
        "aligned": f"{source['paired_endpoint_alignment']['count']}/{pair_n}",
        "actions": action_text,
        "shift": "NR",
        "taxonomy": None,
        "future": source["future_interface"].replace("_", " "),
    }


def _draw_checkpoint_table(report: Report, *, title: str, kicker: str, subtitle: str, rows: list[dict[str, Any]], note: str) -> None:
    top = report.new_page(title, kicker, subtitle)
    c = report.canvas
    values: list[list[Any]] = [["Checkpoint / cohort", "Type", "Episodes", "LEFT", "RIGHT", "Aligned", "Actions", "Mean / median shift", "C/P/T/W/R", "Future interface"]]
    for row in rows:
        taxonomy = "not harmonized" if row["taxonomy"] is None else "/".join(str(row["taxonomy"][key]) for key in ("correct", "pick_failed", "transport_failed", "wrong_side", "release_failed"))
        values.append([row["label"], row["family"], row["n"], row["left"], row["right"], row["aligned"], row["actions"], row["shift"], taxonomy, row["future"]])
    wrapped = [
        [Paragraph(str(value), TABLE_HEAD if row_index == 0 else (TABLE_BOLD if column == 0 else TABLE_TEXT)) for column, value in enumerate(row)]
        for row_index, row in enumerate(values)
    ]
    widths = [128, 34, 45, 42, 42, 49, 49, 84, 91, 182]
    row_height = 51 if len(rows) >= 6 else 57
    table = Table(wrapped, colWidths=widths, rowHeights=[32] + [row_height] * len(rows))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (8, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.45, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [CARD, colors.HexColor("#F1EDE5")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    table_height = 32 + row_height * len(rows)
    table.wrapOn(c, sum(widths), table_height)
    table.drawOn(c, MARGIN, top - table_height)
    rounded_card(c, MARGIN, 49, PAGE_W - 2 * MARGIN, max(70, top - table_height - 62))
    para(c, note, MARGIN + 15, top - table_height - 17, PAGE_W - 2 * MARGIN - 30, SMALL)


def draw_result_tables(report: Report, data: dict[str, Any]) -> None:
    droid_rows = [
        _droid_table_row(data["pi05_phase_a"], label="π0.5 current stack", family="VLA", future="action-only"),
        _droid_table_row(data["groot_phase_a"], label="GR00T N1.7", family="VLA", future="action-only"),
        _droid_table_row(data["edge_phase_a"], label="Cosmos3 Edge", family="WAM", future="452 decoded local futures"),
        _droid_table_row(data["nano_phase_a"], label="Cosmos3 Nano", family="WAM", future="349 decoded local futures"),
        _droid_table_row(data["dream_phase_a"], label="DreamZero", family="WAM", future="54 full-reset decodes + latent futures"),
        _pi0_table_row(data["pi0_compatibility"]),
    ]
    _draw_checkpoint_table(
        report,
        title="Complete checkpoint table · DROID / RoboLab",
        kicker="11-checkpoint result table · 6 DROID identities",
        subtitle="Expanded direct-command cohorts and the separate π0-FAST compatibility cohort; no RoboTwin row is pooled here.",
        rows=droid_rows,
        note="<b>Reading:</b> shift is requested-oriented endpoint redirection; positive follows the requested LEFT→RIGHT ordering. C/P/T/W/R = correct, pick, transport, wrong-side, release. The older frozen π0-FAST 20-episode reference remains a separate historical cohort and is not merged with public-compatibility evidence.",
    )
    robotwin_rows = [
        _robotwin_table_row(data["efficient_robotwin"], label="Efficient-WAM-RT", future="126 decoded coarse futures"),
        _robotwin_table_row(data["fast_robotwin"], label="FastWAM", future="action-only at test time"),
        _robotwin_table_row(data["lingbot_robotwin"], label="LingBot-VA", future="126 latent-only futures"),
        _bounded_robotwin_row(data["bounded_checkpoint_table"], "lingbot_vla_4b_robotwin"),
        _bounded_robotwin_row(data["bounded_checkpoint_table"], "light_wam_robotwin"),
    ]
    _draw_checkpoint_table(
        report,
        title="Complete checkpoint table · RoboTwin",
        kicker="11-checkpoint result table · 5 RoboTwin identities",
        subtitle="Expanded 63-pair replicate cohorts and two bounded references; DROID success rules are not used here.",
        rows=robotwin_rows,
        note="<b>Reading:</b> shift is requested-oriented endpoint redirection; positive follows the requested LEFT→RIGHT ordering. C/P/T/W/R = correct, pick, transport, wrong-side, release. Efficient-WAM-RT's negative mean but positive median reflects a small reverse-tail of large shifts, not a transcription error. <b>Nesting:</b> each expanded WAM has 63 pairs from nine policy seeds within seven scenes. NR is never converted to zero.",
    )


def _ci_text(summary: dict[str, Any], *, scale: float = 1.0, digits: int = 2) -> str:
    interval = summary.get("mean_bootstrap_95", summary)
    return f"[{interval['lower'] * scale:.{digits}f}, {interval['upper'] * scale:.{digits}f}]"


def draw_tier_b_controls(report: Report, data: dict[str, Any]) -> None:
    """Summarize the registered controls that do not belong in Figures 1–7."""

    top = report.new_page(
        "Registered controls refine—and limit—the mechanism claim",
        "Supplementary Tier B evidence",
        "The controls below test stochasticity, scene factors, and cross-arena generality without pooling their estimands.",
    )
    c = report.canvas
    robotwin = data["robotwin_mirror"]
    start = data["start_side"]
    role = data["role_swap"]
    stochastic = data["stochastic_repeats"]

    robotwin_depth = robotwin["full_sample_primary"]["requested_side_depth_interaction"]
    start_outer = start["factor_analysis"]["pairwise_factor_interactions"][
        "target_start_right_minus_target_start_left"
    ]
    start_depth = start_outer["requested_side_depth_interaction_m"]
    start_binary = start_outer["binary_success_interaction"]
    start_cells = start["success_table"]
    role_effect = role["factor_analysis"]["pairwise_factor_interactions"][
        "bowl_target_cube_reference_minus_cube_target_bowl_reference"
    ]
    role_depth = role_effect["requested_side_depth_interaction_m"]
    role_binary = role_effect["binary_success_interaction"]
    stochastic_success = stochastic["success"]
    stochastic_gap = stochastic_success["directional_gap"]
    stochastic_left = stochastic_success["by_direction"]["left"]
    stochastic_right = stochastic_success["by_direction"]["right"]

    cards = [
        (
            "ROBOTTWIN POSITION REFLECTION",
            "FastWAM reverses under reflected movable-object positions",
            f"Requested-depth interaction {robotwin_depth['mean_m']:+.2f} m, 95% CI {_ci_text(robotwin_depth)}, exact sign p={robotwin_depth['paired_sign_test']['p_value']:.3g}. Binary success DiD {robotwin['binary_success_difference_in_differences']['mean']:+.2f}, exact p={robotwin['binary_success_difference_in_differences']['exact_permutation_test']['p_value']:.3g}.",
            "This is cross-arena replication of geometry dependence, not a pooled success estimate: RoboTwin uses a different controller, scenes, and success predicate.",
        ),
        (
            "TARGET START-SIDE",
            "The success advantage follows the target’s initial side",
            f"Target starts LEFT: L {start_cells['target_start_left']['left']['successes']}/27, R {start_cells['target_start_left']['right']['successes']}/27. Target starts RIGHT: L {start_cells['target_start_right']['left']['successes']}/27, R {start_cells['target_start_right']['right']['successes']}/27. The outer binary-gap interaction is {start_binary['mean']:+.3f} (exact p={start_binary['exact_permutation_test']['p_value']:.4f}). Depth changes by {start_depth['mean_m']:+.3f} m, 95% CI {_ci_text(start_depth, digits=3)}.",
            f"The binary interaction is clear; the continuous-depth mean CI excludes zero but its sign test is p={start_depth['paired_sign_test']['p_value']:.3f}, so sign consistency is not claimed. Start side changes geometry, reachability, and policy state together.",
        ),
        (
            "TARGET / REFERENCE ROLE SWAP",
            "Object roles alter the magnitude of directional steering",
            f"Swapping cube and bowl roles changes requested-depth contrast by {role_depth['mean_m']:+.2f} m (95% CI {_ci_text(role_depth)}; p={role_depth['paired_sign_test']['p_value']:.3g}). The binary interaction is {role_binary['mean']:+.2f}, exact p={role_binary['exact_permutation_test']['p_value']:.3g}.",
            "Continuous redirection changes clearly; the binary interaction narrowly misses 0.05. Object semantics and physical affordances change together, so this is not language-only evidence.",
        ),
        (
            "FIXED-SCENE STOCHASTIC REPEATS",
            "Repeated policy samples separate scene effects from policy noise",
            f"Across 27 fixed scenes × 8 policy samples per direction, LEFT succeeds {stochastic_left['successes']}/{stochastic_left['episodes']} and RIGHT {stochastic_right['successes']}/{stochastic_right['episodes']}. The seed-level mean p(RIGHT)−p(LEFT) is {stochastic_gap['mean']:+.2f} (95% cluster-bootstrap CI {_ci_text(stochastic_gap['environment_seed_cluster_bootstrap_95'])}).",
            "The environment seed is the inferential unit; 432 episodes are nested repeats, not 432 independent scenes. Wilson intervals over episodes are descriptive only.",
        ),
    ]

    gap = 12
    card_w = (PAGE_W - 2 * MARGIN - gap) / 2
    card_h = 177
    y_positions = (top - card_h, top - 2 * card_h - gap)
    for index, (label, heading, result, boundary) in enumerate(cards):
        row, column = divmod(index, 2)
        x = MARGIN + column * (card_w + gap)
        y = y_positions[row]
        rounded_card(c, x, y, card_w, card_h)
        c.setFillColor((TEAL, PURPLE, RIGHT, LEFT)[index])
        c.setFont("Arial-Bold", 7.2)
        c.drawString(x + 14, y + card_h - 20, label)
        text_top = para(c, heading, x + 14, y + card_h - 30, card_w - 28, H2) - 6
        text_top = para(c, result, x + 14, text_top, card_w - 28, NOTE) - 8
        para(c, f"<b>Boundary:</b> {boundary}", x + 14, text_top, card_w - 28, SMALL)

    para(
        c,
        "<b>Base-rotation control:</b> failed closed before behavioral inference because the wrist camera inherits the robot-base transform, violating the registered fixed-camera contract. No behavioral denominator or null result was created.",
        MARGIN,
        y_positions[1] - 13,
        PAGE_W - 2 * MARGIN,
        SMALL,
    )


def draw_methods(report: Report, data: dict[str, Any], sources: list[Path]) -> None:
    top = report.new_page(
        "Methods, inference, and claim boundary",
        "Reproducibility",
        "Frozen definitions make the expanded and intervention cohorts comparable without erasing their different scopes.",
    )
    c = report.canvas
    columns = [
        (
            "DESIGN",
            "Matched seed and identical reset within each language pair. Static episode prompt only. No oracle, subtask coach, prompt switching, or progress-conditioned language. Position reflection moves movable-object centers only; the lateral sweep moves the reference object across seven registered positions.",
        ),
        (
            "OUTCOMES",
            "For DROID/RoboLab, correct requires pickup, transport into the requested 45° cone, sustained entry, and detached release. RoboTwin retains its own frozen arena predicate. Continuous fields include signed lateral offset, requested-side depth, timing, and path diagnostics where exposed.",
        ),
        (
            "INFERENCE",
            "Marginal proportions use Wilson intervals. Continuous matched contrasts use 10,000–20,000 seed-level bootstrap resamples, as preregistered per cohort, plus exact two-sided sign tests. Binary reflection effects use exact within-seed layout-label permutation. Phase C is exploratory.",
        ),
    ]
    card_w = (PAGE_W - 2 * MARGIN - 24) / 3
    for index, (label, text_value) in enumerate(columns):
        x = MARGIN + index * (card_w + 12)
        rounded_card(c, x, top - 178, card_w, 178)
        c.setFillColor((TEAL, PURPLE, RIGHT)[index])
        c.setFont("Arial-Bold", 7.2)
        c.drawString(x + 14, top - 20, label)
        para(c, text_value, x + 14, top - 33, card_w - 28, NOTE)
    y = top - 210
    y = para(c, "<b>What the evidence supports</b>", MARGIN, y, 240, H2)
    para(c, "Language frequently perturbs executed behavior. Geometry can reverse the directional competence advantage. Outcomes also vary descriptively across prompt forms in the exploratory block. Sensitivity, requested endpoint ordering, and task completion are therefore distinct estimands.", MARGIN, y - 9, 240, BODY)
    x = MARGIN + 270
    y2 = top - 210
    y2 = para(c, "<b>What it does not identify</b>", x, y2, 240, H2)
    para(c, "The interventions do not identify training-distribution frequency, a unique causal source inside a checkpoint, or a universal VLA–WAM family effect. Robot-base rotation was structurally invalid because the wrist camera inherits the base transform.", x, y2 - 9, 240, BODY)
    x2 = MARGIN + 540
    y3 = top - 210
    y3 = para(c, "<b>Evidence handling</b>", x2, y3, PAGE_W - MARGIN - x2, H2)
    para(c, "Behavioral failures remain in denominators. Infrastructure-invalid and partial attempts are separate. Raw rollouts, videos, and checkpoints remain on the ali-owned PVC; compact JSONL, hashes, manifests, code, and selected media are committed.", x2, y3 - 9, PAGE_W - MARGIN - x2, BODY)
    c.setStrokeColor(LINE)
    c.line(MARGIN, 88, PAGE_W - MARGIN, 88)
    para(c, f"This PDF is generated from {len(sources)} hash-bound inputs. The checkpoint-provenance table contains {len(data['provenance'].get('checkpoints', data['provenance'].get('records', [])))} records; undisclosed training or caption exposure is marked not auditable rather than inferred.", MARGIN, 73, PAGE_W - 2 * MARGIN, SMALL)


def build(output: Path) -> tuple[Path, Path]:
    register_fonts()
    missing = [path for path in list(FIGURES.values()) + list(SOURCES.values()) + list(REQUIRED_MANIFESTS) if not path.is_file()]
    if missing:
        raise FileNotFoundError("report refuses partial evidence; missing:\n" + "\n".join(str(path) for path in missing))
    phase_c_summary_paths = sorted(PHASE_C_DIR.glob("*/**/*_phase_c_summary.json")) + sorted(PHASE_C_DIR.glob("*_phase_c_summary.json"))
    phase_c_summary_paths = sorted(set(phase_c_summary_paths))
    if len(phase_c_summary_paths) != 3:
        raise FileNotFoundError(f"report requires exactly three complete Phase-C summaries, found {len(phase_c_summary_paths)}")
    data = {name: read_json(path) for name, path in SOURCES.items()}
    phase_c = [read_json(path) for path in phase_c_summary_paths]
    manifest_inputs = verify_required_manifests()
    validate_closed_evidence(data, phase_c)

    report = Report(output)
    draw_cover(report, data)
    audit = data["coverage"]["nano_phase_a_margin_sensitivity_reproduction"]
    draw_figure_page(
        report,
        number=1,
        title="Binary completion can conceal a continuous placement asymmetry",
        subtitle="Four measurements on Cosmos3 Nano distinguish a near-ceiling task result from a directional quality gap.",
        finding=f"LEFT and RIGHT task success is nearly balanced, yet RIGHT placement is {audit['right_minus_left_mean_margin_gap_m']*100:.1f} cm deeper on average across 27 matched seeds (exact p={audit['exact_two_sided_sign_test_p_excluding_ties']:.3g}).",
        interpretation="Requested-side depth is the sensitive instrument at binary ceiling. It measures how deeply the object enters the requested region and remains defined on failures.",
        boundary="The figure establishes a checkpoint-specific directional quality gap, not its cause. Position reflection and the dose-response sweep test geometry directly.",
        prompt=True,
    )
    dream_stat, dream_cells = _dream_reflection_text(data["dream_mirror"])
    draw_figure_page(
        report,
        number=2,
        title="Position reflection shifts the directional depth advantage",
        subtitle="The same 27 seeds and four cells are repeated on Nano, π0.5, and DreamZero.",
        finding=f"Nano and π0.5 show strongly negative depth interactions after reflecting movable-object positions. {dream_stat}",
        interpretation=f"{dream_cells} Across three checkpoints, the directional competence gap is layout-dependent rather than a fixed property of the words alone.",
        boundary="The robot base, cameras, prompts, and non-movable geometry are fixed. The experiment does not separate embodiment handedness or identify training distribution.",
        prompt=True,
    )
    slope = data["dose"]["primary_depth_dose_response"]["slope_m_per_m"]
    draw_figure_page(
        report,
        number=3,
        title="Seven reference positions reveal a positive dose-response",
        subtitle="A seven-level sweep converts the reflection result from a binary reversal into a dose-response relationship.",
        finding=f"The requested-depth gap changes by {slope['mean']:.2f} m per meter of bowl displacement (95% bootstrap CI [{slope['ci95'][0]:.2f}, {slope['ci95'][1]:.2f}]); 13/15 seed-level slopes are positive, exact p={slope['sign_test']['two_sided_p']:.3g}.",
        interpretation="Across the seven sampled positions, relocating the reference object and requested regions changes the depth contrast. Binary completion is less sensitive because Nano is close to ceiling.",
        boundary="The fitted zero crossing lies outside the registered support, so no in-support crossing is claimed. The curve is descriptive over the seven tested positions.",
        prompt=True,
    )
    comp = data["competence"]["descriptive_associations"]
    draw_figure_page(
        report,
        number=4,
        title="No monotonic gap–competence relation is detected",
        subtitle="Five separate 54-episode DROID cohorts test the proposed difficulty × competence explanation descriptively.",
        finding=f"There is no monotonic association with the signed gap (Spearman ρ={comp['competence_vs_signed_gap']['spearman']['coefficient']:.2f}, exact p={comp['competence_vs_signed_gap']['spearman']['two_sided_exact_permutation_p']:.2f}) or its magnitude (ρ={comp['competence_vs_absolute_gap']['spearman']['coefficient']:.2f}, p={comp['competence_vs_absolute_gap']['spearman']['two_sided_exact_permutation_p']:.2f}).",
        interpretation="Floor and ceiling mechanically limit the largest observable binary gap. Across these five fixed checkpoints, the observed gaps are non-monotonic; this does not identify their mechanism.",
        boundary="Five checkpoints are too few for a population-level model. The figure is a descriptive constraint, not evidence that competence is irrelevant.",
    )
    draw_figure_page(
        report,
        number=5,
        title="Directional success remains checkpoint- and arena-specific",
        subtitle="Success counts and Wilson intervals provide context for the matched continuous diagnostics.",
        finding="DROID checkpoints span near-floor to near-ceiling performance and include both positive and negative directional gaps. RoboTwin likewise varies across the three WAMs.",
        interpretation="These data do not support a family-level ranking. The meaningful unit is a checkpoint, arena, prompt, and frozen success predicate.",
        boundary="DROID/RoboLab and RoboTwin have different tasks and success rules. Their rates are faceted and never pooled; Wilson intervals are marginal, not paired tests.",
        prompt_text=(
            '<b>DROID direct prompt:</b> "Put the Rubik\'s cube to the {left|right} of the bowl."<br/>'
            '<b>RoboTwin template:</b> "Put the &lt;movable object&gt; to the {left|right} of the &lt;reference object&gt;." '
            'The registry fixes all seven scene-specific noun pairs.'
        ),
    )
    failure_rows = {row["model_id"]: row for row in data["failure"]["results"]}
    draw_figure_page(
        report,
        number=6,
        title="Difficulty alone does not explain every checkpoint’s failures",
        subtitle="Failure-only contingency tests ask whether LEFT and RIGHT differ only in rate or also in failure composition.",
        finding=f"π0.5 and Edge show no detected shape difference (p={failure_rows['pi05_current_stack_droid']['failure_only_exact_test']['p_value']:.3g} and p={failure_rows['cosmos3_edge_policy_droid']['failure_only_exact_test']['p_value']:.3g}), but DreamZero does (p={failure_rows['dreamzero_droid_action_cfg']['failure_only_exact_test']['p_value']:.3g}).",
        interpretation="A same-shape/different-rate difficulty account remains compatible with π0.5 and Edge but is insufficient for DreamZero, whose LEFT failures are transport-heavy while RIGHT failures are pick failures.",
        boundary="Nondetection is not equivalence; the smaller direction-specific failure rows are sparse. The exact tests compare marginal failure composition, not a paired transition model.",
        prompt=True,
    )
    draw_phase_c_page(report, phase_c)
    draw_result_tables(report, data)
    draw_tier_b_controls(report, data)
    source_paths = sorted(set(list(SOURCES.values()) + list(FIGURES.values()) + phase_c_summary_paths + manifest_inputs))
    draw_methods(report, data, source_paths)
    report.finish()
    manifest = {
        "schema_version": "vla-wam-shared-v3-scientific-pdf-manifest-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "status": "complete_figure_led_scientific_report",
        "page_count": report.page,
        "output": file_record(output),
        "inputs": [file_record(path) for path in source_paths],
        "builder": file_record(Path(__file__).resolve()),
        "claim_boundary": "DROID/RoboLab and RoboTwin remain separate; futures are never scored as executions; partial and infrastructure-invalid attempts are excluded from behavioral denominators.",
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, allow_nan=False, sort_keys=True, indent=2) + "\n")
    return output, manifest_path


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
