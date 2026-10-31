"""Reproduce the executed-behavior paper and preserved supporting analyses.

Requires requirements-build.txt and a TeX distribution providing latexmk.
No model inference, network calls, remote jobs, or publication actions.
Visual inspection of the final rendered pages is a separate human/agent check.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from pypdf import PdfReader


def main():
    package = Path(__file__).resolve().parents[1]
    repo = package.parents[1]
    results = package / "results"
    steps = []
    logs = []

    def run(name, command, cwd=repo):
        outcome = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
        logs.append(f"[{name}]\n{outcome.stdout}{outcome.stderr}\n")
        results.mkdir(parents=True, exist_ok=True)
        (results / "validation.log").write_text("\n".join(logs))
        if outcome.returncode:
            raise RuntimeError(f"{name} failed; see results/validation.log")
        steps.append({"step": name, "exit_code": outcome.returncode})
        print(f"Passed: {name}", flush=True)

    run("regression tests", [sys.executable, "-m", "unittest", "discover", "-s", str(package / "tests"), "-v"])
    run("stopping-time controls", [sys.executable, str(package / "analysis/analyze_stopping_controls.py")])
    run("representative H01 media", [sys.executable, str(package / "analysis/package_representative_media.py")])
    run("paper experiment results", [sys.executable, str(package / "analysis/extract_paper_results.py")])
    run("historical evidence replay", [sys.executable, str(package / "analysis/evidence_audit.py")])
    run("private annotation draw", [sys.executable, str(package / "analysis/prepare_annotation_sample.py")])
    run("paper figures", [sys.executable, str(package / "analysis/make_paper_figures.py")])
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise RuntimeError("latexmk is required; add your TeX distribution to PATH")
    run("official-template PDF", [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], package / "paper")
    pdf = package / "paper/main.pdf"
    reader = PdfReader(pdf)
    latex_log = (package / "paper/main.log").read_text()
    forbidden = ["Overfull", "undefined citations", "undefined references", "LaTeX Error"]
    problems = [marker for marker in forbidden if marker in latex_log]
    if len(reader.pages) > 4 or not reader.pages or problems:
        raise RuntimeError(f"PDF check failed: pages={len(reader.pages)}; layout/citation flags={problems}")
    body = "\n".join(page.extract_text() for page in reader.pages)
    for required in ["108", "13/54", "50/54", "26/27", "DreamZero-s2", "31.1", "35.8",
                     "18/27", "17/27", "Results", "Conclusion", "References"]:
        if required not in body:
            raise RuntimeError(f"PDF is missing expected text: {required}")
    if "??" in body:
        raise RuntimeError("Unresolved reference marker in PDF")
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    stopping_path = results / "stopping_controls.json"
    stopping = json.loads(stopping_path.read_text())
    representative_media_path = package / "paper/media/representative_h01_videos.json"
    representative_media = json.loads(representative_media_path.read_text())
    stopping_source_path = "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/nano_v3b001_episodes.jsonl"
    stopping_source = stopping["provenance"]["files"][stopping_source_path]
    expected_stopping_source_sha256 = "ce67aecb2705394e8161fa4b65df7130e62648e2f6d8b6a93408c8da685a7207"
    if stopping_source["sha256"] != expected_stopping_source_sha256:
        raise RuntimeError("Stopping-control input JSONL identity changed")
    report = {
        "status": "working_draft_built_not_submitted",
        "paper_scope": "Executed spatial behavior: Nano and custom DreamZero-s2 position-reflection experiments, reported separately",
        "paper_cohorts": ["nano_reflection", "dreamzero_reflection"],
        "steps": steps,
        "python": platform.python_version(),
        "build_dependencies": {name: importlib.metadata.version(name) for name in ["matplotlib", "pypdf"]},
        "pdf": {"path": "paper/main.pdf", "pages_including_references": len(reader.pages),
                "sha256": digest(pdf), "undefined_references_or_overfull_boxes": False,
                "visual_review": "separate inspection required; see docs/DELIVERY_QA.md"},
        "official_style_sha256": digest(package / "paper/template/corl_2026.sty"),
        "figure_input_sha256": digest(results / "paper_results.json"),
        "annotation_draw_sha256": json.loads((results / "annotation_sample.json").read_text())["provenance"]["draw_sha256"],
        "stopping_control": {
            "scope": stopping["definitions"]["status"],
            "source_commit": stopping["provenance"]["commit"],
            "source_jsonl": {"path": stopping_source_path, **stopping_source},
            "json_sha256": digest(stopping_path),
            "csv_sha256": digest(results / "stopping_controls.csv"),
            "analyzer_sha256": digest(package / "analysis/analyze_stopping_controls.py"),
        },
        "representative_media": {
            "scope": representative_media["claim_boundary"],
            "selection": representative_media["selection"],
            "video_count": representative_media["totals"]["video_count"],
            "video_bytes": representative_media["totals"]["video_bytes"],
            "source_manifest_and_video_file_count": representative_media["totals"][
                "source_manifest_and_video_file_count"
            ],
            "source_manifest_and_video_bytes": representative_media["totals"][
                "source_manifest_and_video_bytes"
            ],
            "manifest_sha256": digest(representative_media_path),
            "packager_sha256": digest(package / "analysis/package_representative_media.py"),
        },
        "corrected_fidelity_measured": False,
        "independent_human_labels_collected": False,
    }
    (results / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
