"""Reproduce the executed-behavior paper and preserved supporting analyses.

Requires the qualified Torch environment, requirements-build.txt and a TeX
distribution providing latexmk.
No model inference, network calls, remote jobs, or publication actions.
Visual inspection of the final rendered pages is a separate human/agent check.
"""
import hashlib
import importlib.metadata
import json
import os
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
    reproducible_output_environment = os.environ.copy()
    reproducible_output_environment.update({
        "SOURCE_DATE_EPOCH": "1786718341",
        "FORCE_SOURCE_DATE": "1",
    })
    required_build_dependencies = ["matplotlib", "numpy", "pypdf", "jsonschema", "torch"]
    try:
        build_dependency_versions = {
            name: importlib.metadata.version(name) for name in required_build_dependencies
        }
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "Build environment is incomplete; follow the Torch-inheriting README recipe"
        ) from error

    def run(name, command, cwd=repo, env=None):
        outcome = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
        logs.append(f"[{name}]\n{outcome.stdout}{outcome.stderr}\n")
        results.mkdir(parents=True, exist_ok=True)
        (results / "validation.log").write_text("\n".join(logs))
        if outcome.returncode:
            raise RuntimeError(f"{name} failed; see results/validation.log")
        steps.append({"step": name, "exit_code": outcome.returncode})
        print(f"Passed: {name}", flush=True)

    run("regression tests", [sys.executable, "-m", "unittest", "discover", "-s", str(package / "tests"), "-v"])
    run("stopping-time controls", [sys.executable, str(package / "analysis/analyze_stopping_controls.py")])
    run("H01 requested-region entry", [sys.executable, str(package / "analysis/analyze_h01_requested_region_entry.py")])
    run("H05 retained lateral dose response", [sys.executable, str(package / "analysis/analyze_h05_lateral_dose_response.py")])
    run("representative H01 media", [sys.executable, str(package / "analysis/package_representative_media.py")])
    run("representative H01 media verification", [sys.executable, str(package / "analysis/verify_representative_media.py")])
    run("paper experiment results", [sys.executable, str(package / "analysis/extract_paper_results.py")])
    run("historical evidence replay", [sys.executable, str(package / "analysis/evidence_audit.py")])
    run("private annotation draw", [sys.executable, str(package / "analysis/prepare_annotation_sample.py")])
    run("paper figures", [sys.executable, str(package / "analysis/make_paper_figures.py")],
        env=reproducible_output_environment)
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise RuntimeError("latexmk is required; add your TeX distribution to PATH")
    run("official-template PDF", [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        package / "paper", env=reproducible_output_environment)
    pdf = package / "paper/main.pdf"
    reader = PdfReader(pdf)
    latex_log = (package / "paper/main.log").read_text()
    forbidden = ["Overfull", "undefined citations", "undefined references", "LaTeX Error"]
    problems = [marker for marker in forbidden if marker in latex_log]
    if len(reader.pages) > 4 or not reader.pages or problems:
        raise RuntimeError(f"PDF check failed: pages={len(reader.pages)}; layout/citation flags={problems}")
    body = "\n".join(page.extract_text() for page in reader.pages)
    for required in ["108", "13/54", "50/54", "26/27", "DreamZero-s2", "31.1", "35.8",
                     "18/27", "17/27", "23/27", "81.4", "104 entries", "1.125", "13/15",
                     "Results", "Conclusion", "References"]:
        if required not in body:
            raise RuntimeError(f"PDF is missing expected text: {required}")
    if "??" in body:
        raise RuntimeError("Unresolved reference marker in PDF")
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    stopping_path = results / "stopping_controls.json"
    stopping = json.loads(stopping_path.read_text())
    entry_path = package / "paper/data/h01_requested_region_entry.json"
    entry = json.loads(entry_path.read_text())
    h05_path = package / "paper/data/h05_lateral_dose_response.json"
    h05 = json.loads(h05_path.read_text())
    representative_media_path = package / "paper/media/representative_h01_videos.json"
    representative_media = json.loads(representative_media_path.read_text())
    representative_verification_path = package / "paper/media/representative_h01_video_verification.json"
    representative_verification = json.loads(representative_verification_path.read_text())
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
        "build_dependencies": build_dependency_versions,
        "reproducible_output_environment": {
            key: reproducible_output_environment[key]
            for key in ("SOURCE_DATE_EPOCH", "FORCE_SOURCE_DATE")
        },
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
        "requested_region_entry": {
            "scope": entry["claim_boundary"],
            "source_commit": entry["provenance"]["commit"],
            "source_jsonl": {
                "path": stopping_source_path,
                **entry["provenance"]["files"][stopping_source_path],
            },
            "cohort": entry["cohort"],
            "condition_summaries": {
                name: {
                    key: value[key]
                    for key in (
                        "episodes", "entry_events", "cap_right_censored_without_entry",
                        "entry_kind_counts", "restricted_waiting_actions",
                    )
                }
                for name, value in entry["conditions"].items()
            },
            "matched_interaction": entry["matched_analysis"]["difference_in_differences_actions"],
            "json_sha256": digest(entry_path),
            "csv_sha256": digest(package / "paper/data/h01_requested_region_entry.csv"),
            "analyzer_sha256": digest(package / "analysis/analyze_h01_requested_region_entry.py"),
        },
        "retained_h05_lateral_dose_response": {
            "scope": h05["claim_boundary"],
            "source_commit": h05["provenance"]["commit"],
            "design": h05["design"],
            "population": h05["authenticated_historical_population"],
            "primary_depth_dose_response": h05["primary_depth_dose_response"],
            "binary_success_secondary": h05["binary_success_secondary"],
            "source_manifest": h05["source_manifest"],
            "historical_final_audit": h05["historical_final_audit"],
            "reproduction": h05["reproduction"],
            "new_activity": h05["new_activity"],
            "gate_state": h05["gate_state"],
            "json_sha256": digest(h05_path),
            "csv_sha256": digest(package / "paper/data/h05_lateral_dose_response.csv"),
            "analyzer_sha256": digest(package / "analysis/analyze_h05_lateral_dose_response.py"),
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
            "verification": {
                "scope": representative_verification["claim_boundary"],
                "totals": representative_verification["totals"],
                "zero_activity": representative_verification["zero_activity"],
                "receipt_sha256": digest(representative_verification_path),
                "verifier_sha256": digest(package / "analysis/verify_representative_media.py"),
            },
        },
        "corrected_fidelity_measured": False,
        "independent_human_labels_collected": False,
    }
    (results / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
