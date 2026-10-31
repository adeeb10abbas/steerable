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
    run("H06 retained graded symmetry", [sys.executable, str(package / "analysis/analyze_h06_graded_symmetry.py")])
    run("H07 H08 retained endpoint symmetry", [sys.executable, str(package / "analysis/analyze_h07_h08_endpoint_symmetry.py")])
    run("H09 H10 retained wording control", [sys.executable, str(package / "analysis/analyze_h09_h10_wording_control.py")])
    run("H11 H12 retained Phase A direct commands", [sys.executable, str(package / "analysis/analyze_h11_h12_phase_a_direct.py")])
    run("H13 retained V1 evaluator corpus", [sys.executable, str(package / "analysis/analyze_h13_v1_forecast_corpus.py")])
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
    if len(reader.pages) != 4 or problems:
        raise RuntimeError(f"PDF check failed: pages={len(reader.pages)}; layout/citation flags={problems}")
    raw_body = "\n".join(page.extract_text() for page in reader.pages)
    # pdfTeX may wrap a word as ``in-\\nference`` even though the rendered word
    # is continuous. Normalize line-end hyphenation and whitespace before
    # asserting the required scientific text.
    body = " ".join(raw_body.replace("-\n", "").split())
    for required in ["108", "13/54", "50/54", "26/27", "DreamZero-s2", "31.1", "35.8",
                     "18/27", "17/27", "23/27", "81.4", "104 entries", "1.125", "13/15",
                     "H07/H08", "Edge", "12.4", "19.2", "endpoint interactions",
                     "H09/H10", "134/160", "123/160", "wording inference was not registered",
                     "H11/H12 are separate fixed-scene Phase-A direct-command cohorts",
                     "Nano 51/54; DreamZero-s2 3/27 LEFT and 17/27 RIGHT",
                     "2,554 retained futures are latents, not per-request RGB decodes",
                     "H13 evaluator-development audit", "3,008 cached forecast frames",
                     "20 matched blocks", "421/752", "331 abstained", "25/97", "413/421",
                     "396/421", "413/752", "744/752", "A3–A9",
                     "annotation/confirmation-gate", "new-inference claim",
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
    h06_path = package / "paper/data/h06_graded_symmetry.json"
    h06 = json.loads(h06_path.read_text())
    h07_h08_path = package / "paper/data/h07_h08_endpoint_symmetry.json"
    h07_h08 = json.loads(h07_h08_path.read_text())
    h09_h10_path = package / "paper/data/h09_h10_wording_control.json"
    h09_h10 = json.loads(h09_h10_path.read_text())
    h11_h12_path = package / "paper/data/h11_h12_phase_a_direct.json"
    h11_h12 = json.loads(h11_h12_path.read_text())
    h13_path = package / "paper/data/h13_v1_forecast_corpus.json"
    h13 = json.loads(h13_path.read_text())
    expected_h09_h10 = {
        "H09": ("cosmos3_nano_policy_droid", 160, 134),
        "H10": ("cosmos3_edge_policy_droid", 160, 123),
    }
    if h09_h10["reproduction"] != {
        "complete_seed_prompt_direction_grids": True,
        "failure_counts_and_taxonomies_exactly_reconciled": True,
        "full_registered_cells_strictly_parsed": 480,
        "h09_h10_compact_episode_rows_strictly_parsed": 320,
        "h09_h10_registered_cells_rejoined": 320,
        "matched_seed_contrast_rows": 120,
        "model_episode_rows": {"H09": 160, "H10": 160},
        "model_request_counts_exactly_reconciled": True,
        "registered_failure_taxonomies_exactly_reproduced": True,
        "registered_relation_endpoint_diagnostics_exactly_reproduced": True,
        "registered_success_counts_and_wilson_intervals_exactly_reproduced": True,
        "requested_margin_endpoint_algebraic_dependency_validated": True,
        "same_seed_direct_control_contrasts_recomputed": 6,
    }:
        raise RuntimeError("H09/H10 reproduction receipt changed")
    historical_counts = {"H09": (1234, 26), "H10": (1365, 37)}
    for cohort, (model, episodes, successes) in expected_h09_h10.items():
        observed = h09_h10["cohorts"][cohort]
        if observed["model_id"] != model:
            raise RuntimeError(f"{cohort} model identity changed")
        if observed["authenticated_population"] != {
            "behavioral_episodes": episodes,
            "directions": 2,
            "infrastructure_attempts_included": False,
            "matched_seed_blocks": 20,
            "missing_value_imputation": "none",
            "model_requests": historical_counts[cohort][0],
            "prompt_forms": 4,
            "valid_behavioral_failures": historical_counts[cohort][1],
            "valid_behavioral_failures_included": True,
        }:
            raise RuntimeError(f"{cohort} authenticated population changed")
        actual_successes = sum(
            form["both_directions"]["successes"]
            for form in observed["condition_summaries"].values()
        )
        if actual_successes != successes:
            raise RuntimeError(f"{cohort} success total changed")
    if len(h09_h10["matched_seed_contrasts"]) != 120:
        raise RuntimeError("H09/H10 contrast-row count changed")
    if any(h09_h10["new_activity"].values()):
        raise RuntimeError("H09/H10 receipt records new activity")
    if any(value is not False for value in h09_h10["gate_state"].values()):
        raise RuntimeError("H09/H10 receipt advances a release gate")
    h09_h10_boundary = h09_h10["claim_boundary"]
    if (
        h09_h10_boundary["wording_inference"] != "not_registered"
        or h09_h10_boundary["confirmatory_wording_claim"] is not False
        or h09_h10_boundary["changes_232_cell_core_bound"] is not False
        or h09_h10_boundary["models_pooled"] is not False
    ):
        raise RuntimeError("H09/H10 claim boundary changed")
    if "/data/users/" in json.dumps(h09_h10):
        raise RuntimeError("H09/H10 canonical output leaked a raw source path")
    if h11_h12["source_manifest"] != {
        "authenticated_bytes": 5_654_967,
        "authenticated_file_count": 10,
        "authenticated_lines": 101_058,
        "compact_git_files_byte_rehashed_now": True,
        "design_and_overlap_helper_bytes": 18_793,
        "design_and_overlap_helper_file_count": 2,
        "design_and_overlap_helper_lines": 383,
        "exported_source_paths_are_git_object_paths_only": True,
        "h11_h12_payload_bytes": 5_636_174,
        "h11_h12_payload_file_count": 8,
        "h11_h12_payload_lines": 100_675,
        "prospective_freeze_ancestry_verified": True,
        "raw_pvc_files_revalidated_now": False,
        "raw_source_paths_exported": False,
    }:
        raise RuntimeError("H11/H12 source authentication receipt changed")
    if h11_h12["reproduction"] != {
        "directional_successes_wilson_intervals_and_failure_taxonomies_exactly_reproduced": True,
        "dream_request_latent_and_official_decode_counts_exactly_reconciled": True,
        "full_phase_a_registry_rows_strictly_parsed": 780,
        "h11_h12_identity_overlap_with_h06_h07_h08_canonical_core": 0,
        "h11_h12_matched_pairs_reconstructed": 54,
        "h11_h12_registered_cells_rejoined": 108,
        "h11_h12_summary_episode_rows_strictly_parsed": 108,
        "historical_v2_baseline_cells_double_counted": 0,
        "infrastructure_invalid_attempts_verified_prebehavior_and_excluded": True,
        "matched_discordance_and_endpoint_ordering_exactly_reproduced": True,
        "model_request_counts_exactly_reconciled": True,
        "models_pooled": False,
        "nano_decoded_future_and_action_chunk_counts_exactly_reconciled": True,
        "preserved_v2_candidate_cells_excluded": 12,
    }:
        raise RuntimeError("H11/H12 reproduction receipt changed")
    if h11_h12["provenance"]["commit"] != "ce561e66f82e95055e39d3d7711691982f6b2086":
        raise RuntimeError("H11/H12 pinned source commit changed")
    if h11_h12["provenance"]["result_commits"] != {
        "H11": "150a362701656a3992c9738f497f1fd09605dc54",
        "H12": "8651aef704f6369b5b514d17ba7fd4bd0913b8d3",
    } or h11_h12["provenance"]["result_commits_are_ancestors_of_pinned_evidence"] is not True:
        raise RuntimeError("H11/H12 result-commit ancestry changed")
    design = h11_h12["design"]
    if (
        design["matched_environment_seed_labels"] != list(range(8303, 8330))
        or design["preserved_v2_seed_labels_excluded"] != [8300, 8301, 8302]
        or design["h06_h07_h08_canonical_core_seed_labels"] != list(range(9400, 9427))
        or design["h11_h12_overlap_with_h06_h07_h08_canonical_core"] != []
    ):
        raise RuntimeError("H11/H12 population identity boundary changed")
    expected_h11_h12 = {
        "H11": ("cosmos3_nano_policy_droid", {"left": 26, "right": 25}, 349, 3, 2),
        "H12": ("dreamzero_droid_action_cfg", {"left": 3, "right": 17}, 2_554, 34, 4),
    }
    for cohort, (model, successes, requests, failures, infra) in expected_h11_h12.items():
        observed = h11_h12["cohorts"][cohort]
        population = observed["authenticated_population"]
        if observed["model_id"] != model or population != {
            "behavioral_episodes": 54,
            "historical_v2_baseline_episodes_counted": 0,
            "infrastructure_invalid_attempts_excluded": infra,
            "matched_pairs": 27,
            "matched_seed_labels": 27,
            "missing_value_imputation": "none",
            "model_requests": requests,
            "preserved_v2_candidate_episodes_excluded": 6,
            "valid_behavioral_failures": failures,
            "valid_behavioral_failures_included": True,
        }:
            raise RuntimeError(f"{cohort} authenticated population changed")
        if {
            relation: observed["directional"][relation]["successes"]
            for relation in ("left", "right")
        } != successes:
            raise RuntimeError(f"{cohort} directional successes changed")
    nano_future = h11_h12["cohorts"]["H11"]["future_accounting"]
    dream_future = h11_h12["cohorts"]["H12"]["future_accounting"]
    if (
        nano_future["model_requests"] != 349
        or nano_future["decoded_rgb_future_records"] != 349
        or nano_future["returned_action_chunks"] != 349
        or dream_future["model_requests"] != 2_554
        or dream_future["retained_latent_future_records"] != 2_554
        or dream_future["official_full_reset_decodes"] != 54
        or dream_future["actual_rollout_videos"] != 54
        or dream_future["decoded_rgb_future_records"] != 0
        or dream_future["latent_records_are_per_request_decoded_rgb"] is not False
    ):
        raise RuntimeError("H11/H12 future-interface accounting changed")
    if len(h11_h12["matched_seed_pairs"]) != 54:
        raise RuntimeError("H11/H12 pair-row count changed")
    if len(h11_h12["new_activity"]) != 13 or any(h11_h12["new_activity"].values()):
        raise RuntimeError("H11/H12 receipt records new activity")
    if len(h11_h12["gate_state"]) != 9 or any(
        value is not False for value in h11_h12["gate_state"].values()
    ):
        raise RuntimeError("H11/H12 receipt advances a release gate")
    h11_h12_boundary = h11_h12["claim_boundary"]
    for flag in (
        "new_model_inference", "new_robot_episodes", "forecast_accuracy_evidence",
        "independent_scene_claim", "annotation_evidence", "equivalence_claim",
        "confirmation_test_or_stage_gate", "models_pooled",
        "historical_v2_baselines_double_counted", "changes_232_cell_core_bound",
    ):
        if h11_h12_boundary[flag] is not False:
            raise RuntimeError(f"H11/H12 claim boundary changed: {flag}")
    if h11_h12_boundary["maximum_core_behavioral_cells"] != 232:
        raise RuntimeError("H11/H12 core-cell ceiling changed")
    serialized_h11_h12 = json.dumps(h11_h12)
    if "/data/users/" in serialized_h11_h12 or "gate_root" in serialized_h11_h12:
        raise RuntimeError("H11/H12 canonical output leaked a raw source path")
    if h13["provenance"]["upstream_commit"] != "ce561e66f82e95055e39d3d7711691982f6b2086":
        raise RuntimeError("H13 pinned source commit changed")
    if h13["provenance"]["audit_commit"] != "4eac280b73be3fcaf6fa4ab12697a153a485b422":
        raise RuntimeError("H13 audit commit changed")
    if h13["provenance"]["bundle_files"] != 5 or h13["provenance"]["bundle_bytes"] != 998_249:
        raise RuntimeError("H13 compact bundle changed")
    h13_sources = h13["provenance"]["upstream_sources"]
    if (
        h13_sources["files"] != 761
        or h13_sources["bytes"] != 7_230_812
        or h13_sources["cache_files"] != 752
        or h13_sources["cache_bytes"] != 5_269_817
        or h13_sources["manifest_sha256"] != "ce7da897256d7583365f876e9697c08639e754dd30951bf476259ce7915b9490"
    ):
        raise RuntimeError("H13 upstream source authentication changed")
    h13_cohort = h13["population"]["cohort"]
    if h13_cohort != {
        "episodes": 80,
        "chunks": 752,
        "frames": 3_008,
        "reliable_frames": 1_460,
        "unreliable_frames": 1_548,
        "frame_rejection_reasons_overlapping": {
            "bowl_cross_camera_disagreement": 338,
            "camera_relation_disagreement": 1_201,
            "cube_cross_camera_disagreement": 710,
            "missing_localization": 1,
        },
    }:
        raise RuntimeError("H13 cohort accounting changed")
    h13_legacy = h13["legacy_replay"]
    if (
        h13_legacy["certain"] != 421
        or h13_legacy["abstained"] != 331
        or h13_legacy["covered_executed_positive"] != 25
        or h13_legacy["executed_positive"] != 97
        or h13_legacy["same_covered_subset_arithmetic"]["observed_agreement_numerator"] != 413
        or h13_legacy["same_covered_subset_arithmetic"]["constant_negative_agreement_numerator"] != 396
        or h13_legacy["unresolved_case_full_corpus_agreement_bounds"]["lower_numerator"] != 413
        or h13_legacy["unresolved_case_full_corpus_agreement_bounds"]["upper_numerator"] != 744
    ):
        raise RuntimeError("H13 legacy evaluator arithmetic changed")
    h13_sampling = h13["annotation_sampling"]
    if (
        h13_sampling["sample_size"] != 160
        or h13_sampling["draw_sha256"] != "ab780ea4e1b91918ca72801ef0641b563725d0a7d935f6de15e1d135b0ed6d47"
        or h13_sampling["media_recovered_and_verified"] != 0
        or h13_sampling["alignment_verified"] != 0
        or h13_sampling["human_labels_collected"] != 0
        or h13_sampling["gate_advanced"] is not False
    ):
        raise RuntimeError("H13 annotation-package boundary changed")
    h13_boundary = h13["claim_boundary"]
    for key in (
        "new_model_inference", "forecast_fidelity_or_accuracy_estimated",
        "annotation_or_release_gate_advanced", "core_232_cell_accounting_changed",
    ):
        if h13_boundary[key] is not False:
            raise RuntimeError(f"H13 claim boundary changed: {key}")
    if h13_boundary["new_behavioral_episodes"] != 0 or h13_boundary["new_human_labels"] != 0:
        raise RuntimeError("H13 receipt records new scientific activity")
    if h13_boundary["maximum_core_behavioral_cells"] != 232:
        raise RuntimeError("H13 core-cell ceiling changed")
    if len(h13["new_activity"]) != 13 or any(h13["new_activity"].values()):
        raise RuntimeError("H13 receipt records new activity")
    if len(h13["gate_state"]) != 9 or any(
        value is not False for value in h13["gate_state"].values()
    ):
        raise RuntimeError("H13 receipt advances a release gate")
    if h13["corrected_fidelity_readiness"]["ready"] is not False:
        raise RuntimeError("H13 unexpectedly claims corrected fidelity readiness")
    serialized_h13 = json.dumps(h13)
    if "/home/" in serialized_h13 or "source_id" in serialized_h13:
        raise RuntimeError("H13 canonical output leaked a private source path or draw identity")
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
        "retained_h06_graded_symmetry": {
            "scope": h06["claim_boundary"],
            "source_commit": h06["provenance"]["commit"],
            "design": h06["design"],
            "population": h06["authenticated_historical_population"],
            "primary_depth_dose_response": h06["primary_depth_dose_response"],
            "binary_success_secondary": h06["binary_success_secondary"],
            "by_level": h06["by_level"],
            "source_manifest": h06["source_manifest"],
            "reproduction": h06["reproduction"],
            "new_activity": h06["new_activity"],
            "gate_state": h06["gate_state"],
            "json_sha256": digest(h06_path),
            "csv_sha256": digest(package / "paper/data/h06_graded_symmetry.csv"),
            "analyzer_sha256": digest(package / "analysis/analyze_h06_graded_symmetry.py"),
        },
        "retained_h07_h08_endpoint_symmetry": {
            "scope": h07_h08["claim_boundary"],
            "source_commit": h07_h08["provenance"]["commit"],
            "design": h07_h08["design"],
            "cohorts": h07_h08["cohorts"],
            "source_manifest": h07_h08["source_manifest"],
            "reproduction": h07_h08["reproduction"],
            "new_activity": h07_h08["new_activity"],
            "gate_state": h07_h08["gate_state"],
            "json_sha256": digest(h07_h08_path),
            "csv_sha256": digest(package / "paper/data/h07_h08_endpoint_symmetry.csv"),
            "analyzer_sha256": digest(package / "analysis/analyze_h07_h08_endpoint_symmetry.py"),
        },
        "retained_h09_h10_wording_control": {
            "scope": h09_h10["claim_boundary"],
            "source_commit": h09_h10["provenance"]["commit"],
            "design": h09_h10["design"],
            "cohorts": h09_h10["cohorts"],
            "matched_seed_contrast_rows": len(h09_h10["matched_seed_contrasts"]),
            "source_manifest": h09_h10["source_manifest"],
            "reproduction": h09_h10["reproduction"],
            "new_activity": h09_h10["new_activity"],
            "gate_state": h09_h10["gate_state"],
            "json_sha256": digest(h09_h10_path),
            "csv_sha256": digest(package / "paper/data/h09_h10_wording_control.csv"),
            "analyzer_sha256": digest(package / "analysis/analyze_h09_h10_wording_control.py"),
        },
        "retained_h11_h12_phase_a": {
            "scope": h11_h12["claim_boundary"],
            "source_commit": h11_h12["provenance"]["commit"],
            "design": h11_h12["design"],
            "cohorts": h11_h12["cohorts"],
            "matched_seed_pair_rows": len(h11_h12["matched_seed_pairs"]),
            "source_manifest": h11_h12["source_manifest"],
            "reproduction": h11_h12["reproduction"],
            "new_activity": h11_h12["new_activity"],
            "gate_state": h11_h12["gate_state"],
            "json_sha256": digest(h11_h12_path),
            "csv_sha256": digest(package / "paper/data/h11_h12_phase_a_direct.csv"),
            "analyzer_sha256": digest(package / "analysis/analyze_h11_h12_phase_a_direct.py"),
        },
        "retained_h13_v1_evaluator_corpus": {
            "scope": h13["claim_boundary"],
            "upstream_commit": h13["provenance"]["upstream_commit"],
            "audit_commit": h13["provenance"]["audit_commit"],
            "source_authentication": h13["provenance"]["upstream_sources"],
            "population": h13["population"],
            "legacy_replay": h13["legacy_replay"],
            "endpoint_sensitivity": h13["endpoint_sensitivity"],
            "annotation_sampling": h13["annotation_sampling"],
            "corrected_fidelity_readiness": h13["corrected_fidelity_readiness"],
            "new_activity": h13["new_activity"],
            "gate_state": h13["gate_state"],
            "json_sha256": digest(h13_path),
            "csv_sha256": digest(package / "paper/data/h13_v1_forecast_corpus.csv"),
            "analyzer_sha256": digest(package / "analysis/analyze_h13_v1_forecast_corpus.py"),
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
