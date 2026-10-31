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
import re
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
    run("H01 request-window execution motion", [sys.executable, str(package / "analysis/analyze_h01_request_window_motion.py")])
    run("H02 retained position reflection", [sys.executable, str(package / "analysis/analyze_h02_position_reflection.py")])
    run("H02 matched entry supplement", [sys.executable, str(package / "analysis/analyze_h02_matched_entry.py")])
    run("H03 H04 retained guidance pilots", [sys.executable, str(package / "analysis/analyze_h03_h04_guidance_pilots.py")])
    run("H05 retained lateral dose response", [sys.executable, str(package / "analysis/analyze_h05_lateral_dose_response.py")])
    run("H06 retained graded symmetry", [sys.executable, str(package / "analysis/analyze_h06_graded_symmetry.py")])
    run("H07 H08 retained endpoint symmetry", [sys.executable, str(package / "analysis/analyze_h07_h08_endpoint_symmetry.py")])
    run("H09 H10 retained wording control", [sys.executable, str(package / "analysis/analyze_h09_h10_wording_control.py")])
    run("H11 H12 retained Phase A direct commands", [sys.executable, str(package / "analysis/analyze_h11_h12_phase_a_direct.py")])
    run("H13 retained V1 evaluator corpus", [sys.executable, str(package / "analysis/analyze_h13_v1_forecast_corpus.py")])
    run("representative H01 media", [sys.executable, str(package / "analysis/package_representative_media.py")])
    run("representative H01 media verification", [sys.executable, str(package / "analysis/verify_representative_media.py")])
    run("paper experiment results", [sys.executable, str(package / "analysis/extract_paper_results.py")])
    run("retained historical evidence coverage", [sys.executable, str(package / "analysis/audit_retained_historical_evidence_coverage.py")])
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
                     "H02 compact audit", "540 descriptors", "Post-result descriptive", "requested-cone-entry indicator 13/54 control versus 53/54 reflected", "42 absent by cap", "0.741", "238.3", "negative 27/27", "not event/imputation", "mixes incidence with entrant indices", "H03/H04", "post-result guidance pilots", "6/6 to 4/6", "3/6 to 4/6",
                     "H07/H08", "Edge", "12.4", "19.2", "endpoint interactions",
                     "H09/H10", "134/160", "123/160", "wording inference was not registered",
                     "H11/H12 are separate fixed-scene Phase-A direct-command cohorts",
                     "Nano 51/54; DreamZero-s2 3/27 LEFT and 17/27 RIGHT",
                     "2,554 retained futures are latents, not per-request RGB decodes",
                     "H13 evaluator-development audit", "3,008 cached forecast frames",
                     "20 matched blocks", "421/752", "331 abstained", "25/97", "413/421",
                     "396/421", "413/752", "744/752", "A3–A9",
                     "request-window audit", "632/738", "106 +32 targets",
                     "annotation/confirmation-gate", "new-inference claim",
                     "All 13 registered historical IDs have authenticated packages",
                     "not pooled inference or confirmation",
                     "Results", "Conclusion", "References"]:
        math_spacing_match = (
            required == "106 +32 targets"
            and re.search(r"106\s*\+32\s*targets", body) is not None
        )
        if required not in body and not math_spacing_match:
            raise RuntimeError(f"PDF is missing expected text: {required}")
    if "??" in body:
        raise RuntimeError("Unresolved reference marker in PDF")
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    stopping_path = results / "stopping_controls.json"
    stopping = json.loads(stopping_path.read_text())
    entry_path = package / "paper/data/h01_requested_region_entry.json"
    entry = json.loads(entry_path.read_text())
    request_window_path = package / "paper/data/h01_request_window_motion.json"
    request_window = json.loads(request_window_path.read_text())
    h02_path = package / "paper/data/h02_position_reflection.json"
    h02 = json.loads(h02_path.read_text())
    h02_matched_path = package / "paper/data/h02_matched_entry.json"
    h02_matched = json.loads(h02_matched_path.read_text())
    h03_h04_path = package / "paper/data/h03_h04_guidance_pilots.json"
    h03_h04 = json.loads(h03_h04_path.read_text())
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
    coverage_path = package / "paper/data/retained_historical_evidence_coverage.json"
    coverage = json.loads(coverage_path.read_text())
    if h02["source_provenance"]["commit"] != "ce561e66f82e95055e39d3d7711691982f6b2086":
        raise RuntimeError("H02 pinned source commit changed")
    if h02["source_provenance"]["authenticated_file_count"] != 9:
        raise RuntimeError("H02 authenticated file count changed")
    if h02["source_provenance"]["authenticated_bytes"] != 831_259:
        raise RuntimeError("H02 authenticated byte count changed")
    if h02["source_provenance"]["authenticated_lines"] != 1_429:
        raise RuntimeError("H02 authenticated line count changed")
    if h02["layout_totals"] != {
        "control": {"episodes": 54, "successes": 13, "failures": 41},
        "position_mirrored": {"episodes": 54, "successes": 50, "failures": 4},
    }:
        raise RuntimeError("H02 layout outcomes changed")
    if h02["declared_raw_artifact_inventory"]["distinct_path_sha_identities"] != 540:
        raise RuntimeError("H02 raw descriptor identity count changed")
    if h02["recorded_requested_region_entry"]["layouts"] != {
        "control": {"episodes": 54, "recorded_entry_events": 13, "no_entry_cap_censored": 41},
        "position_mirrored": {"episodes": 54, "recorded_entry_events": 53, "no_entry_cap_censored": 1},
    }:
        raise RuntimeError("H02 requested-region entry totals changed")
    if h02["recorded_requested_region_entry"]["totals"]["no_entry_cap_censored"] != 42:
        raise RuntimeError("H02 no-entry censoring count changed")
    if h02["recorded_requested_region_entry"]["totals"] != {
        "episodes": 108,
        "recorded_entry_events": 66,
        "no_entry_cap_censored": 42,
        "observed_sustained_entries": 63,
        "goal_terminated_entry_without_observed_three_sample_sustained_run": 3,
        "successes_with_entry": 63,
        "valid_failures_with_entry": 3,
    }:
        raise RuntimeError("H02 entry/censor classification changed")
    if len(h02["matched_entry_censor_blocks"]) != 27:
        raise RuntimeError("H02 matched entry/censor block count changed")
    if h02["schema_version"] != "wmf-retained-h02-position-reflection-v1":
        raise RuntimeError("sealed H02 package schema changed")
    if h02_matched["schema_version"] != "wmf-retained-h02-matched-entry-supplement-v1":
        raise RuntimeError("H02 matched-entry supplement schema changed")
    if h02_matched["population"]["matched_environment_label_blocks"] != 27:
        raise RuntimeError("H02 matched entry aggregate population changed")
    if h02_matched["summaries"]["entry_observed"][
        "block_balanced_position_mirrored_minus_control"
    ] != {
        "n": 27,
        "sum": 20.0,
        "mean": 20 / 27,
        "median": 1.0,
        "minimum": 0.0,
        "maximum": 1.0,
        "sign_counts": {"positive": 25, "zero": 2, "negative": 0},
    }:
        raise RuntimeError("H02 block-balanced recorded-entry indicator contrast changed")
    h02_waiting = h02_matched["summaries"]["cap_restricted_waiting_action_index"][
        "block_balanced_position_mirrored_minus_control"
    ]
    if (
        h02_waiting["sum"] != -6434.0
        or h02_waiting["mean"] != -6434 / 27
        or h02_waiting["median"] != -267.5
        or (h02_waiting["minimum"], h02_waiting["maximum"])
        != (-328.5, -37.5)
        or h02_waiting["sign_counts"]
        != {"positive": 0, "zero": 0, "negative": 27}
    ):
        raise RuntimeError("H02 block-balanced restricted-waiting contrast changed")
    h02_waiting_did = h02_matched["summaries"][
        "cap_restricted_waiting_action_index"
    ]["command_by_layout_difference_in_differences"]
    if (
        h02_waiting_did["sum"] != 662.0
        or h02_waiting_did["mean"] != 662 / 27
        or h02_waiting_did["sign_counts"]
        != {"positive": 19, "zero": 0, "negative": 8}
    ):
        raise RuntimeError("H02 restricted-waiting command interaction changed")
    if h02["audit_operations"] != {
        "compact_git_objects_byte_authenticated": 9,
        "remote_fetches": 0,
        "raw_pvc_files_opened": 0,
        "raw_pvc_bytes_rehashed": 0,
    }:
        raise RuntimeError("H02 audit-operation boundary changed")
    if h02["configuration"]["official_conditional_D1"] is not False:
        raise RuntimeError("H02 custom-s2/D1 boundary changed")
    if any(h02["new_activity"].values()) or any(h02["gate_state"].values()):
        raise RuntimeError("H02 retained audit advanced activity or a gate")
    if h03_h04["provenance"]["commit"] != "ce561e66f82e95055e39d3d7711691982f6b2086":
        raise RuntimeError("H03/H04 pinned source commit changed")
    if h03_h04["source_manifest"] != {
        "authenticated_bytes": 810_860,
        "authenticated_file_count": 7,
        "authenticated_lines": 18_939,
        "compact_git_files_byte_rehashed_now": True,
        "exported_source_paths_are_git_object_paths_only": True,
        "raw_pvc_files_revalidated_now": False,
        "raw_source_paths_exported": False,
    }:
        raise RuntimeError("H03/H04 source authentication receipt changed")
    expected_h03_h04 = {
        "H03": ("cosmos3_nano_policy_droid", "cosmos3_nano_policy_droid", 6, 4),
        "H04": ("dreamzero_droid", "dreamzero_droid_action_cfg", 3, 4),
    }
    expected_h03_h04_configurations = {
        "H03": ("joint_guidance_g3", "conditional_prediction_g1"),
        "H04": ("conditional_action_equivalent_s1", "negative_branch_action_guidance_s2"),
    }
    for cohort, (
        baseline_model,
        intervention_model,
        baseline_successes,
        intervention_successes,
    ) in expected_h03_h04.items():
        observed = h03_h04["cohorts"][cohort]
        if (
            observed["model_id"] != intervention_model
            or observed["baseline_model_id"] != baseline_model
            or observed["intervention_model_id"] != intervention_model
        ):
            raise RuntimeError(f"{cohort} model identity changed")
        if observed["authenticated_population"] != {
            "baseline_behavioral_episodes": 6,
            "intervention_behavioral_episodes": 6,
            "matched_configuration_cells": 6,
            "matched_seed_labels": 3,
            "missing_value_imputation": "none",
            "physical_reset_fingerprints": 1,
        }:
            raise RuntimeError(f"{cohort} authenticated population changed")
        if observed["successes"] != {
            "baseline": baseline_successes,
            "intervention": intervention_successes,
        }:
            raise RuntimeError(f"{cohort} success totals changed")
    if len(h03_h04["matched_cell_contrasts"]) != 12:
        raise RuntimeError("H03/H04 matched-cell count changed")
    for row in h03_h04["matched_cell_contrasts"]:
        baseline_model, intervention_model, _, _ = expected_h03_h04[row["cohort"]]
        if (
            row["baseline_model_id"] != baseline_model
            or row["intervention_model_id"] != intervention_model
            or row["model_id"] != intervention_model
        ):
            raise RuntimeError("H03/H04 matched-cell model identity changed")
    if len(h03_h04["matched_seed_diagnostics"]) != 6:
        raise RuntimeError("H03/H04 matched-seed diagnostic count changed")
    if h03_h04["reproduction"]["source_paired_seed_diagnostics_validated"] != 6:
        raise RuntimeError("H03/H04 paired-seed source validation receipt changed")
    for diagnostic in h03_h04["matched_seed_diagnostics"]:
        cohort = diagnostic["cohort"]
        seed = diagnostic["environment_seed_label"]
        baseline_model, intervention_model, _, _ = expected_h03_h04[cohort]
        baseline_configuration, intervention_configuration = expected_h03_h04_configurations[cohort]
        if (
            diagnostic["model_id"] != intervention_model
            or diagnostic["baseline_model_id"] != baseline_model
            or diagnostic["intervention_model_id"] != intervention_model
            or diagnostic["baseline_configuration_id"] != baseline_configuration
            or diagnostic["intervention_configuration_id"] != intervention_configuration
            or diagnostic["pair_id"] != f"droid_pair_seed_{seed}"
        ):
            raise RuntimeError("H03/H04 matched-seed diagnostic identity changed")
        seed_rows = {
            row["relation"]: row
            for row in h03_h04["matched_cell_contrasts"]
            if row["cohort"] == cohort and row["environment_seed"] == seed
        }
        if set(seed_rows) != {"left", "right"}:
            raise RuntimeError("H03/H04 matched-seed diagnostic join changed")
        baseline_separation = sum(
            seed_rows[relation]["baseline_requested_margin_m"] for relation in ("left", "right")
        )
        intervention_separation = sum(
            seed_rows[relation]["intervention_requested_margin_m"] for relation in ("left", "right")
        )
        baseline_gap = (
            seed_rows["right"]["baseline_requested_margin_m"]
            - seed_rows["left"]["baseline_requested_margin_m"]
        )
        intervention_gap = (
            seed_rows["right"]["intervention_requested_margin_m"]
            - seed_rows["left"]["intervention_requested_margin_m"]
        )
        if (
            diagnostic["baseline_endpoint_ordering"] != "aligned"
            or diagnostic["intervention_endpoint_ordering"] != "aligned"
        ):
            raise RuntimeError("H03/H04 paired endpoint ordering changed")
        expected_diagnostic_metrics = {
            "baseline_endpoint_separation_right_minus_left_m": baseline_separation,
            "intervention_endpoint_separation_right_minus_left_m": intervention_separation,
            "intervention_minus_baseline_endpoint_separation_m": (
                intervention_separation - baseline_separation
            ),
            "baseline_seed_balance_gap_right_minus_left_margin_m": baseline_gap,
            "intervention_seed_balance_gap_right_minus_left_margin_m": intervention_gap,
            "intervention_minus_baseline_seed_balance_gap_m": intervention_gap - baseline_gap,
        }
        for field, expected in expected_diagnostic_metrics.items():
            if abs(diagnostic[field] - expected) > 1e-12:
                raise RuntimeError(f"H03/H04 paired diagnostic metric changed: {field}")
    if len(h03_h04["new_activity"]) != 13 or any(h03_h04["new_activity"].values()):
        raise RuntimeError("H03/H04 receipt records new activity")
    if len(h03_h04["gate_state"]) != 9 or any(
        value is not False for value in h03_h04["gate_state"].values()
    ):
        raise RuntimeError("H03/H04 receipt advances a release gate")
    h03_h04_boundary = h03_h04["claim_boundary"]
    for flag in (
        "powered_or_general_performance_claim",
        "models_pooled",
        "new_confirmation_d2_extension_selected",
        "forecast_accuracy_claim",
        "changes_232_cell_core_bound",
    ):
        if h03_h04_boundary[flag] is not False:
            raise RuntimeError(f"H03/H04 claim boundary changed: {flag}")
    if h03_h04_boundary["historical_h04_custom_s2_retained"] is not True:
        raise RuntimeError("H03/H04 historical custom-s2 retention boundary changed")
    if "/data/users/" in json.dumps(h03_h04):
        raise RuntimeError("H03/H04 canonical output leaked a raw source path")
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
    if coverage["registry"]["registered_historical_ids"] != [
        f"H{number:02d}" for number in range(1, 14)
    ]:
        raise RuntimeError("retained historical ID registry coverage changed")
    if coverage["coverage_summary"] != {
        "H01_packages_with_equivalent_explicit_claim_boundary": 1,
        "additional_primary_or_presentation_roles_counted_as_new_cohorts": 0,
        "duplicate_mappings": 0,
        "grouped_analysis_packages": 9,
        "mapped_cohorts": 13,
        "packages_with_closed_claim_boundaries": 9,
        "packages_with_explicit_zero_activity_and_closed_gate_schema": 8,
        "packaging_coverage_complete": True,
        "registered_cohorts": 13,
        "scientific_confirmation_claims_created_by_this_audit": 0,
        "unmapped_registered_cohorts": 0,
    }:
        raise RuntimeError("retained historical coverage summary changed")
    if coverage["historical_custom_s2_ids_not_official_D1"] != [
        "H02", "H04", "H07", "H12",
    ]:
        raise RuntimeError("historical custom-s2 versus official D1 boundary changed")
    if any(coverage["new_activity"].values()) or any(
        value is not False for value in coverage["gate_state"].values()
    ):
        raise RuntimeError("retained historical coverage audit advanced activity or a gate")
    if any(coverage["preserved_evidence_gaps"].values()):
        raise RuntimeError("retained historical coverage audit closed an unsupported evidence gap")
    coverage_boundary = coverage["claim_boundary"]
    for field in (
        "packaging_coverage_is_scientific_confirmation", "scientific_outcomes_reestimated",
        "models_or_cohorts_pooled", "episode_counts_summed",
        "equivalence_established", "independent_scene_or_layout_generalization_established",
        "language_understanding_established",
        "forecast_accuracy_or_fidelity_estimated", "raw_or_replay_evidence_recovered",
        "new_scientific_evidence_created", "current_worker_or_queue_state_inferred",
        "changes_232_cell_core_bound",
    ):
        if coverage_boundary[field] is not False:
            raise RuntimeError(f"retained historical coverage boundary changed: {field}")
    if coverage_boundary["maximum_core_behavioral_cells"] != 232:
        raise RuntimeError("retained historical coverage changed the core-cell ceiling")
    if request_window["cohort"]["episodes"] != 108 or request_window["coverage"]["requests"] != 738:
        raise RuntimeError("H01 request-window retained population changed")
    if (
        request_window["coverage"]["complete_32_action_windows"],
        request_window["coverage"]["censored_windows"],
    ) != (632, 106):
        raise RuntimeError("H01 request-window coverage/censoring changed")
    if request_window["coverage"]["censored_windows_by_episode_outcome"] != {
        "goal_stopped_success": 100,
        "administrative_cap_censored_failure": 6,
    } or request_window["coverage"]["censored_observed_prefix_actions"]["sum"] != 1748:
        raise RuntimeError("H01 request-window outcome/prefix censoring changed")
    matched_windows = request_window["matched_complete_window_analysis"]
    if {
        layout: matched_windows["by_layout"][layout]["shared_complete_window_pairs"]
        for layout in ("control", "position_mirrored")
    } != {"control": 114, "position_mirrored": 113}:
        raise RuntimeError("H01 matched request-window pair inventory changed")
    if {
        layout: matched_windows["by_layout"][layout][
            "complete_window_records_excluded_without_exact_counterpart_endpoint"
        ]["total"]
        for layout in ("control", "position_mirrored")
    } != {"control": 101, "position_mirrored": 77}:
        raise RuntimeError("H01 matched request-window counterpart censoring changed")
    if {
        layout: {
            timing: matched_windows["by_layout"][layout]["excluded_record_timing"][timing]["total"]
            for timing in (
                "counterpart_request_terminated_before_32_action_endpoint",
                "owner_request_started_at_or_after_counterpart_episode_end",
            )
        }
        for layout in ("control", "position_mirrored")
    } != {
        "control": {
            "counterpart_request_terminated_before_32_action_endpoint": 22,
            "owner_request_started_at_or_after_counterpart_episode_end": 79,
        },
        "position_mirrored": {
            "counterpart_request_terminated_before_32_action_endpoint": 21,
            "owner_request_started_at_or_after_counterpart_episode_end": 56,
        },
    }:
        raise RuntimeError("H01 matched request-window exclusion timing changed")
    if {
        layout: matched_windows["by_layout"][layout][
            "equal_seed_mean_signed_lateral_contrast_m_per_32_action_window"
        ]["sign_counts"]
        for layout in ("control", "position_mirrored")
    } != {
        "control": {"positive": 26, "negative": 0, "exact_zero": 1},
        "position_mirrored": {"positive": 27, "negative": 0, "exact_zero": 0},
    }:
        raise RuntimeError("H01 matched request-window equal-seed signs changed")
    same_support = matched_windows["four_condition_same_request_index_support"]
    if (
        same_support["shared_complete_window_pairs_per_layout"] != 96
        or same_support[
            "position_mirrored_minus_control_mean_contrast_m_per_32_action_window"
        ]["sign_counts"] != {"positive": 15, "negative": 11, "exact_zero": 1}
    ):
        raise RuntimeError("H01 four-condition shared-support audit changed")
    fixed_profile = matched_windows["fixed_request_index_profile"]
    pair_outcome_strata = [
        "success_left__success_right",
        "success_left__cap_failure_right",
        "cap_failure_left__success_right",
        "cap_failure_left__cap_failure_right",
    ]
    joint_pair_outcome_strata = [
        f"control={control}|position_mirrored={reflected}"
        for control in pair_outcome_strata
        for reflected in pair_outcome_strata
    ]
    if (
        request_window["schema_version"]
        != "wmf-retained-h01-request-window-motion-v5"
        or fixed_profile["pair_outcome_strata"] != pair_outcome_strata
        or fixed_profile["joint_pair_outcome_strata"] != joint_pair_outcome_strata
    ):
        raise RuntimeError("H01 fixed-index outcome-attrition schema changed")
    if fixed_profile["layout_observable_seed_labels"] != {
        "control": [27, 27, 26, 18, 8, 5, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        "position_mirrored": [27, 27, 27, 17, 5, 4, 2, 2, 1, 1, 0, 0, 0, 0, 0],
    } or fixed_profile["four_condition_observable_seed_labels"] != [
        27, 27, 26, 12, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]:
        raise RuntimeError("H01 fixed request-index support profile changed")
    first_unavailability_decomposition = fixed_profile[
        "first_unavailability_decomposition"
    ]
    expected_first_losses = {
        "control": [0, 0, 1, 8, 10, 3, 4, 0, 0, 1, 0, 0, 0, 0, 0],
        "position_mirrored": [0, 0, 0, 10, 12, 1, 2, 0, 1, 0, 1, 0, 0, 0, 0],
    }
    expected_limiter_sets = {
        "control": {"left_only": 13, "left_and_right": 3, "right_only": 11},
        "position_mirrored": {
            "left_only": 19, "left_and_right": 6, "right_only": 2,
        },
    }
    for layout in ("control", "position_mirrored"):
        first_loss = first_unavailability_decomposition["by_layout"][layout]
        if (
            first_loss["population_seed_labels"] != 27
            or first_loss["observable_seed_labels_by_request_index"]
            != fixed_profile["layout_observable_seed_labels"][layout]
            or first_loss["first_unavailable_seed_labels_by_request_index"]
            != expected_first_losses[layout]
            or first_loss["limiting_command_set_counts"]
            != expected_limiter_sets[layout]
            or first_loss["limiting_cell_recorded_outcome_counts"]["cap_failure"] != 0
            or set(first_loss["outcome_strata"]) != set(pair_outcome_strata)
        ):
            raise RuntimeError("H01 layout first-unavailability decomposition changed")
    four_condition_first_loss = first_unavailability_decomposition[
        "four_condition"
    ]
    if (
        four_condition_first_loss["observable_seed_labels_by_request_index"]
        != fixed_profile["four_condition_observable_seed_labels"]
        or four_condition_first_loss[
            "first_unavailable_seed_labels_by_request_index"
        ] != [0, 0, 1, 14, 10, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]
        or four_condition_first_loss["limiting_set_size_counts"]
        != {"1": 15, "2": 10, "3": 2, "4": 0}
        or four_condition_first_loss["limiting_cell_recorded_outcome_counts"]
        != {"requested_success": 41, "cap_failure": 0}
        or set(four_condition_first_loss["outcome_strata"])
        != set(joint_pair_outcome_strata)
    ):
        raise RuntimeError("H01 four-condition first-unavailability decomposition changed")
    fixed_rows = fixed_profile["rows"]
    if fixed_rows[1]["by_layout"]["control"]["observed_signed_contrast_m"][
        "sign_counts"
    ] != {"positive": 2, "negative": 6, "exact_zero": 19}:
        raise RuntimeError("H01 universal 32--64 action-window signs changed")
    if fixed_rows[2]["by_layout"]["position_mirrored"][
        "observed_signed_contrast_m"
    ]["sign_counts"] != {"positive": 25, "negative": 1, "exact_zero": 1}:
        raise RuntimeError("H01 64--96 action-window reflected signs changed")
    if fixed_rows[2]["four_condition_same_index_interaction_m"][
        "observed_signed_contrast_m"
    ]["sign_counts"] != {"positive": 15, "negative": 11, "exact_zero": 0}:
        raise RuntimeError("H01 fixed 64--96 four-condition interaction changed")
    expected_pair_outcome_populations = {
        "control": {
            "success_left__success_right": 25,
            "success_left__cap_failure_right": 1,
            "cap_failure_left__success_right": 1,
            "cap_failure_left__cap_failure_right": 0,
        },
        "position_mirrored": {
            "success_left__success_right": 23,
            "success_left__cap_failure_right": 4,
            "cap_failure_left__success_right": 0,
            "cap_failure_left__cap_failure_right": 0,
        },
    }
    expected_pair_outcome_observable = {
        "control": {
            "success_left__success_right": [25, 25, 24, 16, 8, 5, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "success_left__cap_failure_right": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "cap_failure_left__success_right": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "cap_failure_left__cap_failure_right": [0] * 15,
        },
        "position_mirrored": {
            "success_left__success_right": [23, 23, 23, 13, 4, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "success_left__cap_failure_right": [4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "cap_failure_left__success_right": [0] * 15,
            "cap_failure_left__cap_failure_right": [0] * 15,
        },
    }
    for layout in ("control", "position_mirrored"):
        observed_by_stratum = {
            stratum: [] for stratum in pair_outcome_strata
        }
        for row in fixed_rows:
            support = row["by_layout"][layout]
            attrition = support["pair_outcome_attrition"]
            if (
                attrition["population_seed_labels"] != 27
                or attrition["observable_seed_labels"]
                != support["observable_seed_labels"]
                or attrition["unavailable_seed_labels"]
                != support["unavailable_seed_labels"]
                or set(attrition["strata"]) != set(pair_outcome_strata)
            ):
                raise RuntimeError("H01 fixed-index pair-outcome totals changed")
            population_ids = []
            observable_ids = []
            unavailable_ids = []
            for stratum in pair_outcome_strata:
                record = attrition["strata"][stratum]
                if (
                    record["population_seed_labels"]
                    != expected_pair_outcome_populations[layout][stratum]
                    or record["observable_seed_labels"]
                    + record["unavailable_seed_labels"]
                    != record["population_seed_labels"]
                ):
                    raise RuntimeError("H01 pair-outcome stratum population changed")
                observed_by_stratum[stratum].append(
                    record["observable_seed_labels"]
                )
                population_ids.extend(record["population_seed_ids"])
                observable_ids.extend(record["observable_seed_ids"])
                unavailable_ids.extend(record["unavailable_seed_ids"])
            if (
                sorted(population_ids) != list(range(9400, 9427))
                or sorted(observable_ids) != support["observable_seed_ids"]
                or sorted(unavailable_ids) != support["unavailable_seed_ids"]
            ):
                raise RuntimeError("H01 pair-outcome stratum identities changed")
        if observed_by_stratum != expected_pair_outcome_observable[layout]:
            raise RuntimeError("H01 pair-outcome attrition profile changed")
    joint_ss_ss = (
        "control=success_left__success_right|"
        "position_mirrored=success_left__success_right"
    )
    joint_ss_sf = (
        "control=success_left__success_right|"
        "position_mirrored=success_left__cap_failure_right"
    )
    joint_sf_ss = (
        "control=success_left__cap_failure_right|"
        "position_mirrored=success_left__success_right"
    )
    joint_fs_ss = (
        "control=cap_failure_left__success_right|"
        "position_mirrored=success_left__success_right"
    )
    expected_joint_populations = {
        **{stratum: 0 for stratum in joint_pair_outcome_strata},
        joint_ss_ss: 21,
        joint_ss_sf: 4,
        joint_sf_ss: 1,
        joint_fs_ss: 1,
    }
    expected_joint_observable = {
        **{stratum: [0] * 15 for stratum in joint_pair_outcome_strata},
        joint_ss_ss: [21, 21, 20, 8, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        joint_ss_sf: [4, 4, 4, 4, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        joint_sf_ss: [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        joint_fs_ss: [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    observed_joint_by_stratum = {
        stratum: [] for stratum in joint_pair_outcome_strata
    }
    for row in fixed_rows:
        support = row["four_condition_same_index_interaction_m"]
        attrition = support["joint_pair_outcome_attrition"]
        if (
            attrition["population_seed_labels"] != 27
            or attrition["observable_seed_labels"]
            != support["observable_seed_labels"]
            or attrition["unavailable_seed_labels"]
            != support["unavailable_seed_labels"]
            or set(attrition["strata"]) != set(joint_pair_outcome_strata)
        ):
            raise RuntimeError("H01 joint outcome-attrition totals changed")
        population_ids = []
        observable_ids = []
        unavailable_ids = []
        for stratum in joint_pair_outcome_strata:
            record = attrition["strata"][stratum]
            if (
                record["population_seed_labels"]
                != expected_joint_populations[stratum]
                or record["observable_seed_labels"]
                + record["unavailable_seed_labels"]
                != record["population_seed_labels"]
            ):
                raise RuntimeError("H01 joint outcome stratum population changed")
            observed_joint_by_stratum[stratum].append(
                record["observable_seed_labels"]
            )
            population_ids.extend(record["population_seed_ids"])
            observable_ids.extend(record["observable_seed_ids"])
            unavailable_ids.extend(record["unavailable_seed_ids"])
        if (
            sorted(population_ids) != list(range(9400, 9427))
            or sorted(observable_ids) != support["observable_seed_ids"]
            or sorted(unavailable_ids) != support["unavailable_seed_ids"]
        ):
            raise RuntimeError("H01 joint outcome stratum identities changed")
    if observed_joint_by_stratum != expected_joint_observable:
        raise RuntimeError("H01 joint outcome-attrition profile changed")
    request_boundary = request_window["claim_boundary"]
    if request_boundary["retrospective_outcome_stratified_attrition_evidence"] is not True:
        raise RuntimeError("H01 retrospective outcome-attrition evidence flag changed")
    if request_boundary["retrospective_first_unavailability_evidence"] is not True:
        raise RuntimeError("H01 retrospective first-unavailability evidence flag changed")
    if request_boundary["first_loss_arrays_are_algebraic_support_differences"] is not True:
        raise RuntimeError("H01 first-loss algebraic-dependence boundary changed")
    for field in (
        "forecast_arrays_retrieved", "forecast_payload_bytes_reopened_or_reverified",
        "forecast_accuracy_or_skill_estimated",
        "generated_versus_executed_comparison", "A3_A9_completed",
        "physical_time_established", "A6_movement_threshold_established",
        "outcome_stratified_motion_effect_estimated",
        "independent_first_unavailability_estimand_estimated",
        "censoring_independence_established",
        "pair_outcome_establishes_language_or_model_diagnosis",
        "independent_scene_or_layout_generalization_established",
        "language_understanding_established", "new_episodes_or_requests_created",
        "new_labels_or_retrieval_created", "models_or_cohorts_pooled",
        "changes_232_cell_core_bound",
    ):
        if request_boundary[field] is not False:
            raise RuntimeError(f"H01 request-window boundary changed: {field}")
    if request_boundary["maximum_core_behavioral_cells"] != 232:
        raise RuntimeError("H01 request-window package changed the core-cell ceiling")
    if any(request_window["new_activity"].values()) or any(request_window["gate_state"].values()):
        raise RuntimeError("H01 request-window package advanced activity or gates")
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
        "retained_h01_request_window_motion": {
            "scope": request_boundary,
            "source_commit": request_window["provenance"]["commit"],
            "source_jsonl": {
                "path": stopping_source_path,
                **request_window["provenance"]["files"][stopping_source_path],
            },
            "cohort": request_window["cohort"],
            "coverage": request_window["coverage"],
            "matched_complete_window_analysis": matched_windows,
            "new_activity": request_window["new_activity"],
            "gate_state": request_window["gate_state"],
            "json_sha256": digest(request_window_path),
            "csv_sha256": digest(package / "paper/data/h01_request_window_motion.csv"),
            "analyzer_sha256": digest(package / "analysis/analyze_h01_request_window_motion.py"),
        },
        "retained_h02_position_reflection": {
            "scope": h02["claim_boundary"],
            "source_commit": h02["source_provenance"]["commit"],
            "source_authentication": {
                key: h02["source_provenance"][key]
                for key in (
                    "authenticated_file_count", "authenticated_bytes", "authenticated_lines"
                )
            },
            "configuration": h02["configuration"],
            "population": h02["authenticated_population"],
            "condition_outcomes": h02["condition_outcomes"],
            "layout_totals": h02["layout_totals"],
            "failure_taxonomy_counts": h02["failure_taxonomy_counts"],
            "recorded_requested_region_entry": h02["recorded_requested_region_entry"],
            "registered_point_estimands": h02["registered_point_estimands"],
            "descriptive_execution": h02["descriptive_execution"],
            "declared_raw_artifact_inventory": h02["declared_raw_artifact_inventory"],
            "new_activity": h02["new_activity"],
            "gate_state": h02["gate_state"],
            "json_sha256": digest(h02_path),
            "csv_sha256": digest(package / "paper/data/h02_position_reflection.csv"),
            "analyzer_sha256": digest(package / "analysis/analyze_h02_position_reflection.py"),
        },
        "retained_h02_matched_entry_supplement": {
            "scope": h02_matched["claim_boundary"],
            "provenance": h02_matched["provenance"],
            "configuration": h02_matched["configuration"],
            "population": h02_matched["population"],
            "summaries": h02_matched["summaries"],
            "definitions": h02_matched["definitions"],
            "new_activity": h02_matched["new_activity"],
            "gate_state": h02_matched["gate_state"],
            "json_sha256": digest(h02_matched_path),
            "csv_sha256": digest(package / "paper/data/h02_matched_entry.csv"),
            "analyzer_sha256": digest(package / "analysis/analyze_h02_matched_entry.py"),
        },
        "retained_h03_h04_guidance_pilots": {
            "scope": h03_h04["claim_boundary"],
            "source_commit": h03_h04["provenance"]["commit"],
            "design": h03_h04["design"],
            "cohorts": h03_h04["cohorts"],
            "matched_cell_rows": len(h03_h04["matched_cell_contrasts"]),
            "matched_seed_diagnostic_rows": len(h03_h04["matched_seed_diagnostics"]),
            "source_manifest": h03_h04["source_manifest"],
            "reproduction": h03_h04["reproduction"],
            "new_activity": h03_h04["new_activity"],
            "gate_state": h03_h04["gate_state"],
            "json_sha256": digest(h03_h04_path),
            "csv_sha256": digest(package / "paper/data/h03_h04_guidance_pilots.csv"),
            "analyzer_sha256": digest(package / "analysis/analyze_h03_h04_guidance_pilots.py"),
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
        "retained_historical_evidence_coverage": {
            "scope": coverage["claim_boundary"],
            "registry": coverage["registry"],
            "packaging_completion": coverage["packaging_completion"],
            "coverage_summary": coverage["coverage_summary"],
            "historical_custom_s2_ids_not_official_D1": coverage[
                "historical_custom_s2_ids_not_official_D1"
            ],
            "preserved_evidence_gaps": coverage["preserved_evidence_gaps"],
            "new_activity": coverage["new_activity"],
            "gate_state": coverage["gate_state"],
            "json_sha256": digest(coverage_path),
            "csv_sha256": digest(package / "paper/data/retained_historical_evidence_coverage.csv"),
            "auditor_sha256": digest(package / "analysis/audit_retained_historical_evidence_coverage.py"),
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
