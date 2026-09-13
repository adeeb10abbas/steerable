import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / "analysis" / "forecast_evidence_analysis.py"
SPEC = importlib.util.spec_from_file_location("forecast_evidence_analysis", MODULE)
analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analysis)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def consensus_label(cube, bowl=(50.0, 50.0)):
    return {
        "annotation": {
            "cube_resolvability": "resolvable",
            "bowl_resolvability": "resolvable",
            "cube_identity": "rubiks_cube",
            "bowl_identity": "bowl",
            "cube_center_px": list(cube),
            "bowl_center_px": list(bowl),
            "bowl_width_px": 20.0,
            "ambiguity_codes": ["none"],
        },
        "width_px": 100,
        "height_px": 100,
        "restricted_asset_id": "asset_000001",
    }


def endpoint(action_index, cube_y):
    return {
        "action_index": action_index,
        "cube_robot_xyz": (0.0, cube_y, 0.1),
        "bowl_robot_xyz": (0.0, 0.0, 0.1),
    }


def valid_gate():
    return {
        "recording_and_action_chain": "VALIDATED",
        "development_release_freeze": "VALIDATED",
        "confirmation_annotation_freeze": "VALIDATED",
        "blind_human_consensus": "REPRODUCED",
        "technical_statuses_preserved": "VALIDATED",
    }


def install_annotation_quality(context, decisions):
    decisions = sorted(copy.deepcopy(decisions), key=lambda row: row["restricted_asset_id"])
    counts = {
        "images": len(decisions),
        "first_pass_exact_agreements": sum(
            row["decision_source"] == "first_pass_exact_agreement"
            for row in decisions
        ),
        "independently_adjudicated": sum(
            row["decision_source"] == "independent_adjudicator"
            for row in decisions
        ),
    }
    has_adjudication = counts["independently_adjudicated"] > 0
    consensus = analysis.sign_document(
        {
            "counts": counts,
            "labels": decisions,
            "first_pass_response_sha256_by_slot": {
                "rater_a": digest("response-a"),
                "rater_b": digest("response-b"),
            },
            "rater_code_sha256_by_slot": {
                "rater_a": digest("rater-a"),
                "rater_b": digest("rater-b"),
                "adjudicator": digest("adjudicator") if has_adjudication else None,
            },
            "adjudicator_response_sha256": (
                digest("adjudicator-response") if has_adjudication else None
            ),
            "source_restricted_map_sha256": context["sources"]["restricted_map"]["sha256"],
            "adjudication_map_sha256": digest("adjudication-map"),
        }
    )
    consensus_bytes = analysis.canonical_bytes(consensus) + b"\n"
    consensus_sha = hashlib.sha256(consensus_bytes).hexdigest()
    context["sources"]["final_consensus"] = {
        "path": "/final-consensus.json",
        "sha256": consensus_sha,
    }
    context["annotation_quality"] = analysis.annotation_quality_summary(
        consensus,
        final_consensus_sha256=consensus_sha,
    )
    context["annotation_quality_decisions"] = decisions
    context["annotation_quality_consensus_bytes"] = consensus_bytes


def make_context(*, branch="reduced_n3", predicted_x=51.8):
    planned = analysis.load_annotation_module()._planned_cells("confirmation", branch)
    roster = []
    selected = {}
    labels = {}
    endpoints = {}
    episodes = []
    request_inventory = []
    for cell_index, (cell_id, (model, layout, condition)) in enumerate(sorted(planned.items())):
        recording_id = f"recording-{cell_index:03d}"
        roster_row = {
            "cell_id": cell_id,
            "recording_id": recording_id,
            "model_id": model,
            "layout_pair_id": layout,
            "condition_id": condition,
            "recording_status": "valid_complete",
            "executed_action_count": 450,
            "censor_reason": None,
            "recording_receipt_path": f"/raw/{cell_id}.json",
            "recording_receipt_sha256": digest(f"recording:{cell_id}"),
            "action_manifest_path": f"/raw/{cell_id}-actions.json",
            "action_manifest_sha256": digest(f"actions:{cell_id}"),
            "source_video_id": f"video-{cell_index:03d}",
            "source_video_sha256": digest(f"video:{cell_id}"),
        }
        roster.append(roster_row)
        episodes.append(
            {
                "cell_id": cell_id,
                "episode_id": recording_id,
                "recording_status": "valid_complete",
                "request_count": 1,
                "eligible_count": 1,
                "selected_count": 1,
                "zero_eligible": False,
                "eligible_request_inclusion_probability": 1.0,
                "eligible_request_inclusion_probability_exact": "1/1",
            }
        )
        request_id = f"{cell_id}/request_0"
        selected[request_id] = {
            "source_request_id": request_id,
            "cell_id": cell_id,
            "model_id": model,
            "layout_pair_id": layout,
            "condition_id": condition,
            "history_mode": "persistence_at_initial_request",
            "target_physical_time_s": 0.5,
            "early_horizon_supported": False,
        }
        request_inventory.append(
            {
                "source": selected[request_id],
                "timing_camera_action_eligible": True,
                "eligibility_reasons": [],
                "selected": True,
                "eligible_request_inclusion_probability": 1.0,
                "eligible_request_inclusion_probability_exact": "1/1",
            }
        )
        labels[request_id] = {
            "current": consensus_label((50.0, 50.0)),
            "predicted": consensus_label((predicted_x, 50.0)),
            "executed": consensus_label((52.0, 50.0)),
        }
        for role, label in labels[request_id].items():
            label["restricted_asset_id"] = f"asset_{cell_index:03d}_{role}"
        endpoints[cell_id] = {
            "cell_id": cell_id,
            "action_zero": endpoint(0, 0.0),
            "action_450": endpoint(450, 0.1 if condition.endswith("left") else -0.1),
            "first_success_or_action_450": endpoint(
                100 if condition.endswith("left") else 450,
                0.08 if condition.endswith("left") else -0.1,
            ),
            "first_success_action_index": 100 if condition.endswith("left") else None,
        }
    quality_decisions = [
        {
            "restricted_asset_id": label["restricted_asset_id"],
            "decision_source": "first_pass_exact_agreement",
        }
        for role_rows in labels.values()
        for label in role_rows.values()
    ]
    quality_decisions.sort(key=lambda row: row["restricted_asset_id"])
    restricted_map_sha = digest("restricted-map")
    context = {
        "evidence_gate": valid_gate(),
        "branch": branch,
        "selection": {
            "episode_roster": roster,
            "episodes": episodes,
            "requests": request_inventory,
            "alignment_contracts": [
                {
                    "model_id": model,
                    "primary_horizon_s": 0.5,
                    "generated_frame_index": 3,
                    "target_executed_action_offset": 8,
                    "camera_id": "over_shoulder_left_camera",
                    "timestamp_tolerance_s": 1 / 30,
                }
                for model in analysis.MODEL_BRANCHES[branch]
            ],
        },
        "selected": selected,
        "labels": labels,
        "endpoints": endpoints,
        "histories": {},
        "movement_threshold": 0.005,
        "model_specification": {
            "N3": {"checkpoint_revision": "n3-revision", "executed_prefix_cap": 32},
            "D1": {"checkpoint_revision": "d1-revision", "executed_prefix_cap": 8},
        },
        "sources": {
            "restricted_map": {"path": "/restricted-map.json", "sha256": restricted_map_sha},
        },
        "manifest": {"path": "/evidence.json", "sha256": "a" * 64},
    }
    install_annotation_quality(context, quality_decisions)
    return context


def install_early_horizon(context, *, predicted_x=51.0, executed_x=51.0):
    target = {
        "horizon_s": 0.25,
        "generated_frame_index": 1,
        "target_executed_action_offset": 4,
    }
    for alignment in context["selection"]["alignment_contracts"]:
        alignment["early_horizon"] = copy.deepcopy(target)
    for index, request_id in enumerate(sorted(context["selected"])):
        context["selected"][request_id]["early_horizon_supported"] = True
        roles = context["labels"][request_id]
        roles["early_predicted"] = consensus_label((predicted_x, 50.0))
        roles["early_executed"] = consensus_label((executed_x, 50.0))
        roles["early_predicted"]["restricted_asset_id"] = f"asset_{index:03d}_early_predicted"
        roles["early_executed"]["restricted_asset_id"] = f"asset_{index:03d}_early_executed"
    decisions = [
        {
            "restricted_asset_id": label["restricted_asset_id"],
            "decision_source": "first_pass_exact_agreement",
        }
        for role_rows in context["labels"].values()
        for label in role_rows.values()
    ]
    install_annotation_quality(context, decisions)
    return target


class ForecastEvidenceAnalysisTests(unittest.TestCase):
    def test_annotation_quality_is_rederived_from_consensus_decisions(self):
        consensus = {
            "counts": {
                "images": 2,
                "first_pass_exact_agreements": 1,
                "independently_adjudicated": 1,
            },
            "labels": [
                {"restricted_asset_id": "asset_a", "decision_source": "first_pass_exact_agreement"},
                {"restricted_asset_id": "asset_b", "decision_source": "independent_adjudicator"},
            ],
            "first_pass_response_sha256_by_slot": {
                "rater_a": digest("response-a"),
                "rater_b": digest("response-b"),
            },
            "rater_code_sha256_by_slot": {
                "rater_a": digest("rater-a"),
                "rater_b": digest("rater-b"),
                "adjudicator": digest("adjudicator"),
            },
            "adjudicator_response_sha256": digest("adjudicator-response"),
            "source_restricted_map_sha256": digest("restricted-map"),
            "adjudication_map_sha256": digest("adjudication-map"),
        }
        result = analysis.annotation_quality_summary(
            consensus,
            final_consensus_sha256=digest("final-consensus"),
        )
        self.assertEqual(result["first_pass_exact_agreement_rate"], 0.5)
        self.assertEqual(result["independent_adjudication_rate"], 0.5)
        tampered = copy.deepcopy(consensus)
        tampered["counts"]["independently_adjudicated"] = 0
        with self.assertRaisesRegex(
            analysis.AnalysisContractError,
            "counts are inconsistent",
        ):
            analysis.annotation_quality_summary(
                tampered,
                final_consensus_sha256=digest("final-consensus"),
            )

    def test_report_carries_verified_inter_rater_agreement_and_adjudication(self):
        context = make_context()
        decisions = context["annotation_quality_decisions"]
        for row in decisions[:3]:
            row["decision_source"] = "independent_adjudicator"
        install_annotation_quality(context, decisions)
        quality = context["annotation_quality"]
        image_count = len(decisions)
        report = analysis.build_report(context)
        self.assertEqual(report["annotation_quality"], quality)
        self.assertEqual(report["annotation_quality"]["independently_adjudicated"], 3)
        self.assertEqual(
            report["annotation_quality"]["independent_adjudication_rate"],
            3 / image_count,
        )

    def test_adjudicated_consensus_requires_hashed_independent_evidence(self):
        context = make_context()
        decisions = context["annotation_quality_decisions"]
        decisions[0]["decision_source"] = "independent_adjudicator"
        install_annotation_quality(context, decisions)
        quality = context["annotation_quality"]
        quality["rater_code_sha256_by_slot"]["adjudicator"] = None
        quality["adjudicator_response_sha256"] = None
        with self.assertRaisesRegex(
            analysis.AnalysisContractError,
            "adjudicator code sha256",
        ):
            analysis.build_report(context)

    def test_annotation_quality_rejects_detached_or_inconsistent_context(self):
        detached = make_context()
        detached["sources"]["final_consensus"]["sha256"] = digest("another-consensus")
        with self.assertRaisesRegex(
            analysis.AnalysisContractError,
            "detached from final consensus",
        ):
            analysis.build_report(detached)

        empty = make_context()
        install_annotation_quality(empty, [])
        with self.assertRaisesRegex(
            analysis.AnalysisContractError,
            "assets differ from validated request labels",
        ):
            analysis.build_report(empty)

        changed = make_context()
        changed["annotation_quality_decisions"][0]["decision_source"] = (
            "independent_adjudicator"
        )
        with self.assertRaisesRegex(
            analysis.AnalysisContractError,
            "differs from final consensus",
        ):
            analysis.build_report(changed)

    def test_annotation_quality_rejects_reused_adjudicator_identity(self):
        context = make_context()
        decisions = context["annotation_quality_decisions"]
        decisions[0]["decision_source"] = "independent_adjudicator"
        install_annotation_quality(context, decisions)
        quality = context["annotation_quality"]
        quality["rater_code_sha256_by_slot"]["adjudicator"] = (
            quality["rater_code_sha256_by_slot"]["rater_a"]
        )
        with self.assertRaisesRegex(
            analysis.AnalysisContractError,
            "adjudicator is not independent",
        ):
            analysis.build_report(context)

    def test_reduced_branch_reports_positive_skill_without_substitution(self):
        report = analysis.build_report(make_context())
        n3 = report["models"]["N3"]
        self.assertGreater(
            n3["baseline_errors_and_skill"]["forecast_skill_vs_persistence"]["estimate"],
            0,
        )
        self.assertEqual(
            n3["baseline_errors_and_skill"]["forecast_skill_vs_persistence"]["layout_pairs"],
            24,
        )
        self.assertIn("REDUCED_ONE_MODEL_N3_BRANCH", report["study_scope"])
        d1 = next(row for row in report["model_sample_size_table"] if row["model_id"] == "D1")
        self.assertEqual(d1["unrun_due_unqualified_branch"], 96)
        self.assertNotIn("D1", report["models"])
        self.assertEqual(report["payload_sha256"], analysis.payload_hash(report))

    def test_negative_skill_is_preserved(self):
        report = analysis.build_report(make_context(predicted_x=80.0))
        skill = report["models"]["N3"]["baseline_errors_and_skill"]["forecast_skill_vs_persistence"]
        self.assertLess(skill["estimate"], 0)
        self.assertLess(skill["ci95"][1], 0)
        self.assertTrue(report["claim_boundaries"]["either_sign_reported"])

    def test_earlier_horizon_reports_exact_same_request_paired_contrast(self):
        context = make_context()
        target = install_early_horizon(context)
        report = analysis.build_report(context)
        earlier = report["models"]["N3"]["earlier_horizon"]
        self.assertEqual(earlier["status"], "supported_and_observed")
        self.assertEqual(earlier["qualified_target"], target)
        self.assertEqual(earlier["observable_requests"], 96)

        diagonal = math.hypot(100.0, 100.0)
        expected_primary = 1.8 / diagonal
        expected_early = 1.0 / diagonal
        expected_delta = 0.8 / diagonal
        comparison = earlier["paired_skill_comparison"]
        expected = {
            "primary_skill_at_H": expected_primary,
            "early_skill": expected_early,
            "primary_minus_early_skill": expected_delta,
        }
        for key, value in expected.items():
            metric = comparison[key]
            self.assertAlmostEqual(metric["estimate"], value)
            self.assertAlmostEqual(metric["ci95"][0], value)
            self.assertAlmostEqual(metric["ci95"][1], value)
            self.assertEqual(metric["layout_pairs"], 24)
            self.assertEqual(metric["resamples"], 10_000)
            self.assertEqual(metric["seed"], analysis.ANALYSIS_SEED)

    def test_earlier_horizon_pairing_never_subtracts_marginal_request_means(self):
        request = {
            "source_request_id": "missing-early",
            "cell_id": "cell",
            "model_id": "N3",
            "layout_pair_id": "C01",
            "condition_id": "original_left",
            "history_mode": "persistence_at_initial_request",
            "target_physical_time_s": 0.5,
            "early_horizon_supported": True,
        }
        missing_early_labels = {
            "current": consensus_label((50.0, 50.0)),
            "predicted": consensus_label((52.0, 50.0)),
            "executed": consensus_label((52.0, 50.0)),
            "early_predicted": consensus_label((51.0, 50.0)),
            "early_executed": consensus_label((51.0, 50.0)),
        }
        missing_early_labels["early_predicted"]["annotation"]["cube_resolvability"] = "unresolvable"
        marginal_only = analysis.request_metrics(
            request, missing_early_labels, None, movement_threshold=0
        )
        self.assertTrue(marginal_only["primary_observable"])
        self.assertFalse(marginal_only["early_observable"])
        self.assertNotIn("primary_skill_paired_with_early", marginal_only)
        self.assertNotIn("primary_minus_early_skill", marginal_only)

        paired_request = {**request, "source_request_id": "jointly-observable"}
        paired_labels = {
            "current": consensus_label((50.0, 50.0)),
            "predicted": consensus_label((51.5, 50.0)),
            "executed": consensus_label((52.0, 50.0)),
            "early_predicted": consensus_label((50.5, 50.0)),
            "early_executed": consensus_label((51.0, 50.0)),
        }
        paired = analysis.request_metrics(
            paired_request, paired_labels, None, movement_threshold=0
        )
        summary = analysis._cell_summary(
            {
                "cell_id": "cell",
                "model_id": "N3",
                "layout_pair_id": "C01",
                "condition_id": "original_left",
                "recording_status": "valid_complete",
            },
            [marginal_only, paired],
        )
        self.assertNotAlmostEqual(
            summary["means"]["skill"],
            summary["means"]["primary_skill_paired_with_early"],
        )
        self.assertAlmostEqual(
            summary["means"]["primary_skill_paired_with_early"], paired["skill"]
        )
        self.assertAlmostEqual(
            summary["means"]["primary_minus_early_skill"],
            paired["skill"] - paired["early_skill"],
        )

    def test_earlier_horizon_state_boundaries_and_full_model_separation(self):
        unsupported = analysis.build_report(make_context())
        unsupported_early = unsupported["models"]["N3"]["earlier_horizon"]
        self.assertEqual(
            unsupported_early["status"],
            "unsupported_no_earlier_qualified_exposed_target",
        )
        self.assertNotIn("paired_skill_comparison", unsupported_early)

        context = make_context(branch="full_two_model")
        install_early_horizon(context)
        for request_id, request in context["selected"].items():
            if request["model_id"] == "D1":
                context["labels"][request_id]["early_predicted"]["annotation"][
                    "cube_resolvability"
                ] = "unresolvable"
        report = analysis.build_report(context)
        n3_early = report["models"]["N3"]["earlier_horizon"]
        d1_early = report["models"]["D1"]["earlier_horizon"]
        self.assertEqual(n3_early["status"], "supported_and_observed")
        self.assertEqual(
            n3_early["paired_skill_comparison"]["primary_minus_early_skill"][
                "layout_pairs"
            ],
            24,
        )
        self.assertEqual(
            d1_early["status"], "qualified_but_unobservable_in_consensus"
        )
        self.assertEqual(d1_early["observable_requests"], 0)
        self.assertNotIn("paired_skill_comparison", d1_early)

    def test_invalid_censored_and_unrun_cells_stay_separate(self):
        context = make_context()
        affected = {
            "original_left": "technical_invalid",
            "original_right": "valid_censored",
            "reflected_left": "not_run",
        }
        roster = {row["condition_id"]: row for row in context["selection"]["episode_roster"] if row["layout_pair_id"] == "C24"}
        for condition, status in affected.items():
            row = roster[condition]
            row["recording_status"] = status
            request_id = f"{row['cell_id']}/request_0"
            context["selected"].pop(request_id)
            context["labels"].pop(request_id)
            if status == "valid_censored":
                row["executed_action_count"] = 200
                context["endpoints"][row["cell_id"]] = {
                    "cell_id": row["cell_id"],
                    "action_zero": endpoint(0, 0.0),
                    "action_450": None,
                    "first_success_or_action_450": None,
                    "first_success_action_index": None,
                }
            else:
                context["endpoints"].pop(row["cell_id"])
        retained_assets = {
            label["restricted_asset_id"]
            for role_rows in context["labels"].values()
            for label in role_rows.values()
        }
        decisions = [
            row
            for row in context["annotation_quality_decisions"]
            if row["restricted_asset_id"] in retained_assets
        ]
        install_annotation_quality(context, decisions)
        report = analysis.build_report(context)
        table = next(row for row in report["model_sample_size_table"] if row["model_id"] == "N3")
        self.assertEqual(table["technical_invalid"], 1)
        self.assertEqual(table["valid_censored"], 1)
        self.assertEqual(table["unrun"], 1)
        self.assertEqual(table["continuous_complete_layout_pairs"], 23)
        bounds = report["models"]["N3"]["full_design_strict_win_missingness_bounds"]
        self.assertLess(bounds["lower"], bounds["upper"])
        self.assertEqual(report["models"]["N3"]["stopping_control"]["censored_cells_excluded_without_carry_forward"], 1)

    def test_constant_velocity_uses_only_timestamped_preceding_observation(self):
        request = {
            "source_request_id": "r1",
            "cell_id": "c1",
            "model_id": "N3",
            "layout_pair_id": "C01",
            "condition_id": "original_left",
            "history_mode": "preceding_observation",
            "target_physical_time_s": 1.0,
            "early_horizon_supported": False,
        }
        labels = {
            "preceding": consensus_label((49.0, 50.0)),
            "current": consensus_label((50.0, 50.0)),
            "predicted": consensus_label((52.0, 50.0)),
            "executed": consensus_label((52.0, 50.0)),
        }
        row = analysis.request_metrics(
            request,
            labels,
            {"preceding_observation_interval_s": 0.5},
            movement_threshold=0,
        )
        self.assertTrue(row["cv_observable"])
        self.assertAlmostEqual(row["constant_velocity_error"], 0.0)
        with self.assertRaisesRegex(analysis.AnalysisContractError, "timing receipt"):
            analysis.request_metrics(request, labels, None, movement_threshold=0)

    def test_censored_endpoint_receipt_cannot_fabricate_action_450(self):
        roster = {
            "cell_id": "cell",
            "recording_id": "recording",
            "model_id": "N3",
            "recording_status": "valid_censored",
            "recording_receipt_sha256": "a" * 64,
            "action_manifest_sha256": "b" * 64,
            "executed_action_count": 200,
        }
        receipt = analysis.sign_document(
            {
                "schema_version": analysis.ENDPOINT_SCHEMA,
                "study_id": analysis.STUDY_ID,
                "stage": "confirmation",
                "cell_id": "cell",
                "recording_id": "recording",
                "model_id": "N3",
                "recording_status": "valid_censored",
                "recording_receipt_sha256": "a" * 64,
                "action_manifest_sha256": "b" * 64,
                "executed_action_count": 200,
                "first_success_action_index": None,
                "action_zero": {"action_index": 0, "cube_robot_xyz": [0, 0, 0], "bowl_robot_xyz": [0, 0, 0]},
                "action_450": {"action_index": 450, "cube_robot_xyz": [0, 0, 0], "bowl_robot_xyz": [0, 0, 0]},
                "first_success_or_action_450": None,
                "action_zero_observation_id": "obs_000000",
                "action_450_observation_id": "obs_000450",
                "first_success_or_action_450_observation_id": None,
                "source_adapter_completion": {"path": "completion.json", "sha256": "c" * 64},
                "source_adapter_journal": {"path": "events.partial.jsonl", "sha256": "d" * 64},
            }
        )
        with self.assertRaisesRegex(analysis.AnalysisContractError, "fabricates action 450"):
            analysis._validate_endpoint_fields(receipt, roster=roster)

    def test_analysis_refuses_context_without_validated_freeze_and_consensus(self):
        context = make_context()
        context.pop("evidence_gate")
        with self.assertRaisesRegex(analysis.AnalysisContractError, "refuses to run"):
            analysis.build_report(context)
        context = make_context()
        context["labels"].pop(next(iter(context["labels"])))
        with self.assertRaisesRegex(analysis.AnalysisContractError, "cover every selected"):
            analysis.build_report(context)

    def test_release_and_confirmation_freezes_must_bind_exact_consensus_and_decision(self):
        decision = {
            "measurement_usable": True,
            "decided_by": "authorized route",
            "decided_at": "2026-09-13T01:00:00Z",
            "basis": "validated",
            "development_summary_sha256": "b" * 64,
        }
        freeze = {
            "movement_resolution": {
                "threshold_relative_image_diagonal": 0.01,
                "development_summary_sha256": "b" * 64,
            },
            "development_final_consensus": {"path": "consensus.json", "sha256": "c" * 64},
            "development_validation_decision": decision,
        }
        release = {
            "movement_resolution": {"threshold_relative_image_diagonal": 0.01},
            "development_summary": {"path": "summary.json", "sha256": "b" * 64},
            "rubric": {"path": "rubric.json", "sha256": "a" * 64},
            "final_consensus": {"path": "consensus.json", "sha256": "c" * 64},
            "usability_decision": decision,
        }
        analysis.validate_release_annotation_binding(release, freeze, {"rubric_sha256": "a" * 64})
        tampered = copy.deepcopy(release)
        tampered["final_consensus"]["sha256"] = "d" * 64
        with self.assertRaisesRegex(analysis.AnalysisContractError, "consensus hashes"):
            analysis.validate_release_annotation_binding(tampered, freeze, {"rubric_sha256": "a" * 64})
        tampered = copy.deepcopy(release)
        tampered["usability_decision"]["basis"] = "changed"
        with self.assertRaisesRegex(analysis.AnalysisContractError, "usability decisions"):
            analysis.validate_release_annotation_binding(tampered, freeze, {"rubric_sha256": "a" * 64})

    def test_first_success_is_rederived_from_native_action_snapshots(self):
        expected = {
            "action_step": 2,
            "requested_relation": "left",
            "released": True,
            "observation_id": "obs_000002",
        }
        adapter = {
            "completion": {"actions_executed": 2, "first_success": expected},
            "events": [
                {"kind": "environment_step_completed", "payload": {"action_step": 1, "success_predicates": {"left": False, "released": False}}},
                {"kind": "first_success", "payload": expected},
                {"kind": "environment_step_completed", "payload": {"action_step": 2, "success_predicates": {"left": True, "released": True}}},
            ],
        }
        self.assertEqual(analysis._derive_first_success(adapter, "left"), expected)
        tampered = copy.deepcopy(adapter)
        tampered["completion"]["first_success"] = {
            **tampered["completion"]["first_success"],
            "action_step": 1,
        }
        with self.assertRaisesRegex(analysis.AnalysisContractError, "completion first success"):
            analysis._derive_first_success(tampered, "left")

    def test_endpoint_point_is_loaded_from_measurement_only_payload(self):
        class FakeRecording:
            @staticmethod
            def load_payload(_attempt, _descriptor):
                return {
                    "observation_id": "obs_000000",
                    "phase": "settled_reset",
                    "clock": {"physics_time_s": 0.0},
                    "state": {
                        "simulator_state_sample_only_not_policy_input": True,
                        "objects": {
                            "rubiks_cube": {"position_robot_base_m": [[0.1, 0.2, 0.3]]},
                            "bowl": {"position_robot_base_m": [0.4, 0.5, 0.6]},
                        },
                    }
                }

        adapter = {"recording_module": FakeRecording, "attempt_directory": Path("/unused")}
        observations = {
            "obs_000000": {
                "control_step": 0,
                "phase": "settled_reset",
                "clock": {"physics_time_s": 0.0},
                "artifact": {"role": "observation", "bound": True},
            }
        }
        point = analysis._point_from_observation(adapter, observations, "obs_000000")
        self.assertEqual(point["cube_robot_xyz"], (0.1, 0.2, 0.3))
        self.assertEqual(point["bowl_robot_xyz"], (0.4, 0.5, 0.6))

    def test_history_interval_is_recomputed_from_bound_native_camera_clocks(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            completion = root / "completion.json"
            journal = root / "events.partial.jsonl"
            completion.write_text("{}\n", encoding="utf-8")
            journal.write_text("{}\n", encoding="utf-8")
            completion_sha = analysis.sha256_file(completion)
            journal_sha = analysis.sha256_file(journal)
            camera = "over_shoulder_left_camera"

            class FakeRecording:
                @staticmethod
                def load_payload(_attempt, descriptor):
                    return descriptor["test_payload"]

            preceding_clock = {
                "physics_time_s": 2.0,
                "cameras": {camera: {"capture_time_ns": 2_000_000_000}},
            }
            current_clock = {
                "physics_time_s": 2.1,
                "cameras": {camera: {"capture_time_ns": 2_100_000_000}},
            }
            adapter = {
                "completion_path": completion.resolve(),
                "journal_path": journal.resolve(),
                "completion_sha256": completion_sha,
                "journal_sha256": journal_sha,
                "recording_module": FakeRecording,
                "attempt_directory": root,
                "events": [{
                    "kind": "model_request_packed",
                    "payload": {
                        "request_index": 1,
                        "preceding_observation_id": "obs_000031",
                        "current_observation_id": "obs_000032",
                    },
                }],
                "observations": {
                    "obs_000031": {
                        "control_step": 31,
                        "phase": "post_action",
                        "clock": preceding_clock,
                        "artifact": {
                            "role": "observation",
                            "test_payload": {
                                "observation_id": "obs_000031",
                                "phase": "post_action",
                                "clock": preceding_clock,
                            },
                        },
                    },
                    "obs_000032": {
                        "control_step": 32,
                        "phase": "post_action",
                        "clock": current_clock,
                        "artifact": {
                            "role": "observation",
                            "test_payload": {
                                "observation_id": "obs_000032",
                                "phase": "post_action",
                                "clock": current_clock,
                            },
                        },
                    },
                },
            }
            request = {
                "source_request_id": "request-1",
                "cell_id": "cell-1",
                "alignment_receipt_id": "alignment-1",
                "alignment_receipt_sha256": "a" * 64,
                "camera_id": camera,
                "request_index": 1,
                "request_start_action_index": 32,
            }
            receipt = analysis.sign_document({
                "schema_version": analysis.HISTORY_SCHEMA,
                "study_id": analysis.STUDY_ID,
                "stage": "confirmation",
                "source_request_id": "request-1",
                "cell_id": "cell-1",
                "alignment_receipt_id": "alignment-1",
                "alignment_receipt_sha256": "a" * 64,
                "camera_id": camera,
                "preceding_observation_id": "obs_000031",
                "current_observation_id": "obs_000032",
                "preceding_camera_capture_time_ns": 2_000_000_000,
                "current_camera_capture_time_ns": 2_100_000_000,
                "preceding_physics_time_s": 2.0,
                "current_physics_time_s": 2.1,
                "preceding_observation_interval_s": 0.1,
                "source_adapter_completion": {"path": str(completion), "sha256": completion_sha},
                "source_adapter_journal": {"path": str(journal), "sha256": journal_sha},
            })
            result = analysis.validate_history_receipt(
                receipt,
                receipt_path=root / "history.json",
                request=request,
                adapter=adapter,
            )
            self.assertEqual(result["preceding_observation_interval_s"], 0.1)
            tampered = copy.deepcopy(receipt)
            tampered["preceding_observation_interval_s"] = 0.2
            tampered = analysis.sign_document(tampered)
            with self.assertRaisesRegex(analysis.AnalysisContractError, "does not reproduce"):
                analysis.validate_history_receipt(
                    tampered,
                    receipt_path=root / "history.json",
                    request=request,
                    adapter=adapter,
                )

    def test_reference_rejects_leaf_symlink_before_resolution(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            target = root / "target.json"
            target.write_text("{}\n", encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(analysis.AnalysisContractError, "symlink"):
                analysis.resolve_reference(
                    root,
                    {"path": "link.json", "sha256": analysis.sha256_file(target)},
                    "test reference",
                )


if __name__ == "__main__":
    unittest.main()
