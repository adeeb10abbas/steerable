import copy
import contextlib
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


MODULE = Path(__file__).resolve().parents[1] / "analysis" / "compile_forecast_publication.py"
SPEC = importlib.util.spec_from_file_location("compile_forecast_publication", MODULE)
publication = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(publication)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def descriptor(path, public_path=None):
    value = publication.file_descriptor(path)
    if public_path is not None:
        value["path"] = public_path
    return value


def metric(value=0.25):
    rows = [
        {"layout_pair_id": layout, "value": value + index / 1000.0}
        for index, layout in enumerate(publication.LAYOUTS)
    ]
    return {
        "estimate": value,
        "ci95": [value - 0.05, value + 0.05],
        "layout_pairs": 24,
        "resamples": 10_000,
        "seed": 2026091301,
        "resampling_unit": "complete independent base-layout pair with four condition means nested",
        "layout_pair_values": rows,
        "complete_layout_ids": list(publication.LAYOUTS),
        "excluded_incomplete_layout_ids": [],
        "conditional_on_four_condition_observability": True,
    }


class FakeAnalyzer:
    OUTPUT_SCHEMA = publication.FINAL_ANALYSIS_SCHEMA

    def __init__(self, result):
        self.result = result
        self.calls = []

    def analyze_manifest(self, path):
        self.calls.append(Path(path))
        return copy.deepcopy(self.result)


class FakeCompiler:
    COMPILER_SCHEMA = publication.CONFIRMATION_COMPILER_SCHEMA

    def __init__(self):
        self.calls = []

    def _validate_compiled_bundle(self, path):
        self.calls.append(Path(path))


class FakeFixtureValidator:
    def __init__(self, fixture):
        self.fixture = fixture
        self.calls = []

    def validate_fixture_freeze(
        self,
        path,
        expected_sha256,
        *,
        source_root,
        expected_layout_pair_id,
        expected_study_commit,
        deep_validate_selected,
    ):
        self.calls.append(expected_layout_pair_id)
        identity = publication.file_descriptor(path)
        if identity["sha256"] != expected_sha256:
            raise ValueError("fixture hash mismatch")
        selected = next(
            row for row in self.fixture["layouts"]
            if row["layout_pair_id"] == expected_layout_pair_id
        )
        return {
            "fixture_freeze": identity,
            "selected_layout": copy.deepcopy(selected),
            "created_from_study_commit": expected_study_commit,
        }


class PublicationFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.trusted_repo = self.root / "trusted_source"
        self.science_sources = {
            "compiler_source": self.trusted_repo / "workshops/corl2026_world_models/analysis/compile_confirmation_evidence.py",
            "analyzer_source": self.trusted_repo / "workshops/corl2026_world_models/analysis/forecast_evidence_analysis.py",
            "fixture_source": self.trusted_repo / "workshops/corl2026_world_models/experiments/forecast_layout/confirmation_fixture_freeze.py",
        }
        for label, path in self.science_sources.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# synthetic trusted {label}\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=self.trusted_repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=self.trusted_repo, check=True)
        subprocess.run(["git", "config", "user.name", "Synthetic Test"], cwd=self.trusted_repo, check=True)
        subprocess.run(["git", "add", "."], cwd=self.trusted_repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "trusted science source"], cwd=self.trusted_repo, check=True)
        subprocess.run(
            ["git", "branch", "-M", publication.AUTHORIZED_PUBLICATION_BRANCH],
            cwd=self.trusted_repo,
            check=True,
        )
        subprocess.run(
            ["git", "remote", "add", "publish", publication.AUTHORIZED_REPOSITORY_URL],
            cwd=self.trusted_repo,
            check=True,
        )
        self.study_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.trusted_repo, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        self.compiled = self.root / "compiled"
        self.compiled.mkdir()
        self.source_video = self.root / "private_source.mp4"
        self.source_video.write_bytes(b"\x00\x00\x00\x18ftypmp42synthetic-mp4")
        self.analysis_manifest = self.root / "analysis_evidence.json"
        write_json(self.analysis_manifest, {"synthetic": "hash-bound analyzer input"})
        self.request_selection = self.root / "request_selection.json"
        self.request_inventory = self.compiled / "request_inventory.json"
        self.private_inventory_path = self.compiled / "private_video_inventory.json"
        self.fixture_path = self.root / "confirmation_fixture_freeze.json"
        self.pose_path = self.root / "all_pose_manifests.json"
        self.close_path = self.root / "cohort_close.json"
        self.aggregate_path = self.root / "aggregate.json"
        self.receipt_path = self.compiled / "compiler_receipt.json"
        self.analysis_path = self.root / "final_analysis.json"
        self.input_path = self.root / "publication_input.json"
        self._make_fixture()
        self._make_private_evidence()
        self._make_selection()
        self._make_analysis()
        self._make_compiler_receipt()
        self._make_input(copy_videos=False)
        self.fake_analyzer = FakeAnalyzer(self.analysis)
        self.fake_compiler = FakeCompiler()
        self.fake_fixture = FakeFixtureValidator(self.fixture)

    def tearDown(self):
        self.temporary.cleanup()

    def _make_fixture(self):
        pair_rows = {}
        freeze_rows = []
        for index, layout in enumerate(publication.LAYOUTS, start=1):
            candidate_hash = digest(f"candidate:{layout}")
            gate_hash = digest(f"gate:{layout}")
            original = {
                "positions_robot_base_m": {
                    "banana": [0.35, 0.12, 0.02],
                    "bowl": [0.48, 0.16, 0.03],
                    "rubiks_cube": [0.42, -0.11, 0.04],
                },
                "quaternions_wxyz": {
                    "banana": [1.0, 0.0, 0.0, 0.0],
                    "bowl": [1.0, 0.0, 0.0, 0.0],
                    "rubiks_cube": [1.0, 0.0, 0.0, 0.0],
                },
            }
            reflected = copy.deepcopy(original)
            for position in reflected["positions_robot_base_m"].values():
                position[1] *= -1
            pair_rows[layout] = {
                "layout_pair_id": layout,
                "candidate_payload_sha256": candidate_hash,
                "accepted_gate_record_sha256": gate_hash,
                "layouts": {"original": original, "reflected": reflected},
            }
        write_json(self.pose_path, {"layout_pairs": pair_rows})
        pose_descriptor = descriptor(self.pose_path)
        for index, layout in enumerate(publication.LAYOUTS, start=1):
            freeze_rows.append(
                {
                    "layout_pair_id": layout,
                    "environment_seed": 2026091200 + index,
                    "candidate_id": f"{layout}__candidate_00",
                    "candidate_payload_sha256": digest(f"candidate:{layout}"),
                    "accepted_gate_record_sha256": digest(f"gate:{layout}"),
                    "pose_manifest": pose_descriptor,
                }
            )
        self.fixture = publication.sign_document(
            {
                "schema_version": publication.FIXTURE_FREEZE_SCHEMA,
                "study_id": publication.STUDY_ID,
                "namespace": "wmf_ablation_001_20260912",
                "status": "frozen_for_confirmation",
                "created_from_study_commit": self.study_commit,
                "selection_uses_target_model_outcomes": False,
                "candidate_selection_rule": "first passing model-blind physical candidate",
                "layout_count": 24,
                "layouts": freeze_rows,
                "model_request_count": 0,
                "behavioral_action_count": 0,
                "claim_boundary": "synthetic fixture evidence",
            }
        )
        write_json(self.fixture_path, self.fixture)

    def _cell_id(self, layout, condition):
        return f"n3__{layout.lower()}__{condition}"

    def _make_private_evidence(self):
        video = descriptor(self.source_video)
        self.video_rows = []
        self.roster_rows = []
        self.source_rows = []
        for layout in publication.LAYOUTS:
            for condition in publication.CONDITIONS:
                cell = self._cell_id(layout, condition)
                video_id = f"video__{layout.lower()}__{condition}"
                self.video_rows.append(
                    {
                        "cell_id": cell,
                        "recording_id": f"recording__{cell}",
                        "model_id": "N3",
                        "layout_pair_id": layout,
                        "condition_id": condition,
                        "recording_status": "valid_complete",
                        "private_not_for_blind_raters": True,
                        "source_video_id": video_id,
                        "source_video": video,
                    }
                )
                self.roster_rows.append(
                    {
                        "cell_id": cell,
                        "recording_id": f"recording__{cell}",
                        "model_id": "N3",
                        "layout_pair_id": layout,
                        "condition_id": condition,
                        "recording_status": "valid_complete",
                        "executed_action_count": 450,
                        "source_video_id": video_id,
                        "source_video_sha256": video["sha256"],
                    }
                )
                self.source_rows.append(
                    {
                        "cell_id": cell,
                        "model_id": "N3",
                        "layout_pair_id": layout,
                        "condition_id": condition,
                        "recording_status": "valid_complete",
                        "source_video": video,
                    }
                )
        self.private_inventory = publication.sign_document(
            {
                "schema_version": publication.PRIVATE_VIDEO_SCHEMA,
                "study_id": publication.STUDY_ID,
                "stage": "confirmation",
                "cohort_branch": "reduced_n3",
                "visibility": "private_source_evidence_not_blind_annotation_media",
                "videos": self.video_rows,
            }
        )
        write_json(self.private_inventory_path, self.private_inventory)
        write_json(
            self.request_inventory,
            {
                "schema_version": "wmf-forecast-request-inventory-v1",
                "study_id": publication.STUDY_ID,
                "stage": "confirmation",
                "cohort_branch": "reduced_n3",
                "episode_roster": self.roster_rows,
                "requests": [],
            },
        )

    def _make_selection(self):
        self.selection = publication.sign_document(
            {
                "schema_version": "wmf-forecast-request-selection-v1",
                "study_id": publication.STUDY_ID,
                "stage": "confirmation",
                "cohort_branch": "reduced_n3",
                "selection_uses_object_visibility_or_forecast_quality": False,
                "provenance": {
                    "request_inventory_sha256": descriptor(self.request_inventory)["sha256"]
                },
            }
        )
        write_json(self.request_selection, self.selection)

    def _declared_examples(self):
        result = {}
        for condition in publication.CONDITIONS:
            candidates = [
                row for row in self.video_rows
                if row["condition_id"] == condition
                and row["recording_status"] == "valid_complete"
            ]
            winner = min(
                candidates,
                key=lambda row: (
                    publication._selection_rank("N3", condition, row["cell_id"]),
                    row["cell_id"],
                ),
            )
            result[condition] = {
                "cell_id": winner["cell_id"],
                "source_video_id": winner["source_video_id"],
                "source_video_sha256": winner["source_video"]["sha256"],
                "recording_receipt_sha256": digest("recording:" + winner["cell_id"]),
                "selection_rank_sha256": publication._selection_rank(
                    "N3", condition, winner["cell_id"]
                ),
            }
        return result

    def _model_analysis(self):
        coverage = {}
        for condition in publication.CONDITIONS:
            coverage[condition] = {
                "planned_cells": 24,
                "recording_status_counts": {
                    "valid_complete": 24,
                    "valid_censored": 0,
                    "technical_invalid": 0,
                    "not_run": 0,
                },
                "censor_reason_counts": {},
                "request_inventory_count": 24,
                "timing_camera_action_eligible_requests": 24,
                "timing_camera_action_ineligibility_reason_counts": {},
                "zero_eligible_episodes": 0,
                "selected_requests": 24,
                "primary_observable_requests": 24,
                "constant_velocity_observable_requests": 24,
                "early_horizon_observable_requests": 0,
                "unresolved_primary_role_counts": {},
                "unresolved_generated_prediction_requests": 0,
                "nontrivial_ambiguity_code_counts_by_role": {},
            }
        reflection_metric = {
            "estimate": 0.02,
            "ci95": [-0.01, 0.05],
            "layout_pairs": 24,
            "resamples": 10_000,
            "seed": 2026091301,
            "paired_layout_values": [
                {"layout_pair_id": layout, "discrepancy": 0.02}
                for layout in publication.LAYOUTS
            ],
        }
        return {
            "branch_status": "qualified_and_included",
            "baseline_errors_and_skill": {
                "forecast_error": metric(0.15),
                "persistence_error": metric(0.40),
                "constant_velocity_error": metric(0.30),
                "forecast_skill_vs_persistence": metric(0.25),
                "forecast_skill_vs_constant_velocity": metric(0.15),
            },
            "coverage_by_condition": coverage,
            "full_design_strict_win_missingness_bounds": {
                "lower": 0.70,
                "upper": 0.80,
                "strict_win_definition": "forecast_error < persistence_error",
                "missingness_rule": "denominator preserving",
                "weighting": "layout",
                "by_condition": {
                    condition: {"lower": 0.7, "upper": 0.8, "planned_cells": 24}
                    for condition in publication.CONDITIONS
                },
                "layout_bounds": [
                    {"layout_pair_id": layout, "lower": 0.7, "upper": 0.8}
                    for layout in publication.LAYOUTS
                ],
            },
            "reflection_contrasts": {
                "definition": "paired layout effect",
                "by_command": {
                    command: {
                        "actual_reflected_minus_original_motion": reflection_metric,
                        "predicted_reflected_minus_original_motion": reflection_metric,
                        "predicted_minus_actual_contrast_discrepancy": reflection_metric,
                        "eligible_layout_ids": list(publication.LAYOUTS),
                        "paired_layout_values": reflection_metric["paired_layout_values"],
                    }
                    for command in ("left", "right")
                },
                "combined_command_discrepancy": reflection_metric,
            },
            "stopping_control": {"censored_cells_excluded_without_carry_forward": 0},
            "movement_strata": {
                "threshold_relative_image_diagonal": 0.01,
                "stationary": metric(0.2),
                "moving": metric(0.3),
            },
            "movement_decomposition": {
                "actual_relative_motion": metric(0.1),
                "predicted_relative_motion": metric(0.11),
            },
            "earlier_horizon": {
                "status": "unsupported_no_earlier_qualified_exposed_target",
                "qualified_target": None,
                "observable_requests": 0,
            },
            "declared_rule_examples": {
                "declared_rule": (
                    "For each model and condition use metadata hash. No label, error, "
                    "success, or visual outcome enters the rule."
                ),
                "videos": self._declared_examples(),
            },
        }

    def _make_analysis(self):
        self.analysis = publication.sign_document(
            {
                "schema_version": publication.FINAL_ANALYSIS_SCHEMA,
                "study_id": publication.STUDY_ID,
                "stage": "confirmation",
                "cohort_branch": "reduced_n3",
                "study_scope": (
                    "REDUCED_ONE_MODEL_N3_BRANCH; the other primary model is unqualified "
                    "and its 96 confirmation cells were not substituted"
                ),
                "analysis_contract": {
                    "primary": "persistence minus forecast error",
                    "bootstrap_resamples": 10_000,
                    "analysis_seed": 2026091301,
                    "confidence_level": 0.95,
                    "models_pooled": False,
                },
                "model_sample_size_table": [
                    {
                        "model_id": "N3",
                        "branch_status": "qualified_and_included",
                        "checkpoint_revision": "n3-checkpoint",
                        "executed_prefix_cap": 32,
                        "primary_horizon_s": 0.5,
                        "generated_frame_index": 3,
                        "target_executed_action_offset": 16,
                        "camera_id": "over_shoulder_left_camera",
                        "timestamp_tolerance_s": 0.02,
                        "full_design_planned_confirmation_cells": 96,
                        "scientific_roster_cells": 96,
                        "valid_complete": 96,
                        "valid_censored": 0,
                        "technical_invalid": 0,
                        "unrun": 0,
                        "selected_requests": 96,
                        "primary_observable_requests": 96,
                        "continuous_complete_layout_pairs": 24,
                    },
                    {
                        "model_id": "D1",
                        "branch_status": "unqualified_branch_not_run",
                        "checkpoint_revision": "d1-checkpoint",
                        "executed_prefix_cap": 8,
                        "primary_horizon_s": None,
                        "generated_frame_index": None,
                        "target_executed_action_offset": None,
                        "camera_id": None,
                        "timestamp_tolerance_s": None,
                        "full_design_planned_confirmation_cells": 96,
                        "scientific_roster_cells": 0,
                        "valid_complete": 0,
                        "valid_censored": 0,
                        "technical_invalid": 0,
                        "unrun_due_unqualified_branch": 96,
                        "selected_requests": 0,
                        "primary_observable_requests": 0,
                        "continuous_complete_layout_pairs": 0,
                    },
                ],
                "frozen_resource_budget": {},
                "annotation_quality": {
                    "unit": "distinct_blinded_annotation_images",
                    "images": 288,
                    "first_pass_exact_agreements": 250,
                    "independently_adjudicated": 38,
                    "first_pass_exact_agreement_rate": 250 / 288,
                    "independent_adjudication_rate": 38 / 288,
                    "decision_inventory_sha256": digest("decisions"),
                    "final_consensus_sha256": digest("consensus"),
                },
                "models": {"N3": self._model_analysis()},
                "claim_boundaries": {
                    "forecast_accuracy_claim_gate": publication.CLAIM_GATE,
                    "either_sign_reported": True,
                    "between_model_accuracy_ranking_allowed": False,
                    "between_model_reason": "separate within-model reports",
                    "causal_world_model_benefit_claim_allowed": False,
                    "behavioral_success_from_image_centroids_allowed": False,
                    "technical_invalid_is_model_failure": False,
                    "censored_last_state_carried_to_action_450": False,
                },
                "request_level_audit": [],
                "cell_level_audit": [],
                "source_evidence": {
                    "annotation_freeze": {"path": "/restricted/annotation.json", "sha256": digest("annotation")},
                    "final_consensus": {"path": "/restricted/consensus.json", "sha256": digest("consensus")},
                    "request_selection": {
                        "path": str(self.request_selection.resolve()),
                        "sha256": descriptor(self.request_selection)["sha256"],
                    },
                },
                "receipt_evidence": {},
                "analysis_evidence_manifest": {
                    "path": str(self.analysis_manifest.resolve()),
                    "sha256": descriptor(self.analysis_manifest)["sha256"],
                },
            }
        )
        write_json(self.analysis_path, self.analysis)

    def _make_compiler_receipt(self):
        fixture_descriptor = descriptor(self.fixture_path)
        write_json(
            self.aggregate_path,
            {"prerequisites": {"confirmation_fixture_freeze": fixture_descriptor}},
        )
        aggregate_descriptor = descriptor(self.aggregate_path)
        close = {
            "blocks": [
                {
                    "model_id": "N3",
                    "layout_pair_id": layout,
                    "receipt": aggregate_descriptor,
                }
                for layout in publication.LAYOUTS
            ]
        }
        write_json(self.close_path, close)
        self.compiler_receipt = publication.sign_document(
            {
                "schema_version": publication.CONFIRMATION_COMPILER_SCHEMA,
                "study_id": publication.STUDY_ID,
                "status": "compiled_complete_roster",
                "stage": "confirmation",
                "cohort_branch": "reduced_n3",
                "study_commit": self.study_commit,
                "compiler_source": descriptor(self.science_sources["compiler_source"]),
                "final_analyzer_dependency": descriptor(self.science_sources["analyzer_source"]),
                "fixture_freeze_dependency": descriptor(self.science_sources["fixture_source"]),
                "cohort_close_receipt": descriptor(self.close_path),
                "source_root": str(self.trusted_repo.resolve()),
                "counts": {
                    "planned_cells": 96,
                    "valid_complete": 96,
                    "valid_censored": 0,
                    "technical_invalid": 0,
                    "not_run": 0,
                },
                "outputs": {
                    "request_inventory": descriptor(
                        self.request_inventory, "request_inventory.json"
                    ),
                    "private_video_inventory": descriptor(
                        self.private_inventory_path, "private_video_inventory.json"
                    ),
                },
                "source_confirmation_artifacts": self.source_rows,
                "safe_for_request_selection": True,
                "safe_for_analysis_manifest_assembly": False,
                "labels_created": False,
                "scientific_results_computed": False,
                "confirmation_released": False,
            }
        )
        write_json(self.receipt_path, self.compiler_receipt)

    def _make_input(self, *, copy_videos):
        self.input_manifest = publication.sign_document(
            {
                "schema_version": publication.INPUT_SCHEMA,
                "study_id": publication.STUDY_ID,
                "cohort_branch": "reduced_n3",
                "final_analysis": descriptor(self.analysis_path),
                "confirmation_fixture_freeze": descriptor(self.fixture_path),
                "confirmation_compiler_receipt": descriptor(self.receipt_path),
                "private_video_inventory": descriptor(self.private_inventory_path),
                "analyzer_source": descriptor(self.science_sources["analyzer_source"]),
                "publication_source_commit": self.study_commit,
                "copy_selected_videos": copy_videos,
            }
        )
        write_json(self.input_path, self.input_manifest)

    def _refresh_analysis(self):
        unsigned = dict(self.analysis)
        unsigned.pop("payload_sha256", None)
        self.analysis = publication.sign_document(unsigned)
        write_json(self.analysis_path, self.analysis)
        self._make_input(copy_videos=self.input_manifest["copy_selected_videos"])
        self.fake_analyzer.result = copy.deepcopy(self.analysis)

    def _refresh_private_and_receipt(self):
        unsigned = dict(self.private_inventory)
        unsigned.pop("payload_sha256", None)
        self.private_inventory = publication.sign_document(unsigned)
        write_json(self.private_inventory_path, self.private_inventory)
        receipt = dict(self.compiler_receipt)
        receipt.pop("payload_sha256", None)
        receipt["outputs"] = dict(receipt["outputs"])
        receipt["outputs"]["request_inventory"] = descriptor(
            self.request_inventory, "request_inventory.json"
        )
        receipt["outputs"]["private_video_inventory"] = descriptor(
            self.private_inventory_path, "private_video_inventory.json"
        )
        receipt["source_confirmation_artifacts"] = self.source_rows
        self.compiler_receipt = publication.sign_document(receipt)
        write_json(self.receipt_path, self.compiler_receipt)
        self._make_input(copy_videos=self.input_manifest["copy_selected_videos"])

    def _patch_validators(self, *, analyzer=None):
        stack = contextlib.ExitStack()

        @contextlib.contextmanager
        def authenticated_graph(publication_commit, study_commit, *, repository=None):
            self.assertEqual(publication_commit, self.study_commit)
            self.assertEqual(study_commit, self.study_commit)
            yield (
                {
                    "repository_url": publication.AUTHORIZED_REPOSITORY_URL,
                    "control_branch": publication.AUTHORIZED_PUBLICATION_BRANCH,
                    "control_ref": publication.AUTHORIZED_PUBLICATION_REF,
                    "remote_ref_commit": self.study_commit,
                    "read_transport": "literal_public_https_isolated_bare",
                    "git_executable_path": "/usr/bin/git",
                    "git_executable_resolved_path": "/usr/bin/git",
                    "git_version": "git version 2.34.1",
                    "blob_filter_limit_bytes": publication.AUTHENTICATED_BLOB_LIMIT_BYTES,
                    "authenticated_graph_bytes": 4096,
                    "authenticated_graph_max_bytes": publication.MAX_AUTHENTICATED_GRAPH_BYTES,
                    "ls_remote_before_and_after_fetch_match": True,
                    "study_to_publication_to_control_ancestry": True,
                    "local_head_commit": self.study_commit,
                    "checkout_mode": "attached_control_branch",
                    "deployment_remote_alias": "publish",
                },
                self.trusted_repo,
            )

        stack.enter_context(
            mock.patch.object(publication, "_load_analyzer", return_value=analyzer or self.fake_analyzer)
        )
        stack.enter_context(
            mock.patch.object(publication, "_load_confirmation_compiler", return_value=self.fake_compiler)
        )
        stack.enter_context(
            mock.patch.object(publication, "_load_fixture_validator", return_value=self.fake_fixture)
        )
        stack.enter_context(
            mock.patch.object(
                publication,
                "_validate_committed_publication_sources",
                return_value={
                    "publication_compiler_source": publication.file_descriptor(
                        MODULE,
                        public_path="workshops/corl2026_world_models/analysis/compile_forecast_publication.py",
                    ),
                    "publication_contract": publication.file_descriptor(
                        publication.CONTRACT_PATH,
                        public_path="workshops/corl2026_world_models/experiments/forecast_layout/forecast_publication_contract.json",
                    ),
                },
            )
        )
        stack.enter_context(mock.patch.object(publication, "REPOSITORY", self.trusted_repo))
        stack.enter_context(
            mock.patch.object(
                publication,
                "_authenticated_publication_source_graph",
                side_effect=authenticated_graph,
            )
        )
        return stack

    def _make_bare_control_remote(self):
        remote_work = self.root / "remote_work"
        bare_remote = self.root / "authorized.git"
        subprocess.run(
            ["git", "clone", "-q", str(self.trusted_repo), str(remote_work)],
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.invalid"],
            cwd=remote_work,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Synthetic Test"],
            cwd=remote_work,
            check=True,
        )
        descriptor_file = remote_work / "released-descriptor.txt"
        descriptor_file.write_text("descriptor R for source P\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "released-descriptor.txt"], cwd=remote_work, check=True
        )
        subprocess.run(
            ["git", "commit", "-q", "-m", "release descriptor"],
            cwd=remote_work,
            check=True,
        )
        subprocess.run(
            ["git", "clone", "-q", "--bare", str(remote_work), str(bare_remote)],
            check=True,
        )
        subprocess.run(
            ["git", "config", "uploadpack.allowFilter", "true"],
            cwd=bare_remote,
            check=True,
        )
        remote_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=remote_work, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        return bare_remote, remote_head

    def _configure_technical_null_video(self, action_count):
        declared_cells = {
            item["cell_id"]
            for item in self.analysis["models"]["N3"]["declared_rule_examples"]["videos"].values()
        }
        index = next(
            index for index, row in enumerate(self.video_rows)
            if row["cell_id"] not in declared_cells
        )
        cell = self.video_rows[index]["cell_id"]
        condition = self.video_rows[index]["condition_id"]
        self.video_rows[index]["recording_status"] = "technical_invalid"
        self.video_rows[index]["source_video_id"] = None
        self.video_rows[index]["source_video"] = None
        self.roster_rows[index]["recording_status"] = "technical_invalid"
        self.roster_rows[index]["executed_action_count"] = action_count
        self.roster_rows[index]["source_video_id"] = None
        self.roster_rows[index]["source_video_sha256"] = None
        self.source_rows[index]["recording_status"] = "technical_invalid"
        self.source_rows[index]["source_video"] = None
        self.source_rows[index]["adapter_completion"] = {
            "sha256": digest(f"completion:{cell}:{action_count}")
        }
        self.source_rows[index]["adapter_journal"] = {
            "sha256": digest(f"journal:{cell}:{action_count}")
        }
        write_json(
            self.request_inventory,
            {
                "schema_version": "wmf-forecast-request-inventory-v1",
                "study_id": publication.STUDY_ID,
                "stage": "confirmation",
                "cohort_branch": "reduced_n3",
                "episode_roster": self.roster_rows,
                "requests": [],
            },
        )
        self.private_inventory["videos"] = self.video_rows
        row = self.analysis["model_sample_size_table"][0]
        row["valid_complete"], row["technical_invalid"] = 95, 1
        coverage = self.analysis["models"]["N3"]["coverage_by_condition"][condition]
        coverage["recording_status_counts"]["valid_complete"] = 23
        coverage["recording_status_counts"]["technical_invalid"] = 1
        self.compiler_receipt["counts"]["valid_complete"] = 95
        self.compiler_receipt["counts"]["technical_invalid"] = 1
        self._make_selection()
        self.analysis["source_evidence"]["request_selection"] = {
            "path": str(self.request_selection.resolve()),
            "sha256": descriptor(self.request_selection)["sha256"],
        }
        self._refresh_analysis()
        self._refresh_private_and_receipt()
        return cell

    def test_manifest_only_build_is_signed_separate_model_and_deep_replayed(self):
        output = self.root / "publication"
        with self._patch_validators():
            receipt = publication.compile_publication(self.input_path, output)
        self.assertTrue(output.is_dir())
        self.assertEqual(self.fake_analyzer.calls, [self.analysis_manifest.resolve()])
        self.assertEqual(self.fake_compiler.calls, [self.compiled.resolve()])
        self.assertEqual(self.fake_fixture.calls, list(publication.LAYOUTS))
        self.assertEqual(receipt["model_ids"], ["N3"])
        self.assertFalse(receipt["models_pooled"])
        self.assertEqual(receipt["selected_example_count"], 4)
        self.assertEqual(receipt["selected_video_copy_count"], 0)
        remote = receipt["authorized_publication_remote"]
        self.assertEqual(remote["repository_url"], publication.AUTHORIZED_REPOSITORY_URL)
        self.assertEqual(remote["remote_ref_commit"], self.study_commit)
        self.assertEqual(remote["local_head_commit"], self.study_commit)
        self.assertEqual(remote["read_transport"], "literal_public_https_isolated_bare")
        self.assertEqual(remote["deployment_remote_alias"], "publish")
        self.assertFalse((output / "videos").exists())
        paper = publication.load_json(output / "paper_evidence.json", "paper")
        publication.verify_signed(paper, "paper")
        self.assertEqual(set(paper["models"]), {"N3"})
        self.assertFalse(paper["paper_generation_performed"])
        self.assertNotIn(str(self.root), (output / "paper_evidence.json").read_text())
        declared = set(receipt["output_inventory"])
        self.assertEqual(declared, set(publication.STATIC_OUTPUTS))
        self.assertEqual(receipt["output_count_including_self_signed_receipt"], 9)

    def test_optional_video_copies_are_exact_and_fully_bound(self):
        self._make_input(copy_videos=True)
        output = self.root / "publication"
        with self._patch_validators():
            receipt = publication.compile_publication(self.input_path, output)
        self.assertEqual(receipt["selected_video_copy_count"], 4)
        self.assertEqual(len(list((output / "videos").glob("*.mp4"))), 4)
        for row in receipt["selected_video_copies"]:
            target = output / row["path"]
            self.assertEqual(target.read_bytes(), self.source_video.read_bytes())
            self.assertEqual(publication.file_descriptor(target)["sha256"], row["sha256"])
            self.assertEqual(receipt["output_inventory"][row["path"]]["sha256"], row["sha256"])

    def test_null_primary_estimate_fails_without_output(self):
        primary = self.analysis["models"]["N3"]["baseline_errors_and_skill"]["forecast_skill_vs_persistence"]
        primary.update({"estimate": None, "ci95": None, "layout_pairs": 0})
        self._refresh_analysis()
        output = self.root / "publication"
        with self._patch_validators(), self.assertRaisesRegex(
            publication.PublicationContractError, "required primary estimate is null"
        ):
            publication.compile_publication(self.input_path, output)
        self.assertFalse(output.exists())

    def test_model_pooling_or_wrong_status_count_fails(self):
        self.analysis["analysis_contract"]["models_pooled"] = True
        self._refresh_analysis()
        with self._patch_validators(), self.assertRaisesRegex(
            publication.PublicationContractError, "pools models"
        ):
            publication.compile_publication(self.input_path, self.root / "pooled")
        self.analysis["analysis_contract"]["models_pooled"] = False
        self.analysis["model_sample_size_table"][0]["valid_complete"] = 95
        self._refresh_analysis()
        with self._patch_validators(), self.assertRaisesRegex(
            publication.PublicationContractError, "96-cell cohort"
        ):
            publication.compile_publication(self.input_path, self.root / "counts")

    def test_analysis_replay_mismatch_and_missing_claim_gate_fail(self):
        different = copy.deepcopy(self.analysis)
        different["study_scope"] = "different"
        with self._patch_validators(analyzer=FakeAnalyzer(different)), self.assertRaisesRegex(
            publication.PublicationContractError, "does not exactly reproduce"
        ):
            publication.compile_publication(self.input_path, self.root / "replay")
        self.analysis["claim_boundaries"]["forecast_accuracy_claim_gate"] = "NOT_PASSED"
        self._refresh_analysis()
        with self._patch_validators(), self.assertRaisesRegex(
            publication.PublicationContractError, "claim gate"
        ):
            publication.compile_publication(self.input_path, self.root / "gate")

    def test_declared_video_cannot_be_replaced_using_an_outcome(self):
        condition = publication.CONDITIONS[0]
        declared = self.analysis["models"]["N3"]["declared_rule_examples"]["videos"][condition]
        alternative = next(
            row for row in self.video_rows
            if row["condition_id"] == condition and row["cell_id"] != declared["cell_id"]
        )
        declared.update(
            {
                "cell_id": alternative["cell_id"],
                "source_video_id": alternative["source_video_id"],
                "source_video_sha256": alternative["source_video"]["sha256"],
                "selection_rank_sha256": publication._selection_rank(
                    "N3", condition, alternative["cell_id"]
                ),
            }
        )
        self._refresh_analysis()
        with self._patch_validators(), self.assertRaisesRegex(
            publication.PublicationContractError, "outcome-blind rule"
        ):
            publication.compile_publication(self.input_path, self.root / "chosen")

    def test_zero_launch_technical_null_video_is_allowed(self):
        cell = self._configure_technical_null_video(0)
        output = self.root / "technical"
        with self._patch_validators():
            receipt = publication.compile_publication(self.input_path, output)
        self.assertEqual(receipt["selected_example_count"], 4)
        self.assertTrue(output.exists())
        self.assertNotIn(cell, (output / "example_videos.json").read_text())

    def test_nonzero_prefix_technical_null_video_is_allowed(self):
        cell = self._configure_technical_null_video(137)
        output = self.root / "technical-prefix"
        with self._patch_validators():
            receipt = publication.compile_publication(self.input_path, output)
        self.assertEqual(receipt["selected_example_count"], 4)
        self.assertTrue(output.exists())
        self.assertNotIn(cell, (output / "example_videos.json").read_text())

    def test_valid_or_censored_null_video_is_rejected(self):
        index = 0
        self.video_rows[index]["source_video_id"] = None
        self.video_rows[index]["source_video"] = None
        self.roster_rows[index]["source_video_id"] = None
        self.roster_rows[index]["source_video_sha256"] = None
        write_json(self.request_inventory, {"episode_roster": self.roster_rows, "requests": []})
        self.private_inventory["videos"] = self.video_rows
        self._refresh_private_and_receipt()
        with self._patch_validators(), self.assertRaisesRegex(
            publication.PublicationContractError, "source video ID is missing"
        ):
            publication.compile_publication(self.input_path, self.root / "null-valid")

    def test_censored_null_video_is_rejected(self):
        index = 0
        self.video_rows[index]["recording_status"] = "valid_censored"
        self.video_rows[index]["source_video_id"] = None
        self.video_rows[index]["source_video"] = None
        self.roster_rows[index]["recording_status"] = "valid_censored"
        self.roster_rows[index]["source_video_id"] = None
        self.roster_rows[index]["source_video_sha256"] = None
        write_json(self.request_inventory, {"episode_roster": self.roster_rows, "requests": []})
        self.private_inventory["videos"] = self.video_rows
        self._refresh_private_and_receipt()
        with self._patch_validators(), self.assertRaisesRegex(
            publication.PublicationContractError, "source video ID is missing"
        ):
            publication.compile_publication(self.input_path, self.root / "null-censored")

    def test_oversize_video_copy_fails_without_partial_output(self):
        self.source_video.write_bytes(b"x" * (publication.MAX_SELECTED_VIDEO_BYTES + 1))
        video = descriptor(self.source_video)
        for index, row in enumerate(self.video_rows):
            row["source_video"] = video
            self.roster_rows[index]["source_video_sha256"] = video["sha256"]
            self.source_rows[index]["source_video"] = video
        self.private_inventory["videos"] = self.video_rows
        for item in self.analysis["models"]["N3"]["declared_rule_examples"]["videos"].values():
            item["source_video_sha256"] = video["sha256"]
        write_json(
            self.request_inventory,
            {"episode_roster": self.roster_rows, "requests": []},
        )
        self._make_selection()
        self.analysis["source_evidence"]["request_selection"] = {
            "path": str(self.request_selection.resolve()),
            "sha256": descriptor(self.request_selection)["sha256"],
        }
        self._refresh_analysis()
        self._refresh_private_and_receipt()
        self._make_input(copy_videos=True)
        output = self.root / "oversize"
        with self._patch_validators(), self.assertRaisesRegex(
            publication.PublicationContractError, "per-file cap"
        ):
            publication.compile_publication(self.input_path, output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".oversize.tmp-*")), [])

    def test_total_video_copy_cap_fails_without_partial_subset(self):
        self._make_input(copy_videos=True)
        output = self.root / "total-cap"
        per_video = self.source_video.stat().st_size
        with mock.patch.object(
            publication, "MAX_SELECTED_VIDEO_TOTAL_BYTES", per_video * 3
        ), self._patch_validators(), self.assertRaisesRegex(
            publication.PublicationContractError, "total cap"
        ):
            publication.compile_publication(self.input_path, output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".total-cap.tmp-*")), [])

    def test_private_hash_drift_and_wrong_fixture_binding_fail(self):
        self.source_video.write_bytes(self.source_video.read_bytes() + b"drift")
        with self._patch_validators(), self.assertRaisesRegex(
            publication.PublicationContractError, "changed"
        ):
            publication.compile_publication(self.input_path, self.root / "drift")
        self.source_video.write_bytes(b"\x00\x00\x00\x18ftypmp42synthetic-mp4")
        write_json(
            self.aggregate_path,
            {"prerequisites": {"confirmation_fixture_freeze": descriptor(self.pose_path)}},
        )
        close = publication.load_json(self.close_path, "close")
        for row in close["blocks"]:
            row["receipt"] = descriptor(self.aggregate_path)
        write_json(self.close_path, close)
        receipt = dict(self.compiler_receipt)
        receipt.pop("payload_sha256", None)
        receipt["cohort_close_receipt"] = descriptor(self.close_path)
        self.compiler_receipt = publication.sign_document(receipt)
        write_json(self.receipt_path, self.compiler_receipt)
        self._make_input(copy_videos=False)
        with self._patch_validators(), self.assertRaisesRegex(
            publication.PublicationContractError, "different confirmation fixture"
        ):
            publication.compile_publication(self.input_path, self.root / "fixture")

    def test_embedded_private_path_is_rejected(self):
        with self.assertRaisesRegex(publication.PublicationContractError, "absolute private path"):
            publication._assert_no_private_paths(
                {"note": "retained at /data/users/ali/private.mp4"}, "synthetic"
            )
        with self.assertRaisesRegex(publication.PublicationContractError, "absolute private path"):
            publication._assert_no_private_paths(
                {"note": "copied from C:\\Users\\ali\\private.mp4"}, "synthetic"
            )

    def test_publication_sources_must_equal_isolated_remote_blobs(self):
        current = {
            MODULE.resolve().relative_to(publication.REPOSITORY.resolve()).as_posix(): MODULE.read_bytes(),
            publication.CONTRACT_PATH.resolve().relative_to(publication.REPOSITORY.resolve()).as_posix(): publication.CONTRACT_PATH.read_bytes(),
        }

        def exact_run(argv, **kwargs):
            if argv[-2:] == ["rev-parse", "HEAD"]:
                return SimpleNamespace(returncode=0, stdout=("c" * 40 + "\n").encode(), stderr=b"")
            relative = argv[-1].split(":", 1)[1]
            return SimpleNamespace(returncode=0, stdout=current[relative], stderr=b"")

        with mock.patch.object(publication.subprocess, "run", side_effect=exact_run):
            observed = publication._validate_committed_publication_sources(
                "c" * 40, trusted_repository=self.trusted_repo
            )
        self.assertEqual(observed["publication_compiler_source"]["sha256"], publication.sha256_file(MODULE))

        def dirty_run(argv, **kwargs):
            if argv[-2:] == ["rev-parse", "HEAD"]:
                return SimpleNamespace(returncode=0, stdout=("c" * 40 + "\n").encode(), stderr=b"")
            relative = argv[-1].split(":", 1)[1]
            payload = current[relative]
            if relative.endswith("forecast_publication_contract.json"):
                payload += b"dirty"
            return SimpleNamespace(returncode=0, stdout=payload, stderr=b"")

        with mock.patch.object(publication.subprocess, "run", side_effect=dirty_run), self.assertRaisesRegex(
            publication.PublicationContractError, "dirty or differs"
        ):
            publication._validate_committed_publication_sources(
                "c" * 40, trusted_repository=self.trusted_repo
            )

    def test_isolated_round_trip_uses_literal_url_filter_and_exact_ref(self):
        commit = "a" * 40
        calls = []

        def git_bytes(repository, arguments, label, *, protocol="https"):
            calls.append((list(arguments), protocol))
            if arguments[0] == "ls-remote":
                return f"{commit}\t{publication.AUTHORIZED_PUBLICATION_REF}\n".encode()
            if arguments[0] == "fetch":
                return b""
            if arguments == ["rev-parse", f"{publication.AUTHENTICATED_GRAPH_REF}^{{commit}}"]:
                return f"{commit}\n".encode()
            self.fail(f"unexpected Git operation: {arguments}")

        with mock.patch.object(publication, "_isolated_git_bytes", side_effect=git_bytes), mock.patch.object(
            publication, "_authenticated_graph_size", return_value=1024
        ), mock.patch.object(
            publication, "_validate_controlled_isolated_repository_config"
        ) as validate_config, mock.patch.object(
            publication, "_write_controlled_isolated_repository_config"
        ) as rewrite_config:
            observed = publication._remote_ref_round_trip_isolated(
                self.trusted_repo,
                remote_url=publication.AUTHORIZED_REPOSITORY_URL,
                expected_ref=publication.AUTHORIZED_PUBLICATION_REF,
                protocol="https",
            )
        self.assertEqual(observed["before_fetch"], commit)
        self.assertEqual(calls[0][0][3], publication.AUTHORIZED_REPOSITORY_URL)
        self.assertEqual(calls[0][0][4], publication.AUTHORIZED_PUBLICATION_REF)
        fetch = calls[1][0]
        self.assertIn(f"--filter=blob:limit={publication.AUTHENTICATED_BLOB_LIMIT_BYTES}", fetch)
        self.assertEqual(fetch[-2], publication.AUTHORIZED_REPOSITORY_URL)
        self.assertEqual(
            fetch[-1],
            f"+{publication.AUTHORIZED_PUBLICATION_REF}:{publication.AUTHENTICATED_GRAPH_REF}",
        )
        self.assertEqual(calls[-1][0], calls[0][0])
        self.assertTrue(all(protocol == "https" for _, protocol in calls))
        self.assertEqual(rewrite_config.call_count, 1)
        self.assertGreaterEqual(validate_config.call_count, 5)

    def test_production_wrapper_supplies_only_literal_public_https(self):
        captured = {}
        subprocess.run(
            ["git", "config", "core.sshCommand", "/tmp/evil-ssh"],
            cwd=self.trusted_repo,
            check=True,
        )
        subprocess.run(
            ["git", "config", "credential.helper", "/tmp/evil-helper"],
            cwd=self.trusted_repo,
            check=True,
        )

        @contextlib.contextmanager
        def source_graph(**kwargs):
            captured.update(kwargs)
            yield (
                {
                    "repository_url": publication.AUTHORIZED_REPOSITORY_URL,
                    "remote_ref_commit": self.study_commit,
                },
                self.trusted_repo,
            )

        with mock.patch.object(
            publication, "_authenticated_source_graph_from_url", side_effect=source_graph
        ):
            with publication._authenticated_publication_source_graph(
                self.study_commit, self.study_commit, repository=self.trusted_repo
            ) as (observed, graph):
                self.assertEqual(graph, self.trusted_repo)
                self.assertEqual(observed["deployment_remote_alias"], "publish")
        self.assertEqual(captured["remote_url"], publication.AUTHORIZED_REPOSITORY_URL)
        self.assertEqual(captured["protocol"], "https")
        self.assertEqual(captured["publication_commit"], self.study_commit)

    def test_local_route_check_rejects_wrong_credential_and_redirect_urls(self):
        attacks = (
            "https://secret-token@github.com/adeeb10abbas/steerable.git",
            "https://github.com/another-owner/steerable.git",
        )
        for attack in attacks:
            with self.subTest(attack=attack):
                subprocess.run(
                    ["git", "remote", "set-url", "publish", attack],
                    cwd=self.trusted_repo,
                    check=True,
                )
                with self.assertRaises(publication.PublicationContractError) as raised:
                    publication._validate_local_publication_checkout(
                        self.study_commit, self.trusted_repo
                    )
                self.assertNotIn("secret-token", str(raised.exception))
            subprocess.run(
                ["git", "remote", "set-url", "publish", publication.AUTHORIZED_REPOSITORY_URL],
                cwd=self.trusted_repo,
                check=True,
            )
        redirected = "file:///tmp/redirected-secret/"
        subprocess.run(
            ["git", "config", f"url.{redirected}.insteadOf", "https://github.com/"],
            cwd=self.trusted_repo,
            check=True,
        )
        with self.assertRaises(publication.PublicationContractError) as raised:
            publication._validate_local_publication_checkout(
                self.study_commit, self.trusted_repo
            )
        self.assertNotIn("redirected-secret", str(raised.exception))

    def test_wrong_push_repository_is_rejected(self):
        subprocess.run(
            [
                "git", "remote", "set-url", "--push", "publish",
                "https://github.com/adeeb10abbas/not-steerable.git",
            ],
            cwd=self.trusted_repo,
            check=True,
        )
        with self.assertRaisesRegex(
            publication.PublicationContractError, "push does not identify"
        ):
            publication._validate_local_publication_checkout(
                self.study_commit, self.trusted_repo
            )

    def test_attached_wrong_branch_is_rejected_before_transport(self):
        subprocess.run(
            ["git", "branch", "-m", "wrong/publication-branch"],
            cwd=self.trusted_repo,
            check=True,
        )
        with self.assertRaisesRegex(
            publication.PublicationContractError, "attached to a different branch"
        ):
            publication._validate_local_publication_checkout(
                self.study_commit, self.trusted_repo
            )

    def test_detached_cluster_route_accepts_exact_ssh_origin(self):
        subprocess.run(["git", "remote", "remove", "publish"], cwd=self.trusted_repo, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "git@github.com:adeeb10abbas/steerable.git"],
            cwd=self.trusted_repo,
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "-q", "--detach", self.study_commit],
            cwd=self.trusted_repo,
            check=True,
        )
        observed = publication._validate_local_publication_checkout(
            self.study_commit, self.trusted_repo
        )
        self.assertEqual(observed["deployment_remote_alias"], "origin")
        self.assertEqual(observed["checkout_mode"], "detached_immutable_commit")

    def test_source_precedes_descriptor_in_real_isolated_fetched_graph(self):
        bare_remote, remote_head = self._make_bare_control_remote()
        graph_path = None
        with publication._authenticated_source_graph_from_url(
            publication_commit=self.study_commit,
            study_commit=self.study_commit,
            remote_url=bare_remote.resolve().as_uri(),
            protocol="file",
        ) as (observed, graph):
            graph_path = graph
            self.assertTrue(graph.is_dir())
            self.assertNotEqual(remote_head, self.study_commit)
            self.assertEqual(observed["remote_ref_commit"], remote_head)
            self.assertTrue(observed["study_to_publication_to_control_ancestry"])
            self.assertEqual(
                (graph / "config").read_bytes(),
                publication.ISOLATED_REPOSITORY_CONFIG,
            )
        self.assertIsNotNone(graph_path)
        self.assertFalse(graph_path.exists())

    def test_unpublished_or_divergent_source_fails_in_remote_fetched_graph(self):
        bare_remote, _ = self._make_bare_control_remote()
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=self.trusted_repo, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        divergent = subprocess.run(
            ["git", "commit-tree", tree], cwd=self.trusted_repo, check=True,
            input="unpublished divergent source\n", capture_output=True, text=True,
        ).stdout.strip()
        with self.assertRaisesRegex(
            publication.PublicationContractError,
            "publication source commit in isolated remote graph|publication-to-control ancestry",
        ):
            with publication._authenticated_source_graph_from_url(
                publication_commit=divergent,
                study_commit=self.study_commit,
                remote_url=bare_remote.resolve().as_uri(),
                protocol="file",
            ):
                self.fail("divergent source unexpectedly authenticated")

    def test_stale_or_racing_control_ref_is_rejected(self):
        observations = (
            {
                "ref": publication.AUTHORIZED_PUBLICATION_REF,
                "before_fetch": self.study_commit,
                "fetched": "d" * 40,
                "after_fetch": self.study_commit,
            },
            {
                "ref": publication.AUTHORIZED_PUBLICATION_REF,
                "before_fetch": self.study_commit,
                "fetched": self.study_commit,
                "after_fetch": "d" * 40,
            },
        )
        for observation in observations:
            with self.subTest(observation=observation), mock.patch.object(
                publication, "_remote_ref_round_trip_isolated", return_value=observation
            ), self.assertRaisesRegex(
                publication.PublicationContractError, "stale or changed"
            ):
                with publication._authenticated_source_graph_from_url(
                    publication_commit=self.study_commit,
                    study_commit=self.study_commit,
                    remote_url=publication.AUTHORIZED_REPOSITORY_URL,
                    protocol="https",
                ):
                    self.fail("racing control ref unexpectedly authenticated")

    def test_control_ref_change_before_atomic_rename_leaves_no_output(self):
        calls = 0

        @contextlib.contextmanager
        def changing_graph(publication_commit, study_commit, *, repository=None):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise publication.PublicationContractError(
                    "publication control ref was stale or changed during isolated verification"
                )
            yield (
                {
                    "repository_url": publication.AUTHORIZED_REPOSITORY_URL,
                    "control_branch": publication.AUTHORIZED_PUBLICATION_BRANCH,
                    "control_ref": publication.AUTHORIZED_PUBLICATION_REF,
                    "remote_ref_commit": self.study_commit,
                    "read_transport": "literal_public_https_isolated_bare",
                    "git_executable_path": "/usr/bin/git",
                    "git_executable_resolved_path": "/usr/bin/git",
                    "git_version": "git version 2.34.1",
                    "blob_filter_limit_bytes": publication.AUTHENTICATED_BLOB_LIMIT_BYTES,
                    "authenticated_graph_bytes": 4096,
                    "authenticated_graph_max_bytes": publication.MAX_AUTHENTICATED_GRAPH_BYTES,
                    "ls_remote_before_and_after_fetch_match": True,
                    "study_to_publication_to_control_ancestry": True,
                    "local_head_commit": self.study_commit,
                    "checkout_mode": "attached_control_branch",
                    "deployment_remote_alias": "publish",
                },
                self.trusted_repo,
            )

        output = self.root / "remote-raced-publication"
        with self._patch_validators(), mock.patch.object(
            publication, "_authenticated_publication_source_graph", side_effect=changing_graph
        ), self.assertRaisesRegex(
            publication.PublicationContractError, "stale or changed"
        ):
            publication.compile_publication(self.input_path, output)
        self.assertFalse(output.exists())

    def test_isolated_environment_scrubs_config_ssh_helpers_and_exec_path(self):
        dangerous = {
            "GIT_CONFIG_COUNT": "2",
            "GIT_CONFIG_KEY_0": "url.file:///tmp/evil.insteadOf",
            "GIT_CONFIG_VALUE_0": "https://github.com/",
            "GIT_CONFIG_PARAMETERS": "'credential.helper'='/tmp/evil-helper'",
            "GIT_EXEC_PATH": "/tmp/evil-exec",
            "GIT_SSH": "/tmp/evil-ssh",
            "GIT_SSH_COMMAND": "/tmp/evil-ssh -o StrictHostKeyChecking=no",
            "GIT_ASKPASS": "/tmp/evil-askpass",
            "SSH_ASKPASS": "/tmp/evil-askpass",
            "HOME": "/tmp/evil-home",
            "PATH": "/tmp/evil-exec",
            "HTTPS_PROXY": "http://proxy.invalid:8080",
            "SSL_CERT_FILE": "/tmp/test-ca.pem",
        }
        with mock.patch.dict(publication.os.environ, dangerous, clear=False):
            observed = publication._isolated_git_environment()
        self.assertEqual(observed["HTTPS_PROXY"], dangerous["HTTPS_PROXY"])
        self.assertEqual(observed["SSL_CERT_FILE"], dangerous["SSL_CERT_FILE"])
        for key in dangerous:
            if key not in publication.NETWORK_ENVIRONMENT_KEYS and key != "PATH":
                self.assertNotIn(key, observed)
        self.assertEqual(observed["PATH"], "/usr/bin:/bin")
        self.assertEqual(observed["GIT_CONFIG_SYSTEM"], publication.os.devnull)
        self.assertEqual(observed["GIT_CONFIG_GLOBAL"], publication.os.devnull)

    def test_controlled_isolated_config_rejects_helpers_rewrites_and_includes(self):
        graph = self.root / "controlled-config.git"
        graph.mkdir()
        (graph / "config").write_bytes(b"generated init config\n")
        publication._write_controlled_isolated_repository_config(graph)
        publication._validate_controlled_isolated_repository_config(graph)
        self.assertEqual(
            publication.sha256_bytes((graph / "config").read_bytes()),
            publication.ISOLATED_REPOSITORY_CONFIG_SHA256,
        )
        attacks = (
            b"[credential]\n\thelper = /tmp/evil-helper\n",
            b"[core]\n\tsshCommand = /tmp/evil-ssh\n",
            b"[url \"file:///tmp/redirect\"]\n\tinsteadOf = https://github.com/\n",
            b"[include]\n\tpath = /tmp/evil-config\n",
        )
        for attack in attacks:
            with self.subTest(attack=attack):
                (graph / "config").write_bytes(
                    publication.ISOLATED_REPOSITORY_CONFIG + attack
                )
                with self.assertRaisesRegex(
                    publication.PublicationContractError, "config drifted"
                ):
                    publication._validate_controlled_isolated_repository_config(graph)

    def test_isolated_graph_config_drift_during_validation_is_rejected(self):
        bare_remote, _ = self._make_bare_control_remote()
        with self.assertRaisesRegex(
            publication.PublicationContractError, "config drifted"
        ):
            with publication._authenticated_source_graph_from_url(
                publication_commit=self.study_commit,
                study_commit=self.study_commit,
                remote_url=bare_remote.resolve().as_uri(),
                protocol="file",
            ) as (_, graph):
                (graph / "config").write_bytes(
                    publication.ISOLATED_REPOSITORY_CONFIG
                    + b"[credential]\n\thelper = /tmp/evil-helper\n"
                )

    def test_isolated_graph_byte_cap_fails_closed(self):
        graph = self.root / "oversize-graph"
        graph.mkdir()
        (graph / "pack").write_bytes(b"12")
        with mock.patch.object(publication, "MAX_AUTHENTICATED_GRAPH_BYTES", 1), self.assertRaisesRegex(
            publication.PublicationContractError, "exceeds its byte limit"
        ):
            publication._authenticated_graph_size(graph)

    def test_isolated_git_ignores_path_and_uses_controlled_https_config(self):
        captured = {}

        def run(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        cluster_git = {
            "invoked_path": "/usr/local/bin/git",
            "resolved_path": "/usr/bin/git",
        }
        with mock.patch.object(publication, "_resolve_git_executable", return_value=cluster_git), mock.patch.object(
            publication.subprocess, "run", side_effect=run
        ):
            publication._isolated_git_result(
                self.trusted_repo,
                ["ls-remote", publication.AUTHORIZED_REPOSITORY_URL, publication.AUTHORIZED_PUBLICATION_REF],
                "synthetic isolated transport",
            )
        self.assertEqual(captured["argv"][0], "/usr/local/bin/git")
        self.assertIn("credential.helper=", captured["argv"])
        self.assertIn("protocol.https.allow=always", captured["argv"])
        self.assertIn("http.followRedirects=false", captured["argv"])
        self.assertNotIn("origin", captured["argv"])
        self.assertEqual(captured["kwargs"]["cwd"], self.trusted_repo)

    def test_git_executable_safe_symlink_chain_and_malicious_paths(self):
        bin_a = self.root / "usr_bin"
        bin_b = self.root / "usr_local_bin"
        outside = self.root / "outside"
        for directory in (bin_a, bin_b, outside):
            directory.mkdir()
            directory.chmod(0o755)
        target = bin_a / "git"
        target.write_text("synthetic executable\n", encoding="utf-8")
        target.chmod(0o755)
        safe_link = bin_b / "git"
        safe_link.symlink_to(target)
        allowed = (safe_link.as_posix(), target.as_posix())
        observed = publication._validate_git_executable_candidate(
            safe_link.as_posix(),
            allowed_candidates=allowed,
            trusted_directories=(bin_a.as_posix(), bin_b.as_posix()),
            owner_uid=publication.os.getuid(),
        )
        self.assertEqual(observed["resolved_path"], target.as_posix())

        malicious_target = outside / "git"
        malicious_target.write_text("malicious\n", encoding="utf-8")
        malicious_target.chmod(0o755)
        malicious_link = bin_b / "git-malicious"
        malicious_link.symlink_to(malicious_target)
        with self.assertRaisesRegex(publication.PublicationContractError, "leaves trusted"):
            publication._validate_git_executable_candidate(
                malicious_link.as_posix(),
                allowed_candidates=(malicious_link.as_posix(), malicious_target.as_posix()),
                trusted_directories=(bin_a.as_posix(), bin_b.as_posix()),
                owner_uid=publication.os.getuid(),
            )
        target.chmod(0o777)
        with self.assertRaisesRegex(publication.PublicationContractError, "not owned and protected"):
            publication._validate_git_executable_candidate(
                safe_link.as_posix(),
                allowed_candidates=allowed,
                trusted_directories=(bin_a.as_posix(), bin_b.as_posix()),
                owner_uid=publication.os.getuid(),
            )
        with self.assertRaisesRegex(publication.PublicationContractError, "not pinned"):
            publication._validate_git_executable_candidate("/tmp/arbitrary-git")

    def test_isolated_git_failure_does_not_disclose_command_or_stderr(self):
        secret = "https://token-value@github.com/adeeb10abbas/steerable.git"
        failure = subprocess.TimeoutExpired(["/usr/bin/git", "ls-remote", secret], 30)
        with mock.patch.object(publication.subprocess, "run", side_effect=failure), self.assertRaises(
            publication.PublicationContractError
        ) as raised:
            publication._isolated_git_bytes(
                self.trusted_repo, ["ls-remote", secret], "remote network"
            )
        rendered = str(raised.exception)
        self.assertNotIn("token-value", rendered)
        self.assertNotIn(secret, rendered)

    def test_git_replace_refs_cannot_substitute_authenticated_blobs(self):
        relative = (
            "workshops/corl2026_world_models/analysis/forecast_evidence_analysis.py"
        )
        original = self.science_sources["analyzer_source"].read_bytes()
        self.science_sources["analyzer_source"].write_bytes(b"# replacement payload\n")
        subprocess.run(["git", "add", relative], cwd=self.trusted_repo, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "replacement target"],
            cwd=self.trusted_repo,
            check=True,
        )
        replacement = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.trusted_repo, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        subprocess.run(
            ["git", "replace", self.study_commit, replacement],
            cwd=self.trusted_repo,
            check=True,
        )
        observed = publication._isolated_git_bytes(
            self.trusted_repo,
            ["show", f"{self.study_commit}:{relative}"],
            "replacement-resistant historical blob",
        )
        self.assertEqual(observed, original)

    def test_each_receipt_selected_executable_path_is_independently_canonical(self):
        field_by_label = {
            "confirmation_compiler_source": "compiler_source",
            "analyzer_source": "final_analyzer_dependency",
            "fixture_validator_source": "fixture_freeze_dependency",
        }
        for label, field in field_by_label.items():
            with self.subTest(label=label):
                malicious = self.root / f"coordinated-{label}.py"
                malicious.write_text("# caller-selected coordinated replacement\n", encoding="utf-8")
                receipt = copy.deepcopy(self.compiler_receipt)
                receipt[field] = descriptor(malicious)
                supplied_analyzer = (
                    receipt["final_analyzer_dependency"]
                    if field == "final_analyzer_dependency"
                    else self.compiler_receipt["final_analyzer_dependency"]
                )
                with self.assertRaisesRegex(
                    publication.PublicationContractError, "not at its canonical path"
                ):
                    publication._validate_trusted_science_sources(
                        receipt=receipt,
                        receipt_path=self.receipt_path,
                        supplied_analyzer_descriptor=supplied_analyzer,
                        publication_source_commit=self.study_commit,
                        trusted_repository=self.trusted_repo,
                    )

    def test_coordinated_self_hashed_dirty_sources_fail_before_replay(self):
        receipt = copy.deepcopy(self.compiler_receipt)
        for field, key in (
            ("compiler_source", "compiler_source"),
            ("final_analyzer_dependency", "analyzer_source"),
            ("fixture_freeze_dependency", "fixture_source"),
        ):
            path = self.science_sources[key]
            path.write_text(path.read_text(encoding="utf-8") + "# malicious replacement\n", encoding="utf-8")
            receipt[field] = descriptor(path)
        with self.assertRaisesRegex(publication.PublicationContractError, "checkout is dirty"):
            publication._validate_trusted_science_sources(
                receipt=receipt,
                receipt_path=self.receipt_path,
                supplied_analyzer_descriptor=receipt["final_analyzer_dependency"],
                publication_source_commit=self.study_commit,
                trusted_repository=self.trusted_repo,
            )
        self.assertEqual(self.fake_analyzer.calls, [])
        self.assertEqual(self.fake_compiler.calls, [])
        self.assertEqual(self.fake_fixture.calls, [])

    def test_coordinated_committed_descendant_cannot_replace_authenticated_ancestry(self):
        publication_commit = self.study_commit
        for path in self.science_sources.values():
            path.write_text(path.read_text(encoding="utf-8") + "# malicious committed replacement\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=self.trusted_repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "coordinated replacement"], cwd=self.trusted_repo, check=True)
        malicious_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.trusted_repo, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        receipt = copy.deepcopy(self.compiler_receipt)
        receipt["study_commit"] = malicious_commit
        receipt["compiler_source"] = descriptor(self.science_sources["compiler_source"])
        receipt["final_analyzer_dependency"] = descriptor(self.science_sources["analyzer_source"])
        receipt["fixture_freeze_dependency"] = descriptor(self.science_sources["fixture_source"])
        with self.assertRaisesRegex(
            publication.PublicationContractError, "not an ancestor"
        ):
            publication._validate_trusted_science_sources(
                receipt=receipt,
                receipt_path=self.receipt_path,
                supplied_analyzer_descriptor=receipt["final_analyzer_dependency"],
                publication_source_commit=publication_commit,
                trusted_repository=self.trusted_repo,
            )

    def test_coordinated_malicious_ancestor_differs_from_authenticated_head_blobs(self):
        original = {
            key: path.read_bytes() for key, path in self.science_sources.items()
        }
        for path in self.science_sources.values():
            path.write_bytes(path.read_bytes() + b"# malicious historical validator\n")
        subprocess.run(["git", "add", "."], cwd=self.trusted_repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "malicious historical source"], cwd=self.trusted_repo, check=True)
        malicious_study_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.trusted_repo, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        for key, payload in original.items():
            self.science_sources[key].write_bytes(payload)
        subprocess.run(["git", "add", "."], cwd=self.trusted_repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "authenticated publication source"], cwd=self.trusted_repo, check=True)
        publication_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.trusted_repo, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        historical_checkout = self.root / "historical_checkout"
        subprocess.run(
            ["git", "worktree", "add", "-q", "--detach", str(historical_checkout), malicious_study_commit],
            cwd=self.trusted_repo,
            check=True,
        )
        historical = {
            "compiler_source": historical_checkout / "workshops/corl2026_world_models/analysis/compile_confirmation_evidence.py",
            "analyzer_source": historical_checkout / "workshops/corl2026_world_models/analysis/forecast_evidence_analysis.py",
            "fixture_source": historical_checkout / "workshops/corl2026_world_models/experiments/forecast_layout/confirmation_fixture_freeze.py",
        }
        receipt = copy.deepcopy(self.compiler_receipt)
        receipt["source_root"] = str(historical_checkout.resolve())
        receipt["study_commit"] = malicious_study_commit
        receipt["compiler_source"] = descriptor(historical["compiler_source"])
        receipt["final_analyzer_dependency"] = descriptor(historical["analyzer_source"])
        receipt["fixture_freeze_dependency"] = descriptor(historical["fixture_source"])
        with self.assertRaisesRegex(
            publication.PublicationContractError,
            "differs between the study checkout and authenticated publication HEAD",
        ):
            publication._validate_trusted_science_sources(
                receipt=receipt,
                receipt_path=self.receipt_path,
                supplied_analyzer_descriptor=receipt["final_analyzer_dependency"],
                publication_source_commit=publication_commit,
                trusted_repository=self.trusted_repo,
            )

    def test_full_and_reduced_d1_branches_preserve_model_semantics(self):
        full = copy.deepcopy(self.analysis)
        full.pop("payload_sha256")
        full["cohort_branch"] = "full_two_model"
        full["study_scope"] = (
            "FULL_TWO_MODEL_BRANCH_REPORTED_AS_TWO_SEPARATE_WITHIN_MODEL_STUDIES"
        )
        d1_row = copy.deepcopy(full["model_sample_size_table"][0])
        d1_row["model_id"] = "D1"
        d1_row["checkpoint_revision"] = "d1-checkpoint"
        d1_row["executed_prefix_cap"] = 8
        full["model_sample_size_table"][1] = d1_row
        full["models"]["D1"] = copy.deepcopy(full["models"]["N3"])
        full = publication.sign_document(full)
        sample, models = publication._validate_analysis_report(full, "full_two_model")
        self.assertEqual([row["model_id"] for row in sample], ["N3", "D1"])
        self.assertEqual(set(models), {"N3", "D1"})

        reduced = copy.deepcopy(full)
        reduced.pop("payload_sha256")
        reduced["cohort_branch"] = "reduced_d1"
        reduced["study_scope"] = (
            "REDUCED_ONE_MODEL_D1_BRANCH; the other primary model is unqualified "
            "and its 96 confirmation cells were not substituted"
        )
        absent_n3 = copy.deepcopy(self.analysis["model_sample_size_table"][1])
        absent_n3["model_id"] = "N3"
        included_d1 = copy.deepcopy(d1_row)
        reduced["model_sample_size_table"] = [absent_n3, included_d1]
        reduced["models"] = {"D1": copy.deepcopy(full["models"]["D1"])}
        reduced = publication.sign_document(reduced)
        sample, models = publication._validate_analysis_report(reduced, "reduced_d1")
        self.assertEqual(sample[0]["branch_status"], "unqualified_branch_not_run")
        self.assertEqual(set(models), {"D1"})

    def test_existing_output_is_never_replaced(self):
        output = self.root / "existing"
        output.mkdir()
        marker = output / "owned.txt"
        marker.write_text("keep", encoding="utf-8")
        with self._patch_validators(), self.assertRaisesRegex(
            publication.PublicationContractError, "refusing to replace"
        ):
            publication.compile_publication(self.input_path, output)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_csv_and_tex_are_fragments_not_a_paper(self):
        output = self.root / "publication"
        with self._patch_validators():
            publication.compile_publication(self.input_path, output)
        with (output / "model_results.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["model_id"] for row in rows], ["N3", "D1"])
        self.assertEqual(rows[0]["forecast_skill_vs_persistence"], "0.25")
        self.assertEqual(rows[1]["forecast_skill_vs_persistence"], "NA")
        self.assertIn("\\begin{tabular}", (output / "model_results_table.tex").read_text())
        self.assertFalse(any(path.suffix == ".md" for path in output.rglob("*")))


if __name__ == "__main__":
    unittest.main()
