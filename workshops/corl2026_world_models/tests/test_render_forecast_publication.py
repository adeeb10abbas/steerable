import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET


MODULE = Path(__file__).resolve().parents[1] / "analysis" / "render_forecast_publication.py"
SPEC = importlib.util.spec_from_file_location("render_forecast_publication", MODULE)
renderer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(renderer)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sign(value):
    unsigned = copy.deepcopy(value)
    unsigned.pop("payload_sha256", None)
    return renderer.sign_document(unsigned)


def bootstrap_metric(estimate, count=24):
    if estimate is None:
        return {"estimate": None, "ci95": None, "layout_pairs": 0}
    return {
        "estimate": estimate,
        "ci95": [estimate - 0.01, estimate + 0.01],
        "layout_pairs": count,
        "resamples": renderer.BOOTSTRAP_RESAMPLES,
        "seed": renderer.ANALYSIS_SEED,
        "resampling_unit": renderer.RESAMPLING_UNIT,
    }


def skill_metric(estimate):
    rows = [
        {"layout_pair_id": layout, "value": estimate + index / 1000.0}
        for index, layout in enumerate(renderer.LAYOUTS)
    ]
    return {
        **bootstrap_metric(estimate),
        "complete_layout_ids": list(renderer.LAYOUTS),
        "excluded_incomplete_layout_ids": [],
        "layout_pair_values": rows,
    }


def reflection_input(discrepancy=0.02, *, unavailable=False):
    commands = {}
    for command_index, command in enumerate(renderer.COMMANDS):
        if unavailable:
            paired = []
            actual = predicted = difference = bootstrap_metric(None, count=0)
        else:
            actual_value = 0.01 + command_index * 0.002
            predicted_value = actual_value + discrepancy
            paired = [
                {
                    "layout_pair_id": layout,
                    "actual_relative_motion": actual_value,
                    "predicted_relative_motion": predicted_value,
                    "discrepancy": discrepancy,
                }
                for layout in renderer.LAYOUTS
            ]
            actual = bootstrap_metric(actual_value)
            predicted = bootstrap_metric(predicted_value)
            difference = bootstrap_metric(discrepancy)
        commands[command] = {
            "actual_reflected_minus_original_motion": actual,
            "predicted_reflected_minus_original_motion": predicted,
            "predicted_minus_actual_contrast_discrepancy": difference,
            "eligible_layout_ids": [] if unavailable else list(renderer.LAYOUTS),
            "paired_layout_values": paired,
        }
    combined_rows = [] if unavailable else [
        {"layout_pair_id": layout, "discrepancy": discrepancy}
        for layout in renderer.LAYOUTS
    ]
    combined = bootstrap_metric(None if unavailable else discrepancy, count=0 if unavailable else 24)
    combined["paired_layout_values"] = combined_rows
    return {
        "definition": renderer.REFLECTION_DEFINITION,
        "by_command": commands,
        "combined_command_discrepancy": combined,
    }


class RenderFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def descriptor(self, path, public_path):
        return renderer.file_descriptor(path, public_path=public_path)

    def source_authentication(self):
        repository = renderer.REPOSITORY.resolve()
        return {
            "renderer_source_commit": "3" * 40,
            "publication_to_renderer_to_control_ancestry": True,
            "authorized_renderer_remote": {
                "repository_url": renderer.AUTHORIZED_REPOSITORY_URL,
                "control_branch": renderer.AUTHORIZED_PUBLICATION_BRANCH,
                "control_ref": renderer.AUTHORIZED_PUBLICATION_REF,
                "remote_ref_commit": "4" * 40,
                "read_transport": "literal_public_https_isolated_bare",
                "git_executable_path": "/usr/bin/git",
                "git_executable_resolved_path": "/usr/bin/git",
                "git_version": "git version 2.34.1",
                "blob_filter_limit_bytes": renderer.AUTHENTICATED_BLOB_LIMIT_BYTES,
                "authenticated_graph_bytes": 4096,
                "authenticated_graph_max_bytes": renderer.AUTHENTICATED_GRAPH_MAX_BYTES,
                "isolated_repository_config_sha256": renderer.ISOLATED_REPOSITORY_CONFIG_SHA256,
                "ls_remote_before_and_after_fetch_match": True,
                "study_to_publication_to_control_ancestry": True,
                "local_head_commit": "3" * 40,
                "checkout_mode": "attached_control_branch",
                "deployment_remote_alias": "publish",
            },
            "committed_renderer_sources": {
                "renderer_source": self.descriptor(
                    renderer.RENDERER_SOURCE_PATH,
                    renderer.RENDERER_SOURCE_PATH.resolve().relative_to(repository).as_posix(),
                ),
                "renderer_contract": self.descriptor(
                    renderer.CONTRACT_PATH,
                    renderer.CONTRACT_PATH.resolve().relative_to(repository).as_posix(),
                ),
            },
        }

    def make_bundle(self, name="publication", branch="reduced_n3", *, reflection_unavailable=False):
        root = self.root / name
        root.mkdir()
        models = renderer.BRANCH_MODELS[branch]
        final_hash = digest(f"final:{name}:{branch}")
        fixture_hash = digest(f"fixture:{name}:{branch}")
        forecast_models = {}
        timing = {}
        for model_index, model in enumerate(models):
            estimate = 0.20 if model == "N3" else -0.08
            forecast_models[model] = {
                "forecast_skill_vs_persistence": skill_metric(estimate),
                "reflection_layout_effects": reflection_input(
                    0.02 if model == "N3" else -0.015,
                    unavailable=reflection_unavailable,
                ),
            }
            timing[model] = {
                "primary_horizon_s": 0.5 if model == "N3" else 0.4,
                "generated_frame_index": 3 if model == "N3" else 2,
                "target_executed_action_offset": 16 if model == "N3" else 8,
                "executed_prefix_cap": 32 if model == "N3" else 12,
                "camera_id": "over_shoulder_left_camera" if model_index == 0 else "front_camera",
                "timestamp_tolerance_s": 0.02 if model == "N3" else 0.015,
            }
        forecast = sign(
            {
                "schema_version": renderer.FORECAST_INPUT_SCHEMA,
                "study_id": renderer.STUDY_ID,
                "cohort_branch": branch,
                "models_pooled": False,
                "analysis_seed": renderer.ANALYSIS_SEED,
                "layout_bootstrap_resamples": renderer.BOOTSTRAP_RESAMPLES,
                "models": forecast_models,
                "source_final_analysis_sha256": final_hash,
            }
        )
        write_json(root / renderer.FORECAST_FILENAME, forecast)

        layouts = []
        for index, layout in enumerate(renderer.LAYOUTS, start=1):
            original_positions = {
                "banana": [0.30 + index / 1000.0, 0.12 + index / 2000.0, 0.021],
                "bowl": [0.47 + index / 1500.0, 0.16 + index / 2500.0, 0.032],
                "rubiks_cube": [0.41 + index / 1800.0, -0.11 + index / 3000.0, 0.043],
            }
            reflected_positions = {
                object_id: [position[0], -position[1], position[2]]
                for object_id, position in original_positions.items()
            }
            quaternions = {
                "banana": [1.0, 0.0, 0.0, 0.0],
                "bowl": [0.9238795325, 0.0, 0.0, 0.3826834324],
                "rubiks_cube": [0.7071067812, 0.0, 0.0, 0.7071067812],
            }
            layouts.append(
                {
                    "layout_pair_id": layout,
                    "environment_seed": 2026091200 + index,
                    "candidate_id": f"{layout}__candidate_00",
                    "candidate_payload_sha256": digest(f"candidate:{name}:{layout}"),
                    "accepted_gate_record_sha256": digest(f"gate:{name}:{layout}"),
                    "pose_manifest": {
                        "bytes": 1000 + index,
                        "sha256": digest(f"pose:{name}:{layout}"),
                    },
                    "layouts": {
                        "original": {
                            "positions_robot_base_m": original_positions,
                            "quaternions_wxyz": copy.deepcopy(quaternions),
                        },
                        "reflected": {
                            "positions_robot_base_m": reflected_positions,
                            "quaternions_wxyz": copy.deepcopy(quaternions),
                        },
                    },
                }
            )
        scene = sign(
            {
                "schema_version": renderer.SCENE_INPUT_SCHEMA,
                "study_id": renderer.STUDY_ID,
                "cohort_branch": branch,
                "scene_coordinate_frame": "robot_base_m",
                "scene_source": "model_blind_physically_gated_confirmation_fixture_freeze",
                "layout_count": 24,
                "layouts": layouts,
                "model_timing": timing,
                "source_fixture_freeze_sha256": fixture_hash,
                "source_final_analysis_sha256": final_hash,
            }
        )
        write_json(root / renderer.SCENE_FILENAME, scene)

        dummy_payloads = {
            "paper_evidence.json": b"{}\n",
            "model_results.csv": b"model_id\n",
            "coverage_by_condition.csv": b"model_id,condition_id\n",
            "model_results_table.tex": b"% synthetic\n",
            "sample_size_table.tex": b"% synthetic\n",
            "example_videos.json": b"{}\n",
        }
        for relative, payload in dummy_payloads.items():
            (root / relative).write_bytes(payload)
        inventory = {
            relative: self.descriptor(root / relative, relative)
            for relative in sorted(renderer.PUBLICATION_STATIC_OUTPUTS)
        }
        input_bindings = {
            key: {"bytes": 1 + index, "sha256": digest(f"binding:{name}:{key}")}
            for index, key in enumerate(sorted(renderer.INPUT_BINDING_KEYS))
        }
        input_bindings["final_analysis"]["sha256"] = final_hash
        input_bindings["confirmation_fixture_freeze"]["sha256"] = fixture_hash
        receipt = sign(
            {
                "schema_version": renderer.PUBLICATION_RECEIPT_SCHEMA,
                "study_id": renderer.STUDY_ID,
                "status": "compiled_source_only_publication_artifacts",
                "cohort_branch": branch,
                "study_commit": "1" * 40,
                "publication_source_commit": "2" * 40,
                "authorized_publication_remote": {"repository": "https://github.com/adeeb10abbas/steerable.git"},
                "publication_compiler_source": {
                    "path": "workshops/corl2026_world_models/analysis/compile_forecast_publication.py",
                    "bytes": 123,
                    "sha256": digest("publication compiler"),
                },
                "publication_contract": {
                    "path": "workshops/corl2026_world_models/experiments/forecast_layout/forecast_publication_contract.json",
                    "bytes": 456,
                    "sha256": digest("publication contract"),
                },
                "trusted_science_source_authentication": {"authenticated": True},
                "input_bindings": input_bindings,
                "model_ids": list(models),
                "models_pooled": False,
                "layout_count": 24,
                "selected_example_count": 4 * len(models),
                "selected_video_copy_count": 0,
                "selected_video_copy_limits_bytes": {
                    "per_file": 16 * 1024 * 1024,
                    "total": 64 * 1024 * 1024,
                },
                "selected_video_copies": [],
                "output_count_including_self_signed_receipt": len(inventory) + 1,
                "output_inventory": inventory,
                "receipt_self_binding": "payload_sha256",
                "paper_generation_performed": False,
                "labels_created": False,
                "scientific_estimates_recomputed": False,
                "claim_boundary": "Synthetic source-only publication fixture.",
            }
        )
        write_json(root / renderer.PUBLICATION_RECEIPT_FILENAME, receipt)
        return root

    def make_authenticated_source_repository(self, name):
        repository = self.root / name
        repository.mkdir()
        relative_compiler = renderer.PUBLICATION_COMPILER_PATH.resolve().relative_to(
            renderer.REPOSITORY.resolve()
        )
        relative_renderer = renderer.RENDERER_SOURCE_PATH.resolve().relative_to(
            renderer.REPOSITORY.resolve()
        )
        relative_contract = renderer.CONTRACT_PATH.resolve().relative_to(
            renderer.REPOSITORY.resolve()
        )
        compiler_path = repository / relative_compiler
        renderer_path = repository / relative_renderer
        contract_path = repository / relative_contract
        compiler_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(renderer.PUBLICATION_COMPILER_PATH, compiler_path)
        subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repository, check=True)
        subprocess.run(["git", "config", "user.name", "Synthetic Test"], cwd=repository, check=True)
        subprocess.run(["git", "add", "."], cwd=repository, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "publication source"], cwd=repository, check=True)
        publication_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        renderer_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(renderer.RENDERER_SOURCE_PATH, renderer_path)
        shutil.copyfile(renderer.CONTRACT_PATH, contract_path)
        subprocess.run(["git", "add", "."], cwd=repository, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "renderer source"], cwd=repository, check=True)
        subprocess.run(
            ["git", "branch", "-M", renderer.AUTHORIZED_PUBLICATION_BRANCH],
            cwd=repository, check=True,
        )
        renderer_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        subprocess.run(
            ["git", "remote", "add", "origin", renderer.AUTHORIZED_REPOSITORY_URL],
            cwd=repository, check=True,
        )
        remote = self.root / f"{name}.git"
        subprocess.run(["git", "clone", "-q", "--bare", str(repository), str(remote)], check=True)
        subprocess.run(
            ["git", "config", "uploadpack.allowFilter", "true"], cwd=remote, check=True
        )
        receipt = {
            "publication_source_commit": publication_commit,
            "publication_compiler_source": self.descriptor(
                compiler_path, relative_compiler.as_posix()
            ),
        }

        def source_graph_factory(compiler, observed_renderer, observed_publication, observed_repository):
            self.assertEqual(observed_renderer, renderer_commit)
            self.assertEqual(observed_publication, publication_commit)
            self.assertEqual(observed_repository, repository.resolve())
            return compiler._authenticated_source_graph_from_url(
                publication_commit=observed_renderer,
                study_commit=observed_publication,
                remote_url=remote.resolve().as_uri(),
                protocol="file",
            )

        return {
            "repository": repository,
            "compiler_path": compiler_path,
            "renderer_path": renderer_path,
            "contract_path": contract_path,
            "publication_commit": publication_commit,
            "renderer_commit": renderer_commit,
            "remote": remote,
            "receipt": receipt,
            "source_graph_factory": source_graph_factory,
        }

    def rebind_publication_receipt(self, bundle):
        receipt_path = bundle / renderer.PUBLICATION_RECEIPT_FILENAME
        receipt = read_json(receipt_path)
        for relative in receipt["output_inventory"]:
            receipt["output_inventory"][relative] = self.descriptor(bundle / relative, relative)
        write_json(receipt_path, sign(receipt))

    def mutate_signed_input(self, bundle, filename, mutate, *, rebind=True):
        path = bundle / filename
        value = read_json(path)
        mutate(value)
        write_json(path, sign(value))
        if rebind:
            self.rebind_publication_receipt(bundle)

    def render(self, bundle, name="figures"):
        output = self.root / name
        with mock.patch.object(
            renderer,
            "_authenticate_renderer_sources",
            return_value=self.source_authentication(),
        ):
            receipt = renderer.render_publication(bundle, output)
        return output, receipt

    def assert_valid_svg(self, path):
        root = ET.fromstring(path.read_bytes())
        self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")

    def test_reduced_branch_renders_exact_scene_timing_and_separate_model_files(self):
        bundle = self.make_bundle()
        output, receipt = self.render(bundle)
        self.assertEqual(
            {path.name for path in output.iterdir()},
            {
                "actual_scene_layouts.svg", "qualified_timing.svg",
                "forecast_skill_n3.svg", "reflection_layout_effects_n3.svg",
                "render_receipt.json",
            },
        )
        for svg in output.glob("*.svg"):
            self.assert_valid_svg(svg)
        timing = (output / "qualified_timing.svg").read_text(encoding="utf-8")
        for expected in (
            "frame index = 3", "horizon = 0.5 s", "action offset = 16",
            "executed prefix cap = 32 actions", "camera=over_shoulder_left_camera",
            "timestamp tolerance=0.02 s",
            "No generated-frame ↔ physical-time ↔ executed-action equivalence is inferred.",
        ):
            self.assertIn(expected, timing)
        scenes = (output / "actual_scene_layouts.svg").read_text(encoding="utf-8")
        self.assertIn("C01/original/banana", scenes)
        self.assertIn("C24/reflected/rubiks_cube", scenes)
        self.assertIn("position_robot_base_m=[", scenes)
        self.assertIn("quaternion_wxyz=[", scenes)
        self.assertFalse(receipt["frame_to_action_mapping_inferred"])
        self.assertFalse(receipt["scientific_estimates_recomputed"])

    def test_full_branch_renders_each_model_without_pooling(self):
        bundle = self.make_bundle(branch="full_two_model")
        output, receipt = self.render(bundle)
        self.assertEqual(receipt["model_ids"], ["N3", "D1"])
        self.assertFalse(receipt["models_pooled"])
        for model in ("n3", "d1"):
            self.assertTrue((output / f"forecast_skill_{model}.svg").is_file())
            self.assertTrue((output / f"reflection_layout_effects_{model}.svg").is_file())
        n3 = (output / "forecast_skill_n3.svg").read_text(encoding="utf-8")
        d1 = (output / "forecast_skill_d1.svg").read_text(encoding="utf-8")
        self.assertIn("N3 forecast skill", n3)
        self.assertNotIn("D1 forecast skill", n3)
        self.assertIn("D1 forecast skill", d1)
        self.assertNotIn("N3 forecast skill", d1)

    def test_reduced_d1_branch_is_supported(self):
        bundle = self.make_bundle(branch="reduced_d1")
        output, receipt = self.render(bundle)
        self.assertEqual(receipt["model_ids"], ["D1"])
        self.assertTrue((output / "forecast_skill_d1.svg").is_file())
        self.assertFalse((output / "forecast_skill_n3.svg").exists())

    def test_outputs_are_byte_deterministic_across_directories(self):
        bundle = self.make_bundle()
        first, _ = self.render(bundle, "first")
        second, _ = self.render(bundle, "second")
        first_files = {path.name: path.read_bytes() for path in first.iterdir()}
        second_files = {path.name: path.read_bytes() for path in second.iterdir()}
        self.assertEqual(first_files, second_files)

    def test_unavailable_reflection_values_remain_explicitly_null(self):
        bundle = self.make_bundle(reflection_unavailable=True)
        output, _ = self.render(bundle)
        reflection = (output / "reflection_layout_effects_n3.svg").read_text(encoding="utf-8")
        self.assertIn("No reflection contrast values are available", reflection)
        self.assertIn("not estimable (0 layout pairs)", reflection)
        self.assertNotIn("estimate=0", reflection)

    def test_null_required_skill_is_rejected(self):
        bundle = self.make_bundle()
        def mutate(value):
            value["models"]["N3"]["forecast_skill_vs_persistence"] = {
                "estimate": None,
                "ci95": None,
                "layout_pairs": 0,
                "resamples": renderer.BOOTSTRAP_RESAMPLES,
                "seed": renderer.ANALYSIS_SEED,
                "resampling_unit": renderer.RESAMPLING_UNIT,
                "complete_layout_ids": [],
                "excluded_incomplete_layout_ids": list(renderer.LAYOUTS),
                "layout_pair_values": [],
            }
        self.mutate_signed_input(bundle, renderer.FORECAST_FILENAME, mutate)
        with self.assertRaises(renderer.FigureRenderContractError):
            self.render(bundle)

    def test_null_or_unsupported_included_timing_is_rejected(self):
        bundle = self.make_bundle()
        self.mutate_signed_input(
            bundle,
            renderer.SCENE_FILENAME,
            lambda value: value["model_timing"]["N3"].__setitem__("primary_horizon_s", None),
        )
        with self.assertRaises(renderer.FigureRenderContractError):
            self.render(bundle)

    def test_action_offset_is_not_constrained_by_context_cap(self):
        bundle = self.make_bundle()
        self.mutate_signed_input(
            bundle,
            renderer.SCENE_FILENAME,
            lambda value: value["model_timing"]["N3"].update(
                {"target_executed_action_offset": 48, "executed_prefix_cap": 32}
            ),
        )
        output, receipt = self.render(bundle)
        timing = (output / "qualified_timing.svg").read_text(encoding="utf-8")
        self.assertIn("action offset = 48", timing)
        self.assertIn("executed prefix cap = 32 actions", timing)
        self.assertFalse(receipt["frame_to_action_mapping_inferred"])

    def test_swapped_signed_source_is_rejected_by_receipt_binding(self):
        first = self.make_bundle("first")
        second = self.make_bundle("second")
        shutil.copyfile(second / renderer.FORECAST_FILENAME, first / renderer.FORECAST_FILENAME)
        with self.assertRaisesRegex(renderer.FigureRenderContractError, "publication output changed"):
            self.render(first)

    def test_tampered_source_is_rejected(self):
        bundle = self.make_bundle()
        forecast_path = bundle / renderer.FORECAST_FILENAME
        value = read_json(forecast_path)
        value["models"]["N3"]["forecast_skill_vs_persistence"]["estimate"] = 999.0
        write_json(forecast_path, value)
        with self.assertRaises(renderer.FigureRenderContractError):
            self.render(bundle)

    def test_cross_source_final_analysis_mismatch_is_rejected(self):
        bundle = self.make_bundle()
        self.mutate_signed_input(
            bundle,
            renderer.SCENE_FILENAME,
            lambda value: value.__setitem__("source_final_analysis_sha256", digest("other final")),
        )
        with self.assertRaisesRegex(renderer.FigureRenderContractError, "same final analysis"):
            self.render(bundle)

    def test_private_path_in_signed_input_is_rejected(self):
        bundle = self.make_bundle()
        self.mutate_signed_input(
            bundle,
            renderer.SCENE_FILENAME,
            lambda value: value["model_timing"]["N3"].__setitem__("camera_id", "/data/private/camera"),
        )
        with self.assertRaises(renderer.FigureRenderContractError):
            self.render(bundle)

    def test_nonfinite_json_constant_is_rejected(self):
        bundle = self.make_bundle()
        path = bundle / renderer.SCENE_FILENAME
        payload = path.read_text(encoding="utf-8")
        payload = payload.replace('"environment_seed": 2026091201', '"environment_seed": NaN', 1)
        path.write_text(payload, encoding="utf-8")
        self.rebind_publication_receipt(bundle)
        with self.assertRaisesRegex(renderer.FigureRenderContractError, "non-finite JSON constant"):
            self.render(bundle)

    def test_layout_count_drift_is_rejected(self):
        bundle = self.make_bundle()
        self.mutate_signed_input(
            bundle,
            renderer.SCENE_FILENAME,
            lambda value: value.__setitem__("layout_count", 23),
        )
        with self.assertRaises(renderer.FigureRenderContractError):
            self.render(bundle)

    def test_phantom_metric_is_rejected_and_never_rendered(self):
        bundle = self.make_bundle()
        def mutate(value):
            value["models"]["N3"]["phantom_accuracy"] = 1.0
        self.mutate_signed_input(bundle, renderer.FORECAST_FILENAME, mutate)
        with self.assertRaisesRegex(renderer.FigureRenderContractError, "disallowed keys"):
            self.render(bundle)

        clean = self.make_bundle("clean")
        output, _ = self.render(clean, "clean_figures")
        all_svg = "".join(path.read_text(encoding="utf-8") for path in output.glob("*.svg"))
        self.assertNotIn("constant_velocity", all_svg)
        self.assertNotIn("phantom_accuracy", all_svg)

    def test_reflection_count_drift_is_rejected(self):
        bundle = self.make_bundle()
        def mutate(value):
            metric = value["models"]["N3"]["reflection_layout_effects"]["by_command"]["left"][
                "predicted_minus_actual_contrast_discrepancy"
            ]
            metric["layout_pairs"] = 23
        self.mutate_signed_input(bundle, renderer.FORECAST_FILENAME, mutate)
        with self.assertRaisesRegex(renderer.FigureRenderContractError, "layout count"):
            self.render(bundle)

    def test_output_directory_is_no_overwrite_and_input_bundle_remains_immutable(self):
        bundle = self.make_bundle()
        before = {
            path.relative_to(bundle).as_posix(): path.read_bytes()
            for path in bundle.rglob("*") if path.is_file()
        }
        output = self.root / "figures"
        output.mkdir()
        sentinel = output / "sentinel"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(renderer.FigureRenderContractError, "refusing to replace"):
            renderer.render_publication(bundle, output)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
        after = {
            path.relative_to(bundle).as_posix(): path.read_bytes()
            for path in bundle.rglob("*") if path.is_file()
        }
        self.assertEqual(before, after)

    def test_atomic_publish_primitive_never_replaces_a_raced_in_directory(self):
        source = self.root / "staged"
        target = self.root / "raced_target"
        source.mkdir()
        target.mkdir()
        (source / "source").write_text("source", encoding="utf-8")
        (target / "owner").write_text("owner", encoding="utf-8")
        parent_fd, _ = renderer._open_directory_nofollow(self.root, "test parent")
        try:
            with self.assertRaisesRegex(renderer.FigureRenderContractError, "refusing to replace"):
                renderer._rename_directory_noreplace_at(
                    parent_fd, source.name, target.name
                )
        finally:
            os.close(parent_fd)
        self.assertEqual((target / "owner").read_text(encoding="utf-8"), "owner")
        self.assertEqual((source / "source").read_text(encoding="utf-8"), "source")

    def test_output_inside_publication_bundle_is_rejected_without_residue(self):
        bundle = self.make_bundle()
        target = bundle / "figures"
        with self.assertRaisesRegex(renderer.FigureRenderContractError, "immutable publication bundle"):
            renderer.render_publication(bundle, target)
        self.assertFalse(target.exists())

    def test_symlink_parent_escape_creates_no_outside_directory_or_residue(self):
        bundle = self.make_bundle()
        outside = self.root / "outside"
        outside.mkdir()
        link = self.root / "linked_parent"
        link.symlink_to(outside, target_is_directory=True)
        target = link / "missing" / "figures"
        with self.assertRaisesRegex(
            renderer.FigureRenderContractError, "contains a symlink"
        ):
            renderer.render_publication(bundle, target)
        self.assertFalse((outside / "missing").exists())
        self.assertEqual(list(outside.iterdir()), [])
        self.assertTrue(link.is_symlink())

    def test_missing_output_parent_is_not_created(self):
        bundle = self.make_bundle()
        missing = self.root / "missing_parent"
        with self.assertRaisesRegex(
            renderer.FigureRenderContractError, "is missing"
        ):
            renderer.render_publication(bundle, missing / "figures")
        self.assertFalse(missing.exists())

    def test_parent_swap_before_staging_rejects_without_any_residue(self):
        bundle = self.make_bundle()
        parent = self.root / "safe_parent"
        moved = self.root / "moved_parent"
        outside = self.root / "outside_parent"
        parent.mkdir()
        outside.mkdir()
        authentication = self.source_authentication()

        def swap_before_staging(_receipt):
            parent.rename(moved)
            parent.symlink_to(outside, target_is_directory=True)
            return authentication

        with mock.patch.object(
            renderer, "_authenticate_renderer_sources", side_effect=swap_before_staging
        ), self.assertRaisesRegex(
            renderer.FigureRenderContractError, "contains a symlink|changed after"
        ):
            renderer.render_publication(bundle, parent / "figures")
        self.assertEqual(list(outside.iterdir()), [])
        self.assertEqual(list(moved.iterdir()), [])
        self.assertFalse((outside / "figures").exists())
        self.assertFalse((moved / "figures").exists())
        self.assertFalse(any(path.name.startswith(".wmf-figure-render-") for path in moved.iterdir()))

    def test_parent_swap_at_staging_creation_rejects_and_reaps_fd_relative_staging(self):
        bundle = self.make_bundle()
        parent = self.root / "create_parent"
        moved = self.root / "create_parent_moved"
        outside = self.root / "create_outside"
        parent.mkdir()
        outside.mkdir()
        original_create = renderer._create_staging_directory

        def swap_during_create(parent_fd):
            parent.rename(moved)
            parent.symlink_to(outside, target_is_directory=True)
            return original_create(parent_fd)

        with mock.patch.object(
            renderer,
            "_authenticate_renderer_sources",
            return_value=self.source_authentication(),
        ), mock.patch.object(
            renderer, "_create_staging_directory", side_effect=swap_during_create
        ), self.assertRaisesRegex(
            renderer.FigureRenderContractError, "contains a symlink|changed after"
        ):
            renderer.render_publication(bundle, parent / "figures")
        self.assertEqual(list(outside.iterdir()), [])
        self.assertEqual(list(moved.iterdir()), [])
        self.assertFalse((outside / "figures").exists())
        self.assertFalse((moved / "figures").exists())
        self.assertFalse(any(path.name.startswith(".wmf-figure-render-") for path in moved.iterdir()))

    def test_parent_swap_before_rename_rejects_and_reaps_fd_relative_staging(self):
        bundle = self.make_bundle()
        parent = self.root / "rename_parent"
        moved = self.root / "rename_parent_moved"
        outside = self.root / "rename_outside"
        parent.mkdir()
        outside.mkdir()
        authentication = self.source_authentication()
        original_validate = renderer._validate_staged
        calls = []

        def swap_after_second_validation(*args, **kwargs):
            result = original_validate(*args, **kwargs)
            calls.append(1)
            if len(calls) == 2:
                parent.rename(moved)
                parent.symlink_to(outside, target_is_directory=True)
            return result

        with mock.patch.object(
            renderer, "_authenticate_renderer_sources", return_value=authentication
        ), mock.patch.object(
            renderer, "_validate_staged", side_effect=swap_after_second_validation
        ), self.assertRaisesRegex(
            renderer.FigureRenderContractError, "contains a symlink|changed after"
        ):
            renderer.render_publication(bundle, parent / "figures")
        self.assertEqual(len(calls), 2)
        self.assertEqual(list(outside.iterdir()), [])
        self.assertEqual(list(moved.iterdir()), [])
        self.assertFalse((outside / "figures").exists())
        self.assertFalse((moved / "figures").exists())
        self.assertFalse(any(path.name.startswith(".wmf-figure-render-") for path in moved.iterdir()))

    def test_parent_swap_inside_rename_is_rolled_back_without_publication(self):
        bundle = self.make_bundle()
        parent = self.root / "publish_parent"
        moved = self.root / "publish_parent_moved"
        outside = self.root / "publish_outside"
        parent.mkdir()
        outside.mkdir()
        original_rename = renderer._rename_directory_noreplace_at

        def swap_during_rename(parent_fd, source_name, target_name):
            parent.rename(moved)
            parent.symlink_to(outside, target_is_directory=True)
            return original_rename(parent_fd, source_name, target_name)

        with mock.patch.object(
            renderer,
            "_authenticate_renderer_sources",
            return_value=self.source_authentication(),
        ), mock.patch.object(
            renderer, "_rename_directory_noreplace_at", side_effect=swap_during_rename
        ), self.assertRaisesRegex(
            renderer.FigureRenderContractError, "contains a symlink|changed after"
        ):
            renderer.render_publication(bundle, parent / "figures")
        self.assertEqual(list(outside.iterdir()), [])
        self.assertEqual(list(moved.iterdir()), [])
        self.assertFalse((outside / "figures").exists())
        self.assertFalse((moved / "figures").exists())
        self.assertFalse(any(path.name.startswith(".wmf-figure-render-") for path in moved.iterdir()))

    def test_render_receipt_inventory_and_signatures_are_exact(self):
        bundle = self.make_bundle()
        output, receipt = self.render(bundle)
        self.assertEqual(renderer.payload_hash(receipt), receipt["payload_sha256"])
        self.assertEqual(set(receipt["output_inventory"]), {
            "actual_scene_layouts.svg", "qualified_timing.svg",
            "forecast_skill_n3.svg", "reflection_layout_effects_n3.svg",
        })
        for relative, expected in receipt["output_inventory"].items():
            self.assertEqual(self.descriptor(output / relative, relative), expected)
        serialized = (output / "render_receipt.json").read_text(encoding="utf-8")
        for marker in renderer.PRIVATE_PATH_MARKERS:
            self.assertNotIn(marker, serialized)
        self.assertNotIn(str(bundle), serialized)

    def test_render_receipt_rejects_top_level_or_nested_extra_claims(self):
        bundle = self.make_bundle()
        output, receipt = self.render(bundle)
        expected = set(receipt["output_inventory"])
        extra = copy.deepcopy(receipt)
        extra["forecast_accuracy_verified"] = True
        extra = sign(extra)
        with self.assertRaisesRegex(renderer.FigureRenderContractError, "disallowed keys"):
            renderer._validate_render_receipt(extra, expected)
        nested = copy.deepcopy(receipt)
        nested["committed_renderer_sources"]["renderer_source"]["trusted"] = True
        nested = sign(nested)
        with self.assertRaisesRegex(renderer.FigureRenderContractError, "disallowed keys"):
            renderer._validate_render_receipt(nested, expected)
        write_json(output / "render_receipt.json", extra)
        with self.assertRaisesRegex(renderer.FigureRenderContractError, "changed after creation"):
            renderer._validate_staged(
                output,
                receipt,
                expected,
                publication_root=bundle,
                expected_source_authentication=self.source_authentication(),
            )

    def test_render_receipt_rejects_resigned_input_or_source_descriptor_substitution(self):
        bundle = self.make_bundle()
        _, receipt = self.render(bundle)
        expected = set(receipt["output_inventory"])
        authentication = self.source_authentication()
        changed_input = copy.deepcopy(receipt)
        changed_input["input_bindings"][renderer.FORECAST_FILENAME]["sha256"] = "0" * 64
        changed_input = sign(changed_input)
        with self.assertRaisesRegex(renderer.FigureRenderContractError, "publication bundle"):
            renderer._validate_render_receipt(
                changed_input,
                expected,
                publication_root=bundle,
                expected_source_authentication=authentication,
            )
        changed_source = copy.deepcopy(receipt)
        changed_source["committed_renderer_sources"]["renderer_contract"]["sha256"] = "0" * 64
        changed_source = sign(changed_source)
        with self.assertRaisesRegex(renderer.FigureRenderContractError, "current source bytes"):
            renderer._validate_render_receipt(
                changed_source,
                expected,
                publication_root=bundle,
                expected_source_authentication=authentication,
            )

    def test_renderer_sources_authenticate_through_published_isolated_graph(self):
        source = self.make_authenticated_source_repository("authenticated_sources")
        with mock.patch.multiple(
            renderer,
            REPOSITORY=source["repository"],
            PUBLICATION_COMPILER_PATH=source["compiler_path"],
            RENDERER_SOURCE_PATH=source["renderer_path"],
            CONTRACT_PATH=source["contract_path"],
        ):
            observed = renderer._authenticate_renderer_sources(
                source["receipt"], source_graph_factory=source["source_graph_factory"]
            )
        self.assertEqual(observed["renderer_source_commit"], source["renderer_commit"])
        self.assertTrue(observed["publication_to_renderer_to_control_ancestry"])
        self.assertEqual(
            set(observed["committed_renderer_sources"]),
            {"renderer_source", "renderer_contract"},
        )

    def test_coordinated_renderer_and_contract_substitution_is_rejected(self):
        source = self.make_authenticated_source_repository("substituted_sources")
        source["renderer_path"].write_bytes(
            source["renderer_path"].read_bytes() + b"\n# coordinated local replacement\n"
        )
        source["contract_path"].write_bytes(
            source["contract_path"].read_bytes() + b"\n"
        )
        with mock.patch.multiple(
            renderer,
            REPOSITORY=source["repository"],
            PUBLICATION_COMPILER_PATH=source["compiler_path"],
            RENDERER_SOURCE_PATH=source["renderer_path"],
            CONTRACT_PATH=source["contract_path"],
        ), self.assertRaisesRegex(
            renderer.FigureRenderContractError, "dirty or differs"
        ):
            renderer._authenticate_renderer_sources(
                source["receipt"], source_graph_factory=source["source_graph_factory"]
            )

    def test_coordinated_authenticator_and_receipt_substitution_is_rejected(self):
        source = self.make_authenticated_source_repository("substituted_authenticator")
        source["compiler_path"].write_bytes(
            source["compiler_path"].read_bytes() + b"\n# coordinated authenticator replacement\n"
        )
        relative = source["compiler_path"].resolve().relative_to(
            source["repository"].resolve()
        ).as_posix()
        source["receipt"]["publication_compiler_source"] = self.descriptor(
            source["compiler_path"], relative
        )
        with mock.patch.multiple(
            renderer,
            REPOSITORY=source["repository"],
            PUBLICATION_COMPILER_PATH=source["compiler_path"],
            RENDERER_SOURCE_PATH=source["renderer_path"],
            CONTRACT_PATH=source["contract_path"],
        ), self.assertRaisesRegex(
            renderer.FigureRenderContractError, "pinned trusted byte identity"
        ):
            renderer._authenticate_renderer_sources(
                source["receipt"], source_graph_factory=source["source_graph_factory"]
            )

    def test_unpublished_renderer_descendant_is_rejected(self):
        source = self.make_authenticated_source_repository("unpublished_sources")
        source["renderer_path"].write_bytes(
            source["renderer_path"].read_bytes() + b"\n# unpublished replacement\n"
        )
        source["contract_path"].write_bytes(
            source["contract_path"].read_bytes() + b"\n"
        )
        subprocess.run(["git", "add", "."], cwd=source["repository"], check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "unpublished renderer descendant"],
            cwd=source["repository"], check=True,
        )
        unpublished = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source["repository"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()

        def unpublished_factory(compiler, observed_renderer, observed_publication, observed_repository):
            self.assertEqual(observed_renderer, unpublished)
            return compiler._authenticated_source_graph_from_url(
                publication_commit=observed_renderer,
                study_commit=observed_publication,
                remote_url=source["remote"].resolve().as_uri(),
                protocol="file",
            )

        with mock.patch.multiple(
            renderer,
            REPOSITORY=source["repository"],
            PUBLICATION_COMPILER_PATH=source["compiler_path"],
            RENDERER_SOURCE_PATH=source["renderer_path"],
            CONTRACT_PATH=source["contract_path"],
        ), self.assertRaisesRegex(
            renderer.FigureRenderContractError,
            "stable control ref|isolated source graph",
        ):
            renderer._authenticate_renderer_sources(
                source["receipt"], source_graph_factory=unpublished_factory
            )


if __name__ == "__main__":
    unittest.main()
