#!/usr/bin/env python3
"""Fail-closed lineage and value-deduplication audit for retained triangle algebra."""

from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType


RESULTS_COMMIT = "12839900e5c46299e11fff4813e8602949055b86"
REPOSITORY = Path(__file__).resolve().parents[3]
OUTPUT_PATH = (
    REPOSITORY
    / "workshops/corl2026_world_models/results/"
    "retained_residual_triangle_algebra_lineage_deduplication_closure.json"
)
SPEC_DESCRIPTOR = {
    "path": "workshops/corl2026_world_models/experiments/forecast_layout/ablation_spec.json",
    "lines": 238,
    "bytes": 8590,
    "sha256": "93f9ec18ca30a549867e6a642188a7264514df34985ea3fa494b4cd5f9dd1a15",
}
ACCOUNTING_SOURCE_DESCRIPTOR = {
    "path": (
        "workshops/corl2026_world_models/results/"
        "development_annotation_selection_readiness.json"
    ),
    "commit": "1f36670a3d82841642d4ab7397324a342554b664",
    "lines": 363,
    "bytes": 13980,
    "sha256": "1ecd3be3f4a26483581454a625db17da302dc66bbbf558286dd24c30d83a70f0",
}
LINEAGE_INDEX_DESCRIPTOR = {
    "bytes": 7381,
    "sha256": "af3a60d5921a2cf97942f88aac5d2df0a59d1e1839b1037ce08015d857a73e16",
}
WORLD_PROVENANCE_DESCRIPTOR = {
    "bytes": 9573,
    "sha256": "5b5e133fce4deafd022f20200f44b38ca44c3fd7be3b8594d689f34bbf7a281c",
}
PROJECTED_PROVENANCE_DESCRIPTOR = {
    "bytes": 43458,
    "sha256": "4fe80cf34376cc28e9489f02e8bbf6eafe61855c440f5be6653122978c74fb21",
}

LINEAGE_NODES = (
    {
        "turn": "T2849",
        "commit": "711eac7c0253ddd10784d17b205b7da4397b4fb9",
        "tree": "57ca5e11185631b0fc6f8a0da716b0af3ff630ed",
        "parent": "d1b6724cc5e28b5a2dc78d4edff86d026ba0a81a",
        "audit": {
            "path": (
                "workshops/corl2026_world_models/analysis/"
                "audit_retained_center_representation_residual_difference.py"
            ),
            "lines": 1061,
            "bytes": 42188,
            "sha256": "e153680f042aefa61938a7ba9b576eace5a7e0ed9104993a500adf9647b665fa",
        },
        "report": {
            "path": (
                "workshops/corl2026_world_models/results/"
                "retained_center_representation_residual_difference_diagnostic.json"
            ),
            "lines": 2999,
            "bytes": 105930,
            "sha256": "d7b1d6c0e35b93298db8af1b7659885cd088f61a4be46e75f05b0ad172006021",
        },
    },
    {
        "turn": "T2850",
        "commit": "6cec04b0ff7bc98cec039b456b647390aa7b4acd",
        "tree": "1755eb4fc9ff1e74f617a19330daa60d36814357",
        "parent": "74d69ff3ee7b4b26328ef80d86d78d77f42b9db6",
        "audit": {
            "path": (
                "workshops/corl2026_world_models/analysis/"
                "audit_retained_pairwise_relation_reflection_residual.py"
            ),
            "lines": 1158,
            "bytes": 43931,
            "sha256": "729ab2f3303de3abc5e990523c96aec4a875c1413f3378bed8fc3a8c19d79538",
        },
        "report": {
            "path": (
                "workshops/corl2026_world_models/results/"
                "retained_pairwise_relation_reflection_residual_diagnostic.json"
            ),
            "lines": 5986,
            "bytes": 197699,
            "sha256": "4dfcc4da20526d8314f28e41931fd5d3347054b2881b03269bf172dd6ef6c183",
        },
    },
    {
        "turn": "T2851",
        "commit": "0f5312a304e119c0f5660da99b216cab5fe26831",
        "tree": "f0a3f2efba7c29653d01445df686e4e1bff7f036",
        "parent": "0fbeee46221150f85d8bea7e1e53dbcac15e58f6",
        "audit": {
            "path": (
                "workshops/corl2026_world_models/analysis/"
                "audit_retained_residual_triangle_exterior_product.py"
            ),
            "lines": 1020,
            "bytes": 41348,
            "sha256": "78aa64d769e1c7529cf0424e6dd70e2b7b20cb4d50c68b00f298ef24209f6605",
        },
        "report": {
            "path": (
                "workshops/corl2026_world_models/results/"
                "retained_residual_triangle_exterior_product_diagnostic.json"
            ),
            "lines": 3968,
            "bytes": 138932,
            "sha256": "80f6a6783e2e30466bfa20df264b9f2846110638a54fb093d603c80a7fcffbfa",
        },
    },
    {
        "turn": "T2852",
        "commit": "9d7ec27e11aac1c653e9835ba3b30370aee0280e",
        "tree": "21139dedd3ddda2481b56a67fb1ccb320d6f72fb",
        "parent": "1a5a2a4b0056881b56d69e6496f84247392658fa",
        "audit": {
            "path": (
                "workshops/corl2026_world_models/analysis/"
                "audit_retained_residual_triangle_gram_identity.py"
            ),
            "lines": 893,
            "bytes": 36897,
            "sha256": "625bbe0447ebe58c71f73a0e084fd661912e0a7c4628a124be35eb4aaf5aea8a",
        },
        "report": {
            "path": (
                "workshops/corl2026_world_models/results/"
                "retained_residual_triangle_gram_identity_diagnostic.json"
            ),
            "lines": 4025,
            "bytes": 157303,
            "sha256": "9072ecdecedaeef4715946ccdb8b08d166969d39f01dc41db74c60dd6b9c8b2c",
        },
    },
    {
        "turn": "T2853",
        "commit": "8b07203927aebbec82fb730c1a7aeac2af46e088",
        "tree": "e5fdfd9f28c12beb7f95fb2ba884b8983c4b8fd9",
        "parent": "29a9501b1c95e1d1b7a593ae5d33a8e055ae4a30",
        "audit": {
            "path": (
                "workshops/corl2026_world_models/analysis/"
                "audit_retained_residual_triangle_edge_polarization.py"
            ),
            "lines": 840,
            "bytes": 34723,
            "sha256": "7c51bf3099db1bb25852ed773718eb68c2581e3202d747a0d7f3046c4acc1e2d",
        },
        "report": {
            "path": (
                "workshops/corl2026_world_models/results/"
                "retained_residual_triangle_edge_polarization_diagnostic.json"
            ),
            "lines": 5301,
            "bytes": 188378,
            "sha256": "3ddb13226e877819629a4a560fe6e039d22ad425b86fac46a4e4848ce22538da",
        },
    },
    {
        "turn": "T2854",
        "commit": "be873e2f680b387639a0546e34c98c90cd0a3232",
        "tree": "a8499f3f5fa4658e2c12b8d9836b655c9afdf3a3",
        "parent": "38fffe509d4ed4e26bfef2f5111556dc43cfebd3",
        "audit": {
            "path": (
                "workshops/corl2026_world_models/analysis/"
                "audit_retained_residual_triangle_centered_scatter.py"
            ),
            "lines": 975,
            "bytes": 40713,
            "sha256": "7cfd34eeec39dad99cc4c8971a9cd2b5befb249597dababa6ca0c2de5ea7c782",
        },
        "report": {
            "path": (
                "workshops/corl2026_world_models/results/"
                "retained_residual_triangle_centered_scatter_diagnostic.json"
            ),
            "lines": 8709,
            "bytes": 279933,
            "sha256": "2fd1cbf76108e311d7b462fe8e709e8203ff650848fd94c5ce9e9e74d1d2b942",
        },
    },
    {
        "turn": "T2855",
        "commit": "af27b63949e31c9ea486e61bdbdf1a7ee81591e5",
        "tree": "a7c9bf92c34ee07843f5cba4195fed8319e13c04",
        "parent": "ad75a9f412bbab4d501db50be6e3bd97dc3ff324",
        "audit": {
            "path": (
                "workshops/corl2026_world_models/analysis/"
                "audit_retained_residual_triangle_centered_cross_product_identity.py"
            ),
            "lines": 1000,
            "bytes": 41982,
            "sha256": "510e9e691dcfecfcbd5493d606f95e09d82c293fe7c21a218b1de147f7003934",
        },
        "report": {
            "path": (
                "workshops/corl2026_world_models/results/"
                "retained_residual_triangle_centered_cross_product_identity_diagnostic.json"
            ),
            "lines": 7257,
            "bytes": 266773,
            "sha256": "1420d3c9499806b4c89ca851ddc2348bad2eccde7af7762af5595869fd9fd198",
        },
    },
    {
        "turn": "T2856",
        "commit": "376cc7832ee6537ca90b3776408d7f2f8e140cfe",
        "tree": "34e1bc050166552db34577ad7359adb4386ca16b",
        "parent": "9a3e24cb2cf26a18636147020e2217abdd1f450c",
        "audit": {
            "path": (
                "workshops/corl2026_world_models/analysis/"
                "audit_retained_residual_triangle_centered_product_expansion.py"
            ),
            "lines": 996,
            "bytes": 42006,
            "sha256": "a9d015bb544fa77f6e891cdee770a204ddcb74d6c3d04fe6f1d61d34e32ca625",
        },
        "report": {
            "path": (
                "workshops/corl2026_world_models/results/"
                "retained_residual_triangle_centered_product_expansion_diagnostic.json"
            ),
            "lines": 7280,
            "bytes": 263506,
            "sha256": "8b1f7101d10fcd33821f11964c8f76101f6e017c34ddfdc51fbe9af00ea782d7",
        },
    },
    {
        "turn": "T2857",
        "commit": "8bdbeda83479cf65b552a3a78aa6f15dfa2c95c0",
        "tree": "747b65513fab24706ff1dcaa269de14e5f861dce",
        "parent": "35c89d54975a655be52d0070062000612e6b2f75",
        "audit": {
            "path": (
                "workshops/corl2026_world_models/analysis/"
                "audit_retained_residual_triangle_centered_vertex_complement.py"
            ),
            "lines": 1063,
            "bytes": 45344,
            "sha256": "4b2a79a6c9fe59fba2dadfc5e216516ba0ad74692177e8463d95b681d9bfc075",
        },
        "report": {
            "path": (
                "workshops/corl2026_world_models/results/"
                "retained_residual_triangle_centered_vertex_complement_diagnostic.json"
            ),
            "lines": 17350,
            "bytes": 656145,
            "sha256": "f27c83446e7131a0bef780d9474c714a83fb3608fd329d3ce102d8bf679b5f50",
        },
    },
    {
        "turn": "T2858",
        "commit": "1f36670a3d82841642d4ab7397324a342554b664",
        "tree": "afb47c4a7c0baae18544ade0e5c496a80af5bc9c",
        "parent": "8babd0daf7580013f0bb680a822dedcb9d69655b",
        "audit": {
            "path": (
                "workshops/corl2026_world_models/analysis/"
                "audit_retained_residual_triangle_centered_bipartition_product.py"
            ),
            "lines": 1022,
            "bytes": 43650,
            "sha256": "f5e0b114e561a8e619d315ef6e2709e4eb74f3726033308eb4bddb6f702b7137",
        },
        "report": {
            "path": (
                "workshops/corl2026_world_models/results/"
                "retained_residual_triangle_centered_bipartition_product_diagnostic.json"
            ),
            "lines": 14937,
            "bytes": 566772,
            "sha256": "12b7fb74e5b296b8d224bb560232b3478cf2ae62608c12b509e1c68867f3767e",
        },
    },
)

LINEAGE_EDGES = (
    ("T2849", "T2850", "turn2849_center_representation_package"),
    ("T2850", "T2851", "turn2850_pairwise_relation_package"),
    ("T2851", "T2852", "turn2851_exterior_product_package"),
    ("T2852", "T2853", "turn2852_gram_identity_package"),
    ("T2853", "T2854", "turn2853_edge_polarization_package"),
    ("T2854", "T2855", "turn2854_centered_scatter_package"),
    ("T2855", "T2856", "turn2855_centered_component_package"),
    ("T2856", "T2857", "turn2856_centered_product_expansion_package"),
    ("T2857", "T2858", "turn2857_centered_vertex_complement_package"),
)

NODE_ROLES = {
    "T2849": "center-representation residual difference",
    "T2850": "pairwise relation-reflection residual",
    "T2851": "fixed residual-triangle edge source and exterior product",
    "T2852": "residual-triangle Gram identity",
    "T2853": "residual-triangle edge polarization",
    "T2854": "residual-triangle centered scatter",
    "T2855": "centered cross-product identity",
    "T2856": "centered product expansion",
    "T2857": "centered vertex complement",
    "T2858": "two-sided centered bipartition product",
}

EXPECTED_SCHEMAS = {
    "T2849": "wmf-retained-center-representation-residual-difference-diagnostic-v1",
    "T2850": "wmf-retained-pairwise-relation-reflection-residual-diagnostic-v1",
    "T2851": "wmf-retained-residual-triangle-exterior-product-diagnostic-v1",
    "T2852": "wmf-retained-residual-triangle-gram-identity-diagnostic-v1",
    "T2853": "wmf-retained-residual-triangle-edge-polarization-diagnostic-v1",
    "T2854": "wmf-retained-residual-triangle-centered-scatter-diagnostic-v1",
    "T2855": "wmf-retained-residual-triangle-centered-cross-product-identity-diagnostic-v1",
    "T2856": "wmf-retained-residual-triangle-centered-product-expansion-diagnostic-v1",
    "T2857": "wmf-retained-residual-triangle-centered-vertex-complement-diagnostic-v1",
    "T2858": "wmf-retained-residual-triangle-centered-bipartition-product-diagnostic-v1",
}

EXPECTED_DEDUP = {
    "T2849": (0, 0, 0, 0),
    "T2850": (0, 0, 0, 0),
    "T2851": (0, 0, 0, 0),
    "T2852": (279, 191, 0, 191),
    "T2853": (504, 274, 135, 139),
    "T2854": (1008, 524, 4, 520),
    "T2855": (648, 326, 2, 324),
    "T2856": (648, 326, 164, 162),
    "T2857": (1944, 970, 2, 968),
    "T2858": (1458, 486, 486, 0),
}

EXPECTED_BOUNDARY_COUNTS = {
    "T2849": 31,
    "T2850": 33,
    "T2851": 41,
    "T2852": 43,
    "T2853": 46,
    "T2854": 53,
    "T2855": 57,
    "T2856": 59,
    "T2857": 60,
    "T2858": 57,
}

EXPECTED_BOUNDARY_KEYS = {
    "a1_a9_result",
    "a7_result",
    "aabb_envelope_center_validated_as_human_centroid",
    "angle_or_similarity_inferred",
    "camera_crop_or_time_alignment_authenticated",
    "camera_ranking_estimated",
    "causal_effect_estimated",
    "collinearity_or_degeneracy_classified",
    "coordinate_or_frame_invariance_claimed",
    "counterfactual_or_temporal_transition_inferred",
    "covariance_or_statistical_moment_estimated",
    "cross_time_field_join_performed",
    "current_state_inferred",
    "edge_or_metric_ranking_estimated",
    "eigendecomposition_applied",
    "fitting_or_registration_performed",
    "fixture_pass_fail_classification_created",
    "fixture_reference_recovered",
    "forecast_or_persistence_metric",
    "gate_advanced",
    "general_occlusion_analyzed",
    "generated_or_executed_correspondence",
    "human_image_centroids_used",
    "image_footprint_or_object_pixels_inferred",
    "image_footprint_or_silhouette_area_measured",
    "independent_observation",
    "isolated_layout_or_command_effect_estimated",
    "labelability_estimated",
    "matrix_or_tensor_quantity_constructed",
    "measurement_feasibility_qualified",
    "motion_or_temporal_transition_inferred",
    "new_scientific_sample",
    "normalization_applied",
    "novel_numeric_evidence_created",
    "obb_or_exact_mesh_analyzed",
    "observed_centroid_or_center_estimated",
    "orientation_or_chirality_inferred",
    "physical_distance_or_length_measured",
    "physical_pairwise_interaction_inferred",
    "physical_partition_or_set_relation_inferred",
    "physical_plane_normal_inferred",
    "physical_scatter_or_variance_measured",
    "physical_scene_triangle_area_measured",
    "principal_axes_or_eigenvalues_computed",
    "projected_point_center_validated_as_human_centroid",
    "r0_rhat_r_endpoint",
    "raw_camera_bytes_read",
    "raw_pvc_bytes_read",
    "release_authority",
    "representation_ranking_estimated",
    "rgb_or_image_content_analyzed",
    "robot_or_static_scene_occlusion_analyzed",
    "scientific_threshold_applied",
    "silhouette_or_segmentation_analyzed",
    "spectral_property_inferred",
    "square_root_applied",
    "statistical_variance_or_population_moment_estimated",
    "task_success_or_model_rate",
    "tensor_or_physical_inertia_interpreted",
    "translation_invariance_claimed",
    "triangle_inequality_or_metric_classified",
    "vector_cross_product_computed",
    "vertexwise_physical_relation_inferred",
    "visibility_inferred",
}

ACTIVITY_FIELDS = (
    "new_behavioral_cells",
    "new_model_requests",
    "new_actions",
    "new_observations",
    "new_request_selections",
    "labels_created",
)
APPROXIMATE_KEYS = {
    "approximate_m",
    "approximate_m2",
    "approximate_m4",
    "approximate_mm4",
    "approximate_px",
    "approximate_px2",
    "approximate_px4",
}
WORLD_FAMILIES = (
    "center_relation_reflection_residual",
    "aabb_midpoint_relation_reflection_residual",
    "center_minus_aabb_midpoint_relation_residual_difference",
)
PROJECTED_FAMILIES = (
    "point_center_relation_reflection_residual",
    "envelope_center_relation_reflection_residual",
    "point_minus_envelope_relation_residual_difference",
)
OBJECT_PAIR_TO_EDGE = {
    ("banana", "bowl"): "ab",
    ("banana", "rubiks_cube"): "ac",
    ("bowl", "rubiks_cube"): "bc",
}
SENSITIVE_KEY_PARTS = (
    "authorization",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)
SENSITIVE_VALUE_PATTERNS = (
    "-----BEGIN PRIVATE KEY-----",
    "-----BEGIN OPENSSH PRIVATE KEY-----",
    "Bearer ",
    "github_pat_",
    "ghp_",
    "AKIA",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def strict_json_loads(payload: bytes, source: str) -> dict:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant in {source}: {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        require(math.isfinite(parsed), f"Non-finite JSON number in {source}: {value}")
        return parsed

    def unique_pairs(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key in {source}: {key}")
            result[key] = value
        return result

    parsed = json.loads(
        payload.decode("utf-8"),
        parse_constant=reject_constant,
        parse_float=finite_float,
        object_pairs_hook=unique_pairs,
    )
    require(isinstance(parsed, dict), f"Top-level JSON object required: {source}")
    return parsed


def git_bytes(commit: str, path: str) -> bytes:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY),
            "--no-replace-objects",
            "show",
            f"{commit}:{path}",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def git_line(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPOSITORY), "--no-replace-objects", *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def verify_payload(payload: bytes, descriptor: dict, label: str) -> None:
    require(len(payload) == descriptor["bytes"], f"{label} byte count changed")
    require(payload.count(b"\n") == descriptor["lines"], f"{label} line count changed")
    require(
        hashlib.sha256(payload).hexdigest() == descriptor["sha256"],
        f"{label} SHA-256 changed",
    )


def import_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    require(
        specification is not None and specification.loader is not None,
        f"Cannot load {name}",
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def finite_number(value: object, label: str) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"Finite number required: {label}",
    )
    return float(value)


def finite_vector(value: object, length: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == length, f"Vector changed: {label}")
    return [finite_number(item, f"{label}[{index}]") for index, item in enumerate(value)]


def descriptor_projection(node: dict) -> dict:
    return {
        "commit": node["commit"],
        "tree": node["tree"],
        "parent": node["parent"],
        "audit": node["audit"],
        "report": node["report"],
        "canonical_regeneration_verified": True,
    }


def verify_lineage_inputs() -> dict[str, dict]:
    index_bytes = canonical_bytes(list(LINEAGE_NODES))
    require(
        len(index_bytes) == LINEAGE_INDEX_DESCRIPTOR["bytes"],
        "Lineage descriptor index byte count changed",
    )
    require(
        hashlib.sha256(index_bytes).hexdigest() == LINEAGE_INDEX_DESCRIPTOR["sha256"],
        "Lineage descriptor index SHA-256 changed",
    )
    require(len(LINEAGE_NODES) == 10, "Lineage node count changed")
    require(len(LINEAGE_EDGES) == 9, "Lineage edge count changed")
    require(
        [node["turn"] for node in LINEAGE_NODES]
        == [f"T{turn}" for turn in range(2849, 2859)],
        "Lineage turn order changed",
    )
    expected_edge_pairs = [
        (f"T{turn}", f"T{turn + 1}") for turn in range(2849, 2858)
    ]
    observed_edge_pairs = [(source, target) for source, target, _ in LINEAGE_EDGES]
    require(observed_edge_pairs == expected_edge_pairs, "Ordered adjacent edge chain changed")
    require(
        len({source for source, _ in observed_edge_pairs}) == 9
        and len({target for _, target in observed_edge_pairs}) == 9,
        "Lineage edge uniqueness changed",
    )

    reports = {}
    for node in LINEAGE_NODES:
        turn = node["turn"]
        commit = node["commit"]
        require(
            git_line("rev-parse", f"{commit}^{{commit}}") == commit,
            f"{turn} evidence commit does not resolve exactly",
        )
        require(
            git_line("show", "-s", "--format=%T", commit) == node["tree"],
            f"{turn} evidence tree changed",
        )
        require(
            git_line("show", "-s", "--format=%P", commit) == node["parent"],
            f"{turn} evidence parent changed",
        )
        for kind in ("audit", "report"):
            descriptor = node[kind]
            committed = git_bytes(commit, descriptor["path"])
            verify_payload(committed, descriptor, f"Pinned {turn} {kind}")
            local = (REPOSITORY / descriptor["path"]).read_bytes()
            verify_payload(local, descriptor, f"Local {turn} {kind}")
            require(local == committed, f"Local {turn} {kind} differs from pinned blob")
            if kind == "report":
                report = strict_json_loads(committed, descriptor["path"])
                require(
                    report.get("schema_version") == EXPECTED_SCHEMAS[turn],
                    f"{turn} schema changed",
                )
                require(report.get("results_commit") == RESULTS_COMMIT, f"{turn} results pin changed")
                reports[turn] = report

    require(
        isinstance(reports["T2849"].get("turn2848_residual_decomposition_package"), dict),
        "Scoped T2849 root upstream dependency changed",
    )

    node_by_turn = {node["turn"]: node for node in LINEAGE_NODES}
    edge_rows = []
    for source, target, package_key in LINEAGE_EDGES:
        package = reports[target].get(package_key)
        require(isinstance(package, dict), f"{target} predecessor package missing")
        expected = descriptor_projection(node_by_turn[source])
        require(
            {key: package.get(key) for key in expected} == expected,
            f"{target} predecessor descriptor changed",
        )
        edge_rows.append(
            {
                "source": source,
                "target": target,
                "target_package_key": package_key,
                "predecessor_descriptor_revalidated": True,
            }
        )

    tip = LINEAGE_NODES[-1]
    tip_module = import_module(
        REPOSITORY / tip["audit"]["path"],
        "wmf_pinned_turn2858_bipartition_product",
    )
    regenerated_tip = tip_module.audit()
    require(
        tip_module.canonical_bytes(regenerated_tip)
        == git_bytes(tip["commit"], tip["report"]["path"]),
        "Turn2858 canonical regeneration changed",
    )
    require(regenerated_tip == reports["T2858"], "Turn2858 parsed report differs")
    return {"reports": reports, "edge_rows": edge_rows}


def verify_spec() -> dict:
    committed = git_bytes(RESULTS_COMMIT, SPEC_DESCRIPTOR["path"])
    verify_payload(committed, SPEC_DESCRIPTOR, "Pinned ablation specification")
    local = (REPOSITORY / SPEC_DESCRIPTOR["path"]).read_bytes()
    verify_payload(local, SPEC_DESCRIPTOR, "Local ablation specification")
    require(local == committed, "Local ablation specification differs from results pin")
    parsed = strict_json_loads(committed, SPEC_DESCRIPTOR["path"])
    require(parsed.get("core_episode_ceiling") == 232, "Core episode ceiling changed")
    require(parsed.get("launch_ready") is False, "Launch readiness changed")
    require(
        parsed.get("counts_do_not_authorize_execution") is True,
        "Count authorization boundary changed",
    )
    return {
        **SPEC_DESCRIPTOR,
        "commit": RESULTS_COMMIT,
        "core_episode_ceiling": 232,
        "launch_ready": False,
        "counts_do_not_authorize_execution": True,
    }


def verify_frozen_science_accounting() -> dict:
    descriptor = ACCOUNTING_SOURCE_DESCRIPTOR
    committed = git_bytes(descriptor["commit"], descriptor["path"])
    verify_payload(committed, descriptor, "Pinned frozen accounting source")
    local = (REPOSITORY / descriptor["path"]).read_bytes()
    verify_payload(local, descriptor, "Local frozen accounting source")
    require(local == committed, "Local frozen accounting source differs from pinned blob")
    parsed = strict_json_loads(committed, descriptor["path"])
    require(
        parsed.get("schema_version") == "wmf-development-annotation-selection-readiness-v1",
        "Frozen accounting source schema changed",
    )
    expected_snapshot = {
        "active_launched_cells": 0,
        "confirmation_inference_cells": 0,
        "launched_core_cells": 44,
        "nonbehavioral_requests": 18,
        "planned_core_cells": 232,
        "technical_invalid_behavioral_requests": 2,
        "terminal_requests": 1460,
        "terminal_technical_invalid_cells": 4,
        "terminal_valid_cells": 40,
        "valid_behavioral_actions": 18000,
        "valid_behavioral_observations": 18040,
        "valid_behavioral_requests": 1440,
    }
    require(
        parsed.get("global_study_snapshot") == expected_snapshot,
        "Frozen global study snapshot changed",
    )
    design = parsed.get("design_contract")
    require(
        isinstance(design, dict)
        and design.get("confirmation_cells") == 192
        and design.get("core_cell_ceiling") == 232
        and design.get("counts_authorize_execution") is False,
        "Frozen accounting design boundary changed",
    )
    return {
        "source": descriptor,
        "source_schema_version": parsed["schema_version"],
        "snapshot": expected_snapshot,
        "confirmation_design_cells": 192,
        "core_cell_ceiling": 232,
        "counts_authorize_execution": False,
        "accounting_changed_by_this_audit": False,
        "operational_reconciliation_performed": False,
        "scope": "Frozen source snapshot at the pinned Turn2858 evidence commit, not a live-state refresh.",
    }


def verify_activity_boundaries(reports: dict[str, dict]) -> dict:
    boundary_union = set()
    total_boundaries = 0
    root_false_counts = {}
    for turn, report in reports.items():
        boundaries = report.get("evidence_boundaries")
        require(isinstance(boundaries, dict), f"{turn} evidence boundaries missing")
        require(
            len(boundaries) == EXPECTED_BOUNDARY_COUNTS[turn],
            f"{turn} evidence boundary count changed",
        )
        require(
            all(value is False for value in boundaries.values()),
            f"{turn} evidence boundary promoted",
        )
        boundary_union.update(boundaries)
        total_boundaries += len(boundaries)

        for field in ACTIVITY_FIELDS:
            require(type(report.get(field)) is int and report[field] == 0, f"{turn} {field} changed")

        root_true = {key for key, value in report.items() if type(value) is bool and value}
        expected_true = set() if turn == "T2849" else {f"algebraically_derived_from_turn{int(turn[1:]) - 1}"}
        require(root_true == expected_true, f"{turn} root true-boolean set changed")
        root_false = {key for key, value in report.items() if type(value) is bool and not value}
        require(
            {"forecast_accuracy_estimated", "resource_budget_qualified", "confirmation_released"}
            <= root_false,
            f"{turn} root science gates changed",
        )
        root_false_counts[turn] = len(root_false)

        selection = report.get("selection")
        require(isinstance(selection, dict), f"{turn} selection missing")
        require(selection.get("availability_selected") is True, f"{turn} selection basis changed")
        require(
            selection.get("D01_D02_command_confounded") is True
            and selection.get("P00_shared_left_command_with_extra_reflected_right_member") is True,
            f"{turn} confounding boundary changed",
        )
        require(
            selection.get("representative_sample") is False
            and selection.get("full_completed_cohort") is False
            and selection.get("cross_time_field_join_performed") is False,
            f"{turn} selection claim changed",
        )
        isolated_key = (
            "isolated_layout_effect_estimated"
            if turn == "T2849"
            else "isolated_layout_or_command_effect_estimated"
        )
        require(selection.get(isolated_key) is False, f"{turn} isolation claim changed")
        if "outcome_dependent_selection" in selection:
            require(selection["outcome_dependent_selection"] is False, f"{turn} outcome selection changed")

    require(total_boundaries == 480, "Total evidence boundary assertions changed")
    require(boundary_union == EXPECTED_BOUNDARY_KEYS, "Evidence boundary union changed")
    return {
        "per_node_evidence_boundary_counts": EXPECTED_BOUNDARY_COUNTS,
        "total_false_evidence_boundary_assertions": total_boundaries,
        "union_boundary_key_count": len(boundary_union),
        "union_boundary_keys": sorted(boundary_union),
        "activity_counter_fields_per_node": list(ACTIVITY_FIELDS),
        "total_exact_zero_activity_counters": len(reports) * len(ACTIVITY_FIELDS),
        "root_false_boolean_counts": root_false_counts,
        "all_activity_counters_zero": True,
        "all_evidence_boundaries_false": True,
        "all_science_and_release_gates_unadvanced": True,
    }


def provenance_context(provenance: dict, projected: bool) -> tuple:
    require(isinstance(provenance, dict), "Source provenance missing")
    base = (
        provenance.get("phase"),
        provenance.get("layout_pair_id"),
        provenance.get("model_id"),
    )
    return base + ((provenance.get("camera_name"),) if projected else ())


def verify_source_reorganization(reports: dict[str, dict]) -> dict:
    turn2850 = reports["T2850"]
    turn2851 = reports["T2851"]
    source_slot_values = []
    cycle_components = 0
    base_operand_slots = 0
    row_keys = set()

    configurations = (
        (
            False,
            "world_relation_rows",
            "world_triangle_rows",
            WORLD_FAMILIES,
            "m",
            3,
            9,
        ),
        (
            True,
            "projected_relation_rows",
            "projected_determinant_rows",
            PROJECTED_FAMILIES,
            "px",
            2,
            36,
        ),
    )
    for projected, source_key, triangle_key, families, unit, length, expected_rows in configurations:
        source_rows = turn2850.get(source_key)
        triangle_rows = turn2851.get(triangle_key)
        require(
            isinstance(source_rows, list) and len(source_rows) == expected_rows,
            f"Turn2850 {source_key} changed",
        )
        require(
            isinstance(triangle_rows, list) and len(triangle_rows) == expected_rows,
            f"Turn2851 {triangle_key} changed",
        )
        grouped = {}
        for source_index, source_row in enumerate(source_rows):
            require(isinstance(source_row, dict), f"Turn2850 {source_key}[{source_index}] missing")
            provenance = source_row.get("provenance")
            require(isinstance(provenance, dict), "Turn2850 source provenance missing")
            require(provenance.get("pair_direction") == "first_minus_second", "Pair direction changed")
            pair = (provenance.get("first_object_name"), provenance.get("second_object_name"))
            require(pair in OBJECT_PAIR_TO_EDGE, f"Unexpected source object pair: {pair}")
            edge = OBJECT_PAIR_TO_EDGE[pair]
            context = provenance_context(provenance, projected)
            for family in families:
                payload = source_row.get(family)
                require(isinstance(payload, dict), f"Turn2850 {family} payload missing")
                vector = finite_vector(
                    payload.get(f"signed_{unit}"),
                    length,
                    f"Turn2850 {source_key}[{source_index}].{family}",
                )
                key = context + (family,)
                grouped.setdefault(key, {})
                require(edge not in grouped[key], f"Duplicate Turn2850 {edge} row")
                grouped[key][edge] = vector
                source_slot_values.extend((unit, Fraction.from_float(item)) for item in vector)
        require(len(grouped) == expected_rows, f"Turn2850 {source_key} context count changed")
        require(all(set(edges) == {"ab", "ac", "bc"} for edges in grouped.values()), "Edge set changed")

        for row_index, row in enumerate(triangle_rows):
            require(isinstance(row, dict), f"Turn2851 {triangle_key}[{row_index}] missing")
            provenance = row.get("provenance")
            require(isinstance(provenance, dict), "Turn2851 provenance missing")
            family = row.get("family")
            require(family in families, "Turn2851 family changed")
            key = provenance_context(provenance, projected) + (family,)
            require(key in grouped, "Turn2851 source context not found")
            require(key not in row_keys, "Duplicate Turn2851 triangle row key")
            row_keys.add(key)
            require(
                provenance.get("object_order") == ["banana", "bowl", "rubiks_cube"]
                and provenance.get("edge_ab") == "banana_minus_bowl"
                and provenance.get("edge_ac") == "banana_minus_rubiks_cube"
                and provenance.get("edge_bc") == "bowl_minus_rubiks_cube",
                "Turn2851 object or edge order changed",
            )
            exact_edges = {}
            for edge in ("ab", "ac", "bc"):
                vector = finite_vector(
                    row.get(f"edge_{edge}_signed_{unit}"),
                    length,
                    f"Turn2851 {triangle_key}[{row_index}].edge_{edge}",
                )
                require(vector == grouped[key][edge], f"Turn2851 {edge} source value changed")
                exact_edges[edge] = [Fraction.from_float(item) for item in vector]
            for component in range(length):
                require(
                    exact_edges["ab"][component]
                    + exact_edges["bc"][component]
                    - exact_edges["ac"][component]
                    == 0,
                    "Exact residual-triangle edge cycle changed",
                )
                cycle_components += 1
            base_operand_slots += 2 * length
            require(row.get("pair_cycle_identity_exact") is True, "Turn2851 cycle flag changed")

    require(len(row_keys) == 45, "Turn2851 unique row key count changed")
    require(len(source_slot_values) == 297, "Signed source slot count changed")
    require(len(set(source_slot_values)) == 291, "Distinct signed source value count changed")
    require(sum(value == 0 for _, value in source_slot_values) == 2, "Source zero slot count changed")
    require(cycle_components == 99, "Exact cycle component count changed")
    require(base_operand_slots == 198, "Base AB/AC operand slot count changed")

    world_provenance = [row["provenance"] for row in turn2851["world_triangle_rows"]]
    projected_provenance = [row["provenance"] for row in turn2851["projected_determinant_rows"]]
    for value, descriptor, label in (
        (world_provenance, WORLD_PROVENANCE_DESCRIPTOR, "World provenance"),
        (projected_provenance, PROJECTED_PROVENANCE_DESCRIPTOR, "Projected provenance"),
    ):
        encoded = canonical_bytes(value)
        require(len(encoded) == descriptor["bytes"], f"{label} byte count changed")
        require(hashlib.sha256(encoded).hexdigest() == descriptor["sha256"], f"{label} changed")

    return {
        "turn2850_world_relation_rows": 9,
        "turn2850_projected_relation_rows": 36,
        "turn2851_world_triangle_rows": 9,
        "turn2851_projected_determinant_rows": 36,
        "unique_context_family_row_keys": 45,
        "signed_source_value_slots": 297,
        "distinct_dimension_aware_signed_source_values": 291,
        "signed_source_zero_slots": 2,
        "exact_edge_cycle_coordinate_components": 99,
        "world_exact_edge_cycle_coordinate_components": 27,
        "projected_exact_edge_cycle_coordinate_components": 72,
        "base_ab_ac_operand_slots": 198,
        "world_provenance_list": WORLD_PROVENANCE_DESCRIPTOR,
        "projected_provenance_list": PROJECTED_PROVENANCE_DESCRIPTOR,
        "object_order": ["banana", "bowl", "rubiks_cube"],
        "edge_mapping": {
            "ab": "banana_minus_bowl",
            "ac": "banana_minus_rubiks_cube",
            "bc": "bowl_minus_rubiks_cube",
        },
        "source_reorganization_exactly_revalidated": True,
    }


def rational_fingerprint(record: dict, label: str) -> tuple[str, int, int]:
    approximate_keys = [key for key in record if key.startswith("approximate_")]
    require(len(approximate_keys) == 1, f"Exact record approximate key changed: {label}")
    approximate_key = approximate_keys[0]
    require(approximate_key in APPROXIMATE_KEYS, f"Unexpected exact-record dimension: {label}")
    require(
        set(record) == {"numerator", "denominator", approximate_key},
        f"Exact record keys changed: {label}",
    )
    numerator = record["numerator"]
    denominator = record["denominator"]
    require(
        isinstance(numerator, str) and re.fullmatch(r"0|-?[1-9][0-9]*", numerator),
        f"Noncanonical numerator: {label}",
    )
    require(
        isinstance(denominator, str) and re.fullmatch(r"[1-9][0-9]*", denominator),
        f"Noncanonical denominator: {label}",
    )
    value = Fraction(int(numerator), int(denominator))
    require(
        numerator == str(value.numerator) and denominator == str(value.denominator),
        f"Unreduced exact record: {label}",
    )
    approximate = finite_number(record[approximate_key], f"{label}.{approximate_key}")
    require(approximate == float(value), f"Exact-record display value changed: {label}")
    if value == 0:
        require(
            math.copysign(1.0, approximate) == 1.0,
            f"Canonical positive display zero required: {label}",
        )
    return approximate_key, value.numerator, value.denominator


def expected_dimension_key(
    field_name: str,
    ancestor_fields: tuple[str, ...],
    label: str,
) -> tuple[str, str]:
    suffix_pattern = re.compile(r"_(mm4|px4|m4|px2|m2|px|m)$")
    direct = suffix_pattern.search(field_name)
    if direct is not None:
        return f"approximate_{direct.group(1)}", "containing_field_suffix"
    for ancestor in reversed(ancestor_fields):
        inherited = suffix_pattern.search(ancestor)
        if inherited is not None:
            return f"approximate_{inherited.group(1)}", "nearest_unit_bearing_ancestor"
    raise ValueError(f"Exact record lacks a unit-bearing field context: {label}")


def collect_exact_records(
    value: object,
) -> tuple[list[tuple[str, int, int]], list[str], list[str]]:
    records = []
    field_names = []
    binding_modes = []

    def visit(
        item: object,
        path: str,
        field_name: str | None,
        ancestor_fields: tuple[str, ...],
    ) -> None:
        if isinstance(item, dict):
            approximate = [key for key in item if key.startswith("approximate_")]
            exact_candidate = (
                len(item) == 3
                and "numerator" in item
                and "denominator" in item
                and len(approximate) == 1
            )
            if exact_candidate:
                require(field_name is not None, f"Exact record has no containing field: {path}")
                fingerprint = rational_fingerprint(item, path)
                expected_key, binding_mode = expected_dimension_key(
                    field_name,
                    ancestor_fields,
                    path,
                )
                require(
                    fingerprint[0] == expected_key,
                    f"Exact record dimension key does not match serialization context: "
                    f"{path}: expected {expected_key}, found {fingerprint[0]}",
                )
                records.append(fingerprint)
                field_names.append(field_name)
                binding_modes.append(binding_mode)
                return
            for key, child in item.items():
                visit(child, f"{path}.{key}", key, ancestor_fields + (key,))
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]", field_name, ancestor_fields)

    visit(value, "$", None, ())
    return records, field_names, binding_modes


def verify_dimension_mutation_guards(turn2858: dict) -> dict:
    source_path = (
        "$.projected_centered_bipartition_product_rows[0]."
        "exact_centered_bipartition_product_components_px2[0]."
        "bipartition_product_identities[0]."
        "bipartition_product_identity_residual_px2"
    )

    swapped = copy.deepcopy(turn2858)
    swapped_record = swapped["projected_centered_bipartition_product_rows"][0][
        "exact_centered_bipartition_product_components_px2"
    ][0]["bipartition_product_identities"][0][
        "bipartition_product_identity_residual_px2"
    ]
    require(
        swapped_record == {
            "numerator": "0",
            "denominator": "1",
            "approximate_px2": 0.0,
        },
        "Turn2858 mutation-fixture zero record changed",
    )
    swapped_record["approximate_m2"] = swapped_record.pop("approximate_px2")
    try:
        collect_exact_records(swapped)
    except ValueError as error:
        require(
            "dimension key does not match serialization context" in str(error),
            "Cross-dimension zero-key mutation failed for an unexpected reason",
        )
    else:
        raise ValueError("Cross-dimension zero-key mutation was accepted")

    negative_zero = copy.deepcopy(turn2858)
    negative_zero_record = negative_zero["projected_centered_bipartition_product_rows"][0][
        "exact_centered_bipartition_product_components_px2"
    ][0]["bipartition_product_identities"][0][
        "bipartition_product_identity_residual_px2"
    ]
    negative_zero_record["approximate_px2"] = -0.0
    try:
        collect_exact_records(negative_zero)
    except ValueError as error:
        require(
            "Canonical positive display zero required" in str(error),
            "Negative display zero mutation failed for an unexpected reason",
        )
    else:
        raise ValueError("Negative display zero mutation was accepted")

    return {
        "fixture_path": source_path,
        "fixture_exact_value": "0/1",
        "expected_dimension_key": "approximate_px2",
        "cross_dimension_replacement_key": "approximate_m2",
        "cross_dimension_zero_key_swap_rejected": True,
        "negative_display_zero_rejected": True,
        "positive_display_zero_required": True,
    }


def fingerprint_set_descriptor(values: set[tuple[str, int, int]]) -> dict:
    serializable = [
        {"dimension": dimension, "numerator": str(numerator), "denominator": str(denominator)}
        for dimension, numerator, denominator in sorted(values)
    ]
    encoded = canonical_bytes(serializable)
    return {
        "unique_values": len(values),
        "canonical_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def verify_exact_record_deduplication(reports: dict[str, dict]) -> dict:
    prior_union = set()
    per_node = []
    all_field_names = set()
    all_field_dimension_pairs = set()
    all_records = []
    all_binding_modes = []
    per_node_sets = {}
    for turn in [node["turn"] for node in LINEAGE_NODES]:
        records, field_names, binding_modes = collect_exact_records(reports[turn])
        node_set = set(records)
        seen = node_set & prior_union
        first = node_set - prior_union
        expected = EXPECTED_DEDUP[turn]
        observed = (len(records), len(node_set), len(seen), len(first))
        require(observed == expected, f"{turn} exact-record deduplication counts changed")
        prior_union.update(node_set)
        all_records.extend(records)
        all_field_names.update(field_names)
        all_field_dimension_pairs.update(
            (field_name, record[0]) for field_name, record in zip(field_names, records)
        )
        all_binding_modes.extend(binding_modes)
        per_node_sets[turn] = node_set
        per_node.append(
            {
                "turn": turn,
                "serialized_slots": len(records),
                "unique_dimension_aware_values_in_node": len(node_set),
                "seen_in_prior_union": len(seen),
                "first_serialized_in_lineage_here": len(first),
                "node_unique_set": fingerprint_set_descriptor(node_set),
                "cumulative_unique_set": fingerprint_set_descriptor(prior_union),
            }
        )

    require(len(all_records) == 6489, "Total exact-record slot count changed")
    require(sum(row["unique_dimension_aware_values_in_node"] for row in per_node) == 3097, "Per-node unique sum changed")
    require(len(prior_union) == 2304, "Cumulative dimension-aware unique count changed")
    require(len(all_records) - len(prior_union) == 4185, "Repeated exact-record occurrence count changed")
    require(len(all_field_names) == 77, "Exact-record field-name count changed")
    require(len(all_field_dimension_pairs) == 83, "Field/dimension binding count changed")
    require(
        all_binding_modes.count("containing_field_suffix") == 5895
        and all_binding_modes.count("nearest_unit_bearing_ancestor") == 594,
        "Exact-record dimension binding mode counts changed",
    )
    require(sum(numerator == 0 for _, numerator, _ in all_records) == 2664, "Zero record slot count changed")
    unique_zero_values = len({record for record in prior_union if record[1] == 0})
    require(
        unique_zero_values == 6,
        "Dimension-aware zero record count changed",
    )
    dimension_erased_unique = len(
        {(numerator, denominator) for _, numerator, denominator in prior_union}
    )
    require(
        dimension_erased_unique == 2299,
        "Dimension-collapsed secondary count changed",
    )
    require(
        len(prior_union) - dimension_erased_unique == 5 and unique_zero_values - 1 == 5,
        "Dimension-aware mutation guard changed",
    )
    require(
        {dimension for dimension, _, _ in prior_union} == APPROXIMATE_KEYS,
        "Exact-record dimension set changed",
    )
    require(
        per_node_sets["T2858"] <= per_node_sets["T2857"],
        "Turn2858 value set no longer closes inside Turn2857",
    )

    _, turn2858_field_names, _ = collect_exact_records(reports["T2858"])
    complement_product_roles = sum(
        name in {
            "two_vertex_complement_sum_product_m2",
            "two_vertex_complement_sum_product_px2",
        }
        for name in turn2858_field_names
    )
    residual_roles = sum(
        name in {
            "bipartition_product_identity_residual_m2",
            "bipartition_product_identity_residual_px2",
        }
        for name in turn2858_field_names
    )
    focal_roles = sum(
        name in {"focal_vertex_product_m2", "focal_vertex_product_px2"}
        for name in turn2858_field_names
    )
    require(
        (complement_product_roles, residual_roles, focal_roles) == (486, 486, 486),
        "Turn2858 semantic role counts changed",
    )

    return {
        "fingerprint_definition": (
            "(sole approximate_* dimension key, reduced exact numerator, positive exact denominator); "
            "display floats are validated but are not identity keys"
        ),
        "per_node": per_node,
        "total_serialized_exact_rational_record_slots": 6489,
        "sum_of_per_node_unique_counts": 3097,
        "cumulative_dimension_aware_unique_values": 2304,
        "repeated_serialized_slot_occurrences": 4185,
        "distinct_exact_record_field_names": 77,
        "distinct_field_dimension_bindings": 83,
        "containing_field_suffix_bound_record_slots": 5895,
        "nearest_unit_bearing_ancestor_bound_record_slots": 594,
        "all_exact_record_dimension_keys_bound_to_serialization_context": True,
        "exact_record_dimension_keys": sorted(APPROXIMATE_KEYS),
        "zero_record_slots": 2664,
        "dimension_aware_unique_zero_values": 6,
        "secondary_numerator_denominator_only_unique_values": 2299,
        "dimension_erasure_unique_value_delta": 5,
        "dimension_sensitive_zero_value_delta": 5,
        "secondary_dimension_collapsed_count_controls_no_claim": True,
        "turn2858": {
            "serialized_slots": 1458,
            "unique_dimension_aware_values": 486,
            "unique_values_seen_in_prior_lineage": 486,
            "first_serialized_in_lineage_here": 0,
            "new_semantic_product_or_residual_roles": 972,
            "inherited_focal_product_roles": 486,
            "unique_value_set_is_subset_of_turn2857": True,
        },
        "canonical_integer_positive_denominator_and_reduction_checks_passed": True,
        "all_display_numbers_finite_and_exactly_matched": True,
    }


def assert_output_hygiene(value: object) -> None:
    def visit(item: object, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
                require(
                    not any(part in normalized for part in SENSITIVE_KEY_PARTS),
                    f"Sensitive key forbidden in output: {path}.{key}",
                )
                visit(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        elif isinstance(item, str):
            require("\x00" not in item, f"NUL forbidden in output: {path}")
            require(
                not item.startswith(("/", "~", "file://", "ssh://")),
                f"Absolute or raw path forbidden in output: {path}",
            )
            require(
                re.match(r"^[A-Za-z]:[\\/]", item) is None,
                f"Drive path forbidden in output: {path}",
            )
            require(
                "/raw/" not in item and "\\raw\\" not in item,
                f"Raw path forbidden in output: {path}",
            )
            require(
                not any(pattern in item for pattern in SENSITIVE_VALUE_PATTERNS),
                f"Sensitive value forbidden in output: {path}",
            )
            require(
                re.search(r"(?:^|[^A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}", item) is None,
                f"Sensitive value forbidden in output: {path}",
            )

    visit(value, "$")


def audit() -> dict:
    lineage = verify_lineage_inputs()
    reports = lineage["reports"]
    spec = verify_spec()
    activity_boundaries = verify_activity_boundaries(reports)
    source_closure = verify_source_reorganization(reports)
    deduplication = verify_exact_record_deduplication(reports)
    dimension_mutation_guards = verify_dimension_mutation_guards(reports["T2858"])
    frozen_accounting = verify_frozen_science_accounting()

    audit_bytes = sum(node["audit"]["bytes"] for node in LINEAGE_NODES)
    report_bytes = sum(node["report"]["bytes"] for node in LINEAGE_NODES)
    audit_lines = sum(node["audit"]["lines"] for node in LINEAGE_NODES)
    report_lines = sum(node["report"]["lines"] for node in LINEAGE_NODES)
    require((audit_bytes, report_bytes) == (412782, 2821371), "Lineage byte totals changed")
    require((audit_lines, report_lines) == (10028, 77812), "Lineage line totals changed")

    output_boundaries = {
        key: False
        for key in sorted(
            EXPECTED_BOUNDARY_KEYS
            | {
                "algebraic_role_count_is_sample_size",
                "mathematical_exhaustion_claimed",
                "scoped_value_closure_is_scientific_closure",
            }
        )
    }
    require(len(output_boundaries) == 67, "Output evidence boundary count changed")

    report = {
        "schema_version": "wmf-retained-residual-triangle-algebra-lineage-deduplication-closure-v1",
        "results_commit": RESULTS_COMMIT,
        "finding": (
            "The pinned Turn2849-Turn2858 algebraic diagnostic lineage is a 10-node, 9-edge "
            "deterministic algebra DAG over 45 retained context-family rows. Across 6,489 exact "
            "rational record slots it contains 2,304 dimension-aware values; Turn2858 serializes "
            "1,458 slots but its 486 unique values are all already present in Turn2857. Thus "
            "Turn2858 adds zero values and cannot be presented as new numerical or scientific "
            "evidence within the frozen lineage, without claiming that future rational operations "
            "cannot create derived values, mathematical exhaustion or any gate advance."
        ),
        "lineage_manifest": {
            "nodes": list(LINEAGE_NODES),
            "node_roles": NODE_ROLES,
            "edges": lineage["edge_rows"],
            "node_count": 10,
            "edge_count": 9,
            "scope_root": "T2849",
            "scope_tip": "T2858",
            "scope_root_has_upstream_dependency": "T2848",
            "full_transitive_lineage_closure_claimed": False,
            "dependency_semantics": (
                "Adjacent algebraic diagnostic package dependencies; package evidence commits may have "
                "intervening seal commits and are not asserted to be one another's Git parents."
            ),
            "descriptor_index": LINEAGE_INDEX_DESCRIPTOR,
            "file_count": 20,
            "audit_file_count": 10,
            "report_file_count": 10,
            "audit_bytes": audit_bytes,
            "report_bytes": report_bytes,
            "total_bytes": audit_bytes + report_bytes,
            "audit_lines": audit_lines,
            "report_lines": report_lines,
            "total_lines": audit_lines + report_lines,
            "all_blobs_match_local_and_pinned_commit": True,
            "tip_canonical_regeneration_verified": True,
            "tip_regeneration_recursively_revalidates_predecessors": True,
            "git_replace_objects_disabled_for_all_provenance_reads": True,
        },
        "source_and_provenance_closure": source_closure,
        "dimension_aware_exact_record_deduplication": deduplication,
        "dimension_binding_mutation_proofs": dimension_mutation_guards,
        "scoped_stopping_rule": {
            "scope": (
                "Exact rational values serialized by the pinned Turn2849-Turn2858 retained "
                "residual-triangle algebra lineage and its frozen Turn2850 signed source vectors."
            ),
            "current_tip": "T2858",
            "scoped_value_closed_at_tip": True,
            "scoped_value_closure_definition": (
                "Turn2858's unique serialized value set is contained in Turn2857 within the fixed, "
                "pinned T2849-T2858 lineage."
            ),
            "tip_unique_values": 486,
            "tip_unique_values_seen_in_prior_lineage": 486,
            "tip_first_serialized_values": 0,
            "further_same_input_algebraic_diagnostics_recommended": False,
            "recommendation_basis": (
                "Additional re-expression of the same authenticated inputs cannot be presented as "
                "new numerical or scientific evidence within this frozen lineage; this recommendation "
                "is not a claim that algebra cannot derive other rational values."
            ),
            "future_rational_operations_may_create_other_derived_values": True,
            "mathematical_exhaustion_claimed": False,
            "scientific_closure_claimed": False,
            "reopen_only_if": [
                "genuinely new authenticated root input enters scope",
                "a separately authorized non-algebraic scientific question is posed",
            ],
            "not_a_terminal_project_marker": True,
        },
        "frozen_ablation_spec_boundary": spec,
        "frozen_science_accounting": frozen_accounting,
        "audit_activity": {
            "eligibility_decisions": 0,
            "external_system_reads": 0,
            "labels_created": 0,
            "raw_camera_or_pvc_bytes_read": 0,
            "raw_camera_or_pvc_paths_opened": 0,
            "sampling_ranks_created": 0,
            "selection_manifests_created": 0,
        },
        "new_activity": {
            "behavioral_cells": 0,
            "model_requests": 0,
            "actions": 0,
            "observations": 0,
            "request_selections": 0,
            "labels_created": 0,
        },
        "operational_and_science_gates": {
            "a7_result": False,
            "confirmation_released": False,
            "forecast_accuracy_estimated": False,
            "gate_advanced": False,
            "generated_or_executed_correspondence": False,
            "included_in_existing_six_report_annotation_closure": False,
            "independent_observation": False,
            "new_scientific_sample": False,
            "release_authority": False,
        },
        "claim_boundaries": {
            "algebra_only": (
                "Every checked identity and serialized rational value is a deterministic "
                "re-expression of pinned finite stored coordinates."
            ),
            "value_novelty": (
                "First serialization in this lineage is bookkeeping, not new measurement or "
                "scientific evidence; semantic field roles do not create numerical values."
            ),
            "dimension_aware_identity": (
                "Exact values with different approximate_* unit or power keys are never collapsed."
            ),
            "source_population": (
                "The 45 context-family rows are availability-selected, command-confounded and "
                "neither representative nor a completed cohort."
            ),
            "closure_scope": (
                "The stopping rule closes only this frozen same-input algebraic value lineage; it "
                "does not close mathematics, science, annotation, inference or the study."
            ),
            "upstream_scope": (
                "T2849 retains an authenticated T2848 predecessor package. This audit intentionally "
                "closes only the requested T2849-T2858 subgraph and makes no full-transitive-root claim."
            ),
            "role_counts": (
                "Rows, components, identities, rational slots and algebraic roles are dependent "
                "audit counts, never sample sizes or independent observations."
            ),
            "science_annotation_and_release": (
                "No raw image or PVC bytes, labels, selection manifest, forecast estimate, current-"
                "state claim, annotation eligibility, dispatch authority or confirmation release follows."
            ),
        },
        "input_activity_and_boundary_revalidation": activity_boundaries,
        "evidence_boundaries": output_boundaries,
        "included_in_existing_six_report_annotation_closure": False,
        "independent_observation": False,
        "new_scientific_sample": False,
        "novel_numeric_evidence_created": False,
        "mathematical_exhaustion_claimed": False,
        "algebraic_role_count_is_sample_size": False,
        "scoped_value_closure_is_scientific_closure": False,
        "new_behavioral_cells": 0,
        "new_model_requests": 0,
        "new_actions": 0,
        "new_observations": 0,
        "new_request_selections": 0,
        "labels_created": 0,
        "forecast_accuracy_estimated": False,
        "resource_budget_qualified": False,
        "confirmation_released": False,
        "output_hygiene": {
            "absolute_or_raw_paths_included": False,
            "sensitive_material_included": False,
        },
    }
    require(
        all(type(value) is int and value == 0 for value in report["audit_activity"].values())
        and all(type(value) is int and value == 0 for value in report["new_activity"].values()),
        "Output activity changed",
    )
    require(
        len(report["audit_activity"]) + len(report["new_activity"]) == 13,
        "Output activity counter count changed",
    )
    require(
        len(report["operational_and_science_gates"]) == 9
        and all(value is False for value in report["operational_and_science_gates"].values()),
        "Output gate state changed",
    )
    require(all(value is False for value in output_boundaries.values()), "Output boundary promoted")
    assert_output_hygiene(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and close the pinned retained residual-triangle algebra lineage at its "
            "dimension-aware exact-value boundary without advancing science gates."
        )
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        type=Path,
        help="Verify the fixed canonical closure report without modifying it",
    )
    group.add_argument(
        "--write",
        action="store_true",
        help="Atomically write the canonical closure report to its fixed result path",
    )
    arguments = parser.parse_args()
    report = audit()
    encoded = canonical_bytes(report)
    if arguments.check is not None:
        require(arguments.check.resolve() == OUTPUT_PATH.resolve(), "Unexpected check path")
        saved = arguments.check.read_bytes()
        strict_json_loads(saved, str(arguments.check))
        require(saved == encoded, "Saved closure report differs from canonical audit output")
        print(
            "Verified 10 lineage nodes, 9 algebraic dependency edges, 6,489 exact-record slots, "
            "2,304 dimension-aware values and scoped zero-new-value closure at Turn2858; "
            "no science or release gate advanced."
        )
    elif arguments.write:
        temporary = OUTPUT_PATH.with_suffix(".json.tmp")
        temporary.write_bytes(encoded)
        os.replace(temporary, OUTPUT_PATH)
        print(f"Wrote {OUTPUT_PATH.relative_to(REPOSITORY)} ({len(encoded)} bytes)")
    else:
        sys.stdout.buffer.write(encoded)


if __name__ == "__main__":
    main()
