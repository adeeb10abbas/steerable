#!/usr/bin/env python3
"""Audit availability-selected settled object-center mappings without gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


RESULTS_COMMIT = "12839900e5c46299e11fff4813e8602949055b86"
REPOSITORY = Path(__file__).resolve().parents[3]
CAMERA_NAMES = (
    "head_camera",
    "over_shoulder_left_camera",
    "over_shoulder_right_camera",
    "wrist_cam",
)
OBJECT_NAMES = ("banana", "bowl", "rubiks_cube")
SNAPSHOT_BOUNDARY = (
    "Read-only progress snapshot; not a terminal receipt or scientific classification."
)
PINNED_BLOBS = {
    "N3_snapshot_002": (
        "results/jobs/detached-attempt-snapshot-002/publish/diagnostic.json",
        119722,
        "d0afbba59b68dc678bcad886900db8468fc446d8072c29077a0c55e2b70d186e",
    ),
    "N3_snapshot_003": (
        "results/jobs/detached-attempt-snapshot-003/publish/diagnostic.json",
        173811,
        "57f2a346a1bc181fe0901cd7193fd5bbf952a68aa1210b35b70cfedef6a1394d",
    ),
    "N3_terminal": (
        "results/jobs/n3-p00-pilot-004/publish/n3_behavioral_pilot_receipt.json",
        6473,
        "5a7a861acc82615b4eb7aa92308f40f7a0c965e65b6edaf12f776cb64b18b0e0",
    ),
    "D01_snapshot": (
        "results/jobs/d1-development-d01-snapshot-001/publish/diagnostic.json",
        191154,
        "4f3563f92b28e3dcf713ccdf45df43aab745fb2a91562751f78ae5b1133a6883",
    ),
    "D01_terminal": (
        "results/jobs/d1-development-d01-simulator-003/publish/d1_behavioral_development_receipt.json",
        16510,
        "9b1fd7d7479ad275d4baa943b9d8d6a4ed1d223e2d9ca20b3bff0342711ec5eb",
    ),
    "D02_snapshot": (
        "results/jobs/d1-development-d02-snapshot-001/publish/diagnostic.json",
        191132,
        "108f1d7dd1f85c8c3af6b8cbff42a39ba06521aa72bb2e4f19b8bbfcaf6c727e",
    ),
    "D02_terminal": (
        "results/jobs/d1-development-d02-simulator-003/publish/d1_behavioral_development_receipt.json",
        16510,
        "d1dc66c57d36cb3e620731c69a6e41fdf52c3cc2a797c2dae7ee36d63d10c6c9",
    ),
}
SOURCE_COMMITS = {
    "N3": "b8914413f6293c165dd3a57c737f4b977e7e84be",
    "D1": "25ff299ca0d2b9964eb48286990ee2301cb207b8",
}
SOURCE_FILE = {
    "path": (
        "workshops/corl2026_world_models/experiments/forecast_layout/"
        "robolab_fixture_gate_adapter.py"
    ),
    "git_blob": "0113723ed554a7fd2e5014606b37fc7483237813",
    "bytes": 23395,
    "sha256": "28cd6105010892dcabe2f4bfd61277f1e7cf41d0a42086f4879087f4d3268b23",
}
SPEC_SOURCE = {
    "commit": "e67e6c4f990030fa8dc9b40a7e5dc713c99457e5",
    "path": "workshops/corl2026_world_models/docs/ABLATION_SPEC.md",
    "git_blob": "aff0e42e16b4cc542b0f7ff51e2de1dce8b6dadd",
    "bytes": 27094,
    "sha256": "fa8cbbac7bf2eff2e01f73150e5cf073f7e4f177d3c0bfd4fc1f41fb3ea9309f",
}
RESET_SOURCE_FILES = {
    "N3_behavioral_job": {
        "commit": SOURCE_COMMITS["N3"],
        "path": (
            "workshops/corl2026_world_models/experiments/forecast_layout/"
            "n3_behavioral_pilot_job.py"
        ),
        "git_blob": "973000f7a17a17e5ca0c4bac9609dcdc1647cd66",
        "bytes": 110597,
        "sha256": "a648ac0e8290c5b6ed9f42b6868c9ee63b815676bb174258ecc04e064a5d8d4b",
    },
    "D1_behavioral_job": {
        "commit": SOURCE_COMMITS["D1"],
        "path": (
            "workshops/corl2026_world_models/experiments/forecast_layout/"
            "d1_behavioral_pilot_jobs.py"
        ),
        "git_blob": "b9e8eaa5eb1db9933ded1b3f7c0a8514585f19a2",
        "bytes": 157637,
        "sha256": "699b9f8c3c67f90a347191c8d9b0224dbe2c0023993d13867a75bbfd4fa70300",
    },
}
RECORDER_SOURCE_FILE = {
    "path": (
        "workshops/corl2026_world_models/experiments/forecast_layout/"
        "recorder_qualification_job.py"
    ),
    "git_blob": "f152b9df6d4febcbdd096bf3afaa1433c85cee3b",
    "bytes": 72645,
    "sha256": "df8f4d9cf05b72771baa1e2b56e5a90e1622390e01ac06ecbe2908ed2be332ab",
}
FIXTURE_SOURCE_FILE = {
    "path": (
        "workshops/corl2026_world_models/experiments/forecast_layout/"
        "fixture_tasks.py"
    ),
    "git_blob": "9d05d1f0fbc649b2529d5bcc6ab920e585cf3a47",
    "bytes": 16143,
    "sha256": "e3891120e73bd0cd313097b0c6c6d0385cdfa4f240709a3320bbbf2716f8e5ba",
}
PRIOR_PACKAGE_COMMIT = "4e706ef91d187caad397edf1eed2cfdaa03c12ef"
PRIOR_PACKAGE_PINS = {
    "retained_N3_operational_subset": (
        "workshops/corl2026_world_models/results/"
        "retained_n3_pilot_operational_subset.json",
        10620,
        "50cc87385cc5379c9f06de68d2e3a34f186bd36c838e859ac07fe3c94fd74ab4",
    ),
    "retained_success_channel_diagnostic": (
        "workshops/corl2026_world_models/results/"
        "retained_success_channel_diagnostic.json",
        22107,
        "41e757fcd8d95b0412cd2f85a02129d1fce80e213a0dd063cda618a3bbdc5bb3",
    ),
}
N3_ATTEMPT_ID = "n3-p00-pilot-004"
N3_KEYS = (
    "00-wmf1-pilot-P00-N3-reflected-right",
    "01-wmf1-pilot-P00-N3-reflected-left",
    "02-wmf1-pilot-P00-N3-original-left",
)
D1_KEYS = {
    "D01": (
        "00-wmf1-development-D01-D1-original-left",
        "01-wmf1-development-D01-D1-reflected-right",
    ),
    "D02": (
        "00-wmf1-development-D02-D1-reflected-right",
        "01-wmf1-development-D02-D1-original-left",
    ),
}
EXPECTED_OBJECT_L2_M = {
    ("N3", "P00"): {
        "banana": 0.0001423941693542533,
        "bowl": 0.0008224548670985458,
        "rubiks_cube": 0.0006503224760618562,
    },
    ("D1", "D01"): {
        "banana": 0.00014175944634172688,
        "bowl": 0.0007508789535900541,
        "rubiks_cube": 0.0006373677555323029,
    },
    ("D1", "D02"): {
        "banana": 0.00014202822372713433,
        "bowl": 0.0006890814920436979,
        "rubiks_cube": 0.0018614546594330106,
    },
}
EXPECTED_RECEIPTS = {
    "wmf1__pilot__P00__N3__reflected__right": (
        30051, "0a53e9b6df118d79a6361f2f1d1853ec42667120fc1d7caa50d4c3887351b8b9"
    ),
    "wmf1__pilot__P00__N3__reflected__left": (
        30038, "7a1a9ff1b3f31f4f56634a75c27ada25db5c7113f986bf64202e85febbc8794d"
    ),
    "wmf1__pilot__P00__N3__original__left": (
        30028, "805e6723040a4fb5a7e7574a8fe489211a61207379833db5b14924401282220e"
    ),
    "wmf1__development__D01__D1__original__left": (
        70885, "aee559b64e10a8cceb53dc624f604a1d7f73954c0b669fb3190b14e7648645fc"
    ),
    "wmf1__development__D01__D1__reflected__right": (
        70935, "f4cb1f1a1b38dc99c5505e446a8596417718bc34108142cf88e8adda57eac754"
    ),
    "wmf1__development__D02__D1__reflected__right": (
        70892, "225fd27f2411d0e21ceb8ef8eee8d47891f586fd5190f015550270d071b4de78"
    ),
    "wmf1__development__D02__D1__original__left": (
        70873, "0b2dfa0e7d163c9b89539289291a2eb9c959bfa5f9724d64e5df8b3377993d4c"
    ),
}
EXPECTED_GEOMETRY_OCCURRENCES = {
    PINNED_BLOBS["N3_snapshot_002"][0]: 4,
    PINNED_BLOBS["N3_snapshot_003"][0]: 12,
    PINNED_BLOBS["D01_snapshot"][0]: 8,
    PINNED_BLOBS["D02_snapshot"][0]: 8,
}
EXPECTED_CELL_IDENTITIES = {
    N3_KEYS[0]: ("wmf1__pilot__P00__N3__reflected__right", 0, "reflected", "right"),
    N3_KEYS[1]: ("wmf1__pilot__P00__N3__reflected__left", 1, "reflected", "left"),
    N3_KEYS[2]: ("wmf1__pilot__P00__N3__original__left", 2, "original", "left"),
    D1_KEYS["D01"][0]: (
        "wmf1__development__D01__D1__original__left", 0, "original", "left"
    ),
    D1_KEYS["D01"][1]: (
        "wmf1__development__D01__D1__reflected__right", 1, "reflected", "right"
    ),
    D1_KEYS["D02"][0]: (
        "wmf1__development__D02__D1__reflected__right", 0, "reflected", "right"
    ),
    D1_KEYS["D02"][1]: (
        "wmf1__development__D02__D1__original__left", 1, "original", "left"
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def strict_json_loads(payload: bytes, source: str) -> object:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
        value: dict[str, object] = {}
        for key, child in pairs:
            require(key not in value, f"Duplicate JSON key in {source}: {key}")
            value[key] = child
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON value in {source}: {value}")

    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )


def git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(
        [
            "git", "--no-replace-objects", "-C", str(REPOSITORY),
            "show", f"{commit}:{path}",
        ],
        timeout=30,
    )


def git_object_id(commit: str, path: str) -> str:
    return subprocess.check_output(
        [
            "git", "--no-replace-objects", "-C", str(REPOSITORY),
            "rev-parse", f"{commit}:{path}",
        ],
        text=True,
        timeout=30,
    ).strip()


def read_blob(path: str, size: int, digest: str) -> tuple[dict, dict]:
    payload = git_bytes(RESULTS_COMMIT, path)
    require(
        len(payload) == size and hashlib.sha256(payload).hexdigest() == digest,
        f"Pinned published blob differs: {path}",
    )
    decoded = strict_json_loads(payload, path)
    require(isinstance(decoded, dict), f"Pinned JSON root is not an object: {path}")
    return decoded, {
        "git_path": path,
        "bytes": size,
        "sha256": digest,
    }


def key_occurrence_count(value: object, key: str) -> int:
    if isinstance(value, dict):
        return sum(child_key == key for child_key in value) + sum(
            key_occurrence_count(child, key) for child in value.values()
        )
    if isinstance(value, list):
        return sum(key_occurrence_count(child, key) for child in value)
    return 0


def exhaustive_results_json_inventory() -> dict:
    raw = subprocess.check_output(
        [
            "git", "--no-replace-objects", "-C", str(REPOSITORY),
            "ls-tree", "-r", "-z", "-l", RESULTS_COMMIT, "--", "results/jobs",
        ],
        timeout=30,
    )
    entries = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, path_bytes = record.split(b"\t", 1)
        mode, object_type, object_id, size_bytes = metadata.decode("ascii").split()
        path = path_bytes.decode("utf-8")
        if not path.endswith(".json"):
            continue
        require(mode == "100644" and object_type == "blob",
                f"Unexpected results/jobs JSON tree entry: {path}")
        entries.append((path, object_id, int(size_bytes)))
    require(len(entries) == 370,
            "Pinned results/jobs JSON blob count differs")
    require(sum(size for _, _, size in entries) == 1_925_346,
            "Pinned results/jobs JSON byte total differs")

    geometry_occurrences: dict[str, int] = {}
    for path, object_id, size in entries:
        payload = git_bytes(RESULTS_COMMIT, path)
        require(len(payload) == size, f"Git tree size differs: {path}")
        require(
            hashlib.sha1(b"blob " + str(size).encode("ascii") + b"\0" + payload).hexdigest()
            == object_id,
            f"Git blob identity differs while inventorying: {path}",
        )
        decoded = strict_json_loads(payload, path)
        occurrences = key_occurrence_count(decoded, "object_centers_robot_base_m")
        if occurrences:
            geometry_occurrences[path] = occurrences
    require(geometry_occurrences == EXPECTED_GEOMETRY_OCCURRENCES,
            "Results/jobs retained-center occurrence inventory differs")
    return {
        "commit": RESULTS_COMMIT,
        "scope": "results/jobs/**/*.json",
        "strict_json_blobs_parsed": len(entries),
        "strict_json_bytes_parsed": sum(size for _, _, size in entries),
        "object_centers_robot_base_m_occurrences": sum(
            geometry_occurrences.values()
        ),
        "occurrences_by_path": geometry_occurrences,
        "all_occurrences_confined_to_pinned_snapshots": True,
    }


def verify_prior_package_overlap() -> dict:
    packages: dict[str, dict] = {}
    for label, (path, size, digest) in PRIOR_PACKAGE_PINS.items():
        payload = git_bytes(PRIOR_PACKAGE_COMMIT, path)
        require(
            len(payload) == size and hashlib.sha256(payload).hexdigest() == digest,
            f"Pinned prior retained package differs: {label}",
        )
        decoded = strict_json_loads(payload, path)
        require(isinstance(decoded, dict),
                f"Pinned prior retained package root differs: {label}")
        packages[label] = decoded
    n3 = packages["retained_N3_operational_subset"]
    success = packages["retained_success_channel_diagnostic"]
    require(
        n3["retained_capability_scan"]["object_center_occurrences"] == 32
        and n3["retained_capability_scan"]["unique_complete_embedded_cells"] == 7,
        "Prior retained N3 operational scope differs",
    )
    require(
        success["selection"]["included_cells"] == 7,
        "Prior retained success-channel scope differs",
    )
    return {
        "source_commit": PRIOR_PACKAGE_COMMIT,
        "pinned_packages": {
            label: {"git_path": pin[0], "bytes": pin[1], "sha256": pin[2]}
            for label, pin in PRIOR_PACKAGE_PINS.items()
        },
        "already_reported": (
            "The retained N3 operational package already reports the 32 center-field "
            "occurrences and seven unique embedded completed cells; the retained "
            "success-channel package already authenticates the same seven selected "
            "completed cells."
        ),
        "incremental_scope_here": (
            "This diagnostic adds exact numeric three-object center mappings, "
            "camera-copy and repeated-receipt collapse, authenticated reflection "
            "residual vectors, and descriptive per-object L2 residuals only."
        ),
    }


def verify_source_semantics() -> dict:
    observations = []
    source_text: str | None = None
    for model_id, commit in SOURCE_COMMITS.items():
        payload = git_bytes(commit, SOURCE_FILE["path"])
        object_id = git_object_id(commit, SOURCE_FILE["path"])
        require(object_id == SOURCE_FILE["git_blob"],
                f"Pinned fixture adapter Git blob differs: {model_id}")
        require(
            len(payload) == SOURCE_FILE["bytes"]
            and hashlib.sha256(payload).hexdigest() == SOURCE_FILE["sha256"],
            f"Pinned fixture adapter bytes differ: {model_id}",
        )
        text = payload.decode("utf-8")
        require(source_text is None or source_text == text,
                "Fixture adapter differs between recorded model source commits")
        source_text = text
        observations.append({
            "model_id": model_id,
            "source_commit": commit,
            "git_blob": object_id,
        })
    assert source_text is not None
    for fragment in (
        'def _camera_rows(self, env: Any, observation: Mapping[str, Any], world: Any, attempt_dir: Path)',
        'centers = {name: _host(world.get_pose(name, env_id=0)[0]) for name in object_names}',
        'camera_names = ("head_camera", "over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam")',
        'for camera_name in camera_names:',
        '"object_centers_robot_base_m": centers,',
    ):
        require(fragment in source_text,
                f"Pinned fixture-center source fragment absent: {fragment}")

    spec_payload = git_bytes(SPEC_SOURCE["commit"], SPEC_SOURCE["path"])
    require(
        git_object_id(SPEC_SOURCE["commit"], SPEC_SOURCE["path"])
        == SPEC_SOURCE["git_blob"]
        and len(spec_payload) == SPEC_SOURCE["bytes"]
        and hashlib.sha256(spec_payload).hexdigest() == SPEC_SOURCE["sha256"],
        "Pinned ablation specification differs",
    )
    reflection_statement = (
        "Reflection changes cube, bowl and banana center positions (x,y,z) "
        "to (x,-y,z) in robot coordinates."
    )
    require(reflection_statement in spec_payload.decode("utf-8"),
            "Pinned reflection definition differs")

    reset_source_pins = {}
    for label, pin in RESET_SOURCE_FILES.items():
        payload = git_bytes(pin["commit"], pin["path"])
        require(
            git_object_id(pin["commit"], pin["path"]) == pin["git_blob"]
            and len(payload) == pin["bytes"]
            and hashlib.sha256(payload).hexdigest() == pin["sha256"],
            f"Pinned reset-stage job source differs: {label}",
        )
        job_text = payload.decode("utf-8")
        for fragment in (
            "from fixture_tasks import settle_for_recording_reset, success_measurements",
            "collision_sampler, visibility_sampler, physical_evidence = recorder_job._fresh_physical_callbacks(",
            "settled_observation, settled_info, attestation = settle_for_recording_reset(",
            '"fresh_physical_checks": physical_evidence,',
        ):
            require(fragment in job_text,
                    f"Pinned reset-stage job fragment absent: {label}/{fragment}")
        reset_source_pins[label] = {
            "source_commit": pin["commit"],
            "git_path": pin["path"],
            "git_blob": pin["git_blob"],
            "bytes": pin["bytes"],
            "sha256": pin["sha256"],
        }

    recorder_observations = []
    recorder_text: str | None = None
    for model_id, commit in SOURCE_COMMITS.items():
        payload = git_bytes(commit, RECORDER_SOURCE_FILE["path"])
        require(
            git_object_id(commit, RECORDER_SOURCE_FILE["path"])
            == RECORDER_SOURCE_FILE["git_blob"]
            and len(payload) == RECORDER_SOURCE_FILE["bytes"]
            and hashlib.sha256(payload).hexdigest()
            == RECORDER_SOURCE_FILE["sha256"],
            f"Pinned reset callback source differs: {model_id}",
        )
        decoded = payload.decode("utf-8")
        require(recorder_text is None or recorder_text == decoded,
                "Reset callback source differs between model source commits")
        recorder_text = decoded
        recorder_observations.append({
            "model_id": model_id,
            "source_commit": commit,
            "git_blob": RECORDER_SOURCE_FILE["git_blob"],
        })
    assert recorder_text is not None
    for fragment in (
        "def _fresh_physical_callbacks(evidence_root: Path, source_contract: Mapping[str, Any]):",
        'evidence["visibility"] = result',
        "return collision_sampler, visibility_sampler, evidence",
    ):
        require(fragment in recorder_text,
                f"Pinned reset callback fragment absent: {fragment}")

    fixture_observations = []
    fixture_text: str | None = None
    for model_id, commit in SOURCE_COMMITS.items():
        payload = git_bytes(commit, FIXTURE_SOURCE_FILE["path"])
        require(
            git_object_id(commit, FIXTURE_SOURCE_FILE["path"])
            == FIXTURE_SOURCE_FILE["git_blob"]
            and len(payload) == FIXTURE_SOURCE_FILE["bytes"]
            and hashlib.sha256(payload).hexdigest()
            == FIXTURE_SOURCE_FILE["sha256"],
            f"Pinned settle helper source differs: {model_id}",
        )
        decoded = payload.decode("utf-8")
        require(fixture_text is None or fixture_text == decoded,
                "Settle helper source differs between model source commits")
        fixture_text = decoded
        fixture_observations.append({
            "model_id": model_id,
            "source_commit": commit,
            "git_blob": FIXTURE_SOURCE_FILE["git_blob"],
        })
    assert fixture_text is not None
    for fragment in (
        "def settle_for_recording_reset(",
        "for _ in range(settle_steps):",
        "for _ in range(stability_window_steps):",
        'require(stability_passed, "post-reset fixture did not satisfy frozen stability tolerances")',
        "collision_evidence = collision_sampler(env, settled_observation)",
        "visibility_evidence = visibility_sampler(env, settled_observation)",
        '"settled": True,',
        '"settled_observation_returned": True,',
        "return settled_observation, settled_info, receipt",
    ):
        require(fragment in fixture_text,
                f"Pinned settle helper fragment absent: {fragment}")
    return {
        "source_commits": SOURCE_COMMITS,
        "fixture_adapter": {
            "git_path": SOURCE_FILE["path"],
            "bytes": SOURCE_FILE["bytes"],
            "sha256": SOURCE_FILE["sha256"],
            "identical_at_source_commits": observations,
        },
        "center_capture_semantics": (
            "The pinned adapter reads each object's robot-base world pose once into "
            "one centers mapping before the camera loop, then copies that same mapping "
            "into each of the four camera geometry records. Camera copies are not "
            "independent object-position measurements."
        ),
        "ablation_specification": {
            "source_commit": SPEC_SOURCE["commit"],
            "git_path": SPEC_SOURCE["path"],
            "git_blob": SPEC_SOURCE["git_blob"],
            "bytes": SPEC_SOURCE["bytes"],
            "sha256": SPEC_SOURCE["sha256"],
            "authenticated_reflection_definition": reflection_statement,
        },
        "reset_stage_provenance": {
            "behavioral_job_sources": reset_source_pins,
            "shared_callback_source": {
                "git_path": RECORDER_SOURCE_FILE["path"],
                "git_blob": RECORDER_SOURCE_FILE["git_blob"],
                "bytes": RECORDER_SOURCE_FILE["bytes"],
                "sha256": RECORDER_SOURCE_FILE["sha256"],
                "identical_at_source_commits": recorder_observations,
            },
            "shared_settle_helper_source": {
                "git_path": FIXTURE_SOURCE_FILE["path"],
                "git_blob": FIXTURE_SOURCE_FILE["git_blob"],
                "bytes": FIXTURE_SOURCE_FILE["bytes"],
                "sha256": FIXTURE_SOURCE_FILE["sha256"],
                "identical_at_source_commits": fixture_observations,
            },
            "interpretation": (
                "The pinned helper runs settle and stability loops before invoking "
                "the collision and visibility callbacks, returns settled=True, and "
                "the pinned behavioral jobs store the resulting physical_evidence "
                "mapping as fresh_physical_checks in each completed cell receipt."
            ),
        },
    }


def match_terminal_receipt(
    cell: dict,
    terminal: dict,
    expected_suffix: str,
) -> dict:
    encoded = canonical_bytes(cell)
    digest = hashlib.sha256(encoded).hexdigest()
    matches = [
        descriptor for descriptor in terminal["cell_receipts"]
        if descriptor.get("bytes") == len(encoded)
        and descriptor.get("sha256") == digest
    ]
    require(len(matches) == 1,
            f"Embedded receipt lacks unique terminal match: {expected_suffix}")
    require(matches[0]["path"].endswith(expected_suffix),
            f"Terminal receipt path differs: {expected_suffix}")
    return matches[0]


def validate_centers(
    cell: dict, label: str
) -> tuple[dict[str, list[float]], list[dict]]:
    visibility = cell["fresh_physical_checks"]["visibility"]
    require(visibility["passed"] is True,
            f"Retained visibility check is not passed: {label}")
    cameras = visibility["cameras"]
    require(set(cameras) == set(CAMERA_NAMES),
            f"Retained camera inventory differs: {label}")
    center_copies = []
    geometry_hashes = []
    for camera_name in CAMERA_NAMES:
        camera = cameras[camera_name]
        geometry = camera["geometry"]
        geometry_digest = hashlib.sha256(json.dumps(
            geometry,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        require(
            geometry_digest == camera["camera_geometry_source_sha256"]
            == camera["visibility_source_sha256"],
            f"Retained camera geometry digest differs: {label}/{camera_name}",
        )
        geometry_hashes.append({
            "camera_name": camera_name,
            "geometry_sha256": geometry_digest,
            "matches_camera_geometry_source_sha256": True,
            "matches_visibility_source_sha256": True,
        })
        centers = geometry["object_centers_robot_base_m"]
        require(set(centers) == set(OBJECT_NAMES),
                f"Retained object-center inventory differs: {label}/{camera_name}")
        normalized: dict[str, list[float]] = {}
        for object_name in OBJECT_NAMES:
            position = centers[object_name]
            require(
                type(position) is list and len(position) == 3
                and all(type(value) in (int, float) and math.isfinite(value)
                        for value in position),
                f"Retained object center is invalid: {label}/{camera_name}/{object_name}",
            )
            normalized[object_name] = [float(value) for value in position]
        center_copies.append(normalized)
    require(all(value == center_copies[0] for value in center_copies[1:]),
            f"Camera object-center copies differ: {label}")
    return center_copies[0], geometry_hashes


def validate_cell(
    *,
    cell: dict,
    wrapper: dict,
    snapshot_source: str,
    snapshot_at_utc: str,
    snapshot_cell_key: str,
    terminal_source: str,
    terminal: dict,
    attempt_id: str,
    phase: str,
    model_id: str,
    layout_pair_id: str,
) -> dict:
    expected_cell_id, expected_index, expected_arm, expected_command = (
        EXPECTED_CELL_IDENTITIES[snapshot_cell_key]
    )
    require(
        cell["status"] == "passed"
        and cell["cell_id"] == expected_cell_id
        and cell["condition_index"] == expected_index
        and cell["model_config"] == model_id
        and cell["layout_pair_id"] == layout_pair_id
        and cell["layout_arm"] == expected_arm
        and cell["command"] == expected_command
        and cell["actions_executed"] == 450
        and cell["observation_count"] == 451,
        f"Embedded completed-cell identity differs: {snapshot_source}/{snapshot_cell_key}",
    )
    expected_requests = 15 if model_id == "N3" else 57
    require(
        cell["behavioral_episode_count"] == 1
        and cell["behavioral_model_request_count"] == expected_requests
        and cell["generation_qualification_request_count"] == 0,
        f"Embedded completed-cell request accounting differs: {snapshot_cell_key}",
    )
    if model_id == "N3":
        require(
            cell["schema_version"] == "wmf-n3-behavioral-pilot-cell-v1"
            and "phase" not in cell
            and wrapper["last_event_kind"] == "attempt_finalized"
            and wrapper["actions_observed"] == 450
            and wrapper["event_counts"]["attempt_finalized"] == 1
            and wrapper["event_counts"]["environment_step_started"] == 450
            and wrapper["event_counts"]["environment_step_completed"] == 450
            and wrapper["event_counts"]["model_request_completed"] == 15,
            f"N3 embedded receipt schema differs: {snapshot_cell_key}",
        )
    else:
        journal = wrapper["journal"]
        require(
            cell["schema_version"] == "wmf-d1-behavioral-development-cell-v1"
            and cell["phase"] == phase
            and cell["source_pins"]["study_commit"] == SOURCE_COMMITS["D1"],
            f"D1 embedded receipt schema/source differs: {snapshot_cell_key}",
        )
        require(
            journal["exists"] is True
            and journal["last_kind"] == "attempt_finalized"
            and journal["event_counts"]["attempt_finalized"] == 1
            and journal["event_counts"]["environment_step_started"] == 450
            and journal["event_counts"]["environment_step_completed"] == 450
            and journal["event_counts"]["model_request_completed"] == 57
            and journal["max_action_step"] == 450
            and journal["max_request_index"] == 56,
            f"D1 completed wrapper journal differs: {snapshot_cell_key}",
        )
    raw_receipt = match_terminal_receipt(
        cell,
        terminal,
        f"/{attempt_id}/cells/{snapshot_cell_key}/cell_receipt.json",
    )
    encoded = canonical_bytes(cell)
    expected_bytes, expected_digest = EXPECTED_RECEIPTS[cell["cell_id"]]
    require(
        len(encoded) == expected_bytes
        and hashlib.sha256(encoded).hexdigest() == expected_digest,
        f"Pinned canonical completed receipt differs: {snapshot_cell_key}",
    )
    centers, geometry_hashes = validate_centers(
        cell, f"{snapshot_source}/{snapshot_cell_key}"
    )
    return {
        "cell_id": cell["cell_id"],
        "phase": phase,
        "model_id": model_id,
        "layout_pair_id": layout_pair_id,
        "layout_arm": cell["layout_arm"],
        "command": cell["command"],
        "condition_index": cell["condition_index"],
        "candidate_id": cell["candidate_id"],
        "accepted_gate_record_sha256": cell["accepted_gate_record_sha256"],
        "attempt_id": attempt_id,
        "snapshot_source": snapshot_source,
        "snapshot_at_utc": snapshot_at_utc,
        "snapshot_cell_key": snapshot_cell_key,
        "terminal_aggregate_source": terminal_source,
        "canonical_embedded_receipt_bytes": len(encoded),
        "canonical_embedded_receipt_sha256": hashlib.sha256(encoded).hexdigest(),
        "embedded_cell_bytes_match_terminal_descriptor": True,
        "raw_cell_receipt": raw_receipt,
        "camera_names_with_identical_center_copy": list(CAMERA_NAMES),
        "verified_camera_geometry_hashes": geometry_hashes,
        "object_centers_robot_base_m": centers,
    }


def mirror_comparison(
    *,
    model_id: str,
    layout_pair_id: str,
    original: dict,
    reflected_rows: list[dict],
) -> dict:
    require(original["layout_arm"] == "original",
            f"Original comparison row differs: {model_id}/{layout_pair_id}")
    require(reflected_rows and all(row["layout_arm"] == "reflected"
                                   for row in reflected_rows),
            f"Reflected comparison rows differ: {model_id}/{layout_pair_id}")
    reflected = reflected_rows[0]["object_centers_robot_base_m"]
    require(all(row["object_centers_robot_base_m"] == reflected
                for row in reflected_rows[1:]),
            f"Retained same-arm center mappings differ: {model_id}/{layout_pair_id}")
    objects = {}
    for object_name in OBJECT_NAMES:
        original_position = original["object_centers_robot_base_m"][object_name]
        reflected_position = reflected[object_name]
        mirrored_original = [
            original_position[0], -original_position[1], original_position[2]
        ]
        residual = [
            reflected_position[index] - mirrored_original[index]
            for index in range(3)
        ]
        l2_m = math.sqrt(sum(value * value for value in residual))
        objects[object_name] = {
            "original_center_robot_base_m": original_position,
            "mirrored_original_center_robot_base_m": mirrored_original,
            "reflected_center_robot_base_m": reflected_position,
            "mirror_residual_robot_base_m": residual,
            "mirror_residual_l2_m": l2_m,
            "mirror_residual_l2_mm": l2_m * 1000.0,
        }
    expected_l2 = EXPECTED_OBJECT_L2_M[(model_id, layout_pair_id)]
    for object_name in OBJECT_NAMES:
        require(
            math.isclose(
                objects[object_name]["mirror_residual_l2_m"],
                expected_l2[object_name],
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            f"Retained mirror residual differs: "
            f"{model_id}/{layout_pair_id}/{object_name}",
        )
    maximum_object = max(
        OBJECT_NAMES, key=lambda name: objects[name]["mirror_residual_l2_m"]
    )
    require(
        maximum_object == max(OBJECT_NAMES, key=expected_l2.__getitem__),
        f"Retained maximum residual object differs: {model_id}/{layout_pair_id}",
    )
    return {
        "model_id": model_id,
        "layout_pair_id": layout_pair_id,
        "original_cell_id": original["cell_id"],
        "reflected_cell_ids_with_identical_centers": [
            row["cell_id"] for row in reflected_rows
        ],
        "object_residuals": objects,
        "maximum_object_residual": {
            "object": maximum_object,
            "l2_m": objects[maximum_object]["mirror_residual_l2_m"],
            "l2_mm": objects[maximum_object]["mirror_residual_l2_mm"],
        },
        "threshold_applied": False,
        "pass_fail_classification": None,
    }


def audit() -> dict:
    exhaustive_inventory = exhaustive_results_json_inventory()
    values: dict[str, dict] = {}
    sources: dict[str, dict] = {}
    for name, pin in PINNED_BLOBS.items():
        values[name], sources[name] = read_blob(*pin)

    n3_terminal = values["N3_terminal"]
    require(
        n3_terminal["source_commit"] == SOURCE_COMMITS["N3"]
        and n3_terminal["schema_version"] == "wmf-n3-behavioral-pilot-job-v1"
        and n3_terminal["status"] == "passed"
        and n3_terminal["phase"] == "pilot"
        and n3_terminal["model_config"] == "N3"
        and n3_terminal["layout_pair_id"] == "P00"
        and n3_terminal["queue_descriptor"]["job_id"] == N3_ATTEMPT_ID,
        "N3 terminal aggregate identity differs",
    )
    n3_snapshot_002 = values["N3_snapshot_002"]
    n3_snapshot_003 = values["N3_snapshot_003"]
    n3_002_cells = n3_snapshot_002["n3_attempts"][N3_ATTEMPT_ID]["cells"]
    n3_003_cells = n3_snapshot_003["n3_attempts"][N3_ATTEMPT_ID]["cells"]
    require(
        {key for key, wrapper in n3_002_cells.items()
         if isinstance(wrapper.get("cell_receipt.json"), dict)} == {N3_KEYS[0]},
        "N3 snapshot 002 embedded-receipt selection differs",
    )
    require(
        {key for key, wrapper in n3_003_cells.items()
         if isinstance(wrapper.get("cell_receipt.json"), dict)} == set(N3_KEYS),
        "N3 snapshot 003 embedded-receipt selection differs",
    )
    appearances = []
    appearances.append(validate_cell(
        cell=n3_002_cells[N3_KEYS[0]]["cell_receipt.json"],
        wrapper=n3_002_cells[N3_KEYS[0]],
        snapshot_source="N3_snapshot_002",
        snapshot_at_utc=n3_snapshot_002["at_utc"],
        snapshot_cell_key=N3_KEYS[0],
        terminal_source="N3_terminal",
        terminal=n3_terminal,
        attempt_id=N3_ATTEMPT_ID,
        phase="pilot",
        model_id="N3",
        layout_pair_id="P00",
    ))
    for key in N3_KEYS:
        appearances.append(validate_cell(
            cell=n3_003_cells[key]["cell_receipt.json"],
            wrapper=n3_003_cells[key],
            snapshot_source="N3_snapshot_003",
            snapshot_at_utc=n3_snapshot_003["at_utc"],
            snapshot_cell_key=key,
            terminal_source="N3_terminal",
            terminal=n3_terminal,
            attempt_id=N3_ATTEMPT_ID,
            phase="pilot",
            model_id="N3",
            layout_pair_id="P00",
        ))

    for pair in ("D01", "D02"):
        snapshot = values[f"{pair}_snapshot"]
        terminal = values[f"{pair}_terminal"]
        attempt_id = f"d1-development-{pair.lower()}-simulator-003"
        require(
            snapshot["schema_version"] == "wmf-d1-development-readonly-snapshot-v1"
            and snapshot["layout_pair_id"] == pair
            and snapshot["science_claim_boundary"] == SNAPSHOT_BOUNDARY,
            f"D1 snapshot identity differs: {pair}",
        )
        require(
            terminal["study_commit"] == SOURCE_COMMITS["D1"]
            and terminal["schema_version"]
            == "wmf-d1-behavioral-development-simulator-job-v1"
            and terminal["status"] == "passed"
            and terminal["phase"] == "development"
            and terminal["model_config"] == "D1"
            and terminal["layout_pair_id"] == pair
            and terminal["run_id"] == snapshot["run_id"]
            and terminal["queue_descriptor"]["job_id"] == attempt_id,
            f"D1 terminal aggregate identity differs: {pair}",
        )
        cells = snapshot["simulator"]["cells"]
        require(
            {key for key, wrapper in cells.items()
             if isinstance(wrapper.get("cell_receipt"), dict)} == set(D1_KEYS[pair]),
            f"D1 embedded-receipt selection differs: {pair}",
        )
        for key in D1_KEYS[pair]:
            appearances.append(validate_cell(
                cell=cells[key]["cell_receipt"],
                wrapper=cells[key],
                snapshot_source=f"{pair}_snapshot",
                snapshot_at_utc=snapshot["at_utc"],
                snapshot_cell_key=key,
                terminal_source=f"{pair}_terminal",
                terminal=terminal,
                attempt_id=attempt_id,
                phase="development",
                model_id="D1",
                layout_pair_id=pair,
            ))

    n3_partial_002 = n3_002_cells[N3_KEYS[1]]
    n3_partial_003 = n3_003_cells[
        "03-wmf1-pilot-P00-N3-original-right"
    ]
    d01_partial = values["D01_snapshot"]["simulator"]["cells"][
        "02-wmf1-development-D01-D1-original-right"
    ]
    d02_partial = values["D02_snapshot"]["simulator"]["cells"][
        "02-wmf1-development-D02-D1-reflected-left"
    ]
    require(
        n3_partial_002["cell_receipt.json"] is None
        and n3_partial_002["actions_observed"] == 135
        and n3_partial_002["event_counts"]["environment_step_completed"] == 135
        and "attempt_finalized" not in n3_partial_002["event_counts"],
        "N3 snapshot 002 partial-cell boundary differs",
    )
    require(
        n3_partial_003["cell_receipt.json"] is None
        and n3_partial_003["actions_observed"] == 236
        and n3_partial_003["event_counts"]["environment_step_started"] == 236
        and n3_partial_003["event_counts"]["environment_step_completed"] == 235
        and "attempt_finalized" not in n3_partial_003["event_counts"],
        "N3 snapshot 003 partial-cell boundary differs",
    )
    require(
        d01_partial["cell_receipt"] is None
        and d01_partial["journal"]["last_kind"] == "model_request_sent"
        and d01_partial["journal"]["event_counts"]["environment_step_completed"]
        == 128
        and "attempt_finalized" not in d01_partial["journal"]["event_counts"],
        "D01 partial-cell boundary differs",
    )
    require(
        d02_partial["cell_receipt"] is None
        and d02_partial["journal"]["last_kind"] == "model_request_sent"
        and d02_partial["journal"]["event_counts"]["environment_step_completed"]
        == 208
        and "attempt_finalized" not in d02_partial["journal"]["event_counts"],
        "D02 partial-cell boundary differs",
    )
    explicit_exclusions = [
        {
            "source": "N3_snapshot_002",
            "cell_key": N3_KEYS[1],
            "completed_steps_observed": 135,
            "reason": "no embedded finalized receipt in this snapshot",
            "later_complete_receipt_cross_time_joined": False,
        },
        {
            "source": "N3_snapshot_003",
            "cell_key": "03-wmf1-pilot-P00-N3-original-right",
            "started_steps_observed": 236,
            "completed_steps_observed": 235,
            "reason": "no embedded finalized receipt",
        },
        {
            "source": "D01_snapshot",
            "cell_key": "02-wmf1-development-D01-D1-original-right",
            "completed_steps_observed": 128,
            "reason": "no embedded finalized receipt",
        },
        {
            "source": "D02_snapshot",
            "cell_key": "02-wmf1-development-D02-D1-reflected-left",
            "completed_steps_observed": 208,
            "reason": "no embedded finalized receipt",
        },
    ]

    require(len(appearances) == 8,
            "Retained center-field receipt appearance count differs")
    by_cell: dict[str, list[dict]] = {}
    for row in appearances:
        by_cell.setdefault(row["cell_id"], []).append(row)
    require(len(by_cell) == 7,
            "Retained unique completed-cell count differs")
    require(
        sorted(len(rows) for rows in by_cell.values()) == [1, 1, 1, 1, 1, 1, 2],
        "Retained duplicate snapshot appearance pattern differs",
    )

    cells = []
    for cell_id in sorted(by_cell):
        rows = by_cell[cell_id]
        first = rows[0]
        for row in rows[1:]:
            for key in (
                "canonical_embedded_receipt_bytes",
                "canonical_embedded_receipt_sha256",
                "raw_cell_receipt",
                "verified_camera_geometry_hashes",
                "object_centers_robot_base_m",
            ):
                require(row[key] == first[key],
                        f"Duplicate snapshot receipt differs: {cell_id}/{key}")
        cells.append({
            "cell_id": cell_id,
            "phase": first["phase"],
            "model_id": first["model_id"],
            "layout_pair_id": first["layout_pair_id"],
            "layout_arm": first["layout_arm"],
            "command": first["command"],
                "condition_index": first["condition_index"],
            "candidate_id": first["candidate_id"],
            "accepted_gate_record_sha256": first[
                "accepted_gate_record_sha256"
            ],
            "attempt_id": first["attempt_id"],
            "snapshot_appearances": [
                {
                    "source": row["snapshot_source"],
                    "at_utc": row["snapshot_at_utc"],
                    "cell_key": row["snapshot_cell_key"],
                }
                for row in rows
            ],
            "terminal_aggregate_source": first["terminal_aggregate_source"],
            "canonical_embedded_receipt_bytes": first[
                "canonical_embedded_receipt_bytes"
            ],
            "canonical_embedded_receipt_sha256": first[
                "canonical_embedded_receipt_sha256"
            ],
            "embedded_cell_bytes_match_terminal_descriptor": True,
            "raw_cell_receipt": first["raw_cell_receipt"],
            "camera_names_with_identical_center_copy": first[
                "camera_names_with_identical_center_copy"
            ],
            "verified_camera_geometry_hashes": first[
                "verified_camera_geometry_hashes"
            ],
            "object_centers_robot_base_m": first["object_centers_robot_base_m"],
        })

    for model_id, pair in (("N3", "P00"), ("D1", "D01"), ("D1", "D02")):
        pair_rows = [
            row for row in cells
            if row["model_id"] == model_id and row["layout_pair_id"] == pair
        ]
        require(
            len({row["candidate_id"] for row in pair_rows}) == 1
            and len({row["accepted_gate_record_sha256"] for row in pair_rows}) == 1,
            f"Retained pair source identity differs: {model_id}/{pair}",
        )

    terminal_descriptors = [
        descriptor
        for terminal in (
            n3_terminal, values["D01_terminal"], values["D02_terminal"]
        )
        for descriptor in terminal["cell_receipts"]
    ]
    require(len(terminal_descriptors) == 12,
            "Pinned aggregate terminal-descriptor count differs")
    terminal_paths = {descriptor["path"] for descriptor in terminal_descriptors}
    selected_terminal_paths = {row["raw_cell_receipt"]["path"] for row in cells}
    require(len(terminal_paths) == 12 and len(selected_terminal_paths) == 7,
            "Pinned aggregate terminal-descriptor path uniqueness differs")
    unselected_terminal_paths = sorted(terminal_paths - selected_terminal_paths)
    expected_unselected_suffixes = (
        "/n3-p00-pilot-004/cells/03-wmf1-pilot-P00-N3-original-right/cell_receipt.json",
        "/d1-development-d01-simulator-003/cells/02-wmf1-development-D01-D1-original-right/cell_receipt.json",
        "/d1-development-d01-simulator-003/cells/03-wmf1-development-D01-D1-reflected-left/cell_receipt.json",
        "/d1-development-d02-simulator-003/cells/02-wmf1-development-D02-D1-reflected-left/cell_receipt.json",
        "/d1-development-d02-simulator-003/cells/03-wmf1-development-D02-D1-original-right/cell_receipt.json",
    )
    require(
        len(unselected_terminal_paths) == 5
        and all(
            any(path.endswith(suffix) for path in unselected_terminal_paths)
            for suffix in expected_unselected_suffixes
        ),
        "Pinned terminal-only exclusion inventory differs",
    )
    terminal_descriptor_scope = {
        "completed_descriptors_in_three_pinned_terminal_aggregates": 12,
        "descriptors_with_complete_receipts_embedded_in_pinned_snapshots": 7,
        "terminal_descriptors_without_embedded_snapshot_geometry": 5,
        "unselected_terminal_descriptor_paths": unselected_terminal_paths,
        "boundary": (
            "The five terminal-only descriptors do not supply embedded center "
            "mappings in the pinned compact snapshots and are not cross-time joined."
        ),
    }

    def select(model_id: str, pair: str, arm: str) -> list[dict]:
        return [
            row for row in cells
            if row["model_id"] == model_id
            and row["layout_pair_id"] == pair
            and row["layout_arm"] == arm
        ]

    comparisons = []
    for model_id, pair in (("N3", "P00"), ("D1", "D01"), ("D1", "D02")):
        originals = select(model_id, pair, "original")
        reflected = select(model_id, pair, "reflected")
        require(len(originals) == 1 and len(reflected) in (1, 2),
                f"Retained comparison coverage differs: {model_id}/{pair}")
        comparisons.append(mirror_comparison(
            model_id=model_id,
            layout_pair_id=pair,
            original=originals[0],
            reflected_rows=reflected,
        ))

    same_arm_pairs = []
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in cells:
        groups.setdefault(
            (row["model_id"], row["layout_pair_id"], row["layout_arm"]), []
        ).append(row)
    for (model_id, pair, arm), rows in sorted(groups.items()):
        if {row["command"] for row in rows} == {"left", "right"}:
            require(len(rows) == 2
                    and rows[0]["object_centers_robot_base_m"]
                    == rows[1]["object_centers_robot_base_m"],
                    f"Same-arm command center mappings differ: {model_id}/{pair}/{arm}")
            same_arm_pairs.append({
                "model_id": model_id,
                "layout_pair_id": pair,
                "layout_arm": arm,
                "cell_ids": sorted(row["cell_id"] for row in rows),
                "commands": sorted(row["command"] for row in rows),
                "centers_exactly_equal": True,
            })
    require(
        same_arm_pairs == [{
            "model_id": "N3",
            "layout_pair_id": "P00",
            "layout_arm": "reflected",
            "cell_ids": sorted([
                "wmf1__pilot__P00__N3__reflected__left",
                "wmf1__pilot__P00__N3__reflected__right",
            ]),
            "commands": ["left", "right"],
            "centers_exactly_equal": True,
        }],
        "Retained same-arm command-pair availability differs",
    )

    unique_geometry_keys = {
        canonical_bytes(row["object_centers_robot_base_m"])
        for row in cells
    }
    require(len(unique_geometry_keys) == 6,
            "Retained unique arm-level geometry count differs")
    require(
        sum(len(row["verified_camera_geometry_hashes"]) for row in appearances)
        == 32
        and sum(len(row["verified_camera_geometry_hashes"]) for row in cells) == 28,
        "Retained camera-geometry hash accounting differs",
    )

    return {
        "schema_version": "wmf-retained-reset-geometry-diagnostic-v1",
        "results_commit": RESULTS_COMMIT,
        "selection": {
            "availability_selected": True,
            "full_completed_cohort": False,
            "completed_receipt_appearances": 8,
            "unique_completed_cells": 7,
            "unique_arm_level_center_mappings": 6,
            "mirror_comparisons": 3,
            "rule": (
                "Availability-selected complete cell receipts embedded in pinned "
                "compact progress snapshots, each with an exact byte/hash match to a "
                "terminal aggregate descriptor."
            ),
            "representative_sample": False,
        },
        "verification_level": (
            "All 370 results/jobs JSON blobs are strict-parsed; seven exact compact "
            "input blobs yield eight embedded receipt appearances and seven unique "
            "terminal-matched cells; exact specification, adapter, behavioral-job, "
            "and reset-callback sources authenticate interpretation. No raw-PVC, "
            "raw-camera, or live-state revalidation is performed."
        ),
        "exhaustive_results_inventory": exhaustive_inventory,
        "prior_package_overlap": verify_prior_package_overlap(),
        "sources": sources,
        "source_semantics": verify_source_semantics(),
        "duplicate_collapse": {
            "camera_center_field_appearances": 32,
            "camera_geometry_hashes_verified_before_receipt_collapse": 32,
            "camera_geometry_hashes_verified_after_receipt_collapse": 28,
            "camera_copies_per_receipt_appearance": 4,
            "embedded_receipt_appearances": 8,
            "unique_completed_cells": 7,
            "unique_arm_level_center_mappings": 6,
            "repeated_snapshot_cell_id": (
                "wmf1__pilot__P00__N3__reflected__right"
            ),
            "repeated_snapshot_receipt_bytes": 30051,
            "repeated_snapshot_receipt_sha256": (
                "0a53e9b6df118d79a6361f2f1d1853ec42667120fc1d7caa50d4c3887351b8b9"
            ),
        },
        "explicit_partial_exclusions": explicit_exclusions,
        "terminal_descriptor_scope": terminal_descriptor_scope,
        "cross_time_field_join_performed": False,
        "cells": cells,
        "same_arm_command_center_map_pairs": same_arm_pairs,
        "mirror_comparisons": comparisons,
        "finding": (
            "Three availability-selected original/reflected comparisons retain "
            "descriptive settled robot-base center mirror residuals. The maximum "
            "object L2 residual is 0.8224548670985458 mm for N3/P00, "
            "0.7508789535900541 mm for D1/D01, and 1.8614546594330106 mm for "
            "D1/D02. No tolerance or pass/fail threshold is applied."
        ),
        "residual_definition": {
            "source_transform": "(x,y,z) to (x,-y,z) in robot coordinates",
            "signed_vector": (
                "reflected_center_robot_base_m minus "
                "[original_x, -original_y, original_z]"
            ),
            "input_base_unit": "m",
            "l2_base_unit": "m",
            "display_scale_to_mm": 1000.0,
        },
        "claim_boundaries": {
            "stage": (
                "These retained fresh_physical_checks center mappings are produced "
                "by the pinned settle helper after its settle and stability loops. "
                "This audit does not authenticate their ordering against the first "
                "behavioral action, and they are not executed-motion endpoints or "
                "forecast targets."
            ),
            "availability": (
                "Exactly three availability-selected original/reflected comparisons "
                "from seven retained completed cells are described. D01 and D02 "
                "compare original-LEFT with reflected-RIGHT, so layout and command "
                "are confounded in those descriptive pairs. No full-cohort "
                "representativeness, population statistic, uncertainty interval, or "
                "between-model comparison follows."
            ),
            "duplicate_collapse": (
                "Four camera records in one receipt copy one object-center mapping and "
                "are not independent measurements. One exact N3 receipt appears in two "
                "snapshots. The only retained same-arm LEFT/RIGHT center-map pair is "
                "N3/P00/reflected, so its exact center-map equality does not generalize."
            ),
            "science_and_release": (
                "No fixture-tolerance qualification, raw-camera/PVC validation, "
                "current-state inference, A1 executed-motion effect, A2 instruction "
                "effect, A3-A9 result, forecast/task-success/model-rate comparison, "
                "label, gate promotion, dispatch authority, or release follows."
            ),
        },
        "evidence_boundaries": {
            "camera_rows_independent_measurements": False,
            "raw_pvc_bytes_read": 0,
            "raw_camera_bytes_read": 0,
            "pose_acceptance_recomputed": False,
            "threshold_applied": False,
            "current_state_inferred": False,
            "executed_motion_analyzed": False,
            "instruction_effect_estimated": False,
            "forecast_or_persistence_metric": False,
            "task_success_or_model_rate": False,
            "a1_a9_result": False,
            "gate_advanced": False,
            "release_authority": False,
        },
        "new_behavioral_cells": 0,
        "new_model_requests": 0,
        "labels_created": 0,
        "forecast_accuracy_estimated": False,
        "a1_executed_motion_estimated": False,
        "a2_instruction_effect_estimated": False,
        "resource_budget_qualified": False,
        "confirmation_released": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", type=Path,
        help="Verify an existing canonical diagnostic without modifying it",
    )
    args = parser.parse_args()
    encoded = canonical_bytes(audit())
    if args.check:
        require(args.check.read_bytes() == encoded,
                "Saved retained reset-geometry diagnostic differs")
        print(
            "Verified 7 pinned results blobs, 8 embedded appearances, 7 unique "
            "terminal-matched cells, and 3 descriptive object-center comparisons; "
            "diagnostic only, no gate promotion."
        )
    else:
        print(encoded.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
