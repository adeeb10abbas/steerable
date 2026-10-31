#!/usr/bin/env python3
"""Authenticate and package the retained H03/H04 guidance pilots.

H03 and H04 are the completed V2-A015 post-result guidance pilots listed in
ABLATION_SPEC.md.  Each compares six preserved baseline episodes with six
intervention episodes over three matched environment seeds and two commands.
This adapter reads only seven immutable Git blobs, authenticates the disclosed
post-result amendment and compact historical release-gate record, independently
recomputes the cell joins, success transitions and requested-margin summaries,
and emits a small canonical report.  The raw fixed-observation probe bytes are
not present in this package and are not replayed by this adapter.

These pilots are exploratory, post-result evidence.  They are not new model
inference, do not establish a fixed-video intervention, do not qualify core D1,
and do not advance any current forecast-ablation or release gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any, Mapping


PIN = "ce561e66f82e95055e39d3d7711691982f6b2086"
PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
SPEC_PATH = PACKAGE / "docs" / "ABLATION_SPEC.md"
DEFAULT_JSON = PACKAGE / "paper" / "data" / "h03_h04_guidance_pilots.json"
DEFAULT_CSV = PACKAGE / "paper" / "data" / "h03_h04_guidance_pilots.csv"

BASE = "artifacts/vla_wam_shared_v2/pilot/expansion"
COMPARISON = f"{BASE}/cfg_ablation_v2a015_comparison.json"
PREFLIGHT = f"{BASE}/cfg_ablation_v2a015_preflight.json"
AMENDMENT = "artifacts/vla_wam_shared_v2/pilot/post_result_cfg_ablation_v2a015_amendment.json"
NANO_BASELINE = f"{BASE}/cosmos3_nano_policy_droid_direct_gate.json"
NANO_INTERVENTION = f"{BASE}/cosmos3_nano_v2a015_no_cfg_g1_result.json"
DREAMZERO_BASELINE = f"{BASE}/dreamzero_droid_direct_gate.json"
DREAMZERO_INTERVENTION = f"{BASE}/dreamzero_v2a015_action_cfg_s2_result.json"

FREEZE_HEAD = "22218d01c2301cc6ad28c9c1c53905045d8f2e9c"
FROZEN_STUDY_COMMIT = "a55fd133ff5be4d36938afa9a99cf394d33b819f"
RECORDED_HEADS = {
    "post_result_amendment": FREEZE_HEAD,
    "nano_baseline_compilation": "8a9893e643772c35ade334d5e6f5580f7a3e3a16",
    "dreamzero_baseline_compilation": "67016775bd327a43c71f23d44035ce44c06697c1",
    "v2a015_result_compilation": "2f177333545e39e35d586e577a9a4496e4e97f24",
}
SOURCE_BLOB_COMMITS = {
    NANO_BASELINE: "42da83084cb0e638064e4973019c6e0acc5f29e5",
    DREAMZERO_BASELINE: "e4c31fe686618590ca962d16fb606eda78446b7f",
    AMENDMENT: "5eae815eb3fb2433c22ffe9baca43d2f7bb0b410",
    PREFLIGHT: "2f177333545e39e35d586e577a9a4496e4e97f24",
    COMPARISON: "0bcaf371af582d4b7fe397b9e9a33e45c1394934",
    NANO_INTERVENTION: "0bcaf371af582d4b7fe397b9e9a33e45c1394934",
    DREAMZERO_INTERVENTION: "0bcaf371af582d4b7fe397b9e9a33e45c1394934",
}
PHYSICAL_RESET_FINGERPRINT = "8b000f4d2c7c2296bf332984c0b83b47561d796790c32419480665e684635fb6"
EXTERNAL_MODEL_SOURCE_COMMITS = {
    "cosmos3_nano": "411d25b2e35bc441126f48c44a4b93e1c0564274",
    "dreamzero": "ab790c198fbce33503358efbbd4187ce9a89adf3",
}

EXPECTED_SOURCES: dict[str, dict[str, Any]] = {
    AMENDMENT: {
        "git_blob": "113ffdd35b5a3443658cec7fd8df310cf1727ded",
        "sha256": "7c3b98d5e578a5744a463201060ccdacf7462471418b08dc64a2ac29a9a9fece",
        "bytes": 9_965,
        "lines": 169,
    },
    PREFLIGHT: {
        "git_blob": "ebd7d1d6ad8df6a28d7a1de13c8f31503fb1fc89",
        "sha256": "472c09568713fc3e08dbb86d3c39a8da6f328b82b06ff5aadef4e02f1892dd4d",
        "bytes": 18_843,
        "lines": 325,
    },
    COMPARISON: {
        "git_blob": "a4e36cf70b05ecf1f01e800491b0a31356aefc8c",
        "sha256": "0e6daadd9391b7ad38ee316371df0235e34268cc4c32597c6597578ac9f9ed1e",
        "bytes": 96_243,
        "lines": 2_402,
    },
    NANO_BASELINE: {
        "git_blob": "3b810746c3508d7bcc06dcac195ac0fdeaaa743e",
        "sha256": "4a6cc1d61593c7ba5272e1707f6bbe51261f7d23438070992bd75fd9e95fdb93",
        "bytes": 57_097,
        "lines": 1_404,
    },
    NANO_INTERVENTION: {
        "git_blob": "f26d7c53f2e90ca94ad3171337b8e62fe1701739",
        "sha256": "8796f4ab9ea9490ee5b78678bc689d6fd13f27a9551006f8b5d346e202d0cc5c",
        "bytes": 104_115,
        "lines": 2_388,
    },
    DREAMZERO_BASELINE: {
        "git_blob": "beaafdafa68f1f058f7c084457cedf74e5c53c91",
        "sha256": "4c76cdc3ca9eaf227d21d160199408f22e1b3dd7a71176a5a5dbe22223714461",
        "bytes": 254_288,
        "lines": 6_079,
    },
    DREAMZERO_INTERVENTION: {
        "git_blob": "41e9d2cab51066e0aa93b8805f5849a22cd30dbc",
        "sha256": "273b7191cde61b51bf90b2eda04b1910e214f48d34f2347f2a81325069c32444",
        "bytes": 270_309,
        "lines": 6_172,
    },
}

SEEDS = (8300, 8301, 8302)
RELATIONS = ("left", "right")
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
COHORTS: dict[str, dict[str, Any]] = {
    "H03": {
        "comparison_key": "cosmos3_nano",
        "model_family": "cosmos3_nano_policy_droid",
        "model_id": "cosmos3_nano_policy_droid",
        "baseline_model_id": "cosmos3_nano_policy_droid",
        "model_label": "Cosmos3 Nano Policy DROID",
        "baseline_path": NANO_BASELINE,
        "intervention_path": NANO_INTERVENTION,
        "baseline_configuration_id": "joint_guidance_g3",
        "intervention_configuration_id": "conditional_prediction_g1",
        "baseline_label": "g=3 baseline",
        "intervention_label": "g=1 intervention",
    },
    "H04": {
        "comparison_key": "dreamzero",
        "model_family": "dreamzero_droid",
        "model_id": "dreamzero_droid_action_cfg",
        "baseline_model_id": "dreamzero_droid",
        "model_label": "DreamZero DROID",
        "baseline_path": DREAMZERO_BASELINE,
        "intervention_path": DREAMZERO_INTERVENTION,
        "baseline_configuration_id": "conditional_action_equivalent_s1",
        "intervention_configuration_id": "negative_branch_action_guidance_s2",
        "baseline_label": "s=1 conditional-action equivalent",
        "intervention_label": "s=2 CFG-style negative-branch action guidance",
    },
}
EXPECTED_BEHAVIORAL_ACCOUNTING = {
    "H03": {
        "baseline_model_requests": 37,
        "baseline_actions_executed": 1_090,
        "intervention_model_requests": 64,
        "intervention_actions_executed": 1_892,
    },
    "H04": {
        "baseline_model_requests": 265,
        "baseline_actions_executed": 2_090,
        "intervention_model_requests": 245,
        "intervention_actions_executed": 1_931,
    },
}


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "--no-replace-objects", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _require_ancestor(root: Path, ancestor: str, descendant: str, label: str) -> None:
    result = subprocess.run(
        [
            "git", "--no-replace-objects", "-C", str(root),
            "merge-base", "--is-ancestor", ancestor, descendant,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError(f"historical producer ancestry check failed for {label}: {ancestor} -> {descendant}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _reject_nonfinite_tree(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number in {label}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite_tree(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite_tree(item, f"{label}[{index}]")


def parse_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    _reject_nonfinite_tree(value, label)
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _sha256_string(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _close(observed: Any, expected: float, label: str) -> None:
    if not math.isclose(_finite(observed, label), expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} disagrees with independent reproduction")


def read_pinned_sources(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(root)
    resolved = _git(root, "rev-parse", "--verify", PIN + "^{commit}").decode().strip()
    if resolved != PIN:
        raise ValueError("resolved historical evidence commit differs from fixed pin")

    resolved_freeze = _git(root, "rev-parse", "--verify", FROZEN_STUDY_COMMIT + "^{commit}").decode().strip()
    if resolved_freeze != FROZEN_STUDY_COMMIT:
        raise ValueError("resolved V2-A015 frozen study commit differs from fixed identity")
    freeze_parents = _git(root, "show", "-s", "--format=%P", FROZEN_STUDY_COMMIT).decode().strip().split()
    if freeze_parents != [FREEZE_HEAD]:
        raise ValueError("V2-A015 frozen study commit is not the direct child of its recorded pre-request head")
    _require_ancestor(root, FREEZE_HEAD, FROZEN_STUDY_COMMIT, "post-result amendment freeze")
    _require_ancestor(root, FROZEN_STUDY_COMMIT, PIN, "frozen V2-A015 study")
    for label, recorded_head in RECORDED_HEADS.items():
        resolved_head = _git(root, "rev-parse", "--verify", recorded_head + "^{commit}").decode().strip()
        if resolved_head != recorded_head:
            raise ValueError(f"resolved historical recorded head differs for {label}")
        _require_ancestor(root, recorded_head, PIN, label)

    documents: dict[str, Any] = {}
    observed_files: dict[str, dict[str, Any]] = {}
    for path, expected in EXPECTED_SOURCES.items():
        object_id = _git(root, "rev-parse", "--verify", f"{PIN}:{path}").decode().strip()
        if object_id != expected["git_blob"]:
            raise ValueError(f"pinned Git blob mismatch for {path}")
        raw = _git(root, "cat-file", "blob", object_id)
        observed = {
            "git_blob": object_id,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "lines": len(raw.splitlines()),
        }
        if observed != expected:
            raise ValueError(f"pinned source bytes disagree with fixed identity for {path}")
        blob_commit = SOURCE_BLOB_COMMITS[path]
        _require_ancestor(root, blob_commit, PIN, f"blob-bearing commit for {path}")
        produced_object_id = _git(
            root, "rev-parse", "--verify", f"{blob_commit}:{path}"
        ).decode().strip()
        if produced_object_id != object_id:
            raise ValueError(f"authenticated blob-bearing commit disagrees for {path}")
        documents[path] = parse_json(raw, label=path)
        observed_files[path] = observed
    documents["ablation_spec"] = SPEC_PATH.read_text(encoding="utf-8")
    return documents, {
        "commit": PIN,
        "recorded_heads": dict(RECORDED_HEADS),
        "all_recorded_heads_are_ancestors_of_pin": True,
        "source_blob_commits": dict(SOURCE_BLOB_COMMITS),
        "all_source_blob_commits_are_ancestors_of_pin": True,
        "post_result_freeze": {
            "recorded_at_git_head": FREEZE_HEAD,
            "frozen_study_commit": FROZEN_STUDY_COMMIT,
            "recorded_head_is_direct_parent_of_frozen_study_commit": True,
        },
        "files": observed_files,
    }


def _validate_scope(text: str) -> None:
    required = (
        "| H03 | Nano guidance pilot, V2-A015 with V2-A011 baseline | Six g3 baseline episodes: 6/6 successes. Six g1 episodes: 4/6. Three paired labels × two commands per configuration. | Already-completed exploratory guidance ablation |",
        "| H04 | DreamZero guidance pilot, V2-A015 | Six conditional-action-equivalent s1 episodes: 3/6 successes. Six custom s2 episodes: 4/6. | Already-completed exploratory guidance ablation |",
        "H03/H04 include previously collected baselines; do not count those baselines again when totaling another cohort.",
        "No new guidance sweep is required for a paper about the prediction quality of fully disclosed configurations.",
    )
    if any(sentence not in text for sentence in required):
        raise ValueError("ABLATION_SPEC H03/H04 scope binding changed")


def _validate_post_result_amendment(documents: Mapping[str, Any]) -> dict[str, Any]:
    amendment = documents[AMENDMENT]
    if (
        amendment.get("schema_version") != "vla-wam-shared-v2-post-result-cfg-ablation-v1"
        or amendment.get("study_id") != "vla_wam_language_steerability_v2"
        or amendment.get("amendment_id") != "V2-A015"
        or amendment.get("status")
        != "frozen_after_baseline_results_and_before_any_cfg_ablation_model_request_or_behavioral_inference"
        or amendment.get("recorded_at_git_head") != FREEZE_HEAD
    ):
        raise ValueError("V2-A015 post-result amendment freeze identity changed")

    disclosure = amendment.get("known_result_disclosure")
    if not isinstance(disclosure, dict) or disclosure.get("statement") != (
        "Both baseline six-cell results were known when this exploratory ablation was selected. "
        "V2-A015 is a disclosed post-result intervention, not preregistration and not a rewrite "
        "of any frozen result."
    ):
        raise ValueError("V2-A015 known-result disclosure changed")
    expected_baselines = {
        "cosmos3_nano_baseline": {
            "checkpoint": "nvidia/Cosmos3-Nano-Policy-DROID",
            "revision": "6706d7680581c255ff61e0f3bb49d90eac55c79e",
            "source_commit": "411d25b2e35bc441126f48c44a4b93e1c0564274",
            "guidance": 3.0,
            "result": "LEFT 3/3; RIGHT 3/3; 3/3 aligned endpoint pairs",
            "artifact": NANO_BASELINE,
            "sha256": EXPECTED_SOURCES[NANO_BASELINE]["sha256"],
        },
        "dreamzero_baseline": {
            "checkpoint": "GEAR-Dreams/DreamZero-DROID",
            "revision": "96ad344138c66e82536422432ad742f015784942",
            "source_commit": "ab790c198fbce33503358efbbd4187ce9a89adf3",
            "video_guidance": 5.0,
            "action_guidance_equivalent": 1.0,
            "result": "LEFT 2/3; RIGHT 1/3; 3/3 aligned endpoint pairs",
            "artifact": DREAMZERO_BASELINE,
            "sha256": EXPECTED_SOURCES[DREAMZERO_BASELINE]["sha256"],
        },
    }
    for name, expected in expected_baselines.items():
        if disclosure.get(name) != expected:
            raise ValueError(f"V2-A015 disclosed baseline identity changed: {name}")

    arms = amendment.get("arms")
    if not isinstance(arms, list) or len(arms) != 2:
        raise ValueError("V2-A015 amendment arm population changed")
    arm_by_id = {arm.get("arm_id"): arm for arm in arms if isinstance(arm, dict)}
    if set(arm_by_id) != {"cosmos3_nano_no_cfg_g1", "dreamzero_action_cfg_s2"}:
        raise ValueError("V2-A015 amendment arm identities changed")
    nano_arm = arm_by_id["cosmos3_nano_no_cfg_g1"]
    if (
        nano_arm.get("model_id") != "cosmos3_nano_policy_droid"
        or nano_arm.get("checkpoint_revision") != "6706d7680581c255ff61e0f3bb49d90eac55c79e"
        or nano_arm.get("source_commit") != "411d25b2e35bc441126f48c44a4b93e1c0564274"
        or nano_arm.get("guidance") != 1.0
        or nano_arm.get("baseline_guidance") != 3.0
        or nano_arm.get("behavioral_episode_count") != 6
    ):
        raise ValueError("V2-A015 H03 amendment arm identity changed")
    dream_arm = arm_by_id["dreamzero_action_cfg_s2"]
    if (
        dream_arm.get("model_id") != "dreamzero_droid_action_cfg"
        or dream_arm.get("checkpoint_revision") != "96ad344138c66e82536422432ad742f015784942"
        or dream_arm.get("source_commit") != "ab790c198fbce33503358efbbd4187ce9a89adf3"
        or dream_arm.get("action_guidance") != 2.0
        or dream_arm.get("baseline_action_guidance_equivalent") != 1.0
        or dream_arm.get("video_guidance") != 5.0
        or dream_arm.get("behavioral_episode_count") != 6
    ):
        raise ValueError("V2-A015 H04 amendment arm identity changed")

    overlay = amendment.get("dreamzero_source_overlay")
    if not isinstance(overlay, dict):
        raise ValueError("V2-A015 DreamZero overlay record missing")
    expected_overlay = {
        "target": "groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py",
        "target_before_sha256": "7193cd73423472aa252bee73bd80e0d673c89d773ec852e90f50154729b50845",
        "target_after_sha256": "65dc9873aef37563dedf3787fd7b59e0a6d50e575775e38b70edd9e38489f9b8",
        "patch_sha256": "de2b82a1c9f81ee4751fb384158775865f5e5e8fb622134cae3b2c4e8b2a2cc0",
        "derived_server_sha256": "5f9932acc77fe8fc80157622ef3e1db5b552bd31f322768330f2000f9af7d09d",
        "preserved_baseline_server_sha256": "2b0a83e21ee921527a2d0fc5cb9344d0b1f5950ff88906fd441fac2421c5a95f",
    }
    for key, expected in expected_overlay.items():
        if overlay.get(key) != expected:
            raise ValueError(f"V2-A015 DreamZero overlay identity changed: {key}")
    requirements = overlay.get("requirements")
    if not isinstance(requirements, list) or (
        "Scale 1.0 must take the untouched conditional-action branch for exact baseline equivalence."
        not in requirements
    ) or not any("Video CFG, prompt transforms, weights, noise seed" in item for item in requirements):
        raise ValueError("V2-A015 DreamZero overlay equivalence contract changed")

    release_gates = amendment.get("fixed_observation_release_gates")
    if not isinstance(release_gates, dict):
        raise ValueError("V2-A015 fixed-observation release gates missing")
    nano_gates = release_gates.get("cosmos3_nano_no_cfg_g1")
    dream_gates = release_gates.get("dreamzero_action_cfg_s2")
    if not isinstance(nano_gates, list) or not any("bit-identical" in item for item in nano_gates):
        raise ValueError("V2-A015 H03 fixed-input qualification contract changed")
    if (
        not isinstance(dream_gates, list)
        or "The scale-1 overlay returns bit-identical actions and latent video to the unmodified official path on the frozen fixture."
        not in dream_gates
        or "Scale-2 LEFT and RIGHT actions differ under identical observation and official noise seed 1140."
        not in dream_gates
    ):
        raise ValueError("V2-A015 H04 fixed-input qualification contract changed")

    grid = amendment.get("behavioral_grid")
    if not isinstance(grid, dict) or (
        grid.get("prompts") != PROMPTS
        or grid.get("environment_seeds") != list(SEEDS)
        or grid.get("sampling_seed_labels") != list(SEEDS)
        or grid.get("requested_relations") != list(RELATIONS)
        or grid.get("new_behavioral_episode_count") != 12
        or grid.get("prompt_controller") != "episode_static"
        or grid.get("oracle_actions") != 0
        or grid.get("subtask_coach") is not False
        or grid.get("prompt_switching") is not False
        or grid.get("progress_conditioned_language") is not False
    ):
        raise ValueError("V2-A015 frozen behavioral grid changed")
    analysis = amendment.get("analysis")
    inference_boundary = analysis.get("inference_boundary") if isinstance(analysis, dict) else None
    if inference_boundary != (
        "This n=6-per-arm pilot is descriptive and exploratory. Report exact paired discordances "
        "and effect sizes; do not claim a powered improvement from a nonsignificant result."
    ):
        raise ValueError("V2-A015 frozen inference boundary changed")
    followup = amendment.get("optional_followup_not_yet_authorized")
    if not isinstance(followup, dict) or followup.get("authorized_behavioral_episode_count") != 0:
        raise ValueError("V2-A015 optional follow-up authorization boundary changed")
    return {
        "status": "frozen_after_known_baselines_before_v2a015_requests",
        "post_result_disclosed": True,
        "preregistered": False,
        "baseline_results_known_at_selection": True,
        "optional_followup_behavioral_episodes_authorized": 0,
    }


def _validate_historical_qualification(documents: Mapping[str, Any]) -> dict[str, Any]:
    preflight = documents[PREFLIGHT]
    if (
        preflight.get("schema_version") != "vla-wam-shared-v2-cfg-ablation-preflight-v1"
        or preflight.get("amendment_id") != "V2-A015"
        or preflight.get("status") != "complete_twelve_valid_behavioral_cells_both_arms"
    ):
        raise ValueError("V2-A015 historical preflight identity changed")
    study_git = preflight.get("study_git")
    if not isinstance(study_git, dict) or study_git.get("frozen_commit") != FROZEN_STUDY_COMMIT:
        raise ValueError("V2-A015 historical preflight frozen commit changed")
    overlay = preflight.get("dreamzero_overlay")
    amendment_overlay = documents[AMENDMENT].get("dreamzero_source_overlay", {})
    for preflight_key, amendment_key in (
        ("target_before_sha256", "target_before_sha256"),
        ("target_after_sha256", "target_after_sha256"),
        ("source_patch_sha256", "patch_sha256"),
        ("derived_server_sha256", "derived_server_sha256"),
    ):
        if not isinstance(overlay, dict) or overlay.get(preflight_key) != amendment_overlay.get(amendment_key):
            raise ValueError(f"V2-A015 preflight/amendment overlay mismatch: {preflight_key}")
    if overlay.get("git_diff_check") != "passed" or overlay.get("original_worktree_modified") is not False:
        raise ValueError("V2-A015 overlay isolation qualification changed")

    nano_gate = preflight.get("cosmos3_nano_fixed_observation_gate")
    if not isinstance(nano_gate, dict) or (
        nano_gate.get("status") != "passed_behavioral_arm_released"
        or nano_gate.get("behavioral_cells_released") != 6
        or nano_gate.get("behavioral_episode_count") != 0
        or nano_gate.get("fixed_observation_action_request_count") != 3
        or nano_gate.get("fixed_observation_requests_in_behavioral_denominator") != 0
        or nano_gate.get("guidance") != 1.0
        or nano_gate.get("raw_result_sha256") != "96c365164e69e5a7b52e2c07a7e44874db247aec26446ab4f0064c3951f45385"
        or "bit-identical" not in nano_gate.get("result", "")
        or "distinct finite [32,8] actions" not in nano_gate.get("result", "")
    ):
        raise ValueError("V2-A015 H03 historical fixed-input qualification changed")

    dream_gates = preflight.get("dreamzero_fixed_observation_gates")
    if not isinstance(dream_gates, dict) or (
        dream_gates.get("status") != "passed_behavioral_arm_released"
        or dream_gates.get("behavioral_cells_released") != 6
        or dream_gates.get("behavioral_episode_count") != 0
        or dream_gates.get("fixed_observation_action_request_count") != 6
        or dream_gates.get("fixed_observation_requests_in_behavioral_denominator") != 0
    ):
        raise ValueError("V2-A015 H04 historical fixed-input qualification changed")
    s1 = dream_gates.get("action_s1_overlay_equivalence")
    if not isinstance(s1, dict) or (
        s1.get("status") != "passed"
        or s1.get("action_cfg_style_scale") != 1.0
        or s1.get("video_cfg_scale") != 5.0
        or s1.get("raw_result_sha256") != "4663db65362c188fe553acdeded62fb302eb2c7e634333e6ef3cee6e0c705c9a"
        or "bit-exact against the archived official V2-A007 fixed-observation probe" not in s1.get("result", "")
    ):
        raise ValueError("V2-A015 historical s1-overlay equivalence status changed")
    s2 = dream_gates.get("action_s2_release")
    if not isinstance(s2, dict) or (
        s2.get("status") != "passed"
        or s2.get("action_cfg_style_scale") != 2.0
        or s2.get("video_cfg_scale") != 5.0
        or s2.get("raw_result_sha256") != "f4892deb99e54cb815d97f00d7db8d548fa944943e6b4d7bd6fd29bd9ab32b76"
        or "bit-exact" not in s2.get("result", "")
        or "finite [24,8] tensors" not in s2.get("result", "")
    ):
        raise ValueError("V2-A015 historical s2 release qualification changed")

    release = preflight.get("behavioral_release_accounting")
    if release != {
        "authorized_behavioral_episode_count": 12,
        "behavioral_cells_released": 12,
        "behavioral_cells_completed": 12,
        "behavioral_cells_remaining": 0,
        "behavioral_episode_count": 12,
        "dreamzero_action_cfg_s2_remaining": 0,
        "cosmos3_nano_no_cfg_g1_remaining": 0,
    }:
        raise ValueError("V2-A015 historical behavioral release accounting changed")
    nano_behavior = preflight.get("cosmos3_nano_behavioral_arm")
    if not isinstance(nano_behavior, dict) or (
        nano_behavior.get("arm_id") != "cosmos3_nano_no_cfg_g1"
        or nano_behavior.get("status") != "complete_six_valid_behavioral_cells"
        or nano_behavior.get("valid_behavioral_episode_count") != 6
        or nano_behavior.get("requested_success_count") != 4
        or nano_behavior.get("behavioral_model_action_request_count") != 64
        or nano_behavior.get("executed_action_count") != 1_892
        or nano_behavior.get("runtime_intervention_count") != 0
        or nano_behavior.get("success_by_relation") != {"left": "1/3", "right": "3/3"}
        or nano_behavior.get("preserved_g3_baseline", {}).get("artifact_sha256")
        != EXPECTED_SOURCES[NANO_BASELINE]["sha256"]
    ):
        raise ValueError("V2-A015 H03 historical behavioral summary changed")
    dream_behavior = preflight.get("dreamzero_behavioral_arm")
    if not isinstance(dream_behavior, dict) or (
        dream_behavior.get("arm_id") != "dreamzero_action_cfg_s2"
        or dream_behavior.get("status") != "complete_six_valid_behavioral_cells"
        or dream_behavior.get("valid_behavioral_episode_count") != 6
        or dream_behavior.get("requested_success_count") != 4
        or dream_behavior.get("behavioral_model_action_request_count") != 245
        or dream_behavior.get("runtime_thermal_intervention_count") != 0
        or dream_behavior.get("success_by_relation") != {"left": "1/3", "right": "3/3"}
        or dream_behavior.get("preserved_scale1_equivalent_baseline", {}).get("artifact_sha256")
        != EXPECTED_SOURCES[DREAMZERO_BASELINE]["sha256"]
    ):
        raise ValueError("V2-A015 H04 historical behavioral summary changed")
    accounting = preflight.get("inference_accounting")
    if not isinstance(accounting, dict) or (
        accounting.get("model_action_request_count") != 318
        or accounting.get("fixed_observation_action_request_count") != 9
        or accounting.get("behavioral_model_action_request_count") != 309
        or accounting.get("behavioral_episode_count") != 12
        or accounting.get("fixed_observation_requests_in_behavioral_denominator") != 0
        or accounting.get("setup_invalid_attempt_count") != 16
        or accounting.get("setup_invalid_attempts_in_behavioral_denominator") != 0
        or accounting.get("behavioral_denominator_count") != 12
    ):
        raise ValueError("V2-A015 historical inference accounting changed")
    return {
        "record_scope": "authenticated_compact_historical_qualification_record",
        "raw_fixed_observation_probe_bytes_present_in_package": False,
        "raw_fixed_observation_probe_bytes_replayed_now": False,
        "fixed_observation_action_requests_historically_reported": 9,
        "fixed_observation_requests_in_behavioral_denominator": 0,
        "setup_invalid_attempts_historically_reported": 16,
        "setup_invalid_attempts_in_behavioral_denominator": 0,
        "cosmos3_nano_g1_fixed_input_gate_status": "passed_historically_not_replayed_now",
        "dreamzero_s1_overlay_equivalence_status": "passed_historically_not_replayed_now",
        "dreamzero_s2_fixed_input_gate_status": "passed_historically_not_replayed_now",
        "dreamzero_official_noise_seed": 1140,
    }


def _requested_margin(episode: Mapping[str, Any], relation: str, label: str) -> float:
    if "requested_signed_final_margin_m" in episode:
        return _finite(episode["requested_signed_final_margin_m"], label)
    lateral = _finite(episode.get("final_lateral_display_m"), label)
    return -lateral if relation == "left" else lateral


def _episode_map(document: Mapping[str, Any], cohort: str, arm: str) -> dict[tuple[int, str], dict[str, Any]]:
    episodes = document.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 6:
        raise ValueError(f"{cohort} {arm} must contain exactly six episodes")
    expected_model = COHORTS[cohort]["baseline_model_id"]
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise ValueError(f"{cohort} {arm} episode {index} is not an object")
        seed = _integer(episode.get("environment_seed"), f"{cohort}.{arm}.seed")
        relation = episode.get("requested_relation")
        key = (seed, relation)
        if seed not in SEEDS or relation not in RELATIONS or key in rows:
            raise ValueError(f"{cohort} {arm} episode grid changed: {key}")
        model = episode.get("model_id")
        if cohort == "H04" and arm == "intervention":
            if model != "dreamzero_droid_action_cfg":
                raise ValueError("H04 intervention model identity changed")
        elif model != expected_model:
            raise ValueError(f"{cohort} {arm} model identity changed")
        if episode.get("prompt") != PROMPTS[relation] or episode.get("dynamic_prompt_switches") != 0:
            raise ValueError(f"{cohort} {arm} prompt contract changed: {key}")
        if (
            episode.get("prompt_controller") != "episode_static"
            or episode.get("oracle_actions") != 0
            or episode.get("pair_id") != f"droid_pair_seed_{seed}"
        ):
            raise ValueError(f"{cohort} {arm} static prompt/pair identity changed: {key}")
        if episode.get("sampling_seed") != seed:
            raise ValueError(f"{cohort} {arm} environment/sampling seed join changed: {key}")
        if cohort == "H03" and arm == "intervention" and (
            episode.get("amendment_id") != "V2-A015"
            or episode.get("arm_id") != "cosmos3_nano_no_cfg_g1"
            or episode.get("guidance") != 1.0
            or episode.get("baseline_guidance") != 3.0
            or episode.get("checkpoint_revision") != "6706d7680581c255ff61e0f3bb49d90eac55c79e"
            or episode.get("source_commit") != EXTERNAL_MODEL_SOURCE_COMMITS["cosmos3_nano"]
        ):
            raise ValueError(f"H03 intervention episode configuration changed: {key}")
        if cohort == "H04" and arm == "intervention" and (
            episode.get("amendment_id") != "V2-A015"
            or episode.get("arm_id") != "dreamzero_action_cfg_s2"
            or episode.get("action_cfg_style_scale") != 2.0
            or episode.get("baseline_action_cfg_equivalent") != 1.0
            or episode.get("video_cfg_scale") != 5.0
            or episode.get("official_repository_commit") != EXTERNAL_MODEL_SOURCE_COMMITS["dreamzero"]
        ):
            raise ValueError(f"H04 intervention episode configuration changed: {key}")
        success = episode.get("requested_success")
        if type(success) is not bool:
            raise ValueError(f"{cohort} {arm} success must be boolean: {key}")
        actions = _integer(episode.get("actions_executed"), f"{cohort}.{arm}.actions")
        if not 1 <= actions <= 450:
            raise ValueError(f"{cohort} {arm} actions outside retained cap: {key}")
        requests = _integer(episode.get("policy_request_count"), f"{cohort}.{arm}.policy_requests")
        if requests < 1:
            raise ValueError(f"{cohort} {arm} policy request count is not positive: {key}")
        physical = _sha256_string(
            episode.get("physical_initial_state_sha256"),
            f"{cohort}.{arm}.physical_initial_state_sha256",
        )
        if physical != PHYSICAL_RESET_FINGERPRINT:
            raise ValueError(f"{cohort} {arm} physical-reset fingerprint boundary changed: {key}")
        trace = episode.get("executed_action_trace")
        if not isinstance(trace, dict) or not isinstance(trace.get("path"), str) or not trace["path"]:
            raise ValueError(f"{cohort} {arm} executed-action trace path changed: {key}")
        trace_sha256 = _sha256_string(
            trace.get("sha256"), f"{cohort}.{arm}.executed_action_trace.sha256"
        )
        trace_bytes = _integer(trace.get("bytes"), f"{cohort}.{arm}.executed_action_trace.bytes")
        if trace_bytes < 1:
            raise ValueError(f"{cohort} {arm} executed-action trace is empty: {key}")
        completion_status = episode.get("completion_action_status")
        completion_observed = episode.get("completion_actions_observed")
        if success:
            if completion_status is not None and (
                completion_status != "observed_success_event" or completion_observed != actions
            ):
                raise ValueError(f"{cohort} {arm} observed completion semantics changed: {key}")
            completion_right_censored = False
            completion_time_actions: int | None = actions
        else:
            if actions != 450:
                raise ValueError(f"{cohort} {arm} failure is not right-censored at the action cap: {key}")
            if completion_status is not None and (
                completion_status != "right_censored_at_450_action_cap" or completion_observed is not None
            ):
                raise ValueError(f"{cohort} {arm} censored completion semantics changed: {key}")
            completion_right_censored = True
            completion_time_actions = None
        if cohort == "H04" and episode.get("effective_official_model_noise_seed") != 1140:
            raise ValueError(f"{cohort} {arm} effective fixed noise changed: {key}")
        rows[key] = {
            "success": success,
            "policy_request_count": requests,
            "actions_executed": actions,
            "requested_margin_m": _requested_margin(episode, relation, f"{cohort}.{arm}.margin"),
            "physical_initial_state_sha256": physical,
            "executed_action_trace_path": trace["path"],
            "executed_action_trace_path_sha256": hashlib.sha256(trace["path"].encode("utf-8")).hexdigest(),
            "executed_action_trace_sha256": trace_sha256,
            "executed_action_trace_bytes": trace_bytes,
            "completion_right_censored": completion_right_censored,
            "completion_time_actions": completion_time_actions,
        }
    if set(rows) != {(seed, relation) for seed in SEEDS for relation in RELATIONS}:
        raise ValueError(f"{cohort} {arm} does not cover the exact 3 x 2 grid")
    return rows


def _summary(document: Mapping[str, Any], rows: Mapping[tuple[int, str], Mapping[str, Any]], label: str) -> dict[str, Any]:
    source = document.get("summary")
    if not isinstance(source, dict):
        source = document
    successes = sum(row["success"] for row in rows.values())
    by_relation: dict[str, dict[str, Any]] = {}
    for relation in RELATIONS:
        selected = [rows[(seed, relation)] for seed in SEEDS]
        by_relation[relation] = {
            "episodes": 3,
            "successes": sum(row["success"] for row in selected),
            "mean_requested_margin_m": statistics.fmean(row["requested_margin_m"] for row in selected),
        }
        source_relation = source.get("by_direction", {}).get(relation)
        if not isinstance(source_relation, dict):
            source_relation = source.get("success_by_relation", {}).get(relation)
        if not isinstance(source_relation, dict):
            raise ValueError(f"{label} source relation summary missing: {relation}")
        source_trials = source_relation.get("episodes", source_relation.get("trials"))
        if source_trials != 3 or source_relation.get("successes") != by_relation[relation]["successes"]:
            raise ValueError(f"{label} source relation counts changed: {relation}")
    source_episodes = source.get("valid_episode_count", source.get("episode_count"))
    source_successes = source.get("requested_success_count", source.get("successes"))
    if source_episodes != 6 or source_successes != successes:
        raise ValueError(f"{label} source aggregate counts changed")
    return {"episodes": 6, "successes": successes, "failures": 6 - successes, "by_relation": by_relation}


def _validate_configuration(documents: Mapping[str, Any]) -> None:
    nano_base = documents[NANO_BASELINE]
    nano_int = documents[NANO_INTERVENTION]
    if (
        nano_base.get("status") != "complete"
        or nano_base.get("schema_version") != "vla-wam-shared-v2-cosmos3-nano-policy-droid-result-v1"
        or nano_base.get("amendment_id") != "V2-A011"
        or nano_base.get("model_id") != "cosmos3_nano_policy_droid"
        or nano_base.get("arm_id") is not None
        or nano_base.get("compiled_at_git_head") != RECORDED_HEADS["nano_baseline_compilation"]
        or nano_base.get("checkpoint_revision") != "6706d7680581c255ff61e0f3bb49d90eac55c79e"
    ):
        raise ValueError("H03 baseline identity changed")
    nano_cfg = nano_int.get("configuration", {})
    if (
        nano_int.get("status") != "complete"
        or nano_int.get("schema_version") != "vla-wam-shared-v2-cosmos3-nano-v2a015-g1-result-v1"
        or nano_int.get("amendment_id") != "V2-A015"
        or nano_int.get("arm_id") != "cosmos3_nano_no_cfg_g1"
        or nano_int.get("model_id") != "cosmos3_nano_policy_droid"
        or nano_int.get("compiled_at_git_head") != RECORDED_HEADS["v2a015_result_compilation"]
        or nano_cfg.get("guidance") != 1.0
        or nano_cfg.get("baseline_guidance") != 3.0
        or nano_cfg.get("checkpoint_revision") != "6706d7680581c255ff61e0f3bb49d90eac55c79e"
        or nano_cfg.get("source_commit") != EXTERNAL_MODEL_SOURCE_COMMITS["cosmos3_nano"]
    ):
        raise ValueError("H03 intervention identity changed")

    dream_base = documents[DREAMZERO_BASELINE]
    dream_int = documents[DREAMZERO_INTERVENTION]
    if (
        dream_base.get("status") != "complete"
        or dream_base.get("schema_version") != "vla-wam-shared-v2-dreamzero-droid-direct-gate-v1"
        or dream_base.get("amendment_id") != "V2-A007"
        or dream_base.get("model_id") != "dreamzero_droid"
        or dream_base.get("arm_id") is not None
        or dream_base.get("compiled_at_git_head") != RECORDED_HEADS["dreamzero_baseline_compilation"]
        or dream_base.get("future_interface") != "joint_action_and_latent_video_prediction_with_official_decode_path"
    ):
        raise ValueError("H04 conditional-equivalent baseline identity changed")
    dream_cfg = dream_int.get("configuration", {})
    if (
        dream_int.get("status") != "complete"
        or dream_int.get("schema_version") != "vla-wam-shared-v2-dreamzero-v2a015-s2-result-v1"
        or dream_int.get("amendment_id") != "V2-A015"
        or dream_int.get("arm_id") != "dreamzero_action_cfg_s2"
        or dream_int.get("model_id") != "dreamzero_droid_action_cfg"
        or dream_int.get("compiled_at_git_head") != RECORDED_HEADS["v2a015_result_compilation"]
        or dream_cfg.get("action_cfg_style_scale") != 2.0
        or dream_cfg.get("baseline_action_cfg_equivalent") != 1.0
        or dream_cfg.get("video_cfg_scale") != 5.0
        or dream_cfg.get("checkpoint_revision") != "96ad344138c66e82536422432ad742f015784942"
        or dream_cfg.get("source_commit") != EXTERNAL_MODEL_SOURCE_COMMITS["dreamzero"]
    ):
        raise ValueError("H04 s2 intervention identity changed")

    for intervention_path, baseline_path in (
        (NANO_INTERVENTION, NANO_BASELINE),
        (DREAMZERO_INTERVENTION, DREAMZERO_BASELINE),
    ):
        preserved = documents[intervention_path].get("provenance", {}).get("preserved_baseline_result", {})
        expected = EXPECTED_SOURCES[baseline_path]
        if preserved.get("bytes") != expected["bytes"] or preserved.get("sha256") != expected["sha256"]:
            raise ValueError(f"intervention is not bound to authenticated baseline: {intervention_path}")


def _validate_comparison_provenance(comparison: Mapping[str, Any]) -> None:
    expected = {
        "cosmos3_nano_baseline": NANO_BASELINE,
        "cosmos3_nano_intervention": NANO_INTERVENTION,
        "dreamzero_baseline": DREAMZERO_BASELINE,
        "dreamzero_intervention": DREAMZERO_INTERVENTION,
    }
    provenance = comparison.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != set(expected):
        raise ValueError("comparison provenance population changed")
    for name, path in expected.items():
        source = provenance[name]
        identity = EXPECTED_SOURCES[path]
        if source.get("bytes") != identity["bytes"] or source.get("sha256") != identity["sha256"]:
            raise ValueError(f"comparison provenance identity changed: {name}")


def analyze_documents(documents: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    _validate_scope(str(documents["ablation_spec"]))
    post_result_disclosure = _validate_post_result_amendment(documents)
    historical_qualification = _validate_historical_qualification(documents)
    _validate_configuration(documents)
    comparison = documents[COMPARISON]
    if (
        comparison.get("schema_version") != "vla-wam-shared-v2-cfg-ablation-v2a015-comparison-v1"
        or comparison.get("status") != "complete"
        or comparison.get("amendment_id") != "V2-A015"
        or comparison.get("compiled_at_git_head") != RECORDED_HEADS["v2a015_result_compilation"]
        or comparison.get("exact_prompts") != PROMPTS
    ):
        raise ValueError("retained V2-A015 comparison identity changed")
    if "post-result" not in comparison.get("inference_boundary", ""):
        raise ValueError("retained comparison lost its post-result inference boundary")
    _validate_comparison_provenance(comparison)

    cohorts: dict[str, Any] = {}
    matched_rows: list[dict[str, Any]] = []
    matched_seed_diagnostics: list[dict[str, Any]] = []
    action_trace_paths: set[str] = set()
    action_trace_path_hashes: set[str] = set()
    action_trace_sha256s: set[str] = set()
    stable_cell_ids: set[str] = set()
    for cohort, info in COHORTS.items():
        baseline = _episode_map(documents[info["baseline_path"]], cohort, "baseline")
        intervention = _episode_map(documents[info["intervention_path"]], cohort, "intervention")
        baseline_summary = _summary(documents[info["baseline_path"]], baseline, f"{cohort}.baseline")
        intervention_summary = _summary(documents[info["intervention_path"]], intervention, f"{cohort}.intervention")
        physical_fingerprints = {
            row["physical_initial_state_sha256"]
            for source in (baseline, intervention)
            for row in source.values()
        }
        if physical_fingerprints != {PHYSICAL_RESET_FINGERPRINT}:
            raise ValueError(f"{cohort} does not retain the exact physical-reset fingerprint boundary")
        accounting = {
            "baseline_model_requests": sum(row["policy_request_count"] for row in baseline.values()),
            "baseline_actions_executed": sum(row["actions_executed"] for row in baseline.values()),
            "intervention_model_requests": sum(row["policy_request_count"] for row in intervention.values()),
            "intervention_actions_executed": sum(row["actions_executed"] for row in intervention.values()),
        }
        if accounting != EXPECTED_BEHAVIORAL_ACCOUNTING[cohort]:
            raise ValueError(f"{cohort} retained request/action accounting changed")
        compact = comparison.get("comparisons", {}).get(info["comparison_key"])
        if not isinstance(compact, dict):
            raise ValueError(f"retained comparison missing {cohort}")
        if compact.get("baseline_label") != info["baseline_label"] or compact.get("intervention_label") != info["intervention_label"]:
            raise ValueError(f"{cohort} comparison labels changed")
        if compact.get("model") != info["model_label"] or compact.get("exact_prompts") != PROMPTS:
            raise ValueError(f"{cohort} comparison model/prompt identity changed")
        compact_cells = compact.get("cells")
        if not isinstance(compact_cells, list) or len(compact_cells) != 6:
            raise ValueError(f"{cohort} comparison must contain six cells")
        compact_by_id = {row.get("cell_id"): row for row in compact_cells if isinstance(row, dict)}
        if len(compact_by_id) != 6:
            raise ValueError(f"{cohort} comparison cell identities changed")

        transitions: Counter[str] = Counter()
        for seed in SEEDS:
            for relation in RELATIONS:
                key = (seed, relation)
                before, after = baseline[key], intervention[key]
                if before["physical_initial_state_sha256"] != after["physical_initial_state_sha256"]:
                    raise ValueError(f"{cohort} baseline/intervention physical-state join changed: {key}")
                baseline_cell_id = (
                    f"{info['model_family']}::{info['baseline_configuration_id']}::"
                    f"seed{seed}::{relation}"
                )
                intervention_cell_id = (
                    f"{info['model_family']}::{info['intervention_configuration_id']}::"
                    f"seed{seed}::{relation}"
                )
                for stable_id, episode in (
                    (baseline_cell_id, before),
                    (intervention_cell_id, after),
                ):
                    if stable_id in stable_cell_ids:
                        raise ValueError(f"duplicate stable family/configuration/seed/relation identity: {stable_id}")
                    stable_cell_ids.add(stable_id)
                    raw_path = episode["executed_action_trace_path"]
                    path_hash = episode["executed_action_trace_path_sha256"]
                    trace_hash = episode["executed_action_trace_sha256"]
                    if raw_path in action_trace_paths or path_hash in action_trace_path_hashes:
                        raise ValueError("retained executed-action trace path identity is not distinct")
                    if trace_hash in action_trace_sha256s:
                        raise ValueError("retained executed-action trace SHA-256 is not distinct")
                    action_trace_paths.add(raw_path)
                    action_trace_path_hashes.add(path_hash)
                    action_trace_sha256s.add(trace_hash)
                transition = (
                    "unchanged_success" if before["success"] and after["success"]
                    else "regressed_success_to_failure" if before["success"]
                    else "improved_failure_to_success" if after["success"]
                    else "unchanged_failure"
                )
                transitions[transition] += 1
                cell_id = f"seed{seed}_{relation}"
                source_cell = compact_by_id.get(cell_id)
                if source_cell is None or source_cell.get("environment_seed") != seed or source_cell.get("requested_relation") != relation:
                    raise ValueError(f"{cohort} comparison join changed: {cell_id}")
                if source_cell.get("prompt") != PROMPTS[relation] or source_cell.get("physical_initial_state_sha256") != before["physical_initial_state_sha256"]:
                    raise ValueError(f"{cohort} comparison prompt/state changed: {cell_id}")
                labels = (info["baseline_label"], info["intervention_label"])
                source_success = source_cell.get("success", {})
                source_margin = source_cell.get("requested_signed_final_margin_m", {})
                if source_success.get(labels[0]) is not before["success"] or source_success.get(labels[1]) is not after["success"]:
                    raise ValueError(f"{cohort} comparison success changed: {cell_id}")
                if source_success.get("transition") != transition:
                    raise ValueError(f"{cohort} comparison transition changed: {cell_id}")
                _close(source_margin.get(labels[0]), before["requested_margin_m"], f"{cohort}.{cell_id}.baseline_margin")
                _close(source_margin.get(labels[1]), after["requested_margin_m"], f"{cohort}.{cell_id}.intervention_margin")
                matched_rows.append({
                    "cohort": cohort,
                    "model_family": info["model_family"],
                    "model_id": info["model_id"],
                    "baseline_model_id": info["baseline_model_id"],
                    "intervention_model_id": info["model_id"],
                    "environment_seed": seed,
                    "relation": relation,
                    "baseline_cell_id": baseline_cell_id,
                    "intervention_cell_id": intervention_cell_id,
                    "baseline_configuration_id": info["baseline_configuration_id"],
                    "intervention_configuration_id": info["intervention_configuration_id"],
                    "baseline_label": labels[0],
                    "intervention_label": labels[1],
                    "baseline_success": before["success"],
                    "intervention_success": after["success"],
                    "success_transition": transition,
                    "baseline_model_requests": before["policy_request_count"],
                    "intervention_model_requests": after["policy_request_count"],
                    "baseline_actions_executed": before["actions_executed"],
                    "intervention_actions_executed": after["actions_executed"],
                    "baseline_completion_right_censored": before["completion_right_censored"],
                    "intervention_completion_right_censored": after["completion_right_censored"],
                    "baseline_completion_time_actions": before["completion_time_actions"],
                    "intervention_completion_time_actions": after["completion_time_actions"],
                    "baseline_requested_margin_m": before["requested_margin_m"],
                    "intervention_requested_margin_m": after["requested_margin_m"],
                    "intervention_minus_baseline_margin_m": after["requested_margin_m"] - before["requested_margin_m"],
                    "baseline_action_trace_path_sha256": before["executed_action_trace_path_sha256"],
                    "intervention_action_trace_path_sha256": after["executed_action_trace_path_sha256"],
                    "baseline_action_trace_sha256": before["executed_action_trace_sha256"],
                    "intervention_action_trace_sha256": after["executed_action_trace_sha256"],
                    "physical_initial_state_sha256": before["physical_initial_state_sha256"],
                })

            seed_rows = [
                row for row in matched_rows
                if row["cohort"] == cohort and row["environment_seed"] == seed
            ]
            if len(seed_rows) != 2 or {row["relation"] for row in seed_rows} != set(RELATIONS):
                raise ValueError(f"{cohort} matched-seed diagnostic join changed: {seed}")
            matched_seed_diagnostics.append({
                "cohort": cohort,
                "model_family": info["model_family"],
                "model_id": info["model_id"],
                "environment_seed_label": seed,
                "relations": list(RELATIONS),
                "physical_reset_fingerprint_sha256": PHYSICAL_RESET_FINGERPRINT,
                "same_physical_reset_fingerprint_for_both_relations": True,
                "baseline_successes": sum(row["baseline_success"] for row in seed_rows),
                "intervention_successes": sum(row["intervention_success"] for row in seed_rows),
                "baseline_model_requests": sum(row["baseline_model_requests"] for row in seed_rows),
                "intervention_model_requests": sum(row["intervention_model_requests"] for row in seed_rows),
                "baseline_actions_executed": sum(row["baseline_actions_executed"] for row in seed_rows),
                "intervention_actions_executed": sum(row["intervention_actions_executed"] for row in seed_rows),
            })

        expected_transitions = dict(sorted(transitions.items()))
        if compact.get("success", {}).get("exact_paired_transitions") != expected_transitions:
            raise ValueError(f"{cohort} compact transition totals changed")
        if compact.get("success", {}).get("baseline_total") != baseline_summary["successes"]:
            raise ValueError(f"{cohort} compact baseline total changed")
        if compact.get("success", {}).get("intervention_total") != intervention_summary["successes"]:
            raise ValueError(f"{cohort} compact intervention total changed")
        for name, summary in (
            ("baseline_configuration_summary", baseline_summary),
            ("intervention_configuration_summary", intervention_summary),
        ):
            source_summary = compact.get(name, {})
            if source_summary.get("valid_episode_count") != 6 or source_summary.get("requested_success_count") != summary["successes"]:
                raise ValueError(f"{cohort} compact configuration totals changed: {name}")
            for relation in RELATIONS:
                relation_source = source_summary.get("by_direction", {}).get(relation, {})
                relation_expected = summary["by_relation"][relation]
                if relation_source.get("episodes") != 3 or relation_source.get("successes") != relation_expected["successes"]:
                    raise ValueError(f"{cohort} compact direction totals changed: {name}.{relation}")
                _close(
                    relation_source.get("requested_margin_m", {}).get("mean"),
                    relation_expected["mean_requested_margin_m"],
                    f"{cohort}.{name}.{relation}.mean_margin",
                )
        cohorts[cohort] = {
            "model_id": info["model_id"],
            "model_family": info["model_family"],
            "baseline_model_id": info["baseline_model_id"],
            "intervention_model_id": info["model_id"],
            "model_label": info["model_label"],
            "baseline_label": info["baseline_label"],
            "intervention_label": info["intervention_label"],
            "authenticated_population": {
                "baseline_behavioral_episodes": 6,
                "intervention_behavioral_episodes": 6,
                "matched_configuration_cells": 6,
                "matched_seed_labels": 3,
                "missing_value_imputation": "none",
                "physical_reset_fingerprints": 1,
            },
            "successes": {
                "baseline": baseline_summary["successes"],
                "intervention": intervention_summary["successes"],
            },
            "behavioral_request_action_accounting": accounting,
            "matched_environment_seeds": list(SEEDS),
            "baseline": baseline_summary,
            "intervention": intervention_summary,
            "success_transitions": expected_transitions,
            "net_success_change": intervention_summary["successes"] - baseline_summary["successes"],
            "scope": "descriptive_post_result_exploratory_pilot",
        }

    if (
        len(matched_rows) != 12
        or len(matched_seed_diagnostics) != 6
        or len(stable_cell_ids) != 24
        or len(action_trace_paths) != 24
        or len(action_trace_path_hashes) != 24
        or len(action_trace_sha256s) != 24
    ):
        raise ValueError("retained H03/H04 population identity counts changed")
    v2a015_behavioral_requests = sum(
        cohorts[cohort]["behavioral_request_action_accounting"]["intervention_model_requests"]
        for cohort in COHORTS
    )
    if v2a015_behavioral_requests != 309:
        raise ValueError("V2-A015 behavioral request total changed")
    fixed_observation_requests = historical_qualification[
        "fixed_observation_action_requests_historically_reported"
    ]
    if fixed_observation_requests != 9 or v2a015_behavioral_requests + fixed_observation_requests != 318:
        raise ValueError("V2-A015 total model-action request accounting changed")

    return {
        "schema_version": "gm-forecast-ablation-retained-h03-h04-guidance-pilots-v1",
        "provenance": provenance,
        "source_manifest": {
            "authenticated_bytes": sum(item["bytes"] for item in EXPECTED_SOURCES.values()),
            "authenticated_file_count": len(EXPECTED_SOURCES),
            "authenticated_lines": sum(item["lines"] for item in EXPECTED_SOURCES.values()),
            "compact_git_files_byte_rehashed_now": True,
            "exported_source_paths_are_git_object_paths_only": True,
            "raw_pvc_files_revalidated_now": False,
            "raw_source_paths_exported": False,
        },
        "design": {
            "amendment_id": "V2-A015",
            "cohorts": ["H03", "H04"],
            "matched_environment_seeds_per_cohort": 3,
            "relations": list(RELATIONS),
            "episodes_per_arm_per_cohort": 6,
            "historical_episode_rows_authenticated": 24,
            "retained_baseline_behavioral_episodes": 12,
            "new_v2a015_behavioral_episodes": 12,
            "historical_matched_baseline_intervention_cells": 12,
            "stable_family_configuration_seed_relation_identities": 24,
            "distinct_executed_action_trace_paths": 24,
            "distinct_executed_action_trace_sha256": 24,
            "physical_reset_fingerprints": 1,
            "physical_reset_fingerprint_sha256": PHYSICAL_RESET_FINGERPRINT,
            "environment_seed_labels_are_independent_physical_scenes": False,
            "dreamzero_official_model_noise_seed": 1140,
            "dreamzero_noise_seed_varies_with_environment_seed_label": False,
            "inferential_scope": "descriptive_post_result_no_powered_generalization",
        },
        "post_result_disclosure": post_result_disclosure,
        "historical_fixed_input_qualification": historical_qualification,
        "historical_v2a015_request_accounting": {
            "behavioral_model_action_requests": v2a015_behavioral_requests,
            "fixed_observation_action_requests": fixed_observation_requests,
            "total_model_action_requests": v2a015_behavioral_requests + fixed_observation_requests,
            "setup_invalid_attempts": 16,
            "setup_invalid_attempts_in_behavioral_denominator": 0,
            "runtime_interventions": 0,
        },
        "cohorts": cohorts,
        "matched_cell_contrasts": matched_rows,
        "matched_seed_diagnostics": matched_seed_diagnostics,
        "reproduction": {
            "strict_json_duplicate_key_rejection": True,
            "strict_json_nonfinite_rejection": True,
            "authenticated_compact_git_files_rehashed": 7,
            "authenticated_compact_git_bytes_rehashed": 810_860,
            "authenticated_compact_git_lines_counted": 18_939,
            "episode_rows_recomputed": 24,
            "matched_cell_contrasts_recomputed": 12,
            "matched_seed_diagnostics_recomputed": 6,
            "request_action_totals_recomputed": True,
            "right_censoring_recomputed": True,
            "raw_fixed_observation_probe_bytes_replayed": False,
            "external_model_source_commits_locally_authenticated": False,
            "external_model_source_commit_literals": dict(EXTERNAL_MODEL_SOURCE_COMMITS),
        },
        "new_activity": {
            "model_requests": 0,
            "behavioral_actions": 0,
            "behavioral_cells": 0,
            "request_rows": 0,
            "request_selections": 0,
            "annotation_media": 0,
            "human_responses": 0,
            "labels": 0,
            "forecast_accuracy_estimates": 0,
            "retrieval_jobs": 0,
            "queue_descriptors": 0,
            "runnable_manifests": 0,
            "confirmation_inference_cells": 0,
        },
        "gate_state": {
            "current_GM_PVC_worker_state_reconciled": False,
            "live_quiescence_inferred": False,
            "camera_alignment_ready": False,
            "development_annotation_gate_ready": False,
            "annotation_media_ready": False,
            "human_annotation_ready": False,
            "resource_freeze_ready": False,
            "confirmation_release_ready": False,
            "safe_to_release_new_work": False,
        },
        "claim_boundary": {
            "retained_historical_evidence_only": True,
            "exploratory_post_result_only": True,
            "post_result_selection_disclosed": True,
            "raw_fixed_observation_probe_bytes_replayed_now": False,
            "historical_fixed_input_gate_is_current_requalification": False,
            "powered_inference": False,
            "powered_or_general_performance_claim": False,
            "new_model_inference": False,
            "new_robot_episodes": False,
            "new_human_labels": False,
            "forecast_accuracy_evidence": False,
            "forecast_accuracy_claim": False,
            "fixed_video_guidance_intervention": False,
            "official_dreamzero_action_cfg_claim": False,
            "core_D1_qualification": False,
            "H04_s1_may_fill_core_D1": False,
            "new_confirmation_d2_extension_selected": False,
            "historical_h04_custom_s2_retained": True,
            "independent_scene_generalization": False,
            "environment_seed_labels_claimed_as_independent_scenes": False,
            "fixed_noise_labels_claimed_as_independent_noise_replicates": False,
            "models_pooled": False,
            "confirmation_test_or_stage_gate": False,
            "changes_232_cell_core_bound": False,
            "maximum_core_behavioral_cells": 232,
            "boundaries": [
                "H03 and H04 include preserved baselines; these 24 historical rows are not added to current core-cell accounting.",
                "The three repeated environment seeds per cohort are matched pilot labels, not independent physical scenes.",
                "H04 changes action guidance during denoising and does not hold the generated future fixed.",
                "The DreamZero s2 arm is CFG-style negative-branch action guidance, not an official DreamZero action-CFG mode.",
                "All 24 rows share one reset fingerprint, and DreamZero uses fixed official noise seed 1140; seed labels are not independent physical scenes or noise replicates.",
                "The historical conditional-action-equivalent s1 baseline is not a substitute for qualified official core D1.",
                "The compact historical preflight says the fixed-input gates and s1-overlay equivalence passed; this adapter did not replay their raw probe bytes.",
                "No raw PVC file, camera-time alignment, annotation, forecast endpoint, or current cluster state was revalidated.",
            ],
        },
    }


CSV_FIELDS = (
    "cohort", "model_family", "model_id", "baseline_model_id",
    "intervention_model_id", "environment_seed", "relation",
    "baseline_cell_id", "intervention_cell_id", "baseline_configuration_id",
    "intervention_configuration_id", "baseline_label", "intervention_label",
    "baseline_success", "intervention_success", "success_transition",
    "baseline_model_requests", "intervention_model_requests",
    "baseline_actions_executed", "intervention_actions_executed",
    "baseline_completion_right_censored", "intervention_completion_right_censored",
    "baseline_completion_time_actions", "intervention_completion_time_actions",
    "baseline_requested_margin_m", "intervention_requested_margin_m",
    "intervention_minus_baseline_margin_m", "baseline_action_trace_path_sha256",
    "intervention_action_trace_path_sha256", "baseline_action_trace_sha256",
    "intervention_action_trace_sha256", "physical_initial_state_sha256",
)


def render_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def render_csv(report: Mapping[str, Any]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in report["matched_cell_contrasts"]:
        writer.writerow({field: row[field] for field in CSV_FIELDS})
    return stream.getvalue()


def _check_or_write(path: Path, content: str, *, check: bool) -> None:
    if check:
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            raise ValueError(f"canonical output differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--check", action="store_true", help="fail unless checked-in outputs reproduce exactly")
    args = parser.parse_args()
    documents, provenance = read_pinned_sources(ROOT)
    report = analyze_documents(documents, provenance)
    _check_or_write(args.json, render_json(report), check=args.check)
    _check_or_write(args.csv, render_csv(report), check=args.check)
    print(
        f"H03/H04 retained package: {len(report['matched_cell_contrasts'])} paired cells, "
        f"{report['source_manifest']['authenticated_file_count']} authenticated Git objects; "
        f"check={args.check}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
