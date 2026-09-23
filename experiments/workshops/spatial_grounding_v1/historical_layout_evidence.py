"""Compile additive, hash-bound historical DROID layout evidence.

This registry is deliberately not a replacement for the existing inventory or
comparator. It records provenance and unresolved raw bindings, while emitting
comparable pose rows only when an anchored source chain contains measured
root poses and geometric bounds under an explicit coordinate convention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import math
from pathlib import Path
from typing import Any

from .fixtures import Pose, FixtureError
from .lat_workspace_capture import _root_local_offset

REPO_ROOT = Path(__file__).resolve().parents[3]

# Sources audited on divergent tips or added because they were omitted from the
# prior inventory. The commit/blob/hash tuple is the binding; no branch is
# merged or treated as a local checkout.
SOURCE_SPECS = (
    ("forecast-fixture-adapter", "5252511b0bb9f574cd161a479e1129a512328221", "workshops/corl2026_world_models/experiments/forecast_layout/robolab_fixture_gate_adapter.py", "0113723ed554a7fd2e5014606b37fc7483237813", "28cd6105010892dcabe2f4bfd61277f1e7cf41d0a42086f4879087f4d3268b23", "world.get_pose settled actor roots and world.get_bbox corner extrema in the same RoboLab environment-local axes; historical fields label these robot_base_m"),
    ("v1-run-manifest", "167c8a55f040506fbd8d5532c700bac77b5e190d", "artifacts/vla_wam_shared_v1/run_manifest.json", "d772d88bf7f209a348243799faf8ea0a1224fd65", "eadac021a1b96544daf1300843aa9ec243f7d3bd7f1a7c0814c0c3bc06f23b2c", "earlier DROID cohort locations"),
    ("v1-initial-state-schema", "167c8a55f040506fbd8d5532c700bac77b5e190d", "artifacts/vla_wam_shared_v1/initial_state_schema_amendment_006.json", "b79f6cdb20c3c4eff7e6f854c3284b8401eac5d7", "5af527b6c0da114cf8f58ef03d11103641b50431c6282fb0ba6406e3f004f4c0", "initial-state fingerprint schema"),
    ("v3e004-layout-candidate", "167c8a55f040506fbd8d5532c700bac77b5e190d", "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/layout/candidate.json", "3308981ab7951b136089e3adef129cee98785ac7", "7e270419cdbd2a00e36c47a6a5e3d10c7affb225f118e536cc43917da960ba33", "E004 layout definition"),
    ("v3b001-reset-preflight", "167c8a55f040506fbd8d5532c700bac77b5e190d", "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/live_reset_preflight_report.json", "c33c45657f036e5ee5ddf8ae5d8263caa3341f93", "075a3d78fff99a74bfbb3981e77c20b6de20a9d5123e5ed4f6dea10ee2f61bc9", "measured reset attestations"),
    ("forecast-confirmation-inventory", "5252511b0bb9f574cd161a479e1129a512328221", "workshops/corl2026_world_models/execution/20260912/confirmation_fixture_inventory.json", "6fa7a8178cb58c305cc72878f62be71e28368b4a", "1ad3efc394c33ffcdccd08ecbf9748babe95e474f124f02841ba5db4727fc623", "24 prior confirmation identities and PVC pose hashes"),
    ("forecast-layout-contract", "5252511b0bb9f574cd161a479e1129a512328221", "workshops/corl2026_world_models/experiments/forecast_layout/layout_source_contract.json", "7166b468a16ce6590106211dc15904fbfa31fccd", "88a1268ae7f27776fd5246a5808c069b2906399e99bb0e7d8b6f09020a6b85d3", "pinned RoboLab source metadata, not measured reset"),
    ("forecast-layout-candidates", "5252511b0bb9f574cd161a479e1129a512328221", "workshops/corl2026_world_models/experiments/forecast_layout/layout_candidate_pool.json", "485032723d4d83fb8d31d327e4a55faf03403c9c", "ec80f4adc5272ec666c94d753241b84c954ef382bd30e1f2907459bff1cdb2b1", "prior candidates, not replacement SGW candidates"),
    ("forecast-fixture-code", "5252511b0bb9f574cd161a479e1129a512328221", "workshops/corl2026_world_models/experiments/forecast_layout/fixture_layouts.py", "536b67481e0e715dc9827ea1289bc0be3f2209b8", "d124d1052402f278aad3f098f3592aca9aa3c912cea148cbad2c4e329d92b94a", "layout construction provenance"),
    ("forecast-d1-server", "5252511b0bb9f574cd161a479e1129a512328221", "workshops/corl2026_world_models/experiments/forecast_layout/d1_instrumented_server.py", "aa3855eb948d1d0f9351b71c024bd029dc689fef", "085aec66d7deb06cac2c50e61b3205899476185d70720330cb0c586416532801", "DreamZero instrumentation prior art"),
    ("c01-fixture-receipt", "12839900e5c46299e11fff4813e8602949055b86", "results/jobs/fixture-c01-candidate-01/publish/fixture_gate_receipt.json", "cd42b2e8f94baff7802adaea1668a78e2d588db0", "d52ec4b79b4fd848e3b785b52bcc7cec4d571ddd3a0dec5c1f9a9313b497731e", "historical fixture receipt"),
    ("c01-capture-receipt", "12839900e5c46299e11fff4813e8602949055b86", "results/jobs/fixed-observation-c01-001/publish/fixed_observation_job_receipt.json", "c2e17fedc334b147257442e6a7859de02ecbeb7f", "09778e9e5da7bbd2fc073595f6d1a4ea17023bbf7e417a9622dda36b9d25c221", "historical capture receipt; pose remains PVC-bound"),
    ("d1-development-receipt", "12839900e5c46299e11fff4813e8602949055b86", "results/jobs/d1-development-d01-simulator-003/publish/d1_behavioral_development_receipt.json", "dda8071c9c9f524d49fd98d8b6f3b12b397ed319", "9b1fd7d7479ad275d4baa943b9d8d6a4ed1d223e2d9ca20b3bff0342711ec5eb", "four development cells and raw receipt paths"),
    ("development-compiler-receipt", "12839900e5c46299e11fff4813e8602949055b86", "results/jobs/development-evidence-compiler-formal-004/publish/development_evidence_compiler_job_receipt.json", "6ea0e065d1ece6b0d603fbf7bf3b8aa410f59495", "00329a3c991ee6eb0c68996ecaca1aed59ea3017329272cd82b4b5bfb5cee434", "32-cell development compilation, not SGW release"),
    ("v4-continuation", "ab25c99883894ca1019351ff3373472c465817f0", "artifacts/online_correction_v4/continuation_state.json", "2c3375b0914038ec79f7a29fbfb3a7f9f5ebb60f", "61bd0fc03584fe2c63d930948845848b80d450d6ad1ef332a9b8b22e7ef10fe3", "separate campaign accounting"),
    ("v4-containment-registry", "ab25c99883894ca1019351ff3373472c465817f0", "artifacts/online_correction_v4/setup/containment_reset_registry.released.json", "80c0edf9a64acac7162833363f57e1b192907d8b", "bce1999983c378f1ee437e90c6e5dc60c0c1ab0c266e017aa20376d20ba9346c", "different DROID assets"),
    ("v4-object-pair-registry", "ab25c99883894ca1019351ff3373472c465817f0", "artifacts/online_correction_v4/setup/object_pair_reset_registry.released.json", "c68241d869b0fb2cae5c408adb5592540b43aee1", "e7ba17d23c008c47de3ee26374e0439f10da99f51a390fa73a64fb7095f8a71b", "sponge/tray assets, not cube/bowl"),
    ("v4-c8-widowx", "ab25c99883894ca1019351ff3373472c465817f0", "artifacts/online_correction_v4/setup/c8_confirmatory/reset_registry.released.json", "d75e5c22662e723fd546bf550725cf4b69c161ff", "8da6b69ca01e2de15b28df5fe980ae056c04033c9e0b9a3b0d5c1751b269890f", "exclude: SimplerEnv/WidowX"),
    ("v3e007-registration", "aa412b700a063fdc13995be3e6d58072436765d3", "artifacts/vla_wam_shared_v3/phase_e/zero_model_reachability_v3e007/registration.json", "d00cc1cfdba67de2880ae1787a754793c0d181b2", "ded9842301634e958844f3a2fda3abda04b0c92524da381a7e20eaf7d9ed8067", "14 analytical reference layouts, not physical roots"),
    ("v3e007-workspace", "aa412b700a063fdc13995be3e6d58072436765d3", "artifacts/vla_wam_shared_v3/phase_e/zero_model_reachability_v3e007/raw/workspace_summary.json", "36bc35caa0ceed9f8279bcda0f4b72480d97b877", "85a688917a94f946c397d9ee9cd3e4cb6952ba81a40c5b3873e71e172d975906", "analytical reachability only"),
    ("v3e007-evidence", "aa412b700a063fdc13995be3e6d58072436765d3", "artifacts/vla_wam_shared_v3/phase_e/zero_model_reachability_v3e007/results/evidence_manifest.json", "a0c59ecf0ac85d8009d857d25ac1a260c764439b", "402e5476c46ddc25e92bbf3a1a125fa2f5a3eee76bfb9ab9cd263bd75df565c4", "published E007 checksums"),
    ("v3e006-r012-index", "58159325f436dd946f423ea01268194fe27f052c", "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r012/publication/file_inventory.sha256.tsv", "989ede5359a08a14ec4be109e9e98d0fa43c4d03", "2238cb9556fb6243229cb84d9a845e16f8fb640113ee3a2e96ff2c5c45ad9eee", "260-file raw archive index"),
    ("v3e006-r012-lfs", "58159325f436dd946f423ea01268194fe27f052c", "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r012/publication/v3e006_r012_full_raw_a81bc0f.tar.gz", "e9e818e0420906906f0ef1f69dddb8a668373727", "5fbd203d06377f83484ce71b77afbf70d4e6afdb25c1dc5d2187b856f4a896d7", "LFS pointer only; payload unverified"),
)

PVC_POSE_MANIFESTS = tuple(
    (f"C{number:02d}", f"/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/fixed_observations/c{number:02d}_pose_manifest.json", digest)
    for number, digest in (
        (1, "3c2fff3db2b688eeb9b951eb2b4df880a4ddec98a5607d7632ab15ca2a25f9e4"),
        (2, "dc1722af5a3198842b3411f0f74e40ae1f8fd779c755d850d23116e6f0498f9d"),
        (3, "a03551b3bdbd5af1fc2c24f0fa6d31e1484aed92f63e07fabb5758f3afa7a18d"),
        (4, "cc2ab851d0b9911f6ca71a4e7284b0c81285b96aa1daa0a90fbe21e756cc57f8"),
        (5, "ebfc63276cf4a4abffcea76d086fbb6765ff81ef385980ce30bd564a4b3c7b22"),
        (6, "ecf6925c330574280162ccce7ec1bc0b798209abdae1f5b01cc1bb450faf186a"),
        (7, "b902bfaa621c48c307003110703048f8dcd27054023f84cd73e6a546940b1a6b"),
        (8, "f25a00367262512208f101d0a1bf010b3a19571ffc3415f5aecc9314067b0e98"),
        (9, "d92a0ef2dd7b23616971c596f924367ee8152698aaf6cf577a978b370193116a"),
        (10, "6504ab580bec240cd10146ea4be7a11cfabd91eab8cbfe7b58ed6db0ac527272"),
        (11, "b17b027c97f947d229c9c453614cf9aebf13a022e4143468998ec1f096bacbc5"),
        (12, "34905eac6058c41e9a648afc9edb9ee09051e032a70a6d2cb216a148eeb75116"),
        (13, "207f9352f889ad24111017fe0e5c24270a2cc528d53895bfcc5f0838a749bce3"),
        (14, "c5aed5dfc636d06f3d4565f4a64c9c269516a42190f9b6ac2c6bb9da636f7edb"),
        (15, "455edcf91e77e0c5c2576ffe77f4e075436948aa6305c10fe5fad89dfd939e4c"),
        (16, "1e02bc7bba4c861b66b702c47a8ea489b221aa84b548cc5c51309c09f312c028"),
        (17, "5f7298c53a566d1c6c036fb3a5240d007cff5001b800e98f69b9283db19c3b59"),
        (18, "1fcd75b5c2dda02ea095295ef494e1685ae7df8184d21913fd870fdd243ba2ba"),
        (19, "97904d5dd63f87e8d392bc75a20cf0128ae681d00e5eb17cbcd9b5ce8ddfc241"),
        (20, "640af57173d212ba9bc1a4da2c858bf6debc7d51c466aa85243ba054ad51b4ca"),
        (21, "97ad213b356dc682982623f1323c9d6956e2535d06cbf11c1b5a5a36004ea965"),
        (22, "ef75677e068aebee5f827e87cc38469bfbceef05e29cb49a5faf6858ee84974f"),
        (23, "e0d39f23c4c5fd16afa20e888110faa1f934cc5d2be0389ff37ff8a823d90904"),
        (24, "d7c37082210abb25f376038ec769eb739c1269ce8616d262c89c4ab29927de8c"),
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob(repo_root: Path, commit: str, relative: str) -> bytes | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "cat-file", "blob", f"{commit}:{relative}"],
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return None


def _pose_anchor_matches(path: str, entry: dict[str, Any]) -> bool:
    return any(path == expected_path and entry.get("sha256") == digest
               for _, expected_path, digest in PVC_POSE_MANIFESTS)


def _root_only_evidence(pose_export: Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if pose_export is None or not pose_export.is_file():
        return [], []
    payload = json.loads(pose_export.read_text())
    rows = []
    blockers = []
    for path, entry in payload.get("files", {}).items():
        text = entry.get("text")
        if (
            not isinstance(text, str)
            or entry.get("bytes") != len(text.encode())
            or entry.get("sha256") != hashlib.sha256(text.encode()).hexdigest()
        ):
            blockers.append({"manifest_path": path, "reason": "pose export entry failed byte/SHA256 validation"})
            continue
        if not _pose_anchor_matches(path, entry):
            blockers.append({"manifest_path": path, "reason": "pose export differs from its independently pinned historical digest"})
            continue
        try:
            document = json.loads(text)
        except json.JSONDecodeError:
            blockers.append({"manifest_path": path, "reason": "pose export entry is not valid JSON"})
            continue
        for layout_id, pair in document.get("layout_pairs", {}).items():
            for variant, layout in pair.get("layouts", {}).items():
                positions = layout.get("positions_robot_base_m")
                quaternions = layout.get("quaternions_wxyz")
                if not isinstance(positions, dict) or not isinstance(quaternions, dict):
                    continue
                rows.append({
                    "layout_pair_id": layout_id,
                    "variant": variant,
                    "manifest_path": path,
                    "manifest_sha256": entry["sha256"],
                    "positions_robot_base_m": positions,
                    "quaternions_wxyz": quaternions,
                    "asset_provenance_present": bool(pair.get("object_asset_provenance")),
                })
                blockers.append({
                    "layout_pair_id": layout_id,
                    "variant": variant,
                    "reason": "root positions/quaternions are present, but no measured root-local geometric offsets are present in the pose manifest",
                    "manifest_path": path,
                })
    return rows, blockers


def _gate_geometry(
    pose_export: Path | None, gate_export: Path | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if pose_export is None or gate_export is None or not pose_export.is_file() or not gate_export.is_file():
        return [], []
    poses = json.loads(pose_export.read_text()).get("files", {})
    gates = json.loads(gate_export.read_text()).get("files", {})

    def verified(entry: dict[str, Any]) -> bool:
        text = entry.get("text")
        return (
            isinstance(text, str)
            and entry.get("bytes") == len(text.encode())
            and entry.get("sha256") == hashlib.sha256(text.encode()).hexdigest()
        )

    if not all(verified(entry) for entry in (*poses.values(), *gates.values())):
        return [], [{"reason": "pose or gate export entry text failed its declared byte/SHA256 binding"}]
    if not all(_pose_anchor_matches(path, entry) for path, entry in poses.items()):
        return [], [{"reason": "pose export differs from its independently pinned historical digest"}]
    by_layout: dict[str, dict[str, Any]] = {}
    for path, entry in poses.items():
        document = json.loads(entry["text"])
        for layout_id, pair in document.get("layout_pairs", {}).items():
            receipt = pair.get("accepted_gate_attempt_receipt", {})
            by_layout[layout_id] = {
                "manifest_path": path,
                "manifest_sha256": entry["sha256"],
                "receipt_path": receipt.get("path"),
                "receipt_sha256": receipt.get("sha256"),
                "receipt_bytes": receipt.get("bytes"),
                "object_assets": pair.get("object_asset_provenance", {}),
            }
    rows = []
    seen = set()
    blockers = []
    for gate_path, entry in gates.items():
        document = json.loads(entry["text"])
        for capture_index, capture in enumerate(document.get("evaluation", {}).get("capture_evidence", [])):
            layout_id = capture.get("layout_pair_id")
            link = by_layout.get(layout_id)
            if (
                link is None or link["receipt_path"] != gate_path
                or link["receipt_sha256"] != entry["sha256"]
                or link["receipt_bytes"] != entry["bytes"]
            ):
                continue
            configured = capture.get("configured_poses", {})
            settled = capture.get("settled_poses", {})
            for object_name, root in configured.items():
                final = settled.get(object_name)
                if not isinstance(final, dict) or not isinstance(root, dict):
                    continue
                root_position = root.get("position_robot_base_m")
                settled_position = final.get("position_robot_base_m")
                settled_quaternion = final.get("quaternion_wxyz")
                if not all(isinstance(value, list) and len(value) == 3 for value in (root_position, settled_position)):
                    continue
                if not isinstance(settled_quaternion, list) or len(settled_quaternion) != 4:
                    continue
                try:
                    Pose.from_json({"position_m": settled_position, "quaternion_wxyz": settled_quaternion})
                    Pose.from_json({"position_m": root_position, "quaternion_wxyz": root.get("quaternion_wxyz", [])})
                except (FixtureError, TypeError, ValueError) as error:
                    blockers.append({"layout_pair_id": layout_id, "object": object_name, "reason": str(error)})
                    continue
                camera_geometries = [
                    camera.get("geometry", {})
                    for camera in capture.get("cameras", {}).values()
                    if isinstance(camera, dict)
                ]
                aabbs = [geometry.get("object_aabbs_robot_base_m", {}).get(object_name)
                         for geometry in camera_geometries]
                if not aabbs or any(not isinstance(bounds, dict) for bounds in aabbs):
                    continue
                lower = aabbs[0].get("lower")
                upper = aabbs[0].get("upper")
                if not (isinstance(lower, list) and isinstance(upper, list) and len(lower) == len(upper) == 3):
                    continue
                if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in lower + upper) or any(
                    low > high for low, high in zip(lower, upper, strict=True)
                ):
                    blockers.append({"layout_pair_id": layout_id, "object": object_name, "reason": "AABB bounds are nonfinite or inverted"})
                    continue
                if any(
                    bounds.get("lower") != lower or bounds.get("upper") != upper
                    for bounds in aabbs[1:]
                ):
                    blockers.append({
                        "layout_pair_id": layout_id,
                        "variant": capture.get("layout_arm"),
                        "reason": "per-camera AABB bounds are inconsistent",
                    })
                    continue
                center = [(float(lower[i]) + float(upper[i])) / 2 for i in range(3)]
                geometric_offset = _root_local_offset(settled_position, settled_quaternion, center)
                dedupe_key = (
                    layout_id,
                    capture.get("layout_arm"),
                    object_name,
                    tuple(root_position),
                    tuple(settled_position),
                    tuple(settled_quaternion),
                    tuple(lower),
                    tuple(upper),
                )
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                rows.append({
                    "layout_pair_id": layout_id,
                    "variant": capture.get("layout_arm"),
                    "object": object_name,
                    "configured_root_position_robot_base_m": root_position,
                    "settled_root_quaternion_wxyz": settled_quaternion,
                    "settled_position_robot_base_m": settled_position,
                    "reset_displacement_m": [
                        settled_position[i] - root_position[i] for i in range(3)
                    ],
                    "aabb_lower_robot_base_m": lower,
                    "aabb_upper_robot_base_m": upper,
                    "geometric_center_robot_base_m": center,
                    "root_local_geometric_offset_m": geometric_offset,
                    "pose_manifest": link,
                    "gate_receipt": {
                        "path": gate_path,
                        "sha256": entry["sha256"],
                        "bytes": entry["bytes"],
                    },
                    "capture_pointer": f"/evaluation/capture_evidence/{capture_index}",
                    "camera_names": sorted(capture["cameras"]),
                    "geometry_producer_source_id": "forecast-fixture-adapter",
                    "semantic_basis": "pinned fixture adapter uses world.get_pose and world.get_bbox in the same RoboLab environment-local axes (historically labelled robot_base_m); geometric center is AABB midpoint, offset is inverse settled-root rotation, and reset displacement is separate",
                })
    if not rows:
        blockers.append({"reason": "no hash-linked gate receipt supplied for pose manifests"})
    return rows, blockers


def _export_integrity(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"provided": False, "verified_entries": 0}
    entries = json.loads(path.read_text()).get("files", {})
    verified = sum(
        isinstance(entry.get("text"), str)
        and entry.get("bytes") == len(entry["text"].encode())
        and entry.get("sha256") == hashlib.sha256(entry["text"].encode()).hexdigest()
        for entry in entries.values()
    )
    return {"provided": True, "entry_count": len(entries), "verified_entries": verified}


def compile_registry(
    repo_root: Path = REPO_ROOT,
    pose_export: Path | None = None,
    gate_export: Path | None = None,
) -> dict[str, Any]:
    records = []
    unresolved = []
    for source_id, commit, relative, blob, expected, purpose in SOURCE_SPECS:
        path = repo_root / relative
        record = {
            "source_id": source_id,
            "commit": commit,
            "path": relative,
            "git_blob_sha1": blob,
            "expected_sha256": expected,
            "purpose_and_limit": purpose,
            "arena": "excluded_widowx" if "widowx" in purpose.lower() else ("droid_robolab" if "droid" in purpose.lower() or "robolab" in purpose.lower() else "related_source"),
        }
        git_bytes = _git_blob(repo_root, commit, relative)
        if git_bytes is not None:
            actual = hashlib.sha256(git_bytes).hexdigest()
            record["git_object_sha256"] = actual
            record["git_object_bytes"] = len(git_bytes)
            record["git_object_blob_match"] = (
                subprocess.check_output(
                    ["git", "-C", str(repo_root), "rev-parse", f"{commit}:{relative}"],
                    text=True,
                ).strip() == blob
            )
        elif path.is_file():
            actual = _sha256(path)
            record["local_sha256"] = actual
            record["hash_match"] = actual == expected
            if actual != expected:
                unresolved.append({**record, "reason": "local bytes do not match audited commit binding"})
        else:
            record["local_sha256"] = None
            record["hash_match"] = None
            unresolved.append({**record, "reason": "audited commit/path object unavailable locally and source is not checked out"})
        if git_bytes is not None and (actual != expected or not record["git_object_blob_match"]):
            unresolved.append({**record, "reason": "audited Git object does not match the recorded blob or SHA256 binding"})
        records.append(record)
    root_only_rows, root_blockers = _root_only_evidence(pose_export)
    measured_rows, gate_blockers = _gate_geometry(pose_export, gate_export)
    return {
        "schema_version": "sgw-01-historical-droid-layout-evidence-registry-v1",
        "source_policy": {
            "diagnostic_only": True,
            "only_hash_bound_sources": True,
            "comparable_row_rule": "requires complete explicit root position, root quaternion, and geometric offset under one source schema",
            "no_inference_from_partial_coordinates": True,
            "no_release_or_candidate_replacement": True,
        },
        "sources": records,
        "comparable_pose_rows": measured_rows,
        "root_only_pose_rows": root_only_rows,
        "structured_blockers": root_blockers + gate_blockers,
        "export_integrity": {
            "pose_export": _export_integrity(pose_export),
            "gate_export": _export_integrity(gate_export),
        },
        "raw_pvc_paths_still_needed": [
            {
                "layout_pair_id": layout_id,
                "path": path,
                "expected_sha256": expected,
                "reason": "required to inspect actual reset/root quaternion and geometric offsets; audit confirmed path/hash but did not retrieve payload",
            }
            for layout_id, path, expected in PVC_POSE_MANIFESTS
        ],
        "unresolved_bindings": unresolved,
        "excluded_arenas": ["RoboTwin", "SimplerEnv/WidowX"],
        "coverage_status": "incomplete_unresolved_historical_layout_coverage",
        "release_authorization": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose-export", type=Path)
    parser.add_argument("--gate-export", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compile_registry(pose_export=args.pose_export, gate_export=args.gate_export), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
