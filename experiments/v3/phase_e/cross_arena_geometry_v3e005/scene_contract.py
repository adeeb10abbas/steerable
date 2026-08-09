"""Seven-scene, model-blind RoboTwin layout contract for V3-E005.

The registered E005 intervention is deliberately narrower than a new task:
each RoboTwin pair keeps the exact two assets and the exact historical control
reset, while the treatment centers both movable singletons on the calibrated
robot sagittal plane.  RoboTwin native ``-x`` maps to calibrated robot ``+y``;
therefore the layout midline is native ``x == 0``.

This module has no SAPIEN or model dependency.  The candidate binds the seven
source fixtures by task, source seed, asset bytes, prompts, and historical
settled centers.  Full source quaternions are intentionally captured by the
zero-request live gate from the exact deterministic source reset; the gate
then resolves both layouts and hashes the realised poses before inference.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .runtime_contract import (
    AMENDMENT_ID,
    ARENA,
    MODEL_ID,
    QUEUE_SHA256,
    REGISTRATION_SHA256,
    SIMULATOR_COMMIT,
    E005ContractError,
    require,
)


CANDIDATE_SCHEMA = "vla-wam-shared-v3e005-seven-scene-layout-candidate-v1"
LIVE_GATE_SCHEMA = "vla-wam-shared-v3e005-seven-scene-model-blind-gate-v1"
SCENE_IDS = tuple(f"robotwin_pair_{number:02d}" for number in range(3, 10))
LEVELS = (0.0, 1.0)
RELATIONS = ("left", "right")
CAMERAS = ("head_camera", "left_camera", "right_camera")

POSITION_TOLERANCE_M = 0.001
ORIENTATION_TOLERANCE_RAD = math.radians(0.5)
LIVE_POSITION_TOLERANCE_M = 0.003
LIVE_ORIENTATION_TOLERANCE_RAD = math.radians(2.0)
POSITION_INVERSE_M = 10.0
ORIENTATION_INVERSE_RAD = 1.0


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wrap_angle(value: float) -> float:
    require(math.isfinite(float(value)), "angle must be finite")
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def normalize_quaternion(value: Sequence[float]) -> tuple[float, float, float, float]:
    require(len(value) == 4, "quaternion must be wxyz length four")
    row = tuple(float(item) for item in value)
    require(all(math.isfinite(item) for item in row), "quaternion must be finite")
    norm = math.sqrt(sum(item * item for item in row))
    require(norm > 0.0, "quaternion norm is zero")
    return tuple(item / norm for item in row)  # type: ignore[return-value]


def quaternion_multiply(
    a: Sequence[float], b: Sequence[float]
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = normalize_quaternion(a)
    bw, bx, by, bz = normalize_quaternion(b)
    return normalize_quaternion(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def quaternion_yaw(value: Sequence[float]) -> float:
    w, x, y, z = normalize_quaternion(value)
    return wrap_angle(
        math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    )


def with_world_yaw(
    value: Sequence[float], yaw_rad: float
) -> tuple[float, float, float, float]:
    """Replace world-Z yaw while retaining the source reset's roll/pitch."""

    source_yaw = quaternion_yaw(value)
    delta = wrap_angle(float(yaw_rad) - source_yaw)
    rotation = (math.cos(delta / 2.0), 0.0, 0.0, math.sin(delta / 2.0))
    result = quaternion_multiply(rotation, value)
    require(
        abs(wrap_angle(quaternion_yaw(result) - float(yaw_rad))) < 1e-9,
        "yaw replacement failed",
    )
    return result


def angular_distance(a: Sequence[float], b: Sequence[float]) -> float:
    qa = normalize_quaternion(a)
    qb = normalize_quaternion(b)
    dot = abs(sum(x * y for x, y in zip(qa, qb, strict=True)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


@dataclass(frozen=True)
class AssetContract:
    name: str
    model_id: int
    scale_xyz: tuple[float, float, float]
    model_data_sha256: str
    collision_mesh_sha256: str
    visual_mesh_sha256: str

    def to_json(self) -> dict[str, Any]:
        return {
            "asset_identity": [self.name, self.model_id],
            "scale_xyz": list(self.scale_xyz),
            "model_data": {
                "relative_path": f"assets/objects/{self.name}/model_data{self.model_id}.json",
                "sha256": self.model_data_sha256,
            },
            "collision_mesh": {
                "relative_path": f"assets/objects/{self.name}/collision/base{self.model_id}.glb",
                "sha256": self.collision_mesh_sha256,
            },
            "material": {
                "contract": "exact_visual_mesh_bytes",
                "relative_path": f"assets/objects/{self.name}/visual/base{self.model_id}.glb",
                "sha256": self.visual_mesh_sha256,
            },
        }


@dataclass(frozen=True)
class ActorPose:
    position_xyz_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    asset_identity: tuple[str, int]

    def __post_init__(self) -> None:
        require(len(self.position_xyz_m) == 3, "position must be xyz")
        require(
            all(math.isfinite(float(item)) for item in self.position_xyz_m),
            "position must be finite",
        )
        object.__setattr__(
            self,
            "position_xyz_m",
            tuple(float(item) for item in self.position_xyz_m),
        )
        object.__setattr__(
            self, "quaternion_wxyz", normalize_quaternion(self.quaternion_wxyz)
        )
        require(
            isinstance(self.asset_identity, tuple)
            and len(self.asset_identity) == 2
            and isinstance(self.asset_identity[0], str)
            and self.asset_identity[0]
            and type(self.asset_identity[1]) is int,
            "asset identity must be (name, integer id)",
        )

    @property
    def yaw_rad(self) -> float:
        return quaternion_yaw(self.quaternion_wxyz)

    def to_json(self) -> dict[str, Any]:
        return {
            "position_xyz_m": list(self.position_xyz_m),
            "quaternion_wxyz": list(self.quaternion_wxyz),
            "yaw_rad": self.yaw_rad,
            "asset_identity": list(self.asset_identity),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "ActorPose":
        return cls(
            tuple(value["position_xyz_m"]),
            tuple(value["quaternion_wxyz"]),
            (str(value["asset_identity"][0]), int(value["asset_identity"][1])),
        )


def _asset(
    name: str,
    model_id: int,
    scale: tuple[float, float, float],
    model_data: str,
    collision: str,
    visual: str,
) -> AssetContract:
    return AssetContract(name, model_id, scale, model_data, collision, visual)


ASSETS: dict[tuple[str, int], AssetContract] = {
    ("086_woodenblock", 1): _asset("086_woodenblock", 1, (1.0, 1.0, 1.0), "87a426ff27339494ab512c58b797559e2dfb594cef1db4b8ed3452271aca1c78", "249f2e7f85f4ce8cff638537d1decbd663cc3f00c8d52f02ac1e08c8d1074d34", "7dde1e938f953a5297e38de58f81a3cf96b9462857a119d9e303971ba72cc9cc"),
    ("081_playingcards", 1): _asset("081_playingcards", 1, (0.05, 0.05, 0.05), "1fad364b5a4af21b83e4a21b9d11c23d4965840a2835cbfaeac657c21e2cd114", "e468199bc1962acb47f142b4d9f7d00cd30224d67398e0ecbe8c96439525c433", "7e631be493bfd74bf9287a79e846ac1ea5ca6c2cc2a88c5af24e2a4489de3287"),
    ("047_mouse", 0): _asset("047_mouse", 0, (0.5, 0.5, 0.5), "ec9826adaf1ab0a950607cca52ae1f84be4f8713b58697c2b5992e58756853dd", "bb489a586a323b9cda0303b00ae381235a7652d7c2936c314631927b7bf0edea", "29ea5540e000591c21a7bc88a4b3a84f8c5482f9c4f8b560758004afc566209b"),
    ("048_stapler", 2): _asset("048_stapler", 2, (0.05, 0.05, 0.05), "6628636dc87fa391649c7930a9e74fa233c7ec5eaabea5346078db28c2fef0f0", "6f67190620f5c3695ddfe4ffcb016a09fd94966182809f9a580127702dc1c8a6", "b20030751a708ba1ca5773bc3c7d1e674678b34e52a83c5147e6ea281a9239b8"),
    ("081_playingcards", 0): _asset("081_playingcards", 0, (0.05, 0.05, 0.05), "ae60e2c2377edb3ab94ded0ef9f8254f87f6f2eb92a81b77d186d83faee6905d", "a998fa3cb2bc321070132e88de66ad1da112ff04b236726011eb26921446754f", "36b6ebe2bf80c97a34d14e2707014e1694f20fb94229c0e093a36368bbfc72f8"),
    ("073_rubikscube", 1): _asset("073_rubikscube", 1, (0.03, 0.03, 0.03), "d0dcbda4beaa8079cd79d21c81433c6a2b38204f5c07f721b6d292f826df0ff1", "8fc403e4bd15e65481273bee0e857200d4033431cca671f39ed4bb30a7c54622", "384ae3085d937c5b0de1a324d8fe56d59eaab7e4ea3a197f68d79c82a109c59d"),
    ("113_coffee-box", 3): _asset("113_coffee-box", 3, (0.04, 0.04, 0.04), "b5db725a6d42f1f453fc40a7c897f53cf4e059f0111e19a5caade8adbe79398b", "ae07c089afda8fd8f46d79d66da989fb06094c9a9fe1bc4fba95486259eb9be1", "e3463652dc16213070e0c3af6d80b086517f023aaa8b2651248321d82ac598cc"),
    ("075_bread", 6): _asset("075_bread", 6, (0.035, 0.035, 0.035), "9672242c4abd58cad8e3a5802fb15079f145e303ed78048ef541d4011e78acc9", "e7fe1a2bb682baa88297fa79c24e73f3c3706a3142d87f20a602179dfa2d9f70", "8c064f007414a021829139ac7cc6eaa1fffdc52483d0b94642e5fa9279d1cfa1"),
    ("048_stapler", 0): _asset("048_stapler", 0, (0.05, 0.05, 0.05), "57600c0933f3375aefd0e403585d04dfd3c59041f4371d88d30212051d67a4db", "e7a723fdd2016d804b8b5b2ed94953036f2280ca6d29ca03b657429dbecb3102", "92f7f99b39160e18a6037263283f5f4d30e815822257e0d7e5a397f9afd4ce7c"),
    ("081_playingcards", 2): _asset("081_playingcards", 2, (0.05, 0.05, 0.05), "471d179e93258b55e618b869eb62183c9c5565cb54e28229c54c20b55ad8d716", "c411cf82b351df01c9f6dac8d6ad7d57ba0f7b5ecf53c533a3965dffb5a0cd83", "e8e9491f5888cd079a9cd8c6be77d031e72624f965181d0daf188f0293dbd0bf"),
    ("077_phone", 4): _asset("077_phone", 4, (0.078, 0.078, 0.078), "db17091e9b80082b5124c94a3b0070d1b056417a2849cf8bd659251f4c59e59d", "ba289fbf275bccf43b477cc8d504a5baf17802d3131ee3a3b99dc971f7c2e29b", "6a25a9ae794e739a20a3a830603c0c37bf323ab1c387fe091755bd40e1023759"),
    ("073_rubikscube", 0): _asset("073_rubikscube", 0, (0.04, 0.04, 0.04), "3566a537710e2bc7362bb9464147e326e21d00644516ca59e23f922303a82c17", "e8cae86b5036a7af67a44990c9b651d736f99cbf329e2c65a90626fe96829308", "bae57f36990f3287887c7889f01b680ed579250e6915147ce9bb885e1db1c98f"),
    ("086_woodenblock", 0): _asset("086_woodenblock", 0, (1.0, 1.0, 1.0), "789a57fb2d1e2c1a3ed026ebcdf3f1c84905e86d5a7f23ef2392cdfc40f35c2f", "bf937aa1461c72b3b19fecddacf6aa9c273600085b8b557a5e2c528c7e51cd6b", "32df275334333911566565bb0d24074c640b71260f7df7ef23a5a77261009c09"),
}


def _scene(
    number: int,
    task: str,
    target: tuple[str, int],
    reference: tuple[str, int],
    target_xyz: tuple[float, float, float],
    reference_xyz: tuple[float, float, float],
    left_prompt: str,
    right_prompt: str,
    source_result_sha256: str,
) -> dict[str, Any]:
    return {
        "scene_id": f"robotwin_pair_{number:02d}",
        "pair_number": number,
        "anchor_task": task,
        "source_fixture_environment_seed": 4_300_000 + number,
        "target_asset": target,
        "reference_asset": reference,
        "historical_settled_centers": {
            "target": target_xyz,
            "reference": reference_xyz,
        },
        "prompts": {"left": left_prompt, "right": right_prompt},
        "source_result_sha256": source_result_sha256,
    }


SCENES: dict[str, dict[str, Any]] = {
    row["scene_id"]: row
    for row in (
        _scene(3, "place_a2b_right", ("086_woodenblock", 1), ("081_playingcards", 1), (-0.047076620161533356, -0.030880313366651535, 0.7405446767807007), (-0.21130692958831787, -0.1640346497297287, 0.7408550977706909), "Put the small woodenblock to the left of the red playingcards box.", "Put the small woodenblock to the right of the red playingcards box.", "ccb2874ab75fc76a6d3ea2516fc152985f4805d89976fb5a1651134dbf28159b"),
        _scene(4, "place_a2b_left", ("047_mouse", 0), ("048_stapler", 2), (-0.2067875862121582, -0.15669114887714386, 0.7406745553016663), (-0.049057185649871826, -0.03065800853073597, 0.7408447265625), "Put the plastic mouse to the left of the blue stapler.", "Put the plastic mouse to the right of the blue stapler.", "79b813962b2f695d0aeea83e015c8f8bfe420b912564bad4acfaa12194075d65"),
        _scene(5, "place_a2b_right", ("081_playingcards", 0), ("073_rubikscube", 1), (0.028682053089141846, -0.0780719444155693, 0.7408455014228821), (0.051699742674827576, -0.19804884493350983, 0.740509569644928), "Put the box of playingcards to the left of the rubikscube.", "Put the box of playingcards to the right of the rubikscube.", "31946801cb51360ab85a9f3ae79e0b4958bda224e1721c7d0cae8438a3b643a0"),
        _scene(6, "place_a2b_left", ("113_coffee-box", 3), ("081_playingcards", 1), (-0.0416286401450634, -0.13113078474998474, 0.7406562566757202), (-0.044834256172180176, -0.02297043986618519, 0.7408550381660461), "Put the coffee box to the left of the red playingcards box.", "Put the coffee box to the right of the red playingcards box.", "0a31370349dc963032ec93c5fa6f2f26828dcf409ace18a93754996129b0c35e"),
        _scene(7, "place_a2b_right", ("075_bread", 6), ("048_stapler", 0), (0.07497777044773102, -0.19529388844966888, 0.7404320240020752), (0.0632561594247818, -0.00024076629779301584, 0.7403585314750671), "Put the golden bread to the left of the blue stapler.", "Put the golden bread to the right of the blue stapler.", "ebefd347a33e09b7a8842a845129e9a20e1b3020e54b274d9f33181e58553d32"),
        _scene(8, "place_a2b_left", ("081_playingcards", 2), ("077_phone", 4), (0.003525292966514826, -0.08921598643064499, 0.7408636808395386), (0.21135477721691132, -0.19114452600479126, 0.7413461208343506), "Put the box with cards inside to the left of the black phone.", "Put the box with cards inside to the right of the black phone.", "5a1893d099465c4d109ffc206c058cb19b3f3fc5a60a773801dd621b8517d341"),
        _scene(9, "place_a2b_right", ("073_rubikscube", 0), ("086_woodenblock", 0), (0.07919169962406158, -0.18961840867996216, 0.74068284034729), (-0.09130284935235977, -0.086280956864357, 0.7398295402526855), "Put the rubikscube to the left of the brown woodenblock.", "Put the rubikscube to the right of the brown woodenblock.", "8112aecb72cd7059ea2d445cd79169120e35136143614cfa6550d72e8c2a826e"),
    )
}


def candidate_payload() -> dict[str, Any]:
    scene_rows: dict[str, Any] = {}
    for scene_id, source in sorted(SCENES.items()):
        assets = {
            role: ASSETS[source[f"{role}_asset"]].to_json()
            for role in ("target", "reference")
        }
        scene_rows[scene_id] = {
            "pair_number": source["pair_number"],
            "anchor_task": source["anchor_task"],
            "source_fixture_environment_seed": source["source_fixture_environment_seed"],
            "prompts": source["prompts"],
            "inventory": {
                "target_count": 1,
                "reference_count": 1,
                "no_duplicated_reference": True,
                "mirrored_clutter_pairs": [],
                "source_task_has_no_other_movable_actors": True,
            },
            "assets": assets,
            "control_source": {
                "contract": "exact_deterministic_source_reset",
                "historical_settled_centers_native_xyz_m": {
                    role: list(source["historical_settled_centers"][role])
                    for role in ("target", "reference")
                },
                "quaternion_contract": "capture_full_wxyz_from_exact_source_reset_before_request_zero",
                "historical_result_evidence": {
                    "manifest": "artifacts/vla_wam_shared_v3/results/lingbot_va_robotwin_phase_a_evidence_hash_manifest.json",
                    "sha256": source["source_result_sha256"],
                },
            },
            "layouts": {
                "0.00": {
                    "construction": "reuse_live_hash_bound_full_source_poses",
                    "source_pose_required": True,
                },
                "1.00": {
                    "construction": "center_target_and_reference_on_native_x_zero_and_set_world_yaw_zero_preserving_source_roll_pitch",
                    "native_x_m": 0.0,
                    "source_y_z_held_fixed": True,
                    "world_yaw_rad": 0.0,
                    "source_pose_required": True,
                },
            },
        }
    return {
        "schema_version": CANDIDATE_SCHEMA,
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "arena": ARENA,
        "status": "model_blind_candidate_not_released_for_inference",
        "registration_sha256": REGISTRATION_SHA256,
        "queue_sha256": QUEUE_SHA256,
        "simulator_repository_commit": SIMULATOR_COMMIT,
        "model_request_count": 0,
        "model_action_request_count": 0,
        "behavioral_episode_count": 0,
        "coordinate_contract": {
            "calibrated_robot_left_axis": "+robot_y",
            "robot_y_from_robotwin_native": "robot_y=-native_x",
            "sagittal_plane": "robot_y=0 equivalently native_x=0",
            "behavioral_outcome_coordinate": "frozen_RoboTwin_native_axis_and_region",
            "no_DROID_axis_or_predicate_imported": True,
        },
        "levels": [0.0, 1.0],
        "scenes": scene_rows,
        "strict_tolerances": {
            "position_residual_m_strict_upper": POSITION_TOLERANCE_M,
            "orientation_residual_rad_strict_upper": ORIENTATION_TOLERANCE_RAD,
            "midline_residual_m_strict_upper": POSITION_TOLERANCE_M,
            "live_source_position_match_m_upper_inclusive": LIVE_POSITION_TOLERANCE_M,
            "live_pose_match_m_upper_inclusive": LIVE_POSITION_TOLERANCE_M,
            "live_pose_match_rad_upper_inclusive": LIVE_ORIENTATION_TOLERANCE_RAD,
            "occlusion_false_for_all_cameras": list(CAMERAS),
        },
        "live_realisation_contract": {
            "source_reset_before_any_model_request": True,
            "capture_and_hash_full_source_poses": True,
            "resolve_both_layouts_from_captured_source_pose": True,
            "left_right_non_language_reset_fingerprints_identical": True,
            "repeat_resets_minimum": 2,
            "settled_stability_window_required": True,
            "arm_reset_pose_required_and_logged": True,
            "asset_scale_material_bytes_required_and_logged": True,
            "all_registered_camera_views_and_occlusion_checks_required": True,
        },
        "scope_caveat": "This is a symmetric object layout about the calibrated robot midline; the dual-arm robot, joint reset, wrist cameras, and embodiment are not bilaterally symmetric.",
    }


def candidate_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(candidate_payload())).hexdigest()


def load_candidate(
    path: Path,
    expected_sha256: str,
    registration_sha256: str = REGISTRATION_SHA256,
    queue_sha256: str = QUEUE_SHA256,
) -> dict[str, Any]:
    payload = Path(path).read_bytes()
    require(
        hashlib.sha256(payload).hexdigest() == expected_sha256,
        "E005 scene candidate SHA-256 mismatch",
    )
    try:
        value = json.loads(
            payload,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise E005ContractError(f"invalid E005 scene candidate JSON: {error}") from error
    require(value == candidate_payload(), "E005 scene candidate semantic reconstruction changed")
    require(value["registration_sha256"] == registration_sha256, "candidate registration SHA-256 mismatch")
    require(value["queue_sha256"] == queue_sha256, "candidate queue SHA-256 mismatch")
    return value


def _level_key(level: float) -> str:
    numeric = float(level)
    require(numeric in LEVELS, "E005 authorizes only s=0 and s=1")
    return f"{numeric:.2f}"


def layout_for(
    candidate: Mapping[str, Any],
    scene_id: str,
    level: float,
    gate_scene: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a prospective recipe or a gate-resolved concrete layout.

    Before a live gate exists, the return value is the candidate's immutable
    construction recipe.  Once ``gate_scene`` is supplied, the result contains
    concrete target/reference poses suitable for a task subclass.
    """

    require(scene_id in SCENE_IDS, f"unknown E005 scene: {scene_id}")
    key = _level_key(level)
    scene = candidate.get("scenes", {}).get(scene_id)
    require(isinstance(scene, Mapping), f"candidate lacks {scene_id}")
    if gate_scene is None:
        return deepcopy(dict(scene["layouts"][key]))
    require(gate_scene.get("scene_id") == scene_id, "gate scene id mismatch")
    layouts = gate_scene.get("resolved_layouts")
    require(isinstance(layouts, Mapping), f"gate scene {scene_id} lacks resolved layouts")
    layout = layouts.get(key)
    require(isinstance(layout, Mapping), f"gate scene {scene_id} lacks layout {key}")
    require(set(layout) == {"target", "reference"}, "resolved layout inventory drift")
    return deepcopy(dict(layout))


def actor_pose(layout: Mapping[str, Any], role: str) -> ActorPose:
    require(role in {"target", "reference"}, f"unknown actor role: {role}")
    row = layout.get(role)
    require(isinstance(row, Mapping), f"resolved layout lacks {role}")
    return ActorPose.from_json(row)


def symmetric_layout(source_layout: Mapping[str, ActorPose]) -> dict[str, ActorPose]:
    require(set(source_layout) == {"target", "reference"}, "source layout inventory drift")
    output: dict[str, ActorPose] = {}
    for role, pose in source_layout.items():
        _, depth_y, height_z = pose.position_xyz_m
        output[role] = ActorPose(
            (0.0, depth_y, height_z),
            with_world_yaw(pose.quaternion_wxyz, 0.0),
            pose.asset_identity,
        )
    return output


def residuals(layout: Mapping[str, ActorPose]) -> dict[str, float]:
    require(set(layout) == {"target", "reference"}, "layout inventory drift")
    return {
        "position_residual_m": 0.0,
        "orientation_residual_rad": max(
            abs(wrap_angle(2.0 * pose.yaw_rad)) for pose in layout.values()
        ),
        "midline_residual_m": max(
            abs(pose.position_xyz_m[0]) for pose in layout.values()
        ),
    }


def asymmetry_A(layout: Mapping[str, ActorPose]) -> float:
    require(set(layout) == {"target", "reference"}, "layout inventory drift")
    terms: list[float] = []
    for pose in layout.values():
        terms.append(POSITION_INVERSE_M * 2.0 * pose.position_xyz_m[0])
        terms.append(ORIENTATION_INVERSE_RAD * wrap_angle(2.0 * pose.yaw_rad))
    return math.sqrt(sum(value * value for value in terms))


def _asset_snapshot(candidate: Mapping[str, Any], scene_id: str, role: str) -> Mapping[str, Any]:
    row = candidate["scenes"][scene_id]["assets"][role]
    require(isinstance(row, Mapping), f"candidate asset contract missing for {scene_id}/{role}")
    return row


def validate_live_snapshot(
    candidate: Mapping[str, Any],
    scene_id: str,
    level: float,
    snapshot: Mapping[str, Any],
    gate_scene: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed on a model-blind reset snapshot.

    ``gate_scene`` is required for the concrete source quaternion binding.  A
    runtime may pass the enclosing scene row from the live gate report.
    """

    require(gate_scene is not None, "live snapshot validation requires a hash-bound gate scene")
    expected_json = layout_for(candidate, scene_id, level, gate_scene)
    expected = {role: actor_pose(expected_json, role) for role in ("target", "reference")}
    observed_rows = snapshot.get("realised_object_poses")
    require(isinstance(observed_rows, Mapping), "snapshot lacks realised_object_poses")
    observed = {
        role: ActorPose.from_json(observed_rows[role]) for role in ("target", "reference")
    }
    for role in ("target", "reference"):
        require(observed[role].asset_identity == expected[role].asset_identity, f"{scene_id}/{role}: asset drift")
        translation = max(
            abs(a - b)
            for a, b in zip(
                observed[role].position_xyz_m,
                expected[role].position_xyz_m,
                strict=True,
            )
        )
        require(translation <= LIVE_POSITION_TOLERANCE_M, f"{scene_id}/{role}: live position drift")
        require(
            angular_distance(observed[role].quaternion_wxyz, expected[role].quaternion_wxyz)
            <= LIVE_ORIENTATION_TOLERANCE_RAD,
            f"{scene_id}/{role}: live orientation drift",
        )
        observed_asset = snapshot.get("asset_contract", {}).get(role)
        require(observed_asset == _asset_snapshot(candidate, scene_id, role), f"{scene_id}/{role}: scale/material bytes drift")
    occlusion = snapshot.get("occlusion_check")
    require(isinstance(occlusion, Mapping), "snapshot lacks camera occlusion checks")
    require(set(occlusion) == set(CAMERAS), "snapshot camera occlusion inventory drift")
    require(not any(bool(value) for value in occlusion.values()), f"{scene_id}: target is occluded")
    views = snapshot.get("views")
    require(isinstance(views, Mapping) and set(views) == set(CAMERAS), "snapshot camera view inventory drift")
    for camera, row in views.items():
        require(
            isinstance(row, Mapping)
            and type(row.get("target_visible_pixels")) is int
            and row["target_visible_pixels"] > 0,
            f"{scene_id}: target is not actor-segmentation-visible in {camera}",
        )
    require(
        snapshot.get("arm_reset_pose", {}).get("status") == "available",
        f"{scene_id}: arm reset pose unavailable",
    )
    observed_residuals = residuals(observed)
    if math.isclose(float(level), 1.0, abs_tol=1e-12):
        require(observed_residuals["position_residual_m"] < POSITION_TOLERANCE_M, f"{scene_id}: s1 position residual")
        require(observed_residuals["orientation_residual_rad"] < ORIENTATION_TOLERANCE_RAD, f"{scene_id}: s1 orientation residual")
        require(observed_residuals["midline_residual_m"] < POSITION_TOLERANCE_M, f"{scene_id}: s1 midline residual")
    return {
        "scene_id": scene_id,
        "symmetry_level_s": float(level),
        "passed": True,
        **observed_residuals,
        "asymmetry_metric_A": asymmetry_A(observed),
    }


def verify_asset_files(candidate: Mapping[str, Any], simulator_repository: Path) -> dict[str, Any]:
    root = Path(simulator_repository).resolve()
    records: dict[str, Any] = {}
    for scene_id in SCENE_IDS:
        records[scene_id] = {}
        for role in ("target", "reference"):
            contract = _asset_snapshot(candidate, scene_id, role)
            checked: dict[str, Any] = {}
            for key in ("model_data", "collision_mesh", "material"):
                row = contract[key]
                path = root / row["relative_path"]
                require(path.is_file(), f"missing {scene_id}/{role}/{key}: {path}")
                digest = sha256_file(path)
                require(digest == row["sha256"], f"{scene_id}/{role}/{key} SHA-256 drift")
                checked[key] = {
                    "relative_path": row["relative_path"],
                    "sha256": digest,
                    "bytes": path.stat().st_size,
                }
            records[scene_id][role] = checked
    return records
