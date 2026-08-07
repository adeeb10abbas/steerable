#!/usr/bin/env python3
"""Build source-bounded Tier-C checkpoint provenance disclosures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/checkpoint_provenance"
SCHEMA_VERSION = "vla-wam-shared-v3-checkpoint-provenance-v1"
TABLE_SCHEMA_VERSION = "vla-wam-shared-v3-checkpoint-provenance-table-v1"
MANIFEST_SCHEMA_VERSION = "vla-wam-shared-v3-checkpoint-provenance-manifest-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(*paths: str) -> list[dict[str, str]]:
    rows = []
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"path_or_url": relative, "sha256_or_status": sha256(path)})
    return rows


def training(domain: str) -> dict[str, Any]:
    return {
        "disclosure_status": "partially_disclosed",
        "datasets": [{
            "dataset_id": domain,
            "revision": "not_disclosed",
            "split": "not_disclosed",
            "episode_membership": (
                "The released checkpoint/configuration names this target domain, but the "
                "committed study evidence does not disclose the exact training episode multiset."
            ),
            "episode_manifest_sha256": "not_disclosed",
            "source_boundary": "target-domain label; not a training-episode audit",
        }],
        "sampling_or_duplication_policy": "not_disclosed by the checkpoint release or committed study evidence",
        "episode_count": "not_disclosed",
        "trajectory_count": "not_disclosed",
    }


def caption() -> dict[str, str]:
    return {
        "disclosure_status": "not_auditable",
        "left_right_tokens": "unknown",
        "exact_probe_sentences": "unknown",
        "synthetic_or_recaptioned_language": "unknown",
        "audit_method": (
            "No training-caption corpus or episode-to-caption manifest is committed. "
            "Checkpoint names and this study's inference prompts are not treated as training-caption evidence."
        ),
        "matched_examples_sha256": "unknown",
    }


def preprocessing(visual: str, action: str, *, cameras: str, frames: str = "current observation") -> dict[str, str]:
    return {
        "scope": "v3 inference-time interface only",
        "training_preprocessing": "not_disclosed",
        "visual": visual,
        "language": "One frozen static episode prompt; no prompt switching, coaching, or progress-conditioned language.",
        "action": action,
        "frame_sampling": frames,
        "camera_selection": cameras,
    }


def base_record(
    *, model_id: str, family: str, arena: str, artifact: str, revision: str,
    content_hash: str, content_kind: str, runtime_hash: str, runtime_kind: str,
    domain: str, prep: dict[str, str], interface: dict[str, Any], runtime_components: dict[str, str],
    unknowns: list[str], evidence_paths: tuple[str, ...], status: str = "reported_v3_checkpoint_identity",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "model_family": family,
        "arena": arena,
        "study_status": status,
        "checkpoint_identity": {
            "artifact_id": artifact,
            "revision": revision,
            "content_sha256": content_hash,
            "content_hash_kind": content_kind,
            "runtime_identity_sha256": runtime_hash,
            "runtime_hash_kind": runtime_kind,
        },
        "training_episode_multiset": training(domain),
        "preprocessing": prep,
        "caption_exposure": caption(),
        "action_future_interface": interface,
        "runtime_components": runtime_components,
        "known_unknowns": unknowns,
        "source_boundary": (
            "Training-data, caption, and training-preprocessing claims stop at public/committed disclosure. "
            "Observed inference adapters do not establish how the checkpoint was trained."
        ),
        "evidence": evidence(*evidence_paths),
    }


def records() -> list[dict[str, Any]]:
    droid = "DROID"
    twin = "RoboTwin"
    records: list[dict[str, Any]] = []

    records.append(base_record(
        model_id="pi0_fast_droid_vla", family="VLA", arena="droid_robolab",
        artifact="gs://openpi-assets-simeval/pi0_fast_droid_jointpos", revision="not_disclosed",
        content_hash="47b38eb2f17be802c126ef0a7e93b16693823ee2df62b8007f51bb0514baf5c5",
        content_kind="sha256 of the hash-bearing readiness registry that enumerates all 19 checkpoint files (10,844,314,410 bytes)",
        runtime_hash="unknown", runtime_kind="unrecoverable historical runtime identity",
        domain=droid,
        prep=preprocessing(
            "Exact historical inference preprocessing is unavailable with the missing source revisions.",
            "Preserved interface evidence reports action-only 10x8 output; exact historical transforms are unavailable.",
            cameras="not_disclosed for the exact historical runtime",
        ),
        interface={"actions": "action-only, preserved 10x8 interface", "future": "no decoded future", "execution": "historical source-blocked; not rerunnable under this identity"},
        runtime_components={
            "required_openpi_revision": "9e46d3aea26417bfb564227734b95d010aa827e5",
            "required_robolab_revision": "11142d4319e44401e0464866bb5fedf7ec8a8927",
            "availability": "both exact source revisions are absent from every committed bundle and ali-owned checkout",
        },
        unknowns=[
            "Exact training episode multiset, sampling weights, and training preprocessing are not disclosed.",
            "Training caption exposure is not auditable.",
            "The exact historical runtime cannot be reconstructed; newer code would be a different cohort.",
        ],
        evidence_paths=(
            "artifacts/vla_wam_shared_v2/pilot/expansion/pi0_fast_wording_readiness.json",
            "artifacts/vla_wam_shared_v3/droid_direct_registry.json",
        ),
        status="frozen_historical_identity_source_blocked",
    ))

    records.append(base_record(
        model_id="pi0_fast_old_name_config_v3a002", family="VLA", arena="droid_robolab",
        artifact="gs://openpi-assets-simeval/pi0_fast_droid_jointpos", revision="not_disclosed",
        content_hash="47b38eb2f17be802c126ef0a7e93b16693823ee2df62b8007f51bb0514baf5c5",
        content_kind="sha256 of the hash-bearing readiness registry that enumerates all 19 checkpoint files (10,844,314,410 bytes)",
        runtime_hash="unknown", runtime_kind="exact components disclosed; no canonical full-runtime digest committed",
        domain=droid,
        prep=preprocessing(
            "Current public OpenPI DroidInputs(ModelType.PI0_FAST) path; exact numeric training preprocessing remains undisclosed.",
            "DroidOutputs with AbsoluteActions on the first seven dimensions; 10x8 action chunk.",
            cameras="current public pi0_fast_droid_jointpos configuration",
        ),
        interface={"actions": "10x8 action-only output", "future": "no decoded future", "execution": "completed V3-A002 compatibility cohort; not historical recovery"},
        runtime_components={
            "openpi_commit": "235044ed8a1502c0a18338eedc5d7adfe705af05",
            "openpi_tree": "03a4387bedbc0fa1467c367c60fc24e28b61ec6c",
            "robolab_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17",
            "config": "pi0_fast_droid_jointpos",
            "config_source_sha256": "96ddf85d2e9668d9e3d0d4d53f07de7bc0174918fab708fbb17b3bf547b1a54b",
            "uv_lock_sha256": "5e3a275dd30bcf6c1c36a8664307ed70801f17436721de5651870aa38c0b23ef",
        },
        unknowns=[
            "The public compatibility configuration is not proven to be the source revision used to train the checkpoint.",
            "Exact training episode multiset, sampling weights, training preprocessing, and caption exposure are not disclosed.",
            "No canonical full-runtime identity digest was committed; component revisions are independently hash-bound.",
        ],
        evidence_paths=(
            "artifacts/vla_wam_shared_v3/post_result_pi0_fast_old_name_config_amendment.json",
            "artifacts/vla_wam_shared_v3/results/pi0_fast_old_name_config_v3a002_release_gate.json",
            "artifacts/vla_wam_shared_v2/pilot/expansion/pi0_fast_wording_readiness.json",
        ),
        status="completed_distinct_compatibility_cohort",
    ))

    records.append(base_record(
        model_id="pi05_current_stack_droid", family="VLA", arena="droid_robolab",
        artifact="pi05_droid_jointpos_polaris",
        revision="v2a010-manifest-f5a56d9565f9381ccdeeaa165b0495dab6d17a81836cc7b01c5fbc6ab89e74ca",
        content_hash="b193b28b05f9755e24d44a6f5cf3185ca23c2ad3da6c5913370379c82570fbf6",
        content_kind="canonical digest of the complete checkpoint payload file manifest",
        runtime_hash="e73fe7a0cc22db09fa8fdc0babf80dd8ad3280d0502285c6ad1c4d822c7fa532",
        runtime_kind="Phase-A semantic runtime identity",
        domain=droid,
        prep=preprocessing(
            "OpenPI DROID current-stack image pipeline under the registered WRIST_LEFT_RIGHT_HEAD preset.",
            "Joint-position 15x8 action chunk; 15-action open-loop horizon.",
            cameras="WRIST_LEFT_RIGHT_HEAD",
        ),
        interface={"actions": "15x8 joint-position actions", "future": "no decoded future", "execution": "open-loop horizon 15"},
        runtime_components={"openpi_commit": "c23745b5ad24e98f66967ea795a07b2588ed6c79", "robolab_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17", "phase_b_v3b002_runtime_identity_sha256": "6e5e15f3910b0947e8bc4d7b12210f418c7956db8cdf7a28e57d04ca3db87626"},
        unknowns=["Exact training episode membership, sampling weights, training-time numeric transforms, and caption exposure are not disclosed."],
        evidence_paths=(
            "artifacts/vla_wam_shared_v3/results/pi05_current_stack_droid_phase_a_summary.json",
            "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/gates/runtime_identity.json",
            "artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_v2a010_provenance.json",
        ),
    ))

    records.append(base_record(
        model_id="groot_n17_droid_vla", family="VLA", arena="droid_robolab",
        artifact="nvidia/GR00T-N1.7-DROID", revision="05e7cc97e40dbd33b0890c35cc0214fcb0547ab5",
        content_hash="35c6c880d17913458cd1ce97d6590cddf0af2e09399d0cc8ddc371e0c2f3c03f",
        content_kind="canonical digest of the complete required checkpoint file-hash contract",
        runtime_hash="1c9515daaae3b7298310694bd5b9eb0ecdbffb5c71df747f5e1cb0d0e711be64",
        runtime_kind="Phase-A semantic runtime identity",
        domain=droid,
        prep=preprocessing(
            "Two uint8 RGB streams shaped [1,1,180,320,3] plus EEF-9D, seven joint positions, and gripper state.",
            "Checkpoint may return 40x8; the registered execution horizon is eight actions.",
            cameras="video.exterior_image_1_left and video.wrist_image_left",
        ),
        interface={"actions": "up to 40x8 returned; first 8 executed per request", "future": "no decoded future", "embodiment": "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"},
        runtime_components={"isaac_groot_commit": "b9955401d50c92a29258732e3ad6ccd579f1bdc0", "robolab_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17", "backbone": "nvidia/Cosmos-Reason2-2B@9ce19a195e423419c349abfc86fd07178b230561"},
        unknowns=["Exact training episode membership, sampling weights, training preprocessing, and caption exposure are not disclosed."],
        evidence_paths=(
            "artifacts/vla_wam_shared_v3/results/groot_n17_droid_phase_a_summary.json",
            "artifacts/vla_wam_shared_v2/pilot/expansion/groot_n17_droid_readiness.json",
            "experiments/v3/groot_droid/adapter.py",
        ),
    ))

    for model_id, artifact, revision, content_hash, runtime_hash, subsequent in (
        ("cosmos3_edge_policy_droid", "nvidia/Cosmos3-Edge-Policy-DROID", "3ea407af3e156c0af3b4bb6edd85842cc9a58777", "b58d38088b3baad884a44ff9587ba10584a573f15e2cf7b08b836336cb53e48e", "e92f68c02345042190a415a67e3eafbb12b35fded6d59d77074c74cb28ef1940", {}),
        ("cosmos3_nano_policy_droid", "nvidia/Cosmos3-Nano-Policy-DROID", "6706d7680581c255ff61e0f3bb49d90eac55c79e", "cf76fcba7008061ecf95ec08b1b21815a6ffcb2ae9878fa11fb64a5eafb2e246", "d4bc4ab7d03fd1d1041f0bcc384d34321f3bd7b16c0c4cf517b62b8a1a2160e2", {"v3b005_runtime_identity_sha256": "2aa9a2db99fb80a0bf82e9e326e17a4dbcbac8830dd800cb3f84c5bea8579287", "v3b009_runtime_identity_sha256": "a60cb8cc7139bc86eaef6a87553623b14867279604fe2a69149161ab8d44c61e"}),
    ):
        summary_name = model_id + "_phase_a_summary.json"
        registry = "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_edge_droid_v2_registry.json" if "edge" in model_id else "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_v2a011_registry.json"
        records.append(base_record(
            model_id=model_id, family="WAM", arena="droid_robolab", artifact=artifact, revision=revision,
            content_hash=content_hash, content_kind="canonical digest of the complete checkpoint payload manifest",
            runtime_hash=runtime_hash, runtime_kind="Phase-A semantic runtime identity", domain=droid,
            prep=preprocessing(
                "Official packed 540x640 uint8 RGB conditioning image plus joint and gripper state.",
                "32x8 joint-position action chunk.", cameras="official WRIST_LEFT_RIGHT_HEAD client composition"),
            interface={"actions": "32x8 joint-position actions", "future": "33-frame decoded RGB future exposed per request", "future_scoring": "retained separately; never substituted for execution"},
            runtime_components={"cosmos_framework_commit": "411d25b2e35bc441126f48c44a4b93e1c0564274", "robolab_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17", **subsequent},
            unknowns=["Exact training episode membership, sampling weights, training preprocessing, and caption exposure are not disclosed."],
            evidence_paths=(f"artifacts/vla_wam_shared_v3/results/{summary_name}", registry, "experiments/v3/cosmos_droid/contract.py"),
        ))

    records.append(base_record(
        model_id="dreamzero_droid_action_cfg", family="WAM", arena="droid_robolab",
        artifact="GEAR-Dreams/DreamZero-DROID", revision="96ad344138c66e82536422432ad742f015784942",
        content_hash="b260d781c59b84a9a325184a31dcabd88f7874c69a7b5e514b78e6fabbc3af68",
        content_kind="canonical digest of the complete checkpoint payload manifest",
        runtime_hash="a590a06f0389d4c5c75f59e83ef71ce6e5e0af9bdb55bf71e0ca775dd544b6c3",
        runtime_kind="Phase-A semantic runtime identity", domain=droid,
        prep=preprocessing(
            "Three padded uint8 180x320 views plus joint state, zero Cartesian vector, and gripper state.",
            "Raw 24x8 joint-action output; first eight actions executed per request.",
            cameras="exterior_image_0_left, exterior_image_1_left, wrist_image_left"),
        interface={"actions": "24x8 returned; first 8 executed per request", "future": "latent video with official reset decode", "guidance": "derived action CFG=2; video CFG=5"},
        runtime_components={"dreamzero_commit": "ab790c198fbce33503358efbbd4187ce9a89adf3", "robolab_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17"},
        unknowns=["Exact training episode membership, sampling weights, training preprocessing, and caption exposure are not disclosed."],
        evidence_paths=(
            "artifacts/vla_wam_shared_v3/results/dreamzero_droid_action_cfg_phase_a_summary.json",
            "artifacts/vla_wam_shared_v2/pilot/expansion/dreamzero_droid_readiness.json",
            "experiments/v3/dreamzero_droid/client.py",
        ),
    ))

    twin_common = "artifacts/vla_wam_shared_v2/pilot/execution_configs.json"
    twin_contract = "experiments/v3/robotwin_wams/contract.py"
    records.append(base_record(
        model_id="efficient_wam_rt_robotwin", family="WAM", arena="robotwin",
        artifact="jiajun0613/Efficient-WAM_RoboTwin/Efficient-WAM-RT", revision="81280a79e8ac69dd6ffb9ce8698e00d122ec07fd",
        content_hash="7f22e356fe8a0d209ecb8aed312b485e2a7aa6307a5f82edd0bf162c3c530d82",
        content_kind="sha256 of the hash-bearing checkpoint manifest artifact",
        runtime_hash="946d1ec792ed0875fd9aab2f3feb369ccc0a719743c5aa50b02b62682ca43a76",
        runtime_kind="runtime identity payload sha256", domain=twin,
        prep=preprocessing("Native Efficient-WAM-RT RoboTwin runner; exact training transforms are not disclosed.", "16-action chunks over ten inference steps.", cameras="native checkpoint runner camera contract", frames="native runner sequence contract"),
        interface={"actions": "16-action chunks", "future": "decoded coarse future video, maximum one retained chunk", "execution": "10 inference steps"},
        runtime_components={"model_source_commit": "b0b6cfabcbd68d18888866e958c677ce640f0412", "robotwin_commit": "0bd8e76fde3afcffa4b30a3e3e8f92a206aa66cc"},
        unknowns=["Exact training episode membership, sampling weights, numeric training preprocessing, and caption exposure are not disclosed."],
        evidence_paths=("artifacts/vla_wam_shared_v3/results/efficient_wam_rt_robotwin_phase_a_summary.json", twin_contract, twin_common),
    ))

    records.append(base_record(
        model_id="fastwam_robotwin", family="WAM", arena="robotwin",
        artifact="yuanty/fastwam/robotwin_uncond_3cam_384.pt", revision="139eebb6d90cdd9bdbbe465f72c6edc9ad5a518a",
        content_hash="1987af9cdeadad49cd56e5500416f7f8c0ed89b1273737679f78a26d0978839a",
        content_kind="sha256 of the hash-bearing checkpoint manifest artifact",
        runtime_hash="ecce9f86a54c4e1aa07031b9dd1ec42da965e4b34bb5c56090d551c1bffe52c0",
        runtime_kind="runtime identity payload sha256", domain=twin,
        prep=preprocessing("Native FastWAM three-camera RoboTwin runner; exact numeric training transforms are not disclosed.", "32-action prediction horizon with replan interval 24.", cameras="head, left wrist, right wrist", frames="native runner sequence contract"),
        interface={"actions": "32-action horizon; replan at 24", "future": "action-only at test time", "guidance": "CFG=2"},
        runtime_components={"model_source_commit": "068d3fd70c89df3726c09893f47b75a624b20c02", "robotwin_commit": "068d3fd70c89df3726c09893f47b75a624b20c02"},
        unknowns=["Exact training episode membership, sampling weights, numeric training preprocessing, and caption exposure are not disclosed."],
        evidence_paths=("artifacts/vla_wam_shared_v3/results/fastwam_robotwin_phase_a_summary.json", twin_contract, twin_common),
    ))

    records.append(base_record(
        model_id="lingbot_va_robotwin", family="WAM", arena="robotwin",
        artifact="lerobot/lingbot_va_robotwin + robbyant/lingbot-va-posttrain-robotwin",
        revision="lerobot/lingbot_va_robotwin@d1e1f93a84eaf9bca9880856fda800cc98cc8eaa;robbyant/lingbot-va-posttrain-robotwin@8c9dea8abbc5c91cc9e18bc3264b8915083bbe70",
        content_hash="91d32f57b7edbb9b624ef5e64e0440177c529a3bf099fc1aae5c51d1ac847c18",
        content_kind="sha256 of the hash-bearing composite checkpoint manifest artifact",
        runtime_hash="f2af400c5d6fac539564c7d2a0f3ff76479f98120896e333bc50cf3615e41e89",
        runtime_kind="runtime identity payload sha256", domain=twin,
        prep=preprocessing("Native LingBot-VA RoboTwin runner; exact numeric training transforms are not disclosed.", "Exposed action trajectory from the native runner.", cameras="native checkpoint runner camera contract", frames="native runner sequence contract"),
        interface={"actions": "exposed native action trajectory", "future": "latent-only tensor [1,48,2,24,20], not decoded", "guidance": "guidance scale 5; action guidance scale 1"},
        runtime_components={"model_source_commit": "d42efbc04e502057dab4b18bb14770cc48e85131", "robotwin_commit": "0aeea2d669c0f8516f4d5785f0aa33ba812c14b4"},
        unknowns=["Exact training episode membership, sampling weights, numeric training preprocessing, and caption exposure are not disclosed.", "The exposed future is latent-only; no behavioral video is inferred from it."],
        evidence_paths=("artifacts/vla_wam_shared_v3/results/lingbot_va_robotwin_phase_a_summary.json", twin_contract, twin_common),
    ))
    return records


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = records()
    expected_files = {f"{row['model_id']}.json" for row in rows}
    for old in OUT.glob("*.json"):
        if old.name not in expected_files | {"checkpoint_provenance_table.json", "checkpoint_provenance_manifest.json"}:
            raise RuntimeError(f"unexpected existing JSON output: {old}")
    for row in rows:
        write_json(OUT / f"{row['model_id']}.json", row)

    table_rows = []
    for row in rows:
        checkpoint = row["checkpoint_identity"]
        table_rows.append({
            "model_id": row["model_id"], "model_family": row["model_family"], "arena": row["arena"],
            "artifact_id": checkpoint["artifact_id"], "revision": checkpoint["revision"],
            "content_sha256": checkpoint["content_sha256"], "runtime_identity_sha256": checkpoint["runtime_identity_sha256"],
            "training_multiset_disclosure": row["training_episode_multiset"]["disclosure_status"],
            "caption_exposure_disclosure": row["caption_exposure"]["disclosure_status"],
            "inference_visual": row["preprocessing"]["visual"],
            "action_interface": row["action_future_interface"]["actions"],
            "future_interface": row["action_future_interface"]["future"],
            "known_unknown_count": len(row["known_unknowns"]),
        })
    table = {"schema_version": TABLE_SCHEMA_VERSION, "record_count": len(table_rows), "records": table_rows}
    table_json = OUT / "checkpoint_provenance_table.json"
    write_json(table_json, table)

    lines = [
        "# V3 checkpoint provenance", "",
        "Training episode membership, sampling, training preprocessing, and caption exposure are not reconstructed from model names or inference adapters. `unknown` and `not_disclosed` are evidence boundaries, not negative findings.", "",
        "| Checkpoint | Family / arena | Artifact revision | Inference interface | Future interface | Training multiset | Caption audit |", "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in table_rows:
        revision = row["revision"] if len(row["revision"]) < 30 else row["revision"][:12] + "…"
        lines.append(
            f"| `{row['model_id']}` | {row['model_family']} / {row['arena']} | `{revision}` | "
            f"{row['action_interface']} | {row['future_interface']} | {row['training_multiset_disclosure']} | {row['caption_exposure_disclosure']} |"
        )
    lines += ["", "The historical π0-FAST row and the V3-A002 compatibility row are distinct identities and must not be pooled. DROID/RoboLab and RoboTwin remain separate arenas.", ""]
    table_md = OUT / "checkpoint_provenance_table.md"
    table_md.write_text("\n".join(lines), encoding="utf-8")

    files = [OUT / f"{row['model_id']}.json" for row in rows] + [table_json, table_md]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "record_count": len(rows),
        "schema_path": "artifacts/vla_wam_shared_v3/prospective_tier_b/checkpoint_provenance.schema.json",
        "schema_sha256": sha256(ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/checkpoint_provenance.schema.json"),
        "builder_path": "tools/build_v3_checkpoint_provenance.py",
        "builder_sha256": sha256(Path(__file__)),
        "files": [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files],
        "boundaries": [
            "No unknown training fact is inferred from a model name, target-domain label, or inference adapter.",
            "Historical pi0_fast_droid_vla and pi0_fast_old_name_config_v3a002 are distinct cohorts.",
            "DROID/RoboLab and RoboTwin remain separate arenas.",
        ],
    }
    write_json(OUT / "checkpoint_provenance_manifest.json", manifest)
    print(f"wrote {len(rows)} provenance records to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
