"""Seal the independent bi review without modifying any original evidence."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from PIL import Image


SOURCE = "5f7a0911b7384b98cc2094b4054ad75f9c3b226b"
CAPTURES = {
    "0-height-left": "d8c961be41b65d4d6860f550466a2a0b000d44bd147f631901f6de31d98b9148",
    "1-height-right": "35327020323b406969eef7613d94946bd776001567d0dcc4e223dfc3330ad3b7",
    "2-dist-left": "beb0991e7f11bbd57f4e762627ceffd478aeeab1e91a6f4f65723e2c41d31b32",
    "3-dist-right": "7a8ace8867e0e1807518b9ca073c2004fc80fe9b9c11a1a489d57a49857ee19a",
}
CODE = {
    "prospective_family_scene.py": "15ad70278c9b3af5ecc8439e656f35617d0d013edc8fe03e8d61ba4dd5dd3345",
    "prospective_family_capture.py": "0ee4c519f64ef21b8af2e80d78ccfe029fbde02cf498069baff52ade9fa6215f",
    "prospective_family_capture_task.py": "8c0965c71a5bdedbb21beb9289057231b38dd6ab5c1a8dfc3d19fb6db827706d",
    "lat_workspace_capture.py": "c3e1b3d2a5131331e4891823bb74464cc85f27500fb46a287b8f08c82986e919",
}
HISTORICAL = {
    "sgw-bf-review-analysis/independent_scene_review.md":
        "53672965768c32429d7aa3fd34edb17cb4fe39e7fec487c9312c1d52c15a5d3f",
    "sgw-bf-review-analysis/independent_scene_review_manifest.json":
        "0ba08e859b40ee9062c482fd91336cdf11e3abc4fdfabc7536563faf87809bae",
    "prepare_sgw_scene_review.py":
        "d590f61c454769ea92dcc4cec9e95e345352691006f3c204adf5cc8d5920623f",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    root = args.root.resolve()
    analysis = root / "sgw-bi-review-analysis"
    evidence = root / "sgw-bi-review-evidence"
    output = analysis / "independent_scene_review_manifest.json"
    assert not output.exists(), "Do not replace a sealed review."

    def binding(path):
        path = path.resolve()
        assert path.is_relative_to(root) and path.is_file()
        return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest(path)}

    def read(path):
        return json.loads(path.read_text())

    def git_binding(path, expected=None):
        content = subprocess.run(
            ["git", "-C", str(args.repo), "show", f"{SOURCE}:{path}"],
            check=True, capture_output=True,
        ).stdout
        sha = hashlib.sha256(content).hexdigest()
        if expected is not None:
            assert sha == expected
        return {"path": path, "commit": SOURCE, "bytes": len(content), "sha256": sha}

    archive = binding(root / "sgw-bi-review-evidence.tar.gz")
    assert archive["bytes"] == 124817255
    assert archive["sha256"] == "0305c508b25cead38263399209269544526377c7a94c7c0c079baa7c2f3e286e"
    export_binding = binding(evidence / "export_manifest.json")
    assert export_binding["sha256"] == "cf5797a7d6cf171bdffe1d6d3bf8ae4de1f9fcd46bb44bf736704a538709025c"
    export = read(evidence / "export_manifest.json")
    assert export["source_commit"] == SOURCE and len(export["files"]) == 118
    exported = []
    for row in export["files"]:
        item = binding(evidence / row["export_path"])
        assert item["bytes"] == row["bytes"] and item["sha256"] == row["sha256"]
        exported.append({**item, "original_source_path": row["source_path"]})

    geometry = read(analysis / "geometry_and_integrity.json")
    repairs = read(analysis / "repair_checks.json")
    media = read(analysis / "media_index.json")
    details = read(analysis / "repair_images.json")
    assert geometry["study_source_commit"] == repairs["study_source_commit"] == SOURCE
    assert geometry["verified_exported_files"] == 118
    scene_rows = []
    for scene, expected in CAPTURES.items():
        g = geometry["scenes"][scene]
        r = repairs["scenes"][scene]
        cap = read(evidence / scene / "capture.json")
        assert digest(evidence / scene / "capture.json") == g["capture_sha256"] == r["capture_sha256"] == expected
        assert cap["study_source_commit"] == SOURCE
        assert cap["model_request_count"] == cap["behavioral_episode_count"] == g["physics_actions"] == 0
        assert g["decoded_video_frames"] == 121 and g["local_recording_files_verified"] == 22
        assert g["recorded_warmup_arrays_not_exported"] == 115
        assert g["neutral_signed_relation_m"] == 0 and g["neutral_within_frozen_5mm"]
        assert r["counterbalance_unchanged_from_bf"] and r["banana_within_measured_table_xy_bounds"]
        scene_rows.append({
            "scene": scene, "disposition": "ACCEPT_NATIVE_VISUAL_SETUP_ONLY",
            "confidence_setup_visibility_out_of_10": 9,
            "category_caveat": "plate/puck/pad uncertain" if "-dist-" in scene else None,
            "capture": binding(evidence / scene / "capture.json"),
            "scene_manifest": binding(evidence / scene / "scene/manifest.json"),
            "capture_receipt_content_sha256": g["capture_receipt_content_sha256"],
            "scene_manifest_content_sha256": g["manifest_content_sha256"],
            "overlay_sha256": g["overlay_sha256"], "video_sha256": g["video_sha256"],
            "counterbalance": g["counterbalance"], "render_samples_inspected": [0, 1, 10, 30, 60, 120],
            "decoded_video_frames": 121, "locally_verified_recording_files": 22,
            "recorded_warmup_arrays_not_exported": 115,
        })

    assert len(media) == 12 and len(details["details"]) == 4
    for item in media:
        assert item["render_frame"] == 120 and item["warmup_render_frame_ids"] == [0, 1, 10, 30, 60, 120]
        original = np.load(evidence / item["source_array"], allow_pickle=False)
        image = np.asarray(Image.open(analysis / item["png"]))
        assert np.array_equal(original, image)
        assert digest(analysis / item["png"]) == item["png_sha256"]
        assert digest(analysis / item["warmup_sheet"]) == item["warmup_sheet_sha256"]
    for item in details["details"]:
        original = np.load(evidence / item["source_array"], allow_pickle=False)
        x0, y0, x1, y1 = item["source_rectangle_xyxy"]
        assert np.array_equal(np.asarray(Image.open(analysis / item["png"])), original[y0:y1, x0:x1])
        assert digest(analysis / item["png"]) == item["png_sha256"]
    assert digest(analysis / details["overview"]["path"]) == details["overview"]["sha256"]

    policy = root / "sgw-bi-policy-inputs"
    replay = root / "sgw-bi-policy-replay/sgw-bi-policy-inputs"
    assert digest(policy / "manifest.json") == "065a192aef51b65870ed885d176292ef15b8e7198d8b17ac66ef77c217169b41"
    assert digest(replay / "manifest.json") == "4d2e735fcd84ff0e4f36386980a8e629645b2fe11a0185dbad48d642908d2bc2"
    verification = read(analysis / "policy_replay_verification.json")
    assert verification["images_bit_identical"] == 20
    policy_rows = read(policy / "manifest.json")["images"]
    assert len(policy_rows) == 20
    for item in policy_rows:
        name = Path(item["path"]).name
        assert digest(policy / name) == digest(replay / name) == item["png_sha256"]
        pixels = np.asarray(Image.open(policy / name))
        assert list(pixels.shape) == item["shape"] and pixels.dtype == np.uint8
        assert hashlib.sha256(pixels.tobytes()).hexdigest() == item["array_sha256"]

    checkpoint = read(root / "sgw-native-nano-checkpoint-config-bi.json")
    config_bytes = (json.dumps(checkpoint["config"], indent=2) + "\n").encode()
    assert len(config_bytes) == checkpoint["bytes"] == 7735
    assert hashlib.sha256(config_bytes).hexdigest() == checkpoint["sha256"]
    assert "dataloader_train" not in checkpoint["config"]
    for audit_name in ("sgw-native-image-utils-ao.json", "sgw-native-nano-observation-at.json",
                       "sgw-native-nano-transform-bi.json"):
        audit = read(root / audit_name)
        assert hashlib.sha256(audit["text"].encode()).hexdigest() == audit["sha256"]
    for audit in read(root / "sgw-native-nano-spatial-config-bi.json")["files"].values():
        assert hashlib.sha256(audit["text"].encode()).hexdigest() == audit["sha256"]

    history = []
    for name, expected in HISTORICAL.items():
        row = binding(root / name)
        assert row["sha256"] == expected
        history.append(row)
    for report in (analysis / "independent_scene_review.md", analysis / "policy_input_review.md"):
        text = report.read_text()
        for expected in CAPTURES.values():
            assert expected in text
        for target in re.findall(r"\]\(([^)]+)\)", text):
            assert (report.parent / target).resolve().is_file(), (report, target)

    companions = [
        binding(root / name)
        for name in (
            "prepare_sgw_bi_scene_review.py", "prepare_sgw_scene_review.py",
            "prepare_sgw_bi_policy_inputs.py", "verify_sgw_bi_policy_replay.py",
            "seal_sgw_bi_scene_review.py", "sgw-native-image-utils-ao.json",
            "sgw-native-nano-observation-at.json", "sgw-native-nano-transform-bi.json",
            "sgw-native-nano-spatial-config-bi.json", "sgw-native-nano-checkpoint-config-bi.json",
            "sgw-bi-policy-inputs/manifest.json", "sgw-bi-policy-replay/sgw-bi-policy-inputs/manifest.json",
        )
    ]
    result = {
        "schema_version": "sgw-01-independent-bi-native-and-policy-scene-review-v1",
        "review_id": "SGW-01-independent-Astra-ENG-007-bi-20260923",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": {"project_session_id": "0f08c820-0979-467b-908f-fb307177a60d",
                     "model": "gpt-6-astra", "reasoning": "maximum exposed",
                     "same_reviewer_followup_not_additional_rater": True},
        "claim_boundary": (
            "Visual engineering acceptance of these exact four captures, not fixture/controller "
            "qualification, candidate acceptance, live model-tensor parity, semantic recognition, "
            "prediction annotation, behavioral release, or model failure attribution."
        ),
        "findings": {
            "F1": "resolved: supports grounded for recorded initial states",
            "F2": "resolved for recorded initial states; per-candidate clearance still required",
            "F3": "contrast improved; plate category and wrist occlusion caveats remain",
        },
        "archive": archive, "export_manifest": export_binding, "verified_exported_files": exported,
        "export_job_identity": {"name": export["job_name"], "uid": export["job_uid"],
                                "claim_boundary": "export/coordinator identity, not a new cluster-status query"},
        "study_source_commit": SOURCE,
        "source_bindings": [
            git_binding(f"experiments/workshops/spatial_grounding_v1/{name}", expected)
            for name, expected in CODE.items()
        ],
        "contract_bindings": [
            git_binding(path) for path in (
                "docs/SGW_HEIGHT_DIST_CAPTURE_REQUIREMENTS.md",
                "docs/SGW_PROSPECTIVE_HEIGHT_DIST_CAPTURE.md",
                "experiments/workshops/spatial_grounding_v1/spec/EXPERIMENT_SPEC.md",
                "artifacts/workshops/spatial_grounding_v1/STATUS.md",
                "artifacts/workshops/spatial_grounding_v1/continuation_state.json",
            )
        ],
        "scenes": scene_rows,
        "reviewed_media": {"native_full_resolution_final_views": 12, "lossless_warmup_samples": 72,
                           "decoded_video_samples_inspected": 24, "complete_videos_decoded": 4,
                           "total_video_frames_decoded": 484, "sample_indices": [0, 1, 10, 30, 60, 120],
                           "D1_API_images_opened": 12, "N3_server_images_opened": 4,
                           "N3_spatial_images_opened": 4, "policy_images_CPU_bit_identical": 20},
        "analysis_files": [binding(path) for path in sorted(analysis.rglob("*")) if path.is_file()],
        "policy_images": [
            {**binding(policy / Path(item["path"]).name), "scene": item["scene"], "slot": item["slot"],
             "shape": item["shape"], "rgb_sha256": item["array_sha256"]}
            for item in policy_rows
        ],
        "companions": companions, "bf_report_manifest_and_verifier_unchanged": history,
        "missing_or_unqualified": [
            "115 additional recorded warmup NPYs per scene not exported locally",
            "inherited external USD/MDL/texture payload bytes not included in the export",
            "sustained physical stability: render warmup advances no physics",
            "measured controller calibration and downstream six-trial qualification",
            "live request-time internal model tensors and policy category recognition",
            "generated future evidence and later independent blinded prediction annotation",
        ],
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    assert read(output) == result
    for group in ("analysis_files", "policy_images", "companions", "verified_exported_files",
                  "bf_report_manifest_and_verifier_unchanged"):
        for item in result[group]:
            assert binding(root / item["path"]) == {key: item[key] for key in ("path", "bytes", "sha256")}
    print(json.dumps({
        "sealed_manifest": binding(output),
        "reports": [binding(analysis / name) for name in ("independent_scene_review.md", "policy_input_review.md")],
        "verified_exported_files": 118, "bound_analysis_files": len(result["analysis_files"]),
        "bound_policy_images": len(result["policy_images"]),
    }, indent=2))


if __name__ == "__main__":
    main()
