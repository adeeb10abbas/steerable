"""Recheck the bounded historical lineage export without granting coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def selected(root: Path, row: dict) -> dict:
    path = (root / row["export_path"]).resolve()
    require(path.is_relative_to(root.resolve()), "export path escapes its root")
    raw = path.read_bytes()
    require(len(raw) == row["export_bytes"] and digest(raw) == row["export_sha256"],
            "selected export byte/hash mismatch")
    value = json.loads(raw)
    require(value["source_sha256"] == row["expected_sha256"]
            and value["source_bytes"] == row["expected_bytes"], "raw source binding mismatch")
    return value


def audit(repo: Path, export: Path, prior: Path, inputs: Path) -> dict:
    config = json.loads(inputs.read_text())["data"]
    request = json.loads(config["requests.json"])
    prior_raw = (prior / "manifest.json").read_bytes()
    require(digest(prior_raw) == request["prior_source_export_manifest_sha256"],
            "prior export manifest mismatch")
    manifest_raw = (export / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    for key, filename in (
        ("request_sha256", "requests.json"),
        ("extractor_sha256", "historical_layout_streaming.py"),
        ("launcher_sha256", "run.py"),
    ):
        require(manifest[key] == digest(config[filename].encode()), f"{key} mismatch")
    require(request["release_authorization"] is False and request["retain_source_lineage"] is True,
            "unexpected extraction authority")
    require(manifest["model_requests"] == manifest["behavioral_episodes"] == manifest["allocated_gpus"] == 0
            and manifest["release_permitted"] is False, "unexpected export authority")
    old_records = json.loads(prior_raw)["records"]
    require(len(manifest["records"]) == len(old_records) == len(request["records"]),
            "source inventory size differs")
    records = []
    for row, old_row, expected in zip(manifest["records"], old_records, request["records"], strict=True):
        for key in ("cohort_id", "path", "expected_sha256", "expected_bytes"):
            require(row[key] == old_row[key] == expected[key], f"source inventory {key} mismatch")
        current = selected(export, row)
        previous = selected(prior, old_row)
        require(current["selection_contract"] == request["selection_contract"], "selection contract mismatch")
        for group in (
            "fresh_reset_objects", "candidate_state_objects", "full_reset_comparisons",
            "reference_bounds", "geometry_preflight_identity",
        ):
            require(current[group] == previous[group], f"prior {group} changed")
        lineage = {item["json_pointer"]: item["value"] for item in current["source_lineage"]}
        require(len(lineage) == len(current["source_lineage"]) == 2, "lineage pointers differ")
        source = lineage["/execution_evidence/construction_source"]
        bindings = lineage["/execution_evidence/input_bindings"]
        revision = source["study_commit"]
        require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "invalid exact source commit")
        prefix, separator, relative = source["path"].partition("/experiments/")
        require(bool(separator) and bool(relative), "producer path lacks its source-root boundary")
        prefix += "/"
        verified = []
        external = []
        producer = None
        for role, binding in {"construction_source": source, **bindings}.items():
            if not binding["path"].startswith(prefix):
                external.append({"role": role, **binding})
                continue
            relative = binding["path"][len(prefix):]
            require(not PurePosixPath(relative).is_absolute() and ".." not in PurePosixPath(relative).parts,
                    "invalid repository-relative binding")
            raw = subprocess.check_output(["git", "-C", str(repo), "cat-file", "blob", f"{revision}:{relative}"])
            require(len(raw) == binding["bytes"] and digest(raw) == binding["sha256"],
                    f"{row['cohort_id']} {role} differs from its recorded Git object")
            verified.append({"role": role, "path": relative, "bytes": len(raw), "sha256": digest(raw)})
            if role == "construction_source":
                producer = raw.decode().splitlines()
        require(producer is not None, "producer source is not verified")
        getters = {}
        for name, expression in (
            ("object_position", '"position_world_m": _host(data.root_pos_w[0])'),
            ("object_quaternion", '"quaternion_world_wxyz": _host(data.root_quat_w[0])'),
            ("environment_origin", '"scene_env_origin_world_m": env_origin.tolist()'),
        ):
            locations = [number for number, line in enumerate(producer, 1) if expression in line]
            require(len(locations) == 1, f"ambiguous or absent source expression: {name}")
            getters[name] = {"expression": expression, "line": locations[0]}
        frames = current["frame_identity"]
        require(bool(frames), "no source-defined frame records")
        frame_pointers = [item["json_pointer"] for item in frames]
        require(len(frame_pointers) == len(set(frame_pointers)), "duplicate frame records")
        records.append({
            "cohort_id": row["cohort_id"], "raw_source": row["path"],
            "raw_source_sha256": row["expected_sha256"], "raw_source_bytes": row["expected_bytes"],
            "selected_export_sha256": row["export_sha256"], "source_commit": revision,
            "verified_git_bindings": verified, "external_bindings_not_verified_by_this_audit": external,
            "source_expressions": getters, "frame_record_count": len(frames),
            "all_frame_identity_checks_passed": all(item["value"]["passed"] is True for item in frames),
            "observed_scene_environment_origins_world_m": sorted({
                tuple(item["value"]["scene_env_origin_world_m"]) for item in frames
            }),
            "fresh_reset_object_records": len(current["fresh_reset_objects"]),
            "candidate_state_object_records": len(current["candidate_state_objects"]),
            "prior_selected_fields_unchanged": True,
        })
    return {
        "schema_version": "sgw-01-historical-source-lineage-audit-v1",
        "export_manifest_sha256": digest(manifest_raw),
        "prior_export_manifest_sha256": digest(prior_raw),
        "inputs_manifest_sha256": digest(inputs.read_bytes()),
        "auditor_sha256": digest(Path(__file__).read_bytes()),
        "records": records, "model_requests": 0, "behavioral_episodes": 0,
        "release_permitted": False, "historical_population_coverage_complete": False,
        "status": "verified_selected_source_lineage_not_complete_geometry_or_population_coverage",
        "claim_boundary": (
            "The exact producers read native world actor roots; the separately named EEF "
            "fields must not redefine object-root semantics. Origin records apply only to "
            "their named snapshots. Source/scene bindings are not an exhaustive layout "
            "population, asset-payload verification, or measured root-local center geometry. "
            "No historical acceptance or SGW identity is required for later comparison."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--prior-export-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(Path(__file__).resolve().parents[1], args.export_root, args.prior_export_root, args.inputs)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
