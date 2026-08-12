#!/usr/bin/env python3
"""Validate retained V3-E006 zero-request infrastructure evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_binding(binding: Mapping[str, Any], *, verify_raw: bool) -> None:
    if not {"path", "bytes", "sha256"} <= set(binding):
        raise AssertionError("incomplete evidence binding")
    if not isinstance(binding["bytes"], int) or binding["bytes"] < 0:
        raise AssertionError("invalid evidence byte count")
    if not isinstance(binding["sha256"], str) or len(binding["sha256"]) != 64:
        raise AssertionError("invalid evidence SHA-256")
    path = Path(str(binding["path"]))
    if verify_raw:
        if not path.is_file() or path.stat().st_size != binding["bytes"] or sha256(path) != binding["sha256"]:
            raise AssertionError(f"target evidence differs: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--verify-raw", action="store_true")
    parser.add_argument("--preflight-root", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    artifact = root / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006/gates/model_blind_infrastructure_invalid.jsonl"
    rows = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 2:
        raise AssertionError("expected the two retained model-blind infrastructure-invalid attempts")
    commits = set()
    for row in rows:
        if row.get("schema_version") != "vla-wam-shared-v3e006-model-blind-infrastructure-invalid-v2":
            raise AssertionError("invalid infrastructure schema")
        for key in ("model_request_count", "behavioral_episode_count", "behavioral_action_count", "state_candidate_count"):
            if row.get(key) != 0:
                raise AssertionError(f"infrastructure attempt has nonzero {key}")
        if row.get("behavioral_denominator_included") is not False or row.get("candidate_gate_passed") is not False:
            raise AssertionError("infrastructure attempt entered a scientific denominator")
        if len(row.get("invocation", {}).get("argv", [])) < 20:
            raise AssertionError("full invocation not retained")
        commits.add(row["invocation"]["study_commit"])
        for binding in row["input_bindings"].values():
            verify_binding(binding, verify_raw=args.verify_raw)
        verify_binding(row["construction_source"], verify_raw=False)
        for binding in row["raw_sources"].values():
            verify_binding(binding, verify_raw=args.verify_raw)
    if commits != {
        "011e61396d7831001e9614f3929108a0202535fc",
        "0c733d29b36dbadd4eba4009a1c3887ef50367a8",
    }:
        raise AssertionError("retained infrastructure commit set differs")

    lineage_path = root / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006/source_lineage.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    for row in lineage["commits"]:
        completed = subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"{row['sha']}^{{commit}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode:
            raise AssertionError(f"source-lineage SHA is not a commit: {row['sha']}")
    preflight_status = None
    if args.preflight_root is not None:
        preflight_root = args.preflight_root.resolve()
        launch_path = preflight_root / "preflight_launch.json"
        result_path = preflight_root / "harness_result.json"
        if not launch_path.is_file() or not result_path.is_file():
            raise AssertionError("preflight launch/result evidence incomplete")
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if launch.get("schema_version") != "vla-wam-shared-v3e006-zero-model-preflight-launch-v1":
            raise AssertionError("preflight launch schema differs")
        if result.get("schema_version") != "vla-wam-shared-v3e006-zero-model-preflight-harness-result-v1":
            raise AssertionError("preflight result schema differs")
        verify_binding(launch["harness_source"], verify_raw=True)
        verify_binding(launch["python_interpreter"], verify_raw=True)
        if launch.get("child_argv", [None])[0] != launch["python_interpreter"]["path"]:
            raise AssertionError("child argv does not preserve the registered Python interpreter path")
        if Path(launch["python_interpreter"]["path"]).resolve() != Path(launch["python_interpreter"]["resolved_path"]):
            raise AssertionError("registered Python interpreter target differs")
        verify_binding(result["harness_source"], verify_raw=True)
        if result["harness_source"] != launch["harness_source"]:
            raise AssertionError("preflight harness source binding differs")
        if not isinstance(launch.get("outer_argv"), list) or len(launch["outer_argv"]) < 20:
            raise AssertionError("complete outer preflight invocation is missing")
        for binding_row in launch.get("input_bindings", {}).values():
            verify_binding(binding_row, verify_raw=True)
        for binding_row in launch.get("diagnostic_inputs", []):
            verify_binding(binding_row, verify_raw=True)
        for key in ("model_request_count", "behavioral_episode_count", "state_candidate_count"):
            if result.get(key) != 0:
                raise AssertionError(f"preflight has nonzero {key}")
        if result.get("behavioral_denominator_included") is not False or result.get("candidate_denominator_included") is not False:
            raise AssertionError("preflight entered a scientific denominator")
        verify_binding(result["launch"], verify_raw=True)
        verify_binding(result["runtime_log"], verify_raw=True)
        if result.get("passed") is True:
            verify_binding(result["child_report"], verify_raw=True)
            child = json.loads(Path(result["child_report"]["path"]).read_text(encoding="utf-8"))
            if child.get("status") != "passed_generic_zero_model_cuda_vulkan_isaac_physics_render_health_preflight":
                raise AssertionError("child preflight status differs")
            verify_binding(child["source"], verify_raw=True)
            for binding_row in child.get("bound_inputs", {}).values():
                verify_binding(binding_row, verify_raw=True)
            for binding_row in child.get("diagnostic_inputs", []):
                verify_binding(binding_row, verify_raw=True)
        elif result.get("status") != "infrastructure_invalid_zero_model_health_preflight":
            raise AssertionError("failed preflight lacks invalid status")
        preflight_status = result["status"]
    print(json.dumps({"passed": True, "retained_attempts": len(rows), "verified_raw": args.verify_raw, "preflight_status": preflight_status}, sort_keys=True))


if __name__ == "__main__":
    main()
