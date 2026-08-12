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
    parser.add_argument("--candidate-root", type=Path)
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

    candidate_ledger_path = root / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006/gates/model_blind_candidate_infrastructure_invalid.jsonl"
    candidate_rows = [
        json.loads(line) for line in candidate_ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(candidate_rows) != 4:
        raise AssertionError("expected the four retained candidate/prelaunch invalid attempts")
    for row in candidate_rows:
        if row.get("schema_version") != "vla-wam-shared-v3e006-model-blind-candidate-infrastructure-invalid-v1":
            raise AssertionError("invalid candidate infrastructure schema")
        for key in ("model_request_count", "behavioral_episode_count", "state_candidate_count"):
            if row.get(key) != 0:
                raise AssertionError(f"candidate infrastructure attempt has nonzero {key}")
        if row.get("behavioral_denominator_included") is not False or row.get("candidate_gate_passed") is not False:
            raise AssertionError("invalid candidate attempt entered a scientific denominator")
        verify_binding(row["invocation"]["exact_argv_text"], verify_raw=args.verify_raw)
        if args.verify_raw:
            invocation_text = Path(row["invocation"]["exact_argv_text"]["path"]).read_text(encoding="utf-8")
            if "canonical_stage_localization_v3e006.state_construction_gate" not in invocation_text:
                raise AssertionError("candidate invalid exact argv differs")
        for binding_row in row.get("raw_sources", {}).values():
            verify_binding(binding_row, verify_raw=args.verify_raw)
        if "construction_source" in row:
            verify_binding(row["construction_source"], verify_raw=args.verify_raw)
    if "robolab/assets/scenes" not in candidate_rows[0]["invocation"]["wrong_path"]:
        raise AssertionError("prelaunch-invalid wrong path is not retained")

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
        expected_home = str(preflight_root / "cache" / "home")
        if launch.get("environment", {}).get("HOME") != expected_home:
            raise AssertionError("preflight did not retain its unique writable HOME")
        if args.verify_raw and not Path(expected_home).is_dir():
            raise AssertionError("preflight writable HOME is missing")
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
            verify_binding(child["pose_helper_source"], verify_raw=True)
            verify_binding(child["installed_pose_api_source"], verify_raw=True)
            if child["pose_helper_source"] != launch["input_bindings"]["pose_helper_source"]:
                raise AssertionError("child pose helper differs from launch binding")
            if child["installed_pose_api_source"] != launch["input_bindings"]["installed_pose_api_source"]:
                raise AssertionError("child installed pose API differs from launch binding")
            for binding_row in child.get("bound_inputs", {}).values():
                verify_binding(binding_row, verify_raw=True)
            for binding_row in child.get("diagnostic_inputs", []):
                verify_binding(binding_row, verify_raw=True)
            runtime_path = Path(launch["input_bindings"]["runtime_contract"]["path"])
            runtime_contract = json.loads(runtime_path.read_text(encoding="utf-8"))
            renderer = runtime_contract["components"]["renderer"]["contract"]
            if child.get("runtime_contract_observation", {}).get("canonical_contract_sha256") != runtime_contract.get("canonical_contract_sha256"):
                raise AssertionError("child runtime observation differs from the frozen exact E004 contract")
            if child.get("app_launcher_runtime") != {
                "renderer": renderer["renderer"],
                "rendering_mode": renderer["rendering_type"],
                "device": renderer["device"],
                "headless": renderer["headless"],
            }:
                raise AssertionError("child AppLauncher runtime differs from the frozen E004 renderer contract")
        else:
            if result.get("status") != "infrastructure_invalid_zero_model_health_preflight":
                raise AssertionError("failed preflight lacks invalid status")
            if result.get("child_report") is not None:
                verify_binding(result["child_report"], verify_raw=True)
                child = json.loads(Path(result["child_report"]["path"]).read_text(encoding="utf-8"))
                if child.get("status") != "infrastructure_invalid_zero_model_runtime_health_preflight" or child.get("passed") is not False:
                    raise AssertionError("failed child report lacks infrastructure-invalid status")
                for key in ("model_request_count", "behavioral_episode_count", "state_candidate_count"):
                    if child.get(key) != 0:
                        raise AssertionError(f"failed child preflight has nonzero {key}")
                verify_binding(child["source"], verify_raw=True)
                verify_binding(child["pose_helper_source"], verify_raw=True)
                if child["pose_helper_source"] != launch["input_bindings"]["pose_helper_source"]:
                    raise AssertionError("failed child pose helper differs from launch binding")
                if child.get("installed_pose_api_source") is not None:
                    verify_binding(child["installed_pose_api_source"], verify_raw=True)
                    if child["installed_pose_api_source"] != launch["input_bindings"]["installed_pose_api_source"]:
                        raise AssertionError("failed child installed pose API differs from launch binding")
                for binding_row in child.get("bound_inputs", {}).values():
                    verify_binding(binding_row, verify_raw=True)
                for binding_row in child.get("diagnostic_inputs", []):
                    verify_binding(binding_row, verify_raw=True)
                runtime_path = Path(launch["input_bindings"]["runtime_contract"]["path"])
                runtime_contract = json.loads(runtime_path.read_text(encoding="utf-8"))
                renderer = runtime_contract["components"]["renderer"]["contract"]
                if child.get("runtime_contract_observation", {}).get("canonical_contract_sha256") != runtime_contract.get("canonical_contract_sha256"):
                    raise AssertionError("failed child runtime observation differs from frozen E004")
                if child.get("app_launcher_runtime") != {
                    "renderer": renderer["renderer"],
                    "rendering_mode": renderer["rendering_type"],
                    "device": renderer["device"],
                    "headless": renderer["headless"],
                }:
                    raise AssertionError("failed child AppLauncher runtime differs from frozen E004")
                if not isinstance(child.get("invocation"), list) or len(child["invocation"]) < 20:
                    raise AssertionError("failed child invocation is incomplete")
                child_environment = child.get("environment", {})
                if not child_environment or any(
                    value != launch.get("environment", {}).get(key) for key, value in child_environment.items()
                ):
                    raise AssertionError("failed child environment differs from retained launch")
                if not child.get("error", {}).get("type") or not child.get("error", {}).get("traceback"):
                    raise AssertionError("failed child report lacks retained exception evidence")
                if result.get("child_status") != child["status"]:
                    raise AssertionError("outer result does not bind the failed child status")
        preflight_status = result["status"]
    candidate_status = None
    candidate_runtime_log = None
    if args.candidate_root is not None:
        candidate_root = args.candidate_root.resolve()
        failure_path = candidate_root / "state_construction_failure.json"
        passed_path = candidate_root / "state_candidate.json"
        if failure_path.is_file() == passed_path.is_file():
            raise AssertionError("candidate root must contain exactly one passed/failure report")
        report_path = passed_path if passed_path.is_file() else failure_path
        report = json.loads(report_path.read_text(encoding="utf-8"))
        evidence = report["execution_evidence"] if passed_path.is_file() else report
        expected_status = (
            "passed_model_blind_state_construction_not_released_for_inference"
            if passed_path.is_file()
            else "infrastructure_invalid_model_blind_state_construction"
        )
        if report.get("status") != expected_status:
            raise AssertionError("state-construction report status differs")
        if report.get("model_request_count") != 0 or report.get("behavioral_episode_count") != 0:
            raise AssertionError("state-construction report has a model/behavior count")
        if passed_path.is_file():
            if report.get("passed") is not True or report.get("state_candidate_count") != 1:
                raise AssertionError("passed candidate count/status differs")
            if evidence.get("candidate_gate_passed") is not True:
                raise AssertionError("passed candidate execution gate differs")
        else:
            if report.get("passed") is not False or report.get("state_candidate_count") != 0:
                raise AssertionError("failed candidate count/status differs")
            if report.get("candidate_gate_passed") is not False:
                raise AssertionError("failed candidate entered candidate denominator")
            if not report.get("error", {}).get("type") or not report.get("error", {}).get("traceback"):
                raise AssertionError("failed state construction lacks exact traceback")
        verify_binding(evidence["construction_source"], verify_raw=True)
        for binding_row in evidence.get("input_bindings", {}).values():
            verify_binding(binding_row, verify_raw=True)
        for binding_row in evidence.get("passed_health_preflight", {}).values():
            verify_binding(binding_row, verify_raw=True)
        health = json.loads(Path(evidence["passed_health_preflight"]["harness_result"]["path"]).read_text(encoding="utf-8"))
        if health.get("status") != "passed_generic_zero_model_health_preflight" or health.get("passed") is not True:
            raise AssertionError("candidate is not bound to the passed formal health preflight")
        runtime_path = Path(evidence["runtime_log"]["path"])
        if not runtime_path.is_file():
            raise AssertionError("candidate runtime log is missing")
        candidate_runtime_log = {
            "path": str(runtime_path),
            "bytes": runtime_path.stat().st_size,
            "sha256": sha256(runtime_path),
        }
        candidate_status = report["status"]
    print(
        json.dumps(
            {
                "passed": True,
                "retained_attempts": len(rows),
                "retained_candidate_attempts": len(candidate_rows),
                "verified_raw": args.verify_raw,
                "preflight_status": preflight_status,
                "candidate_status": candidate_status,
                "candidate_runtime_log": candidate_runtime_log,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
