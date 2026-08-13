"""Offline A004 source-push verification installed only by A004 launchers."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
import subprocess
from typing import Any, Mapping

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import ContractError, require, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[3]
PINNED_RECEIPT_SCHEMA = "vla-wam-shared-v3c002r001-activation-v4-source-gate-v3"
_RECEIPT: dict[str, Any] | None = None


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)


def install_from_environment() -> dict[str, Any]:
    """Bind the immutable orchestrator receipt; never contact or fetch a remote."""
    global _RECEIPT
    path_text = os.environ.get("V3C002_A004_PINNED_RECEIPT")
    expected = os.environ.get("V3C002_A004_PINNED_RECEIPT_SHA256")
    require(isinstance(path_text, str) and path_text and isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None, "A004 pinned receipt environment is missing")
    path = Path(path_text).resolve()
    require(path.is_file() and sha256_file(path) == expected, "A004 pinned receipt bytes changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict) and value.get("schema_version") == PINNED_RECEIPT_SCHEMA and value.get("status") == "passed_universal_a004_source_and_registration_pushed_before_behavior" and value.get("passed") is True and value.get("pushed") is True, "A004 pinned receipt did not pass")
    require(value.get("a004_model_requests_before_registration") == 0 and value.get("a004_behavioral_episodes_before_registration") == 0, "A004 pinned receipt is retrospective")
    head = value.get("remote_head_at_gate")
    require(isinstance(head, str) and re.fullmatch(r"[0-9a-f]{40}", head) is not None, "A004 pinned remote head is invalid")
    require(_git("cat-file", "-e", f"{head}^{{commit}}").returncode == 0, "A004 pinned head is unavailable locally")
    _RECEIPT = value
    return value


def _head() -> str:
    require(_RECEIPT is not None, "A004 pinned source verification was not installed")
    return str(_RECEIPT["remote_head_at_gate"])


def verify_pinned_local_ancestry(commit: Any, *, label: str) -> None:
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None, f"{label} commit is invalid")
    require(_git("cat-file", "-e", f"{commit}^{{commit}}").returncode == 0, f"{label} commit is unavailable locally")
    require(_git("merge-base", "--is-ancestor", commit, _head()).returncode == 0, f"{label} commit is outside pinned pushed ancestry")


def verify_parent_pushed_source(source_gate: Mapping[str, Any]) -> None:
    require(isinstance(source_gate, Mapping), "source gate is invalid")
    require(isinstance(source_gate.get("branch"), str) and source_gate["branch"] and isinstance(source_gate.get("remote"), str) and source_gate["remote"], "source gate remote/branch missing")
    verify_pinned_local_ancestry(source_gate.get("source_commit"), label="source gate")


def verify_r001_pushed_gate(source_gate: Mapping[str, Any], repair: Mapping[str, Any]) -> None:
    """Exact frozen R001 non-network checks plus pinned local ancestry."""
    from experiments.v3.phase_c_semantic_equivalence_v3c002r001 import contract as r001

    require(source_gate.get("schema_version") == r001.SOURCE_GATE_SCHEMA and source_gate.get("status") == "passed_repair_source_and_registration_pushed", "repair source gate did not pass")
    require(source_gate.get("passed") is True and source_gate.get("pushed") is True, "repair source is not pushed")
    require(source_gate.get("repair_registration_sha256") == sha256_file(r001.resolve(source_gate["repair_registration"])), "repair source gate registration changed")
    require(source_gate.get("implementation_commit") == repair.get("runtime", {}).get("repair_wrapper_implementation_commit"), "repair implementation/source gate lineage changed")
    require(source_gate.get("queue", {}).get("sha256") == repair.get("queue", {}).get("sha256") and source_gate.get("assignment_manifest", {}).get("sha256") == repair.get("assignment_manifest", {}).get("sha256"), "repair source gate queue/assignment changed")
    require(isinstance(source_gate.get("remote"), str) and source_gate["remote"] and isinstance(source_gate.get("branch"), str) and source_gate["branch"], "repair source remote/branch missing")
    for key in ("implementation_commit", "registration_commit"):
        verify_pinned_local_ancestry(source_gate.get(key), label=f"repair {key}")


def install_contract_monkeypatches() -> dict[str, Any]:
    """Patch imported contracts in memory; frozen source files remain untouched."""
    receipt = install_from_environment()
    from experiments.v3.phase_c_semantic_equivalence_v3c002 import contract as parent
    from experiments.v3.phase_c_semantic_equivalence_v3c002r001 import contract as r001
    parent._verify_pushed_source_commit = verify_parent_pushed_source
    r001.verify_pushed_gate = verify_r001_pushed_gate
    return receipt
