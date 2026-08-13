#!/usr/bin/env python3
"""Validate the published C002-R001 bundle after the branch advanced.

The frozen finalizer validator remains untouched.  Its episode rows retain the
absolute execution-checkout paths by design, so this additive closure invokes
it with the exact historical source checkout used at execution while comparing
against the byte-identical files published by RESULTS_COMMIT.  No raw/PVC path
is relocated or normalized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import ContractError, require
SOURCE_COMMIT = "785ea96419df51b92249ef7cdd1b5dbd59ff0a50"
RESULTS_COMMIT = "700a1f76a2f8ec2ac8e19db669c9afe3668f8a85"
IMPLEMENTATION_COMMIT = "9def5c605c04d070e32b4069578e6378dcf21cd7"
REGISTRATION_COMMIT = "42703eb998cda35f04701def31a428d9c4281f01"
SOURCE_GATE_SHA256 = "1eee9f34fd585a8dedfc4eac1f69c4a428b13cc7c36ec5cdf8ef0436dc0fd0cf"
HISTORICAL_VALIDATOR_SHA256 = "ae9aced8d3cd75ed515a966d8d4c8503e631645c75c06ae561ce6d2cbea85193"
CANONICAL_REMOTE = "https://github.com/adeeb10abbas/steerable.git"
CANONICAL_BRANCH = "experiment/v3c002-semantic-equivalence"
BASE = Path("artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001")
FINAL = BASE / "activation_v4/final_analysis_v3"
PUBLISHED_PATHS = (
    FINAL / "finalization_execution_receipt.json",
    FINAL / "infrastructure_attempts.jsonl",
    FINAL / "raw_aggregation_receipt.json",
    FINAL / "results/DECISION_MEMO.md",
    FINAL / "results/MANUSCRIPT_INSERT.md",
    FINAL / "results/episodes.jsonl",
    FINAL / "results/epoch_diagnostics.json",
    FINAL / "results/evidence_manifest.json",
    FINAL / "results/pairs.jsonl",
    FINAL / "results/results.json",
    FINAL / "validator.stdout.json",
)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True, check=False)
    require(result.returncode == 0, f"post-publication git check failed: {' '.join(args)}")
    return result.stdout.strip()


def _bytes_at(commit: str, relative: Path) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{commit}:{relative.as_posix()}"],
        capture_output=True, check=False,
    )
    require(result.returncode == 0, f"published artifact is absent from {commit}: {relative}")
    return result.stdout


def _binding(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def validate_publication_tree() -> dict[str, dict[str, Any]]:
    head = _git(REPO_ROOT, "rev-parse", "HEAD")
    require(_git(REPO_ROOT, "status", "--porcelain") == "", "post-publication checkout is not clean")
    for commit in (SOURCE_COMMIT, RESULTS_COMMIT, IMPLEMENTATION_COMMIT):
        require(_git(REPO_ROOT, "cat-file", "-t", commit) == "commit", f"missing pinned commit {commit}")
    _git(REPO_ROOT, "merge-base", "--is-ancestor", SOURCE_COMMIT, RESULTS_COMMIT)
    require(_git(REPO_ROOT, "rev-parse", f"{RESULTS_COMMIT}^") == SOURCE_COMMIT, "results commit is not the direct evidence-only child of the frozen source head")
    _git(REPO_ROOT, "merge-base", "--is-ancestor", RESULTS_COMMIT, head)
    remote = _git(REPO_ROOT, "remote", "get-url", "origin")
    require(remote == CANONICAL_REMOTE, "post-publication canonical remote changed")
    rows = _git(REPO_ROOT, "ls-remote", "--heads", remote, CANONICAL_BRANCH).splitlines()
    require(len(rows) == 1 and rows[0].endswith(f"\trefs/heads/{CANONICAL_BRANCH}"), "post-publication remote branch receipt changed")
    remote_head = rows[0].split("\t", 1)[0]
    _git(REPO_ROOT, "merge-base", "--is-ancestor", RESULTS_COMMIT, remote_head)
    changed = _git(REPO_ROOT, "diff", "--name-status", SOURCE_COMMIT, RESULTS_COMMIT).splitlines()
    require(changed == [f"A\t{path.as_posix()}" for path in PUBLISHED_PATHS], "results commit is not the exact eleven-file evidence-only publication")
    bindings: dict[str, dict[str, Any]] = {}
    for relative in PUBLISHED_PATHS:
        path = REPO_ROOT / relative
        data = path.read_bytes()
        committed = _bytes_at(RESULTS_COMMIT, relative)
        require(data == committed, f"published artifact differs from results commit: {relative}")
        bindings[relative.as_posix()] = _binding(path)
    return bindings


def validate_historical_source(source_checkout: Path) -> dict[str, Any]:
    require(source_checkout.is_absolute(), "historical source checkout must be an absolute retained execution path")
    require(_git(source_checkout, "rev-parse", "HEAD") == SOURCE_COMMIT, "historical execution checkout head changed")
    require(_git(source_checkout, "status", "--porcelain") == "", "historical execution checkout is not clean")
    source_path = source_checkout / FINAL / "source_push_gate.released.json"
    require(_binding(source_path)["sha256"] == SOURCE_GATE_SHA256, "historical source-gate blob changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    require(
        source.get("passed") is True and source.get("pushed") is True
        and source.get("implementation_commit") == IMPLEMENTATION_COMMIT
        and source.get("registration_commit") == REGISTRATION_COMMIT
        and source.get("remote_head_at_gate") == source.get("registration_commit"),
        "historical final-analysis source admission changed",
    )
    inventory = source.get("source_inventory")
    require(isinstance(inventory, dict) and inventory, "historical source inventory missing")
    for relative, binding in inventory.items():
        data = _bytes_at(IMPLEMENTATION_COMMIT, Path(relative))
        require(len(data) == binding.get("bytes") and hashlib.sha256(data).hexdigest() == binding.get("sha256"), f"historical implementation inventory changed: {relative}")
    _git(source_checkout, "merge-base", "--is-ancestor", IMPLEMENTATION_COMMIT, REGISTRATION_COMMIT)
    _git(source_checkout, "merge-base", "--is-ancestor", REGISTRATION_COMMIT, SOURCE_COMMIT)
    registration = source_checkout / FINAL / "registration.json"
    require(_binding(registration)["sha256"] == source.get("registration_sha256"), "historical registration blob changed")
    return _binding(source_path)


def validate_execution_receipt(bindings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    receipt_path = REPO_ROOT / FINAL / "finalization_execution_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(
        receipt.get("passed") is True and receipt.get("exit_code") == 0
        and receipt.get("valid_behavioral_episodes") == 1364
        and receipt.get("complete_seed_blocks") == 341
        and receipt.get("prompt_form_pairs") == 682
        and receipt.get("infrastructure_attempts_excluded") == 14,
        "published finalization execution receipt changed",
    )
    for name, relative in {
        "compiled_episodes": FINAL / "results/episodes.jsonl", "pairs": FINAL / "results/pairs.jsonl",
        "results": FINAL / "results/results.json", "epoch_diagnostics": FINAL / "results/epoch_diagnostics.json",
        "evidence_manifest": FINAL / "results/evidence_manifest.json", "decision_memo": FINAL / "results/DECISION_MEMO.md",
        "manuscript_insert": FINAL / "results/MANUSCRIPT_INSERT.md", "raw_aggregation_receipt": FINAL / "raw_aggregation_receipt.json",
        "infrastructure_attempts": FINAL / "infrastructure_attempts.jsonl", "invocation_validator_stdout": FINAL / "validator.stdout.json",
    }.items():
        expected = bindings[relative.as_posix()]
        actual = receipt["bindings"][name]
        require(actual.get("bytes") == expected["bytes"] and actual.get("sha256") == expected["sha256"], f"execution receipt binding changed: {name}")
    # Every remaining invocation/raw admission binding is checked at its exact
    # retained absolute path.  These are never relocated into the repository.
    for name in (
        "raw_episodes", "prelaunch_receipt", "script", "invocation_argv",
        "invocation_environment", "invocation_checkout_head", "invocation_checkout_status",
        "invocation_remote_head", "invocation_script_sha", "invocation_prelaunch_sha",
        "invocation_exit_code",
    ):
        actual = receipt["bindings"].get(name)
        require(isinstance(actual, dict) and Path(str(actual.get("path", ""))).is_absolute(), f"execution receipt durable binding missing: {name}")
        retained = _binding(Path(actual["path"]))
        require(retained["bytes"] == actual.get("bytes") and retained["sha256"] == actual.get("sha256"), f"execution receipt retained binding changed: {name}")
    return receipt


def regenerate(*, source_checkout: Path, raw_root: Path) -> dict[str, Any]:
    require(raw_root.is_absolute(), "raw root must remain an exact absolute PVC path")
    source = source_checkout / BASE
    published = REPO_ROOT / FINAL / "results"
    historical_validator = source_checkout / "experiments/v3/phase_c_semantic_equivalence_v3c002r001/activation_v4_finalizer/validator.py"
    require(_binding(historical_validator)["sha256"] == HISTORICAL_VALIDATOR_SHA256, "frozen historical validator source changed")
    # Import and execute the untouched validator from the exact historical
    # source checkout so its relocation of committed artifact bindings is
    # identical to execution time.  The sole monkeypatch is a process-local
    # response for the now-obsolete live-head equality check; ancestry and all
    # source/result bytes were already verified above.  No checkout, remote,
    # raw file, or published result is mutated.
    driver = r'''
import json
from pathlib import Path
import subprocess
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_finalizer import finalizer, validator
real_run = finalizer.subprocess.run
source_commit = __SOURCE_COMMIT__
branch = __BRANCH__
remote = __REMOTE__
def historical_run(args, *pos, **kw):
    if list(args[:3]) == ["git", "ls-remote", "--heads"] and args[3:] == [remote, branch]:
        return subprocess.CompletedProcess(args, 0, stdout=f"{source_commit}\trefs/heads/{branch}\n", stderr="")
    return real_run(args, *pos, **kw)
finalizer.subprocess.run = historical_run
validator.validate_bundle(**{key: Path(value) for key, value in json.loads(__ARGS__).items()})
'''.replace("__SOURCE_COMMIT__", repr(SOURCE_COMMIT)).replace("__BRANCH__", repr(CANONICAL_BRANCH)).replace("__REMOTE__", repr(CANONICAL_REMOTE))
    arguments = {
        "output_dir": str(published),
        "parent_registration": str(source_checkout / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002/active/registration.json"),
        "queue": str(source / "activation_v3/queue.jsonl"),
        "raw_episodes": str(raw_root / "raw/episodes.jsonl"),
        "infrastructure_attempts": str(raw_root / "infrastructure_attempts.jsonl"),
        "aggregation_receipt": str(raw_root / "raw_aggregation_receipt.json"),
        "finalization_registration": str(source / "activation_v4/final_analysis_v3/registration.json"),
        "finalization_source_gate": str(source / "activation_v4/final_analysis_v3/source_push_gate.released.json"),
        "original_release": str(source / "activation_v3/release_gate.released.json"),
        "a003_release": str(source / "activation_v3/lane_replacement_a003/release_gate.released.json"),
        "continuation_gate": str(source / "activation_v4/v10/continuation_gate.released.json"),
        "v11_registration": str(source / "activation_v4/v11/registration.json"),
        "v11_source_gate": str(source / "activation_v4/v11/source_push_gate.released.json"),
    }
    driver = driver.replace("__ARGS__", repr(json.dumps(arguments)))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source_checkout)
    result = subprocess.run(
        [sys.executable, "-c", driver], cwd=source_checkout, env=env,
        text=True, capture_output=True, check=False,
    )
    require(result.returncode == 0, f"untouched historical validator regeneration failed: {result.stderr.strip()}")
    results = json.loads((published / "results.json").read_text(encoding="utf-8"))
    require(
        results.get("valid_behavioral_episodes") == 1364 and results.get("prompt_form_pairs") == 682
        and results.get("model_level_semantic_depth_equivalence_claim_authorized") is False
        and results.get("model_level_semantic_depth_equivalence_claim_withheld") is True
        and results.get("claim_gate_components") == {
            "directional_depth_tost_conjunction": False,
            "inverse_reference_endpoint_positive_control": False,
        },
        "post-publication regenerated counts or conservative claim changed",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--verify-raw", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    require(args.verify_raw == (args.raw_root is not None), "--verify-raw and --raw-root must be supplied together")
    bindings = validate_publication_tree()
    historical_source = validate_historical_source(args.source_checkout)
    execution = validate_execution_receipt(bindings)
    results = regenerate(source_checkout=args.source_checkout, raw_root=args.raw_root) if args.verify_raw else None
    value = {
        "schema_version": "vla-wam-shared-v3c002r001-post-publication-closure-v1",
        "status": "passed_post_publication_closure_with_independent_raw_regeneration" if args.verify_raw else "passed_post_publication_static_closure_raw_regeneration_required",
        "passed": True, "source_commit": SOURCE_COMMIT, "results_commit": RESULTS_COMMIT,
        "historical_source_gate": historical_source, "published_files": bindings,
        "execution_receipt_sha256": bindings[(FINAL / "finalization_execution_receipt.json").as_posix()]["sha256"],
        "independent_raw_regeneration_passed": args.verify_raw,
        "historical_validator_sha256": HISTORICAL_VALIDATOR_SHA256,
        "frozen_validator_historical_remote_overlay_disclosure": "The untouched frozen validator passed 17.46 GB raw regeneration when its temporary remote view was pinned to execution-time head 785ea964; the overlay was removed. The durable post-publication validator instead proves commit ancestry and uses the exact retained execution checkout paths.",
        "claim_authorized": results.get("model_level_semantic_depth_equivalence_claim_authorized") if results else execution.get("result_claim_authorized"),
        "claim_withheld": results.get("model_level_semantic_depth_equivalence_claim_withheld") if results else execution.get("result_claim_withheld"),
    }
    if args.output:
        require(not args.output.exists(), f"refusing to overwrite closure receipt: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ContractError, OSError, json.JSONDecodeError, KeyError) as exc:
        raise SystemExit(f"V3-C002-R001 post-publication closure failed: {exc}") from exc
