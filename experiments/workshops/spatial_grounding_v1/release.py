"""Fail-closed SGW-01 release creation and concrete Kubernetes Job rendering."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Mapping

from .contract import (ContractError, REQUIRED_BINDING_FIELDS, STAGE_EPISODES, canonical_bytes,
                       load_json, sha256_file, validate_stage_authorizations)
from .recorder import atomic_json


_TOKEN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _queue_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def create_release(*, output: Path, release_id: str, protocol: Path, prompts: Path, planned_queue: Path,
                   fixtures: Path, runtime_binding: Path, resource_owner: str) -> Path:
    if output.exists():
        raise ContractError("release output must be a new directory")
    binding = load_json(runtime_binding, "runtime binding")
    missing = REQUIRED_BINDING_FIELDS - set(binding)
    if missing or binding.get("resource_owner") != resource_owner or "@sha256:" not in str(binding.get("worker_image_digest")):
        raise ContractError("runtime binding is incomplete, unowned, or uses a mutable image")
    fixture = load_json(fixtures, "fixture receipt")
    if fixture.get("status") != "qualified" or not fixture.get("fixture_sha256") or not fixture.get("time_map_sha256"):
        raise ContractError("fixtures require qualified hash-bound receipts")
    authorizations = validate_stage_authorizations(binding.get("stage_authorizations"))
    rows = _queue_rows(planned_queue)
    if len(rows) != 1044:
        raise ContractError("only the frozen 1044-cell queue can be released")
    output.mkdir(parents=True)
    for source, name in ((protocol, "protocol.json"), (prompts, "prompts.json"), (fixtures, "fixtures.json"), (runtime_binding, "runtime_binding.json")):
        shutil.copyfile(source, output / name)
    queue: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        if row.get("status") != "PLANNED_NOT_RELEASED":
            raise ContractError("release source queue must remain frozen and unreleased")
        row.update({"release_id": release_id, "status": "RELEASED",
                    "fixture_sha256": fixture["fixture_sha256"], "runtime_sha256": sha256_file(runtime_binding),
                    "time_map_sha256": fixture["time_map_sha256"]})
        queue.append(row)
    with (output / "queue.jsonl").open("wb") as stream:
        for row in queue:
            stream.write(canonical_bytes(row))
    receipt = {"schema_version": "sgw-01-release-v1", "release_id": release_id, "resource_owner": resource_owner,
               "source_queue_sha256": sha256_file(planned_queue), "cell_count": len(queue), "status": "released",
               "stage_authorizations": authorizations}
    atomic_json(output / "release_receipt.json", receipt)
    hashes = {name: sha256_file(output / name) for name in ("protocol.json", "prompts.json", "queue.jsonl", "fixtures.json", "runtime_binding.json", "release_receipt.json")}
    atomic_json(output / "hashes.json", hashes)
    return output


def render_job(*, release: Path, template: Path, output: Path, model: str, family: str, stage: str) -> Path:
    binding = load_json(Path(release) / "runtime_binding.json", "runtime binding")
    if model not in {"N3", "D1"} or family not in {"LAT", "HEIGHT", "DIST"} or stage not in STAGE_EPISODES:
        raise ContractError("invalid Job partition")
    limits = binding.get("cpu_memory_limits")
    if not isinstance(limits, dict) or not {"cpu_request", "memory_request", "cpu_limit", "memory_limit"}.issubset(limits):
        raise ContractError("runtime binding lacks concrete CPU/memory limits")
    gpu_counts = binding.get("model_gpu_counts")
    if not isinstance(gpu_counts, dict) or model not in gpu_counts:
        raise ContractError("runtime binding lacks model GPU count")
    values = {
        "JOB_NAME": f"sgw-01-{Path(release).name.lower()}-{model.lower()}-{family.lower()}-{stage.lower()}",
        "VERIFIED_NAMESPACE": binding["namespace"], "RELEASE_LABEL": Path(release).name,
        "VERIFIED_IMAGE_AT_SHA256_DIGEST": binding["worker_image_digest"],
        "VERIFIED_SOURCE_ROOT": binding.get("source_root", "/workspace/steerable"),
        "IMMUTABLE_RELEASE_PATH": str(Path(release).resolve()), "MODEL_ID": model, "FAMILY_ID": family, "STAGE_ID": stage,
        "PARTITION_EPISODES": str(STAGE_EPISODES[stage]), "VERIFIED_CPU_REQUEST": str(limits["cpu_request"]),
        "VERIFIED_MEMORY_REQUEST": str(limits["memory_request"]),
        "VERIFIED_CPU_LIMIT": str(limits["cpu_limit"]),
        "VERIFIED_MEMORY_LIMIT": str(limits["memory_limit"]),
        "VERIFIED_GPU_COUNT": str(gpu_counts[model]), "VERIFIED_PVC_MOUNT": binding["pvc_mount_path"],
        "VERIFIED_PVC_NAME": binding["pvc_name"],
    }
    text = Path(template).read_text(encoding="utf-8")
    rendered = _TOKEN.sub(lambda match: values.get(match.group(1), match.group(0)), text)
    if _TOKEN.search(rendered) or ":latest" in rendered:
        raise ContractError("Job rendering left unresolved or mutable fields")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--output", type=Path, required=True); create.add_argument("--release-id", required=True)
    create.add_argument("--protocol", type=Path, required=True); create.add_argument("--prompts", type=Path, required=True)
    create.add_argument("--planned-queue", type=Path, required=True); create.add_argument("--fixtures", type=Path, required=True)
    create.add_argument("--runtime-binding", type=Path, required=True); create.add_argument("--resource-owner", required=True)
    render = commands.add_parser("render-job")
    render.add_argument("--release", type=Path, required=True); render.add_argument("--template", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True); render.add_argument("--model", required=True)
    render.add_argument("--family", required=True); render.add_argument("--stage", required=True)
    args = parser.parse_args()
    result = (create_release(output=args.output, release_id=args.release_id, protocol=args.protocol, prompts=args.prompts,
              planned_queue=args.planned_queue, fixtures=args.fixtures, runtime_binding=args.runtime_binding,
              resource_owner=args.resource_owner) if args.command == "create" else
              render_job(release=args.release, template=args.template, output=args.output, model=args.model,
                         family=args.family, stage=args.stage))
    print(result)


if __name__ == "__main__":
    main()
