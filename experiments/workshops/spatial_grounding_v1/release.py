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
                       load_json, load_release, sha256_file, validate_stage_authorizations)
from .recorder import atomic_json


_TOKEN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _queue_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def create_release(*, output: Path, release_id: str, protocol: Path, prompts: Path, planned_queue: Path,
                   fixtures: Path, runtime_binding: Path, resource_owner: str, stage: str,
                   model: str, family: str) -> Path:
    if output.exists():
        raise ContractError("release output must be a new directory")
    if stage not in STAGE_EPISODES:
        raise ContractError("release stage must be P, D, or C")
    if model not in {"N3", "D1"} or family not in {"LAT", "HEIGHT", "DIST"}:
        raise ContractError("release must select one registered model/family branch")
    binding = load_json(runtime_binding, "runtime binding")
    missing = REQUIRED_BINDING_FIELDS - set(binding)
    if (missing or any(binding.get(key) in (None, "", {}, []) for key in REQUIRED_BINDING_FIELDS)
            or binding.get("resource_owner") != resource_owner or "@sha256:" not in str(binding.get("worker_image_digest"))):
        raise ContractError("runtime binding is incomplete, unowned, or uses a mutable image")
    fixture = load_json(fixtures, "fixture receipt")
    layouts = fixture.get("layouts")
    time_maps = fixture.get("time_maps")
    if fixture.get("status") != "qualified" or not isinstance(layouts, dict) or not isinstance(time_maps, dict):
        raise ContractError("fixtures require qualified per-layout and per-model hash-bound receipts")
    authorizations = binding.get("stage_authorizations")
    validate_stage_authorizations(authorizations, stage)
    rows = _queue_rows(planned_queue)
    if len(rows) != 1044:
        raise ContractError("only the frozen 1044-cell queue can be released")
    output.mkdir(parents=True)
    for source, name in ((protocol, "protocol.json"), (prompts, "prompts.json"), (fixtures, "fixtures.json"), (runtime_binding, "runtime_binding.json")):
        shutil.copyfile(source, output / name)
    queue: list[dict[str, Any]] = []
    for row in rows:
        if (row["stage"], row["model"], row["family"]) != (stage, model, family):
            continue
        row = dict(row)
        if row.get("status") != "PLANNED_NOT_RELEASED":
            raise ContractError("release source queue must remain frozen and unreleased")
        layout = layouts.get(row["layout_id"])
        fixture_hash = layout.get("fixture_sha256") if isinstance(layout, dict) else None
        time_hash = time_maps.get(model)
        if not isinstance(fixture_hash, str) or not fixture_hash or not isinstance(time_hash, str) or not time_hash:
            raise ContractError(f"unqualified layout/runtime binding for {row['layout_id']}/{model}")
        row.update({"release_id": release_id, "status": "RELEASED",
                    "fixture_sha256": fixture_hash, "runtime_sha256": sha256_file(runtime_binding),
                    "time_map_sha256": time_hash})
        queue.append(row)
    if len(queue) != STAGE_EPISODES[stage]:
        raise ContractError("selected branch does not match its frozen stage ceiling")
    with (output / "queue.jsonl").open("wb") as stream:
        for row in queue:
            stream.write(canonical_bytes(row))
    receipt = {"schema_version": "sgw-01-release-v1", "release_id": release_id, "resource_owner": resource_owner,
               "source_queue_sha256": sha256_file(planned_queue), "stage": stage, "model": model, "family": family,
               "cell_count": len(queue), "status": "released",
               "stage_authorizations": authorizations}
    atomic_json(output / "release_receipt.json", receipt)
    hashes = {name: sha256_file(output / name) for name in ("protocol.json", "prompts.json", "queue.jsonl", "fixtures.json", "runtime_binding.json", "release_receipt.json")}
    atomic_json(output / "hashes.json", hashes)
    return output


def render_job(*, release: Path, template: Path, output: Path, model: str, family: str, stage: str) -> Path:
    loaded = load_release(release)
    loaded.partition(model, family, stage)
    binding = loaded.binding
    if model not in {"N3", "D1"} or family not in {"LAT", "HEIGHT", "DIST"} or stage not in STAGE_EPISODES:
        raise ContractError("invalid Job partition")
    limits = binding.get("cpu_memory_limits")
    if not isinstance(limits, dict) or not {"cpu_request", "memory_request", "cpu_limit", "memory_limit"}.issubset(limits):
        raise ContractError("runtime binding lacks concrete CPU/memory limits")
    gpu_counts = binding.get("model_gpu_counts")
    if not isinstance(gpu_counts, dict) or model not in gpu_counts:
        raise ContractError("runtime binding lacks model GPU count")
    if sum(int(value) for value in gpu_counts.values()) > 4 or int(gpu_counts[model]) < 1:
        raise ContractError("runtime binding exceeds the SGW-01 four-GPU ceiling")
    source_root = binding.get("source_root")
    if not isinstance(source_root, str) or not source_root:
        raise ContractError("runtime binding lacks verified source_root")
    values = {
        "JOB_NAME": f"sgw-01-{Path(release).name.lower()}-{model.lower()}-{family.lower()}-{stage.lower()}",
        "VERIFIED_NAMESPACE": binding["namespace"], "RELEASE_LABEL": Path(release).name,
        "VERIFIED_IMAGE_AT_SHA256_DIGEST": binding["worker_image_digest"],
        "VERIFIED_SOURCE_ROOT": source_root,
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
    create.add_argument("--stage", choices=("P", "D", "C"), required=True)
    create.add_argument("--model", choices=("N3", "D1"), required=True)
    create.add_argument("--family", choices=("LAT", "HEIGHT", "DIST"), required=True)
    render = commands.add_parser("render-job")
    render.add_argument("--release", type=Path, required=True); render.add_argument("--template", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True); render.add_argument("--model", required=True)
    render.add_argument("--family", required=True); render.add_argument("--stage", required=True)
    args = parser.parse_args()
    result = (create_release(output=args.output, release_id=args.release_id, protocol=args.protocol, prompts=args.prompts,
              planned_queue=args.planned_queue, fixtures=args.fixtures, runtime_binding=args.runtime_binding,
              resource_owner=args.resource_owner, stage=args.stage, model=args.model, family=args.family) if args.command == "create" else
              render_job(release=args.release, template=args.template, output=args.output, model=args.model,
                         family=args.family, stage=args.stage))
    print(result)


if __name__ == "__main__":
    main()
