#!/usr/bin/env python3
"""Run the frozen 12-call Cosmos-Reason2 visual-planning diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import transformers


LEFT = "Put the Rubik's cube to the left of the bowl."
RIGHT = "Put the Rubik's cube to the right of the bowl."
ROBOT_COT = (
    'You are given the task "{task_instruction}". Specify the 2D trajectory your '
    'end effector should follow in pixel space. Return the trajectory coordinates in '
    'JSON format like this: {{"point_2d": [x, y], "label": "gripper trajectory"}}.'
)
PIXELS_PER_TOKEN = 32**2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_candidates(text: str) -> list[str]:
    candidates = [text.strip()]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    return candidates


def _parse_output(text: str) -> dict[str, Any]:
    for candidate in _json_candidates(text):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            point = value.get("point_2d")
            if (
                isinstance(point, list)
                and len(point) == 2
                and all(isinstance(item, (int, float)) for item in point)
            ):
                return {
                    "json_parsed": True,
                    "point_2d": [float(point[0]), float(point[1])],
                    "label": value.get("label"),
                }
            return {"json_parsed": True, "point_2d": None, "parsed_json": value}
    return {"json_parsed": False, "point_2d": None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, action="append", required=True)
    parser.add_argument("--fixture-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if len(args.fixture) != 3 or len(args.fixture_manifest) != 3:
        parser.error("Exactly three fixtures and three fixture manifests are required")

    fixture_rows = []
    for fixture, manifest_path in zip(
        args.fixture, args.fixture_manifest, strict=True
    ):
        manifest = json.loads(manifest_path.read_text())
        if manifest["npz_sha256"] != _sha256(fixture):
            raise ValueError(f"Fixture hash mismatch: {fixture}")
        seed = int(manifest["environment_seed"])
        if seed not in (8300, 8301, 8302):
            raise ValueError(f"Unexpected fixture seed {seed}")
        with np.load(fixture, allow_pickle=False) as archive:
            exterior = np.asarray(archive["video.exterior_image_1_left"])
        if exterior.shape != (1, 1, 180, 320, 3) or exterior.dtype != np.uint8:
            raise ValueError(f"Unexpected exterior image: {exterior.shape}/{exterior.dtype}")
        fixture_rows.append(
            {
                "seed": seed,
                "fixture": fixture,
                "fixture_sha256": _sha256(fixture),
                "manifest": manifest_path,
                "manifest_sha256": _sha256(manifest_path),
                "image": Image.fromarray(exterior[0, 0], mode="RGB"),
            }
        )
    fixture_rows.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in fixture_rows] != [8300, 8301, 8302]:
        raise ValueError("Fixtures must cover seeds 8300-8302 exactly")

    transformers.set_seed(0)
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=True,
    )
    processor = transformers.Qwen3VLProcessor.from_pretrained(
        args.model, local_files_only=True
    )
    processor.image_processor.size = {
        "shortest_edge": 256 * PIXELS_PER_TOKEN,
        "longest_edge": 8192 * PIXELS_PER_TOKEN,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for fixture_row in fixture_rows:
        for relation, task_instruction in (("left", LEFT), ("right", RIGHT)):
            user_prompt = ROBOT_COT.format(task_instruction=task_instruction)
            for repeat in (0, 1):
                conversation = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": fixture_row["image"]},
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ]
                torch.manual_seed(fixture_row["seed"])
                inputs = processor.apply_chat_template(
                    conversation,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                    )
                trimmed = generated[:, inputs.input_ids.shape[1] :]
                raw_output = processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                output_path = (
                    args.output_dir
                    / f"seed{fixture_row['seed']}_{relation}_repeat{repeat}.txt"
                )
                output_path.write_text(raw_output)
                parsed = _parse_output(raw_output)
                records.append(
                    {
                        "environment_seed": fixture_row["seed"],
                        "relation": relation,
                        "repeat": repeat,
                        "task_instruction": task_instruction,
                        "user_prompt": user_prompt,
                        "fixture_sha256": fixture_row["fixture_sha256"],
                        "raw_output_path": str(output_path),
                        "raw_output_sha256": _sha256(output_path),
                        "raw_output": raw_output,
                        **parsed,
                    }
                )

    pair_rows = []
    for seed in (8300, 8301, 8302):
        rows = [row for row in records if row["environment_seed"] == seed]
        left = sorted((row for row in rows if row["relation"] == "left"), key=lambda row: row["repeat"])
        right = sorted((row for row in rows if row["relation"] == "right"), key=lambda row: row["repeat"])
        left_repeat_exact = left[0]["raw_output"] == left[1]["raw_output"]
        right_repeat_exact = right[0]["raw_output"] == right[1]["raw_output"]
        left_point = left[0]["point_2d"]
        right_point = right[0]["point_2d"]
        x_ordering = None
        if left_point is not None and right_point is not None:
            x_ordering = float(right_point[0] - left_point[0])
        pair_rows.append(
            {
                "environment_seed": seed,
                "left_repeat_raw_exact": left_repeat_exact,
                "right_repeat_raw_exact": right_repeat_exact,
                "left_point_2d": left_point,
                "right_point_2d": right_point,
                "right_minus_left_x_px": x_ordering,
                "requested_side_ordering_aligned": x_ordering is not None and x_ordering > 0,
            }
        )

    manifest = {
        "schema_version": "vla-wam-shared-v2-cosmos-reason2-static-diagnostic-v1",
        "status": "complete",
        "model_path": str(args.model),
        "model_config_sha256": _sha256(args.model / "config.json"),
        "model_weights_sha256": _sha256(args.model / "model.safetensors"),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "dtype": "float16",
            "attention": "sdpa",
            "system_prompt": "You are a helpful assistant.",
            "user_prompt_template": ROBOT_COT,
            "image_source": "video.exterior_image_1_left",
        },
        "fixtures": [
            {
                "environment_seed": row["seed"],
                "fixture_path": str(row["fixture"]),
                "fixture_sha256": row["fixture_sha256"],
                "manifest_path": str(row["manifest"]),
                "manifest_sha256": row["manifest_sha256"],
            }
            for row in fixture_rows
        ],
        "records": records,
        "pairs": pair_rows,
        "summary": {
            "call_count": len(records),
            "json_parse_count": sum(row["json_parsed"] for row in records),
            "point_parse_count": sum(row["point_2d"] is not None for row in records),
            "exact_repeat_condition_count": sum(
                int(row["left_repeat_raw_exact"]) + int(row["right_repeat_raw_exact"])
                for row in pair_rows
            ),
            "exact_repeat_condition_total": 6,
            "aligned_requested_side_pair_count": sum(
                row["requested_side_ordering_aligned"] for row in pair_rows
            ),
            "requested_side_pair_total": 3,
        },
        "claim_boundary": (
            "Text-output visual-planning diagnostic only: zero behavioral episodes, "
            "no robot-success rate, and no generated-future score."
        ),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
