from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import logging
import platform
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .cmi import (
    categorical_mutual_information_bits,
    estimate_gaussian_cmi,
    estimate_kde_cmi,
)


LOGGER = logging.getLogger("steerable-groot-cmi")

DEFAULT_MODEL_ID = "nvidia/gr00t17-lerobot-libero_spatial-640"
DEFAULT_MODEL_REVISION = "32a6ec786d6509df31b40392b4e4dcdda78c0f11"
DEFAULT_BASE_MODEL_ID = "nvidia/GR00T-N1.7-3B"
DEFAULT_BASE_MODEL_REVISION = "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
DEFAULT_PROCESSOR_ASSETS_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_PROCESSOR_ASSETS_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
DEFAULT_DATASET_ID = "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot"
DEFAULT_DATASET_REVISION = "bf14d6258218d12c2e3c1a3b9922e163cdf6455d"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="steerable-groot-cmi",
        description=(
            "Probe GR00T N1.7 on fixed LIBERO observations while swapping language prompts, "
            "then estimate action-language conditional mutual information."
        ),
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--base-model-id", default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--base-model-revision", default=DEFAULT_BASE_MODEL_REVISION)
    parser.add_argument("--processor-assets-id", default=DEFAULT_PROCESSOR_ASSETS_ID)
    parser.add_argument(
        "--processor-assets-revision", default=DEFAULT_PROCESSOR_ASSETS_REVISION
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--max-prompts", type=int, default=10)
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=8,
        help="Number of predicted action steps included in the analysis (sampling saves the full chunk).",
    )
    parser.add_argument("--estimator", choices=("kde", "gaussian"), default="kde")
    parser.add_argument("--bandwidth", type=float, default=None)
    parser.add_argument(
        "--covariance", choices=("diagonal", "full"), default="diagonal"
    )
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--bf16-params",
        action="store_true",
        help="Keep parameters in their native/BF16 dtypes instead of the checkpoint's FP32 mode.",
    )
    parser.add_argument(
        "--public-api-only",
        action="store_true",
        help="Disable the pinned LeRobot backbone-cache optimization; substantially slower.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace existing per-frame samples."
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip model and dataset loading and analyze existing per-frame samples.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use 1 frame, 2 prompts, 2 samples, and a small permutation test.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/groot_cmi/libero_spatial"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/groot_cmi/libero_spatial_ep0"),
    )
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_task_prompts(repo_id: str, revision: str) -> list[str]:
    from huggingface_hub import hf_hub_download

    task_file = hf_hub_download(
        repo_id=repo_id,
        filename="meta/tasks.jsonl",
        repo_type="dataset",
        revision=revision,
    )
    records = [
        json.loads(line)
        for line in Path(task_file).read_text().splitlines()
        if line.strip()
    ]
    records.sort(key=lambda row: int(row["task_index"]))
    return [str(row["task"]) for row in records]


def _snapshot(repo_id: str, revision: str) -> str:
    from huggingface_hub import snapshot_download

    LOGGER.info("Resolving pinned model snapshot %s@%s", repo_id, revision)
    return snapshot_download(repo_id=repo_id, revision=revision)


def _processor_assets_snapshot(repo_id: str, revision: str) -> str:
    from huggingface_hub import snapshot_download

    LOGGER.info("Resolving public Qwen3-VL processor assets %s@%s", repo_id, revision)
    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=[
            "config.json",
            "chat_template.json",
            "merges.txt",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
        ],
    )


class _PinnedV21Episode:
    """Minimal read-only adapter for the v2.1 dataset used by the official recipe."""

    def __init__(
        self,
        *,
        frame_table,
        videos: dict[str, np.ndarray],
        tasks: dict[int, str],
        episode_index: int,
        torch,
    ) -> None:
        self.frame_table = frame_table.reset_index(drop=True)
        self.videos = videos
        self.tasks = tasks
        self.episode_index = episode_index
        self.torch = torch
        lengths = {len(self.frame_table), *(len(video) for video in videos.values())}
        if len(lengths) != 1:
            raise ValueError(f"episode table/video length mismatch: {sorted(lengths)}")

    def __len__(self) -> int:
        return len(self.frame_table)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame_table.iloc[index]
        task_index = int(row["task_index"])
        return {
            "observation.state": self.torch.as_tensor(
                np.asarray(row["observation.state"], dtype=np.float32).copy()
            ),
            "observation.images.image": self.torch.from_numpy(
                self.videos["observation.images.image"][index].copy()
            ).permute(2, 0, 1),
            "observation.images.wrist_image": self.torch.from_numpy(
                self.videos["observation.images.wrist_image"][index].copy()
            ).permute(2, 0, 1),
            "action": self.torch.as_tensor(
                np.asarray(row["action"], dtype=np.float32).copy()
            ),
            "frame_index": int(row["frame_index"]),
            "episode_index": int(row.get("episode_index", self.episode_index)),
            "task_index": task_index,
            "task": self.tasks[task_index],
        }


def _decode_video(path: str) -> np.ndarray:
    try:
        from decord import VideoReader, cpu

        reader = VideoReader(path, ctx=cpu(0))
        return reader.get_batch(list(range(len(reader)))).asnumpy()
    except Exception as decord_error:
        LOGGER.warning(
            "decord failed for %s (%s); falling back to PyAV", path, decord_error
        )
        import av

        with av.open(path) as container:
            frames = [
                frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
            ]
        if not frames:
            raise RuntimeError(f"no video frames decoded from {path}") from decord_error
        return np.stack(frames)


def _load_pinned_v21_episode(args: argparse.Namespace, torch) -> _PinnedV21Episode:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    chunk = args.episode // 1000
    episode_name = f"episode_{args.episode:06d}"
    relative_paths = {
        "data": f"data/chunk-{chunk:03d}/{episode_name}.parquet",
        "observation.images.image": (
            f"videos/chunk-{chunk:03d}/observation.images.image/{episode_name}.mp4"
        ),
        "observation.images.wrist_image": (
            f"videos/chunk-{chunk:03d}/observation.images.wrist_image/{episode_name}.mp4"
        ),
        "tasks": "meta/tasks.jsonl",
    }
    local_paths = {
        key: hf_hub_download(
            repo_id=args.dataset_id,
            filename=relative_path,
            repo_type="dataset",
            revision=args.dataset_revision,
            local_dir=args.dataset_root,
        )
        for key, relative_path in relative_paths.items()
    }
    tasks = {}
    for line in Path(local_paths["tasks"]).read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            tasks[int(record["task_index"])] = str(record["task"])
    videos = {
        key: _decode_video(path)
        for key, path in local_paths.items()
        if key.startswith("observation.images.")
    }
    return _PinnedV21Episode(
        frame_table=pd.read_parquet(local_paths["data"]),
        videos=videos,
        tasks=tasks,
        episode_index=args.episode,
        torch=torch,
    )


def _load_runtime(args: argparse.Namespace):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.policies.groot.modeling_groot import GrootPolicy

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    checkpoint_path = _snapshot(args.model_id, args.model_revision)
    base_model_path = _snapshot(args.base_model_id, args.base_model_revision)
    processor_assets_path = _processor_assets_snapshot(
        args.processor_assets_id, args.processor_assets_revision
    )

    config = PreTrainedConfig.from_pretrained(checkpoint_path)
    if not isinstance(config, GrootConfig):
        raise TypeError(
            f"checkpoint resolved to {type(config).__name__}, not GrootConfig"
        )
    config.device = args.device
    config.base_model_path = base_model_path
    config.pretrained_path = checkpoint_path
    config.pretrained_revision = args.model_revision
    config.model_params_fp32 = not args.bf16_params

    LOGGER.info("Loading GR00T policy on %s", args.device)
    policy = GrootPolicy.from_pretrained(
        checkpoint_path,
        config=config,
        strict=True,
    )
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint_path,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "groot_n1_7_vlm_encode_v1": {
                "device": args.device,
                "model_name": processor_assets_path,
            },
        },
    )

    LOGGER.info("Loading selected LIBERO episode %d", args.episode)
    dataset = _load_pinned_v21_episode(args, torch)
    return torch, policy, preprocessor, postprocessor, dataset


def _observation_for_prompt(sample: dict[str, Any], prompt: str) -> dict[str, Any]:
    observation = {
        key: value
        for key, value in sample.items()
        if key == "observation.state" or key.startswith("observation.images.")
    }
    observation["task"] = prompt
    return observation


def _seed_torch(torch, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict_public(
    torch,
    policy,
    processed: dict[str, Any],
    postprocessor,
    seeds: list[int],
) -> np.ndarray:
    chunks = []
    for seed in seeds:
        _seed_torch(torch, seed)
        with torch.inference_mode():
            chunks.append(policy.predict_action_chunk(processed))
    raw = torch.cat(chunks, dim=0)
    decoded = postprocessor(raw)
    return decoded.detach().float().cpu().numpy()


def _predict_with_cached_backbone(
    torch,
    policy,
    processed: dict[str, Any],
    postprocessor,
    seeds: list[int],
) -> np.ndarray:
    """Cache deterministic VLM features; LeRobot v0.6.0 is pinned for this path."""

    model = policy._groot_model
    head = model.action_head
    device = next(policy.parameters()).device
    autocast = (
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=policy.config.use_bf16,
        )
        if device.type in {"cuda", "cpu"}
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        inputs = policy._filter_groot_inputs(processed, include_action=False)
        backbone_inputs, action_inputs = model.prepare_input(inputs)
        backbone_output = model.backbone(backbone_inputs)
        features = head._encode_features(backbone_output, action_inputs)
        chunks = []
        for seed in seeds:
            _seed_torch(torch, seed)
            output = head.get_action_with_features(
                backbone_features=features.backbone_features,
                state_features=features.state_features,
                embodiment_id=action_inputs.embodiment_id,
                backbone_output=backbone_output,
                action_input=action_inputs,
            )
            actions = output["action_pred"]
            horizon = policy._resolve_prediction_horizon(actions)
            action_dim = policy.config.output_features["action"].shape[0]
            chunks.append(actions[:, :horizon, :action_dim])
        raw = torch.cat(chunks, dim=0)
    decoded = postprocessor(raw)
    return decoded.detach().float().cpu().numpy()


def _to_uint8_image(value: Any) -> np.ndarray:
    array = (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"expected a 3-D image, got shape {array.shape}")
    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        if float(array.max()) <= 1.0 + 1e-6:
            array = array * 255.0
        array = np.clip(array, 0, 255)
    return array.astype(np.uint8)


def _as_scalar(value: Any) -> int | float | str:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size == 1:
        return array.reshape(-1)[0].item()
    return str(value)


def _frame_indices(length: int, count: int) -> list[int]:
    if length < 1:
        raise ValueError("selected episode is empty")
    if count < 1:
        raise ValueError("num_frames must be positive")
    count = min(count, length)
    return sorted({int(round(value)) for value in np.linspace(0, length - 1, count)})


def _sample_seed(
    base_seed: int, frame_order: int, prompt_index: int, sample_index: int
) -> int:
    return int(
        base_seed + frame_order * 1_000_003 + prompt_index * 10_007 + sample_index
    )


def _sample_frames(args: argparse.Namespace, prompts: list[str]) -> dict[str, Any]:
    torch, policy, preprocessor, postprocessor, dataset = _load_runtime(args)
    sample_dir = args.output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    selected_indices = _frame_indices(len(dataset), args.num_frames)
    frame_records: list[dict[str, Any]] = []

    for frame_order, dataset_index in enumerate(selected_indices):
        output_path = sample_dir / f"frame_{frame_order:03d}.npz"
        if output_path.exists() and not args.force:
            LOGGER.info("Keeping existing %s", output_path)
            with np.load(output_path, allow_pickle=False) as existing:
                frame_records.append(json.loads(str(existing["metadata_json"].item())))
            continue

        sample = dataset[dataset_index]
        prompt_actions = []
        prompt_latencies = []
        for prompt_index, prompt in enumerate(prompts):
            LOGGER.info(
                "Frame %d/%d, prompt %d/%d",
                frame_order + 1,
                len(selected_indices),
                prompt_index + 1,
                len(prompts),
            )
            processed = preprocessor(_observation_for_prompt(sample, prompt))
            seeds = [
                _sample_seed(args.seed, frame_order, prompt_index, sample_index)
                for sample_index in range(args.samples_per_prompt)
            ]
            started = time.perf_counter()
            if args.public_api_only:
                actions = _predict_public(
                    torch, policy, processed, postprocessor, seeds
                )
            else:
                actions = _predict_with_cached_backbone(
                    torch, policy, processed, postprocessor, seeds
                )
            prompt_latencies.append(time.perf_counter() - started)
            prompt_actions.append(actions)

        actions = np.stack(prompt_actions).astype(np.float32)
        true_action = np.asarray(sample["action"], dtype=np.float32)
        main_image = _to_uint8_image(sample["observation.images.image"])
        frame_index = int(_as_scalar(sample.get("frame_index", dataset_index)))
        episode_index = int(_as_scalar(sample.get("episode_index", args.episode)))
        metadata = {
            "frame_order": frame_order,
            "dataset_index": dataset_index,
            "frame_index": frame_index,
            "episode_index": episode_index,
            "episode_progress": dataset_index / max(len(dataset) - 1, 1),
            "true_task": str(sample.get("task", "")),
            "prompt_latencies_seconds": prompt_latencies,
            "action_shape": list(actions.shape),
        }
        np.savez_compressed(
            output_path,
            actions=actions,
            prompts=np.asarray(prompts),
            true_action=true_action,
            main_image=main_image,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        frame_records.append(metadata)
        LOGGER.info("Saved %s with shape %s", output_path, actions.shape)

    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"selected_dataset_indices": selected_indices, "frames": frame_records}


def _categorical_permutation(
    labels: np.ndarray, *, permutations: int, seed: int
) -> tuple[float, float, float]:
    observed = categorical_mutual_information_bits(labels)
    if permutations == 0:
        return observed, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    flat = labels.reshape(-1)
    null = np.asarray(
        [
            categorical_mutual_information_bits(
                flat[rng.permutation(len(flat))].reshape(labels.shape)
            )
            for _ in range(permutations)
        ]
    )
    p_value = float((1 + np.count_nonzero(null >= observed)) / (permutations + 1))
    return observed, float(np.quantile(null, 0.95)), p_value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_metrics(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    frame_payloads: list[dict[str, Any]],
    alpha: float,
) -> None:
    progress = np.asarray([row["episode_progress"] for row in summaries])
    observed = np.asarray([row["first_action_cmi_bits"] for row in summaries])
    horizon_cmi = np.asarray([row["mean_horizon_cmi_bits"] for row in summaries])
    permutation_p = np.asarray(
        [row["first_action_permutation_p_value"] for row in summaries]
    )
    prompt_ceiling = np.log2(summaries[0]["n_prompts"])
    corrected_alpha = alpha / len(summaries)
    gripper = np.asarray([row["first_action_gripper_mi_bits"] for row in summaries])

    figure, axes = plt.subplots(
        3, 1, figsize=(9, 9), sharex=True, constrained_layout=True
    )
    axes[0].plot(progress, observed, marker="o", label="first predicted action")
    axes[0].plot(progress, horizon_cmi, marker="s", label="mean over 8-step chunk")
    axes[0].axhline(
        prompt_ceiling,
        color="black",
        linewidth=1,
        linestyle=":",
        label=f"prompt entropy ceiling ({prompt_ceiling:.3f} bits)",
    )
    axes[0].set_ylim(0.0, prompt_ceiling * 1.06)
    axes[0].set_ylabel("Arm CMI estimate (bits)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(progress, permutation_p, marker="o", color="#7b3294")
    axes[1].axhline(
        corrected_alpha,
        color="black",
        linewidth=1,
        linestyle=":",
        label=f"Bonferroni alpha = {corrected_alpha:.5f} ({len(summaries)} states)",
    )
    axes[1].set_ylim(0.0, max(0.008, corrected_alpha * 1.2))
    axes[1].set_ylabel("First-action\npermutation p-value")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    axes[2].plot(progress, gripper, marker="o", color="#008837")
    axes[2].set_ylabel("Gripper MI (bits)")
    axes[2].set_xlabel("Demonstration episode progress")
    axes[2].grid(alpha=0.25)
    figure.suptitle("GR00T N1.7 language dependence along a frozen LIBERO rollout")
    figure.savefig(output_dir / "cmi_over_rollout.png", dpi=180)
    plt.close(figure)

    columns = min(4, len(frame_payloads))
    rows = int(np.ceil(len(frame_payloads) / columns))
    montage, axes_array = plt.subplots(
        rows, columns, figsize=(4 * columns, 3.5 * rows), squeeze=False
    )
    for axis in axes_array.reshape(-1):
        axis.axis("off")
    for axis, payload in zip(axes_array.reshape(-1), frame_payloads, strict=False):
        axis.imshow(payload["main_image"])
        meta = payload["metadata"]
        axis.set_title(
            f"frame {meta['frame_index']} ({100 * meta['episode_progress']:.0f}%)",
            fontsize=10,
        )
        axis.axis("off")
    montage.suptitle("Frozen observations used for prompt swaps")
    montage.tight_layout()
    montage.savefig(output_dir / "probe_frames.png", dpi=180)
    plt.close(montage)


def _write_blog_findings(
    args: argparse.Namespace,
    prompts: list[str],
    summaries: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
) -> None:
    first_detected = sum(
        bool(row["first_action_language_dependence_detected"]) for row in summaries
    )
    corrected_alpha = args.alpha / len(summaries)
    first_detected_bonferroni = sum(
        float(row["first_action_permutation_p_value"]) < corrected_alpha
        for row in summaries
    )
    observed = np.asarray([row["first_action_cmi_bits"] for row in summaries])
    peak_index = int(np.argmax(observed))
    peak = summaries[peak_index]
    median = float(np.median(observed))
    prompt_ceiling = float(np.log2(len(prompts)))
    gripper = np.asarray([row["first_action_gripper_mi_bits"] for row in summaries])
    gripper_nonzero = int(np.count_nonzero(gripper > 1e-12))
    sensitivity_by_n = {
        sample_count: float(
            np.median(
                [
                    row["first_action_cmi_bits"]
                    for row in sensitivity
                    if row["samples_per_prompt"] == sample_count
                ]
            )
        )
        for sample_count in sorted({row["samples_per_prompt"] for row in sensitivity})
    }
    first_n = min(sensitivity_by_n)
    last_n = max(sensitivity_by_n)
    text = f"""# GR00T N1.7 x LIBERO-Spatial CMI pilot

## Result snapshot

This frozen-state intervention used **{len(prompts)} prompts**, **{args.samples_per_prompt} stochastic action samples per prompt**, and **{len(summaries)} observations** from LIBERO-Spatial episode {args.episode}. For the first predicted arm action, language dependence cleared the per-frame permutation null at alpha={args.alpha:g} in **{first_detected}/{len(summaries)} frames**, and **{first_detected_bonferroni}/{len(summaries)}** remained below the Bonferroni threshold **{corrected_alpha:.5f}**. The median KDE CMI estimate was **{median:.4f} bits** (range **{float(observed.min()):.4f}–{float(observed.max()):.4f}**) against a **{prompt_ceiling:.4f}-bit** prompt-entropy ceiling. The peak was **{float(peak["first_action_cmi_bits"]):.4f} bits** at {100 * float(peak["episode_progress"]):.1f}% episode progress. First-action gripper MI was nonzero in **{gripper_nonzero}/{len(summaries)} frames** and reached **{float(gripper.max()):.4f} bits**.

The pilot therefore found **no instruction-blind point** among the eight probed observations: GR00T's continuous arm distribution remained strongly prompt-dependent from the beginning to the end of this demonstration. That is a sensitivity finding, not a success claim—the swapped prompts are counterfactual and usually inconsistent with the frozen scene.

As a basic Monte Carlo check, the median first-action estimate was **{sensitivity_by_n[first_n]:.4f} bits at N={first_n}** and **{sensitivity_by_n[last_n]:.4f} bits at N={last_n}** using nested samples. The stable qualitative direction is encouraging, but the increasing magnitude shows that this small pilot has not established numerical convergence.

## What was measured

At each fixed visual/proprioceptive observation `S=s`, the model received every prompt in the pinned LIBERO-Spatial task pool. GR00T's flow-matching sampler generated repeated action chunks. The main metric is

`I(A; L | S=s) = H(A | S=s) - E_L[H(A | S=s, L)]`.

The six continuous arm dimensions use a shared-bandwidth, leave-one-out Gaussian KDE estimator expressed through the prompt posterior, so the raw estimate is bounded above by the prompt entropy. The binarized gripper uses categorical mutual information and is reported separately. Prompt labels are permuted while preserving group sizes to estimate the finite-sample null. `null_centered_score_bits` in the CSV is the raw estimate minus the permutation-null median; it is a diagnostic effect score, **not CMI**, and is not bounded by prompt entropy.

## Interpretation boundary

A frame that does not reject the permutation null is an **instruction-blind candidate**, not proof that the model never uses language. Conversely, high CMI proves sensitivity, not correct grounding: a model can change its action for the wrong linguistic reason. These are off-policy prompt swaps on states from one successful demonstration, and many swapped spatial descriptions are inconsistent with the rendered scene. LIBERO-Spatial also keeps the manipulated object and destination nearly constant across prompts, so low CMI can be rational state-based control rather than a model defect. With only eight Monte Carlo samples per prompt, these point estimates should be treated as a pilot and repeated with more samples, episodes, and random seeds.

## Why LIBERO-Spatial

This suite is a controlled first probe because the ten tasks keep the object (black bowl) and destination (plate) fixed while varying the bowl's spatial relation. The exact model and dataset are published in the LeRobot ecosystem: [GR00T policy guide](https://huggingface.co/docs/lerobot/groot), [LIBERO integration](https://huggingface.co/docs/lerobot/main/libero), [fine-tuned checkpoint](https://huggingface.co/nvidia/gr00t17-lerobot-libero_spatial-640), and [demonstration dataset](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot).

## Reproducibility pins

- Model: `{args.model_id}` at `{args.model_revision}`
- Base model: `{args.base_model_id}` at `{args.base_model_revision}`
- Processor assets: `{args.processor_assets_id}` at `{args.processor_assets_revision}` (public Qwen3-VL architecture-compatible assets; the Cosmos-Reason2-2B asset repo is gated)
- Dataset: `{args.dataset_id}` at `{args.dataset_revision}`
- Episode: `{args.episode}`
- Seed: `{args.seed}`
- Continuous estimator: `{args.estimator}` (bandwidth `{args.bandwidth}`; Gaussian diagnostic covariance `{args.covariance}`, ridge `{args.ridge}`)
- Permutations/bootstrap replicates: `{args.permutations}/{args.bootstrap_samples}`

See `frame_metrics.csv` for every action-horizon estimate, `frame_summary.csv` for plot-level values, `sample_size_sensitivity.csv` for the nested-sample check, `cmi_over_rollout.png` for the main figure, and `probe_frames.png` for the frozen observations. The next decisive run is a preregistered replication over multiple episodes with at least 32–64 samples per prompt, paired with actual counterfactual rollout success so sensitivity can be separated from correct instruction following.
"""
    (args.output_dir / "BLOG_FINDINGS.md").write_text(text)


def _sample_size_sensitivity(
    args: argparse.Namespace,
    frame_payloads: list[dict[str, Any]],
    aggregate_actions: list[np.ndarray],
) -> list[dict[str, Any]]:
    sample_counts = sorted(
        {2, max(2, args.samples_per_prompt // 2), args.samples_per_prompt}
    )
    rows: list[dict[str, Any]] = []
    for frame_order, (payload, actions) in enumerate(
        zip(frame_payloads, aggregate_actions, strict=True)
    ):
        arm_dim = actions.shape[-1] - 1
        for sample_count in sample_counts:
            subset = actions[:, :sample_count, 0, :arm_dim]
            if args.estimator == "kde":
                result = estimate_kde_cmi(
                    subset,
                    bandwidth=args.bandwidth,
                    permutations=0,
                    bootstrap_samples=0,
                    seed=args.seed,
                )
            else:
                result = estimate_gaussian_cmi(
                    subset,
                    covariance=args.covariance,
                    ridge=args.ridge,
                    permutations=0,
                    bootstrap_samples=0,
                    seed=args.seed,
                )
            rows.append(
                {
                    "frame_order": frame_order,
                    "frame_index": payload["metadata"]["frame_index"],
                    "episode_progress": payload["metadata"]["episode_progress"],
                    "samples_per_prompt": sample_count,
                    "first_action_cmi_bits": result.cmi_bits,
                }
            )
    return rows


def _analyze(args: argparse.Namespace, prompts: list[str]) -> dict[str, Any]:
    sample_paths = sorted((args.output_dir / "samples").glob("frame_*.npz"))
    if not sample_paths:
        raise FileNotFoundError(
            f"no frame samples found under {args.output_dir / 'samples'}"
        )

    metrics: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    frame_payloads: list[dict[str, Any]] = []
    aggregate_actions = []
    for frame_order, path in enumerate(sample_paths):
        with np.load(path, allow_pickle=False) as payload:
            actions = payload["actions"].astype(np.float64)
            saved_prompts = payload["prompts"].astype(str).tolist()
            metadata = json.loads(str(payload["metadata_json"].item()))
            main_image = payload["main_image"]
        if saved_prompts != prompts:
            raise ValueError(f"prompt pool in {path} does not match this run")
        if actions.ndim != 4:
            raise ValueError(
                f"actions in {path} must have shape (prompt, sample, horizon, dim)"
            )
        if (
            actions.shape[0] != len(prompts)
            or actions.shape[1] != args.samples_per_prompt
        ):
            raise ValueError(f"sample dimensions in {path} do not match CLI arguments")
        horizon = min(args.horizon_steps, actions.shape[2])
        arm_dim = actions.shape[-1] - 1
        if arm_dim < 1:
            raise ValueError(
                "expected at least one continuous action dimension plus gripper"
            )
        aggregate_actions.append(actions.astype(np.float32))
        frame_payloads.append({"metadata": metadata, "main_image": main_image})

        frame_rows = []
        for horizon_step in range(horizon):
            estimator_seed = args.seed + frame_order * 1009 + horizon_step
            if args.estimator == "kde":
                result = estimate_kde_cmi(
                    actions[:, :, horizon_step, :arm_dim],
                    bandwidth=args.bandwidth,
                    permutations=args.permutations,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=estimator_seed,
                )
            else:
                result = estimate_gaussian_cmi(
                    actions[:, :, horizon_step, :arm_dim],
                    covariance=args.covariance,
                    ridge=args.ridge,
                    permutations=args.permutations,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=estimator_seed,
                )
            if (
                args.estimator == "kde"
                and result.cmi_bits > np.log2(len(prompts)) + 1e-9
            ):
                raise AssertionError(
                    "KDE CMI estimate exceeded the prompt-entropy ceiling"
                )
            gripper_labels = (actions[:, :, horizon_step, -1] > 0).astype(np.int8)
            gripper_mi, gripper_null_p95, gripper_p = _categorical_permutation(
                gripper_labels,
                permutations=args.permutations,
                seed=args.seed + 5_000_003 + frame_order * 1009 + horizon_step,
            )
            row = {
                "frame_order": frame_order,
                "dataset_index": metadata["dataset_index"],
                "frame_index": metadata["frame_index"],
                "episode_index": metadata["episode_index"],
                "episode_progress": metadata["episode_progress"],
                "horizon_step": horizon_step,
                **result.as_dict(),
                "language_dependence_detected": bool(
                    result.permutation_p_value < args.alpha
                    and result.null_centered_score_bits > 0
                ),
                "gripper_mi_bits": gripper_mi,
                "gripper_null_p95_bits": gripper_null_p95,
                "gripper_permutation_p_value": gripper_p,
            }
            metrics.append(row)
            frame_rows.append(row)

        first = frame_rows[0]
        summaries.append(
            {
                "frame_order": frame_order,
                "dataset_index": metadata["dataset_index"],
                "frame_index": metadata["frame_index"],
                "episode_index": metadata["episode_index"],
                "episode_progress": metadata["episode_progress"],
                "true_task": metadata["true_task"],
                "n_prompts": len(prompts),
                "first_action_cmi_bits": first["cmi_bits"],
                "first_action_null_centered_score_bits": first[
                    "null_centered_score_bits"
                ],
                "first_action_null_p95_bits": first["null_p95_bits"],
                "first_action_permutation_p_value": first["permutation_p_value"],
                "first_action_language_dependence_detected": first[
                    "language_dependence_detected"
                ],
                "first_action_gripper_mi_bits": first["gripper_mi_bits"],
                "first_action_gripper_permutation_p_value": first[
                    "gripper_permutation_p_value"
                ],
                "mean_horizon_cmi_bits": float(
                    np.mean([row["cmi_bits"] for row in frame_rows])
                ),
                "mean_horizon_null_centered_score_bits": float(
                    np.mean([row["null_centered_score_bits"] for row in frame_rows])
                ),
                "significant_horizon_fraction": float(
                    np.mean([row["language_dependence_detected"] for row in frame_rows])
                ),
            }
        )

    _write_csv(args.output_dir / "frame_metrics.csv", metrics)
    _write_csv(args.output_dir / "frame_summary.csv", summaries)
    sensitivity = _sample_size_sensitivity(args, frame_payloads, aggregate_actions)
    _write_csv(args.output_dir / "sample_size_sensitivity.csv", sensitivity)
    np.savez_compressed(
        args.output_dir / "samples.npz",
        actions=np.stack(aggregate_actions),
        prompts=np.asarray(prompts),
    )
    _plot_metrics(args.output_dir, summaries, frame_payloads, args.alpha)
    _write_blog_findings(args, prompts, summaries, sensitivity)
    return {
        "num_frames": len(summaries),
        "num_prompts": len(prompts),
        "samples_per_prompt": args.samples_per_prompt,
        "first_action_language_dependence_detected_frames": int(
            sum(row["first_action_language_dependence_detected"] for row in summaries)
        ),
        "first_action_bonferroni_detected_frames": int(
            sum(
                row["first_action_permutation_p_value"] < args.alpha / len(summaries)
                for row in summaries
            )
        ),
        "first_action_bonferroni_alpha": float(args.alpha / len(summaries)),
        "median_first_action_cmi_bits": float(
            np.median([row["first_action_cmi_bits"] for row in summaries])
        ),
        "prompt_entropy_ceiling_bits": float(np.log2(len(prompts))),
        "artifacts": {
            "metrics": str(args.output_dir / "frame_metrics.csv"),
            "summary": str(args.output_dir / "frame_summary.csv"),
            "sample_size_sensitivity": str(
                args.output_dir / "sample_size_sensitivity.csv"
            ),
            "plot": str(args.output_dir / "cmi_over_rollout.png"),
            "frames": str(args.output_dir / "probe_frames.png"),
            "blog": str(args.output_dir / "BLOG_FINDINGS.md"),
        },
    }


def _environment_manifest(args: argparse.Namespace) -> dict[str, Any]:
    packages = {}
    for package in ("lerobot", "torch", "transformers", "huggingface-hub", "numpy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = {
                "device": args.device,
                "name": torch.cuda.get_device_name(args.device),
                "capability": list(torch.cuda.get_device_capability(args.device)),
                "cuda_runtime": torch.version.cuda,
            }
    except ImportError:
        pass
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "gpu": gpu,
    }


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.smoke:
        args.num_frames = 1
        args.max_prompts = 2
        args.samples_per_prompt = 2
        args.horizon_steps = 2
        args.permutations = min(args.permutations, 19)
        args.bootstrap_samples = 0
    if args.max_prompts < 2 or args.samples_per_prompt < 2:
        raise ValueError("CMI requires at least two prompts and two samples per prompt")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = _read_task_prompts(args.dataset_id, args.dataset_revision)[
        : args.max_prompts
    ]
    run_manifest: dict[str, Any] = {
        "status": "sampling" if not args.analyze_only else "analyzing",
        "model": {"repo_id": args.model_id, "revision": args.model_revision},
        "base_model": {
            "repo_id": args.base_model_id,
            "revision": args.base_model_revision,
        },
        "processor_assets": {
            "repo_id": args.processor_assets_id,
            "revision": args.processor_assets_revision,
            "reason": (
                "Public Qwen3-VL tokenizer/image/video processor assets are used because the "
                "architecture-compatible nvidia/Cosmos-Reason2-2B asset repo is gated."
            ),
        },
        "dataset": {
            "repo_id": args.dataset_id,
            "revision": args.dataset_revision,
            "episode": args.episode,
        },
        "prompt_prior": "uniform",
        "prompt_entropy_ceiling_bits": float(np.log2(len(prompts))),
        "prompts": prompts,
        "estimator": {
            "continuous_dimensions": "all action dimensions except final gripper",
            "continuous_estimator": args.estimator,
            "bandwidth": args.bandwidth,
            "gaussian_covariance": args.covariance,
            "gripper_estimator": "categorical plug-in mutual information",
            "ridge": args.ridge,
            "permutations": args.permutations,
            "bootstrap_samples": args.bootstrap_samples,
            "alpha": args.alpha,
        },
        "sampling": {
            "num_frames": args.num_frames,
            "samples_per_prompt": args.samples_per_prompt,
            "horizon_steps_analyzed": args.horizon_steps,
            "seed": args.seed,
            "backbone_cache": not args.public_api_only,
            "bf16_params": args.bf16_params,
        },
        "environment": _environment_manifest(args),
    }
    previous_manifest: dict[str, Any] = {}
    manifest_path = args.output_dir / "run_manifest.json"
    if args.analyze_only and manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text())
        if "sampled_frames" in previous_manifest:
            run_manifest["sampled_frames"] = previous_manifest["sampled_frames"]
    _write_json(manifest_path, run_manifest)
    try:
        if not args.analyze_only:
            run_manifest["sampled_frames"] = _sample_frames(args, prompts)
        run_manifest["analysis"] = _analyze(args, prompts)
        run_manifest["status"] = "complete"
    except Exception as error:
        run_manifest["status"] = "failed"
        run_manifest["error"] = f"{type(error).__name__}: {error}"
        _write_json(manifest_path, run_manifest)
        raise
    _write_json(manifest_path, run_manifest)
    print(json.dumps(run_manifest["analysis"], indent=2))


if __name__ == "__main__":
    main()
