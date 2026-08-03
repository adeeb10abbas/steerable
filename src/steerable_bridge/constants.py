from __future__ import annotations

from dataclasses import dataclass


ANNOTATION_REPO = "Embodied-CoT/steering_features_bridge"
ANNOTATION_REVISION = "094f1f7259148e03619e73b45d7dff54995e7003"
LEROBOT_REPO = "IPEC-COMMUNITY/bridge_orig_lerobot"
LEROBOT_REVISION = "0e9d76d07e9df3ea3eba257b2520d4913833fad2"

DEFAULT_SEED = 20_260_723
DEFAULT_FPS = 5


@dataclass(frozen=True)
class HostedFile:
    repo: str
    revision: str
    path: str
    size: int
    local_group: str

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/datasets/{self.repo}/resolve/"
            f"{self.revision}/{self.path}"
        )


HOSTED_FILES = (
    HostedFile(
        ANNOTATION_REPO,
        ANNOTATION_REVISION,
        "traj_idx_key_map.json",
        17_149_202,
        "steering_features_bridge",
    ),
    HostedFile(
        ANNOTATION_REPO,
        ANNOTATION_REVISION,
        "step_to_subtask_dict.json",
        49_836_416,
        "steering_features_bridge",
    ),
    HostedFile(
        ANNOTATION_REPO,
        ANNOTATION_REVISION,
        "subtask_level_commands.json",
        81_107_672,
        "steering_features_bridge",
    ),
    HostedFile(
        ANNOTATION_REPO,
        ANNOTATION_REVISION,
        "rationales.json",
        84_532_755,
        "steering_features_bridge",
    ),
    HostedFile(
        LEROBOT_REPO,
        LEROBOT_REVISION,
        "meta/info.json",
        4_579,
        "bridge_orig_lerobot",
    ),
    HostedFile(
        LEROBOT_REPO,
        LEROBOT_REVISION,
        "meta/episodes.jsonl",
        4_351_502,
        "bridge_orig_lerobot",
    ),
    HostedFile(
        LEROBOT_REPO,
        LEROBOT_REVISION,
        "meta/tasks.jsonl",
        1_637_992,
        "bridge_orig_lerobot",
    ),
)


PILOT_SPLIT_COUNTS = {"train": 128, "validation": 32, "test": 32}
TARGET_SPLIT_COUNTS = {"train": 512, "validation": 64, "test": 128}
