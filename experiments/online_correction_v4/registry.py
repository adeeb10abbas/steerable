"""Episode and execution-group registry loading from the planning manifest."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

from experiments.online_correction_v4.contracts import EpisodeManifestRow


def load_manifest_jsonl(path: Path) -> list[EpisodeManifestRow]:
    rows: list[EpisodeManifestRow] = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        import json

        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_no}: JSONL record must be an object")
        rows.append(EpisodeManifestRow.from_manifest_dict(payload))
    return rows


def load_manifest_from_config(config_path: Path) -> list[EpisodeManifestRow]:
    import importlib.util
    import json

    repo_root = config_path.parents[2]
    spec = importlib.util.spec_from_file_location(
        "online_correction_v4_planning",
        repo_root / "tools/online_correction_v4.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load planning helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads(config_path.read_text())
    config_sha = module.digest_bytes(config_path.read_bytes())
    manifest = module.build_manifest(config, config_sha)
    return [EpisodeManifestRow.from_manifest_dict(row) for row in manifest]


@dataclass
class ExecutionGroup:
    group_id: str
    policy: str
    fixture: str
    rows: list[EpisodeManifestRow] = field(default_factory=list)

    @property
    def episode_ids(self) -> list[str]:
        return [row.episode_id for row in self.rows]

    def ordered_rows(self) -> list[EpisodeManifestRow]:
        return sorted(self.rows, key=lambda row: row.execution_order)


@dataclass
class CampaignRegistry:
    rows: list[EpisodeManifestRow]
    config_sha256: str
    by_episode_id: dict[str, EpisodeManifestRow] = field(default_factory=dict)
    by_execution_group: dict[str, ExecutionGroup] = field(default_factory=dict)
    by_block: dict[str, list[EpisodeManifestRow]] = field(default_factory=dict)
    control_index: dict[str, EpisodeManifestRow] = field(default_factory=dict)

    @classmethod
    def from_rows(cls, rows: Iterable[EpisodeManifestRow], *, config_sha256: str) -> CampaignRegistry:
        materialized = list(rows)
        registry = cls(rows=materialized, config_sha256=config_sha256)
        groups: dict[str, list[EpisodeManifestRow]] = defaultdict(list)
        for row in materialized:
            if row.episode_id in registry.by_episode_id:
                raise ValueError(f"duplicate episode_id {row.episode_id}")
            registry.by_episode_id[row.episode_id] = row
            groups[row.execution_group].append(row)
            registry.by_block.setdefault(row.block_key, []).append(row)
            registry.control_index[row.episode_id] = row
        for group_id, group_rows in groups.items():
            policy, fixture = group_id.split(":", 1)
            registry.by_execution_group[group_id] = ExecutionGroup(
                group_id=group_id,
                policy=policy,
                fixture=fixture,
                rows=list(group_rows),
            )
        return registry

    @classmethod
    def from_manifest_path(cls, path: Path) -> CampaignRegistry:
        rows = load_manifest_jsonl(path)
        if not rows:
            raise ValueError("manifest is empty")
        return cls.from_rows(rows, config_sha256=rows[0].config_sha256)

    def get(self, episode_id: str) -> EpisodeManifestRow:
        return self.by_episode_id[episode_id]

    def resolve_controls(self, row: EpisodeManifestRow) -> list[EpisodeManifestRow]:
        return [self.by_episode_id[eid] for eid in row.reuse_episode_ids]

    def iter_groups(self) -> Iterator[ExecutionGroup]:
        for group_id in sorted(self.by_execution_group):
            yield self.by_execution_group[group_id]

    def families_present(self) -> set[str]:
        return {row.family for row in self.rows}

    def episodes_for_family(self, family_id: str) -> list[EpisodeManifestRow]:
        return [row for row in self.rows if row.family == family_id]

    def missing_controls(self, family_id: str) -> list[str]:
        missing: list[str] = []
        for row in self.episodes_for_family(family_id):
            for eid in row.reuse_episode_ids:
                if eid not in self.by_episode_id:
                    missing.append(eid)
        return missing

    def group_dependencies_satisfied(
        self,
        group: ExecutionGroup,
        *,
        accepted_episode_ids: set[str],
    ) -> tuple[bool, list[str]]:
        unsatisfied: list[str] = []
        for row in group.rows:
            for control_id in row.reuse_episode_ids:
                if control_id not in accepted_episode_ids:
                    unsatisfied.append(control_id)
        return (not unsatisfied, unsatisfied)
