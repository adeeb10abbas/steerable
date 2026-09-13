from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[3]
FORECAST_ROOT = REPO_ROOT / "workshops/corl2026_world_models/experiments/forecast_layout"
if str(FORECAST_ROOT) not in sys.path:
    sys.path.insert(0, str(FORECAST_ROOT))

import d1_development_block_jobs as development


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module(
    "build_d1_development_attempt003_release",
    FORECAST_ROOT / "build_d1_development_attempt003_release.py",
)
cluster_queue = _load_module(
    "d1_attempt003_cluster_queue_contract",
    REPO_ROOT / "workshops/corl2026_world_models/execution/20260912/autonomy/cluster_queue.py",
)


class D1DevelopmentAttempt003ReleaseTests(unittest.TestCase):
    def _historical_jobs(self) -> dict[str, dict]:
        relative = (
            "workshops/corl2026_world_models/execution/20260912/autonomy/"
            "cluster_queue.json"
        )
        result = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "show",
                f"{builder.BASE_RELEASE_COMMIT}:{relative}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        queue = json.loads(result.stdout)
        wanted = set(builder.BASE_DESCRIPTOR_SHA256)
        jobs = {job["job_id"]: job for job in queue["jobs"] if job["job_id"] in wanted}
        self.assertEqual(set(jobs), wanted)
        return jobs

    @staticmethod
    def _scrub_attempt_argv(argv: list[str]) -> list[str]:
        scrubbed: list[str] = []
        index = 0
        while index < len(argv):
            if argv[index] == "--pair-admission-timeout-seconds":
                index += 2
                continue
            value = argv[index]
            if value in {builder.BASE_SOURCE_COMMIT, builder.SOURCE_COMMIT}:
                value = "<source-commit>"
            if value.startswith("d1-development-d0"):
                value = re.sub(r"-00[23]\Z", "-<attempt>", value)
            scrubbed.append(value)
            index += 1
        return scrubbed

    @staticmethod
    def _option(argv: list[str], name: str) -> str:
        indices = [index for index, value in enumerate(argv) if value == name]
        if len(indices) != 1 or indices[0] + 1 >= len(argv):
            raise AssertionError(f"option is not singular: {name}")
        return argv[indices[0] + 1]

    def test_fragment_is_exactly_six_frozen_descriptor_only_jobs(self) -> None:
        fragment = builder.build_release_fragment()
        self.assertEqual(fragment["status"], "descriptor_only_not_dispatched")
        self.assertEqual(fragment["runtime_source_commit"], builder.SOURCE_COMMIT)
        self.assertEqual(fragment["maximum_new_behavioral_cells"], 12)
        self.assertEqual(fragment["descriptor_sha256"], builder.EXPECTED_DESCRIPTOR_SHA256)
        self.assertEqual(
            [job["job_id"] for job in fragment["jobs"]],
            [
                "d1-development-d01-server-003",
                "d1-development-d01-simulator-003",
                "d1-development-d02-server-003",
                "d1-development-d02-simulator-003",
                "d1-development-d04-server-003",
                "d1-development-d04-simulator-003",
            ],
        )
        self.assertEqual(
            [job["role"] for job in fragment["jobs"]],
            [builder.SERVER_ROLE, builder.SIMULATOR_ROLE] * 3,
        )
        serialized_jobs = json.dumps(fragment["jobs"], sort_keys=True)
        self.assertNotIn("d1-development-d03", serialized_jobs)
        self.assertNotRegex(serialized_jobs, r"d1-development-d0[124].*-002")
        for job in fragment["jobs"]:
            self.assertIs(job["released"], True)
            self.assertEqual(job["source_commit"], builder.SOURCE_COMMIT)
            self.assertEqual(job["max_wall_seconds"], 60000)
            self.assertEqual(job["publish_log_tail_bytes"], 8192)
            self.assertEqual(
                self._option(job["argv"], "--pair-admission-timeout-seconds"),
                "900",
            )
            self.assertEqual(self._option(job["argv"], "--source-root"), "{source_root}")
            self.assertEqual(self._option(job["argv"], "--job-dir"), "{job_dir}")

    def test_all_attempt002_scientific_bindings_are_preserved(self) -> None:
        historical = self._historical_jobs()
        fragment = builder.build_release_fragment()
        for current in fragment["jobs"]:
            mode = current["argv"][2]
            layout = self._option(current["argv"], "--layout-pair-id").lower()
            role = "server" if mode == "server-job" else "simulator"
            previous_id = f"d1-development-{layout}-{role}-002"
            previous = historical[previous_id]
            with self.subTest(job_id=current["job_id"]):
                self.assertEqual(
                    builder.descriptor_sha256(previous),
                    builder.BASE_DESCRIPTOR_SHA256[previous_id],
                )
                self.assertEqual(current["released"], previous["released"])
                self.assertEqual(current["role"], previous["role"])
                self.assertEqual(current["max_wall_seconds"], previous["max_wall_seconds"])
                self.assertEqual(
                    current["publish_log_tail_bytes"],
                    previous["publish_log_tail_bytes"],
                )
                self.assertEqual(
                    self._scrub_attempt_argv(current["argv"]),
                    self._scrub_attempt_argv(previous["argv"]),
                )

        release_path = (
            "workshops/corl2026_world_models/execution/20260912/"
            "d1_development_attempt002_release.json"
        )
        result = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "show",
                f"{builder.BASE_RELEASE_COMMIT}:{release_path}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        historical_release = json.loads(result.stdout)
        expected_preflights = {
            row["layout_pair_id"]: row
            for row in historical_release["prerequisite_preflights"]
        }
        self.assertEqual(builder.PREREQUISITE_PREFLIGHTS, expected_preflights)

    def test_hash_bytes_match_cluster_queue_normalization_exactly(self) -> None:
        for job in builder.build_release_fragment()["jobs"]:
            with self.subTest(job_id=job["job_id"]):
                self.assertEqual(
                    builder.normalized_descriptor(job),
                    cluster_queue.normalize_job(job),
                )
                self.assertEqual(
                    builder.descriptor_bytes(job),
                    cluster_queue.encode(cluster_queue.normalize_job(job)),
                )

    def test_runtime_validator_accepts_every_synthetic_queue_descriptor(self) -> None:
        fragment = builder.build_release_fragment()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            development.pilot, "verify_clean_git", return_value=None
        ):
            jobs_root = Path(temporary) / "control/jobs"
            for job in fragment["jobs"]:
                job_dir = jobs_root / job["job_id"]
                job_dir.mkdir(parents=True)
                normalized = cluster_queue.normalize_job(job)
                (job_dir / "descriptor.json").write_bytes(cluster_queue.encode(normalized))
                argv = job["argv"]
                mode = argv[2]
                layout = self._option(argv, "--layout-pair-id")
                paired_option = (
                    "--simulator-job-id" if mode == "server-job" else "--server-job-id"
                )
                block = development.load_development_block(REPO_ROOT, layout)
                with self.subTest(job_id=job["job_id"]):
                    identity = development.validate_queue_invocation(
                        source_root=REPO_ROOT,
                        job_dir=job_dir,
                        study_commit=builder.SOURCE_COMMIT,
                        job_id=job["job_id"],
                        expected_role=job["role"],
                        expected_mode=mode,
                        paired_job_id=self._option(argv, paired_option),
                        run_id=self._option(argv, "--run-id"),
                        block=block,
                        simulator_worker_role=builder.SIMULATOR_ROLE,
                        pair_admission_timeout_seconds=(
                            builder.PAIR_ADMISSION_TIMEOUT_SECONDS
                        ),
                    )
                    self.assertEqual(
                        identity["sha256"],
                        fragment["descriptor_sha256"][job["job_id"]],
                    )

    def test_builder_is_deterministic_and_does_not_edit_active_queue(self) -> None:
        active = (
            REPO_ROOT
            / "workshops/corl2026_world_models/execution/20260912/autonomy/"
            "cluster_queue.json"
        )
        before = active.read_bytes()
        first = builder.build_release_fragment()
        second = builder.build_release_fragment()
        self.assertEqual(first, second)
        self.assertEqual(active.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
