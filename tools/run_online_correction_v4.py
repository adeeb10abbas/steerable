#!/usr/bin/env python3
"""Execute or validate one registered V4 DROID episode through the live adapter layer.

This entrypoint never fabricates receipts or runs models unless every frozen
binding is present and ``--dry-run`` / ``--validate-only`` are not set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "docs/online_correction_v4/campaign.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or validate one registered V4 online-correction episode through "
            "the DROID/RoboLab adapter layer."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Absolute path to the frozen planning manifest JSONL.",
    )
    parser.add_argument(
        "--runtime-lock",
        type=Path,
        required=True,
        help="Absolute path to the released runtime lock JSON.",
    )
    parser.add_argument(
        "--episode-id",
        required=True,
        help="Registered episode identifier from the manifest.",
    )
    parser.add_argument(
        "--attempt-id",
        required=True,
        help="Execution attempt identifier for write-once evidence.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Absolute output parent for attempt evidence.",
    )
    parser.add_argument(
        "--campaign-config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Campaign config used to bind config_sha256.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate bindings and print the launch plan without touching RoboLab or policy servers.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Alias for --dry-run.",
    )
    parser.add_argument(
        "--policy-host",
        default=None,
        help="Policy server host for live execution.",
    )
    parser.add_argument(
        "--policy-port",
        type=int,
        default=None,
        help="Policy server port for live execution.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    dry_run = args.dry_run or args.validate_only

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from experiments.online_correction_v4.contracts import TimingConfig
    from experiments.online_correction_v4.droid_contract import (
        DroidContractError,
        LaunchArgs,
        build_launch_plan,
        find_manifest_row,
        load_manifest_rows,
        sha256_bytes,
        sha256_file,
        validate_manifest_row_against_lock,
        validate_runtime_lock,
    )

    launch_args = LaunchArgs(
        manifest_path=args.manifest.resolve(),
        runtime_lock_path=args.runtime_lock.resolve(),
        episode_id=args.episode_id,
        attempt_id=args.attempt_id,
        output_dir=args.output.resolve(),
        dry_run=dry_run,
        validate_only=dry_run,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
    )
    try:
        plan = build_launch_plan(
            launch_args,
            study_root=ROOT,
            campaign_config_path=args.campaign_config.resolve(),
        )
    except DroidContractError as exc:
        print(f"[V4 DROID] blocked: {exc}", file=sys.stderr)
        return 2

    if dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if args.policy_host is None or args.policy_port is None:
        print(
            "[V4 DROID] blocked: --policy-host and --policy-port are required for live execution",
            file=sys.stderr,
        )
        return 2

    try:
        from experiments.online_correction_v4.droid_bindings import (
            build_episode_runner,
            build_live_binding,
            query_schedule_for_manifest,
            resolve_motion_direction,
            resolve_prompt_text,
        )

        config_raw = json.loads(args.campaign_config.read_text(encoding="utf-8"))
        config_sha = sha256_file(args.campaign_config.resolve())
        rows, manifest_sha = load_manifest_rows(launch_args.manifest_path)
        lock = validate_runtime_lock(
            launch_args.runtime_lock_path,
            expected_config_sha256=config_sha,
            expected_manifest_sha256=manifest_sha,
        )
        row = find_manifest_row(rows, launch_args.episode_id)
        manifest = validate_manifest_row_against_lock(row, lock=lock, manifest_sha256=manifest_sha)
        policy_binding = lock.policies[manifest.factors["policy"]]
        fixture_binding = lock.fixtures[manifest.fixture]
        prompt_text = resolve_prompt_text(manifest)
        prompt_sha256 = sha256_bytes(prompt_text.encode("utf-8"))
        timing = TimingConfig.from_mapping(config_raw["timing"])
        schedule = query_schedule_for_manifest(manifest)
        binding = build_live_binding(
            manifest=manifest,
            lock=lock,
            policy_binding=policy_binding,
            fixture_binding=fixture_binding,
            prompt_text=prompt_text,
            prompt_sha256=prompt_sha256,
            runtime_identity_sha256=lock.runner_sha256,
            timing=timing,
            schedule=schedule,
            policy_host=args.policy_host,
            policy_port=args.policy_port,
            output_dir=launch_args.output_dir,
        )
        fixture_config = config_raw.get("fixtures", {}).get(manifest.fixture, {})
        nominal_translation_m = fixture_config.get("nominal_translation_m")
        if (
            isinstance(nominal_translation_m, bool)
            or not isinstance(nominal_translation_m, (int, float))
            or float(nominal_translation_m) <= 0.0
        ):
            raise RuntimeError(
                f"campaign fixture {manifest.fixture!r} lacks a positive "
                "nominal_translation_m"
            )
        displacement_m = float(nominal_translation_m) * float(
            fixture_binding.calibration_scale
        )
        runner, _finalizer = build_episode_runner(
            binding,
            output_dir=launch_args.output_dir,
            attempt_id=launch_args.attempt_id,
            displacement_m=displacement_m,
            motion_direction=resolve_motion_direction(manifest),
            scenario=manifest.factors.get("scenario", "move_stop"),
            motion_config=config_raw["motion"],
        )
        try:
            result = runner.run()
        finally:
            from experiments.online_correction_v4.droid_robolab import close_live_droid_stack

            close_live_droid_stack(policy=binding.policy)
        print(
            json.dumps(
                {"status": result.attempt_status, "end_reason": result.end_reason.value},
                indent=2,
            )
        )
        return 0
    except DroidContractError as exc:
        print(f"[V4 DROID] blocked: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        from experiments.online_correction_v4.motion import MotionDirectionError

        if isinstance(exc, MotionDirectionError):
            print(f"[V4 DROID] blocked: {exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
