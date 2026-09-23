"""Isaac Lab entrypoint for the existing SGW-01 partition worker.

Release and partition validation happen before Isaac imports or model
construction.  The existing ``worker.run_partition`` remains the sole owner
of allocation checks, adapter creation, cell execution, recording, retries,
and cleanup.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--model", choices=("N3", "D1"), required=True)
    parser.add_argument("--family", choices=("LAT", "HEIGHT", "DIST"), required=True)
    parser.add_argument("--stage", choices=("P", "D", "C"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-valid-episodes", type=int, required=True)
    parser.add_argument("--max-cell-attempts", type=int, default=3)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    from robolab.eval.runner import add_common_eval_args
    from isaaclab.app import AppLauncher

    add_common_eval_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _preflight(args: argparse.Namespace) -> Any:
    from .contract import ContractError, load_release
    from . import worker

    release = load_release(args.release)
    cells = release.partition(args.model, args.family, args.stage)
    if args.max_valid_episodes != len(cells) or args.max_cell_attempts != 3:
        raise ContractError("partition limits must equal the frozen stage ceiling and three total attempts")
    if worker._stage_authorized(release, args.stage) is False:
        raise ContractError(f"stage {args.stage} is not authorized by the release")
    return release


def _configure_app(args: argparse.Namespace) -> Any:
    assigned = os.environ.get("SGW01_SIMULATOR_DEVICE", "").strip()
    if not assigned:
        raise RuntimeError("SGW01_SIMULATOR_DEVICE is required for the native simulator")
    required = ("device", "headless", "enable_cameras", "num_envs", "rendering_mode")
    if any(not hasattr(args, name) for name in required):
        raise RuntimeError("AppLauncher arguments lack the required SGW-01 renderer/device fields")
    from isaaclab.app import AppLauncher

    args.device = assigned
    args.headless = True
    args.enable_cameras = True
    args.num_envs = 1
    args.rendering_mode = "balanced"
    launcher = AppLauncher(args)
    app = getattr(launcher, "app", None)
    if app is None or not callable(getattr(app, "close", None)):
        raise RuntimeError("AppLauncher must expose an app.close() shutdown hook")
    return launcher


def run(args: argparse.Namespace) -> int:
    release = _preflight(args)
    launcher = _configure_app(args)
    try:
        from . import worker

        return worker.run_partition(
            release,
            model=args.model,
            family=args.family,
            stage=args.stage,
            max_valid=args.max_valid_episodes,
            max_attempts=args.max_cell_attempts,
            worker_id=f"{args.model}-{args.family}-{args.stage}-{os.getpid()}",
            heartbeat_seconds=args.heartbeat_seconds,
        )
    finally:
        app = getattr(launcher, "app", launcher)
        close = getattr(app, "close", None)
        if not callable(close):
            raise RuntimeError("AppLauncher app.close() is unavailable during shutdown")
        close()


def main() -> None:
    args = _parser().parse_args()
    try:
        code = run(args)
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        code = 44
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        code = 42
    raise SystemExit(code)


if __name__ == "__main__":
    main()
