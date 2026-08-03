#!/usr/bin/env python3
"""Fail-fast readiness gate for the official GR00T N1.7 DROID server.

This command deliberately uses only the Python standard library.  It verifies
the single controlled-access prerequisite with a <=64 KiB config request and
never resolves model weights, constructs a policy, or starts a server.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


CHECKPOINT_ID = "nvidia/GR00T-N1.7-DROID"
COSMOS_ID = "nvidia/Cosmos-Reason2-2B"
COSMOS_CONFIG_URL = f"https://huggingface.co/{COSMOS_ID}/resolve/main/config.json"
EXPECTED_SERVER_SCRIPT = Path("gr00t/eval/run_gr00t_server.py")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check GR00T N1.7 DROID controlled access before any model download or server startup."
        )
    )
    parser.add_argument(
        "--isaac-groot-dir",
        type=Path,
        default=Path("/home/ali/projects/Isaac-GR00T"),
        help="Official Isaac-GR00T checkout to validate after gated access succeeds.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=15.0,
        help="Timeout for the small Cosmos config access probe.",
    )
    return parser


def _token() -> str | None:
    """Read an existing Hugging Face login without printing its secret."""

    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    token_path = hf_home / "token"
    try:
        value = token_path.read_text().strip()
    except FileNotFoundError:
        return None
    return value or None


def _result(status: str, *, detail: str, **extra: object) -> None:
    print(
        json.dumps(
            {
                "checkpoint": CHECKPOINT_ID,
                "cosmos_backbone": COSMOS_ID,
                "status": status,
                "detail": detail,
                **extra,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _probe_cosmos_access(timeout_seconds: float) -> tuple[bool, str]:
    headers = {"User-Agent": "steerable-groot-droid-preflight/1"}
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(COSMOS_CONFIG_URL, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            # The model is ~4.9 GB; intentionally cap this probe at a tiny config file.
            config = response.read(64 * 1024)
    except urllib.error.HTTPError as error:
        if error.code in {401, 403}:
            return False, f"Hugging Face returned HTTP {error.code} for gated Cosmos config"
        return False, f"Hugging Face returned HTTP {error.code} for Cosmos config"
    except urllib.error.URLError as error:
        return False, f"network error while probing Cosmos config: {error.reason}"

    if not config:
        return False, "Cosmos config response was empty"
    return True, "authenticated Cosmos config access confirmed; no model tensors were downloaded"


def main() -> int:
    args = _parser().parse_args()
    if sys.version_info[:2] != (3, 12):
        _result(
            "blocked_python",
            detail=(
                "The current official Isaac-GR00T runtime is pinned to CPython 3.12; rerun with "
                "`uv run --no-project --python 3.12 experiments/groot_droid/preflight.py`."
            ),
            detected_python=sys.version.split()[0],
        )
        return 10

    accessible, detail = _probe_cosmos_access(args.timeout_seconds)
    if not accessible:
        _result(
            "blocked_gated_cosmos_access",
            detail=detail,
            required_one_time_action=(
                "Use the Hugging Face account that has accepted access for "
                "https://huggingface.co/nvidia/Cosmos-Reason2-2B, then run `hf auth login` "
                "on this host."
            ),
            model_tensors_downloaded=False,
        )
        return 20

    server_script = args.isaac_groot_dir / EXPECTED_SERVER_SCRIPT
    if not server_script.is_file():
        _result(
            "blocked_official_server_checkout_missing",
            detail=(
                f"Cosmos access passed, but expected official server entrypoint is absent: {server_script}"
            ),
            next_command=(
                "git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git "
                f"{args.isaac_groot_dir}"
            ),
            model_tensors_downloaded=False,
        )
        return 30

    _result(
        "ready_for_server_contract_smoke",
        detail=(
            "Python 3.12, gated Cosmos access, and the official server entrypoint are present. "
            "No policy was loaded by this preflight."
        ),
        server_script=str(server_script),
        model_tensors_downloaded=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
