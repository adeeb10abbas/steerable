#!/usr/bin/env python3
"""Run the pinned GR00T server with auditable per-request sampling seeds.

The released N1.7 server forwards an ``options`` dictionary but the policy does
not consume it.  The v2 matched-language design needs LEFT and RIGHT replans to
use the same deterministic seed schedule, so this wrapper handles only the
``sampling_seed`` option and otherwise delegates to the official policy.
"""

from __future__ import annotations

import argparse
import random
from typing import Any

import numpy as np
import torch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
from gr00t.policy.server_client import PolicyServer


class SeededGr00tPolicy(Gr00tPolicy):
    """Honor one measurement-only seed option without changing policy inputs."""

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not options or "sampling_seed" not in options:
            return super()._get_action(observation, options)

        seed = int(options["sampling_seed"])
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                np.random.seed(seed % (2**32))
                random.seed(seed)
                action, info = super()._get_action(observation, options)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)

        info = dict(info)
        info["sampling_seed"] = seed
        return action, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--embodiment-tag",
        default="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    policy = SeededGr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(args.embodiment_tag),
        model_path=args.model_path,
        device=args.device,
        strict=True,
    )
    wrapped = Gr00tSimPolicyWrapper(policy)
    with PolicyServer(wrapped, host=args.host, port=args.port) as server:
        server.run()


if __name__ == "__main__":
    main()
