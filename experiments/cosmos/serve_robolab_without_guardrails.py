"""Launch the official Cosmos RoboLab policy server without safety guardrails.

The released RoboLab server does not expose ``GuardrailOverrides.guardrails``
on its command line, so it otherwise downloads the separately gated
Cosmos-Guardrail1 repository before loading a policy.  This wrapper changes
only that setup flag; policy checkpoint loading, transforms, sampling, prompt
formatting, action output, and decoded-video output stay on the pinned official
server implementation.
"""

from __future__ import annotations

from typing import Any

from cosmos_framework.scripts import action_policy_server_robolab as server


_official_build_setup_args = server.RobolabPolicyService._build_setup_args


def _build_setup_args_without_guardrails(
    self: server.RobolabPolicyService,
    args: server.RobolabServerArgs,
) -> Any:
    setup_args = _official_build_setup_args(self, args)
    return setup_args.model_copy(update={"guardrails": False})


server.RobolabPolicyService._build_setup_args = _build_setup_args_without_guardrails


if __name__ == "__main__":
    server.main()
