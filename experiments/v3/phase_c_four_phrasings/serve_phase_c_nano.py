"""Compatibility entry point for the registered Cosmos Phase-C overlay."""

from cosmos_framework.scripts import action_policy_server_robolab as server

from experiments.v3.phase_c_four_phrasings import serve_phase_c_cosmos  # noqa: F401


if __name__ == "__main__":
    server.main()
