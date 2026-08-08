"""Dedicated Cosmos3 Edge E004 server entry point (no simulator launch)."""

from .cosmos_server import run_server


if __name__ == "__main__":
    run_server("cosmos3_edge_policy_droid")
