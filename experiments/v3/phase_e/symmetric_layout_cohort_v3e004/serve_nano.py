"""Dedicated Cosmos3 Nano E004 server entry point (no simulator launch)."""

from .cosmos_server import run_server


if __name__ == "__main__":
    run_server("cosmos3_nano_policy_droid")
