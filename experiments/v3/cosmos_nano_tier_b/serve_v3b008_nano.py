#!/usr/bin/env python3
"""Dedicated port-18018 Cosmos3 Nano server for V3-B008 only."""

from experiments.v3.cosmos_nano_tier_b.server import run_server


if __name__ == "__main__":
    run_server("V3-B008")

