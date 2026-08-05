"""Fail-closed v3 RoboTwin adapters for the three frozen WAM stacks."""

from .contract import AdapterError, AuthorizedPair, MODEL_SPECS, load_authorized_pair

__all__ = ["AdapterError", "AuthorizedPair", "MODEL_SPECS", "load_authorized_pair"]
