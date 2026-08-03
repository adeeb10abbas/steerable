from __future__ import annotations

import ast
from pathlib import Path


LINGBOT_RUNNER = Path(
    "/home/ali/projects/lerobot-lingbot/experiments/lingbot_language_gate/closed_loop_language_gate.py"
)


def test_lingbot_trace_is_the_exact_absolute_action_passed_to_environment() -> None:
    tree = ast.parse(LINGBOT_RUNNER.read_text())
    run_episode = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_episode"
    )
    take_action_calls = [
        node
        for node in ast.walk(run_episode)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "take_action"
    ]
    assert len(take_action_calls) == 1
    assert isinstance(take_action_calls[0].args[0], ast.Name)
    assert take_action_calls[0].args[0].id == "executed_action"

    save_calls = [
        node
        for node in ast.walk(run_episode)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "savez_compressed"
    ]
    assert len(save_calls) == 1
    executed_keyword = next(keyword for keyword in save_calls[0].keywords if keyword.arg == "executed")
    assert isinstance(executed_keyword.value, ast.Name)
    assert executed_keyword.value.id == "executed_action_trace"

    provenance_calls = [
        node
        for node in ast.walk(run_episode)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "action_trace_record"
    ]
    assert len(provenance_calls) == 1
    assert isinstance(provenance_calls[0].args[1], ast.Name)
    assert provenance_calls[0].args[1].id == "executed_action_trace"
