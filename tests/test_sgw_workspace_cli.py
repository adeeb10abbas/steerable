"""Workspace capture must preserve receipt paths during two-stage CLI parsing."""

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from experiments.workshops.spatial_grounding_v1 import lat_workspace_capture


def test_renderer_argument_does_not_abbreviate_renderer_receipt(tmp_path, monkeypatch):
    assets = tmp_path / "assets.json"
    receipt = tmp_path / "renderer.json"
    assets.write_text("{}")
    receipt.write_text("{}")
    monkeypatch.setattr(lat_workspace_capture, "is_git_worktree", lambda path: path == tmp_path)

    def add_common_args(parser):
        parser.add_argument("--renderer")
        parser.add_argument("--rendering-type")

    def add_app_args(parser):
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--rendering_mode")

    for name in ("isaaclab", "isaaclab.app", "robolab", "robolab.eval", "robolab.eval.runner"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["isaaclab.app"].AppLauncher = SimpleNamespace(add_app_launcher_args=add_app_args)
    sys.modules["robolab.eval.runner"].add_common_eval_args = add_common_args
    monkeypatch.setattr(sys, "argv", [
        "lat_workspace_capture",
        "--study-root", str(Path(__file__).resolve().parents[1]),
        "--robolab-root", str(tmp_path),
        "--assets-manifest", str(assets),
        "--renderer-receipt", str(receipt),
        "--output", str(tmp_path / "workspace.json"),
        "--headless", "--renderer", "realtime",
        "--rendering-type", "balanced", "--rendering_mode", "balanced",
    ])

    args = lat_workspace_capture.parse_args()

    assert args.renderer_receipt == receipt
    assert args.assets_manifest == assets
    assert args.renderer == "realtime"
    assert args.rendering_type == args.rendering_mode == "balanced"
