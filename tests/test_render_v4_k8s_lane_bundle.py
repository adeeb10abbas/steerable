from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "tools/render_v4_k8s_lane_bundle.py"
SPEC = importlib.util.spec_from_file_location("render_v4_k8s_lane_bundle", SOURCE)
assert SPEC is not None and SPEC.loader is not None
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)


def _replace_sources(spec_path: Path, tmp_path: Path) -> Path:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    for role in ("policy", "simulator"):
        for index, binding in enumerate(spec[role]["file_bindings"]):
            source = tmp_path / f"{role}-{index}.bin"
            source.write_bytes(f"{role}-{index}\n".encode())
            binding["source"] = str(source)
    output = tmp_path / "spec.json"
    output.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def test_renderer_is_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    spec = _replace_sources(ROOT / "deploy/k8s/v4_lane_bundle/spec.example.json", tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    hashes_a = RENDERER.render(spec, first)
    hashes_b = RENDERER.render(spec, second)
    assert hashes_a == hashes_b
    assert {
        "configmap.yaml",
        "scripts-configmap.yaml",
        "policy-job.yaml",
        "policy-service.yaml",
        "simulator-job.yaml",
        "kustomization.yaml",
    }.issubset(hashes_a)
    assert any(name.startswith("scripts/") for name in hashes_a)
    configmap = (first / "configmap.yaml").read_text(encoding="utf-8")
    assert "immutable: true" in configmap
    assert "kube.context:" in configmap
    assert "v4-lane-id" in (first / "policy-job.yaml").read_text(encoding="utf-8")
    assert "/opt/v4-lane/scripts/lane_entrypoint.py" in (first / "policy-job.yaml").read_text(encoding="utf-8")
    assert '"file_bindings"' in configmap
    assert '"http_healthz_after_checkpoint_load"' in configmap
    assert '"infrastructure_qualification_only_no_scientific_behavior"' in configmap
    assert "--checkpoint-loaded" in (first / "policy-job.yaml").read_text(encoding="utf-8")
    assert "--launch-config" in (first / "policy-job.yaml").read_text(encoding="utf-8")
    with pytest.raises(RENDERER.RenderError, match="refusing to overwrite"):
        RENDERER.render(spec, first)


def test_renderer_rejects_policy_port_transcription(tmp_path: Path) -> None:
    spec_path = _replace_sources(ROOT / "deploy/k8s/v4_lane_bundle/spec.example.json", tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["policy"]["experiment_argv"][-1] = "9999"
    spec_path.write_text(json.dumps(spec) + "\n", encoding="utf-8")
    with pytest.raises(RENDERER.RenderError, match="port differs"):
        RENDERER.render(spec_path, tmp_path / "bad")


def test_renderer_rejects_unknown_keys(tmp_path: Path) -> None:
    spec_path = _replace_sources(ROOT / "deploy/k8s/v4_lane_bundle/spec.example.json", tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["simulator"]["silently_ignored_typo"] = True
    spec_path.write_text(json.dumps(spec) + "\n", encoding="utf-8")
    with pytest.raises(RENDERER.RenderError, match="unknown keys"):
        RENDERER.render(spec_path, tmp_path / "bad")


def test_renderer_rejects_missing_required_runtime_binding(tmp_path: Path) -> None:
    spec_path = _replace_sources(ROOT / "deploy/k8s/v4_lane_bundle/spec.example.json", tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["policy"]["file_bindings"] = [
        row for row in spec["policy"]["file_bindings"] if not row["path"].endswith("check_policy_ready.py")
    ]
    spec_path.write_text(json.dumps(spec) + "\n", encoding="utf-8")
    with pytest.raises(RENDERER.RenderError, match="omit required runtime inputs"):
        RENDERER.render(spec_path, tmp_path / "bad")


def test_http_health_contract_requires_bound_server_semantics(tmp_path: Path) -> None:
    spec_path = _replace_sources(ROOT / "deploy/k8s/v4_lane_bundle/spec.example.json", tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["policy"]["file_bindings"] = [
        row
        for row in spec["policy"]["file_bindings"]
        if not row["path"].endswith("/src/openpi/serving/websocket_policy_server.py")
    ]
    spec_path.write_text(json.dumps(spec) + "\n", encoding="utf-8")
    with pytest.raises(RENDERER.RenderError, match="omit HTTP health server semantics"):
        RENDERER.render(spec_path, tmp_path / "bad")


def test_renderer_accepts_configured_gpu_classes(tmp_path: Path) -> None:
    spec_path = _replace_sources(ROOT / "deploy/k8s/v4_lane_bundle/spec.example.json", tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["policy"]["gpu_product"] = "NVIDIA-H100-80GB-HBM3"
    spec["policy"]["expected_gpu_name"] = "NVIDIA H100 80GB HBM3"
    spec["simulator"]["gpu_product"] = "NVIDIA-B200"
    spec["simulator"]["expected_gpu_name"] = "NVIDIA B200"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output = tmp_path / "b200-h100"
    RENDERER.render(spec_path, output)
    policy_job = (output / "policy-job.yaml").read_text(encoding="utf-8")
    simulator_job = (output / "simulator-job.yaml").read_text(encoding="utf-8")
    assert "NVIDIA-H100-80GB-HBM3" in policy_job
    assert "NVIDIA-B200" in simulator_job
    configmap = (output / "configmap.yaml").read_text(encoding="utf-8")
    assert "NVIDIA-H100-80GB-HBM3" in configmap
    assert "NVIDIA-B200" in configmap


def test_renderer_rejects_qualification_with_behavioral_runner(tmp_path: Path) -> None:
    spec_path = _replace_sources(ROOT / "deploy/k8s/v4_lane_bundle/spec.example.json", tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    runner = tmp_path / "run_online_correction_v4_episodes.py"
    runner.write_text("# qualification stub\n", encoding="utf-8")
    spec["simulator"]["experiment_argv"] = ["/usr/bin/python3", str(runner)]
    spec["simulator"]["file_bindings"].append({"source": str(runner), "path": str(runner)})
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RENDERER.RenderError, match="online_correction_v4|qualification_only|forbids"):
        RENDERER.render(spec_path, tmp_path / "bad")


def test_renderer_rejects_behavioral_true_argv(tmp_path: Path) -> None:
    spec_path = _replace_sources(ROOT / "deploy/k8s/v4_lane_bundle/spec.example.json", tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["qualification_only"] = False
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RENDERER.RenderError, match="/usr/bin/true"):
        RENDERER.render(spec_path, tmp_path / "bad")


def test_renderer_accepts_behavioral_v4_runner_binding(tmp_path: Path) -> None:
    spec_path = _replace_sources(ROOT / "deploy/k8s/v4_lane_bundle/spec.example.json", tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    runner = tmp_path / "run_online_correction_v4_episodes.py"
    runner.write_text("# behavioral stub\n", encoding="utf-8")
    mounted = f"/data/runner/{runner.name}"
    spec["qualification_only"] = False
    spec["simulator"]["experiment_argv"] = [
        spec["runtime"]["simulator"]["python_bin"],
        mounted,
        "--queue",
        "/data/queue.jsonl",
    ]
    spec["simulator"]["file_bindings"].append({"source": str(runner), "path": mounted})
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output = tmp_path / "behavioral"
    RENDERER.render(spec_path, output)
    configmap = (output / "configmap.yaml").read_text(encoding="utf-8")
    assert "online_correction_v4" in configmap
    assert "infrastructure_qualification_only_no_scientific_behavior" not in configmap


def test_vendored_openpi_health_server_is_exact_c23745b_source() -> None:
    source = ROOT / "deploy/k8s/v4_lane_bundle/frozen_sources/openpi_websocket_policy_server.c23745b.py"
    payload = source.read_bytes()
    assert len(payload) == 3051
    assert hashlib.sha256(payload).hexdigest() == "1370d345e6c3c5b8f15573050e485e60a5b423d1df33e24b237805e6b442b026"


def test_render_probe_preserves_failure_across_isaac_shutdown() -> None:
    source = (ROOT / "deploy/k8s/v4_lane_bundle/scripts/isaac_render_probe.py").read_text(
        encoding="utf-8"
    )
    assert 'parser.add_argument("--num-envs"' in source
    assert "except BaseException as exc:" in source
    assert "traceback.print_exc()" in source
    assert "raise original.with_traceback(original_traceback)" in source
    assert source.index("except BaseException as exc:") < source.index("simulation_app.close()")
