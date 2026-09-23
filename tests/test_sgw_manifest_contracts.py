from pathlib import Path

import yaml


MANIFESTS = Path(__file__).resolve().parents[1] / "handoff" / "k8s"


def manifest(attempt):
    return yaml.safe_load((MANIFESTS / f"sgw01-ali-lat-workspace-20260922{attempt}.yaml").read_text())


def test_failed_m_exposes_the_flow_mapping_capability_split():
    container = manifest("m")["spec"]["template"]["spec"]["containers"][0]
    entry = next(item for item in container["env"] if item["name"] == "NVIDIA_DRIVER_CAPABILITIES")
    assert entry == {
        "name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute",
        "utility": None, "graphics": None, "display": None,
    }


def test_replacement_preserves_the_entire_proven_runtime_configuration():
    previous = manifest("g")
    replacement = manifest("r")
    previous["metadata"]["name"] = replacement["metadata"]["name"]
    command = previous["spec"]["template"]["spec"]["containers"][0]["args"][0]
    command = command.replace(
        "b429ddd8c3e7be59f63759c1820e352029e48b8c",
        "35e627eda09f24f00995fc2dce4dbee362c0bc02",
    ).replace("/source/b429ddd", "/source/35e627e-q").replace(
        "lat-workspace-20260922g", "lat-workspace-20260922r",
    )
    previous["spec"]["template"]["spec"]["containers"][0]["args"] = [command]
    assert replacement == previous
    environment = replacement["spec"]["template"]["spec"]["containers"][0]["env"]
    assert all(set(entry) == {"name", "value"} and isinstance(entry["value"], str) for entry in environment)
    assert next(entry["value"] for entry in environment if entry["name"] == "NVIDIA_DRIVER_CAPABILITIES") == (
        "compute,utility,graphics,display"
    )
