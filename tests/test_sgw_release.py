import json
from pathlib import Path

import pytest

from experiments.workshops.spatial_grounding_v1.contract import ContractError
from experiments.workshops.spatial_grounding_v1.release import create_release, render_job
from tests.test_sgw_contract import make_release


def _update_binding(release: Path, binding: dict) -> None:
    (release / "runtime_binding.json").write_text(json.dumps(binding))
    hashes = json.loads((release / "hashes.json").read_text())
    import hashlib
    hashes["runtime_binding.json"] = hashlib.sha256((release / "runtime_binding.json").read_bytes()).hexdigest()
    (release / "hashes.json").write_text(json.dumps(hashes))


def test_render_job_resolves_every_template_token(tmp_path: Path) -> None:
    release = make_release(tmp_path)
    binding = json.loads((release / "runtime_binding.json").read_text())
    binding["cpu_memory_limits"] = {"cpu_request": "2", "memory_request": "4Gi", "cpu_limit": "4", "memory_limit": "8Gi"}
    _update_binding(release, binding)
    template = Path("experiments/workshops/spatial_grounding_v1/spec/kubernetes/worker-job.yaml.in")
    output = tmp_path / "job.yaml"
    render_job(release=release, template=template, output=output, model="N3", family="LAT", stage="P")
    text = output.read_text()
    assert "${" not in text
    assert "@sha256:" in text
    assert '"6"' in text


def test_render_job_rejects_unresolved_placeholder(tmp_path: Path) -> None:
    release = make_release(tmp_path)
    binding = json.loads((release / "runtime_binding.json").read_text())
    binding["cpu_memory_limits"] = {"cpu_request": "2", "memory_request": "4Gi", "cpu_limit": "4", "memory_limit": "8Gi"}
    _update_binding(release, binding)
    template = tmp_path / "bad.yaml"
    template.write_text("image: ${MISSING}\n")
    with pytest.raises(ContractError, match="unresolved"):
        render_job(release=release, template=template, output=tmp_path / "job.yaml",
                   model="N3", family="LAT", stage="P")


def test_create_release_consumes_frozen_csv_registry(tmp_path: Path) -> None:
    source = Path("experiments/workshops/spatial_grounding_v1/spec")
    binding = json.loads((make_release(tmp_path) / "runtime_binding.json").read_text())
    binding["cpu_memory_limits"] = {"cpu_request": "2", "memory_request": "4Gi", "cpu_limit": "4", "memory_limit": "8Gi"}
    binding.update({"model_code_commits": {"N3": "a"}, "checkpoint_hashes": {"N3": "b"},
                    "policy_ports": {"N3": 1}, "frame_time_mapping_hashes": {"N3": "c"}})
    binding_path = tmp_path / "binding.json"
    binding_path.write_text(json.dumps(binding))
    fixtures = tmp_path / "fixtures.json"
    fixtures.write_text(json.dumps({
        "status": "qualified",
        "layouts": {"LAT-P01": {"fixture_sha256": "f" * 64}},
        "time_maps": {"N3": "t" * 64},
    }))
    output = create_release(output=tmp_path / "released", release_id="sgw-test",
                            protocol=source / "protocol.json", prompts=source / "prompts.json",
                            planned_queue=source / "planned_cells.csv", fixtures=fixtures,
                            runtime_binding=binding_path, resource_owner="ali", stage="P",
                            model="N3", family="LAT")
    queue = (output / "queue.jsonl").read_text().splitlines()
    assert len(queue) == 6
    assert json.loads(queue[0])["status"] == "RELEASED"
