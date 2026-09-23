"""Release one frozen zero-model capture Job after its GPU predecessor completes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import ssl
import sys
import time
from typing import Any, Mapping
import urllib.parse
import urllib.error
import urllib.request


def spec_digest(job: Mapping[str, Any]) -> str:
    spec = dict(job["spec"])
    spec.pop("suspend", None)
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_job(job: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    metadata = job["metadata"]
    if (
        metadata["name"] != expected["name"]
        or metadata["uid"] != expected["uid"]
        or spec_digest(job) != expected["spec_sha256"]
        or metadata.get("labels", {}).get("owner") != "ali"
        or metadata.get("labels", {}).get("app.kubernetes.io/name") != "sgw-01"
    ):
        raise ValueError("Job identity or frozen specification changed")


def terminal_state(job: Mapping[str, Any]) -> str | None:
    conditions = {row["type"] for row in job.get("status", {}).get("conditions", [])
                  if row.get("status") == "True"}
    if "Failed" in conditions:
        return "failed"
    if "Complete" in conditions:
        return "complete"
    return None


def predecessor_released(job: Mapping[str, Any]) -> bool:
    state = terminal_state(job)
    if state == "failed":
        raise ValueError("predecessor failed; automatic next-phase release is blocked")
    status = job.get("status", {})
    return (
        state == "complete"
        and status.get("active", 0) == 0
        and status.get("terminating", 0) == 0
        and status.get("succeeded") == job["spec"]["completions"]
        and not status.get("failedIndexes")
    )


def activation_patch(target: Mapping[str, Any]) -> list[dict[str, Any]]:
    if target["spec"].get("suspend") is not True:
        raise ValueError("capture Job is not suspended")
    return [
        {"op": "test", "path": "/metadata/uid", "value": target["metadata"]["uid"]},
        {"op": "test", "path": "/metadata/resourceVersion", "value": target["metadata"]["resourceVersion"]},
        {"op": "test", "path": "/spec/suspend", "value": True},
        {"op": "replace", "path": "/spec/suspend", "value": False},
    ]


class Cluster:
    def __init__(self, namespace: str, credentials: Path) -> None:
        host, port = os.environ["KUBERNETES_SERVICE_HOST"], os.environ["KUBERNETES_SERVICE_PORT_HTTPS"]
        self.base = f"https://{host}:{port}/apis/batch/v1/namespaces/{urllib.parse.quote(namespace, safe='')}/jobs/"
        self.credentials = credentials
        self.tls = ssl.create_default_context(cafile=str(credentials / "ca.crt"))

    def request(self, name: str, patch: list[dict[str, Any]] | None = None) -> Mapping[str, Any]:
        for attempt in range(3):
            try:
                return self._request(name, patch)
            except (urllib.error.URLError, TimeoutError) as error:
                code = getattr(error, "code", None)
                if code is not None and code not in {409, 429, 500, 502, 503, 504}:
                    raise
                print(json.dumps({"api_attempt_failed": attempt + 1, "http_status": code,
                                  "job": name, "exception_type": type(error).__name__}), flush=True)
                if patch is not None:
                    observed = self._request(name)
                    if observed["spec"].get("suspend") is False and observed["metadata"]["uid"] == patch[0]["value"]:
                        return observed
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable API retry state")

    def _request(self, name: str, patch: list[dict[str, Any]] | None = None) -> Mapping[str, Any]:
        token = (self.credentials / "token").read_text().strip()
        request = urllib.request.Request(
            self.base + urllib.parse.quote(name, safe=""),
            data=json.dumps(patch).encode() if patch is not None else None,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json-patch+json" if patch is not None else "application/json",
            },
            method="PATCH" if patch is not None else "GET",
        )
        with urllib.request.urlopen(request, context=self.tls, timeout=30) as response:
            return json.loads(response.read())


def run(plan: Mapping[str, Any], cluster: Any, output: Path, *, clock=time.monotonic, wait=time.sleep) -> None:
    if plan.get("scope") != "zero_model_prospective_capture_only" or plan.get("gpu_ceiling") != 4:
        raise ValueError("sequencer plan is not the bounded zero-model capture plan")
    output.mkdir(parents=True, exist_ok=False)
    deadline = clock() + plan["timeout_seconds"]
    previous = None
    result = "blocked"
    try:
        while clock() < deadline:
            producer = cluster.request(plan["predecessor"]["name"])
            target = cluster.request(plan["target"]["name"])
            check_job(producer, plan["predecessor"])
            check_job(target, plan["target"])
            containers = target["spec"]["template"]["spec"]["containers"]
            gpu_count = sum(int(row["resources"]["limits"].get("nvidia.com/gpu", 0)) for row in containers)
            if not 0 < gpu_count * target["spec"]["parallelism"] <= plan["gpu_ceiling"]:
                raise ValueError("capture Job exceeds the frozen GPU ceiling")
            released = predecessor_released(producer)
            suspended = target["spec"].get("suspend") is True
            if not released and not suspended:
                raise ValueError("capture Job activated before predecessor released its allocation")
            state = {
                "predecessor_succeeded": producer.get("status", {}).get("succeeded", 0),
                "predecessor_released": released,
                "target_suspended": suspended,
                "target_active": target.get("status", {}).get("active", 0),
                "target_terminal": terminal_state(target),
            }
            if state != previous:
                with (output / "events.jsonl").open("a") as stream:
                    stream.write(json.dumps({**state, "observed_at_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                previous = state
            if released and suspended:
                updated = cluster.request(plan["target"]["name"], activation_patch(target))
                check_job(updated, plan["target"])
                if updated["spec"].get("suspend") is not False:
                    raise ValueError("capture Job activation was not persisted")
                continue
            if terminal_state(target) == "failed":
                raise ValueError("zero-model capture Job failed; no automatic retry")
            if terminal_state(target) == "complete":
                if target.get("status", {}).get("succeeded") != target["spec"]["completions"]:
                    raise ValueError("capture Job completed without every frozen index")
                result = "zero_model_capture_job_completed_requires_evidence_review"
                return
            wait(15)
        raise TimeoutError("bounded capture sequencing deadline expired")
    finally:
        error = sys.exc_info()[1]
        with (output / "receipt.json").open("x") as stream:
            json.dump({
                "status": result, "last_observed": previous,
                "predecessor": plan["predecessor"], "target": plan["target"],
                "model_requests": 0, "behavioral_episodes": 0, "release_permitted": False,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": {"type": type(error).__name__, "message": str(error)} if error else None,
            }, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--credentials", type=Path, default=Path("/var/run/sgw-kubernetes"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.plan.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.expected_plan_sha256:
        raise ValueError("sequencer plan hash differs")
    plan = json.loads(raw)
    run(plan, Cluster(plan["namespace"], args.credentials), args.output)


if __name__ == "__main__":
    main()
