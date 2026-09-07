#!/usr/bin/env python3
"""HTTP policy bridge for C8 GR00T Bridge/WidowX live episodes."""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class GrootBridgePolicyServer:
    def __init__(
        self,
        *,
        checkpoint_path: Path,
        integration_root: Path,
        device: str = "cuda",
    ) -> None:
        os.environ.setdefault("GROOT_HF_LOCAL_FIRST", "1")
        os.environ.setdefault("GROOT_PATCH_MISTRAL", "1")
        os.environ.setdefault("HF_HOME", "/data/users/ali/vla_wam/hf_home")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        sys.path.insert(0, str(integration_root))
        from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

        self._policy = Gr00tPolicy(
            embodiment_tag="SIMPLER_ENV_WIDOWX",
            model_path=str(checkpoint_path),
            device=device,
            strict=True,
        )
        self._wrapped = Gr00tSimPolicyWrapper(self._policy)
        self._modality_configs = self._policy.modality_configs

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        from experiments.online_correction_v4.droid_groot_observation import batched_observation

        processed = request.get("processed_observation")
        prompt = request.get("prompt") or request.get("instruction")
        if not isinstance(processed, dict):
            raise ValueError("processed_observation is required")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt is required")
        batch = batched_observation(
            observation=processed,
            modality_configs=self._modality_configs,
            prompt=prompt,
            np=np,
        )
        seed = request.get("sampling_seed")
        if seed is not None:
            from gr00t.utils.determinism import seed_everything

            seed_everything(int(seed))
        self._wrapped.reset()
        actions, _info = self._wrapped.get_action(batch)
        serializable = {key: value.tolist() for key, value in actions.items()}
        return {
            "action": serializable,
            "sampling_seed": seed,
        }


def _build_handler(server: GrootBridgePolicyServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path != "/healthz":
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

        def do_POST(self) -> None:
            if self.path != "/v4/infer":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(length)
            try:
                request = json.loads(payload.decode("utf-8"))
                if not isinstance(request, dict):
                    raise ValueError("request must be an object")
                response = server.infer(request)
            except Exception as exc:  # noqa: BLE001
                body = json.dumps({"error": str(exc)}, sort_keys=True).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
                return
            body = json.dumps(response, sort_keys=True).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--integration-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    server_impl = GrootBridgePolicyServer(
        checkpoint_path=args.checkpoint_path.resolve(),
        integration_root=args.integration_root.resolve(),
        device=args.device,
    )
    httpd = ThreadingHTTPServer((args.host, args.port), _build_handler(server_impl))
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
