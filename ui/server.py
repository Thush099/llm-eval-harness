"""Lightweight local web UI server for the LLM evaluation harness.

Serves the leaderboard workbench in ``ui/index.html`` and exposes a JSON API that runs
the real ``Pipeline``. Standard library only. Start with ``python ui/server.py`` and open
the printed URL.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import EvalHarnessConfig  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.utils import load_data  # noqa: E402

INDEX_HTML = (REPO_ROOT / "ui" / "index.html").read_text(encoding="utf-8")

DEFAULT_INPUT = REPO_ROOT / "data" / "sample_eval.jsonl"
MAX_SAMPLES_BYTES = 2_000_000


def run_eval(
    samples_jsonl: Optional[str] = None,
    faithfulness_threshold: float = 0.7,
    geval_threshold: float = 0.7,
) -> dict[str, Any]:
    """Run the evaluation harness over samples and return the leaderboard.

    Without ``samples_jsonl``, the bundled sample evaluation file is used. Raises ValueError
    with a clear message for invalid inputs, matching the CLI's boundary validation.
    """
    if samples_jsonl is None:
        samples = load_data(DEFAULT_INPUT)
    else:
        if not samples_jsonl.strip():
            raise ValueError("samples_jsonl must be a non-empty string")
        if len(samples_jsonl.encode("utf-8")) > MAX_SAMPLES_BYTES:
            raise ValueError(f"samples_jsonl exceeds {MAX_SAMPLES_BYTES} bytes")
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(samples_jsonl)
            tmp_path = handle.name
        try:
            samples = load_data(tmp_path)
        finally:
            os.unlink(tmp_path)

    config = EvalHarnessConfig(
        faithfulness_threshold=faithfulness_threshold,
        geval_threshold=geval_threshold,
    )
    pipeline = Pipeline(config=config)
    output = pipeline.run(samples)
    return {
        "leaderboard": [entry.to_dict() for entry in output.leaderboard],
        "sample_count": len(output.evaluations),
    }


def build_server(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Create (but do not start) the UI HTTP server bound to ``host:port``."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/health":
                self._send_json({"status": "ok"})
            elif self.path == "/api/sample":
                self._send_json({"samples_jsonl": DEFAULT_INPUT.read_text(encoding="utf-8")})
            else:
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)

        def do_POST(self) -> None:
            if self.path != "/api/evaluate":
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                result = run_eval(
                    samples_jsonl=payload.get("samples_jsonl"),
                    faithfulness_threshold=payload.get("faithfulness_threshold", 0.7),
                    geval_threshold=payload.get("geval_threshold", 0.7),
                )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the LLM eval leaderboard UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8128)
    args = parser.parse_args(argv)

    server = build_server(args.host, args.port)
    print(f"LLM eval leaderboard running at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())