"""Tests for the LLM eval leaderboard UI server (ui/server.py)."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from ui.server import build_server, run_eval

SAMPLE = json.dumps(
    {
        "model": "m1",
        "question": "What is 2+2?",
        "context": ["Basic arithmetic: 2+2 equals 4."],
        "response": "2+2 equals 4.",
    }
)


@pytest.fixture()
def server():
    srv = build_server("127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _post(base: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base}/api/evaluate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request)
        return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_evaluate_endpoint_runs_real_harness(server):
    status, payload = _post(server, {"samples_jsonl": SAMPLE})
    assert status == 200
    assert payload["sample_count"] == 1
    entry = payload["leaderboard"][0]
    assert entry["model"] == "m1"
    assert 0.0 <= entry["composite_score"] <= 1.0
    assert "faithfulness" in entry and "g_eval" in entry
    # serializes nested MetricStats to plain dicts
    assert isinstance(entry["faithfulness"], dict)


def test_default_samples_produce_ranked_leaderboard(server):
    status, payload = _post(server, {})
    assert status == 200
    assert len(payload["leaderboard"]) >= 2
    ranks = [e["rank"] for e in payload["leaderboard"]]
    assert ranks == sorted(ranks)


def test_health_sample_and_index_served(server):
    assert json.loads(urllib.request.urlopen(server + "/api/health").read())["status"] == "ok"
    sample = json.loads(urllib.request.urlopen(server + "/api/sample").read())
    assert "samples_jsonl" in sample and sample["samples_jsonl"].strip()
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert "leaderboard" in html and "/api/evaluate" in html


@pytest.mark.parametrize(
    "payload,message",
    [
        ({"samples_jsonl": "   "}, "samples_jsonl must be a non-empty string"),
        ({"samples_jsonl": "{bad json"}, "Malformed JSONL"),
        ({"samples_jsonl": '{"question": "q"}'}, "missing required field"),
    ],
)
def test_invalid_inputs_rejected_with_clear_message(server, payload, message):
    status, body = _post(server, payload)
    assert status == 400
    assert message in body["error"]


def test_run_eval_serializes_nested_stats():
    result = run_eval(samples_jsonl=SAMPLE)
    json.dumps(result)  # must be JSON-serializable end to end
    assert result["leaderboard"][0]["g_eval"]["mean"] >= 0.0