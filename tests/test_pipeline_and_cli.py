"""Integration tests for input parsing, Pipeline, error handling, and CLI."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from main import main as cli_main
from src.pipeline import Pipeline
from src.utils import load_data

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_EVAL = REPO_ROOT / "data" / "sample_eval.jsonl"


def test_sample_eval_pipeline_ranks_grounded_above_incomplete_and_hallucinated() -> None:
    samples = load_data(SAMPLE_EVAL)
    assert len(samples) == 15

    output = Pipeline().run(samples)
    assert [entry.model for entry in output.leaderboard] == [
        "grounded-rag-v2",
        "concise-mini-v1",
        "unconstrained-draft-v1",
    ]
    grounded, incomplete, hallucinated = output.leaderboard
    assert grounded.faithfulness.mean == pytest.approx(1.0)
    assert grounded.pass_rate == pytest.approx(1.0)
    assert incomplete.faithfulness.mean > hallucinated.faithfulness.mean
    assert incomplete.criterion_means["completeness"] < grounded.criterion_means["completeness"]
    assert hallucinated.claim_summary["contradicted"] >= 4


def test_empty_and_malformed_input_handling(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    assert load_data(empty_file) == []

    empty_out = Pipeline().run([])
    assert empty_out.leaderboard == []
    assert empty_out.evaluations == []
    assert "No evaluation samples found" in empty_out.markdown

    bad_json = tmp_path / "bad.jsonl"
    bad_json.write_text("{not valid json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed JSONL"):
        load_data(bad_json)

    missing_fields = tmp_path / "missing.jsonl"
    missing_fields.write_text('{"model": "m1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field"):
        load_data(missing_fields)

    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "does_not_exist.jsonl")


def test_cli_execution_and_file_exports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    json_dir = tmp_path / "out_dir"
    md_file = tmp_path / "leaderboard.md"

    exit_code = cli_main([
        "--input-data",
        str(SAMPLE_EVAL),
        "--output-path",
        str(json_dir),
        "--markdown",
        str(md_file),
        "--json",
    ])
    assert exit_code == 0

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    assert len(stdout_payload["models"]) == 3
    assert stdout_payload["models"][0]["model"] == "grounded-rag-v2"

    assert (json_dir / "leaderboard.json").exists()
    assert md_file.exists()
    assert "grounded-rag-v2" in md_file.read_text(encoding="utf-8")


def test_cli_subprocess_default_run() -> None:
    proc = subprocess.run(
        [sys.executable, "main.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Evaluated 15 samples across 3 models." in proc.stdout
    assert "grounded-rag-v2" in proc.stdout
    assert "unconstrained-draft-v1" in proc.stdout


def _entry(composite: float):
    from src.models import LeaderboardEntry, MetricStats

    stats = MetricStats(mean=0.0, min=0.0, max=0.0)
    return LeaderboardEntry(
        rank=1,
        model="m",
        sample_count=1,
        composite_score=composite,
        pass_rate=0.0,
        faithfulness=stats,
        g_eval=stats,
        criterion_means={},
    )


def test_exit_code_no_threshold_always_succeeds() -> None:
    from main import exit_code_for

    assert exit_code_for([_entry(0.0)], None) == 0


def test_exit_code_gates_on_min_composite() -> None:
    from main import exit_code_for

    assert exit_code_for([_entry(0.4)], 0.5) == 1
    assert exit_code_for([_entry(0.5)], 0.5) == 0
    assert exit_code_for([_entry(0.9), _entry(0.4)], 0.5) == 1
    assert exit_code_for([], 0.5) == 0


def test_exit_code_rejects_invalid_threshold() -> None:
    from main import exit_code_for

    for bad in (float("nan"), float("inf"), -0.1, 1.5):
        with pytest.raises(ValueError, match="min_composite must be a finite value"):
            exit_code_for([_entry(1.0)], bad)


def test_cli_min_composite_gates_the_run() -> None:
    assert cli_main(["--min-composite", "0.0"]) == 0
    assert cli_main(["--min-composite", "1.0"]) == 1


def test_load_data_json_variants_and_boundaries(tmp_path: Path) -> None:
    """The JSON loader accepts {samples: [...]} wrappers and bare arrays, and
    rejects malformed JSON and non-list payloads with clear errors."""
    good = {"model": "m", "question": "q", "context": ["c"], "response": "r"}

    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"samples": [good, dict(good)]}), encoding="utf-8")
    assert len(load_data(wrapped)) == 2

    array = tmp_path / "array.json"
    array.write_text(json.dumps([good]), encoding="utf-8")
    loaded = load_data(array)
    assert len(loaded) == 1 and loaded[0].model == "m"

    malformed = tmp_path / "bad.json"
    malformed.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed JSON"):
        load_data(malformed)

    not_list = tmp_path / "obj.json"
    not_list.write_text(json.dumps({"foo": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a JSON array"):
        load_data(not_list)
