"""Unit tests for GEvalScorer rubric weights, JudgeBackend, and LeaderboardGenerator."""

import json
import math
from pathlib import Path

import pytest

from config import EvalHarnessConfig
from src.models import (
    FaithfulnessResult,
    GEvalResult,
    GEvalScorer,
    LeaderboardGenerator,
    SampleEvaluation,
)


@pytest.mark.parametrize(
    "overrides",
    [
        {"faithfulness_threshold": float("nan")},
        {"geval_threshold": 1.1},
        {"faithfulness_support_threshold": -0.1},
        {"composite_geval_weight": float("inf")},
        {"composite_faithfulness_weight": 0.0, "composite_geval_weight": 0.0},
        {"composite_faithfulness_weight": 1e308, "composite_geval_weight": 1e308},
        {"rubric_weights": {"coherence": float("nan")}},
        {"rubric_weights": {"coherence": 1e308, "relevance": 1e308}},
        {"rubric_weights": {"coherence": 0.0}},
    ],
)
def test_eval_harness_config_rejects_invalid_scoring_settings(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        EvalHarnessConfig(**overrides)  # type: ignore[arg-type]


def test_geval_completeness_relevance_and_conciseness_discrimination() -> None:
    scorer = GEvalScorer()
    question = "What parameters does Okapi BM25 use for saturation and length normalization?"
    context = [
        "Okapi BM25 uses k1=1.5 for term frequency saturation and b=0.75 for document length normalization."
    ]
    reference = "k1=1.5 controls term frequency saturation and b=0.75 controls document length normalization."

    complete_resp = (
        "Okapi BM25 uses k1=1.5 for term frequency saturation and b=0.75 for document length normalization."
    )
    incomplete_resp = "Okapi BM25 uses k1=1.5."
    off_topic_resp = "Sourdough starter ferments flour and water with wild yeast overnight."
    bloated_resp = (
        "As an AI language model, certainly! It is important to note that Okapi BM25 "
        "uses k1=1.5 for term frequency saturation and b=0.75 for document length normalization, "
        "and it is worth repeating that k1=1.5 and b=0.75 are used repeatedly over and over."
    )

    r_complete = scorer.evaluate(question, complete_resp, context, reference)
    r_incomplete = scorer.evaluate(question, incomplete_resp, context, reference)
    r_off_topic = scorer.evaluate(question, off_topic_resp, context, reference)
    r_bloated = scorer.evaluate(question, bloated_resp, context, reference)

    assert r_complete.criterion_scores["completeness"] > r_incomplete.criterion_scores["completeness"]
    assert r_complete.criterion_scores["relevance"] > r_off_topic.criterion_scores["relevance"]
    assert r_complete.criterion_scores["conciseness"] > r_bloated.criterion_scores["conciseness"]
    assert r_complete.overall_score > r_incomplete.overall_score
    assert r_complete.overall_score > r_off_topic.overall_score


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), float("-inf")])
def test_geval_rejects_non_finite_backend_scores(invalid_score: float) -> None:
    class InvalidBackend:
        def score_criterion(
            self, criterion, rubric_description, question, response, context, reference_answer=None
        ):
            return invalid_score

    scorer = GEvalScorer(weights={"coherence": 1.0}, backend=InvalidBackend())

    with pytest.raises(ValueError, match="non-finite score"):
        scorer.evaluate("question", "response")


@pytest.mark.parametrize(
    "weights",
    [
        {"coherence": float("nan")},
        {"coherence": float("inf")},
        {"coherence": 1e308, "relevance": 1e308},
    ],
)
def test_geval_rejects_non_finite_or_overflowing_weights(weights: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        GEvalScorer(weights=weights)


def test_geval_custom_weights_and_pluggable_backend() -> None:
    class FixedBackend:
        def score_criterion(self, criterion, rubric_description, question, response, context, reference_answer=None):
            mapping = {
                "coherence": 1.0,
                "relevance": 0.5,
                "conciseness": 0.0,
                "completeness": 0.25,
            }
            return mapping[criterion]

    weights = {
        "coherence": 1.0,
        "relevance": 2.0,
        "conciseness": 1.0,
        "completeness": 4.0,
    }
    scorer = GEvalScorer(weights=weights, backend=FixedBackend())
    result = scorer.evaluate("q", "r")

    expected = (1 * 1.0 + 2 * 0.5 + 1 * 0.0 + 4 * 0.25) / 8.0
    assert math.isclose(result.overall_score, expected, rel_tol=1e-9)
    assert math.isclose(sum(result.weights.values()), 1.0, rel_tol=1e-9)

    with pytest.raises(ValueError):
        GEvalScorer(weights={})
    with pytest.raises(ValueError):
        GEvalScorer(weights={"coherence": -0.5})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"faithfulness_threshold": float("nan")},
        {"faithfulness_threshold": 1.1},
        {"geval_threshold": -0.1},
        {"faithfulness_weight": float("inf")},
        {"faithfulness_weight": -1.0},
        {"faithfulness_weight": 1e308, "geval_weight": 1e308},
        {"faithfulness_weight": 0.0, "geval_weight": 0.0},
        {"faithfulness_threshold": True},
        {"geval_threshold": "x"},
        {"faithfulness_weight": None},
    ],
)
def test_leaderboard_generator_rejects_invalid_thresholds_and_weights(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        LeaderboardGenerator(**kwargs)


def test_leaderboard_aggregation_math_and_exports(tmp_path: Path) -> None:
    gen = LeaderboardGenerator(faithfulness_threshold=0.75, geval_threshold=0.70)

    def make_eval(sample_id: str, model: str, faith: float, geval: float, sup: int, contra: int) -> SampleEvaluation:
        return SampleEvaluation(
            sample_id=sample_id,
            model=model,
            question="Q?",
            faithfulness=FaithfulnessResult(
                score=faith,
                claims=[],
                supported_count=sup,
                unsupported_count=0,
                contradicted_count=contra,
            ),
            g_eval=GEvalResult(
                overall_score=geval,
                criterion_scores={
                    "coherence": geval,
                    "relevance": geval,
                    "conciseness": geval,
                    "completeness": geval,
                },
                weights={"coherence": 0.25, "relevance": 0.25, "conciseness": 0.25, "completeness": 0.25},
            ),
            passed=gen.is_passing(faith, geval),
        )

    evals = [
        make_eval("s1", "alpha", 1.0, 0.9, 2, 0),
        make_eval("s2", "alpha", 0.8, 0.7, 1, 0),
        make_eval("s3", "beta", 0.5, 0.8, 1, 1),
        make_eval("s4", "beta", 0.9, 0.6, 2, 0),
    ]
    entries = gen.aggregate(evals)
    assert [e.model for e in entries] == ["alpha", "beta"]
    assert entries[0].rank == 1
    assert entries[1].rank == 2

    alpha = entries[0]
    assert alpha.sample_count == 2
    assert alpha.faithfulness.mean == pytest.approx(0.9)
    assert alpha.faithfulness.min == pytest.approx(0.8)
    assert alpha.faithfulness.max == pytest.approx(1.0)
    assert alpha.g_eval.mean == pytest.approx(0.8)
    assert alpha.pass_rate == pytest.approx(1.0)
    assert alpha.composite_score == pytest.approx(0.85)
    assert alpha.claim_summary == {"supported": 3, "unsupported": 0, "contradicted": 0}

    beta = entries[1]
    assert beta.pass_rate == pytest.approx(0.0)
    assert beta.claim_summary == {"supported": 3, "unsupported": 0, "contradicted": 1}

    json_path = gen.export_json(entries, tmp_path, evaluations=evals)
    assert json_path.name == "leaderboard.json"
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(loaded["models"]) == 2
    assert len(loaded["samples"]) == 4

    md_path = gen.export_markdown(entries, tmp_path / "report.md")
    md_text = md_path.read_text(encoding="utf-8")
    assert "| Rank | Model |" in md_text
    assert "| 1 | alpha |" in md_text
    assert "G-Eval Rubric Breakdown" in md_text


def test_json_report_includes_score_distributions() -> None:
    """The JSON report carries the per-sample score series the UI plots as spreads."""
    gen = LeaderboardGenerator(faithfulness_threshold=0.75, geval_threshold=0.70)

    def make_eval(sample_id: str, faith: float, geval: float) -> SampleEvaluation:
        return SampleEvaluation(
            sample_id=sample_id,
            model="alpha",
            question="Q?",
            faithfulness=FaithfulnessResult(
                score=faith, claims=[], supported_count=1,
                unsupported_count=0, contradicted_count=0,
            ),
            g_eval=GEvalResult(
                overall_score=geval,
                criterion_scores={"coherence": geval, "relevance": geval - 0.1},
                weights={"coherence": 0.5, "relevance": 0.5},
            ),
            passed=gen.is_passing(faith, geval),
        )

    evals = [make_eval("s1", 1.0, 0.9), make_eval("s2", 0.8, 0.7)]
    entries = gen.aggregate(evals)
    payload = gen.to_dict(entries, evaluations=evals)

    dist = payload["distributions"]
    assert dist["alpha"]["faithfulness"] == [1.0, 0.8]
    assert dist["alpha"]["g_eval"] == [0.9, 0.7]
    assert dist["alpha"]["criteria"]["coherence"] == [0.9, 0.7]
    assert dist["alpha"]["criteria"]["relevance"] == [pytest.approx(0.8), pytest.approx(0.6)]
    assert "distributions" not in gen.to_dict(entries), "no evaluations → no series"
