"""Unit tests for FaithfulnessEvaluator claim verification, numbers, and polarity."""

import pytest

from src.models import FaithfulnessEvaluator


@pytest.mark.parametrize(
    "attr", ["support_threshold", "alignment_threshold", "contradiction_penalty"]
)
@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), "x", None, True]
)
def test_faithfulness_evaluator_rejects_invalid_threshold(attr: str, value: object) -> None:
    with pytest.raises(ValueError, match=attr):
        FaithfulnessEvaluator(**{attr: value})


@pytest.mark.parametrize("attr", ["support_threshold", "alignment_threshold"])
@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_faithfulness_evaluator_rejects_out_of_range_score_threshold(
    attr: str, value: float
) -> None:
    with pytest.raises(ValueError, match=attr):
        FaithfulnessEvaluator(**{attr: value})


def test_faithfulness_evaluator_rejects_negative_contradiction_penalty() -> None:
    with pytest.raises(ValueError, match="contradiction_penalty"):
        FaithfulnessEvaluator(contradiction_penalty=-0.1)


def test_faithfulness_evaluator_accepts_valid_thresholds() -> None:
    evaluator = FaithfulnessEvaluator(
        support_threshold=0.0, alignment_threshold=1.0, contradiction_penalty=2.5
    )
    assert evaluator.support_threshold == 0.0
    assert evaluator.alignment_threshold == 1.0
    assert evaluator.contradiction_penalty == 2.5


def test_grounded_answer_scores_higher_than_hallucinated_numbers_and_ungrounded_claims() -> None:
    evaluator = FaithfulnessEvaluator()
    context = [
        "The Transformer architecture was introduced in 2017 with 6 encoder layers "
        "and achieved 28.4 BLEU on WMT14."
    ]
    grounded = (
        "The Transformer was introduced in 2017 using 6 encoder layers, "
        "achieving 28.4 BLEU on WMT14."
    )
    hallucinated_nums = (
        "The Transformer was introduced in 2019 using 12 encoder layers, "
        "achieving 41.2 BLEU on WMT14."
    )
    ungrounded = (
        "The model was invented in Zurich on quantum processors for autonomous rockets."
    )

    res_grounded = evaluator.evaluate(grounded, context)
    res_hallucinated = evaluator.evaluate(hallucinated_nums, context)
    res_ungrounded = evaluator.evaluate(ungrounded, context)

    assert res_grounded.score == 1.0
    assert res_grounded.supported_count == 1
    assert res_grounded.contradicted_count == 0

    assert res_hallucinated.contradicted_count == 1
    assert res_hallucinated.claims[0].verdict == "contradicted"
    assert res_hallucinated.score < res_grounded.score

    assert res_ungrounded.unsupported_count == 1
    assert res_ungrounded.claims[0].verdict == "unsupported"
    assert res_ungrounded.score < res_grounded.score


def test_partial_contradiction_scores_strictly_below_ungrounded_and_grounded() -> None:
    evaluator = FaithfulnessEvaluator()
    context = [
        "Okapi BM25 uses k1=1.5 for term saturation. "
        "Reciprocal Rank Fusion uses smoothing constant k=60."
    ]
    all_supported = (
        "Okapi BM25 uses k1=1.5 for term saturation. "
        "Reciprocal Rank Fusion uses smoothing constant k=60."
    )
    one_unsupported = (
        "Okapi BM25 uses k1=1.5 for term saturation. "
        "It also trains multi-billion parameter vision transformers on video."
    )
    one_contradicted = (
        "Okapi BM25 uses k1=1.5 for term saturation. "
        "Reciprocal Rank Fusion uses smoothing constant k=120."
    )

    s_ok = evaluator.evaluate(all_supported, context).score
    s_unsup = evaluator.evaluate(one_unsupported, context).score
    s_contra = evaluator.evaluate(one_contradicted, context).score

    assert s_ok == 1.0
    assert s_unsup < s_ok
    assert s_contra < s_unsup


def test_negation_and_contrast_polarity_checks() -> None:
    evaluator = FaithfulnessEvaluator()
    context = [
        "Semantic caching matches query embeddings rather than requiring exact string equality. "
        "Sliding overlap prevents sentences from being severed across boundaries."
    ]
    faithful_paraphrase = (
        "Semantic caching does not require exact string equality. "
        "Sliding overlap ensures sentences are not severed across boundaries."
    )
    contradicted_negation = (
        "Semantic caching requires exact string equality. "
        "Sliding overlap allows sentences to be severed."
    )

    res_ok = evaluator.evaluate(faithful_paraphrase, context)
    assert res_ok.score == 1.0
    assert res_ok.contradicted_count == 0

    res_bad = evaluator.evaluate(contradicted_negation, context)
    assert res_bad.contradicted_count >= 1
    assert res_bad.score < res_ok.score


def test_antonym_conflict_and_empty_inputs() -> None:
    evaluator = FaithfulnessEvaluator()
    context = ["Caching decreases query latency across repeated requests."]
    res_antonym = evaluator.evaluate(
        "Caching increases query latency across repeated requests.", context
    )
    assert res_antonym.contradicted_count == 1
    assert res_antonym.claims[0].verdict == "contradicted"

    assert evaluator.evaluate("", context).score == 0.0
    res_no_ctx = evaluator.evaluate("Any factual claim.", [])
    assert res_no_ctx.score == 0.0
    assert res_no_ctx.unsupported_count == 1
