"""Faithfulness evaluation, rubric-based G-Eval scoring, and leaderboard reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
from typing import (
    Any,
    Dict,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

from config import default_rubric_weights
from src.utils import (
    ANTONYM_PAIRS,
    NEGATION_WORDS,
    EvalSample,
    compute_idf_map,
    extract_content_tokens,
    extract_numbers,
    make_bigrams,
    split_claims,
    split_context_propositions,
    split_sentences,
    stem_token,
    tokenize,
)

ClaimLabel = Literal["supported", "unsupported", "contradicted"]

_BOILERPLATE_PATTERNS = (
    "as an ai",
    "as a language model",
    "sure, here is",
    "certainly!",
    "it is important to note that",
    "in summary, it is worth noting",
)


@dataclass(frozen=True)
class ClaimVerdict:
    """Per-claim faithfulness verification result."""

    claim: str
    verdict: ClaimLabel
    support_score: float
    matched_context: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize claim verdict to a dictionary."""
        return asdict(self)


@dataclass
class FaithfulnessResult:
    """Aggregated claim-level faithfulness evaluation for one response."""

    score: float
    claims: list[ClaimVerdict]
    supported_count: int
    unsupported_count: int
    contradicted_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize faithfulness result to a dictionary."""
        return {
            "score": self.score,
            "supported_count": self.supported_count,
            "unsupported_count": self.unsupported_count,
            "contradicted_count": self.contradicted_count,
            "claims": [claim.to_dict() for claim in self.claims],
        }


def _has_number_match(target: float, candidates: Sequence[float], rel_tol: float = 1e-4) -> bool:
    return any(math.isclose(target, cand, rel_tol=rel_tol, abs_tol=1e-6) for cand in candidates)


def _negation_parity(text: str, extra_negated: bool = False) -> bool:
    """Return True when text carries negative polarity."""
    tokens = tokenize(text)
    neg_count = sum(1 for tok in tokens if tok in NEGATION_WORDS)
    if extra_negated:
        neg_count += 1
    return (neg_count % 2) == 1


def _has_antonym_conflict(claim_tokens: set[str], prop_tokens: set[str]) -> Optional[str]:
    for left, right in ANTONYM_PAIRS:
        if (left in claim_tokens and right in prop_tokens) or (
            right in claim_tokens and left in prop_tokens
        ):
            return f"antonym conflict ({left} vs {right})"
    return None


class FaithfulnessEvaluator:
    """Evaluates response faithfulness at the atomic claim level against reference context."""

    def __init__(
        self,
        support_threshold: float = 0.55,
        alignment_threshold: float = 0.25,
        contradiction_penalty: float = 0.20,
    ) -> None:
        self.support_threshold = support_threshold
        self.alignment_threshold = alignment_threshold
        self.contradiction_penalty = contradiction_penalty

    def _compute_overlap(
        self,
        claim_stems: Sequence[str],
        reference_stems: set[str],
        reference_bigrams: set[tuple[str, str]],
        idf_map: Dict[str, float],
        default_idf: float,
    ) -> float:
        unique_stems = list(dict.fromkeys(claim_stems))
        if not unique_stems:
            return 0.0

        matched = [s for s in unique_stems if s in reference_stems]
        if not matched:
            return 0.0

        token_cov = len(matched) / len(unique_stems)
        total_idf = sum(idf_map.get(s, default_idf) for s in unique_stems)
        matched_idf = sum(idf_map.get(s, default_idf) for s in matched)
        idf_overlap = matched_idf / total_idf if total_idf > 0.0 else token_cov

        claim_bgs = make_bigrams(claim_stems)
        bg_cov = (
            sum(1 for bg in claim_bgs if bg in reference_bigrams) / len(claim_bgs)
            if claim_bgs
            else token_cov
        )
        return float(0.40 * token_cov + 0.40 * idf_overlap + 0.20 * bg_cov)

    def evaluate(
        self,
        response: str,
        context: Union[str, Sequence[str]],
    ) -> FaithfulnessResult:
        """Split response into claims and check support, numeric consistency, and polarity."""
        context_list = [context] if isinstance(context, str) else list(context)
        context_list = [c for c in context_list if c and c.strip()]

        if not response or not response.strip():
            return FaithfulnessResult(
                score=0.0,
                claims=[],
                supported_count=0,
                unsupported_count=0,
                contradicted_count=0,
            )

        claims = split_claims(response)
        if not claims:
            return FaithfulnessResult(
                score=0.0,
                claims=[],
                supported_count=0,
                unsupported_count=0,
                contradicted_count=0,
            )

        if not context_list:
            verdicts = [
                ClaimVerdict(
                    claim=claim,
                    verdict="unsupported",
                    support_score=0.0,
                    matched_context=None,
                    reason="no reference context provided",
                )
                for claim in claims
            ]
            return FaithfulnessResult(
                score=0.0,
                claims=verdicts,
                supported_count=0,
                unsupported_count=len(verdicts),
                contradicted_count=0,
            )

        propositions = split_context_propositions(context_list)
        if not propositions:
            propositions = [(c, False) for c in context_list]

        idf_map = compute_idf_map([p[0] for p in propositions], stem=True)
        default_idf = max(idf_map.values(), default=1.0)

        all_ctx_stems: set[str] = set()
        all_ctx_bigrams: set[tuple[str, str]] = set()
        all_ctx_numbers: list[float] = []
        prop_index: list[tuple[str, bool, set[str], set[tuple[str, str]], list[float], set[str]]] = []

        for prop_text, has_contrast_neg in propositions:
            p_stems_seq = extract_content_tokens(prop_text, stem=True)
            p_stems = set(p_stems_seq)
            p_bigrams = set(make_bigrams(p_stems_seq))
            p_numbers = extract_numbers(prop_text)
            p_raw_tokens = set(tokenize(prop_text))

            all_ctx_stems.update(p_stems)
            all_ctx_bigrams.update(p_bigrams)
            all_ctx_numbers.extend(p_numbers)
            prop_index.append(
                (prop_text, has_contrast_neg, p_stems, p_bigrams, p_numbers, p_raw_tokens)
            )

        verdicts: list[ClaimVerdict] = []
        for claim in claims:
            c_stems_seq = extract_content_tokens(claim, stem=True)
            c_raw_tokens = set(tokenize(claim))
            c_numbers = extract_numbers(claim)
            c_neg = _negation_parity(claim)

            full_support = self._compute_overlap(
                c_stems_seq, all_ctx_stems, all_ctx_bigrams, idf_map, default_idf
            )

            best_prop_text: Optional[str] = None
            best_prop_overlap = -1.0
            best_prop_info = None
            for prop_info in prop_index:
                prop_text, _, p_stems, p_bigrams, _, _ = prop_info
                overlap = self._compute_overlap(
                    c_stems_seq, p_stems, p_bigrams, idf_map, default_idf
                )
                if overlap > best_prop_overlap:
                    best_prop_overlap = overlap
                    best_prop_text = prop_text
                    best_prop_info = prop_info

            missing_numbers = [
                num for num in c_numbers if not _has_number_match(num, all_ctx_numbers)
            ]

            contradiction_reason: Optional[str] = None
            if best_prop_info is not None and best_prop_overlap >= self.alignment_threshold:
                _, p_has_contrast_neg, _, _, p_numbers, p_raw_tokens = best_prop_info
                p_neg = _negation_parity(best_prop_text or "", extra_negated=p_has_contrast_neg)

                if missing_numbers and p_numbers:
                    contradiction_reason = (
                        f"numeric mismatch: claim states {missing_numbers} vs context {p_numbers}"
                    )
                elif c_neg != p_neg:
                    contradiction_reason = "negation polarity mismatch with aligned context"
                else:
                    antonym = _has_antonym_conflict(c_raw_tokens, p_raw_tokens)
                    if antonym is not None:
                        contradiction_reason = antonym

            if contradiction_reason is not None:
                verdicts.append(
                    ClaimVerdict(
                        claim=claim,
                        verdict="contradicted",
                        support_score=0.0,
                        matched_context=best_prop_text,
                        reason=contradiction_reason,
                    )
                )
            elif missing_numbers:
                verdicts.append(
                    ClaimVerdict(
                        claim=claim,
                        verdict="unsupported",
                        support_score=min(full_support, 0.49),
                        matched_context=best_prop_text if best_prop_overlap > 0 else None,
                        reason=f"ungrounded numeric claim(s): {missing_numbers}",
                    )
                )
            elif full_support >= self.support_threshold:
                verdicts.append(
                    ClaimVerdict(
                        claim=claim,
                        verdict="supported",
                        support_score=full_support,
                        matched_context=best_prop_text,
                        reason="entailed by reference context",
                    )
                )
            else:
                verdicts.append(
                    ClaimVerdict(
                        claim=claim,
                        verdict="unsupported",
                        support_score=full_support,
                        matched_context=best_prop_text if best_prop_overlap > 0 else None,
                        reason="insufficient lexical/semantic support in context",
                    )
                )

        supported = sum(1 for v in verdicts if v.verdict == "supported")
        unsupported = sum(1 for v in verdicts if v.verdict == "unsupported")
        contradicted = sum(1 for v in verdicts if v.verdict == "contradicted")
        total = len(verdicts)

        raw_score = (supported - self.contradiction_penalty * contradicted) / total
        score = float(max(0.0, min(1.0, raw_score)))
        return FaithfulnessResult(
            score=score,
            claims=verdicts,
            supported_count=supported,
            unsupported_count=unsupported,
            contradicted_count=contradicted,
        )

    def score(self, sample: EvalSample) -> FaithfulnessResult:
        """Convenience wrapper accepting an EvalSample."""
        return self.evaluate(sample.response, sample.context)


DEFAULT_RUBRIC_DESCRIPTIONS: Dict[str, str] = {
    "coherence": "Logical sentence structure, clarity, grammatical flow, and non-repetition.",
    "relevance": "Direct alignment with the user's question without off-topic drift.",
    "conciseness": "Information density without filler boilerplate or redundant phrasing.",
    "completeness": "Coverage of all essential facts, numbers, and nuances required by the question.",
}


@runtime_checkable
class JudgeBackend(Protocol):
    """Protocol for pluggable LLM or heuristic G-Eval scoring backends."""

    def score_criterion(
        self,
        criterion: str,
        rubric_description: str,
        question: str,
        response: str,
        context: Sequence[str],
        reference_answer: Optional[str] = None,
    ) -> float:
        """Score a single rubric criterion in [0.0, 1.0]."""
        ...


@dataclass
class GEvalResult:
    """Weighted rubric evaluation output for one model response."""

    overall_score: float
    criterion_scores: Dict[str, float]
    weights: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize G-Eval result to a dictionary."""
        return {
            "overall_score": self.overall_score,
            "criterion_scores": dict(self.criterion_scores),
            "weights": dict(self.weights),
        }


class LocalHeuristicJudgeBackend:
    """Deterministic offline heuristic implementation of the G-Eval JudgeBackend protocol."""

    def _score_coherence(self, response: str) -> float:
        tokens = tokenize(response)
        if not tokens:
            return 0.0

        sentences = split_sentences(response)
        if not sentences:
            return 0.0

        # Check proper sentence punctuation & readable sentence length
        terminated = sum(1 for s in sentences if s[-1] in ".!?") / len(sentences)
        lengths = [len(tokenize(s)) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        if 6.0 <= mean_len <= 32.0:
            len_score = 1.0
        elif mean_len < 6.0:
            len_score = max(0.3, mean_len / 6.0)
        else:
            len_score = max(0.3, 32.0 / mean_len)

        # Penalize repetitive tokens and duplicated bigrams
        bigrams = make_bigrams(tokens)
        bigram_diversity = (len(set(bigrams)) / len(bigrams)) if bigrams else 1.0
        lexical_diversity = min(1.0, (len(set(tokens)) / len(tokens)) * 1.35)

        return float(
            max(
                0.0,
                min(
                    1.0,
                    0.30 * terminated
                    + 0.25 * len_score
                    + 0.25 * bigram_diversity
                    + 0.20 * lexical_diversity,
                ),
            )
        )

    def _score_relevance(
        self,
        question: str,
        response: str,
        context: Sequence[str],
    ) -> float:
        q_stems = extract_content_tokens(question, stem=True)
        r_stems_seq = extract_content_tokens(response, stem=True)
        if not q_stems or not r_stems_seq:
            return 0.0

        r_stems = set(r_stems_seq)
        idf_map = compute_idf_map([question, *context, response], stem=True)
        default_idf = max(idf_map.values(), default=1.0)

        unique_q = list(dict.fromkeys(q_stems))
        matched_q = [t for t in unique_q if t in r_stems]
        if not matched_q:
            return 0.0

        q_cov = len(matched_q) / len(unique_q)
        total_idf = sum(idf_map.get(t, default_idf) for t in unique_q)
        matched_idf = sum(idf_map.get(t, default_idf) for t in matched_q)
        idf_cov = matched_idf / total_idf if total_idf > 0.0 else q_cov

        # Blend question coverage with on-topic focus against question + context vocabulary
        topic_vocab = set(q_stems)
        for ctx in context:
            topic_vocab.update(extract_content_tokens(ctx, stem=True))
        on_topic_ratio = (
            sum(1 for t in r_stems_seq if t in topic_vocab) / len(r_stems_seq)
            if topic_vocab
            else 1.0
        )
        return float(max(0.0, min(1.0, 0.45 * q_cov + 0.35 * idf_cov + 0.20 * on_topic_ratio)))

    def _score_conciseness(
        self,
        question: str,
        response: str,
        context: Sequence[str],
        reference_answer: Optional[str],
    ) -> float:
        tokens = tokenize(response)
        if not tokens:
            return 0.0

        lower = response.lower()
        filler_hits = sum(1 for phrase in _BOILERPLATE_PATTERNS if phrase in lower)
        filler_penalty = min(0.5, 0.25 * filler_hits)

        content_tokens = extract_content_tokens(response, stem=True)
        unique_ratio = len(set(content_tokens)) / len(content_tokens) if content_tokens else 0.0
        bigrams = make_bigrams(tokens)
        bigram_non_redundancy = (len(set(bigrams)) / len(bigrams)) if bigrams else 1.0

        ref_text = reference_answer or " ".join(context)
        ref_len = max(12, len(tokenize(ref_text)))
        resp_len = len(tokens)
        if resp_len <= ref_len * 1.4:
            brevity = 1.0
        else:
            brevity = max(0.25, (ref_len * 1.4) / resp_len)

        raw = 0.40 * unique_ratio + 0.30 * bigram_non_redundancy + 0.30 * brevity - filler_penalty
        return float(max(0.0, min(1.0, raw)))

    def _score_completeness(
        self,
        question: str,
        response: str,
        context: Sequence[str],
        reference_answer: Optional[str],
    ) -> float:
        r_stems = set(extract_content_tokens(response, stem=True))
        if not r_stems:
            return 0.0

        if reference_answer and reference_answer.strip():
            target_text = reference_answer
        else:
            q_stems = set(extract_content_tokens(question, stem=True))
            relevant_sents = []
            for ctx in context:
                for sent in split_sentences(ctx):
                    s_stems = set(extract_content_tokens(sent, stem=True))
                    if s_stems & q_stems:
                        relevant_sents.append(sent)
            target_text = " ".join(relevant_sents) if relevant_sents else " ".join(context)

        target_stems = list(dict.fromkeys(extract_content_tokens(target_text, stem=True)))
        if not target_stems:
            return 1.0 if r_stems else 0.0

        matched = sum(1 for t in target_stems if t in r_stems)
        concept_cov = matched / len(target_stems)

        target_nums = extract_numbers(target_text)
        resp_nums = extract_numbers(response)
        if target_nums:
            num_cov = sum(1 for n in target_nums if _has_number_match(n, resp_nums)) / len(
                target_nums
            )
            return float(max(0.0, min(1.0, 0.70 * concept_cov + 0.30 * num_cov)))
        return float(max(0.0, min(1.0, concept_cov)))

    def score_criterion(
        self,
        criterion: str,
        rubric_description: str,
        question: str,
        response: str,
        context: Sequence[str],
        reference_answer: Optional[str] = None,
    ) -> float:
        """Compute deterministic heuristic rubric score in [0.0, 1.0]."""
        key = criterion.strip().lower()
        if key == "coherence":
            return self._score_coherence(response)
        if key == "relevance":
            return self._score_relevance(question, response, context)
        if key == "conciseness":
            return self._score_conciseness(question, response, context, reference_answer)
        if key == "completeness":
            return self._score_completeness(question, response, context, reference_answer)

        # Fallback for custom user-defined rubric criteria: overlap with description + response quality
        desc_stems = set(extract_content_tokens(rubric_description, stem=True))
        resp_stems = set(extract_content_tokens(response, stem=True))
        if not resp_stems:
            return 0.0
        desc_cov = len(desc_stems & resp_stems) / len(desc_stems) if desc_stems else 0.5
        return float(max(0.0, min(1.0, 0.5 * self._score_coherence(response) + 0.5 * desc_cov)))


class GEvalScorer:
    """Configurable G-Eval rubric evaluator supporting local heuristics or pluggable LLM judges."""

    def __init__(
        self,
        weights: Optional[Mapping[str, float]] = None,
        rubric_descriptions: Optional[Mapping[str, str]] = None,
        backend: Optional[JudgeBackend] = None,
    ) -> None:
        raw_weights = dict(weights) if weights is not None else default_rubric_weights()
        if not raw_weights:
            raise ValueError("Rubric weights cannot be empty.")
        if any(w < 0.0 for w in raw_weights.values()):
            raise ValueError("Rubric weights must be non-negative.")

        total_weight = sum(raw_weights.values())
        if total_weight <= 0.0:
            raise ValueError("Sum of rubric weights must be strictly positive.")

        self.weights: Dict[str, float] = {
            k: float(v) / total_weight for k, v in raw_weights.items()
        }
        self.rubric_descriptions: Dict[str, str] = {
            **DEFAULT_RUBRIC_DESCRIPTIONS,
            **(dict(rubric_descriptions) if rubric_descriptions else {}),
        }
        self.backend: JudgeBackend = backend if backend is not None else LocalHeuristicJudgeBackend()

    def evaluate(
        self,
        question: str,
        response: str,
        context: Union[str, Sequence[str]] = (),
        reference_answer: Optional[str] = None,
    ) -> GEvalResult:
        """Evaluate a response across all configured rubric criteria."""
        ctx_list = [context] if isinstance(context, str) else list(context)
        criterion_scores: Dict[str, float] = {}
        overall = 0.0

        for criterion, weight in self.weights.items():
            description = self.rubric_descriptions.get(criterion, criterion)
            raw_score = float(
                self.backend.score_criterion(
                    criterion=criterion,
                    rubric_description=description,
                    question=question,
                    response=response,
                    context=ctx_list,
                    reference_answer=reference_answer,
                )
            )
            clamped = max(0.0, min(1.0, raw_score))
            criterion_scores[criterion] = clamped
            overall += weight * clamped

        return GEvalResult(
            overall_score=float(max(0.0, min(1.0, overall))),
            criterion_scores=criterion_scores,
            weights=dict(self.weights),
        )

    def score(self, sample: EvalSample) -> GEvalResult:
        """Convenience wrapper accepting an EvalSample."""
        return self.evaluate(
            question=sample.question,
            response=sample.response,
            context=sample.context,
            reference_answer=sample.reference_answer,
        )


@dataclass(frozen=True)
class SampleEvaluation:
    """Combined faithfulness and G-Eval scores for a single sample."""

    sample_id: str
    model: str
    question: str
    faithfulness: FaithfulnessResult
    g_eval: GEvalResult
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        """Serialize per-sample evaluation result."""
        return {
            "sample_id": self.sample_id,
            "model": self.model,
            "question": self.question,
            "passed": self.passed,
            "faithfulness": self.faithfulness.to_dict(),
            "g_eval": self.g_eval.to_dict(),
        }


@dataclass(frozen=True)
class MetricStats:
    """Mean, minimum, and maximum summary statistics for one metric."""

    mean: float
    min: float
    max: float

    def to_dict(self) -> Dict[str, float]:
        """Serialize metric stats."""
        return asdict(self)


@dataclass
class LeaderboardEntry:
    """Aggregated evaluation summary for a single model."""

    rank: int
    model: str
    sample_count: int
    composite_score: float
    pass_rate: float
    faithfulness: MetricStats
    g_eval: MetricStats
    criterion_means: Dict[str, float]
    claim_summary: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize leaderboard row to a JSON-compatible dictionary."""
        return {
            "rank": self.rank,
            "model": self.model,
            "sample_count": self.sample_count,
            "composite_score": round(self.composite_score, 4),
            "pass_rate": round(self.pass_rate, 4),
            "faithfulness": {k: round(v, 4) for k, v in self.faithfulness.to_dict().items()},
            "g_eval": {k: round(v, 4) for k, v in self.g_eval.to_dict().items()},
            "criterion_means": {k: round(v, 4) for k, v in self.criterion_means.items()},
            "claim_summary": dict(self.claim_summary),
        }


def _compute_stats(values: Sequence[float]) -> MetricStats:
    if not values:
        return MetricStats(mean=0.0, min=0.0, max=0.0)
    return MetricStats(
        mean=float(sum(values) / len(values)),
        min=float(min(values)),
        max=float(max(values)),
    )


class LeaderboardGenerator:
    """Aggregates per-model scores and exports JSON, Markdown, and console leaderboards."""

    def __init__(
        self,
        faithfulness_threshold: float = 0.70,
        geval_threshold: float = 0.60,
        faithfulness_weight: float = 0.50,
        geval_weight: float = 0.50,
    ) -> None:
        total_w = faithfulness_weight + geval_weight
        if total_w <= 0.0:
            raise ValueError("Composite score weights must sum to a positive value.")
        self.faithfulness_threshold = faithfulness_threshold
        self.geval_threshold = geval_threshold
        self.faithfulness_weight = faithfulness_weight / total_w
        self.geval_weight = geval_weight / total_w

    def is_passing(
        self,
        faithfulness_score: float,
        geval_score: float,
    ) -> bool:
        """Determine whether a sample meets both faithfulness and G-Eval thresholds."""
        return (
            faithfulness_score >= self.faithfulness_threshold
            and geval_score >= self.geval_threshold
        )

    def aggregate(
        self,
        evaluations: Sequence[SampleEvaluation],
    ) -> list[LeaderboardEntry]:
        """Compute per-model mean/min/max/pass-rate summaries ranked descending."""
        if not evaluations:
            return []

        grouped: Dict[str, list[SampleEvaluation]] = {}
        for item in evaluations:
            grouped.setdefault(item.model, []).append(item)

        entries: list[LeaderboardEntry] = []
        for model_name, items in grouped.items():
            faith_scores = [ev.faithfulness.score for ev in items]
            geval_scores = [ev.g_eval.overall_score for ev in items]
            faith_stats = _compute_stats(faith_scores)
            geval_stats = _compute_stats(geval_scores)

            pass_count = sum(1 for ev in items if ev.passed)
            pass_rate = pass_count / len(items)
            composite = (
                self.faithfulness_weight * faith_stats.mean
                + self.geval_weight * geval_stats.mean
            )

            all_criteria = list(dict.fromkeys(
                crit for ev in items for crit in ev.g_eval.criterion_scores
            ))
            criterion_means = {
                crit: float(
                    sum(ev.g_eval.criterion_scores.get(crit, 0.0) for ev in items)
                    / len(items)
                )
                for crit in all_criteria
            }

            claim_summary = {
                "supported": sum(ev.faithfulness.supported_count for ev in items),
                "unsupported": sum(ev.faithfulness.unsupported_count for ev in items),
                "contradicted": sum(ev.faithfulness.contradicted_count for ev in items),
            }

            entries.append(
                LeaderboardEntry(
                    rank=0,
                    model=model_name,
                    sample_count=len(items),
                    composite_score=composite,
                    pass_rate=pass_rate,
                    faithfulness=faith_stats,
                    g_eval=geval_stats,
                    criterion_means=criterion_means,
                    claim_summary=claim_summary,
                )
            )

        entries.sort(
            key=lambda entry: (
                -entry.composite_score,
                -entry.faithfulness.mean,
                -entry.g_eval.mean,
                entry.model,
            )
        )
        for rank, entry in enumerate(entries, start=1):
            entry.rank = rank
        return entries

    def to_dict(
        self,
        entries: Sequence[LeaderboardEntry],
        evaluations: Optional[Sequence[SampleEvaluation]] = None,
    ) -> Dict[str, Any]:
        """Convert leaderboard entries (and optional sample evaluations) to structured JSON dict."""
        payload: Dict[str, Any] = {
            "metadata": {
                "model_count": len(entries),
                "sample_count": sum(e.sample_count for e in entries),
                "faithfulness_threshold": self.faithfulness_threshold,
                "geval_threshold": self.geval_threshold,
            },
            "models": [entry.to_dict() for entry in entries],
        }
        if evaluations is not None:
            payload["samples"] = [ev.to_dict() for ev in evaluations]
        return payload

    def to_json(
        self,
        entries: Sequence[LeaderboardEntry],
        evaluations: Optional[Sequence[SampleEvaluation]] = None,
        indent: int = 2,
    ) -> str:
        """Render leaderboard report as formatted JSON text."""
        return json.dumps(self.to_dict(entries, evaluations=evaluations), indent=indent)

    def export_json(
        self,
        entries: Sequence[LeaderboardEntry],
        path: Union[str, Path],
        evaluations: Optional[Sequence[SampleEvaluation]] = None,
    ) -> Path:
        """Write structured leaderboard JSON to disk (creating leaderboard.json if given a dir)."""
        target = Path(path)
        if target.suffix.lower() != ".json":
            target = target / "leaderboard.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            self.to_json(entries, evaluations=evaluations) + "\n",
            encoding="utf-8",
        )
        return target

    def to_markdown(self, entries: Sequence[LeaderboardEntry]) -> str:
        """Render leaderboard entries as a formatted Markdown report."""
        if not entries:
            return "# LLM Evaluation Leaderboard\n\n_No evaluation samples found._\n"

        lines = [
            "# LLM Evaluation Leaderboard",
            "",
            "| Rank | Model | Samples | Composite | Faithfulness (Mean / Min–Max) | G-Eval (Mean / Min–Max) | Pass Rate | Claims (Sup/Unsup/Contra) |",
            "|---:|---|---:|---:|---|---|---:|---|",
        ]
        for entry in entries:
            claims = entry.claim_summary
            claims_str = (
                f"{claims.get('supported', 0)}/"
                f"{claims.get('unsupported', 0)}/"
                f"{claims.get('contradicted', 0)}"
            )
            lines.append(
                f"| {entry.rank} | {entry.model} | {entry.sample_count} | "
                f"{entry.composite_score:.4f} | "
                f"{entry.faithfulness.mean:.4f} ({entry.faithfulness.min:.2f}–{entry.faithfulness.max:.2f}) | "
                f"{entry.g_eval.mean:.4f} ({entry.g_eval.min:.2f}–{entry.g_eval.max:.2f}) | "
                f"{entry.pass_rate * 100:.1f}% | {claims_str} |"
            )

        criteria = list(dict.fromkeys(c for e in entries for c in e.criterion_means))
        if criteria:
            header_cols = " | ".join(c.capitalize() for c in criteria)
            sep_cols = "|".join("---:" for _ in criteria)
            lines.extend([
                "",
                "## G-Eval Rubric Breakdown",
                "",
                f"| Rank | Model | {header_cols} |",
                f"|---:|---|{sep_cols}|",
            ])
            for entry in entries:
                vals = " | ".join(f"{entry.criterion_means.get(c, 0.0):.4f}" for c in criteria)
                lines.append(f"| {entry.rank} | {entry.model} | {vals} |")

        return "\n".join(lines) + "\n"

    def export_markdown(
        self,
        entries: Sequence[LeaderboardEntry],
        path: Union[str, Path],
    ) -> Path:
        """Write formatted Markdown leaderboard to disk."""
        target = Path(path)
        if target.is_dir() or not target.suffix:
            target = target / "leaderboard.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_markdown(entries), encoding="utf-8")
        return target

    def format_console_table(self, entries: Sequence[LeaderboardEntry]) -> str:
        """Format a crisp fixed-width ASCII table for terminal output."""
        if not entries:
            return "No evaluation samples to display."

        headers = [
            "Rank",
            "Model",
            "N",
            "Composite",
            "Faithfulness (Mean/Min/Max)",
            "G-Eval (Mean/Min/Max)",
            "Pass Rate",
            "Claims S/U/C",
        ]
        rows = []
        for entry in entries:
            c = entry.claim_summary
            rows.append([
                str(entry.rank),
                entry.model,
                str(entry.sample_count),
                f"{entry.composite_score:.4f}",
                f"{entry.faithfulness.mean:.4f} ({entry.faithfulness.min:.2f}/{entry.faithfulness.max:.2f})",
                f"{entry.g_eval.mean:.4f} ({entry.g_eval.min:.2f}/{entry.g_eval.max:.2f})",
                f"{entry.pass_rate * 100:.1f}%",
                f"{c.get('supported', 0)}/{c.get('unsupported', 0)}/{c.get('contradicted', 0)}",
            ])

        widths = [
            max(len(headers[col]), *(len(row[col]) for row in rows))
            for col in range(len(headers))
        ]
        fmt = " | ".join(
            f"{{:<{w}}}" if idx == 1 else f"{{:>{w}}}"
            for idx, w in enumerate(widths)
        )
        divider = "-+-".join("-" * w for w in widths)
        lines = [fmt.format(*headers), divider]
        lines.extend(fmt.format(*row) for row in rows)
        return "\n".join(lines)
