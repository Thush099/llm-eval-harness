"""End-to-end LLM evaluation pipeline orchestrating Faithfulness, G-Eval, and Leaderboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from config import EvalHarnessConfig
from src.models import (
    FaithfulnessEvaluator,
    GEvalScorer,
    JudgeBackend,
    LeaderboardEntry,
    LeaderboardGenerator,
    SampleEvaluation,
)
from src.utils import EvalSample


@dataclass
class EvaluationOutput:
    """Complete result returned by Pipeline.run()."""

    leaderboard: List[LeaderboardEntry]
    evaluations: List[SampleEvaluation]
    report_dict: Dict[str, Any]
    markdown: str


class Pipeline:
    """Evaluates LLM outputs across faithfulness and G-Eval criteria."""

    def __init__(
        self,
        config: Optional[EvalHarnessConfig] = None,
        faithfulness_evaluator: Optional[FaithfulnessEvaluator] = None,
        g_eval_scorer: Optional[GEvalScorer] = None,
        leaderboard_generator: Optional[LeaderboardGenerator] = None,
        judge_backend: Optional[JudgeBackend] = None,
    ) -> None:
        self.config = config or EvalHarnessConfig()
        self.faithfulness_evaluator = faithfulness_evaluator or FaithfulnessEvaluator(
            support_threshold=self.config.faithfulness_support_threshold,
        )
        self.g_eval_scorer = g_eval_scorer or GEvalScorer(
            weights=self.config.rubric_weights,
            backend=judge_backend,
        )
        self.leaderboard_generator = leaderboard_generator or LeaderboardGenerator(
            faithfulness_threshold=self.config.faithfulness_threshold,
            geval_threshold=self.config.geval_threshold,
            faithfulness_weight=self.config.composite_faithfulness_weight,
            geval_weight=self.config.composite_geval_weight,
        )

    def evaluate_samples(self, samples: Sequence[EvalSample]) -> List[SampleEvaluation]:
        """Run faithfulness and G-Eval scoring for each sample."""
        evaluations: List[SampleEvaluation] = []
        for sample in samples:
            faith_res = self.faithfulness_evaluator.score(sample)
            geval_res = self.g_eval_scorer.score(sample)
            passed = self.leaderboard_generator.is_passing(
                faithfulness_score=faith_res.score,
                geval_score=geval_res.overall_score,
            )
            evaluations.append(
                SampleEvaluation(
                    sample_id=sample.sample_id,
                    model=sample.model,
                    question=sample.question,
                    faithfulness=faith_res,
                    g_eval=geval_res,
                    passed=passed,
                )
            )
        return evaluations

    def run(self, samples: Sequence[EvalSample]) -> EvaluationOutput:
        """Evaluate samples and build structured leaderboard outputs."""
        evaluations = self.evaluate_samples(samples)
        leaderboard = self.leaderboard_generator.aggregate(evaluations)
        report_dict = self.leaderboard_generator.to_dict(
            leaderboard, evaluations=evaluations
        )
        markdown = self.leaderboard_generator.to_markdown(leaderboard)
        return EvaluationOutput(
            leaderboard=leaderboard,
            evaluations=evaluations,
            report_dict=report_dict,
            markdown=markdown,
        )
