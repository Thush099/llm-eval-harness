"""Configuration dataclasses and constants for the LLM evaluation harness."""

from dataclasses import dataclass, field
import logging
import math
from typing import Dict, Optional


def default_rubric_weights() -> Dict[str, float]:
    """Return default G-Eval criterion weights summing to 1.0."""
    return {
        "coherence": 0.25,
        "relevance": 0.30,
        "conciseness": 0.20,
        "completeness": 0.25,
    }


@dataclass
class EvalHarnessConfig:
    """Configuration for faithfulness, G-Eval rubric scoring, and leaderboard reporting."""

    input_data_path: str = "data/sample_eval.jsonl"
    output_path: Optional[str] = None
    markdown_path: Optional[str] = None
    faithfulness_threshold: float = 0.70
    geval_threshold: float = 0.60
    faithfulness_support_threshold: float = 0.55
    composite_faithfulness_weight: float = 0.50
    composite_geval_weight: float = 0.50
    rubric_weights: Dict[str, float] = field(default_factory=default_rubric_weights)

    def __post_init__(self) -> None:
        for name, value in (
            ("faithfulness_threshold", self.faithfulness_threshold),
            ("geval_threshold", self.geval_threshold),
            ("faithfulness_support_threshold", self.faithfulness_support_threshold),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")

        for name, value in (
            ("composite_faithfulness_weight", self.composite_faithfulness_weight),
            ("composite_geval_weight", self.composite_geval_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        composite_weight_sum = self.composite_faithfulness_weight + self.composite_geval_weight
        if not math.isfinite(composite_weight_sum) or composite_weight_sum <= 0.0:
            raise ValueError("Composite score weights must sum to a finite positive value.")

        if not self.rubric_weights:
            raise ValueError("Rubric weights cannot be empty.")
        for name, value in self.rubric_weights.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"Rubric weight for '{name}' must be finite and non-negative")
        rubric_weight_sum = sum(self.rubric_weights.values())
        if not math.isfinite(rubric_weight_sum) or rubric_weight_sum <= 0.0:
            raise ValueError("Sum of rubric weights must be finite and strictly positive.")


class Constants:
    """Logging and formatting constants."""

    LOG_LEVEL = logging.INFO


class Config:
    """Default application configuration."""

    EVAL_HARNESS = EvalHarnessConfig()
