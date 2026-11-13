"""Configuration dataclasses and constants for the LLM evaluation harness."""

from dataclasses import dataclass, field
import logging
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


class Constants:
    """Logging and formatting constants."""

    LOG_LEVEL = logging.INFO


class Config:
    """Default application configuration."""

    EVAL_HARNESS = EvalHarnessConfig()
