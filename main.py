"""CLI entrypoint for running the LLM evaluation harness."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Sequence

from config import EvalHarnessConfig
from src.models import LeaderboardEntry
from src.pipeline import Pipeline
from src.utils import load_data

DEFAULT_INPUT_DATA = Path(__file__).resolve().parent / "data" / "sample_eval.jsonl"


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the LLM evaluation harness CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM responses with claim-level faithfulness and G-Eval rubric scoring."
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default=str(DEFAULT_INPUT_DATA),
        help="Path to input JSONL or JSON evaluation file (default: data/sample_eval.jsonl).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional path to write structured leaderboard JSON (file or directory).",
    )
    parser.add_argument(
        "--markdown",
        type=str,
        default=None,
        help="Optional path to write Markdown leaderboard report.",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print structured JSON leaderboard output to stdout instead of table.",
    )
    parser.add_argument(
        "--faithfulness-threshold",
        type=float,
        default=0.70,
        help="Minimum faithfulness score required for a sample to pass.",
    )
    parser.add_argument(
        "--geval-threshold",
        type=float,
        default=0.60,
        help="Minimum G-Eval score required for a sample to pass.",
    )
    parser.add_argument(
        "--min-composite",
        type=float,
        default=None,
        help=(
            "Exit non-zero when any model's composite leaderboard score is below "
            "this threshold in [0, 1], so the harness can gate a CI job."
        ),
    )
    return parser


def exit_code_for(leaderboard: Sequence[LeaderboardEntry], min_composite: Optional[float]) -> int:
    """Map a leaderboard to a process exit code under a score gate.

    Returns 0 when no threshold is given or every model meets it, and 1 when any
    model's composite score is below the threshold. The threshold must be a
    finite value in [0, 1]; anything else raises ``ValueError`` so a typo cannot
    silently disable gating.
    """
    if min_composite is None:
        return 0
    if not math.isfinite(min_composite) or not 0.0 <= min_composite <= 1.0:
        raise ValueError(
            "min_composite must be a finite value in [0, 1]; "
            f"got {min_composite!r}"
        )
    return 1 if any(entry.composite_score < min_composite for entry in leaderboard) else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run evaluation harness from CLI arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)

    samples = load_data(args.input_data)
    config = EvalHarnessConfig(
        input_data_path=args.input_data,
        output_path=args.output_path,
        markdown_path=args.markdown,
        faithfulness_threshold=args.faithfulness_threshold,
        geval_threshold=args.geval_threshold,
    )
    pipeline = Pipeline(config=config)
    output = pipeline.run(samples)

    if args.output_path:
        pipeline.leaderboard_generator.export_json(
            output.leaderboard,
            args.output_path,
            evaluations=output.evaluations,
        )
    if args.markdown:
        pipeline.leaderboard_generator.export_markdown(
            output.leaderboard,
            args.markdown,
        )

    if args.json_output:
        print(
            pipeline.leaderboard_generator.to_json(
                output.leaderboard, evaluations=output.evaluations
            )
        )
    else:
        print(
            f"Evaluated {len(samples)} samples across {len(output.leaderboard)} models.\n"
        )
        print(pipeline.leaderboard_generator.format_console_table(output.leaderboard))
    return exit_code_for(output.leaderboard, args.min_composite)


if __name__ == "__main__":
    raise SystemExit(main())
