# LLM Evaluation Harness

A modular, deterministic evaluation harness for Large Language Model (LLM) and RAG outputs, featuring claim-level faithfulness verification, configurable G-Eval rubric scoring, and JSON/Markdown leaderboard reporting.

## Features

- **Claim-Level Faithfulness (`FaithfulnessEvaluator`)**:
  - Splits model responses into atomic claims and clauses while preserving decimals and abbreviations.
  - Compares each claim against reference context passages using stemmed token coverage, bigram overlap, and smoothed IDF weighting.
  - Detects **numeric contradictions** (e.g., hallucinated dates, metrics, or parameters) and **negation/contrast polarity or antonym conflicts**, assigning per-claim verdicts (`supported`, `unsupported`, `contradicted`) and an overall faithfulness score in `[0.0, 1.0]`.
- **Configurable G-Eval Rubric Scoring (`GEvalScorer`)**:
  - Evaluates `coherence`, `relevance`, `conciseness`, and `completeness` with customizable rubric weights.
  - Ships with a deterministic offline `LocalHeuristicJudgeBackend` and a clean `JudgeBackend` protocol for plugging in OpenAI, Ollama, or local LLM judges with zero hardcoded credentials.
- **Leaderboard Aggregation & Export (`LeaderboardGenerator`)**:
  - Aggregates per-model `mean`, `min`, `max`, `pass_rate`, composite rank, rubric criterion breakdowns, and `supported/unsupported/contradicted` claim totals.
  - Exports structured JSON (`leaderboard.json`), formatted Markdown tables, and aligned console tables.
- **Bundled Sample Dataset**: Includes `data/sample_eval.jsonl` comparing 3 models (`grounded-rag-v2`, `concise-mini-v1`, `unconstrained-draft-v1`) across 5 QA prompts covering grounded, incomplete, and hallucinated-number responses.

## Architecture

```text
JSONL / JSON Samples (question, context, response, reference_answer)
         │
         ├──► FaithfulnessEvaluator (claim splitting, IDF overlap, numeric & polarity checks)
         │
         └──► GEvalScorer (coherence, relevance, conciseness, completeness; JudgeBackend protocol)
                   │
                   ▼
         LeaderboardGenerator (mean/min/max/pass-rate aggregation)
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
  Console / Markdown    leaderboard.json
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the evaluation harness directly against the bundled benchmark dataset:

```bash
python main.py
```

Or specify custom paths and export formats:

```bash
python main.py \
  --input-data data/sample_eval.jsonl \
  --output-path outputs/leaderboard.json \
  --markdown outputs/leaderboard.md
```

Supported CLI options:

- `--input-data`: Path to `.jsonl` or `.json` evaluation records (default: `data/sample_eval.jsonl`)
- `--output-path`: Optional file or directory path for structured JSON output
- `--markdown`: Optional file or directory path for Markdown leaderboard output
- `--json`: Print structured JSON output to stdout
- `--faithfulness-threshold`: Minimum faithfulness score for passing (default: `0.70`)
- `--geval-threshold`: Minimum G-Eval score for passing (default: `0.60`)

### Sample CLI Output

```text
$ python main.py
Evaluated 15 samples across 3 models.

Rank | Model                  | N | Composite | Faithfulness (Mean/Min/Max) | G-Eval (Mean/Min/Max) | Pass Rate | Claims S/U/C
-----+------------------------+---+-----------+-----------------------------+-----------------------+-----------+-------------
   1 | grounded-rag-v2        | 5 |    0.9404 |          1.0000 (1.00/1.00) |    0.8808 (0.81/0.92) |    100.0% |        9/0/0
   2 | concise-mini-v1        | 5 |    0.8049 |          1.0000 (1.00/1.00) |    0.6097 (0.56/0.70) |     40.0% |        5/0/0
   3 | unconstrained-draft-v1 | 5 |    0.3855 |          0.0800 (0.00/0.40) |    0.6910 (0.61/0.72) |      0.0% |        1/0/5
```

## Project Layout

```text
llm-eval-harness/
├── config.py
├── main.py
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── data/
│   └── sample_eval.jsonl
├── src/
│   ├── __init__.py
│   ├── models.py
│   ├── pipeline.py
│   └── utils.py
└── tests/
    ├── test_faithfulness.py
    ├── test_geval_and_leaderboard.py
    └── test_pipeline_and_cli.py
```

## Testing

```bash
pytest -v
```
