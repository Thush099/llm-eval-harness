# LLM Evaluation Harness
## Introduction
This project provides a comprehensive evaluation harness for Large Language Models (LLMs) using G-EVAL and faithfulness scoring.

## Problem Statement
Evaluating LLMs is a complex task that requires careful consideration of various metrics and scoring systems. This project aims to provide a standardized evaluation framework for LLMs.

## Architecture
```
+---------------+
|  Input Data  |
+---------------+
        |
        |
        v
+---------------+
|  Data Preprocessing  |
+---------------+
        |
        |
        v
+---------------+
|  G-EVAL Scoring  |
+---------------+
        |
        |
        v
+---------------+
|  Faithfulness Scoring  |
+---------------+
        |
        |
        v
+---------------+
|  Leaderboard Generation  |
+---------------+
        |
        |
        v
+---------------+
|  Output Results  |
+---------------+
```

## Installation
To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

## Usage
To run the evaluation harness, use the following command:
```bash
python main.py --input-data path/to/input/data --output-path path/to/output/results
```

## Sample Output
The evaluation harness will generate a leaderboard output in the following format:
```json
{
  "models": [
    {
      "name": "Model 1",
      "g-eval-score": 0.8,
      "faithfulness-score": 0.9
    },
    {
      "name": "Model 2",
      "g-eval-score": 0.7,
      "faithfulness-score": 0.8
    }
  ]
}

## Design Decisions
The evaluation harness uses a modular design, with separate components for data preprocessing, G-EVAL scoring, faithfulness scoring, and leaderboard generation. This allows for easy extension and modification of the evaluation framework.
