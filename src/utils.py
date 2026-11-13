"""Text normalization, claim splitting, numeric extraction, and I/O helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "how",
    "in", "into", "is", "it", "its", "may", "might", "more", "most", "of",
    "on", "or", "our", "should", "such", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "to", "was",
    "we", "were", "what", "when", "where", "which", "who", "whom", "why",
    "will", "with", "would", "so", "also", "often", "typically", "usually",
})

NEGATION_WORDS = frozenset({
    "not", "no", "never", "neither", "nor", "without", "cannot", "cant",
    "doesnt", "dont", "didnt", "isnt", "arent", "wasnt", "werent", "wont",
    "wouldnt", "shouldnt", "couldnt", "fails", "fail", "failed", "unable",
    "lack", "lacks", "lacking", "impossible", "false", "absent", "none",
    "prevent", "prevents", "prevented", "preventing",
    "avoid", "avoids", "avoided", "avoiding",
})

ANTONYM_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("increase", "decrease"),
    ("increases", "decreases"),
    ("increased", "decreased"),
    ("higher", "lower"),
    ("more", "less"),
    ("faster", "slower"),
    ("always", "never"),
    ("true", "false"),
    ("enable", "disable"),
    ("enables", "disables"),
    ("enabled", "disabled"),
    ("support", "prevent"),
    ("supports", "prevents"),
    ("include", "exclude"),
    ("includes", "excludes"),
    ("required", "optional"),
    ("exact", "approximate"),
    ("early", "late"),
    ("before", "after"),
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ABBREVIATION_RE = re.compile(
    r"\b(?:e\.g|i\.e|vs|etc|fig|dr|mr|mrs|ms|prof|inc|ltd|al)\.",
    re.IGNORECASE,
)
_DECIMAL_RE = re.compile(r"(?<=\d)\.(?=\d)")
_NUMBER_RE = re.compile(
    r"(?<![a-zA-Z0-9.])"
    r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"(?:\s*(%|percent|thousand|million|billion|k|m|b))?"
    r"(?![a-zA-Z0-9]|\.\d)",
    re.IGNORECASE,
)
_CONTRAST_CLAUSE_RE = re.compile(
    r"\s+(?:rather than|instead of|as opposed to)\s+",
    re.IGNORECASE,
)
_CONJUNCTION_SPLIT_RE = re.compile(
    r"(?:;\s+|,\s+(?:however|whereas|while|but|so)\s+|\s+because\s+|\s+so\s+(?:that\s+)?(?!much\b|many\b|fast\b|well\b|far\b|long\b|often\b|little\b|few\b|soon\b|on\b))",
    re.IGNORECASE,
)
_CONTRACTION_RE = re.compile(r"['’]")


@dataclass(frozen=True)
class EvalSample:
    """Single model evaluation sample loaded from JSONL or JSON input."""

    sample_id: str
    model: str
    question: str
    context: List[str]
    response: str
    reference_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def context_text(self) -> str:
        """Return joined context passages."""
        return "\n".join(c for c in self.context if c).strip()


def tokenize(text: str) -> List[str]:
    """Lowercase and extract alphanumeric tokens, normalizing contractions."""
    if not text:
        return []
    cleaned = _CONTRACTION_RE.sub("", text.lower())
    return _TOKEN_RE.findall(cleaned)


def stem_token(token: str) -> str:
    """Apply conservative suffix normalization for lexical alignment."""
    if len(token) <= 3 or token.isdigit():
        return token
    stem = token
    if len(stem) > 4 and stem.endswith("ies"):
        stem = f"{stem[:-3]}y"
    elif stem.endswith(("ches", "shes", "sses", "xes", "zes")) and len(stem) > 5:
        stem = stem[:-2]
    elif stem.endswith("s") and not stem.endswith("ss") and len(stem) > 4:
        stem = stem[:-1]

    if len(stem) > 5 and stem.endswith("ing"):
        stem = stem[:-3]
    elif len(stem) > 4 and stem.endswith("ed"):
        stem = stem[:-2]

    if len(stem) > 4 and stem.endswith("e"):
        stem = stem[:-1]
    return stem


def extract_content_tokens(text: str, stem: bool = False) -> List[str]:
    """Extract non-stopword tokens; optionally apply light stemming."""
    tokens = tokenize(text)
    content = [t for t in tokens if t not in STOPWORDS]
    chosen = content if content else tokens
    return [stem_token(t) for t in chosen] if stem else chosen


def make_bigrams(tokens: Sequence[str]) -> List[Tuple[str, str]]:
    """Return ordered adjacent token pairs."""
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


def split_sentences(text: str) -> List[str]:
    """Split text into sentences while preserving decimals and abbreviations."""
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return []

    placeholder = "\u0000"
    masked = _DECIMAL_RE.sub(placeholder, cleaned)
    masked = _ABBREVIATION_RE.sub(
        lambda m: m.group(0).replace(".", placeholder),
        masked,
    )
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])", masked)
    return [
        part.replace(placeholder, ".").strip()
        for part in parts
        if part.replace(placeholder, ".").strip()
    ]


def split_claims(text: str) -> List[str]:
    """Decompose a response into atomic claims/clauses for faithfulness checks."""
    claims: List[str] = []
    for sentence in split_sentences(text):
        sub_parts = _CONJUNCTION_SPLIT_RE.split(sentence)
        accepted = [part.strip(" ,;") for part in sub_parts if part.strip(" ,;")]
        if len(accepted) > 1 and all(len(tokenize(p)) >= 4 for p in accepted):
            claims.extend(accepted)
        else:
            claims.append(sentence.strip())
    return [c for c in claims if tokenize(c)]


def split_context_propositions(context: Sequence[str]) -> List[Tuple[str, bool]]:
    """Split context passages into sentences and contrastive sub-clauses.

    Returns tuples of (proposition_text, has_contrast_negation_prefix).
    """
    propositions: List[Tuple[str, bool]] = []
    for passage in context:
        for sentence in split_sentences(passage):
            contrast_parts = _CONTRAST_CLAUSE_RE.split(sentence)
            if len(contrast_parts) == 1:
                propositions.append((sentence, False))
            else:
                for idx, part in enumerate(contrast_parts):
                    stripped = part.strip(" ,.")
                    if stripped:
                        propositions.append((stripped, idx > 0))
    return propositions


def extract_numbers(text: str) -> List[float]:
    """Extract normalized numeric values (percentages, commas, magnitude suffixes)."""
    if not text:
        return []

    multipliers = {
        "k": 1_000.0,
        "thousand": 1_000.0,
        "m": 1_000_000.0,
        "million": 1_000_000.0,
        "b": 1_000_000_000.0,
        "billion": 1_000_000_000.0,
    }
    numbers: List[float] = []
    for match in _NUMBER_RE.finditer(text):
        raw_num = match.group(1).replace(",", "")
        unit = (match.group(2) or "").lower()
        try:
            val = float(raw_num)
        except ValueError:
            continue
        if unit in multipliers:
            val *= multipliers[unit]
        numbers.append(val)
    return numbers


def compute_idf_map(texts: Iterable[str], stem: bool = True) -> Dict[str, float]:
    """Compute smoothed IDF weights across texts."""
    doc_freqs: Counter[str] = Counter()
    n_docs = 0
    for text in texts:
        n_docs += 1
        tokens = extract_content_tokens(text, stem=stem)
        doc_freqs.update(set(tokens))

    if n_docs == 0:
        return {}

    return {
        term: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        for term, df in doc_freqs.items()
    }


def _normalize_sample(record: Any, index: int) -> EvalSample:
    if not isinstance(record, Mapping):
        raise ValueError(f"Expected JSON object at record {index}, got {type(record).__name__}.")

    model = record.get("model")
    question = record.get("question")
    response = record.get("response", record.get("answer"))
    raw_context = record.get("context", record.get("contexts"))

    missing = [
        name
        for name, val in (
            ("model", model),
            ("question", question),
            ("context", raw_context),
            ("response", response),
        )
        if val is None
    ]
    if missing:
        raise ValueError(
            f"Record {index} missing required field(s): {', '.join(missing)}."
        )

    if isinstance(raw_context, str):
        context = [raw_context] if raw_context.strip() else []
    elif isinstance(raw_context, list):
        context = [str(item) for item in raw_context if str(item).strip()]
    else:
        raise ValueError(
            f"Record {index} field 'context' must be a string or list of strings."
        )

    sample_id = str(record.get("id") or record.get("sample_id") or f"sample-{index}")
    ref = record.get("reference_answer")
    known = {"id", "sample_id", "model", "question", "context", "contexts", "response", "answer", "reference_answer"}
    metadata = {k: v for k, v in record.items() if k not in known}

    return EvalSample(
        sample_id=sample_id,
        model=str(model),
        question=str(question),
        context=context,
        response=str(response),
        reference_answer=str(ref) if ref is not None else None,
        metadata=metadata,
    )


def load_data(path: Union[str, Path]) -> List[EvalSample]:
    """Load evaluation samples from a JSONL or JSON file."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Input data file not found: {data_path}")

    raw_text = data_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    if data_path.suffix.lower() == ".json":
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in {data_path}: {exc}") from exc
        records = payload.get("samples", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON array or 'samples' list in {data_path}.")
        return [_normalize_sample(rec, idx) for idx, rec in enumerate(records, start=1)]

    samples: List[EvalSample] = []
    for line_no, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSONL in {data_path} at line {line_no}: {exc}"
            ) from exc
        samples.append(_normalize_sample(record, line_no))
    return samples


def save_results(path: Union[str, Path], results: Any, indent: int = 2) -> Path:
    """Serialize structured evaluation results to JSON on disk."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=indent) + "\n", encoding="utf-8")
    return out_path
