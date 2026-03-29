"""Metrics for modality agreement and report alignment."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Sequence

from .normalize import normalize_text, unique_tokens


def exact_match(text_a: object, text_b: object) -> float:
    """Return 1.0 when normalized texts match exactly."""
    return float(normalize_text(text_a) == normalize_text(text_b))


def jaccard_similarity(text_a: object, text_b: object) -> float:
    """Compute token-level Jaccard similarity."""
    set_a = unique_tokens(text_a)
    set_b = unique_tokens(text_b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def sequence_similarity(text_a: object, text_b: object) -> float:
    """Compute difflib SequenceMatcher ratio on normalized text."""
    return SequenceMatcher(None, normalize_text(text_a), normalize_text(text_b)).ratio()


def rouge_l_score(text_a: object, text_b: object) -> float | None:
    """Compute ROUGE-L F1 if rouge-score is installed."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return None
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(normalize_text(text_a), normalize_text(text_b))
    return float(scores["rougeL"].fmeasure)


class SemanticSimilarityScorer:
    """Lazy sentence-transformers cosine similarity scorer."""

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer, util
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed") from exc
        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._util = util

    def score_pairs(self, text_pairs: Sequence[tuple[str, str]]) -> List[float]:
        """Return cosine similarity for each text pair."""
        if not text_pairs:
            return []
        first_texts = [normalize_text(a) for a, _ in text_pairs]
        second_texts = [normalize_text(b) for _, b in text_pairs]
        embeddings_a = self._model.encode(first_texts, convert_to_tensor=True, show_progress_bar=False)
        embeddings_b = self._model.encode(second_texts, convert_to_tensor=True, show_progress_bar=False)
        similarities = self._util.cos_sim(embeddings_a, embeddings_b).diagonal()
        return [float(value) for value in similarities]


def label_prf1(predicted: Iterable[str], reference: Iterable[str]) -> Dict[str, float]:
    """Compute precision, recall, and F1 for label sets."""
    pred_set = set(predicted)
    ref_set = set(reference)
    tp = len(pred_set & ref_set)
    fp = len(pred_set - ref_set)
    fn = len(ref_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate_label_scores(per_row_scores: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Compute micro and macro averages from per-row PRF1 results."""
    if not per_row_scores:
        return {
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
        }

    total_tp = sum(score["tp"] for score in per_row_scores)
    total_fp = sum(score["fp"] for score in per_row_scores)
    total_fn = sum(score["fn"] for score in per_row_scores)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    macro_precision = sum(score["precision"] for score in per_row_scores) / len(per_row_scores)
    macro_recall = sum(score["recall"] for score in per_row_scores) / len(per_row_scores)
    macro_f1 = sum(score["f1"] for score in per_row_scores) / len(per_row_scores)

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def majority_winner(values: Dict[str, float]) -> List[str]:
    """Return the modality or modalities with the highest score."""
    if not values:
        return []
    best = max(values.values())
    return sorted([name for name, value in values.items() if value == best])
