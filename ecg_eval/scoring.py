"""Per-answer evaluation logic for question-type-aware ECG scoring."""

from __future__ import annotations

from typing import Dict, List

from .label_extractor import ECGLabelExtractor
from .metrics import SemanticSimilarityScorer, jaccard_similarity, label_prf1, sequence_similarity
from .normalize import normalize_text
from .parsing import extract_leads, normalize_binary_answer, normalize_multiclass_answer, overlap_with_range, parse_numeric_answer


class AnswerEvaluator:
    """Evaluate one answer against a derived question-specific target."""

    def __init__(self, extractor: ECGLabelExtractor | None = None, semantic_scorer: SemanticSimilarityScorer | None = None) -> None:
        self.extractor = extractor or ECGLabelExtractor()
        self.semantic_scorer = semantic_scorer

    def evaluate(self, answer: str, prompt: str, report: str, question_type: str, derived_gt: Dict[str, object]) -> Dict[str, object]:
        """Return a structured evaluation payload for one answer."""
        if question_type == "binary_classification":
            return self._evaluate_binary(answer, prompt, derived_gt)
        if question_type == "multiclass_classification":
            return self._evaluate_multiclass(answer, prompt, derived_gt)
        if question_type == "numeric":
            return self._evaluate_numeric(answer, prompt, derived_gt)
        if question_type == "lead_based":
            return self._evaluate_leads(answer, derived_gt)
        if question_type == "diagnosis_label":
            return self._evaluate_diagnosis(answer, report, derived_gt)
        if question_type == "summary_generation":
            return self._evaluate_summary(answer, report, derived_gt)
        return {"scorable": False, "warning": f"Unsupported question type: {question_type}"}

    def _base_payload(self, derived_gt: Dict[str, object]) -> Dict[str, object]:
        return {
            "scorable": bool(derived_gt.get("scorable", False)),
            "warning": derived_gt.get("warning"),
            "notes": list(derived_gt.get("notes", [])),
        }

    def _evaluate_binary(self, answer: str, prompt: str, derived_gt: Dict[str, object]) -> Dict[str, object]:
        payload = self._base_payload(derived_gt)
        parsed = normalize_binary_answer(answer, prompt)
        payload.update(parsed)
        payload["true_class"] = derived_gt.get("true_class")
        if not payload["scorable"]:
            return payload
        if parsed["predicted_class"] is None:
            payload["scorable"] = False
            payload["warning"] = payload.get("warning") or "Could not normalize binary answer."
            return payload
        payload["correctness"] = float(parsed["predicted_class"] == derived_gt.get("true_class"))
        payload["primary_score"] = payload["correctness"]
        return payload

    def _evaluate_multiclass(self, answer: str, prompt: str, derived_gt: Dict[str, object]) -> Dict[str, object]:
        payload = self._base_payload(derived_gt)
        parsed = normalize_multiclass_answer(answer, prompt)
        payload.update(parsed)
        payload["true_class"] = derived_gt.get("true_class")
        if not payload["scorable"]:
            return payload
        if parsed["predicted_class"] is None:
            payload["scorable"] = False
            payload["warning"] = payload.get("warning") or "Could not normalize multiclass answer."
            return payload
        payload["exact_match"] = float(parsed["predicted_class"] == derived_gt.get("true_class"))
        payload["primary_score"] = payload["exact_match"]
        return payload

    def _evaluate_numeric(self, answer: str, prompt: str, derived_gt: Dict[str, object]) -> Dict[str, object]:
        payload = self._base_payload(derived_gt)
        parsed = parse_numeric_answer(answer)
        payload.update(parsed)
        payload["true_numeric"] = derived_gt.get("true_numeric")
        payload["true_range"] = derived_gt.get("true_range")
        payload["true_descriptor"] = derived_gt.get("true_descriptor")
        payload["exact_match"] = None
        payload["tolerance_match"] = None
        if not payload["scorable"]:
            return payload
        payload["tolerance_match"] = overlap_with_range(parsed.get("predicted_numeric"), parsed.get("predicted_range"), derived_gt.get("true_range"))
        if payload["true_numeric"] is not None and parsed.get("predicted_numeric") is not None:
            payload["exact_match"] = float(parsed["predicted_numeric"] == payload["true_numeric"])
        elif payload["true_range"] is not None and parsed.get("predicted_range") is not None:
            low, high = payload["true_range"]
            pred_low, pred_high = parsed["predicted_range"]
            payload["exact_match"] = float(pred_low == low and pred_high == high)
        if payload["tolerance_match"] is None and payload["predicted_numeric"] is None and payload["predicted_range"] is None:
            payload["scorable"] = False
            payload["warning"] = payload.get("warning") or "Could not parse numeric answer."
            return payload
        payload["primary_score"] = float(payload["tolerance_match"]) if payload["tolerance_match"] is not None else 0.0
        return payload

    def _evaluate_leads(self, answer: str, derived_gt: Dict[str, object]) -> Dict[str, object]:
        payload = self._base_payload(derived_gt)
        predicted_leads = extract_leads(answer)
        true_leads = list(derived_gt.get("true_leads", []))
        payload["predicted_leads"] = predicted_leads
        payload["true_leads"] = true_leads
        if not payload["scorable"]:
            return payload
        if not predicted_leads:
            payload["scorable"] = False
            payload["warning"] = payload.get("warning") or "Could not parse lead answer."
            return payload
        prf = label_prf1(predicted_leads, true_leads)
        payload.update(
            {
                "lead_precision": prf["precision"],
                "lead_recall": prf["recall"],
                "lead_f1": prf["f1"],
                "exact_set_match": float(set(predicted_leads) == set(true_leads)),
                "tp": prf["tp"],
                "fp": prf["fp"],
                "fn": prf["fn"],
                "primary_score": prf["f1"],
            }
        )
        return payload

    def _evaluate_diagnosis(self, answer: str, report: str, derived_gt: Dict[str, object]) -> Dict[str, object]:
        payload = self._base_payload(derived_gt)
        predicted_labels = self.extractor.extract_non_generic(answer) or self.extractor.extract(answer)
        true_labels = list(derived_gt.get("true_labels", []))
        prf = label_prf1(predicted_labels, true_labels)
        payload.update(
            {
                "predicted_labels": predicted_labels,
                "true_labels": true_labels,
                "tp": prf["tp"],
                "fp": prf["fp"],
                "fn": prf["fn"],
                "precision": prf["precision"],
                "recall": prf["recall"],
                "f1": prf["f1"],
                "jaccard": jaccard_similarity(" ".join(predicted_labels), " ".join(true_labels)),
                "sequence_similarity": sequence_similarity(answer, report),
                "primary_score": prf["f1"],
            }
        )
        if not payload["scorable"]:
            return payload
        return payload

    def _evaluate_summary(self, answer: str, report: str, derived_gt: Dict[str, object]) -> Dict[str, object]:
        payload = self._base_payload(derived_gt)
        predicted_labels = self.extractor.extract_non_generic(answer) or self.extractor.extract(answer)
        true_labels = list(derived_gt.get("true_labels", []))
        prf = label_prf1(predicted_labels, true_labels)
        payload.update(
            {
                "predicted_labels": predicted_labels,
                "true_labels": true_labels,
                "label_precision": prf["precision"],
                "label_recall": prf["recall"],
                "label_f1": prf["f1"],
                "text_jaccard": jaccard_similarity(answer, derived_gt.get("target_text", report)),
                "sequence_similarity": sequence_similarity(answer, derived_gt.get("target_text", report)),
            }
        )
        if self.semantic_scorer is not None:
            payload["semantic_similarity"] = self.semantic_scorer.score_pairs([(answer, derived_gt.get("target_text", report))])[0]
        payload["primary_score"] = sum(
            value for value in [payload["label_f1"], payload["text_jaccard"], payload["sequence_similarity"]] if value is not None
        ) / 3.0
        return payload


def evaluate_answer(answer: str, prompt: str, report: str, question_type: str, derived_gt: Dict[str, object], extractor: ECGLabelExtractor | None = None, semantic_scorer: SemanticSimilarityScorer | None = None) -> Dict[str, object]:
    """Convenience wrapper for question-type-aware answer scoring."""
    return AnswerEvaluator(extractor=extractor, semantic_scorer=semantic_scorer).evaluate(
        answer=answer,
        prompt=prompt,
        report=report,
        question_type=question_type,
        derived_gt=derived_gt,
    )
