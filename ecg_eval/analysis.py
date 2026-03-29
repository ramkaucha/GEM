"""Question-type-aware ECG evaluation workflows."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from .ground_truth import GroundTruthDeriver
from .io_utils import build_merged_dataframe, build_single_prediction_dataframe, ensure_outdir, write_json, write_jsonl
from .label_extractor import ECGLabelExtractor
from .metrics import SemanticSimilarityScorer
from .questioning import build_question_classification, classify_question, load_question_overrides, prompt_key
from .scoring import AnswerEvaluator

MODALITIES = ["image", "ecg", "both"]
QUESTION_TYPES = [
    "binary_classification",
    "multiclass_classification",
    "numeric",
    "lead_based",
    "diagnosis_label",
    "summary_generation",
]


def _maybe_get_semantic_scorer() -> SemanticSimilarityScorer | None:
    try:
        return SemanticSimilarityScorer()
    except RuntimeError:
        return None


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _flatten_record(record: dict) -> dict:
    flat = {
        "study_id": record["study_id"],
        "question_id": record["question_id"],
        "prompt": record["prompt"],
        "question_type": record["question_type"],
        "question_key": record["question_key"],
        "modality": record["modality"],
        "answer": record["answer"],
        "scorable": record["scorable"],
        "weakly_scorable": record["weakly_scorable"],
        "warning": record.get("warning"),
        "notes": " | ".join(record.get("notes", [])),
    }
    for source, prefix in [(record.get("derived_ground_truth", {}), "gt"), (record.get("evaluation", {}), "eval")]:
        for key, value in source.items():
            if isinstance(value, (dict, list)):
                flat[f"{prefix}_{key}"] = str(value)
            else:
                flat[f"{prefix}_{key}"] = value
    return flat


def _evaluate_long_frame(frame: pd.DataFrame, overrides: Dict[str, str]) -> tuple[List[dict], Dict[str, dict], Dict[str, bool]]:
    extractor = ECGLabelExtractor()
    deriver = GroundTruthDeriver(extractor=extractor)
    semantic_scorer = _maybe_get_semantic_scorer()
    evaluator = AnswerEvaluator(extractor=extractor, semantic_scorer=semantic_scorer)
    classification = build_question_classification(frame["prompt"].unique().tolist(), overrides=overrides)

    records: List[dict] = []
    for row in frame.to_dict(orient="records"):
        question_type = classify_question(row["prompt"], overrides=overrides)
        ground_truth = deriver.derive(prompt=row["prompt"], report=row["report"], question_type=question_type)
        evaluation = evaluator.evaluate(
            answer=row["answer"],
            prompt=row["prompt"],
            report=row["report"],
            question_type=question_type,
            derived_gt=ground_truth,
        )
        notes = list(dict.fromkeys(list(ground_truth.get("notes", [])) + list(evaluation.get("notes", []))))
        record = {
            "study_id": int(row["study_id"]),
            "question_id": row["question_id"],
            "prompt": row["prompt"],
            "question_type": question_type,
            "question_key": prompt_key(row["prompt"]),
            "report": row["report"],
            "modality": row["modality"],
            "answer": row["answer"],
            "derived_ground_truth": _json_safe(ground_truth),
            "evaluation": _json_safe(evaluation),
            "scorable": bool(evaluation.get("scorable", ground_truth.get("scorable", False))),
            "weakly_scorable": bool(ground_truth.get("weakly_scorable", False)),
            "warning": evaluation.get("warning") or ground_truth.get("warning"),
            "notes": notes,
        }
        records.append(record)

    optional = {
        "semantic_similarity": semantic_scorer is not None,
    }
    return records, classification, optional


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _micro_macro_from_records(records: Sequence[dict], precision_key: str, recall_key: str, f1_key: str, tp_key: str, fp_key: str, fn_key: str) -> dict:
    if not records:
        return {
            "scorable_count": 0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
        }
    tp = sum(float(record["evaluation"].get(tp_key, 0.0)) for record in records)
    fp = sum(float(record["evaluation"].get(fp_key, 0.0)) for record in records)
    fn = sum(float(record["evaluation"].get(fn_key, 0.0)) for record in records)
    micro_precision = tp / (tp + fp) if (tp + fp) else 0.0
    micro_recall = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0.0
    return {
        "scorable_count": len(records),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": _mean([float(record["evaluation"].get(precision_key, 0.0)) for record in records]),
        "macro_recall": _mean([float(record["evaluation"].get(recall_key, 0.0)) for record in records]),
        "macro_f1": _mean([float(record["evaluation"].get(f1_key, 0.0)) for record in records]),
    }


def _aggregate_type_records(records: Sequence[dict]) -> dict:
    if not records:
        return {"scorable_count": 0, "unscorable_count": 0}
    qtype = records[0]["question_type"]
    scorable = [record for record in records if record["scorable"]]
    summary = {
        "scorable_count": len(scorable),
        "unscorable_count": len(records) - len(scorable),
    }
    if qtype == "binary_classification":
        summary["accuracy"] = _mean([float(record["evaluation"].get("correctness", 0.0)) for record in scorable])
    elif qtype == "multiclass_classification":
        summary["accuracy"] = _mean([float(record["evaluation"].get("exact_match", 0.0)) for record in scorable])
    elif qtype == "numeric":
        exact_values = [record["evaluation"].get("exact_match") for record in scorable if record["evaluation"].get("exact_match") is not None]
        tolerance_values = [float(record["evaluation"].get("tolerance_match", 0.0)) for record in scorable if record["evaluation"].get("tolerance_match") is not None]
        summary.update(
            {
                "exact_match_rate": _mean([float(value) for value in exact_values]) if exact_values else 0.0,
                "exact_scorable_count": len(exact_values),
                "tolerance_match_rate": _mean(tolerance_values),
            }
        )
    elif qtype == "lead_based":
        summary.update(_micro_macro_from_records(scorable, "lead_precision", "lead_recall", "lead_f1", "tp", "fp", "fn"))
    elif qtype == "diagnosis_label":
        summary.update(_micro_macro_from_records(scorable, "precision", "recall", "f1", "tp", "fp", "fn"))
        summary["avg_jaccard"] = _mean([float(record["evaluation"].get("jaccard", 0.0)) for record in scorable])
        summary["avg_sequence_similarity"] = _mean([float(record["evaluation"].get("sequence_similarity", 0.0)) for record in scorable])
    elif qtype == "summary_generation":
        summary.update(
            {
                "avg_label_f1": _mean([float(record["evaluation"].get("label_f1", 0.0)) for record in scorable]),
                "avg_text_jaccard": _mean([float(record["evaluation"].get("text_jaccard", 0.0)) for record in scorable]),
                "avg_sequence_similarity": _mean([float(record["evaluation"].get("sequence_similarity", 0.0)) for record in scorable]),
            }
        )
        semantic = [record["evaluation"].get("semantic_similarity") for record in scorable if record["evaluation"].get("semantic_similarity") is not None]
        if semantic:
            summary["avg_semantic_similarity"] = _mean([float(value) for value in semantic])
    return summary


def _aggregate_records(records: Sequence[dict]) -> dict:
    by_modality: Dict[str, dict] = {}
    for modality in sorted(set(record["modality"] for record in records)):
        modality_records = [record for record in records if record["modality"] == modality]
        by_modality[modality] = {
            "overall": {
                "sample_count": len(modality_records),
                "scorable_count": sum(int(record["scorable"]) for record in modality_records),
                "unscorable_count": sum(int(not record["scorable"]) for record in modality_records),
            },
            "by_question_type": {},
        }
        for question_type in QUESTION_TYPES:
            type_records = [record for record in modality_records if record["question_type"] == question_type]
            if type_records:
                by_modality[modality]["by_question_type"][question_type] = _aggregate_type_records(type_records)

    by_question_type: Dict[str, dict] = {}
    for question_type in QUESTION_TYPES:
        type_records = [record for record in records if record["question_type"] == question_type]
        if type_records:
            by_question_type[question_type] = _aggregate_type_records(type_records)
    return {"by_modality": by_modality, "by_question_type": by_question_type}


def _wins_by_type(records: Sequence[dict]) -> dict:
    wins: Dict[str, Counter[str]] = defaultdict(Counter)
    both_better: Counter[str] = Counter()
    both_adds: Counter[str] = Counter()

    grouped: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        grouped[record["question_id"]].append(record)

    for question_id, group in grouped.items():
        if len(group) < 3:
            continue
        qtype = group[0]["question_type"]
        if any(record["question_type"] != qtype for record in group):
            continue
        scorable_group = [record for record in group if record["scorable"]]
        if len(scorable_group) < 3:
            continue
        score_map = {record["modality"]: float(record["evaluation"].get("primary_score", 0.0)) for record in scorable_group}
        if len(score_map) != 3:
            continue
        best = max(score_map.values())
        winners = sorted([modality for modality, score in score_map.items() if score == best])
        if len(winners) == 1:
            wins[qtype][winners[0]] += 1
        else:
            wins[qtype]["tie"] += 1
        if score_map["both"] > score_map["image"] and score_map["both"] > score_map["ecg"]:
            both_better[qtype] += 1

        if qtype in {"diagnosis_label", "summary_generation"}:
            img_labels = set(group[[record["modality"] for record in group].index("image")]["evaluation"].get("predicted_labels", []))
            ecg_labels = set(group[[record["modality"] for record in group].index("ecg")]["evaluation"].get("predicted_labels", []))
            both_labels = set(group[[record["modality"] for record in group].index("both")]["evaluation"].get("predicted_labels", []))
            true_labels = set(group[[record["modality"] for record in group].index("both")]["derived_ground_truth"].get("true_labels", []))
            if (both_labels - (img_labels | ecg_labels)) & true_labels:
                both_adds[qtype] += 1

    return {
        "wins_by_type": {question_type: dict(counter) for question_type, counter in wins.items()},
        "both_better_than_unimodal_by_type": dict(both_better),
        "both_adds_report_labels_by_type": dict(both_adds),
    }



def _build_combined_question_records(records: Sequence[dict]) -> List[dict]:
    """Group modality-specific rows into one combined record per question."""
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        grouped[record["question_id"]].append(record)

    combined: List[dict] = []
    for question_id in sorted(grouped):
        group = sorted(grouped[question_id], key=lambda item: item["modality"])
        first = group[0]
        combined_record = {
            "study_id": first["study_id"],
            "question_id": question_id,
            "prompt": first["prompt"],
            "question_type": first["question_type"],
            "question_key": first["question_key"],
            "report": first["report"],
            "answers": [
                {
                    "modality": record["modality"],
                    "text": record["answer"],
                    "scorable": record["scorable"],
                    "warning": record.get("warning"),
                    "derived_ground_truth": record.get("derived_ground_truth"),
                    "evaluation": record.get("evaluation"),
                }
                for record in group
            ],
        }
        combined.append(combined_record)
    return combined
def _build_error_cases(records: Sequence[dict]) -> dict:
    categories = {
        "unknown_ground_truth": [],
        "parsing_failures": [],
        "incorrect_predictions": [],
    }
    for record in records:
        slim = {
            "study_id": record["study_id"],
            "question_id": record["question_id"],
            "prompt": record["prompt"],
            "question_type": record["question_type"],
            "modality": record["modality"],
            "answer": record["answer"],
            "warning": record.get("warning"),
            "notes": record.get("notes", []),
            "derived_ground_truth": record.get("derived_ground_truth"),
            "evaluation": record.get("evaluation"),
        }
        if not record["scorable"]:
            categories["unknown_ground_truth"].append(slim)
            continue
        evaluation = record["evaluation"]
        if evaluation.get("predicted_class") is None and evaluation.get("predicted_numeric") is None and not evaluation.get("predicted_labels") and not evaluation.get("predicted_leads"):
            categories["parsing_failures"].append(slim)
            continue
        if float(evaluation.get("primary_score", 0.0)) == 0.0:
            categories["incorrect_predictions"].append(slim)
    return categories


def _print_summary(records: Sequence[dict], aggregate: dict, optional_dependencies: dict) -> None:
    print("\n=== Type-Aware ECG Evaluation Summary ===")
    print(f"Samples evaluated: {len(records)}")
    print(f"Scorable samples: {sum(int(record['scorable']) for record in records)}")
    print(f"Unscorable samples: {sum(int(not record['scorable']) for record in records)}")
    print(f"Optional dependencies: semantic_similarity={optional_dependencies.get('semantic_similarity', False)}")
    for modality, payload in aggregate["by_modality"].items():
        print(f"\n[{modality}] scorable={payload['overall']['scorable_count']} / {payload['overall']['sample_count']}")
        for question_type, metrics in payload["by_question_type"].items():
            headline = metrics.get("accuracy")
            if headline is None:
                headline = metrics.get("micro_f1")
            if headline is None:
                headline = metrics.get("avg_label_f1")
            if headline is None:
                headline = metrics.get("tolerance_match_rate", 0.0)
            print(f"  {question_type}: score={headline:.4f}, scorable={metrics['scorable_count']}")


def run_single_modality_evaluation(reports_path: str, prediction_path: str, outdir: str, modality_name: str = "prediction", question_map_path: str | None = None) -> dict:
    """Evaluate one prediction JSONL file with question-type-aware scoring."""
    overrides = load_question_overrides(question_map_path)
    outdir_path = ensure_outdir(outdir)
    frame = build_single_prediction_dataframe(reports_path, prediction_path, modality_name=modality_name)
    records, classification, optional = _evaluate_long_frame(frame, overrides)
    aggregate = _aggregate_records(records)
    error_cases = _build_error_cases(records)

    write_json(outdir_path / "question_classification.json", classification)
    write_jsonl(outdir_path / "per_sample_results.jsonl", records)
    write_jsonl(outdir_path / "combined_samples.jsonl", _build_combined_question_records(records))
    write_json(outdir_path / "aggregate_results.json", {**aggregate, "optional_dependencies": optional})
    write_json(outdir_path / "error_cases.json", error_cases)
    pd.DataFrame([_flatten_record(record) for record in records]).to_csv(outdir_path / "per_sample_results.csv", index=False)

    _print_summary(records, aggregate, optional)
    return {"records": records, "aggregate": aggregate, "classification": classification, "error_cases": error_cases}


def run_multimodal_evaluation(reports_path: str, img_path: str, ecg_path: str, both_path: str, outdir: str, question_map_path: str | None = None) -> dict:
    """Evaluate image, ECG, and combined predictions with question-type-aware scoring."""
    overrides = load_question_overrides(question_map_path)
    outdir_path = ensure_outdir(outdir)
    merged = build_merged_dataframe(reports_path, img_path, ecg_path, both_path)
    long_rows: List[dict] = []
    for row in merged.to_dict(orient="records"):
        for modality, answer_column in [("image", "answer_img"), ("ecg", "answer_ecg"), ("both", "answer_both")]:
            long_rows.append(
                {
                    "study_id": row["study_id"],
                    "question_id": row["question_id"],
                    "prompt": row["prompt"],
                    "report": row["report"],
                    "modality": modality,
                    "answer": row[answer_column],
                }
            )
    long_frame = pd.DataFrame(long_rows)
    records, classification, optional = _evaluate_long_frame(long_frame, overrides)
    aggregate = _aggregate_records(records)
    win_summary = _wins_by_type(records)
    error_cases = _build_error_cases(records)

    write_json(outdir_path / "question_classification.json", classification)
    write_jsonl(outdir_path / "per_sample_results.jsonl", records)
    write_jsonl(outdir_path / "combined_samples.jsonl", _build_combined_question_records(records))
    write_json(outdir_path / "aggregate_results.json", {**aggregate, **win_summary, "optional_dependencies": optional})
    write_json(outdir_path / "error_cases.json", error_cases)
    pd.DataFrame([_flatten_record(record) for record in records]).to_csv(outdir_path / "per_sample_results.csv", index=False)
    merged.to_csv(outdir_path / "merged_predictions.csv", index=False)

    _print_summary(records, aggregate, optional)
    return {
        "records": records,
        "aggregate": aggregate,
        "classification": classification,
        "error_cases": error_cases,
        "wins": win_summary,
    }


