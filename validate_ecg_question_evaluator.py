#!/usr/bin/env python
"""Lightweight sanity checks for the question-type-aware ECG evaluator."""

from __future__ import annotations

from ecg_eval.ground_truth import derive_ground_truth
from ecg_eval.questioning import classify_question
from ecg_eval.scoring import evaluate_answer


def check_case(report: str, prompt: str, answer: str, expected_gt: object, expected_score: object, field: str = "true_class") -> None:
    question_type = classify_question(prompt)
    gt = derive_ground_truth(prompt, report, question_type)
    assert gt.get(field) == expected_gt, f"Expected GT {expected_gt!r}, got {gt.get(field)!r} for prompt {prompt!r}"
    result = evaluate_answer(answer, prompt, report, question_type, gt)
    if "correctness" in result:
        assert result["correctness"] == expected_score, f"Expected correctness {expected_score}, got {result['correctness']}"
    elif "exact_match" in result:
        assert result["exact_match"] == expected_score, f"Expected exact_match {expected_score}, got {result['exact_match']}"
    elif "f1" in result:
        assert round(result["f1"], 6) == round(expected_score, 6), f"Expected f1 {expected_score}, got {result['f1']}"


def main() -> int:
    report = "sinus rhythm. rightward axis. borderline ecg."
    prompt = "Is the rhythm regular or irregular?"
    check_case(report, prompt, "The rhythm is regular.", "regular", 1.0)
    check_case(report, prompt, "The rhythm is irregular due to atrial fibrillation.", "regular", 0.0)

    axis_prompt = "What is the likely frontal plane QRS axis: normal, left axis deviation, right axis deviation, or extreme axis deviation?"
    question_type = classify_question(axis_prompt)
    gt = derive_ground_truth(axis_prompt, report, question_type)
    assert gt.get("true_class") == "right axis deviation"

    af_report = "atrial fibrillation with slow ventricular response. abnormal ecg."
    af_prompt = "Is the rhythm regular or irregular?"
    gt = derive_ground_truth(af_prompt, af_report, classify_question(af_prompt))
    assert gt.get("true_class") == "irregular"

    normal_report = "sinus rhythm. normal ecg."
    overall_prompt = "Does this ECG appear normal or abnormal overall?"
    gt = derive_ground_truth(overall_prompt, normal_report, classify_question(overall_prompt))
    assert gt.get("true_class") == "normal"

    diagnosis_prompt = "What is the most likely underlying rhythm?"
    diagnosis_type = classify_question(diagnosis_prompt)
    diagnosis_gt = derive_ground_truth(diagnosis_prompt, af_report, diagnosis_type)
    diagnosis_result = evaluate_answer("Atrial fibrillation", diagnosis_prompt, af_report, diagnosis_type, diagnosis_gt)
    assert diagnosis_result["f1"] == 1.0

    print("Validation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
