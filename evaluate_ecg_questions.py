#!/usr/bin/env python
"""Evaluate one ECG prediction file with question-type-aware scoring.

Usage:
    python evaluate_ecg_questions.py \
      --reports report_data.csv \
      --input answers/answer_img.jsonl \
      --output-dir outputs/ecg_question_eval_img \
      --modality image \
      --question-map ecg_eval/question_type_overrides.example.json
"""

from __future__ import annotations

import argparse
import sys

from ecg_eval.analysis import run_single_modality_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", default="report_data.csv", help="Path to the report CSV.")
    parser.add_argument("--input", required=True, help="Path to a prediction JSONL file.")
    parser.add_argument("--output-dir", required=True, help="Directory for evaluation outputs.")
    parser.add_argument("--modality", default="prediction", help="Modality name to store in outputs.")
    parser.add_argument("--question-map", default=None, help="Optional prompt override JSON.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_single_modality_evaluation(
            reports_path=args.reports,
            prediction_path=args.input,
            outdir=args.output_dir,
            modality_name=args.modality,
            question_map_path=args.question_map,
        )
    except Exception as exc:
        parser.exit(status=1, message=f"Evaluation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
