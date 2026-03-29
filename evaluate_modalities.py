#!/usr/bin/env python
"""Evaluate image, ECG, and combined ECG answers with question-type-aware scoring.

Usage:
    python evaluate_modalities.py \
      --reports report_data.csv \
      --img answers/answer_img.jsonl \
      --ecg answers/answer_ecg.jsonl \
      --both answers/answer_both.jsonl \
      --outdir outputs/modality_eval \
      --question-map ecg_eval/question_type_overrides.example.json
"""

from __future__ import annotations

import argparse
import sys

from ecg_eval.analysis import run_multimodal_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", required=True, help="Path to cleaned_report_data.csv or equivalent report CSV.")
    parser.add_argument("--img", required=True, help="Path to image-only answer JSONL.")
    parser.add_argument("--ecg", required=True, help="Path to ECG-only answer JSONL.")
    parser.add_argument("--both", required=True, help="Path to ECG+image answer JSONL.")
    parser.add_argument("--outdir", required=True, help="Directory for evaluation outputs.")
    parser.add_argument("--question-map", default=None, help="Optional prompt override JSON.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_multimodal_evaluation(
            reports_path=args.reports,
            img_path=args.img,
            ecg_path=args.ecg,
            both_path=args.both,
            outdir=args.outdir,
            question_map_path=args.question_map,
        )
    except Exception as exc:
        parser.exit(status=1, message=f"Evaluation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
