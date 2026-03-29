#!/usr/bin/env python
"""Print the ECG prompt classifications for inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

from ecg_eval.io_utils import load_jsonl, write_json
from ecg_eval.questioning import build_question_classification, clean_prompt, load_question_overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="questions.json", help="Path to questions.json.")
    parser.add_argument("--question-map", default=None, help="Optional prompt override JSON.")
    parser.add_argument("--out", default=None, help="Optional path for question_classification.json output.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = load_jsonl(args.questions) if str(args.questions).endswith(".jsonl") else None
    if payload is None:
        import json
        payload = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    prompts = [item["conversations"][0]["value"].replace("<image>", "").strip() for item in payload]
    classification = build_question_classification(prompts, overrides=load_question_overrides(args.question_map))
    for prompt, info in classification.items():
        print(f"{info['question_key']:32s} | {info['question_type']:24s} | {prompt}")
    if args.out:
        write_json(args.out, classification)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
