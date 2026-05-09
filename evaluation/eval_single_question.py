"""Evaluate single-question inference outputs against AHA_Description.

Pairs each row in the JSONL produced by `model_ecg_resume.py --single-question`
with the corresponding row in `preprocess/reference/metadata_with_description.csv`
and scores the model output via rapidfuzz token_set_ratio.

OR-handling: descriptions like "Sinus bradycardia OR Sinus arrhythmia" are
split on ' OR ' and each alternative is scored separately. A record counts as
correct if ANY alternative scores at or above --threshold (default 70).

Outputs:
    --output-csv: per-record table with modality, best_score, matched_alt, reference, match
    stdout:      per-modality aggregate summary (count, accuracy, avg score)
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

try:
    from rapidfuzz.fuzz import token_set_ratio
except ImportError:  # pragma: no cover
    print(
        "rapidfuzz is required. Install it with:\n"
        "    pip install rapidfuzz",
        file=sys.stderr,
    )
    sys.exit(1)


# Whitespace-only normalization: lowercase and collapse runs of whitespace.
# We deliberately don't strip punctuation — token_set_ratio handles that well.
_WHITESPACE_RE = re.compile(r"\s+")
# AHA_Description uses literal "OR" (uppercase, surrounded by spaces) to separate alternatives.
_OR_RE = re.compile(r"\s+OR\s+")


def normalize(text: str) -> str:
    if text is None:
        return ""
    return _WHITESPACE_RE.sub(" ", text.lower().strip())


def load_metadata(csv_path: str) -> dict:
    """Build {record_id: {"raw": str, "alternatives": [normalized_str, ...]}}."""
    mapping = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = (row.get("ECG_ID") or "").strip()
            if not rid:
                continue
            desc = (row.get("AHA_Description") or "").strip()
            alternatives = [normalize(p) for p in _OR_RE.split(desc) if p.strip()]
            mapping[rid] = {"raw": desc, "alternatives": alternatives}
    return mapping


def evaluate(answers_jsonl: str, metadata_csv: str, output_csv: str, threshold: float):
    metadata = load_metadata(metadata_csv)
    if not metadata:
        print(f"WARNING: no rows loaded from {metadata_csv}", file=sys.stderr)

    rows = []
    with open(answers_jsonl, "r", encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            rec = json.loads(raw_line)
            rid = rec.get("record_id") or rec.get("question_id")
            modality = (rec.get("metadata") or {}).get("modality", "unknown")
            output_norm = normalize(rec.get("text", ""))

            ref = metadata.get(rid)
            if ref is None:
                # Record produced an answer but isn't in the metadata CSV.
                # Surface it explicitly rather than silently dropping it.
                rows.append({
                    "record_id": rid,
                    "modality": modality,
                    "best_score": "",
                    "matched_alt": "",
                    "reference": "",
                    "match": "missing_metadata",
                })
                continue

            best_alt = ""
            best_score = 0.0
            for alt in ref["alternatives"]:
                if not alt:
                    continue
                # token_set_ratio is order-independent and tolerant of extra padding
                # words in the model output, which is the common failure mode here.
                score = token_set_ratio(alt, output_norm)
                if score > best_score:
                    best_score = score
                    best_alt = alt

            rows.append({
                "record_id": rid,
                "modality": modality,
                "best_score": f"{best_score:.2f}",
                "matched_alt": best_alt,
                "reference": ref["raw"],
                "match": "yes" if best_score >= threshold else "no",
            })

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["record_id", "modality", "best_score", "matched_alt", "reference", "match"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate per-modality.
    by_modality = defaultdict(lambda: {"total": 0, "matched": 0, "scored": 0, "score_sum": 0.0, "missing_meta": 0})
    for r in rows:
        m = r["modality"]
        agg = by_modality[m]
        agg["total"] += 1
        if r["match"] == "yes":
            agg["matched"] += 1
        elif r["match"] == "missing_metadata":
            agg["missing_meta"] += 1
        if r["best_score"] != "":
            agg["scored"] += 1
            agg["score_sum"] += float(r["best_score"])

    print(f"Wrote per-record evaluation to {output_csv}")
    print(f"Threshold: {threshold}")
    print()
    header = f"{'Modality':<10} {'Total':<8} {'Match':<8} {'Acc%':<8} {'AvgScore':<10} {'NoMeta':<8}"
    print(header)
    print("-" * len(header))
    for m in sorted(by_modality.keys()):
        s = by_modality[m]
        acc = (s["matched"] / s["total"] * 100.0) if s["total"] else 0.0
        avg = (s["score_sum"] / s["scored"]) if s["scored"] else 0.0
        print(f"{m:<10} {s['total']:<8} {s['matched']:<8} {acc:<8.2f} {avg:<10.2f} {s['missing_meta']:<8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers-file", required=True, help="JSONL produced by --single-question inference")
    parser.add_argument(
        "--metadata-csv",
        default="preprocess/reference/metadata_with_description.csv",
        help="Reference CSV with ECG_ID and AHA_Description columns",
    )
    parser.add_argument("--output-csv", required=True, help="Where to write the per-record evaluation table")
    parser.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="Token-set-ratio threshold for counting a match (0-100). Default 70.",
    )
    args = parser.parse_args()
    evaluate(args.answers_file, args.metadata_csv, args.output_csv, args.threshold)
