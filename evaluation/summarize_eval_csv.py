"""Print the per-modality summary table from an eval.csv produced by eval_single_question.py.

Use when the original job's log was lost or you want to re-summarize without rerunning inference.

Default behaviour: uses the `match` column already in the CSV (yes/no/missing_metadata).
If --threshold is passed, the threshold is re-applied to `best_score` from scratch,
so you can sweep thresholds without re-running inference.

Examples:
    python summarize_eval_csv.py --eval-csv eval_outputs/GEM/single_question_1k/eval.csv
    python summarize_eval_csv.py --eval-csv eval.csv --threshold 80
"""

import argparse
import csv
import sys
from collections import defaultdict


def summarize(eval_csv: str, threshold):
    by_modality = defaultdict(lambda: {
        "total": 0,
        "matched": 0,
        "missing_meta": 0,
        "scored": 0,
        "score_sum": 0.0,
    })

    with open(eval_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mod = row.get("modality", "unknown")
            agg = by_modality[mod]
            agg["total"] += 1

            # best_score is empty when the record had no metadata entry.
            best_score_raw = row.get("best_score", "")
            if best_score_raw != "":
                try:
                    bs = float(best_score_raw)
                    agg["scored"] += 1
                    agg["score_sum"] += bs
                except ValueError:
                    bs = None
            else:
                bs = None

            # Determine match: use threshold override if provided, else trust CSV's match column.
            if threshold is not None:
                if bs is None:
                    agg["missing_meta"] += 1
                elif bs >= threshold:
                    agg["matched"] += 1
            else:
                m = (row.get("match") or "").strip().lower()
                if m == "yes":
                    agg["matched"] += 1
                elif m == "missing_metadata":
                    agg["missing_meta"] += 1

    if not by_modality:
        print(f"No rows found in {eval_csv}", file=sys.stderr)
        sys.exit(1)

    if threshold is not None:
        print(f"Threshold (override): {threshold}")
    else:
        print("Threshold: (using `match` column as-is from CSV)")

    header = f"{'Modality':<10} {'Total':<8} {'Match':<8} {'Acc%':<8} {'AvgScore':<10} {'NoMeta':<8}"
    print(header)
    print("-" * len(header))

    for mod in sorted(by_modality.keys()):
        s = by_modality[mod]
        # Accuracy denominator: exclude missing_metadata rows from the bottom of the fraction
        # so we don't penalize the model for our own data gaps. Falls back to total if everything
        # is missing (avoids div-by-zero).
        denom = s["total"] - s["missing_meta"]
        denom = denom if denom > 0 else s["total"]
        acc = (s["matched"] / denom * 100.0) if denom else 0.0
        avg = (s["score_sum"] / s["scored"]) if s["scored"] else 0.0
        print(f"{mod:<10} {s['total']:<8} {s['matched']:<8} {acc:<8.2f} {avg:<10.2f} {s['missing_meta']:<8}")

    # Overall row across all modalities.
    total_all = sum(s["total"] for s in by_modality.values())
    matched_all = sum(s["matched"] for s in by_modality.values())
    missing_all = sum(s["missing_meta"] for s in by_modality.values())
    scored_all = sum(s["scored"] for s in by_modality.values())
    score_sum_all = sum(s["score_sum"] for s in by_modality.values())
    denom_all = total_all - missing_all
    denom_all = denom_all if denom_all > 0 else total_all
    acc_all = (matched_all / denom_all * 100.0) if denom_all else 0.0
    avg_all = (score_sum_all / scored_all) if scored_all else 0.0
    print("-" * len(header))
    print(f"{'OVERALL':<10} {total_all:<8} {matched_all:<8} {acc_all:<8.2f} {avg_all:<10.2f} {missing_all:<8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--eval-csv", required=True, help="Path to eval.csv produced by eval_single_question.py")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="If set, re-apply this threshold to best_score (overrides the CSV's match column).",
    )
    args = parser.parse_args()
    summarize(args.eval_csv, args.threshold)
