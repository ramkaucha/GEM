"""Input and output utilities for ECG evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


REQUIRED_REPORT_COLUMNS = {"study_id", "total_report"}
REQUIRED_PREDICTION_COLUMNS = {"question_id", "prompt", "text"}


def recover_study_id(question_id: str) -> int:
    """Parse study_id from a question_id like 42149331_q09."""
    prefix = str(question_id).split("_", 1)[0]
    return int(prefix)


def load_reports(report_path: str | Path) -> pd.DataFrame:
    """Load the report CSV and validate required columns."""
    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"Report CSV not found: {report_path}")
    reports = pd.read_csv(report_path)
    missing = REQUIRED_REPORT_COLUMNS - set(reports.columns)
    if missing:
        raise ValueError(f"Report CSV is missing required columns: {sorted(missing)}")
    reports = reports.copy()
    reports["study_id"] = reports["study_id"].astype(int)
    reports["total_report"] = reports["total_report"].fillna("").astype(str)
    return reports


def load_jsonl(path: str | Path) -> List[dict]:
    """Load a JSONL file into a list of dictionaries."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")
    rows: List[dict] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {file_path}:{line_number}") from exc
    return rows


def load_predictions(path: str | Path, answer_column: str) -> pd.DataFrame:
    """Load one prediction file and standardize the output columns."""
    records = load_jsonl(path)
    frame = pd.DataFrame(records)
    missing = REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction file {path} is missing required columns: {sorted(missing)}")
    frame = frame[["question_id", "prompt", "text"]].copy()
    frame["study_id"] = frame["question_id"].map(recover_study_id)
    frame.rename(columns={"text": answer_column}, inplace=True)
    frame[answer_column] = frame[answer_column].fillna("").astype(str)
    frame["prompt"] = frame["prompt"].fillna("").astype(str)
    return frame


def build_single_prediction_dataframe(reports_path: str | Path, prediction_path: str | Path, modality_name: str = "prediction") -> pd.DataFrame:
    """Join one prediction file with the report CSV."""
    reports = load_reports(reports_path)[["study_id", "total_report"]].rename(columns={"total_report": "report"})
    predictions = load_predictions(prediction_path, "answer").copy()
    predictions["modality"] = modality_name
    merged = predictions.merge(reports, on="study_id", how="left")
    merged = merged.dropna(subset=["report"]).copy()
    return merged[["question_id", "study_id", "prompt", "report", "answer", "modality"]].sort_values(["study_id", "question_id"]).reset_index(drop=True)


def build_merged_dataframe(
    reports_path: str | Path,
    img_path: str | Path,
    ecg_path: str | Path,
    both_path: str | Path,
) -> pd.DataFrame:
    """Join reports and predictions across all modalities."""
    reports = load_reports(reports_path)[["study_id", "total_report"]].rename(columns={"total_report": "report"})
    img = load_predictions(img_path, "answer_img")
    ecg = load_predictions(ecg_path, "answer_ecg")
    both = load_predictions(both_path, "answer_both")

    merged = img.merge(ecg[["question_id", "answer_ecg"]], on="question_id", how="outer")
    merged = merged.merge(both[["question_id", "answer_both"]], on="question_id", how="outer")
    merged["study_id"] = merged["question_id"].map(recover_study_id)

    prompt_frame = pd.concat(
        [
            img[["question_id", "prompt"]],
            ecg[["question_id", "prompt"]],
            both[["question_id", "prompt"]],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["question_id"], keep="first")
    merged = merged.drop(columns=["prompt"], errors="ignore").merge(prompt_frame, on="question_id", how="left")
    merged = merged.merge(reports, on="study_id", how="left")
    merged = merged.dropna(subset=["answer_img", "answer_ecg", "answer_both", "report"]).copy()
    merged["report"] = merged["report"].fillna("").astype(str)
    merged = merged[
        ["question_id", "study_id", "prompt", "report", "answer_img", "answer_ecg", "answer_both"]
    ].sort_values(["study_id", "question_id"])
    return merged.reset_index(drop=True)


def ensure_outdir(outdir: str | Path) -> Path:
    """Create the output directory if needed."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def write_json(path: str | Path, payload: Dict) -> None:
    """Write a JSON file with indentation."""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Write newline-delimited JSON rows."""
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def stringify_labels(labels: Iterable[str]) -> str:
    """Serialize labels for CSV output."""
    return "|".join(labels)
