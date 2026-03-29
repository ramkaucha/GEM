"""Prompt classification and prompt-intent helpers for ECG question evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .normalize import normalize_text

QUESTION_TYPES = {
    "binary_classification",
    "multiclass_classification",
    "numeric",
    "lead_based",
    "diagnosis_label",
    "summary_generation",
}


def clean_prompt(prompt: object) -> str:
    """Normalize a prompt while removing multimodal markup."""
    value = str(prompt or "")
    value = value.replace("<image>", " ").replace("<ecg>", " ")
    return normalize_text(value)


def load_question_overrides(path: str | Path | None) -> Dict[str, str]:
    """Load optional prompt->question_type overrides."""
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Question override file not found: {file_path}")
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    overrides = {clean_prompt(prompt): value for prompt, value in payload.items()}
    invalid = sorted(set(overrides.values()) - QUESTION_TYPES)
    if invalid:
        raise ValueError(f"Invalid override question types: {invalid}")
    return overrides


def prompt_key(prompt: str) -> str:
    """Return a stable prompt-intent key for a normalized ECG question."""
    value = clean_prompt(prompt)
    rules: List[Tuple[str, str]] = [
        ("what is the heart rate in bpm", "heart_rate"),
        ("is the rhythm regular or irregular", "rhythm_regularity"),
        ("what is the most likely underlying rhythm", "underlying_rhythm"),
        ("are p waves present before each qrs complex", "p_waves_before_qrs"),
        ("is there evidence of sinus rhythm", "evidence_sinus_rhythm"),
        ("is there evidence of atrial fibrillation", "evidence_atrial_fibrillation"),
        ("is there evidence of atrial flutter", "evidence_atrial_flutter"),
        ("are there premature beats present", "premature_beats"),
        ("is there any pause or dropped beat visible", "pause_or_dropped_beat"),
        ("does this ecg suggest tachycardia, bradycardia, or a normal rate", "rate_class"),
        ("estimate the pr interval", "pr_interval"),
        ("estimate the qrs duration", "qrs_duration"),
        ("estimate the qt interval or qtc", "qt_interval"),
        ("is there evidence of first-degree av block", "first_degree_av_block"),
        ("is there evidence of second-degree av block", "second_degree_av_block"),
        ("is there evidence of third-degree av block", "third_degree_av_block"),
        ("is there evidence of right bundle branch block", "right_bundle_branch_block"),
        ("is there evidence of left bundle branch block", "left_bundle_branch_block"),
        ("what is the likely frontal plane qrs axis", "axis_class"),
        ("is there any intraventricular conduction delay present", "intraventricular_conduction_delay"),
        ("are there abnormal p-wave findings suggesting atrial enlargement", "atrial_enlargement"),
        ("are there q waves present, and if so are they pathologic", "q_waves"),
        ("is there r-wave progression across the precordial leads, and is it normal", "r_wave_progression"),
        ("is there poor r-wave progression", "poor_r_wave_progression"),
        ("is there st-segment elevation present", "st_elevation"),
        ("is there st-segment depression present", "st_depression"),
        ("are there t-wave inversions present", "t_wave_inversion"),
        ("are there peaked t waves or other signs of hyperkalemia", "hyperkalemia_signs"),
        ("are there u waves present", "u_waves"),
        ("are there signs of ventricular hypertrophy", "ventricular_hypertrophy"),
        ("what is the most likely primary diagnosis from this ecg", "primary_diagnosis"),
        ("list the top 3 diagnostic considerations for this ecg", "diagnostic_considerations"),
        ("what are the key ecg findings supporting your interpretation", "key_findings"),
        ("which leads contain the most important abnormal findings", "lead_abnormalities"),
        ("is there evidence of acute ischemia or infarction", "acute_ischemia_or_infarction"),
        ("if infarction is suspected, what territory is most likely involved", "infarction_territory"),
        ("is this ecg more consistent with a supraventricular or ventricular process", "supraventricular_vs_ventricular"),
        ("does this ecg appear normal or abnormal overall", "overall_normal_abnormal"),
        ("summarize this ecg in 3 to 5 concise clinical statements", "ecg_summary"),
        ("provide a final impression as if writing an ecg report", "ecg_impression"),
    ]
    for pattern, key in rules:
        if pattern in value:
            return key
    return "unknown"


def classify_question(prompt: str, overrides: Dict[str, str] | None = None) -> str:
    """Classify a prompt into a question type using rules and optional overrides."""
    normalized = clean_prompt(prompt)
    if overrides and normalized in overrides:
        return overrides[normalized]

    key = prompt_key(normalized)
    if key in {"ecg_summary", "ecg_impression"}:
        return "summary_generation"
    if key in {"lead_abnormalities"}:
        return "lead_based"
    if key in {"underlying_rhythm", "primary_diagnosis", "diagnostic_considerations", "key_findings"}:
        return "diagnosis_label"
    if key in {"heart_rate", "pr_interval", "qrs_duration", "qt_interval"}:
        return "numeric"
    if key in {"rate_class", "axis_class", "q_waves", "r_wave_progression", "infarction_territory", "supraventricular_vs_ventricular"}:
        return "multiclass_classification"
    if normalized.startswith("is ") or normalized.startswith("are ") or normalized.startswith("does "):
        return "binary_classification"
    return "summary_generation"


def build_question_classification(prompts: Iterable[str], overrides: Dict[str, str] | None = None) -> Dict[str, dict]:
    """Return an inspectable prompt->classification payload."""
    mapping: Dict[str, dict] = {}
    overrides = overrides or {}
    for prompt in sorted(set(str(prompt) for prompt in prompts)):
        normalized = clean_prompt(prompt)
        mapping[prompt] = {
            "normalized_prompt": normalized,
            "question_key": prompt_key(prompt),
            "question_type": classify_question(prompt, overrides=overrides),
            "overridden": normalized in overrides,
        }
    return mapping
