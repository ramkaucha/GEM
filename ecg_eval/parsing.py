"""Answer normalization and extraction helpers for ECG question evaluation."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

from .normalize import deduplicate_ordered, normalize_text
from .questioning import prompt_key

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
_LEAD_PATTERN = re.compile(r"\b(?:lead|leads)?\s*(I{1,3}|aVR|aVL|aVF|V[1-6])\b", re.IGNORECASE)
_LEAD_RANGE_PATTERN = re.compile(r"\b(V[1-6])\s*[-/]\s*(V?)([1-6])\b", re.IGNORECASE)
_NUMBER_RANGE_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:to|-)\s*(-?\d+(?:\.\d+)?)")
_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])(-?\d+(?:\.\d+)?)")


def normalize_binary_answer(answer: str, prompt: str) -> Dict[str, object]:
    """Normalize a verbose answer into a binary or paired class when possible."""
    text = normalize_text(answer)
    key = prompt_key(prompt)
    notes: List[str] = []
    predicted = None

    if key == "rhythm_regularity":
        if re.search(r"\birregular\b", text):
            predicted = "irregular"
        elif re.search(r"\bregular\b", text):
            predicted = "regular"
    elif key == "overall_normal_abnormal":
        if "borderline" in text:
            predicted = "abnormal"
            notes.append("Mapped borderline answer to abnormal for strict scoring.")
        elif re.search(r"\babnormal\b", text):
            predicted = "abnormal"
        elif re.search(r"\bnormal\b", text):
            predicted = "normal"
    else:
        negative_patterns = [
            r"\bno\b",
            r"\bnot present\b",
            r"\babsent\b",
            r"\bwithout evidence\b",
            r"\bno evidence\b",
            r"\bdoes not appear\b",
            r"\bnot seen\b",
            r"\bnone\b",
            r"\bnegative\b",
        ]
        positive_patterns = [
            r"\byes\b",
            r"\bpresent\b",
            r"\bevidence of\b",
            r"\bseen\b",
            r"\bpositive\b",
            r"\bthere is\b",
            r"\bthere are\b",
        ]
        if any(re.search(pattern, text) for pattern in negative_patterns):
            predicted = "no"
        elif any(re.search(pattern, text) for pattern in positive_patterns):
            predicted = "yes"

    return {
        "predicted_class": predicted,
        "confidence": 1.0 if predicted is not None else 0.0,
        "notes": notes,
    }


def normalize_multiclass_answer(answer: str, prompt: str) -> Dict[str, object]:
    """Normalize a multiclass answer into a fixed class vocabulary."""
    text = normalize_text(answer)
    key = prompt_key(prompt)
    predicted = None
    classes: List[str] = []

    if key == "rate_class":
        mapping = [
            ("tachycardia", [r"\btachycardia\b", r"\btachy\b"]),
            ("bradycardia", [r"\bbradycardia\b", r"\bbrady\b"]),
            ("normal rate", [r"\bnormal rate\b", r"\bnormal\b"]),
        ]
    elif key == "axis_class":
        mapping = [
            ("left axis deviation", [r"\bleft axis deviation\b", r"\bleftward axis\b"]),
            ("right axis deviation", [r"\bright axis deviation\b", r"\brightward axis\b"]),
            ("extreme axis deviation", [r"\bextreme axis\b"]),
            ("normal", [r"\bnormal axis\b", r"\bnormal\b"]),
        ]
    elif key == "q_waves":
        mapping = [
            ("pathologic q waves", [r"\bpathologic q waves?\b", r"\bpathological q waves?\b", r"\bold infarct\b", r"\binfarct\b"]),
            ("q waves present", [r"\bq waves?\b"]),
            ("no q waves", [r"\bno q waves?\b", r"\babsent\b", r"\bnone\b"]),
        ]
    elif key == "r_wave_progression":
        mapping = [
            ("poor r-wave progression", [r"\bpoor r[- ]wave progression\b"]),
            ("normal progression", [r"\bnormal progression\b", r"\bnormal\b"]),
            ("abnormal progression", [r"\babnormal progression\b"]),
        ]
    elif key == "infarction_territory":
        mapping = [
            ("inferior", [r"\binferior\b"]),
            ("anterior", [r"\banterior\b"]),
            ("anteroseptal", [r"\banteroseptal\b", r"\bseptal\b"]),
            ("lateral", [r"\blateral\b"]),
        ]
    elif key == "supraventricular_vs_ventricular":
        mapping = [
            ("supraventricular", [r"\bsupraventricular\b", r"\batrial\b", r"\bsinus\b"]),
            ("ventricular", [r"\bventricular\b"]),
        ]
    else:
        mapping = []

    for label, patterns in mapping:
        if any(re.search(pattern, text) for pattern in patterns):
            classes.append(label)
    if classes:
        predicted = classes[0]
    return {
        "predicted_class": predicted,
        "candidate_classes": deduplicate_ordered(classes),
    }


def parse_numeric_answer(answer: str) -> Dict[str, object]:
    """Parse a numeric answer into value and/or range."""
    text = normalize_text(answer)
    range_match = _NUMBER_RANGE_PATTERN.search(text)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        if low > high:
            low, high = high, low
        return {
            "predicted_numeric": None,
            "predicted_range": [low, high],
            "raw_numbers": [low, high],
        }

    numbers = [float(match.group(1)) for match in _NUMBER_PATTERN.finditer(text)]
    numeric_value = numbers[0] if numbers else None
    return {
        "predicted_numeric": numeric_value,
        "predicted_range": None,
        "raw_numbers": numbers,
    }


def expand_lead_range(start: str, end: str) -> List[str]:
    """Expand a precordial lead range like V1-V4 into individual leads."""
    first = start.upper()
    second = end.upper()
    if not first.startswith("V"):
        first = f"V{first}"
    if not second.startswith("V"):
        second = f"V{second}"
    start_idx = int(first[1:])
    end_idx = int(second[1:])
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    return [f"V{index}" for index in range(start_idx, end_idx + 1)]


def normalize_lead(lead: str) -> str:
    """Normalize a lead token into canonical ECG lead casing."""
    token = lead.strip()
    upper = token.upper()
    if upper in {"I", "II", "III"}:
        return upper
    if upper in {"AVR", "AVL", "AVF"}:
        return f"aV{upper[-1]}"
    if upper.startswith("V") and upper[1:].isdigit():
        return f"V{int(upper[1:])}"
    return token


def extract_leads(text: str) -> List[str]:
    """Extract normalized ECG leads from free text."""
    normalized = normalize_text(text)
    leads: List[str] = []
    for start, prefix, end in _LEAD_RANGE_PATTERN.findall(normalized):
        leads.extend(expand_lead_range(start, f"{prefix}{end}"))
    for lead in _LEAD_PATTERN.findall(normalized):
        leads.append(normalize_lead(lead))
    present = {normalize_lead(item) for item in leads}
    return [lead for lead in LEAD_ORDER if lead in present]


def overlap_with_range(predicted_numeric: float | None, predicted_range: Sequence[float] | None, true_range: Sequence[float] | None) -> bool | None:
    """Check whether a parsed numeric answer is compatible with a true range."""
    if true_range is None:
        return None
    true_low, true_high = true_range
    if predicted_numeric is not None:
        return true_low <= predicted_numeric <= true_high
    if predicted_range is not None:
        pred_low, pred_high = predicted_range
        return not (pred_high < true_low or pred_low > true_high)
    return False
