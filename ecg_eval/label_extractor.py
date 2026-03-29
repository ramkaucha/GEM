"""Rule-based ECG finding extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Pattern, Tuple

from .normalize import deduplicate_ordered, normalize_text


DEFAULT_LABEL_PATTERNS: Dict[str, List[str]] = {
    "sinus rhythm": [r"\bsinus rhythm\b", r"\bnormal sinus rhythm\b"],
    "sinus tachycardia": [r"\bsinus tachycardia\b"],
    "sinus bradycardia": [r"\bsinus bradycardia\b", r"\bslow ventricular response\b"],
    "atrial fibrillation": [r"\batrial fibrillation\b", r"\ba[- ]fib\b", r"\bafib\b"],
    "atrial flutter": [r"\batrial flutter\b"],
    "pvc": [
        r"\bpvc\b",
        r"\bpvcs\b",
        r"\bpremature ventricular contractions?\b",
        r"\bventricular extrasystoles?\b",
        r"\baberrant ventricular conduction\b",
        r"\bventricular premature beats?\b",
    ],
    "pac": [
        r"\bpac\b",
        r"\bpacs\b",
        r"\bpremature atrial contractions?\b",
        r"\bsupraventricular extrasystoles?\b",
        r"\bpremature supraventricular beats?\b",
    ],
    "pause": [r"\bpause\b", r"\bsinus pause\b"],
    "dropped beat": [r"\bdropped beat\b"],
    "bigeminy": [r"\bbigeminy\b"],
    "trigeminy": [r"\btrigeminy\b"],
    "right bundle branch block": [r"\bright bundle branch block\b", r"\brbbb\b"],
    "left bundle branch block": [r"\bleft bundle branch block\b", r"\blbbb\b"],
    "bifascicular block": [r"\bbifascicular block\b"],
    "first degree av block": [
        r"\bfirst degree a[- ]?v block\b",
        r"\b1st degree a[- ]?v block\b",
        r"\bfirst degree av block\b",
        r"\bprolonged pr\b",
    ],
    "second degree av block": [r"\bsecond degree a[- ]?v block\b", r"\bmobitz\b"],
    "third degree av block": [r"\bthird degree a[- ]?v block\b", r"\bcomplete heart block\b"],
    "left axis deviation": [r"\bleft axis deviation\b", r"\bleftward axis\b"],
    "right axis deviation": [r"\bright axis deviation\b", r"\brightward axis\b"],
    "extreme axis deviation": [r"\bextreme axis deviation\b"],
    "inferior infarct": [r"\binferior infarct\b"],
    "anterior infarct": [r"\banterior infarct\b", r"\banteroseptal infarct\b", r"\bseptal infarct\b"],
    "lateral infarct": [r"\blateral infarct\b", r"\bhigh lateral infarct\b"],
    "st elevation": [r"\bst elevation\b", r"\bst[- ]segment elevation\b"],
    "st depression": [r"\bst depression\b", r"\bst[- ]segment depression\b"],
    "t wave inversion": [r"\bt wave inversion\b", r"\bt[- ]wave inversion\b", r"\bt waves? inverted\b"],
    "prolonged qt": [r"\bprolonged qt\b"],
    "prolonged qtc": [r"\bprolonged qtc\b", r"\bqtc prolong"],
    "low qrs voltage": [r"\blow qrs voltage\b", r"\blow voltage qrs\b", r"\blow limb lead voltage\b"],
    "lvh": [r"\blvh\b", r"\bleft ventricular hypertrophy\b"],
    "left atrial enlargement": [r"\bleft atrial enlargement\b"],
    "right atrial abnormality": [r"\bright atrial abnormality\b", r"\bright atrial enlargement\b"],
    "abnormal ecg": [r"\babnormal ecg\b"],
    "normal ecg": [r"\bnormal ecg\b"],
    "borderline ecg": [r"\bborderline ecg\b"],
    "intraventricular conduction delay": [
        r"\bintraventricular conduction delay\b",
        r"\bintraventricular conduction defect\b",
        r"\biv conduction defect\b",
        r"\bnon[- ]specific intraventricular block\b",
    ],
    "atrial enlargement": [r"\batrial enlargement\b", r"\batrial abnormality\b"],
    "q waves": [r"\bq waves?\b"],
    "poor r-wave progression": [r"\bpoor r[- ]wave progression\b"],
    "hyperkalemia": [r"\bhyperkalemia\b", r"\bpeaked t waves?\b"],
    "u waves": [r"\bu waves?\b"],
    "ischemia": [r"\bischemia\b", r"\bischemic\b", r"\bmyocardial injury\b"],
}

GENERIC_LABELS = {"abnormal ecg", "normal ecg", "borderline ecg"}


@dataclass(frozen=True)
class LabelMatch:
    """Structured label extraction result."""

    labels: List[str]
    canonical_to_patterns: Dict[str, List[str]]


class ECGLabelExtractor:
    """Regex-backed extractor for common ECG findings."""

    def __init__(self, label_patterns: Dict[str, Iterable[str]] | None = None) -> None:
        patterns = label_patterns or DEFAULT_LABEL_PATTERNS
        self.label_patterns = {label: list(values) for label, values in patterns.items()}
        self.compiled_patterns: Dict[str, List[Pattern[str]]] = {
            label: [re.compile(pattern) for pattern in pattern_list]
            for label, pattern_list in self.label_patterns.items()
        }

    def extract(self, text: object) -> List[str]:
        """Extract canonical labels from text in text-order where possible."""
        normalized = normalize_text(text)
        indexed_matches: List[Tuple[int, str]] = []
        for canonical, patterns in self.compiled_patterns.items():
            positions = [match.start() for pattern in patterns for match in pattern.finditer(normalized)]
            if positions:
                indexed_matches.append((min(positions), canonical))
        ordered = [label for _, label in sorted(indexed_matches, key=lambda item: (item[0], item[1]))]
        return deduplicate_ordered(ordered)

    def extract_non_generic(self, text: object) -> List[str]:
        """Extract labels excluding generic normal/abnormal summary labels."""
        return [label for label in self.extract(text) if label not in GENERIC_LABELS]

    def extract_with_details(self, text: object) -> LabelMatch:
        """Return labels plus the source pattern map for debugging."""
        labels = self.extract(text)
        return LabelMatch(labels=labels, canonical_to_patterns=self.label_patterns)
