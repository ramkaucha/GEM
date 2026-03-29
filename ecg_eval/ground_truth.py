"""Question-specific ground-truth derivation from ECG reports."""

from __future__ import annotations

import re
from typing import Dict, List

from .label_extractor import ECGLabelExtractor
from .normalize import normalize_text
from .parsing import extract_leads
from .questioning import prompt_key

REGION_TO_LEADS = {
    "inferior": ["II", "III", "aVF"],
    "anterior": ["V2", "V3", "V4"],
    "anteroseptal": ["V1", "V2", "V3", "V4"],
    "lateral": ["I", "aVL", "V5", "V6"],
}


class GroundTruthDeriver:
    """Derive conservative, question-specific targets from a report."""

    def __init__(self, extractor: ECGLabelExtractor | None = None) -> None:
        self.extractor = extractor or ECGLabelExtractor()

    def derive(self, prompt: str, report: str, question_type: str) -> Dict[str, object]:
        """Return a structured ground-truth payload for one prompt/report pair."""
        normalized_report = normalize_text(report)
        report_labels = self.extractor.extract(report)
        report_labels_non_generic = self.extractor.extract_non_generic(report)
        key = prompt_key(prompt)
        base = {
            "question_key": key,
            "question_type": question_type,
            "report_labels": report_labels,
            "report_labels_non_generic": report_labels_non_generic,
            "scorable": False,
            "weakly_scorable": False,
            "warning": None,
            "notes": [],
        }

        if question_type == "binary_classification":
            base.update(self._derive_binary(key, normalized_report, report_labels, report_labels_non_generic))
        elif question_type == "multiclass_classification":
            base.update(self._derive_multiclass(key, normalized_report, report_labels, report_labels_non_generic))
        elif question_type == "numeric":
            base.update(self._derive_numeric(key, normalized_report, report_labels))
        elif question_type == "lead_based":
            base.update(self._derive_leads(normalized_report, report_labels_non_generic))
        elif question_type == "diagnosis_label":
            base.update(self._derive_diagnosis(key, report_labels_non_generic, report_labels))
        elif question_type == "summary_generation":
            base.update(self._derive_summary(report, report_labels_non_generic, report_labels))
        return base

    def _derive_binary(self, key: str, report: str, labels: List[str], non_generic: List[str]) -> Dict[str, object]:
        notes: List[str] = []
        true_class = None

        if key == "rhythm_regularity":
            if "atrial fibrillation" in labels:
                true_class = "irregular"
            elif any(label in labels for label in ["sinus rhythm", "sinus bradycardia", "sinus tachycardia"]):
                true_class = "regular"
            else:
                return {"warning": "Unable to infer rhythm regularity conservatively."}
            return {"scorable": True, "true_class": true_class, "notes": notes}

        if key == "p_waves_before_qrs":
            if any(label in labels for label in ["sinus rhythm", "sinus bradycardia", "sinus tachycardia"]):
                true_class = "yes"
            elif any(label in labels for label in ["atrial fibrillation", "atrial flutter"]):
                true_class = "no"
            else:
                return {"warning": "Report does not safely specify P-wave relation to QRS."}
            return {"scorable": True, "true_class": true_class, "notes": notes}

        if key == "overall_normal_abnormal":
            if "normal ecg" in labels:
                true_class = "normal"
            elif "abnormal ecg" in labels:
                true_class = "abnormal"
            elif "borderline ecg" in labels:
                true_class = "abnormal"
                notes.append("Mapped borderline ECG to abnormal for strict binary scoring.")
            elif non_generic:
                true_class = "abnormal"
                notes.append("Mapped presence of non-generic report findings to abnormal.")
            else:
                return {"warning": "Unable to infer normal/abnormal status."}
            return {"scorable": True, "true_class": true_class, "notes": notes}

        label_mapping = {
            "evidence_sinus_rhythm": ["sinus rhythm", "sinus bradycardia", "sinus tachycardia"],
            "evidence_atrial_fibrillation": ["atrial fibrillation"],
            "evidence_atrial_flutter": ["atrial flutter"],
            "premature_beats": ["pvc", "pac", "bigeminy", "trigeminy"],
            "pause_or_dropped_beat": ["pause", "dropped beat"],
            "first_degree_av_block": ["first degree av block"],
            "second_degree_av_block": ["second degree av block"],
            "third_degree_av_block": ["third degree av block"],
            "right_bundle_branch_block": ["right bundle branch block"],
            "left_bundle_branch_block": ["left bundle branch block"],
            "intraventricular_conduction_delay": ["intraventricular conduction delay", "right bundle branch block", "left bundle branch block", "bifascicular block"],
            "atrial_enlargement": ["atrial enlargement", "left atrial enlargement", "right atrial abnormality"],
            "poor_r_wave_progression": ["poor r-wave progression"],
            "st_elevation": ["st elevation"],
            "st_depression": ["st depression"],
            "t_wave_inversion": ["t wave inversion"],
            "hyperkalemia_signs": ["hyperkalemia"],
            "u_waves": ["u waves"],
            "ventricular_hypertrophy": ["lvh"],
        }
        positives = label_mapping.get(key)
        if positives is not None:
            true_class = "yes" if any(label in labels for label in positives) else "no"
            if true_class == "no":
                notes.append("Derived negative from absence of salient report finding.")
            return {"scorable": True, "true_class": true_class, "notes": notes}

        if key == "acute_ischemia_or_infarction":
            if any(label in labels for label in ["ischemia", "st elevation", "st depression", "inferior infarct", "anterior infarct", "lateral infarct"]):
                true_class = "yes"
                if "age undetermined" in report:
                    notes.append("Positive for ischemia/infarction, but acuity is ambiguous in report wording.")
                return {"scorable": True, "true_class": true_class, "notes": notes}
            if "normal ecg" in labels or "borderline ecg" in labels:
                return {"scorable": True, "true_class": "no", "notes": notes}
            return {"warning": "Ischemia/infarction status is too ambiguous to score conservatively."}

        return {"warning": f"No binary ground-truth rule implemented for {key}."}

    def _derive_multiclass(self, key: str, report: str, labels: List[str], non_generic: List[str]) -> Dict[str, object]:
        if key == "rate_class":
            if "sinus tachycardia" in labels:
                return {"scorable": True, "true_class": "tachycardia"}
            if "sinus bradycardia" in labels:
                return {"scorable": True, "true_class": "bradycardia"}
            if any(label in labels for label in ["sinus rhythm", "normal ecg", "borderline ecg"]):
                return {"scorable": True, "true_class": "normal rate"}
            return {"warning": "Unable to derive rate class from report."}

        if key == "axis_class":
            if "left axis deviation" in labels:
                return {"scorable": True, "true_class": "left axis deviation"}
            if "right axis deviation" in labels:
                return {"scorable": True, "true_class": "right axis deviation"}
            if "extreme axis deviation" in labels:
                return {"scorable": True, "true_class": "extreme axis deviation"}
            return {"scorable": True, "true_class": "normal", "notes": ["No axis deviation mentioned; treated as normal axis."]}

        if key == "q_waves":
            if any(label in labels for label in ["inferior infarct", "anterior infarct", "lateral infarct", "q waves"]):
                return {"scorable": True, "true_class": "pathologic q waves"}
            if "normal ecg" in labels:
                return {"scorable": True, "true_class": "no q waves"}
            return {"warning": "Report does not safely determine Q-wave class."}

        if key == "r_wave_progression":
            if "poor r-wave progression" in labels:
                return {"scorable": True, "true_class": "poor r-wave progression"}
            if "normal ecg" in labels:
                return {"scorable": True, "true_class": "normal progression"}
            return {"warning": "Report does not explicitly describe R-wave progression."}

        if key == "infarction_territory":
            if "inferior infarct" in labels:
                return {"scorable": True, "true_class": "inferior"}
            if "anterior infarct" in labels:
                return {"scorable": True, "true_class": "anterior" if "anteroseptal" not in report else "anteroseptal"}
            if "lateral infarct" in labels:
                return {"scorable": True, "true_class": "lateral"}
            return {"warning": "No infarct territory stated in report."}

        if key == "supraventricular_vs_ventricular":
            has_supra = any(label in labels for label in ["sinus rhythm", "sinus tachycardia", "sinus bradycardia", "atrial fibrillation", "atrial flutter", "pac"])
            has_ventricular = any(label in labels for label in ["pvc"])
            if has_supra and not has_ventricular:
                return {"scorable": True, "true_class": "supraventricular"}
            if has_ventricular and not has_supra:
                return {"scorable": True, "true_class": "ventricular"}
            return {"warning": "Report mixes supraventricular and ventricular clues."}

        return {"warning": f"No multiclass ground-truth rule implemented for {key}."}

    def _derive_numeric(self, key: str, report: str, labels: List[str]) -> Dict[str, object]:
        if key == "heart_rate":
            if "sinus bradycardia" in labels:
                return {"scorable": True, "true_range": [0.0, 59.0], "tolerance": 0.0, "notes": ["Derived only a bradycardic rate range from report."]}
            if "sinus tachycardia" in labels:
                return {"scorable": True, "true_range": [101.0, 250.0], "tolerance": 0.0, "notes": ["Derived only a tachycardic rate range from report."]}
            if any(label in labels for label in ["sinus rhythm", "normal ecg", "borderline ecg"]):
                return {"scorable": True, "true_range": [60.0, 100.0], "tolerance": 0.0, "notes": ["Derived only a normal rate range from report."]}
            return {"warning": "Report does not safely provide heart-rate ground truth."}

        if key == "pr_interval":
            if "first degree av block" in labels:
                return {
                    "scorable": True,
                    "true_range": [200.0, 400.0],
                    "tolerance": 20.0,
                    "true_descriptor": "prolonged",
                    "notes": ["Estimated PR lower bound from first-degree AV block wording; exact PR is unknown."],
                }
            return {"warning": "Report does not safely provide PR interval."}

        if key == "qrs_duration":
            if any(label in labels for label in ["right bundle branch block", "left bundle branch block", "intraventricular conduction delay", "bifascicular block"]):
                return {
                    "scorable": True,
                    "true_range": [120.0, 220.0],
                    "tolerance": 20.0,
                    "true_descriptor": "wide",
                    "notes": ["Estimated wide-QRS range from conduction-abnormality wording; exact duration is unknown."],
                }
            if "normal ecg" in labels:
                return {
                    "scorable": True,
                    "true_range": [60.0, 110.0],
                    "tolerance": 20.0,
                    "true_descriptor": "narrow",
                    "notes": ["Estimated narrow-QRS range from normal report wording."],
                }
            return {"warning": "Report does not safely provide QRS duration."}

        if key == "qt_interval":
            if any(label in labels for label in ["prolonged qt", "prolonged qtc"]):
                return {
                    "scorable": True,
                    "true_range": [460.0, 650.0],
                    "tolerance": 20.0,
                    "true_descriptor": "prolonged",
                    "notes": ["Estimated prolonged-QT range from report wording; exact QT/QTc is unknown."],
                }
            return {"warning": "Report does not safely provide QT/QTc interval."}

        return {"warning": f"No numeric ground-truth rule implemented for {key}."}

    def _derive_leads(self, report: str, labels: List[str]) -> Dict[str, object]:
        leads = extract_leads(report)
        inferred_regions: List[str] = []
        if not leads:
            for region, region_leads in REGION_TO_LEADS.items():
                if region in report:
                    leads.extend(region_leads)
                    inferred_regions.append(region)
            if "anteroseptal infarct" in report:
                leads.extend(REGION_TO_LEADS["anteroseptal"])
                inferred_regions.append("anteroseptal")
        leads = sorted(set(leads), key=lambda item: ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"].index(item)) if leads else []
        if not leads:
            return {"warning": "Report does not specify leads strongly enough for lead-based scoring."}
        payload = {"scorable": True, "true_leads": leads}
        if inferred_regions:
            payload["weakly_scorable"] = True
            payload["notes"] = [f"Lead set inferred from territory/region wording: {', '.join(sorted(set(inferred_regions)))}."]
        return payload

    def _derive_diagnosis(self, key: str, non_generic: List[str], labels: List[str]) -> Dict[str, object]:
        if key == "underlying_rhythm":
            rhythm_labels = [label for label in non_generic if label in {"sinus rhythm", "sinus tachycardia", "sinus bradycardia", "atrial fibrillation", "atrial flutter"}]
            if rhythm_labels:
                return {"scorable": True, "true_labels": rhythm_labels[:1]}
            return {"warning": "Report does not specify an underlying rhythm label."}

        if key == "primary_diagnosis":
            candidates = [label for label in non_generic if label not in {"left axis deviation", "right axis deviation", "st depression", "st elevation", "t wave inversion"}]
            if candidates:
                return {"scorable": True, "true_labels": candidates[:1]}
            if non_generic:
                return {"scorable": True, "true_labels": non_generic[:1]}
            return {"warning": "No diagnosis label available in report."}

        if key == "diagnostic_considerations":
            if non_generic:
                return {"scorable": True, "true_labels": non_generic[:3]}
            return {"warning": "No report labels available for diagnostic considerations."}

        if key == "key_findings":
            if non_generic:
                return {"scorable": True, "true_labels": non_generic}
            if labels:
                return {"scorable": True, "true_labels": labels}
            return {"warning": "No report labels available for key findings."}

        return {"warning": f"No diagnosis-label rule implemented for {key}."}

    def _derive_summary(self, report: str, non_generic: List[str], labels: List[str]) -> Dict[str, object]:
        target_labels = non_generic or labels
        return {
            "scorable": True,
            "true_labels": target_labels,
            "target_text": report,
        }


def derive_ground_truth(prompt: str, report: str, question_type: str, extractor: ECGLabelExtractor | None = None) -> Dict[str, object]:
    """Convenience wrapper for question-specific report ground truth."""
    return GroundTruthDeriver(extractor=extractor).derive(prompt=prompt, report=report, question_type=question_type)
