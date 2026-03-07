"""Phase 809 — Organ impairment dose adjustment (combined renal + hepatic)."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OrganImpairmentDosingResult:
    drug_name: str
    renal_function: str  # eGFR description: "normal", "mild", "moderate", "severe", "ESRD"
    hepatic_function: str  # Child-Pugh class: "A", "B", "C", "normal"
    egfr_ml_per_min: float
    child_pugh_score: int
    fe_urine: float
    fm_hepatic: float  # fraction metabolized hepatically
    renal_adjustment_factor: float
    hepatic_adjustment_factor: float
    combined_adjustment_factor: float  # product, clamped [0.1, 1.0]
    recommended_dose_mg: float
    recommended_interval_h: float
    monitoring_required: bool
    contraindicated: bool  # if combined_factor < 0.2 → possible CI
    notes: str


def _renal_function_description(egfr: float) -> str:
    """Classify eGFR into renal function category."""
    if egfr >= 90:
        return "normal"
    elif egfr >= 60:
        return "mild"
    elif egfr >= 30:
        return "moderate"
    elif egfr >= 15:
        return "severe"
    else:
        return "ESRD"


def child_pugh_class(score: int) -> str:
    """Return Child-Pugh class (A/B/C) or 'normal' for score < 5."""
    if score < 5:
        return "normal"
    elif score <= 6:
        return "A"
    elif score <= 9:
        return "B"
    else:
        return "C"


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def calculate_organ_impairment_dose(
    drug_name: str,
    normal_dose_mg: float,
    normal_interval_h: float,
    fe_urine: float,  # fraction excreted unchanged renally
    fm_hepatic: float,  # fraction metabolized hepatically
    egfr_ml_per_min: float = 100.0,
    child_pugh_score: int = 5,  # 5-6=A, 7-9=B, 10-15=C
    normal_egfr: float = 100.0,
) -> OrganImpairmentDosingResult:
    """Calculate dose adjustment for combined organ impairment."""
    # Validate inputs
    if not (0.0 <= fe_urine <= 1.0):
        raise ValueError(f"fe_urine must be in [0, 1], got {fe_urine}")
    if not (0.0 <= fm_hepatic <= 1.0):
        raise ValueError(f"fm_hepatic must be in [0, 1], got {fm_hepatic}")
    if normal_dose_mg <= 0:
        raise ValueError(f"normal_dose_mg must be positive, got {normal_dose_mg}")
    if normal_interval_h <= 0:
        raise ValueError(f"normal_interval_h must be positive, got {normal_interval_h}")
    if egfr_ml_per_min < 0:
        raise ValueError(f"egfr_ml_per_min must be non-negative, got {egfr_ml_per_min}")
    if not (5 <= child_pugh_score <= 15):
        raise ValueError(f"child_pugh_score must be in [5, 15], got {child_pugh_score}")

    # Renal adjustment factor
    # rf = 1 - fe_urine * (1 - egfr/normal_egfr), clamped [0.1, 1.0]
    egfr_ratio = egfr_ml_per_min / normal_egfr if normal_egfr > 0 else 0.0
    rf_raw = 1.0 - fe_urine * (1.0 - egfr_ratio)
    renal_factor = _clamp(rf_raw, 0.1, 1.0)

    # Hepatic adjustment factor from Child-Pugh
    cp_class = child_pugh_class(child_pugh_score)
    if cp_class in ("normal", "A"):
        cyp_reduction = 0.0
    elif cp_class == "B":
        cyp_reduction = 0.3
    else:  # C
        cyp_reduction = 0.5

    hf_raw = 1.0 - fm_hepatic * cyp_reduction
    hepatic_factor = _clamp(hf_raw, 0.25, 1.0)

    # Combined factor
    combined_raw = renal_factor * hepatic_factor
    combined_factor = _clamp(combined_raw, 0.1, 1.0)

    # Recommended dose
    recommended_dose = normal_dose_mg * combined_factor

    # Recommended interval (keep same interval; dose adjustment approach)
    recommended_interval = normal_interval_h

    # Monitoring and contraindication
    monitoring_required = combined_factor < 0.7
    contraindicated = combined_factor < 0.2 and (egfr_ml_per_min < 15 or child_pugh_score >= 12)

    # Build notes
    note_parts = []
    renal_desc = _renal_function_description(egfr_ml_per_min)
    if renal_desc != "normal":
        note_parts.append(f"Renal impairment ({renal_desc}, eGFR={egfr_ml_per_min:.0f})")
    if cp_class not in ("normal", "A"):
        note_parts.append(f"Hepatic impairment (Child-Pugh {cp_class}, score={child_pugh_score})")
    if contraindicated:
        note_parts.append("Possible contraindication: use with extreme caution or avoid")
    elif monitoring_required:
        note_parts.append("Enhanced monitoring recommended")
    if not note_parts:
        note_parts.append("No dose adjustment required")

    notes = "; ".join(note_parts)

    return OrganImpairmentDosingResult(
        drug_name=drug_name,
        renal_function=renal_desc,
        hepatic_function=cp_class,
        egfr_ml_per_min=egfr_ml_per_min,
        child_pugh_score=child_pugh_score,
        fe_urine=fe_urine,
        fm_hepatic=fm_hepatic,
        renal_adjustment_factor=renal_factor,
        hepatic_adjustment_factor=hepatic_factor,
        combined_adjustment_factor=combined_factor,
        recommended_dose_mg=recommended_dose,
        recommended_interval_h=recommended_interval,
        monitoring_required=monitoring_required,
        contraindicated=contraindicated,
        notes=notes,
    )


def screen_organ_impairment(
    drug_name: str,
    normal_dose_mg: float,
    normal_interval_h: float,
    fe_urine: float,
    fm_hepatic: float,
    scenarios: list[dict[str, Any]],
) -> list[OrganImpairmentDosingResult]:
    """Screen multiple organ impairment scenarios for a drug.

    Each scenario dict may contain:
      - egfr_ml_per_min (float, default 100)
      - child_pugh_score (int, default 5)
      - normal_egfr (float, default 100)
    """
    results = []
    for scenario in scenarios:
        egfr = float(scenario.get("egfr_ml_per_min", 100.0))
        cp_score = int(scenario.get("child_pugh_score", 5))
        n_egfr = float(scenario.get("normal_egfr", 100.0))
        result = calculate_organ_impairment_dose(
            drug_name=drug_name,
            normal_dose_mg=normal_dose_mg,
            normal_interval_h=normal_interval_h,
            fe_urine=fe_urine,
            fm_hepatic=fm_hepatic,
            egfr_ml_per_min=egfr,
            child_pugh_score=cp_score,
            normal_egfr=n_egfr,
        )
        results.append(result)
    return results
