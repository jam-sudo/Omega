"""Carcinogenicity risk prediction from structural features (alert-based model)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CarcinogenicityResult:
    drug_name: str
    carcinogenicity_risk: str  # "negative","low_concern","moderate_concern","high_concern"
    risk_score: float  # 0-100
    structural_alerts: tuple  # triggered alert names
    genotoxic_potential: bool
    epigenetic_concern: bool  # non-genotoxic carcinogenicity mechanisms
    hormesis_concern: bool  # hormonal/receptor-mediated
    threshold_dose_mg: float  # estimated NOAEL proxy (simplified)
    td50_mg_per_kg_day: float  # estimated TD50 (crude model)
    safety_margin_fold: float  # TD50 / (daily_dose/70kg)
    regulatory_flag: str  # "clean","flag_for_review","likely_carcinogen"
    confidence: str  # "high","medium","low"
    notes: str


def estimate_td50(
    logP: float,
    mw: float,
    n_alerts: int,
    genotoxic: bool,
) -> float:
    """
    Estimate TD50 (mg/kg/day) using simplified QSAR.
    Higher alerts and genotoxicity -> lower TD50.
    """
    base_td50 = 1000.0  # mg/kg/day for "clean" drug

    # Each alert divides TD50 by 3
    for _ in range(n_alerts):
        base_td50 /= 3.0

    # Genotoxic: divide by 10
    if genotoxic:
        base_td50 /= 10.0

    # Biopersistence (logP > 5)
    if logP > 5.0:
        base_td50 /= 2.0

    # Enforce minimum
    return max(0.01, base_td50)


def predict_carcinogenicity(
    drug_name: str,
    daily_dose_mg: float,
    logP: float,
    mw: float,
    n_aromatic_rings: int = 0,
    has_nitro_group: bool = False,
    has_azo_linkage: bool = False,
    has_alkylating_group: bool = False,
    has_michael_acceptor: bool = False,
    has_epoxide: bool = False,
    has_intercalator: bool = False,
    n_hbd: int = 2,
    psa: float = 80.0,
    is_hormonal: bool = False,
) -> CarcinogenicityResult:
    """Predict carcinogenicity risk from structural and physicochemical features."""
    alerts = []
    score = 0.0

    # --- Genotoxic alerts ---
    genotoxic_potential = False

    if has_nitro_group:
        alerts.append("nitro_reduction")
        score += 30.0
        genotoxic_potential = True

    if has_azo_linkage:
        alerts.append("azo_reduction")
        score += 25.0
        genotoxic_potential = True

    if has_alkylating_group:
        alerts.append("alkylation")
        score += 35.0
        genotoxic_potential = True

    if has_epoxide:
        alerts.append("epoxide_reactivity")
        score += 30.0
        genotoxic_potential = True

    if has_michael_acceptor:
        alerts.append("michael_addition")
        score += 20.0
        genotoxic_potential = True

    # --- Non-genotoxic / epigenetic alerts ---
    epigenetic_from_structure = False

    if n_aromatic_rings >= 3:
        alerts.append("polycyclic_aromatic")
        score += 15.0
        epigenetic_from_structure = True

    if has_intercalator:
        alerts.append("dna_intercalation")
        score += 20.0
        epigenetic_from_structure = True

    # --- Hormonal / receptor-mediated ---
    hormesis_concern = False
    if is_hormonal:
        alerts.append("hormonal_disruption")
        score += 10.0
        hormesis_concern = True
        epigenetic_from_structure = True  # hormonal is an epigenetic mechanism

    epigenetic_concern = epigenetic_from_structure

    # --- Physicochemical modifiers ---
    if logP > 5.0:
        score += 5.0  # biopersistence

    if mw > 500.0:
        score += 5.0  # potential accumulation

    # Clamp score to [0, 100]
    score = max(0.0, min(100.0, score))

    # --- TD50 estimation ---
    n_alerts = len(alerts)
    td50 = estimate_td50(logP, mw, n_alerts, genotoxic_potential)

    # --- Safety margin ---
    daily_dose_per_kg = daily_dose_mg / 70.0
    if daily_dose_per_kg > 1e-12:
        safety_margin = td50 / daily_dose_per_kg
    else:
        safety_margin = 10000.0
    safety_margin = min(10000.0, safety_margin)

    # --- Threshold dose (NOAEL proxy) ---
    threshold_dose_mg = td50 * 70.0 * 0.1  # 10% of TD50 for 70kg human

    # --- Carcinogenicity risk classification ---
    if score < 10.0:
        carcinogenicity_risk = "negative"
    elif score < 30.0:
        carcinogenicity_risk = "low_concern"
    elif score < 60.0:
        carcinogenicity_risk = "moderate_concern"
    else:
        carcinogenicity_risk = "high_concern"

    # --- Regulatory flag ---
    if genotoxic_potential:
        regulatory_flag = "likely_carcinogen"
    elif score > 30.0:
        regulatory_flag = "flag_for_review"
    else:
        regulatory_flag = "clean"

    # --- Confidence ---
    if genotoxic_potential:
        confidence = "high"
    elif score > 10.0:
        confidence = "medium"
    else:
        confidence = "low"

    # --- Notes ---
    alert_str = ", ".join(alerts) if alerts else "none"
    notes = (
        f"Structural alerts: {alert_str}. "
        f"Score={score:.1f}/100, genotoxic={genotoxic_potential}, "
        f"epigenetic={epigenetic_concern}, hormesis={hormesis_concern}. "
        f"TD50={td50:.2f} mg/kg/day, safety margin={safety_margin:.1f}x."
    )

    return CarcinogenicityResult(
        drug_name=drug_name,
        carcinogenicity_risk=carcinogenicity_risk,
        risk_score=score,
        structural_alerts=tuple(alerts),
        genotoxic_potential=genotoxic_potential,
        epigenetic_concern=epigenetic_concern,
        hormesis_concern=hormesis_concern,
        threshold_dose_mg=threshold_dose_mg,
        td50_mg_per_kg_day=td50,
        safety_margin_fold=safety_margin,
        regulatory_flag=regulatory_flag,
        confidence=confidence,
        notes=notes,
    )
