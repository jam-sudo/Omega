"""Phase 364 — Carcinogenicity Structural Alert Screener.

Screen drug candidates for carcinogenicity structural alerts using ICH S1/M7 guidelines.
"""

from __future__ import annotations

from dataclasses import dataclass

# Alert weights per ICH M7 potency classification
_ALERT_WEIGHTS: dict = {
    "nitrosamine": 30,
    "aromatic_amine": 20,
    "aflatoxin": 30,
    "alkylating": 25,
    "intercalating": 15,
    "epoxide": 20,
    "michael_acceptor": 10,
    "polycyclic_aromatic": 15,
    "hydrazine": 20,
    "azo_dye": 10,
}

# Mapping from smiles_features keys to alert names
_FEATURE_TO_ALERT: dict = {
    "has_nitrosamine": "nitrosamine",
    "has_aromatic_amine": "aromatic_amine",
    "has_aflatoxin_scaffold": "aflatoxin",
    "has_alkylating_group": "alkylating",
    "has_intercalating_scaffold": "intercalating",
    "has_epoxide": "epoxide",
    "has_michael_acceptor": "michael_acceptor",
    "has_polycyclic_aromatic": "polycyclic_aromatic",
    "has_hydrazine": "hydrazine",
    "has_azo_dye": "azo_dye",
}

# ICH M7 regulatory limits (µg/day)
_REGULATORY_LIMITS: dict = {
    "cohort_of_concern": 0.03,
    "high_concern": 1.5,
    "moderate_concern": 10.0,
    "low_concern": 150.0,
}

# Cohort of concern alerts (highest potency per ICH M7)
_COHORT_OF_CONCERN_ALERTS: set = {"nitrosamine", "aflatoxin", "alkylating"}


@dataclass
class CarcinogenicityScreenResult:
    """Result of carcinogenicity structural alert screening."""

    drug_name: str
    alerts_present: list
    raw_alert_score: float
    carcinogenicity_score: float
    ich_m7_class: str
    regulatory_limit_ug_per_day: float
    cohort_of_concern: bool
    needs_mutagenicity_testing: bool
    notes: str


def screen_carcinogenicity(
    drug_name: str,
    smiles_features: dict,
) -> CarcinogenicityScreenResult:
    """Screen a drug for carcinogenicity structural alerts.

    Parameters
    ----------
    drug_name:
        Name of the drug or candidate.
    smiles_features:
        Dictionary of boolean structural features and physicochemical properties.
        Expected keys:
            has_nitrosamine, has_aromatic_amine, has_aflatoxin_scaffold,
            has_alkylating_group, has_intercalating_scaffold, has_epoxide,
            has_michael_acceptor, has_polycyclic_aromatic, has_hydrazine,
            has_azo_dye  (all bool)
            logP (float), mw_Da (float)

    Returns
    -------
    CarcinogenicityScreenResult
    """
    # --- Validation ---
    logP = smiles_features.get("logP", 2.0)
    mw_Da = smiles_features.get("mw_Da", 300.0)

    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if not (-5.0 <= logP <= 10.0):
        raise ValueError(f"logP must be in [-5, 10], got {logP}")

    # --- Identify fired alerts ---
    alerts_present: list = []
    raw_score = 0.0

    for feature_key, alert_name in _FEATURE_TO_ALERT.items():
        if smiles_features.get(feature_key, False):
            alerts_present.append(alert_name)
            raw_score += _ALERT_WEIGHTS[alert_name]

    # --- Normalize score ---
    carcinogenicity_score = min(100.0, raw_score)

    # --- Cohort of concern determination ---
    is_cohort_of_concern = any(a in _COHORT_OF_CONCERN_ALERTS for a in alerts_present)

    # --- ICH M7 class ---
    if is_cohort_of_concern:
        ich_m7_class = "cohort_of_concern"
    elif carcinogenicity_score >= 50.0:
        ich_m7_class = "high_concern"
    elif carcinogenicity_score >= 20.0:
        ich_m7_class = "moderate_concern"
    else:
        ich_m7_class = "low_concern"

    # --- Regulatory limit ---
    regulatory_limit = _REGULATORY_LIMITS[ich_m7_class]

    # --- Mutagenicity testing needed ---
    needs_mutagenicity_testing = carcinogenicity_score >= 20.0

    # --- Notes ---
    note_parts = []
    if not alerts_present:
        note_parts.append("No structural alerts detected; low carcinogenicity concern.")
    else:
        note_parts.append(f"Alerts fired: {', '.join(alerts_present)}.")
    if is_cohort_of_concern:
        note_parts.append(
            "Cohort of concern: ICH M7 requires specific risk management (TTC 0.03 µg/day)."
        )
    if needs_mutagenicity_testing:
        note_parts.append("Mutagenicity testing (Ames assay) recommended per ICH M7.")
    note_parts.append(f"Regulatory limit: {regulatory_limit} µg/day (ICH M7 {ich_m7_class}).")
    if logP > 5.0:
        note_parts.append("High logP may increase membrane partitioning and exposure.")
    notes = " ".join(note_parts)

    return CarcinogenicityScreenResult(
        drug_name=drug_name,
        alerts_present=alerts_present,
        raw_alert_score=raw_score,
        carcinogenicity_score=carcinogenicity_score,
        ich_m7_class=ich_m7_class,
        regulatory_limit_ug_per_day=regulatory_limit,
        cohort_of_concern=is_cohort_of_concern,
        needs_mutagenicity_testing=needs_mutagenicity_testing,
        notes=notes,
    )
