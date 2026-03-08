"""Phase 371 — Cytokine Release Syndrome (CRS) Risk Predictor.

Predicts CRS risk for immunotherapy drugs (bispecific antibodies, CAR-T, mAbs).
"""

from dataclasses import dataclass

VALID_DRUG_CLASSES = {"bispecific_ab", "car_t", "monoclonal_ab", "adc"}
VALID_ANTIGENS = {"cd3", "cd19", "cd20", "cd38", "her2", "other"}

BASE_RISK = {
    "bispecific_ab": 35,
    "car_t": 50,
    "monoclonal_ab": 10,
    "adc": 8,
}

ANTIGEN_MODIFIER = {
    "cd3": 20,
    "cd19": 20,
    "cd20": 10,
    "cd38": 10,
    "her2": 5,
    "other": 5,
}

CRS_GRADES = [
    (20, "grade_0"),
    (40, "grade_1"),
    (60, "grade_2"),
    (80, "grade_3"),
]


@dataclass
class CRSRiskResult:
    """Result of CRS risk prediction."""

    drug_name: str
    drug_class: str
    crs_risk_score: float
    predicted_crs_grade: str
    score_components: dict
    risk_factors: list
    mitigation_strategies: list
    monitoring_frequency: str
    notes: str


def _predict_crs_grade(score: float) -> str:
    for threshold, grade in CRS_GRADES:
        if score < threshold:
            return grade
    return "grade_4"


def _monitoring_frequency(grade: str) -> str:
    if grade in {"grade_0", "grade_1"}:
        return "standard"
    if grade == "grade_2":
        return "enhanced"
    return "intensive"


def _mitigation_strategies(grade: str, premedication: bool) -> list:
    strategies = []
    if not premedication:
        strategies.append("Administer corticosteroid premedication")
    if grade in {"grade_1", "grade_2"}:
        strategies.append("Slow infusion rate")
        strategies.append("Step-up dosing regimen")
        strategies.append("Antipyretic premedication")
    if grade in {"grade_2", "grade_3"}:
        strategies.append("Tocilizumab (IL-6R antagonist) on standby")
        strategies.append("Hospitalization for monitoring")
    if grade in {"grade_3", "grade_4"}:
        strategies.append("ICU-level monitoring required")
        strategies.append("High-dose corticosteroids")
        strategies.append("Vasopressor support if needed")
    if grade == "grade_0":
        strategies.append("Standard outpatient monitoring")
    return strategies


def predict_crs_risk(
    drug_name: str,
    drug_class: str,
    target_antigen: str,
    dose_mg_per_kg: float,
    t_cell_engagement: bool,
    cytokine_amplification_factor: float,
    infusion_rate_mg_per_h: float,
    premedication: bool,
) -> CRSRiskResult:
    """Predict CRS risk score and grade for immunotherapy agents.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    drug_class:
        One of "bispecific_ab", "car_t", "monoclonal_ab", "adc".
    target_antigen:
        Target antigen: "cd3", "cd19", "cd20", "cd38", "her2", or "other".
    dose_mg_per_kg:
        Dose in mg/kg (must be > 0).
    t_cell_engagement:
        True if drug redirects T cells (highest CRS risk).
    cytokine_amplification_factor:
        Tumor burden proxy in range [1, 10].
    infusion_rate_mg_per_h:
        Infusion rate in mg/h (must be > 0).
    premedication:
        True if steroid pretreatment administered.

    Returns
    -------
    CRSRiskResult
        Comprehensive CRS risk assessment.
    """
    if drug_class not in VALID_DRUG_CLASSES:
        raise ValueError(
            f"Invalid drug_class '{drug_class}'. Must be one of {sorted(VALID_DRUG_CLASSES)}."
        )
    if target_antigen not in VALID_ANTIGENS:
        raise ValueError(
            f"Invalid target_antigen '{target_antigen}'. Must be one of {sorted(VALID_ANTIGENS)}."
        )
    if dose_mg_per_kg <= 0:
        raise ValueError(f"dose_mg_per_kg must be > 0, got {dose_mg_per_kg}.")
    if infusion_rate_mg_per_h <= 0:
        raise ValueError(f"infusion_rate_mg_per_h must be > 0, got {infusion_rate_mg_per_h}.")
    if not (1.0 <= cytokine_amplification_factor <= 10.0):
        raise ValueError(
            f"cytokine_amplification_factor must be in [1, 10], got {cytokine_amplification_factor}."
        )

    components: dict = {}

    # Base risk from drug class
    base = BASE_RISK[drug_class]
    components["base_class_risk"] = base

    # Antigen density modifier
    antigen_score = ANTIGEN_MODIFIER[target_antigen]
    components["antigen_density"] = antigen_score

    # Dose modifier
    dose_score = min(20.0, dose_mg_per_kg * 5.0)
    components["dose_modifier"] = dose_score

    # T-cell engagement
    t_cell_score = 15.0 if t_cell_engagement else 0.0
    components["t_cell_engagement"] = t_cell_score

    # Cytokine amplification
    amp_score = min(20.0, cytokine_amplification_factor * 2.0)
    components["cytokine_amplification"] = amp_score

    # Infusion rate
    if infusion_rate_mg_per_h < 0.1:
        infusion_score = 0.0
    elif infusion_rate_mg_per_h <= 1.0:
        infusion_score = 5.0
    else:
        infusion_score = 15.0
    components["infusion_rate"] = infusion_score

    # Premedication reduction
    premed_score = -10.0 if premedication else 0.0
    components["premedication"] = premed_score

    raw = (
        base + antigen_score + dose_score + t_cell_score + amp_score + infusion_score + premed_score
    )
    crs_risk_score = min(100.0, max(0.0, raw))

    grade = _predict_crs_grade(crs_risk_score)
    monitoring = _monitoring_frequency(grade)
    mitigations = _mitigation_strategies(grade, premedication)

    risk_factors = [name for name, score in components.items() if score > 0]

    notes_parts = [
        f"Drug class '{drug_class}' with target '{target_antigen}'.",
        f"Raw score: {raw:.1f}, clamped to {crs_risk_score:.1f}.",
        f"CRS grade predicted: {grade}.",
    ]
    if t_cell_engagement:
        notes_parts.append("T-cell redirecting therapy — high CRS risk.")
    if premedication:
        notes_parts.append("Premedication applied (-10 risk reduction).")

    return CRSRiskResult(
        drug_name=drug_name,
        drug_class=drug_class,
        crs_risk_score=crs_risk_score,
        predicted_crs_grade=grade,
        score_components=components,
        risk_factors=risk_factors,
        mitigation_strategies=mitigations,
        monitoring_frequency=monitoring,
        notes=" ".join(notes_parts),
    )
