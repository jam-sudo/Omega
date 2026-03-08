"""Phase 917 — Drug Thyroid Gland Distribution.

Predicts drug concentration in thyroid gland, relevant for radioiodine,
amiodarone (iodine-rich), thyroid-disrupting chemicals, and antithyroid drugs.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ThyroidDistributionResult:
    """Result of thyroid gland drug distribution prediction."""

    drug_name: str
    logp: float
    mw_Da: float
    iodine_content_pct: float
    thyroid_plasma_ratio: float
    predicted_thyroid_conc_ug_g: float
    plasma_conc_input_mg_L: float
    nis_substrate_probability: float
    thyroid_disruption_score: float
    thyroid_toxicity_concern: bool
    notes: str


def predict_thyroid_distribution(
    drug_name: str,
    logp: float = 2.0,
    mw_Da: float = 400.0,
    plasma_conc_mg_L: float = 1.0,
    iodine_content_pct: float = 0.0,
    fu_plasma: float = 0.5,
) -> ThyroidDistributionResult:
    """Predict drug distribution to thyroid gland.

    Args:
        drug_name: Name of the drug.
        logp: Octanol-water partition coefficient.
        mw_Da: Molecular weight in Daltons.
        plasma_conc_mg_L: Plasma concentration (mg/L).
        iodine_content_pct: Weight percent iodine in drug (e.g., amiodarone ~37%).
        fu_plasma: Fraction unbound in plasma.

    Returns:
        ThyroidDistributionResult with distribution parameters.

    Raises:
        ValueError: If input parameters are invalid.
    """
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if plasma_conc_mg_L <= 0:
        raise ValueError(f"plasma_conc_mg_L must be > 0, got {plasma_conc_mg_L}")
    if iodine_content_pct < 0:
        raise ValueError(f"iodine_content_pct must be >= 0, got {iodine_content_pct}")

    # NIS substrate probability
    nis_prob = 0.05
    if iodine_content_pct > 10.0:
        nis_prob += 0.5
    if mw_Da < 300 and logp < 0:
        nis_prob += 0.2
    nis_prob = max(0.0, min(1.0, nis_prob))

    # Thyroid:plasma ratio
    thyroid_plasma_ratio = 0.5  # passive baseline
    if iodine_content_pct > 10.0:
        thyroid_plasma_ratio *= 20.0  # NIS concentrating effect
    elif nis_prob > 0.3:
        thyroid_plasma_ratio *= 5.0
    thyroid_plasma_ratio = max(0.1, min(1000.0, thyroid_plasma_ratio))

    # Predicted thyroid concentration (ug/g)
    # density of thyroid tissue ~1.05 g/mL; convert mg/L to ug/mL first
    # thyroid_conc_ug_g = ratio * plasma_conc_mg_L * fu_plasma * 1000 / density
    thyroid_conc = thyroid_plasma_ratio * plasma_conc_mg_L * fu_plasma * 1000.0 / 1.05
    thyroid_conc = max(0.0, min(1.0e6, thyroid_conc))

    # Thyroid disruption score (0-100)
    disruption_score = iodine_content_pct * 2.0 + thyroid_plasma_ratio * 0.5 + nis_prob * 20.0
    disruption_score = min(100.0, disruption_score)

    # Toxicity concern
    thyroid_toxicity_concern = disruption_score > 40.0

    # Notes
    if iodine_content_pct > 30.0:
        notes = (
            "High iodine content drug — significant thyroid accumulation. "
            "Monitor thyroid function tests."
        )
    elif thyroid_toxicity_concern:
        notes = "Thyroid disruption potential detected — monitor TSH and T3/T4 levels"
    else:
        notes = "Low thyroid distribution risk"

    return ThyroidDistributionResult(
        drug_name=drug_name,
        logp=logp,
        mw_Da=mw_Da,
        iodine_content_pct=iodine_content_pct,
        thyroid_plasma_ratio=thyroid_plasma_ratio,
        predicted_thyroid_conc_ug_g=thyroid_conc,
        plasma_conc_input_mg_L=plasma_conc_mg_L,
        nis_substrate_probability=nis_prob,
        thyroid_disruption_score=disruption_score,
        thyroid_toxicity_concern=thyroid_toxicity_concern,
        notes=notes,
    )


def screen_thyroid_safety(candidates: list) -> list:
    """Screen a list of drug candidates for thyroid safety.

    Args:
        candidates: List of dicts with keys:
            - "name": drug name (required)
            - "logp": logP value (required)
            - "mw_Da": molecular weight, default 400
            - "iodine_content_pct": iodine weight percent, default 0

    Returns:
        List of ThyroidDistributionResult sorted by thyroid_disruption_score descending.
    """
    results = []
    for cand in candidates:
        result = predict_thyroid_distribution(
            drug_name=cand["name"],
            logp=cand["logp"],
            mw_Da=cand.get("mw_Da", 400.0),
            iodine_content_pct=cand.get("iodine_content_pct", 0.0),
        )
        results.append(result)
    results.sort(key=lambda r: r.thyroid_disruption_score, reverse=True)
    return results
