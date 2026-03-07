"""Sebum and skin appendage drug distribution prediction.

Predicts drug distribution in sebum and skin appendages (hair follicles,
sebaceous glands). Relevant for acne treatments, alopecia drugs, antifungals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SebumSkinDistributionResult:
    """Result of sebum and skin appendage distribution prediction."""

    drug_name: str
    logp: float
    mw_Da: float
    sebum_partition_coefficient: float
    follicular_penetration_fraction: float
    sebum_reservoir_t_half_h: float
    topical_penetration_enhancement: float
    acne_target_conc_achievable: bool
    predicted_sebum_conc_ug_g: float
    plasma_conc_input_mg_L: float
    notes: str


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _pow10(x: float) -> float:
    """10**x using stdlib math."""
    import math

    return math.pow(10.0, x)


def predict_sebum_distribution(
    drug_name: str,
    logp: float = 3.0,
    mw_Da: float = 350.0,
    plasma_conc_mg_L: float = 0.1,
) -> SebumSkinDistributionResult:
    """Predict drug distribution into sebum and skin appendages.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logp:
        Octanol-water partition coefficient (log scale).
    mw_Da:
        Molecular weight in Daltons.
    plasma_conc_mg_L:
        Plasma drug concentration in mg/L.

    Returns
    -------
    SebumSkinDistributionResult
    """
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if plasma_conc_mg_L <= 0:
        raise ValueError("plasma_conc_mg_L must be > 0")

    # Sebum partition coefficient: sebum:plasma
    raw_pc = _pow10(logp * 0.5)
    sebum_pc = _clamp(max(1.0, raw_pc), 1.0, 1000.0)

    # Follicular penetration fraction
    if mw_Da < 500 and logp > 0:
        raw_fpf = 0.1 + logp * 0.05
        raw_fpf = _clamp(raw_fpf, 0.01, 0.4)
    else:
        if mw_Da < 500 and logp <= 0:
            # Still compute base then scale
            raw_fpf = _clamp(0.1 + logp * 0.05, 0.01, 0.4)
            raw_fpf = max(0.01, raw_fpf * 0.3)
        else:
            raw_fpf_base = _clamp(0.1 + logp * 0.05, 0.01, 0.4)
            raw_fpf = max(0.01, raw_fpf_base * 0.3)
    follicular_penetration_fraction = raw_fpf

    # Sebum reservoir half-life (h)
    sebum_reservoir_t_half_h = _clamp(sebum_pc * 0.5, 0.5, 72.0)

    # Topical penetration enhancement (fold vs no follicular pathway)
    topical_penetration_enhancement = 1.0 + follicular_penetration_fraction * 3.0

    # Predicted sebum concentration (µg/g sebum)
    raw_sebum_conc = sebum_pc * plasma_conc_mg_L * 1000.0
    predicted_sebum_conc_ug_g = _clamp(raw_sebum_conc, 0.0, 1e6)

    # Acne target achievability
    acne_target_conc_achievable = predicted_sebum_conc_ug_g >= 1.0

    # Notes
    if acne_target_conc_achievable:
        notes = (
            "Therapeutic sebum concentrations achievable — suitable for acne/folliculitis treatment"
        )
    elif logp > 4:
        notes = (
            "High sebum accumulation — prolonged follicular reservoir. Consider once-daily dosing."
        )
    else:
        notes = "Moderate sebaceous penetration"

    # Override notes if logp > 4 but acne target also achievable — acne note takes priority
    # (already handled: if acne_target_conc_achievable check is first)
    # Additional logp > 4 note appended only when acne target NOT achievable
    # but logp > 4: covered above

    return SebumSkinDistributionResult(
        drug_name=drug_name,
        logp=logp,
        mw_Da=mw_Da,
        sebum_partition_coefficient=sebum_pc,
        follicular_penetration_fraction=follicular_penetration_fraction,
        sebum_reservoir_t_half_h=sebum_reservoir_t_half_h,
        topical_penetration_enhancement=topical_penetration_enhancement,
        acne_target_conc_achievable=acne_target_conc_achievable,
        predicted_sebum_conc_ug_g=predicted_sebum_conc_ug_g,
        plasma_conc_input_mg_L=plasma_conc_mg_L,
        notes=notes,
    )


def screen_follicular_drugs(
    candidates: list[dict],
) -> list[SebumSkinDistributionResult]:
    """Screen a list of drug candidates for follicular/sebum distribution.

    Parameters
    ----------
    candidates:
        List of dicts with keys: "name" (str), "logp" (float),
        optional "mw_Da" (float, default 350).

    Returns
    -------
    List of SebumSkinDistributionResult sorted by sebum_partition_coefficient
    descending.
    """
    results = []
    for cand in candidates:
        name = cand["name"]
        logp = float(cand["logp"])
        mw_Da = float(cand.get("mw_Da", 350.0))
        result = predict_sebum_distribution(
            drug_name=name,
            logp=logp,
            mw_Da=mw_Da,
        )
        results.append(result)

    results.sort(key=lambda r: r.sebum_partition_coefficient, reverse=True)
    return results
