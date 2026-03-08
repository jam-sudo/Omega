"""Phase 907 — Drug Aqueous Humor Distribution.

Predict drug distribution into aqueous humor for glaucoma treatments
and anterior segment ocular pharmacology.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AqueousHumorDistributionResult:
    """Result of aqueous humor distribution prediction."""

    drug_name: str
    logp: float
    mw_Da: float
    ah_plasma_ratio: float
    predicted_ah_conc_ug_mL: float
    plasma_conc_input_mg_L: float
    ah_turnover_h: float
    bab_permeability_class: str
    anterior_penetration_score: float
    iop_lowering_potential: bool
    notes: str


def predict_aqueous_humor_distribution(
    drug_name: str,
    logp: float = 2.0,
    mw_Da: float = 300.0,
    plasma_conc_mg_L: float = 1.0,
    fu_plasma: float = 0.5,
    has_bab_transport: bool = False,
) -> AqueousHumorDistributionResult:
    """Predict drug distribution into aqueous humor.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logp:
        Lipophilicity (log P octanol/water).
    mw_Da:
        Molecular weight in Daltons.
    plasma_conc_mg_L:
        Plasma drug concentration (mg/L).
    fu_plasma:
        Fraction unbound in plasma (0–1).
    has_bab_transport:
        True if drug is an active transport substrate across the BAB.

    Returns
    -------
    AqueousHumorDistributionResult
    """
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if plasma_conc_mg_L <= 0:
        raise ValueError(f"plasma_conc_mg_L must be > 0, got {plasma_conc_mg_L}")

    # Blood-aqueous barrier permeability score
    bab_score = logp - mw_Da / 500.0

    if bab_score > 1.0:
        bab_permeability_class = "high"
        ah_plasma_ratio_base = 0.5
    elif bab_score > -0.5:
        bab_permeability_class = "moderate"
        ah_plasma_ratio_base = 0.15
    else:
        bab_permeability_class = "low"
        ah_plasma_ratio_base = 0.02

    ah_plasma_ratio = ah_plasma_ratio_base
    if has_bab_transport:
        ah_plasma_ratio *= 3.0

    # Clamp to [0.001, 5.0]
    ah_plasma_ratio = max(0.001, min(5.0, ah_plasma_ratio))

    # Predicted AH concentration: ratio × plasma × fu × 1000 (mg/L → µg/mL)
    predicted_ah_conc_ug_mL = ah_plasma_ratio * plasma_conc_mg_L * fu_plasma * 1000.0

    # AH turnover half-life (~0.46/h → t½ ≈ 1.5 h)
    ah_turnover_h = 1.5

    # Anterior penetration score (0–100)
    transport_bonus = 20.0 if has_bab_transport else 0.0
    anterior_penetration_score = min(100.0, ah_plasma_ratio * 100.0 + transport_bonus)

    # IOP-lowering potential
    iop_lowering_potential = anterior_penetration_score > 50.0

    # Notes
    if iop_lowering_potential:
        notes = "Good anterior segment penetration — suitable for glaucoma treatment"
    elif bab_permeability_class == "high":
        notes = "High BAB permeability — achieves therapeutic AH concentrations"
    else:
        notes = "Limited aqueous humor distribution — topical route may be preferred"

    return AqueousHumorDistributionResult(
        drug_name=drug_name,
        logp=logp,
        mw_Da=mw_Da,
        ah_plasma_ratio=ah_plasma_ratio,
        predicted_ah_conc_ug_mL=predicted_ah_conc_ug_mL,
        plasma_conc_input_mg_L=plasma_conc_mg_L,
        ah_turnover_h=ah_turnover_h,
        bab_permeability_class=bab_permeability_class,
        anterior_penetration_score=anterior_penetration_score,
        iop_lowering_potential=iop_lowering_potential,
        notes=notes,
    )


def screen_glaucoma_candidates(
    candidates: list[dict],
) -> list[AqueousHumorDistributionResult]:
    """Screen multiple drug candidates for glaucoma suitability.

    Parameters
    ----------
    candidates:
        List of dicts with keys: "name" (str), "logp" (float),
        optional "mw_Da" (float, default 300), optional "has_bab_transport" (bool).

    Returns
    -------
    List of results sorted by anterior_penetration_score descending.
    """
    results = []
    for cand in candidates:
        name = cand["name"]
        logp = float(cand.get("logp", 2.0))
        mw_Da = float(cand.get("mw_Da", 300.0))
        has_bab_transport = bool(cand.get("has_bab_transport", False))
        result = predict_aqueous_humor_distribution(
            drug_name=name,
            logp=logp,
            mw_Da=mw_Da,
            has_bab_transport=has_bab_transport,
        )
        results.append(result)

    results.sort(key=lambda r: r.anterior_penetration_score, reverse=True)
    return results
