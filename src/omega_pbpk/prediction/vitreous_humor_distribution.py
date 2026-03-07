"""Phase 908 — Drug Vitreous Humor Distribution.

Predict drug distribution in vitreous humor for retinal diseases
such as AMD and diabetic retinopathy.
"""

from __future__ import annotations

from dataclasses import dataclass

_V_VITREOUS_ML = 4.0  # human vitreous volume in mL


@dataclass(frozen=True)
class VitreousHumorDistributionResult:
    """Result of vitreous humor distribution prediction."""

    drug_name: str
    logp: float
    mw_Da: float
    route: str
    dose_mg: float
    predicted_vh_cmax_ug_mL: float
    predicted_vh_t_half_days: float
    brb_plasma_to_vh_ratio: float
    monthly_dosing_feasible: bool
    anti_vegf_suitable: bool
    retinal_penetration_fraction: float
    notes: str


def predict_vitreous_distribution(
    drug_name: str,
    dose_mg: float = 1.0,
    logp: float = 2.0,
    mw_Da: float = 300.0,
    route: str = "intravitreal",
    plasma_conc_mg_L: float | None = None,
) -> VitreousHumorDistributionResult:
    """Predict drug distribution in vitreous humor.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in mg (for intravitreal route).
    logp:
        Lipophilicity (log P octanol/water).
    mw_Da:
        Molecular weight in Daltons.
    route:
        Administration route: "intravitreal" or "systemic".
    plasma_conc_mg_L:
        Plasma concentration (mg/L); used for systemic route.

    Returns
    -------
    VitreousHumorDistributionResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if route not in {"intravitreal", "systemic"}:
        raise ValueError(f"route must be 'intravitreal' or 'systemic', got '{route}'")

    # BRB permeability ratio
    if mw_Da > 1000.0:
        brb_ratio = 0.01
    elif mw_Da >= 300.0:
        brb_ratio = 0.05
    else:
        brb_ratio = 0.1

    if route == "intravitreal":
        predicted_vh_cmax_ug_mL = dose_mg * 1000.0 / _V_VITREOUS_ML
        brb_plasma_to_vh_ratio = 0.0
    else:
        plasma = plasma_conc_mg_L if plasma_conc_mg_L is not None else 1.0
        predicted_vh_cmax_ug_mL = plasma * brb_ratio * 1000.0
        brb_plasma_to_vh_ratio = brb_ratio

    # Vitreous half-life
    if mw_Da <= 1000.0:
        # Small molecule: 3.0 + logp * 0.5 days, clamp [0.5, 10.0]
        t_half_days = 3.0 + logp * 0.5
        t_half_days = max(0.5, min(10.0, t_half_days))
        retinal_penetration_fraction = 0.3
    else:
        # Large molecule / mAb: 20.0 + (mw_Da / 150000) * 10, clamp [10, 60]
        t_half_days = 20.0 + (mw_Da / 150000.0) * 10.0
        t_half_days = max(10.0, min(60.0, t_half_days))
        retinal_penetration_fraction = 0.1

    monthly_dosing_feasible = t_half_days > 15.0
    anti_vegf_suitable = mw_Da > 10000.0 and t_half_days > 20.0

    # Notes
    if anti_vegf_suitable:
        notes = (
            "Suitable for intravitreal anti-VEGF therapy — "
            "long vitreal t_half allows monthly/bimonthly dosing"
        )
    elif monthly_dosing_feasible:
        notes = "Monthly intravitreal dosing feasible"
    else:
        notes = "Frequent intravitreal dosing required — consider sustained release implant"

    return VitreousHumorDistributionResult(
        drug_name=drug_name,
        logp=logp,
        mw_Da=mw_Da,
        route=route,
        dose_mg=dose_mg,
        predicted_vh_cmax_ug_mL=predicted_vh_cmax_ug_mL,
        predicted_vh_t_half_days=t_half_days,
        brb_plasma_to_vh_ratio=brb_plasma_to_vh_ratio,
        monthly_dosing_feasible=monthly_dosing_feasible,
        anti_vegf_suitable=anti_vegf_suitable,
        retinal_penetration_fraction=retinal_penetration_fraction,
        notes=notes,
    )


def compare_intravitreal_doses(
    drug_name: str,
    doses_mg: list[float],
    logp: float = 2.0,
    mw_Da: float = 300.0,
) -> list[VitreousHumorDistributionResult]:
    """Compare multiple intravitreal doses for a single drug.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    doses_mg:
        List of doses in mg to compare.
    logp:
        Lipophilicity.
    mw_Da:
        Molecular weight in Daltons.

    Returns
    -------
    List of results sorted by predicted_vh_cmax_ug_mL descending.
    """
    results = []
    for dose in doses_mg:
        result = predict_vitreous_distribution(
            drug_name=drug_name,
            dose_mg=dose,
            logp=logp,
            mw_Da=mw_Da,
            route="intravitreal",
        )
        results.append(result)

    results.sort(key=lambda r: r.predicted_vh_cmax_ug_mL, reverse=True)
    return results
