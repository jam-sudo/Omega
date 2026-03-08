"""
Phase 434 — Clearance route predictor.

Predicts hepatic vs renal vs biliary vs other clearance pathways
from molecular properties using rule-based, literature-derived heuristics.
Pure Python, stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClearanceRouteResult:
    """Predicted clearance pathway fractions and clinical flags."""

    drug_name: str
    mw_Da: float
    logp: float
    pka: float
    fu_plasma: float
    hepatic_cl_fraction: float
    renal_cl_fraction: float
    biliary_cl_fraction: float
    other_cl_fraction: float
    predicted_total_cl_L_per_h: float
    predicted_hepatic_cl_L_per_h: float
    predicted_renal_cl_L_per_h: float
    primary_route: str
    renal_sensitivity: str
    hepatic_cyp_dependency: str
    dose_adjustment_ckd: str
    dose_adjustment_hepatic: str
    notes: str


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def predict_clearance_route(
    drug_name: str,
    mw_Da: float,
    logp: float,
    pka: float,
    ionization_type: str,
    fu_plasma: float,
) -> ClearanceRouteResult:
    """
    Predict clearance route fractions from molecular properties.

    Parameters
    ----------
    drug_name : str
    mw_Da : float  -- molecular weight (Da)
    logp : float   -- octanol-water partition coefficient
    pka : float    -- ionization constant
    ionization_type : str  -- "acid", "base", or "neutral"
    fu_plasma : float  -- fraction unbound in plasma (0-1]

    Returns
    -------
    ClearanceRouteResult
    """
    # --- Validation ---
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if not (0 < fu_plasma <= 1):
        raise ValueError("fu_plasma must be in (0, 1]")
    valid_ionization = {"acid", "base", "neutral"}
    if ionization_type not in valid_ionization:
        raise ValueError(
            f"ionization_type must be one of {valid_ionization}, got '{ionization_type}'"
        )

    # --- Determine ionization state at physiological pH 7.4 ---
    ph_phys = 7.4
    if ionization_type == "acid":
        # fraction ionized at pH 7.4
        fraction_ionized = 1.0 / (1.0 + 10.0 ** (pka - ph_phys))
    elif ionization_type == "base":
        fraction_ionized = 1.0 / (1.0 + 10.0 ** (ph_phys - pka))
    else:
        fraction_ionized = 0.0

    is_ionized = fraction_ionized > 0.5

    # --- Renal fraction ---
    if mw_Da > 500:
        renal_fraction = 0.05
    elif logp < 1:
        if not is_ionized:
            renal_fraction = 0.6
        else:
            renal_fraction = 0.4
    elif logp > 2:
        renal_fraction = 0.1
    else:
        # moderate logP (1-2)
        renal_fraction = 0.25

    renal_fraction = _clamp(renal_fraction, 0.05, 0.9)

    # --- Hepatic fraction ---
    if logp > 1:
        hepatic_fraction = 0.7 - (renal_fraction - 0.1)
    else:
        hepatic_fraction = max(0.05, 0.5 - renal_fraction * 0.5)

    hepatic_fraction = _clamp(hepatic_fraction, 0.05, 0.9)

    # Ensure hepatic + renal <= 0.95
    if hepatic_fraction + renal_fraction > 0.95:
        # scale proportionally
        total = hepatic_fraction + renal_fraction
        hepatic_fraction = hepatic_fraction * 0.95 / total
        renal_fraction = renal_fraction * 0.95 / total

    # --- Biliary fraction ---
    if mw_Da > 400 and logp > 2:
        biliary_fraction = 0.1
    else:
        biliary_fraction = 0.02

    # Ensure biliary doesn't exceed remaining budget
    remaining = 1.0 - hepatic_fraction - renal_fraction
    biliary_fraction = min(biliary_fraction, remaining)
    biliary_fraction = max(0.0, biliary_fraction)

    # --- Other fraction ---
    other_cl_fraction = max(0.0, 1.0 - hepatic_fraction - renal_fraction - biliary_fraction)

    # --- Total CL ---
    cl_total = 10.0 * fu_plasma * (1.0 + logp * 0.2)
    cl_total = _clamp(cl_total, 0.5, 150.0)

    predicted_hepatic_cl = cl_total * hepatic_fraction
    predicted_renal_cl = cl_total * renal_fraction

    # --- Primary route ---
    fractions = {
        "hepatic": hepatic_fraction,
        "renal": renal_fraction,
        "biliary": biliary_fraction,
        "other": other_cl_fraction,
    }
    primary_route = max(fractions, key=lambda k: fractions[k])

    # --- Clinical flags ---
    renal_sensitivity = "high" if renal_fraction > 0.5 else "low"
    hepatic_cyp_dependency = "high" if (logp > 1 and hepatic_fraction > 0.5) else "low"
    dose_adjustment_ckd = "required" if renal_fraction > 0.3 else "not_required"
    dose_adjustment_hepatic = "required" if hepatic_fraction > 0.5 else "not_required"

    notes = (
        f"Primary clearance: {primary_route} ({fractions[primary_route]:.2f}). "
        f"CKD adjustment: {dose_adjustment_ckd}. "
        f"Hepatic adjustment: {dose_adjustment_hepatic}. "
        f"Renal sensitivity: {renal_sensitivity}. "
        f"CYP dependency: {hepatic_cyp_dependency}."
    )

    return ClearanceRouteResult(
        drug_name=drug_name,
        mw_Da=mw_Da,
        logp=logp,
        pka=pka,
        fu_plasma=fu_plasma,
        hepatic_cl_fraction=hepatic_fraction,
        renal_cl_fraction=renal_fraction,
        biliary_cl_fraction=biliary_fraction,
        other_cl_fraction=other_cl_fraction,
        predicted_total_cl_L_per_h=cl_total,
        predicted_hepatic_cl_L_per_h=predicted_hepatic_cl,
        predicted_renal_cl_L_per_h=predicted_renal_cl,
        primary_route=primary_route,
        renal_sensitivity=renal_sensitivity,
        hepatic_cyp_dependency=hepatic_cyp_dependency,
        dose_adjustment_ckd=dose_adjustment_ckd,
        dose_adjustment_hepatic=dose_adjustment_hepatic,
        notes=notes,
    )


def screen_clearance_routes(
    compounds_list: list[dict],
) -> list[ClearanceRouteResult]:
    """
    Predict clearance routes for a list of compounds.

    Parameters
    ----------
    compounds_list : list of dicts, each with keys:
        drug_name, mw_Da, logp, pka, ionization_type, fu_plasma

    Returns
    -------
    List[ClearanceRouteResult] sorted by predicted_total_cl_L_per_h descending
    """
    results = []
    for compound in compounds_list:
        r = predict_clearance_route(
            drug_name=compound["drug_name"],
            mw_Da=compound["mw_Da"],
            logp=compound["logp"],
            pka=compound["pka"],
            ionization_type=compound["ionization_type"],
            fu_plasma=compound["fu_plasma"],
        )
        results.append(r)

    results.sort(key=lambda x: x.predicted_total_cl_L_per_h, reverse=True)
    return results
