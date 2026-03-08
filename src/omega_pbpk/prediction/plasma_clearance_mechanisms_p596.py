"""Plasma clearance mechanism decomposition.

Predicts and decomposes total plasma clearance into hepatic, renal, and
other components using physiological parameters. Supports dose adjustment
recommendations for renal impairment.

References
----------
- Rowland M & Tozer TN, Clinical Pharmacokinetics and Pharmacodynamics. 2011.
- Wilkinson GR & Shand DG, Clin Pharmacol Ther. 1975;18:377-390 (well-stirred model)
- Swan SK, Am J Med. 1994;97(Suppl 5A):1S-3S (renal dose adjustment)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Physiological constants
# ---------------------------------------------------------------------------
_GFR_NORMAL_L_PER_H = 7.2  # 120 mL/min = 7.2 L/h
_HEPATIC_BLOOD_FLOW_L_PER_H = 90.0  # ~1500 mL/min = 90 L/h
_GFR_SEVERE_IMPAIRMENT = 15.0  # mL/min, severe renal impairment
_GFR_NORMAL_ML_MIN = 120.0  # mL/min


# ---------------------------------------------------------------------------
# Result dataclass (frozen — no list/dict fields)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlasmaClearanceMechanismsResult:
    """Decomposed plasma clearance result."""

    drug_name: str
    cl_total_L_per_h: float
    cl_hepatic_L_per_h: float
    cl_renal_L_per_h: float
    cl_other_L_per_h: float
    f_hepatic: float
    f_renal: float
    f_other: float
    half_life_h: float
    extraction_ratio_hepatic: float
    clearance_class: str
    primary_elimination: str
    dose_adjustment_renal_imp: float
    notes: str


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------
def predict_clearance_mechanisms(
    drug_name: str,
    cl_total_L_per_h: float,
    fu_plasma: float,
    logP: float,
    mw_Da: float,
    vd_L: float,
    pka: float = 7.0,
    renal_secretion: bool = False,
) -> PlasmaClearanceMechanismsResult:
    """Predict and decompose total plasma clearance.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    cl_total_L_per_h:
        Total plasma clearance [L/h].
    fu_plasma:
        Fraction unbound in plasma (0 < fu_plasma <= 1).
    logP:
        Octanol-water partition coefficient.
    mw_Da:
        Molecular weight in Daltons.
    vd_L:
        Volume of distribution [L].
    pka:
        Acid dissociation constant. Default 7.0.
    renal_secretion:
        If True, active tubular secretion doubles renal clearance.

    Returns
    -------
    PlasmaClearanceMechanismsResult
    """
    # Validation
    if cl_total_L_per_h <= 0:
        raise ValueError(f"cl_total_L_per_h must be > 0, got {cl_total_L_per_h}")
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")

    # Renal clearance estimate
    cl_renal = _GFR_NORMAL_L_PER_H * fu_plasma
    if renal_secretion:
        cl_renal *= 2.0
    # Cap at total clearance
    cl_renal = min(cl_renal, cl_total_L_per_h)

    # Hepatic clearance (remainder after renal)
    cl_hepatic = cl_total_L_per_h - cl_renal
    if cl_hepatic < 0:
        cl_hepatic = 0.0
        cl_renal = cl_total_L_per_h

    # Other clearance (e.g., biliary, lymphatic, pulmonary)
    cl_other = max(0.0, cl_total_L_per_h - cl_hepatic - cl_renal)

    # Fractional contributions
    f_hepatic = cl_hepatic / cl_total_L_per_h
    f_renal = cl_renal / cl_total_L_per_h
    f_other = 1.0 - f_hepatic - f_renal
    f_other = max(0.0, f_other)  # guard against floating point

    # Half-life
    half_life = vd_L * math.log(2.0) / cl_total_L_per_h

    # Hepatic extraction ratio
    extraction_ratio = cl_hepatic / _HEPATIC_BLOOD_FLOW_L_PER_H
    extraction_ratio = max(0.0, min(1.0, extraction_ratio))

    # Clearance class based on hepatic extraction ratio
    if extraction_ratio < 0.3:
        clearance_class = "low"
    elif extraction_ratio < 0.7:
        clearance_class = "intermediate"
    else:
        clearance_class = "high"

    # Primary elimination pathway
    if f_hepatic > 0.7:
        primary_elimination = "hepatic"
    elif f_renal > 0.7:
        primary_elimination = "renal"
    elif f_hepatic > 0.3 and f_renal > 0.3:
        primary_elimination = "mixed"
    else:
        primary_elimination = "other"

    # Dose adjustment for severe renal impairment (GFR = 15 mL/min)
    if f_renal < 0.3:
        dose_adjustment = 1.0  # no significant renal contribution
    else:
        # Proportional reduction in clearance due to GFR decline
        gfr_fraction_remaining = _GFR_SEVERE_IMPAIRMENT / _GFR_NORMAL_ML_MIN
        dose_adjustment = 1.0 - f_renal * (1.0 - gfr_fraction_remaining)

    notes = (
        f"GFR-based CLr={cl_renal:.3f} L/h; "
        f"ER_hepatic={extraction_ratio:.3f}; "
        f"t1/2={half_life:.2f} h"
    )

    return PlasmaClearanceMechanismsResult(
        drug_name=drug_name,
        cl_total_L_per_h=cl_total_L_per_h,
        cl_hepatic_L_per_h=cl_hepatic,
        cl_renal_L_per_h=cl_renal,
        cl_other_L_per_h=cl_other,
        f_hepatic=f_hepatic,
        f_renal=f_renal,
        f_other=f_other,
        half_life_h=half_life,
        extraction_ratio_hepatic=extraction_ratio,
        clearance_class=clearance_class,
        primary_elimination=primary_elimination,
        dose_adjustment_renal_imp=dose_adjustment,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Comparison function
# ---------------------------------------------------------------------------
def compare_clearance_routes(compounds: list) -> list:
    """Compare clearance mechanisms across multiple compounds.

    Parameters
    ----------
    compounds:
        List of dicts, each with keys matching ``predict_clearance_mechanisms``
        parameters: ``drug_name``, ``cl_total_L_per_h``, ``fu_plasma``,
        ``logP``, ``mw_Da``, ``vd_L``, and optionally ``pka``,
        ``renal_secretion``.

    Returns
    -------
    list
        List of ``PlasmaClearanceMechanismsResult`` sorted by ``f_hepatic``
        descending.
    """
    results = []
    for cpd in compounds:
        result = predict_clearance_mechanisms(
            drug_name=cpd["drug_name"],
            cl_total_L_per_h=cpd["cl_total_L_per_h"],
            fu_plasma=cpd["fu_plasma"],
            logP=cpd["logP"],
            mw_Da=cpd["mw_Da"],
            vd_L=cpd["vd_L"],
            pka=cpd.get("pka", 7.0),
            renal_secretion=cpd.get("renal_secretion", False),
        )
        results.append(result)

    # Sort by f_hepatic descending
    results.sort(key=lambda r: r.f_hepatic, reverse=True)
    return results
