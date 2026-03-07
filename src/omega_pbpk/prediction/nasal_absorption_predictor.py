"""Phase 235 — Nasal absorption predictor.

Predicts nasal drug absorption based on physicochemical properties
(MW, logP, pKa) and nasal region physiology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Region lookup table
# ---------------------------------------------------------------------------

_NASAL_REGIONS: dict[str, dict[str, float]] = {
    "anterior": {"peff_cm_s": 1e-5, "surface_area_cm2": 20.0},
    "posterior": {"peff_cm_s": 5e-5, "surface_area_cm2": 60.0},
    "olfactory": {"peff_cm_s": 2e-4, "surface_area_cm2": 5.0},
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NasalAbsorptionResult:
    """Result of nasal absorption prediction.

    Attributes
    ----------
    drug_name : Drug identifier string.
    mw : Molecular weight (g/mol).
    logp : Octanol-water partition coefficient.
    pka : Acid dissociation constant.
    dose_mg : Dose administered nasally (mg).
    nasal_region : Region of nasal cavity ("anterior", "posterior", "olfactory").
    peff_cm_s : Effective permeability after MW correction (cm/s).
    fraction_absorbed : Fraction of dose absorbed (0-1).
    c_plasma_peak_mg_L : Estimated peak plasma concentration (mg/L).
    t_peak_min : Estimated time to peak plasma concentration (min).
    bioavailability_pct : Nasal bioavailability as percentage.
    absorption_class : "rapid" (<10 min), "moderate" (10-30 min), or "slow" (>30 min).
    notes : Summary notes describing key modifiers.
    """

    drug_name: str
    mw: float
    logp: float
    pka: float
    dose_mg: float
    nasal_region: str
    peff_cm_s: float
    fraction_absorbed: float
    c_plasma_peak_mg_L: float
    t_peak_min: float
    bioavailability_pct: float
    absorption_class: str
    notes: str


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------


def predict_nasal_absorption(
    drug_name: str,
    mw: float,
    logp: float,
    pka: float,
    dose_mg: float,
    nasal_region: str = "posterior",
    vd_L: float = 50.0,
    cl_L_per_h: float = 5.0,
) -> NasalAbsorptionResult:
    """Predict nasal absorption for a drug based on physicochemical properties.

    Parameters
    ----------
    drug_name : Drug identifier string.
    mw : Molecular weight (g/mol). Must be > 0.
    logp : Octanol-water partition coefficient.
    pka : Acid dissociation constant.
    dose_mg : Nasal dose (mg). Must be > 0.
    nasal_region : Nasal region ("anterior", "posterior", "olfactory").
    vd_L : Volume of distribution (L). Must be > 0.
    cl_L_per_h : Systemic clearance (L/h). Must be > 0.

    Returns
    -------
    NasalAbsorptionResult

    Raises
    ------
    ValueError
        If mw <= 0, dose_mg <= 0, nasal_region unknown, vd_L <= 0, or cl_L_per_h <= 0.
    """
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if nasal_region not in _NASAL_REGIONS:
        raise ValueError(
            f"nasal_region must be one of {list(_NASAL_REGIONS.keys())}, got '{nasal_region}'"
        )
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")

    region_props = _NASAL_REGIONS[nasal_region]
    peff_base = region_props["peff_cm_s"]
    surface_area_cm2 = region_props["surface_area_cm2"]

    # MW penalty: exponential decay above 300 g/mol
    mw_excess = max(0.0, mw - 300.0)
    peff_mod = peff_base * math.exp(-0.003 * mw_excess)

    # Fraction absorbed (simplified model)
    fraction_absorbed = min(1.0, peff_mod * surface_area_cm2 * 3600.0 / dose_mg)

    # Peak plasma concentration (mg/L)
    c_plasma_peak_mg_L = fraction_absorbed * dose_mg / vd_L

    # Analytical 1-compartment t_peak estimate
    # ka in /min: Peff_mod (cm/s) * surface_area (cm2) * 60 s/min / (Vd (L) * 1000 cm3/L)
    ke_per_h = cl_L_per_h / vd_L  # /h
    ka_per_min = peff_mod * surface_area_cm2 * 60.0 / (vd_L * 1000.0)
    ke_per_min = ke_per_h / 60.0

    if ka_per_min <= ke_per_min:
        t_peak_min = 5.0
    else:
        t_peak_min = math.log(ka_per_min / ke_per_min) / (ka_per_min - ke_per_min) * 60.0

    # Bioavailability percent
    bioavailability_pct = fraction_absorbed * 100.0

    # Absorption class
    if t_peak_min < 10.0:
        absorption_class = "rapid"
    elif t_peak_min <= 30.0:
        absorption_class = "moderate"
    else:
        absorption_class = "slow"

    # Notes
    notes_parts = [f"region={nasal_region}"]
    if mw_excess > 0:
        notes_parts.append(
            f"MW penalty: exp(-0.003*{mw_excess:.0f})={math.exp(-0.003 * mw_excess):.3f}"
        )
    notes_parts.append(f"Peff_mod={peff_mod:.2e}cm/s")
    notes = "; ".join(notes_parts)

    return NasalAbsorptionResult(
        drug_name=drug_name,
        mw=mw,
        logp=logp,
        pka=pka,
        dose_mg=dose_mg,
        nasal_region=nasal_region,
        peff_cm_s=peff_mod,
        fraction_absorbed=fraction_absorbed,
        c_plasma_peak_mg_L=c_plasma_peak_mg_L,
        t_peak_min=t_peak_min,
        bioavailability_pct=bioavailability_pct,
        absorption_class=absorption_class,
        notes=notes,
    )


def compare_nasal_regions(
    drug_name: str,
    mw: float,
    logp: float,
    pka: float,
    dose_mg: float,
    vd_L: float = 50.0,
    cl_L_per_h: float = 5.0,
) -> list:
    """Compare nasal absorption across all three nasal regions.

    Parameters
    ----------
    drug_name : Drug identifier string.
    mw : Molecular weight (g/mol).
    logp : Octanol-water partition coefficient.
    pka : Acid dissociation constant.
    dose_mg : Nasal dose (mg).
    vd_L : Volume of distribution (L).
    cl_L_per_h : Systemic clearance (L/h).

    Returns
    -------
    list[NasalAbsorptionResult]
        Results for all three regions, sorted by fraction_absorbed descending.
    """
    results = []
    for region in _NASAL_REGIONS:
        result = predict_nasal_absorption(
            drug_name=drug_name,
            mw=mw,
            logp=logp,
            pka=pka,
            dose_mg=dose_mg,
            nasal_region=region,
            vd_L=vd_L,
            cl_L_per_h=cl_L_per_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.fraction_absorbed, reverse=True)
    return results


__all__ = [
    "NasalAbsorptionResult",
    "predict_nasal_absorption",
    "compare_nasal_regions",
]
