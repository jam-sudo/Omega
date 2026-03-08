"""Dermal Absorption Risk Assessment.

Assesses the risk of dermal absorption for occupational/consumer safety
using the Potts-Guy permeability model.

Reference: Potts & Guy (1992). Predicting skin permeability.
  log10(Kp) = 0.71*logP - 0.0061*MW - 6.3
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass (frozen=True — no list/dict fields)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DermalAbsorptionRiskResult:
    """Result of dermal absorption risk assessment."""

    substance_name: str
    logP: float
    mw_Da: float
    kp_cm_h: float
    flux_mg_cm2_h: float
    skin_area_cm2: float
    exposure_duration_h: float
    conc_mg_cm3: float
    dermal_absorption_fraction: float
    total_absorbed_mg: float
    body_burden_mg_L: float
    risk_class: str
    ppe_required: bool
    notes: str


# ---------------------------------------------------------------------------
# Risk classification helper
# ---------------------------------------------------------------------------

_RISK_THRESHOLDS = [
    (0.001, "negligible"),
    (0.01, "low"),
    (0.1, "moderate"),
]


def _classify_risk(body_burden_mg_L: float) -> str:
    """Classify risk based on body burden (mg/L)."""
    for threshold, label in _RISK_THRESHOLDS:
        if body_burden_mg_L < threshold:
            return label
    return "high"


# ---------------------------------------------------------------------------
# Assessment function
# ---------------------------------------------------------------------------


def assess_dermal_absorption_risk(
    substance_name: str,
    logP: float,
    mw_Da: float,
    conc_mg_cm3: float,
    skin_area_cm2: float = 100.0,
    exposure_duration_h: float = 8.0,
    vd_body_L: float = 42.0,
) -> DermalAbsorptionRiskResult:
    """Assess dermal absorption risk using the Potts-Guy model.

    Parameters
    ----------
    substance_name:
        Name of the substance.
    logP:
        Log octanol-water partition coefficient.
    mw_Da:
        Molecular weight (Da).
    conc_mg_cm3:
        Concentration of substance on skin surface (mg/cm3).
    skin_area_cm2:
        Skin surface area exposed (cm2). Default 100 cm2.
    exposure_duration_h:
        Duration of skin exposure (h). Default 8 h.
    vd_body_L:
        Volume of distribution for body burden estimate (L). Default 42 L.

    Returns
    -------
    DermalAbsorptionRiskResult
    """
    # --- Validation ---
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if conc_mg_cm3 < 0:
        raise ValueError("conc_mg_cm3 must be >= 0")
    if skin_area_cm2 <= 0:
        raise ValueError("skin_area_cm2 must be > 0")
    if exposure_duration_h <= 0:
        raise ValueError("exposure_duration_h must be > 0")
    if vd_body_L <= 0:
        raise ValueError("vd_body_L must be > 0")

    # --- Potts-Guy permeability ---
    log10_kp = 0.71 * logP - 0.0061 * mw_Da - 6.3
    kp_cm_h = 10.0**log10_kp  # cm/h

    # --- Flux ---
    flux_mg_cm2_h = kp_cm_h * conc_mg_cm3  # mg/cm2/h

    # --- Dermal absorption fraction ---
    # Clamped to [0.01, 1.0]
    dermal_absorption_fraction = min(1.0, max(0.01, kp_cm_h * 10.0))

    # --- Total absorbed ---
    # systemic_dose_mg_h * duration * fraction
    systemic_dose_rate = flux_mg_cm2_h * skin_area_cm2  # mg/h
    total_absorbed_mg = systemic_dose_rate * exposure_duration_h * dermal_absorption_fraction

    # --- Body burden ---
    body_burden_mg_L = total_absorbed_mg / vd_body_L

    # --- Risk classification ---
    risk_class = _classify_risk(body_burden_mg_L)
    ppe_required = risk_class in {"moderate", "high"}

    notes = (
        f"log10(Kp)={log10_kp:.3f}; "
        f"Kp={kp_cm_h:.2e} cm/h; "
        f"flux={flux_mg_cm2_h:.2e} mg/cm2/h; "
        f"absorbed_fraction={dermal_absorption_fraction:.3f}; "
        f"risk={risk_class}"
    )

    return DermalAbsorptionRiskResult(
        substance_name=substance_name,
        logP=logP,
        mw_Da=mw_Da,
        kp_cm_h=kp_cm_h,
        flux_mg_cm2_h=flux_mg_cm2_h,
        skin_area_cm2=skin_area_cm2,
        exposure_duration_h=exposure_duration_h,
        conc_mg_cm3=conc_mg_cm3,
        dermal_absorption_fraction=dermal_absorption_fraction,
        total_absorbed_mg=total_absorbed_mg,
        body_burden_mg_L=body_burden_mg_L,
        risk_class=risk_class,
        ppe_required=ppe_required,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Comparison function
# ---------------------------------------------------------------------------


def compare_exposure_scenarios(
    substance_name: str,
    logP: float,
    mw_Da: float,
    conc_mg_cm3: float,
    areas_cm2: list,
    exposure_duration_h: float = 8.0,
    vd_body_L: float = 42.0,
) -> list:
    """Compare body burden across different skin exposure areas.

    Returns list of DermalAbsorptionRiskResult sorted by body_burden_mg_L descending.

    Parameters
    ----------
    substance_name:
        Name of the substance.
    logP:
        Log octanol-water partition coefficient.
    mw_Da:
        Molecular weight (Da).
    conc_mg_cm3:
        Concentration on skin surface (mg/cm3).
    areas_cm2:
        List of skin exposure areas (cm2) to compare.
    exposure_duration_h:
        Duration of exposure (h). Default 8 h.
    vd_body_L:
        Body volume of distribution (L). Default 42 L.
    """
    results = []
    for area in areas_cm2:
        res = assess_dermal_absorption_risk(
            substance_name=substance_name,
            logP=logP,
            mw_Da=mw_Da,
            conc_mg_cm3=conc_mg_cm3,
            skin_area_cm2=area,
            exposure_duration_h=exposure_duration_h,
            vd_body_L=vd_body_L,
        )
        results.append(res)

    results.sort(key=lambda r: r.body_burden_mg_L, reverse=True)
    return results
