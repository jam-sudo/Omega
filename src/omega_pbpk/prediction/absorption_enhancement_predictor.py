"""
Phase 233: Absorption Enhancement Predictor

Predicts how absorption enhancers improve drug permeability and
oral bioavailability.  Pure-Python, stdlib only (math, dataclasses).
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Lookup table for enhancer types
# ---------------------------------------------------------------------------

_ENHANCER_TABLE = {
    "surfactant": {"Peff_multiplier": 3.0, "f_increase": 0.15},
    "lipid": {"Peff_multiplier": 2.0, "f_increase": 0.10},
    "cyclodextrin": {"Peff_multiplier": 1.5, "f_increase": 0.08},
    "chitosan": {"Peff_multiplier": 2.5, "f_increase": 0.12},
    "none": {"Peff_multiplier": 1.0, "f_increase": 0.00},
}

VALID_ENHANCER_TYPES = list(_ENHANCER_TABLE.keys())


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbsorptionEnhancerResult:
    drug_name: str
    enhancer_type: str
    baseline_peff_cm_s: float
    enhanced_peff_cm_s: float
    baseline_f_oral: float
    enhanced_f_oral: float
    enhancement_ratio: float
    bioavailability_improvement_pct: float
    notes: str


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------


def predict_absorption_enhancement(
    drug_name: str,
    baseline_peff_cm_s: float,
    baseline_f_oral: float,
    enhancer_type: str,
    dose_mg: float = 10.0,
) -> AbsorptionEnhancerResult:
    """
    Predict the effect of an absorption enhancer on drug permeability and
    oral bioavailability.

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    baseline_peff_cm_s : float
        Baseline effective intestinal permeability (cm/s). Must be > 0.
    baseline_f_oral : float
        Baseline oral bioavailability fraction (0, 1]. Must be in (0, 1].
    enhancer_type : str
        Type of absorption enhancer. One of: surfactant, lipid, cyclodextrin,
        chitosan, none.
    dose_mg : float
        Dose in mg (informational; default 10.0).

    Returns
    -------
    AbsorptionEnhancerResult
    """
    # --- Validation ---
    if baseline_peff_cm_s <= 0:
        raise ValueError(f"baseline_peff_cm_s must be > 0, got {baseline_peff_cm_s}")
    if baseline_f_oral <= 0 or baseline_f_oral > 1.0:
        raise ValueError(f"baseline_f_oral must be in (0, 1], got {baseline_f_oral}")
    if enhancer_type not in _ENHANCER_TABLE:
        raise ValueError(
            f"enhancer_type must be one of {VALID_ENHANCER_TYPES}, got '{enhancer_type}'"
        )

    params = _ENHANCER_TABLE[enhancer_type]
    peff_mult = params["Peff_multiplier"]
    f_inc = params["f_increase"]

    # --- Calculations ---
    enhanced_peff = baseline_peff_cm_s * peff_mult
    enhanced_f = min(1.0, baseline_f_oral + f_inc)
    enhancement_ratio = enhanced_peff / baseline_peff_cm_s
    bav_improvement_pct = (enhanced_f - baseline_f_oral) / baseline_f_oral * 100.0

    # --- Notes ---
    if enhancer_type == "none":
        notes = "No enhancer applied; permeability and bioavailability unchanged."
    else:
        notes = (
            f"{enhancer_type.capitalize()} enhancer increases Peff by "
            f"{peff_mult:.1f}x and adds {f_inc * 100:.0f}% absolute F increase; "
            f"enhanced_f_oral capped at 1.0 if applicable."
        )

    return AbsorptionEnhancerResult(
        drug_name=drug_name,
        enhancer_type=enhancer_type,
        baseline_peff_cm_s=baseline_peff_cm_s,
        enhanced_peff_cm_s=enhanced_peff,
        baseline_f_oral=baseline_f_oral,
        enhanced_f_oral=enhanced_f,
        enhancement_ratio=enhancement_ratio,
        bioavailability_improvement_pct=bav_improvement_pct,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Screening function
# ---------------------------------------------------------------------------


def screen_enhancers(
    drug_name: str,
    baseline_peff_cm_s: float,
    baseline_f_oral: float,
) -> list[AbsorptionEnhancerResult]:
    """
    Screen all enhancer types for a given drug and return results sorted by
    enhanced_f_oral descending.

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    baseline_peff_cm_s : float
        Baseline effective permeability (cm/s). Must be > 0.
    baseline_f_oral : float
        Baseline oral bioavailability fraction (0, 1].

    Returns
    -------
    list[AbsorptionEnhancerResult]
        One result per enhancer type, sorted by enhanced_f_oral descending.
    """
    results = []
    for enh_type in _ENHANCER_TABLE:
        r = predict_absorption_enhancement(
            drug_name=drug_name,
            baseline_peff_cm_s=baseline_peff_cm_s,
            baseline_f_oral=baseline_f_oral,
            enhancer_type=enh_type,
        )
        results.append(r)

    results.sort(key=lambda r: r.enhanced_f_oral, reverse=True)
    return results
