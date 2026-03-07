"""BCS Dose Number and Dose-Dependent Absorption Prediction.

Implements the FDA BCS dose number (Do) calculation and predicts
dose-dependent absorption behavior. A high Do indicates that the drug
dose is not completely dissolved in GI fluid, leading to
dissolution-limited absorption.

FDA BCS dose number: Do = dose_mg / (V_GI_mL * solubility_mg_mL)
  Do < 1: dose completely dissolved → absorption limited by permeability
  Do >= 1: dose not completely dissolved → dissolution-limited

Permeability threshold: Peff >= 2e-4 cm/s → high permeability (BCS).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HIGH_PERM_THRESHOLD_CM_S: float = 2.0e-4  # cm/s — FDA BCS permeability cutoff
_DEFAULT_GI_VOLUME_ML: float = 250.0  # mL — FDA standard GI fluid volume
_INTESTINAL_RADIUS_CM: float = 1.75  # cm — effective intestinal radius
_RESIDENCE_TIME_H: float = 3.0  # h — small intestinal residence time


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BCSRiskResult:
    """BCS risk assessment result for a given drug/dose combination."""

    drug_name: str
    dose_mg: float
    solubility_mg_mL_pH68: float
    peff_cm_s: float
    logP: float
    mw: float
    dose_number: float
    predicted_fa: float
    bcs_class: str  # "I", "II", "III", or "IV"
    biowaiver_eligible: bool
    dissolution_limited: bool
    permeability_limited: bool
    risk_level: str  # "low", "moderate", "high"
    recommendation: str
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def dose_number(
    dose_mg: float,
    solubility_mg_mL: float,
    v_gi_mL: float = _DEFAULT_GI_VOLUME_ML,
) -> float:
    """Compute FDA BCS dose number.

    Do = dose_mg / (v_gi_mL * solubility_mg_mL)

    Args:
        dose_mg: Administered dose in mg.
        solubility_mg_mL: Drug solubility at pH 6.8 in mg/mL.
        v_gi_mL: GI fluid volume in mL (default 250 mL per FDA BCS guidance).

    Returns:
        Dimensionless dose number Do.

    Raises:
        ValueError: If dose_mg or solubility_mg_mL are not positive.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")

    return dose_mg / (v_gi_mL * solubility_mg_mL)


def predict_fa_from_do(do: float, peff_cm_s: float) -> float:
    """Predict fraction absorbed (fa) from dose number and permeability.

    Uses a cylindrical intestine model for permeability-limited fa:
      ka = 2 * Peff / R_intestine  (s⁻¹)
      fa_perm = 1 - exp(-ka * tau)  where tau = residence time in seconds

    If Do < 1 (sufficient dissolution):
      fa = fa_perm (permeability-limited)

    If Do >= 1 (dissolution-limited):
      fa = fa_perm / Do  (linearly reduced by dose number)

    Result is clamped to [0, 1].

    Args:
        do: Dimensionless dose number.
        peff_cm_s: Effective intestinal permeability (cm/s).

    Returns:
        Predicted fraction absorbed in [0, 1].
    """
    tau_s = _RESIDENCE_TIME_H * 3600.0  # seconds
    ka_per_s = 2.0 * peff_cm_s / _INTESTINAL_RADIUS_CM
    fa_perm = 1.0 - math.exp(-ka_per_s * tau_s)
    fa_perm = min(max(fa_perm, 0.0), 1.0)

    if do < 1.0:
        fa = fa_perm
    else:
        fa = fa_perm / do

    return min(max(fa, 0.0), 1.0)


def _classify_bcs(do: float, peff_cm_s: float) -> str:
    """Classify BCS class from dose number and permeability."""
    high_perm = peff_cm_s >= _HIGH_PERM_THRESHOLD_CM_S
    diss_limited = do >= 1.0

    if not diss_limited and high_perm:
        return "I"
    if diss_limited and high_perm:
        return "II"
    if not diss_limited and not high_perm:
        return "III"
    return "IV"


def _risk_level_from_class(bcs_class: str) -> str:
    """Map BCS class to risk level string."""
    if bcs_class == "I":
        return "low"
    if bcs_class in ("II", "III"):
        return "moderate"
    return "high"  # Class IV


_RECOMMENDATIONS: dict[str, str] = {
    "I": (
        "BCS Class I: high solubility, high permeability. "
        "Biowaiver eligible for immediate-release solid oral dosage forms "
        "(FDA BCS guidance). Good oral bioavailability expected."
    ),
    "II": (
        "BCS Class II: low solubility, high permeability. "
        "Absorption is dissolution-limited — not biowaiver eligible. "
        "Consider formulation strategies such as particle size reduction, "
        "amorphous solid dispersions, or lipid-based formulations."
    ),
    "III": (
        "BCS Class III: high solubility, low permeability. "
        "Absorption is permeability-limited — biowaiver not applicable. "
        "Permeation enhancers or prodrug strategies may improve absorption."
    ),
    "IV": (
        "BCS Class IV: low solubility, low permeability. "
        "Most challenging class — poor oral bioavailability expected. "
        "Significant formulation innovation (e.g., nanocrystals, co-solvents) "
        "and/or prodrug approaches required. Consider IV or alternative routes."
    ),
}


def bcs_risk_assessment(
    drug_name: str,
    dose_mg: float,
    solubility_mg_mL_pH68: float,
    peff_cm_s: float,
    logP: float,
    mw: float,
) -> BCSRiskResult:
    """Perform BCS risk assessment for a drug at a given dose.

    Computes the dose number, classifies the BCS class, predicts fa,
    and determines biowaiver eligibility and risk level.

    Args:
        drug_name: Name of the drug.
        dose_mg: Dose in mg.
        solubility_mg_mL_pH68: Aqueous solubility at pH 6.8 in mg/mL.
        peff_cm_s: Effective intestinal permeability in cm/s.
        logP: Octanol-water partition coefficient (informational).
        mw: Molecular weight in g/mol (informational).

    Returns:
        BCSRiskResult with classification, predicted fa, and guidance.

    Raises:
        ValueError: If dose, solubility, or peff are not positive.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if solubility_mg_mL_pH68 <= 0:
        raise ValueError("solubility_mg_mL_pH68 must be > 0")
    if peff_cm_s <= 0:
        raise ValueError("peff_cm_s must be > 0")

    do = dose_number(dose_mg, solubility_mg_mL_pH68)
    fa = predict_fa_from_do(do, peff_cm_s)
    bcs_class = _classify_bcs(do, peff_cm_s)

    biowaiver_eligible = bcs_class == "I"
    dissolution_limited = do >= 1.0
    permeability_limited = peff_cm_s < _HIGH_PERM_THRESHOLD_CM_S
    risk_level = _risk_level_from_class(bcs_class)
    recommendation = _RECOMMENDATIONS[bcs_class]

    notes: list[str] = []
    notes.append(f"Dose number (Do): {do:.3f} (threshold: 1.0)")
    notes.append(
        f"Peff: {peff_cm_s:.2e} cm/s (high-perm threshold: {_HIGH_PERM_THRESHOLD_CM_S:.1e} cm/s)"
    )
    if logP > 5:
        notes.append(f"logP={logP:.1f} > 5: poor aqueous solubility and potential P-gp substrate.")
    if mw > 500:
        notes.append(f"MW={mw:.0f} g/mol > 500: may have reduced permeability (Lipinski rule).")

    return BCSRiskResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        solubility_mg_mL_pH68=solubility_mg_mL_pH68,
        peff_cm_s=peff_cm_s,
        logP=logP,
        mw=mw,
        dose_number=round(do, 6),
        predicted_fa=round(fa, 6),
        bcs_class=bcs_class,
        biowaiver_eligible=biowaiver_eligible,
        dissolution_limited=dissolution_limited,
        permeability_limited=permeability_limited,
        risk_level=risk_level,
        recommendation=recommendation,
        notes=notes,
    )


def dose_escalation_fa(
    drug_name: str,
    solubility_mg_mL: float,
    peff_cm_s: float,
    doses_mg: list[float],
) -> list[BCSRiskResult]:
    """Predict how fraction absorbed changes with dose (dose-dependent absorption).

    Shows the nonlinear relationship between dose and bioavailability for
    dissolution-limited drugs. At high doses (Do >= 1), fa decreases as
    dose increases.

    Args:
        drug_name: Name of the drug.
        solubility_mg_mL: Drug solubility at pH 6.8 in mg/mL.
        peff_cm_s: Effective intestinal permeability in cm/s.
        doses_mg: List of doses in mg to evaluate.

    Returns:
        List of BCSRiskResult sorted by dose_mg ascending.

    Raises:
        ValueError: If solubility_mg_mL or peff_cm_s are not positive,
            or if doses_mg is empty.
    """
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")
    if peff_cm_s <= 0:
        raise ValueError("peff_cm_s must be > 0")
    if not doses_mg:
        raise ValueError("doses_mg must not be empty")

    results: list[BCSRiskResult] = []
    for d in doses_mg:
        result = bcs_risk_assessment(
            drug_name=drug_name,
            dose_mg=d,
            solubility_mg_mL_pH68=solubility_mg_mL,
            peff_cm_s=peff_cm_s,
            logP=0.0,
            mw=300.0,
        )
        results.append(result)

    return sorted(results, key=lambda r: r.dose_mg)


__all__ = [
    "BCSRiskResult",
    "dose_number",
    "predict_fa_from_do",
    "bcs_risk_assessment",
    "dose_escalation_fa",
]
