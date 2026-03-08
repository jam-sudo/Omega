"""Phase 1109 — Drug Subcutaneous Bioavailability Prediction.

Predicts bioavailability of subcutaneously injected drugs using a two-pathway
model (capillary vs lymphatic absorption) based on MW, logP, injection site,
and formulation parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SCBioavailabilityResult:
    """Result of SC bioavailability prediction."""

    drug_name: str
    mw_Da: float
    logP: float
    dose_mg: float
    injection_volume_mL: float
    injection_site: str
    f_capillary_absorption: float
    f_lymphatic_absorption: float
    f_sc_bioavailability: float
    ka_effective_per_h: float
    tmax_predicted_h: float
    absorption_route: str
    dose_feasibility: str
    notes: str


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_VALID_SITES = {"abdomen", "thigh", "arm"}

_SITE_MULTIPLIER = {
    "abdomen": 1.0,
    "thigh": 0.85,
    "arm": 0.90,
}


def _validate_inputs(
    mw_Da: float,
    logP: float,
    dose_mg: float,
    injection_volume_mL: float,
    injection_site: str,
) -> None:
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if injection_volume_mL <= 0:
        raise ValueError(f"injection_volume_mL must be > 0, got {injection_volume_mL}")
    if injection_site not in _VALID_SITES:
        raise ValueError(f"injection_site must be one of {_VALID_SITES}, got '{injection_site}'")
    if not (-5 <= logP <= 10):
        raise ValueError(f"logP must be in [-5, 10], got {logP}")


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------


def _sigmoid_mw(mw_Da: float) -> float:
    """Fraction absorbed via lymphatic route."""
    exponent = -(mw_Da - 15000.0) / 5000.0
    return 1.0 / (1.0 + math.exp(exponent))


def predict_sc_bioavailability(
    drug_name: str,
    mw_Da: float,
    logP: float,
    dose_mg: float,
    solubility_mg_mL: float = 100.0,
    injection_volume_mL: float = 1.0,
    injection_site: str = "abdomen",
) -> SCBioavailabilityResult:
    """Predict SC bioavailability for a drug.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    mw_Da:
        Molecular weight in Daltons.
    logP:
        Octanol-water partition coefficient.
    dose_mg:
        Dose in milligrams.
    solubility_mg_mL:
        Aqueous solubility in mg/mL (default 100).
    injection_volume_mL:
        Volume of injection in mL (default 1.0).
    injection_site:
        One of "abdomen", "thigh", "arm" (default "abdomen").

    Returns
    -------
    SCBioavailabilityResult
    """
    _validate_inputs(mw_Da, logP, dose_mg, injection_volume_mL, injection_site)

    # --- Pathway fractions ---
    f_lymph = _sigmoid_mw(mw_Da)
    f_cap = 1.0 - f_lymph

    # --- Solubility check ---
    f_soluble = 1.0 if solubility_mg_mL * injection_volume_mL >= dose_mg * 0.9 else 0.7

    # --- Pathway bioavailabilities ---
    F_cap = 0.95 * f_soluble
    F_lymph = 0.85

    # --- Combined SC F ---
    F_sc = f_cap * F_cap + f_lymph * F_lymph

    # --- Lipophilicity penalty ---
    if logP > 3:
        F_sc *= 0.85

    # --- Site multiplier ---
    site_mult = _SITE_MULTIPLIER[injection_site]
    F_sc *= site_mult

    # --- Volume penalty ---
    if injection_volume_mL > 2.0:
        F_sc *= 1.0 - 0.1 * (injection_volume_mL - 2.0)

    # Clamp to [0, 1]
    F_sc = max(0.0, min(1.0, F_sc))

    # --- Effective absorption rate ---
    ka_effective = f_cap * 0.5 + f_lymph * 0.05

    # --- tmax approximation ---
    tmax = 1.0 / ka_effective if ka_effective > 0 else float("inf")

    # --- Absorption route classification ---
    if f_cap >= 0.7:
        absorption_route = "capillary_dominant"
    elif f_lymph >= 0.7:
        absorption_route = "lymphatic_dominant"
    else:
        absorption_route = "mixed"

    # --- Dose feasibility ---
    if injection_volume_mL <= 2.0:
        dose_feasibility = "feasible"
    elif injection_volume_mL <= 4.0:
        dose_feasibility = "borderline"
    else:
        dose_feasibility = "challenging"

    # --- Notes ---
    note_parts: list[str] = []
    if logP > 3:
        note_parts.append("high logP reduces F_sc due to local tissue binding")
    if injection_volume_mL > 2.0:
        note_parts.append(
            f"volume {injection_volume_mL:.1f} mL reduces F_sc by "
            f"{10 * (injection_volume_mL - 2.0):.0f}%"
        )
    if f_soluble < 1.0:
        note_parts.append("insufficient solubility at injection volume")
    if injection_site != "abdomen":
        note_parts.append(f"site '{injection_site}' applies {site_mult:.0%} absorption multiplier")
    notes = "; ".join(note_parts) if note_parts else "standard SC absorption"

    return SCBioavailabilityResult(
        drug_name=drug_name,
        mw_Da=mw_Da,
        logP=logP,
        dose_mg=dose_mg,
        injection_volume_mL=injection_volume_mL,
        injection_site=injection_site,
        f_capillary_absorption=f_cap,
        f_lymphatic_absorption=f_lymph,
        f_sc_bioavailability=F_sc,
        ka_effective_per_h=ka_effective,
        tmax_predicted_h=tmax,
        absorption_route=absorption_route,
        dose_feasibility=dose_feasibility,
        notes=notes,
    )


def compare_injection_sites(
    drug_name: str,
    mw_Da: float,
    logP: float,
    dose_mg: float,
    solubility_mg_mL: float = 100.0,
    injection_volume_mL: float = 1.0,
) -> list[SCBioavailabilityResult]:
    """Compare SC bioavailability across all three injection sites.

    Returns a list of results sorted by f_sc_bioavailability descending.
    """
    results = [
        predict_sc_bioavailability(
            drug_name=drug_name,
            mw_Da=mw_Da,
            logP=logP,
            dose_mg=dose_mg,
            solubility_mg_mL=solubility_mg_mL,
            injection_volume_mL=injection_volume_mL,
            injection_site=site,
        )
        for site in _VALID_SITES
    ]
    results.sort(key=lambda r: r.f_sc_bioavailability, reverse=True)
    return results
