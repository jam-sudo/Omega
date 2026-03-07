"""Drug Crystal Dissolution Kinetics — Phase 887.

Models dissolution of crystalline drug polymorphs with different crystal habits
and solubilities. Different crystal forms (anhydrous, hydrate, solvate, amorphous)
have different intrinsic dissolution rates (IDR) and thermodynamic solubilities.

Amorphous > metastable polymorph > stable polymorph in terms of dissolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "CrystalDissolutionResult",
    "predict_crystal_dissolution",
    "compare_crystal_forms",
]

_VALID_FORMS = {"amorphous", "anhydrous", "hydrate", "solvate", "metastable"}


@dataclass(frozen=True)
class CrystalDissolutionResult:
    """Result of crystal dissolution kinetics prediction."""

    drug_name: str
    crystal_form: str
    dose_mg: float
    solubility_mg_mL: float  # thermodynamic solubility
    idr_mg_cm2_min: float  # intrinsic dissolution rate
    dissolution_rate_constant_per_min: float  # k_d (per min)
    t50_dissolution_min: float  # time to 50% dissolved
    t90_dissolution_min: float  # time to 90% dissolved
    fraction_dissolved_at_30min: float
    fraction_dissolved_at_60min: float
    f_abs_predicted: float  # predicted oral fraction absorbed
    dissolution_limited: bool  # True if dissolution limits absorption
    notes: str


def _default_solubility(crystal_form: str, logp: float) -> float:
    """Return default thermodynamic solubility (mg/mL) by crystal form."""
    base = max(0.1, math.pow(10, logp * -0.3 + 1.5))
    multipliers = {
        "amorphous": 3.0,
        "anhydrous": 1.0,
        "hydrate": 0.8,
        "solvate": 1.2,
        "metastable": 1.5,
    }
    return base * multipliers[crystal_form]


def predict_crystal_dissolution(
    drug_name: str,
    dose_mg: float,
    crystal_form: str = "anhydrous",
    surface_area_cm2: float = 100.0,
    volume_mL: float = 250.0,
    solubility_mg_mL: float | None = None,
    logp: float = 2.0,
    pka: float | None = None,
    ph: float = 6.8,
) -> CrystalDissolutionResult:
    """Predict dissolution kinetics for a crystalline drug form.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams.
    crystal_form:
        Crystal form: "amorphous", "anhydrous", "hydrate", "solvate", "metastable".
    surface_area_cm2:
        Particle surface area available for dissolution (cm²).
    volume_mL:
        Dissolution medium volume (mL).
    solubility_mg_mL:
        Thermodynamic solubility; if None, estimated from logP and crystal form.
    logp:
        Octanol-water partition coefficient.
    pka:
        pKa of drug (unused in base model, reserved for future pH correction).
    ph:
        pH of dissolution medium.

    Returns
    -------
    CrystalDissolutionResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if crystal_form not in _VALID_FORMS:
        raise ValueError(
            f"crystal_form must be one of {sorted(_VALID_FORMS)}, got {crystal_form!r}"
        )
    if surface_area_cm2 <= 0:
        raise ValueError(f"surface_area_cm2 must be > 0, got {surface_area_cm2}")
    if volume_mL <= 0:
        raise ValueError(f"volume_mL must be > 0, got {volume_mL}")

    # Solubility
    sol = (
        solubility_mg_mL
        if solubility_mg_mL is not None
        else _default_solubility(crystal_form, logp)
    )

    # IDR: diffusion layer model h=0.2 cm, D~1e-6 cm²/s → IDR = Cs * D/h
    # Simplified: IDR (mg/cm²/min) = solubility * 0.005
    idr = sol * 0.005

    # First-order dissolution rate constant (per min)
    k_d = idr * surface_area_cm2 / dose_mg

    # Dissolution profile: F(t) = 1 - exp(-k_d * t), clamped to [0, 1]
    if k_d > 0:
        t50 = math.log(2) / k_d
        t90 = 2.302585 / k_d  # ln(10)/k_d
    else:
        t50 = float("inf")
        t90 = float("inf")

    f30 = min(1.0, 1.0 - math.exp(-k_d * 30))
    f60 = min(1.0, 1.0 - math.exp(-k_d * 60))

    f_abs = min(1.0, f60 * 0.85)
    diss_limited = f60 < 0.8

    # Build notes
    if crystal_form == "amorphous":
        notes = "Amorphous form — highest dissolution but physical instability risk"
    elif diss_limited:
        notes = (
            "Dissolution-limited absorption — consider supersaturation or particle size reduction"
        )
    else:
        notes = "Complete dissolution expected within 60 min"

    return CrystalDissolutionResult(
        drug_name=drug_name,
        crystal_form=crystal_form,
        dose_mg=dose_mg,
        solubility_mg_mL=sol,
        idr_mg_cm2_min=idr,
        dissolution_rate_constant_per_min=k_d,
        t50_dissolution_min=t50,
        t90_dissolution_min=t90,
        fraction_dissolved_at_30min=f30,
        fraction_dissolved_at_60min=f60,
        f_abs_predicted=f_abs,
        dissolution_limited=diss_limited,
        notes=notes,
    )


def compare_crystal_forms(
    drug_name: str,
    dose_mg: float,
    logp: float = 2.0,
    **kwargs,
) -> list[CrystalDissolutionResult]:
    """Compare all crystal forms for a drug, sorted by predicted absorption.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams.
    logp:
        Octanol-water partition coefficient.
    **kwargs:
        Additional keyword arguments forwarded to predict_crystal_dissolution
        (e.g. surface_area_cm2, volume_mL, ph).

    Returns
    -------
    List of 5 CrystalDissolutionResult, one per form, sorted by
    f_abs_predicted descending.
    """
    results = [
        predict_crystal_dissolution(drug_name, dose_mg, crystal_form=form, logp=logp, **kwargs)
        for form in sorted(_VALID_FORMS)
    ]
    return sorted(results, key=lambda r: r.f_abs_predicted, reverse=True)
