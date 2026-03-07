"""
Crystal form effects on oral bioavailability.

Supports amorphous, polymorph_I, polymorph_II, hydrate, and solvate crystal forms.
Uses Q10 temperature correction and dose-number based fa estimation.
"""

import math

VALID_FORMS = {"amorphous", "polymorph_I", "polymorph_II", "hydrate", "solvate"}

# Solubility multipliers relative to polymorph_I (reference = 1.0)
_FORM_MULTIPLIERS = {
    "amorphous": 20.0,
    "polymorph_I": 1.0,
    "polymorph_II": 0.6,
    "hydrate": 0.8,
    "solvate": 1.2,
}

_FORM_NOTES = {
    "amorphous": "Amorphous form: ~20x higher solubility due to lack of crystal lattice energy.",
    "polymorph_I": "Polymorph I: reference crystalline form.",
    "polymorph_II": "Polymorph II: lower energy polymorph with reduced solubility.",
    "hydrate": "Hydrate: water molecules in lattice reduce effective solubility slightly.",
    "solvate": "Solvate: organic solvent in lattice modestly increases solubility.",
}


def crystal_form_solubility(
    form: str,
    intrinsic_solubility_mg_L: float,
    temperature_C: float,
) -> dict:
    """
    Compute solubility of a crystal form at a given temperature.

    Q10 rule: solubility doubles per 10 degrees C rise from 25 C reference.

    Parameters
    ----------
    form : str
        One of 'amorphous', 'polymorph_I', 'polymorph_II', 'hydrate', 'solvate'.
    intrinsic_solubility_mg_L : float
        Intrinsic solubility of polymorph_I at 25 C (mg/L).
    temperature_C : float
        Temperature in Celsius.

    Returns
    -------
    dict with keys: form, solubility_mg_L, temperature_C, fold_vs_polymorph_I, notes
    """
    if form not in VALID_FORMS:
        raise ValueError(f"Invalid crystal form '{form}'. Must be one of {sorted(VALID_FORMS)}.")
    if intrinsic_solubility_mg_L < 0:
        raise ValueError(
            f"intrinsic_solubility_mg_L must be non-negative, got {intrinsic_solubility_mg_L}."
        )

    multiplier = _FORM_MULTIPLIERS[form]
    # Q10 temperature correction: doubles per 10 C above 25 C reference
    delta_t = temperature_C - 25.0
    temp_factor = math.pow(2.0, delta_t / 10.0)

    solubility = intrinsic_solubility_mg_L * multiplier * temp_factor
    fold_vs_pi = multiplier * temp_factor  # fold relative to polymorph_I at same temp

    return {
        "form": form,
        "solubility_mg_L": solubility,
        "temperature_C": temperature_C,
        "fold_vs_polymorph_I": fold_vs_pi,
        "notes": _FORM_NOTES[form],
    }


def predict_f_oral_from_crystal(
    solubility_mg_L: float,
    dose_mg: float,
    peff_cm_s: float,
    gi_volume_L: float = 0.25,
) -> float:
    """
    Predict oral fraction absorbed (fa) from crystal form solubility.

    Uses dose number (Do) approach:
      Do = dose_mg / (solubility_mg_L * gi_volume_L)
      fa = 1.0 if Do <= 1 (dissolution not rate-limiting)
      fa = 1/Do if Do > 1 (solubility-limited)

    Parameters
    ----------
    solubility_mg_L : float
        Drug solubility in mg/L.
    dose_mg : float
        Administered dose in mg.
    peff_cm_s : float
        Effective intestinal permeability in cm/s (not used in Do model directly,
        retained for API consistency and future extension).
    gi_volume_L : float
        GI fluid volume in L (default 0.25 L).

    Returns
    -------
    float : fa in [0, 1]
    """
    if solubility_mg_L < 0:
        raise ValueError("solubility_mg_L must be non-negative.")
    if dose_mg < 0:
        raise ValueError("dose_mg must be non-negative.")
    if gi_volume_L <= 0:
        raise ValueError("gi_volume_L must be positive.")

    if solubility_mg_L == 0:
        return 0.0

    do = dose_mg / (solubility_mg_L * gi_volume_L)

    if do <= 1.0:
        fa = 1.0
    else:
        fa = 1.0 / do

    # Clamp to [0, 1]
    return max(0.0, min(1.0, fa))


def compare_crystal_forms(
    drug_name: str,
    dose_mg: float,
    peff_cm_s: float,
    intrinsic_solubility_mg_L: float,
    temperature_C: float = 25.0,
) -> list:
    """
    Compare oral fa across all 5 crystal forms.

    Parameters
    ----------
    drug_name : str
        Name of the drug (used for labeling).
    dose_mg : float
        Administered dose in mg.
    peff_cm_s : float
        Effective intestinal permeability in cm/s.
    intrinsic_solubility_mg_L : float
        Intrinsic solubility of polymorph_I at 25 C (mg/L).
    temperature_C : float
        Temperature in Celsius (default 25.0).

    Returns
    -------
    list of dicts sorted by fa descending, each with:
        form, solubility_mg_L, fa, fold_vs_polymorph_I
    """
    if dose_mg < 0:
        raise ValueError("dose_mg must be non-negative.")
    if intrinsic_solubility_mg_L < 0:
        raise ValueError("intrinsic_solubility_mg_L must be non-negative.")

    results = []
    for form in VALID_FORMS:
        sol_info = crystal_form_solubility(form, intrinsic_solubility_mg_L, temperature_C)
        sol = sol_info["solubility_mg_L"]
        fa = predict_f_oral_from_crystal(sol, dose_mg, peff_cm_s)
        results.append(
            {
                "form": form,
                "solubility_mg_L": sol,
                "fa": fa,
                "fold_vs_polymorph_I": sol_info["fold_vs_polymorph_I"],
            }
        )

    results.sort(key=lambda x: x["fa"], reverse=True)
    return results


def crystal_conversion_risk(
    storage_rh_pct: float,
    temperature_C: float,
    form: str,
) -> dict:
    """
    Assess risk of crystal form conversion during storage.

    Risk levels:
      'high':     RH > 75% AND temperature > 40 C (especially for amorphous)
      'moderate': RH 60-75% OR temperature 30-40 C
      'low':      otherwise

    Parameters
    ----------
    storage_rh_pct : float
        Relative humidity in percent.
    temperature_C : float
        Storage temperature in Celsius.
    form : str
        Crystal form of the drug.

    Returns
    -------
    dict with keys: risk_level, notes
    """
    if form not in VALID_FORMS:
        raise ValueError(f"Invalid crystal form '{form}'. Must be one of {sorted(VALID_FORMS)}.")

    high_rh = storage_rh_pct > 75.0
    moderate_rh = 60.0 <= storage_rh_pct <= 75.0
    high_temp = temperature_C > 40.0
    moderate_temp = 30.0 <= temperature_C <= 40.0

    if high_rh and high_temp:
        risk_level = "high"
        if form == "amorphous":
            notes = (
                "High risk: amorphous form is especially prone to crystallization "
                "under high humidity and elevated temperature. Store below 40 C and "
                "RH < 60%."
            )
        else:
            notes = (
                "High risk: elevated humidity and temperature may drive polymorphic "
                f"conversion or hydrate/solvate formation for {form}."
            )
    elif moderate_rh or moderate_temp:
        risk_level = "moderate"
        notes = (
            f"Moderate risk: intermediate humidity or temperature conditions may "
            f"slowly promote crystal form changes for {form}. Monitor regularly."
        )
    else:
        risk_level = "low"
        notes = (
            f"Low risk: current storage conditions (RH={storage_rh_pct}%, "
            f"T={temperature_C} C) are acceptable for {form}."
        )

    return {
        "risk_level": risk_level,
        "notes": notes,
    }
