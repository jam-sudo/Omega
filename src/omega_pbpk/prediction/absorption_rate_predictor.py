"""Predict oral absorption rate constant (ka) and lag time from physicochemical properties.

Based on BCS correlations: ka depends on dissolution rate, membrane permeability, GI transit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AbsorptionRatePrediction:
    """Result of an absorption rate prediction."""

    compound_name: str
    ka_per_h: float
    lag_time_h: float
    predicted_tmax_h: float  # given ke=0.2/h as reference
    dissolution_limited: bool  # True if dissolution is rate-limiting
    permeability_limited: bool  # True if Papp < 5e-6 cm/s
    bcs_class: str  # "I", "II", "III", "IV"
    f_abs_predicted: float  # predicted fraction absorbed (0-1)
    notes: str


def _classify_bcs(
    logP: float,
    mw: float,
    psa: float,
    solubility_mg_mL: float,
    dose_mg: float,
    dose_volume_mL: float = 250.0,
) -> tuple[str, bool, bool]:
    """Classify compound into BCS class I-IV.

    Returns (bcs_class, high_perm, high_sol).
    """
    # High permeability: logP > 1 AND psa < 140 AND mw < 500
    high_perm = (logP > 1.0) and (psa < 140.0) and (mw < 500.0)

    # High solubility: solubility * dose_volume > dose
    high_sol = (solubility_mg_mL * dose_volume_mL) > dose_mg

    if high_perm and high_sol:
        bcs_class = "I"
    elif high_perm and not high_sol:
        bcs_class = "II"
    elif not high_perm and high_sol:
        bcs_class = "III"
    else:
        bcs_class = "IV"

    return bcs_class, high_perm, high_sol


def _permeability_factor(logP: float, psa: float) -> float:
    """Compute permeability factor from logP and PSA."""
    logp_sigmoid = 1.0 if logP > 1.0 else 0.5
    return (1.0 / (1.0 + psa / 60.0)) * logp_sigmoid


def _dissolution_factor(particle_size_um: float) -> float:
    """Compute dissolution factor from particle size (smaller = faster)."""
    return 1.0 / (1.0 + particle_size_um / 50.0)


def _compute_lag_time(
    logP: float, pka_acid: float | None, pka_base: float | None, gastric_pH: float
) -> float:
    """Estimate lag time based on ionization in gastric environment."""
    base_lag = 0.25  # h (standard gastroduodenal transit contribution)

    ionization_delay = 0.0

    # Acid drug: ionized at gastric pH (pH 1.5) if pKa < 4
    if pka_acid is not None:
        if pka_acid < 4.0:
            # Mostly ionized in stomach, delay absorption
            ionization_delay += 0.1
        else:
            ionization_delay += 0.0

    # Base drug: protonated (ionized) at gastric pH
    if pka_base is not None:
        if pka_base > gastric_pH + 2:
            # Strongly protonated in stomach, slow gastric absorption
            ionization_delay += 0.1

    return base_lag + ionization_delay


def _predict_papp(logP: float, mw: float, psa: float) -> float:
    """Predict apparent permeability Papp (cm/s) from physicochemical properties.

    Uses a simplified model based on logP, MW, and PSA.
    Returns Papp in cm/s.
    """
    # Simplified permeability prediction
    # High logP, low PSA, low MW → high Papp
    log_papp = -5.5 + 0.3 * logP - 0.005 * psa - 0.003 * (mw - 300.0) / 100.0
    return 10.0**log_papp


def predict_absorption_rate(
    compound_name: str,
    logP: float,
    mw: float,
    psa: float,
    solubility_mg_mL: float,
    particle_size_um: float = 100.0,
    dose_mg: float = 100.0,
    pka_acid: float | None = None,
    pka_base: float | None = None,
    gastric_pH: float = 1.5,
    intestinal_pH: float = 6.5,
) -> AbsorptionRatePrediction:
    """Predict oral absorption rate constant ka and related PK parameters.

    Parameters
    ----------
    compound_name: name of the compound
    logP: lipophilicity
    mw: molecular weight (Da)
    psa: polar surface area (A^2)
    solubility_mg_mL: aqueous solubility
    particle_size_um: particle diameter (default 100 um)
    dose_mg: dose amount (default 100 mg)
    pka_acid: optional acid pKa
    pka_base: optional base pKa
    gastric_pH: default 1.5 (fasted)
    intestinal_pH: default 6.5
    """
    # Input validation
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if psa < 0:
        raise ValueError(f"psa must be >= 0, got {psa}")
    if solubility_mg_mL < 0:
        raise ValueError(f"solubility_mg_mL must be >= 0, got {solubility_mg_mL}")
    if particle_size_um <= 0:
        raise ValueError(f"particle_size_um must be > 0, got {particle_size_um}")

    # Base ka (per h) — starting value calibrated for typical oral drugs
    base_ka = 1.5  # per h

    # Compute factors
    perm_factor = _permeability_factor(logP, psa)
    diss_factor = _dissolution_factor(particle_size_um)

    # ka: combined effect of permeability and dissolution
    ka = base_ka * perm_factor * diss_factor

    # Ensure ka is positive and reasonable
    ka = max(ka, 0.01)

    # Lag time
    lag_time_h = _compute_lag_time(logP, pka_acid, pka_base, gastric_pH)

    # Predicted tmax using reference ke=0.2/h (flip-flop corrected)
    ke_ref = 0.2  # per h
    if abs(ka - ke_ref) > 1e-6:
        predicted_tmax_h = math.log(ka / ke_ref) / (ka - ke_ref)
    else:
        predicted_tmax_h = 1.0 / ka
    predicted_tmax_h = max(predicted_tmax_h, 0.01) + lag_time_h

    # BCS classification
    bcs_class, high_perm, high_sol = _classify_bcs(logP, mw, psa, solubility_mg_mL, dose_mg)

    # Predicted Papp (cm/s)
    papp = _predict_papp(logP, mw, psa)
    permeability_limited = papp < 5e-6

    # Dissolution limited: large particles or low solubility
    # If dissolution factor is < 0.5 (particle_size > 50 um) and low solubility → dissolution limited
    dissolution_limited = (particle_size_um > 50.0) or (not high_sol)

    # Predicted fraction absorbed
    # BCS-based estimate
    if bcs_class == "I":
        f_abs_base = 0.90
    elif bcs_class == "II":
        f_abs_base = 0.60
    elif bcs_class == "III":
        f_abs_base = 0.50
    else:  # IV
        f_abs_base = 0.20

    # Adjust for particle size (large particles reduce fa)
    particle_penalty = 1.0 / (1.0 + particle_size_um / 200.0)
    f_abs_predicted = f_abs_base * particle_penalty

    # Cap and floor
    f_abs_predicted = max(0.05, min(1.0, f_abs_predicted))

    notes = (
        f"BCS class {bcs_class} compound. "
        f"Papp~{papp:.2e} cm/s ({'low' if permeability_limited else 'high'} perm). "
        f"Dissolution {'limited' if dissolution_limited else 'adequate'} "
        f"(particle size={particle_size_um} um). "
        f"ka={ka:.3f}/h, lag={lag_time_h:.2f}h, fa~{f_abs_predicted:.2f}."
    )

    return AbsorptionRatePrediction(
        compound_name=compound_name,
        ka_per_h=ka,
        lag_time_h=lag_time_h,
        predicted_tmax_h=predicted_tmax_h,
        dissolution_limited=dissolution_limited,
        permeability_limited=permeability_limited,
        bcs_class=bcs_class,
        f_abs_predicted=f_abs_predicted,
        notes=notes,
    )


def screen_absorption(compounds: list[dict]) -> list[AbsorptionRatePrediction]:
    """Screen multiple compounds, sorted by ka descending.

    Each dict should contain keyword arguments for predict_absorption_rate.
    """
    if not compounds:
        return []

    results = []
    for compound in compounds:
        result = predict_absorption_rate(**compound)
        results.append(result)

    # Sort by ka descending
    results.sort(key=lambda r: r.ka_per_h, reverse=True)
    return results
