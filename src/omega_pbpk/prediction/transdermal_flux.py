"""
Phase 366 — Transdermal Flux Prediction

Predict transdermal flux and permeability coefficient from physicochemical
properties using the Potts-Guy (1992) and Johnson (1995) models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TransdermalFluxResult:
    compound_name: str
    logP: float
    mw: float

    # Flux predictions
    log_kp_potts_guy: float      # log10(Kp) from Potts-Guy (cm/h)
    kp_cm_per_h: float           # Permeability coefficient (cm/h)

    log_kp_johnson: float        # Johnson 1995 (alternative model)
    kp_johnson_cm_per_h: float

    # Flux at unit concentration
    flux_ug_per_cm2_per_h: float   # At 1 mg/mL donor concentration

    # Feasibility
    flux_therapeutic_ug_per_cm2_per_h: float  # Needed for therapeutic effect
    patch_area_cm2_needed: float              # Area for target systemic dose
    feasibility: str  # "feasible" (<50 cm2), "marginal" (50-100 cm2), "infeasible" (>100 cm2)

    # Lag time
    lag_time_h: float  # Estimated lag time

    # Classification
    transdermal_potential: str  # "high", "moderate", "low"

    notes: str


def predict_transdermal_flux(
    compound_name: str,
    logP: float,
    mw: float,
    target_dose_mg_per_day: float = 10.0,
    patch_concentration_mg_mL: float = 50.0,
    donor_concentration_mg_mL: float = 1.0,
) -> TransdermalFluxResult:
    """Predict transdermal permeability and flux.

    Parameters
    ----------
    compound_name:
        Name of the compound.
    logP:
        Octanol-water partition coefficient (log10).
    mw:
        Molecular weight (Da).
    target_dose_mg_per_day:
        Desired systemic dose delivered per day (mg/day).
    patch_concentration_mg_mL:
        Drug concentration in the patch vehicle (mg/mL).
    donor_concentration_mg_mL:
        Donor compartment concentration for unit-flux calculation (mg/mL).

    Returns
    -------
    TransdermalFluxResult
    """
    # --- Input validation ---
    if mw <= 0:
        raise ValueError("mw must be > 0.")
    if donor_concentration_mg_mL <= 0:
        raise ValueError("donor_concentration_mg_mL must be > 0.")
    if target_dose_mg_per_day <= 0:
        raise ValueError("target_dose_mg_per_day must be > 0.")
    if patch_concentration_mg_mL <= 0:
        raise ValueError("patch_concentration_mg_mL must be > 0.")

    # --- Potts-Guy 1992 ---
    # log Kp (cm/s) = -6.3 + 0.71 * logP - 0.0061 * MW
    log_kp_pg_cm_s = -6.3 + 0.71 * logP - 0.0061 * mw
    kp_pg_cm_per_h = (10.0 ** log_kp_pg_cm_s) * 3600.0
    # log10(Kp in cm/h)
    log_kp_potts_guy = math.log10(kp_pg_cm_per_h) if kp_pg_cm_per_h > 0 else -float("inf")

    # --- Johnson 1995 ---
    # log Kp (cm/s) = -6.43 + 0.50 * logP - 0.007 * MW
    log_kp_j_cm_s = -6.43 + 0.50 * logP - 0.007 * mw
    kp_j_cm_per_h = (10.0 ** log_kp_j_cm_s) * 3600.0
    log_kp_johnson = math.log10(kp_j_cm_per_h) if kp_j_cm_per_h > 0 else -float("inf")

    # --- Flux at donor_concentration (µg/cm²/h) ---
    # J = Kp (cm/h) * C (mg/mL) * 1000 µg/mg
    flux_ug_per_cm2_per_h = kp_pg_cm_per_h * donor_concentration_mg_mL * 1000.0

    # --- Therapeutic flux and required patch area ---
    # Target delivery rate (mg/h)
    target_rate_mg_per_h = target_dose_mg_per_day / 24.0
    # J_therapeutic * area = target_rate
    # J = Kp * patch_concentration (mg/mL) * 1000 (µg/mg) → µg/cm²/h
    # We need area in cm² such that:
    #   Kp (cm/h) * patch_concentration (mg/mL) * area (cm²) = target_rate (mg/h)
    #     → area = target_rate / (Kp * patch_concentration)  [cm²]
    flux_therapeutic_ug_per_cm2_per_h = kp_pg_cm_per_h * patch_concentration_mg_mL * 1000.0
    if kp_pg_cm_per_h > 0 and patch_concentration_mg_mL > 0:
        patch_area_cm2_needed = target_rate_mg_per_h / (
            kp_pg_cm_per_h * patch_concentration_mg_mL
        )
    else:
        patch_area_cm2_needed = float("inf")

    # --- Feasibility ---
    if patch_area_cm2_needed < 50.0:
        feasibility = "feasible"
    elif patch_area_cm2_needed <= 100.0:
        feasibility = "marginal"
    else:
        feasibility = "infeasible"

    # --- Lag time ---
    # h_sc = 10 µm = 0.001 cm (stratum corneum thickness)
    # D_sc ≈ 10^(-9) * exp(logP * 0.1)  [cm²/h, rough approximation]
    h_sc = 0.001  # cm
    d_sc = 1e-9 * math.exp(logP * 0.1)  # cm²/h
    lag_time_h = (h_sc ** 2) / (6.0 * d_sc)

    # --- Transdermal potential ---
    if kp_pg_cm_per_h > 1e-3 and mw < 400 and 1.0 <= logP <= 4.0:
        transdermal_potential = "high"
    elif kp_pg_cm_per_h > 1e-5 and mw < 500:
        transdermal_potential = "moderate"
    else:
        transdermal_potential = "low"

    notes = (
        f"Potts-Guy Kp={kp_pg_cm_per_h:.3e} cm/h, "
        f"Johnson Kp={kp_j_cm_per_h:.3e} cm/h, "
        f"area needed={patch_area_cm2_needed:.1f} cm², "
        f"lag time={lag_time_h:.2f} h."
    )

    return TransdermalFluxResult(
        compound_name=compound_name,
        logP=logP,
        mw=mw,
        log_kp_potts_guy=log_kp_potts_guy,
        kp_cm_per_h=kp_pg_cm_per_h,
        log_kp_johnson=log_kp_johnson,
        kp_johnson_cm_per_h=kp_j_cm_per_h,
        flux_ug_per_cm2_per_h=flux_ug_per_cm2_per_h,
        flux_therapeutic_ug_per_cm2_per_h=flux_therapeutic_ug_per_cm2_per_h,
        patch_area_cm2_needed=patch_area_cm2_needed,
        feasibility=feasibility,
        lag_time_h=lag_time_h,
        transdermal_potential=transdermal_potential,
        notes=notes,
    )


def screen_transdermal_candidates(
    compounds: list[dict],
) -> list[TransdermalFluxResult]:
    """Screen a list of compounds for transdermal potential.

    Parameters
    ----------
    compounds:
        List of dicts each with keys ``name``, ``logP``, ``mw``.

    Returns
    -------
    List of :class:`TransdermalFluxResult` sorted by ``kp_cm_per_h`` descending.
    """
    results = [
        predict_transdermal_flux(
            compound_name=c["name"],
            logP=float(c["logP"]),
            mw=float(c["mw"]),
        )
        for c in compounds
    ]
    return sorted(results, key=lambda r: r.kp_cm_per_h, reverse=True)


def compare_flux_models(
    compound_name: str,
    logP: float,
    mw: float,
) -> dict:
    """Compare Potts-Guy and Johnson Kp predictions.

    Parameters
    ----------
    compound_name:
        Name of the compound.
    logP:
        Octanol-water partition coefficient.
    mw:
        Molecular weight (Da).

    Returns
    -------
    Dict with keys ``potts_guy_kp_cm_per_h``, ``johnson_kp_cm_per_h``, ``average_kp_cm_per_h``.
    """
    result = predict_transdermal_flux(compound_name, logP, mw)
    avg_kp = (result.kp_cm_per_h + result.kp_johnson_cm_per_h) / 2.0
    return {
        "compound_name": compound_name,
        "potts_guy_kp_cm_per_h": result.kp_cm_per_h,
        "johnson_kp_cm_per_h": result.kp_johnson_cm_per_h,
        "average_kp_cm_per_h": avg_kp,
    }
