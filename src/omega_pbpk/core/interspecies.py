"""Allometric interspecies scaling — animal-to-human PK prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ── Species reference data ──────────────────────────────────────────


@dataclass(frozen=True)
class SpeciesPhysiology:
    name: str
    body_weight_kg: float
    brain_weight_g: float
    mlp_years: float  # maximum lifespan potential


MOUSE = SpeciesPhysiology("mouse", 0.02, 0.4, 2.7)
RAT = SpeciesPhysiology("rat", 0.25, 1.8, 4.5)
RABBIT = SpeciesPhysiology("rabbit", 3.5, 10.0, 13.0)
DOG = SpeciesPhysiology("dog", 10.0, 72.0, 20.0)
MONKEY = SpeciesPhysiology("monkey", 5.0, 62.0, 35.0)
HUMAN = SpeciesPhysiology("human", 70.0, 1400.0, 80.0)

SPECIES_DB: dict[str, SpeciesPhysiology] = {
    s.name: s for s in [MOUSE, RAT, RABBIT, DOG, MONKEY, HUMAN]
}


# ── Data containers ─────────────────────────────────────────────────


@dataclass(frozen=True)
class AnimalPKData:
    species: str  # must be key in SPECIES_DB
    cl_mL_min_kg: float  # clearance in mL/min/kg
    vd_L_kg: float  # volume of distribution in L/kg
    thalf_h: float  # elimination half-life in hours
    dose_mg_kg: float = 1.0


@dataclass(frozen=True)
class AllometricFit:
    coefficient_a: float
    exponent_b: float
    r_squared: float
    human_predicted: float  # value predicted at human BW


@dataclass(frozen=True)
class InterspeciesResult:
    cl_human_L_per_h: float
    vd_human_L: float
    thalf_human_h: float
    method_cl: str
    method_vd: str
    fit_cl: AllometricFit | None
    fit_vd: AllometricFit | None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HumanPKPrediction:
    cl_L_per_h: float
    vd_L: float
    thalf_h: float
    dose_mg: float
    method_used: str
    fit_quality: float | None  # r² of CL fit, None if single species
    warnings: list[str] = field(default_factory=list)


# ── Core functions ──────────────────────────────────────────────────


def scale_single_species(
    animal_value: float,
    animal_bw_kg: float,
    human_bw_kg: float = 70.0,
    exponent: float = 0.75,
) -> float:
    """Scale a PK parameter from one species to another using allometry."""
    if animal_bw_kg <= 0 or human_bw_kg <= 0:
        raise ValueError("Body weights must be > 0")
    return animal_value * (human_bw_kg / animal_bw_kg) ** exponent


def allometric_regression(
    species_bw_kg: list[float],
    species_values: list[float],
    human_bw_kg: float = 70.0,
) -> AllometricFit:
    """Fit log-log allometric regression: Y = a * BW^b."""
    bw = np.asarray(species_bw_kg, dtype=float)
    vals = np.asarray(species_values, dtype=float)
    if len(bw) < 2:
        raise ValueError("Need at least 2 species for regression")

    log_bw = np.log(bw)
    log_vals = np.log(vals)

    # Linear regression in log-log space: log(Y) = log(a) + b*log(BW)
    coeffs = np.polyfit(log_bw, log_vals, 1)
    b = float(coeffs[0])
    log_a = float(coeffs[1])
    a = math.exp(log_a)

    # R-squared
    predicted = log_a + b * log_bw
    ss_res = float(np.sum((log_vals - predicted) ** 2))
    ss_tot = float(np.sum((log_vals - np.mean(log_vals)) ** 2))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    human_pred = a * human_bw_kg**b

    return AllometricFit(
        coefficient_a=a,
        exponent_b=b,
        r_squared=r_sq,
        human_predicted=human_pred,
    )


def correct_mlp(
    cl_values: list[float],
    bw_values: list[float],
    mlp_values: list[float],
    human_bw_kg: float = 70.0,
    human_mlp_years: float = 80.0,
) -> float:
    """MLP-corrected allometric scaling: regress CL×MLP vs BW."""
    cl = np.asarray(cl_values, dtype=float)
    mlp = np.asarray(mlp_values, dtype=float)
    corrected = cl * mlp
    fit = allometric_regression(bw_values, corrected.tolist(), human_bw_kg)
    return fit.human_predicted / human_mlp_years


def correct_brain_weight(
    cl_values: list[float],
    bw_values: list[float],
    brain_weights_g: list[float],
    human_bw_kg: float = 70.0,
    human_brain_g: float = 1400.0,
) -> float:
    """Brain weight-corrected allometric scaling: regress CL×BrW vs BW."""
    cl = np.asarray(cl_values, dtype=float)
    brw = np.asarray(brain_weights_g, dtype=float)
    corrected = cl * brw
    fit = allometric_regression(bw_values, corrected.tolist(), human_bw_kg)
    return fit.human_predicted / human_brain_g


def _apply_rule_of_exponents(
    cl_total_values: list[float],  # total CL in L/h (not per kg)
    bw_values: list[float],
    species_names: list[str],
    human_bw_kg: float = 70.0,
) -> tuple[float, str, AllometricFit]:
    """Apply Mahmood & Balian rule of exponents for CL."""
    fit = allometric_regression(bw_values, cl_total_values, human_bw_kg)
    exp = fit.exponent_b

    if exp <= 0.70:
        return fit.human_predicted, "simple_allometry", fit
    elif exp <= 1.0:
        # MLP correction
        mlp_vals = [SPECIES_DB[s].mlp_years for s in species_names]
        cl_mlp = correct_mlp(cl_total_values, bw_values, mlp_vals, human_bw_kg)
        return cl_mlp, "mlp_correction", fit
    else:
        # Brain weight correction
        brw_vals = [SPECIES_DB[s].brain_weight_g for s in species_names]
        cl_brw = correct_brain_weight(cl_total_values, bw_values, brw_vals, human_bw_kg)
        return cl_brw, "brain_weight_correction", fit


def predict_human_pk(
    animal_data: list[AnimalPKData],
    human_bw_kg: float = 70.0,
) -> HumanPKPrediction:
    """Predict human PK parameters from animal data using allometric scaling.

    For single species: uses standard exponents (CL: 0.75, Vd: 1.0, t½: 0.25).
    For ≥3 species: applies rule of exponents (Mahmood & Balian 1996).
    For 2 species: simple allometric regression.
    """
    if not animal_data:
        raise ValueError("At least one species required")

    warnings: list[str] = []

    # Validate species
    for ad in animal_data:
        if ad.species not in SPECIES_DB:
            raise ValueError(f"Unknown species: {ad.species}. Known: {list(SPECIES_DB.keys())}")

    # Convert to total (non-weight-normalized) values
    species_names = [ad.species for ad in animal_data]
    bw_values = [SPECIES_DB[ad.species].body_weight_kg for ad in animal_data]

    # CL: mL/min/kg → L/h total = mL/min/kg × BW_kg × 60/1000
    cl_total = [
        ad.cl_mL_min_kg * bw * 60.0 / 1000.0 for ad, bw in zip(animal_data, bw_values, strict=True)
    ]
    # Vd: L/kg → L total = L/kg × BW_kg
    vd_total = [ad.vd_L_kg * bw for ad, bw in zip(animal_data, bw_values, strict=True)]

    if len(animal_data) == 1:
        # Single species: fixed exponents
        ad = animal_data[0]
        bw = bw_values[0]
        cl_human = scale_single_species(cl_total[0], bw, human_bw_kg, 0.75)
        vd_human = scale_single_species(vd_total[0], bw, human_bw_kg, 1.0)
        thalf_human = scale_single_species(ad.thalf_h, bw, human_bw_kg, 0.25)
        dose_human = ad.dose_mg_kg * human_bw_kg
        method = "single_species_fixed_exponents"
        fit_quality = None
        warnings.append("Single species scaling — consider multi-species data for reliability")

    elif len(animal_data) == 2:
        # Two species: simple regression (no rule of exponents — need ≥3)
        fit_cl = allometric_regression(bw_values, cl_total, human_bw_kg)
        fit_vd = allometric_regression(bw_values, vd_total, human_bw_kg)
        cl_human = fit_cl.human_predicted
        vd_human = fit_vd.human_predicted
        thalf_human = math.log(2) * vd_human / cl_human if cl_human > 0 else 0.0
        dose_human = animal_data[0].dose_mg_kg * human_bw_kg
        method = "two_species_regression"
        fit_quality = fit_cl.r_squared
        warnings.append("Only 2 species — rule of exponents requires ≥3 species")

    else:
        # ≥3 species: apply rule of exponents for CL
        cl_human, method_cl, fit_cl = _apply_rule_of_exponents(
            cl_total, bw_values, species_names, human_bw_kg
        )
        fit_vd = allometric_regression(bw_values, vd_total, human_bw_kg)
        vd_human = fit_vd.human_predicted
        thalf_human = math.log(2) * vd_human / cl_human if cl_human > 0 else 0.0
        dose_human = animal_data[0].dose_mg_kg * human_bw_kg
        method = f"rule_of_exponents:{method_cl}"
        fit_quality = fit_cl.r_squared

        if fit_cl.r_squared < 0.9:
            warnings.append(f"Poor CL fit (R²={fit_cl.r_squared:.3f}) — prediction unreliable")
        if fit_vd.r_squared < 0.9:
            warnings.append(f"Poor Vd fit (R²={fit_vd.r_squared:.3f}) — Vd prediction unreliable")

    # Sanity checks
    if cl_human <= 0:
        warnings.append("Predicted CL ≤ 0 — check input data")
        cl_human = max(cl_human, 0.001)
    if vd_human <= 0:
        warnings.append("Predicted Vd ≤ 0 — check input data")
        vd_human = max(vd_human, 0.001)

    return HumanPKPrediction(
        cl_L_per_h=cl_human,
        vd_L=vd_human,
        thalf_h=thalf_human,
        dose_mg=dose_human,
        method_used=method,
        fit_quality=fit_quality,
        warnings=warnings,
    )


__all__ = [
    "SpeciesPhysiology",
    "MOUSE",
    "RAT",
    "RABBIT",
    "DOG",
    "MONKEY",
    "HUMAN",
    "SPECIES_DB",
    "AnimalPKData",
    "AllometricFit",
    "InterspeciesResult",
    "HumanPKPrediction",
    "scale_single_species",
    "allometric_regression",
    "correct_mlp",
    "correct_brain_weight",
    "predict_human_pk",
]
