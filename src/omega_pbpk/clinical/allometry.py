"""Allometric scaling for cross-species PK prediction.

Implements fixed-exponent allometric scaling (Boxenbaum 1982) and
multi-species regression for human PK prediction from preclinical data.

Reference: Mahmood I, Balian JD. J Pharm Sci 1996;85:103-106.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Standard allometric exponents (Boxenbaum 1982)
EXPONENTS = {
    "cl": 0.75,         # Clearance ∝ BW^0.75
    "vd": 1.00,         # Volume of distribution ∝ BW^1.0
    "t_half": 0.25,     # Half-life ∝ BW^0.25 (derived from CL/Vd)
    "flow": 0.75,       # Blood flow ∝ BW^0.75
    "cmax": -0.25,      # Cmax ∝ BW^-0.25 (for fixed mg/kg dose)
}

# Reference body weights (kg)
BODY_WEIGHTS = {
    "mouse": 0.02,
    "rat": 0.25,
    "rabbit": 2.5,
    "dog": 10.0,
    "monkey": 5.0,
    "human": 70.0,
}


@dataclass(frozen=True)
class AllometricPrediction:
    """Result of allometric scaling to human."""
    species_source: str
    cl_human_L_per_h: float
    vd_human_L: float
    t_half_human_h: float
    f_oral_assumed: float
    method: str
    r_squared: float | None = None   # For multi-species regression


def scale_single_species(
    species: str,
    cl_L_per_h_per_kg: float,
    vd_L_per_kg: float,
    human_bw_kg: float = 70.0,
    cl_exponent: float = EXPONENTS["cl"],
    vd_exponent: float = EXPONENTS["vd"],
) -> AllometricPrediction:
    """Scale PK from a single preclinical species to human using fixed exponents.

    Args:
        species: Source species ("rat", "dog", "monkey", etc.)
        cl_L_per_h_per_kg: CL per body weight in source species (L/h/kg)
        vd_L_per_kg: Vd per body weight in source species (L/kg)
        human_bw_kg: Target human body weight
        cl_exponent: Allometric exponent for CL (default 0.75)
        vd_exponent: Allometric exponent for Vd (default 1.0)

    Returns:
        AllometricPrediction with human CL, Vd, t½
    """
    animal_bw = BODY_WEIGHTS.get(species.lower(), 0.25)

    # Compute species-level values (not normalized)
    cl_animal = cl_L_per_h_per_kg * animal_bw
    vd_animal = vd_L_per_kg * animal_bw

    # Allometric coefficient: Y = a × BW^b → a = Y_animal / BW_animal^b
    a_cl = cl_animal / (animal_bw ** cl_exponent)
    a_vd = vd_animal / (animal_bw ** vd_exponent)

    cl_human = a_cl * (human_bw_kg ** cl_exponent)
    vd_human = a_vd * (human_bw_kg ** vd_exponent)
    t_half = float(0.693 * vd_human / max(cl_human, 1e-9))

    return AllometricPrediction(
        species_source=species,
        cl_human_L_per_h=float(cl_human),
        vd_human_L=float(vd_human),
        t_half_human_h=t_half,
        f_oral_assumed=1.0,
        method="fixed_exponent",
    )


def scale_multi_species(
    species_data: dict[str, tuple[float, float]],
    human_bw_kg: float = 70.0,
) -> AllometricPrediction:
    """Scale PK from multiple preclinical species using log-log regression.

    Args:
        species_data: {species_name: (cl_L_per_h_per_kg, vd_L_per_kg)}
        human_bw_kg: Target human body weight

    Returns:
        AllometricPrediction based on regression across species
    """
    if len(species_data) < 2:
        # Fall back to single-species
        species, (cl, vd) = list(species_data.items())[0]
        return scale_single_species(species, cl, vd, human_bw_kg)

    # Build arrays for regression
    bw_arr, cl_arr, vd_arr = [], [], []
    for sp, (cl_per_kg, vd_per_kg) in species_data.items():
        bw = BODY_WEIGHTS.get(sp.lower(), 0.25)
        bw_arr.append(bw)
        cl_arr.append(cl_per_kg * bw)
        vd_arr.append(vd_per_kg * bw)

    bw_log = np.log(bw_arr)
    cl_log = np.log(np.maximum(cl_arr, 1e-9))
    vd_log = np.log(np.maximum(vd_arr, 1e-9))

    # Log-log regression: log(Y) = log(a) + b·log(BW)
    b_cl, log_a_cl = np.polyfit(bw_log, cl_log, 1)
    b_vd, log_a_vd = np.polyfit(bw_log, vd_log, 1)

    # R² for CL regression
    cl_pred = b_cl * bw_log + log_a_cl
    ss_res = np.sum((cl_log - cl_pred) ** 2)
    ss_tot = np.sum((cl_log - cl_log.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-12))

    cl_human = float(math.exp(log_a_cl) * (human_bw_kg ** b_cl))
    vd_human = float(math.exp(log_a_vd) * (human_bw_kg ** b_vd))
    t_half = float(0.693 * vd_human / max(cl_human, 1e-9))

    logger.info(
        f"Multi-species allometry: CL exponent={b_cl:.2f}, Vd exponent={b_vd:.2f}, R²={r2:.3f}"
    )

    return AllometricPrediction(
        species_source=f"multi({','.join(species_data.keys())})",
        cl_human_L_per_h=cl_human,
        vd_human_L=vd_human,
        t_half_human_h=t_half,
        f_oral_assumed=1.0,
        method="log_log_regression",
        r_squared=r2,
    )


def predict_human_from_preclinical(
    smiles: str,
    preclinical_data: dict[str, tuple[float, float]],
    dose_mg: float = 100.0,
    human_bw_kg: float = 70.0,
) -> dict[str, float]:
    """Full preclinical → human PK prediction pipeline.

    Args:
        smiles: Drug SMILES (for ADME prediction)
        preclinical_data: {species: (cl_L_per_h_per_kg, vd_L_per_kg)}
        dose_mg: Human dose to simulate
        human_bw_kg: Human body weight

    Returns:
        dict with predicted human CL, Vd, t½, Cmax (estimated), AUC
    """
    if len(preclinical_data) == 1:
        sp, vals = list(preclinical_data.items())[0]
        pred = scale_single_species(sp, vals[0], vals[1], human_bw_kg)
    else:
        pred = scale_multi_species(preclinical_data, human_bw_kg)

    # Estimated Cmax and AUC (simple 1-compartment estimate)
    cl = pred.cl_human_L_per_h
    vd = pred.vd_human_L
    cmax_est = dose_mg / max(vd, 1.0)
    auc_est = dose_mg / max(cl, 1e-9)

    return {
        "cl_L_per_h": cl,
        "vd_L": vd,
        "t_half_h": pred.t_half_human_h,
        "cmax_est_mg_L": cmax_est,
        "auc_est_mg_h_L": auc_est,
        "method": pred.method,
        "r_squared": pred.r_squared or float("nan"),
        "species_source": pred.species_source,
    }
