"""Allometric interspecies CL/Vd prediction using power-law regression.

Predicts human PK parameters (CL, Vd, t1/2) from animal data via the simple
allometric power law:  Y = a * BW^b

Log-log linear regression (np.polyfit) is used to estimate a and b.

References
----------
- Mahmood I & Balian JD. J Pharm Sci. 1996;85(10):1101-6
- Boxenbaum H. J Pharmacokinet Biopharm. 1982;10(2):201-27
- Ward KW & Smith BR. Drug Metab Dispos. 2004;32(6):603-11
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AllometricFit:
    """Result of allometric log-log regression.

    Attributes
    ----------
    coefficient_a : Allometric coefficient (Y = a * BW^b).
    exponent_b : Allometric exponent (typically ~0.75 for CL, ~1.0 for Vd).
    r_squared : Coefficient of determination (R^2) of log-log fit.
    n_species : Number of species used in regression.
    pk_param : PK parameter fitted ("CL" or "Vd").
    """

    coefficient_a: float
    exponent_b: float
    r_squared: float
    n_species: int
    pk_param: str


@dataclass(frozen=True)
class AllometricAnalysisResult:
    """Full allometric analysis predicting human CL, Vd, and derived PK.

    Attributes
    ----------
    cl_fit : AllometricFit for CL.
    vd_fit : AllometricFit for Vd.
    human_cl_L_per_h : Predicted human CL (L/h).
    human_vd_L : Predicted human Vd (L).
    predicted_t_half_h : Predicted human terminal half-life (h).
    notes : Summary text.
    """

    cl_fit: AllometricFit
    vd_fit: AllometricFit
    human_cl_L_per_h: float
    human_vd_L: float
    predicted_t_half_h: float
    notes: str


def fit_allometric(
    animal_data: list[dict],
    pk_param: str,
) -> AllometricFit:
    """Fit allometric power law: Y = a * BW^b from animal data.

    Parameters
    ----------
    animal_data : List of dicts, each with keys:
                  - species (str): species name (informational only)
                  - bw_kg (float): body weight (kg), must be > 0
                  - value (float): PK parameter value, must be > 0
    pk_param : PK parameter label, e.g. "CL" or "Vd".

    Returns
    -------
    AllometricFit with fitted coefficient_a, exponent_b, r_squared.

    Raises
    ------
    ValueError : If fewer than 2 data points, or bw_kg/value <= 0.
    """
    if len(animal_data) < 2:
        raise ValueError(
            f"At least 2 data points required for allometric regression, got {len(animal_data)}"
        )

    bw_vals = []
    y_vals = []
    for i, entry in enumerate(animal_data):
        bw = entry["bw_kg"]
        val = entry["value"]
        if bw <= 0:
            raise ValueError(f"bw_kg must be > 0 at index {i}, got {bw}")
        if val <= 0:
            raise ValueError(f"value must be > 0 at index {i}, got {val}")
        bw_vals.append(bw)
        y_vals.append(val)

    bw_arr = np.asarray(bw_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)

    log_bw = np.log(bw_arr)
    log_y = np.log(y_arr)

    # Linear regression in log-log space: log(Y) = log(a) + b*log(BW)
    coeffs = np.polyfit(log_bw, log_y, 1)
    b = float(coeffs[0])
    log_a = float(coeffs[1])
    a = math.exp(log_a)

    # R-squared in log-log space
    predicted_log_y = log_a + b * log_bw
    ss_res = float(np.sum((log_y - predicted_log_y) ** 2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return AllometricFit(
        coefficient_a=a,
        exponent_b=b,
        r_squared=r_sq,
        n_species=len(animal_data),
        pk_param=pk_param,
    )


def predict_human_pk(fit: AllometricFit, human_bw_kg: float = 70.0) -> float:
    """Predict human PK parameter using fitted allometric relationship.

    Parameters
    ----------
    fit : AllometricFit from fit_allometric().
    human_bw_kg : Human body weight (kg). Default 70.0.

    Returns
    -------
    Predicted human PK value (same units as input to fit_allometric).

    Raises
    ------
    ValueError : If human_bw_kg <= 0.
    """
    if human_bw_kg <= 0:
        raise ValueError(f"human_bw_kg must be > 0, got {human_bw_kg}")
    return fit.coefficient_a * (human_bw_kg**fit.exponent_b)


def full_allometric_analysis(
    animal_data_cl: list[dict],
    animal_data_vd: list[dict],
    human_bw_kg: float = 70.0,
) -> AllometricAnalysisResult:
    """Fit allometric models for both CL and Vd and predict human PK.

    Also estimates terminal half-life and steady-state Css for a typical dose.

    Parameters
    ----------
    animal_data_cl : Animal data for CL fitting. Each dict: species, bw_kg, value (L/h).
    animal_data_vd : Animal data for Vd fitting. Each dict: species, bw_kg, value (L).
    human_bw_kg : Assumed human body weight (kg). Default 70.0.

    Returns
    -------
    AllometricAnalysisResult with fits, predicted human PK, and derived metrics.

    Raises
    ------
    ValueError : If either dataset has fewer than 2 data points.
    """
    cl_fit = fit_allometric(animal_data_cl, pk_param="CL")
    vd_fit = fit_allometric(animal_data_vd, pk_param="Vd")

    human_cl = predict_human_pk(cl_fit, human_bw_kg)
    human_vd = predict_human_pk(vd_fit, human_bw_kg)

    # Terminal half-life: t1/2 = ln(2) * Vd / CL
    t_half = math.log(2.0) * human_vd / human_cl if human_cl > 0 else float("inf")

    notes = (
        f"Allometric analysis: CL=a*BW^b (a={cl_fit.coefficient_a:.4f}, "
        f"b={cl_fit.exponent_b:.3f}, R²={cl_fit.r_squared:.3f}); "
        f"Vd=a*BW^b (a={vd_fit.coefficient_a:.4f}, "
        f"b={vd_fit.exponent_b:.3f}, R²={vd_fit.r_squared:.3f}). "
        f"Human ({human_bw_kg:.0f} kg): CL={human_cl:.3f} L/h, "
        f"Vd={human_vd:.2f} L, t1/2={t_half:.2f} h."
    )

    return AllometricAnalysisResult(
        cl_fit=cl_fit,
        vd_fit=vd_fit,
        human_cl_L_per_h=human_cl,
        human_vd_L=human_vd,
        predicted_t_half_h=t_half,
        notes=notes,
    )


__all__ = [
    "AllometricFit",
    "AllometricAnalysisResult",
    "fit_allometric",
    "predict_human_pk",
    "full_allometric_analysis",
]
