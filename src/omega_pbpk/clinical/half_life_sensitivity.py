"""Phase 504 — Plasma Half-Life Sensitivity Analysis.

Analyze how changes in CL and Vd affect plasma half-life and exposure
metrics using a 1-compartment analytical model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HalfLifeSensitivityResult:
    """Result of a plasma half-life sensitivity analysis."""

    drug_name: str
    baseline_cl_L_per_h: float
    baseline_vd_L: float
    baseline_dose_mg: float

    t_half_h: float
    auc_mg_h_per_L: float
    cmax_mg_L: float
    css_avg_mg_L: float

    # Normalized sensitivity: d(t_half)/d(CL) = -1.0 for 1-cpt
    cl_sensitivity: float
    # Normalized sensitivity: d(t_half)/d(Vd) = +1.0 for 1-cpt
    vd_sensitivity: float
    # Always "CL" for 1-cpt because CL is in denominator
    dominant_parameter: str
    notes: str


def analyze_half_life_sensitivity(
    drug_name: str,
    baseline_cl_L_per_h: float,
    baseline_vd_L: float,
    baseline_dose_mg: float,
    f_oral: float = 1.0,
    dosing_interval_h: float = 24.0,
) -> HalfLifeSensitivityResult:
    """Compute 1-cpt PK metrics and normalized sensitivity coefficients.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    baseline_cl_L_per_h:
        Baseline clearance (L/h).
    baseline_vd_L:
        Baseline volume of distribution (L).
    baseline_dose_mg:
        Dose in mg.
    f_oral:
        Oral bioavailability fraction (0-1].
    dosing_interval_h:
        Dosing interval for steady-state Css calculation (h).

    Returns
    -------
    HalfLifeSensitivityResult
    """
    if baseline_cl_L_per_h <= 0:
        raise ValueError("baseline_cl_L_per_h must be > 0")
    if baseline_vd_L <= 0:
        raise ValueError("baseline_vd_L must be > 0")
    if baseline_dose_mg <= 0:
        raise ValueError("baseline_dose_mg must be > 0")
    if not (0 < f_oral <= 1):
        raise ValueError("f_oral must be in (0, 1]")

    t_half_h = 0.693 * baseline_vd_L / baseline_cl_L_per_h
    auc_mg_h_per_L = baseline_dose_mg / baseline_cl_L_per_h
    cmax_mg_L = baseline_dose_mg / baseline_vd_L
    css_avg_mg_L = f_oral * baseline_dose_mg / (baseline_cl_L_per_h * dosing_interval_h)

    # For a 1-cpt model t_half = 0.693 * Vd / CL:
    # - Normalized d(ln t_half)/d(ln CL) = -1.0
    # - Normalized d(ln t_half)/d(ln Vd) = +1.0
    cl_sensitivity = -1.0
    vd_sensitivity = 1.0
    dominant_parameter = "CL"

    notes = (
        f"1-cpt analytical; t_half={t_half_h:.2f}h; "
        f"AUC={auc_mg_h_per_L:.2f}; Cmax={cmax_mg_L:.4f}; "
        f"Css_avg={css_avg_mg_L:.4f}"
    )

    return HalfLifeSensitivityResult(
        drug_name=drug_name,
        baseline_cl_L_per_h=baseline_cl_L_per_h,
        baseline_vd_L=baseline_vd_L,
        baseline_dose_mg=baseline_dose_mg,
        t_half_h=t_half_h,
        auc_mg_h_per_L=auc_mg_h_per_L,
        cmax_mg_L=cmax_mg_L,
        css_avg_mg_L=css_avg_mg_L,
        cl_sensitivity=cl_sensitivity,
        vd_sensitivity=vd_sensitivity,
        dominant_parameter=dominant_parameter,
        notes=notes,
    )


def sweep_cl_vd(
    drug_name: str,
    baseline_cl_L_per_h: float,
    baseline_vd_L: float,
    baseline_dose_mg: float,
    n_steps: int = 5,
) -> list[dict]:
    """Sweep CL and Vd each over [0.5x, 2.0x] baseline and compute t_half.

    Parameters
    ----------
    drug_name:
        Name of the drug (unused in computation, included for context).
    baseline_cl_L_per_h:
        Baseline clearance (L/h).
    baseline_vd_L:
        Baseline volume of distribution (L).
    baseline_dose_mg:
        Dose in mg (unused in t_half, included for completeness).
    n_steps:
        Number of steps for each axis (both CL and Vd). Default 5.

    Returns
    -------
    List of dicts with keys ``cl_factor``, ``vd_factor``, ``t_half_h``,
    sorted by t_half_h ascending.
    """
    if n_steps < 2:
        n_steps = 2

    # Build factor arrays: linearly spaced from 0.5 to 2.0
    step_size_cl = (2.0 - 0.5) / (n_steps - 1)
    step_size_vd = (2.0 - 0.5) / (n_steps - 1)

    cl_factors = [0.5 + i * step_size_cl for i in range(n_steps)]
    vd_factors = [0.5 + i * step_size_vd for i in range(n_steps)]

    results = []
    for cl_f in cl_factors:
        for vd_f in vd_factors:
            cl = baseline_cl_L_per_h * cl_f
            vd = baseline_vd_L * vd_f
            t_half = 0.693 * vd / cl
            results.append(
                {
                    "cl_factor": cl_f,
                    "vd_factor": vd_f,
                    "t_half_h": t_half,
                }
            )

    results.sort(key=lambda d: d["t_half_h"])
    return results
