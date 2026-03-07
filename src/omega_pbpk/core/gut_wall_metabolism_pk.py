"""Phase 869 — Gut Wall Metabolism PBPK (Pang-Rowland model).

Models first-pass intestinal (gut wall) metabolism and hepatic first-pass
to predict oral bioavailability and plasma concentration-time profiles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GutWallMetabolismResult:
    """Result of gut wall metabolism PBPK simulation."""

    drug_name: str
    dose_mg: float
    fa: float
    clint_gut_mL_per_min: float
    clint_hep_mL_per_min: float
    fg_gut_availability: float
    fh_hepatic_availability: float
    f_bioavailability: float
    effective_dose_mg: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    gut_extraction_ratio: float
    hepatic_extraction_ratio: float
    extraction_dominant: str
    notes: str


def _compute_fg(clint_gut_mL_per_min: float, fug: float, q_villous_mL_per_min: float) -> float:
    """Compute intestinal availability using Pang-Rowland model."""
    numerator = clint_gut_mL_per_min * fug
    fg = 1.0 - numerator / (q_villous_mL_per_min + numerator)
    # Clamp to [0, 1] for numerical safety
    return max(0.0, min(1.0, fg))


def _compute_fh(clint_hep_mL_per_min: float, fup: float, q_hepatic_mL_per_min: float) -> float:
    """Compute hepatic availability using well-stirred model."""
    numerator = clint_hep_mL_per_min * fup
    eh = numerator / (q_hepatic_mL_per_min + numerator)
    fh = 1.0 - eh
    return max(0.0, min(1.0, fh))


def _simulate_1cpt_oral(
    dose_eff_mg: float,
    ka_per_h: float,
    ke_per_h: float,
    vd_L: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[list[float], list[float]]:
    """Simulate 1-compartment oral PK analytically."""
    times: list[float] = []
    concs: list[float] = []

    # Handle edge case where ka == ke to avoid division by zero
    if abs(ka_per_h - ke_per_h) < 1e-9:
        ke_per_h = ke_per_h * 1.0001

    prefactor = (dose_eff_mg * ka_per_h) / (vd_L * (ka_per_h - ke_per_h))

    t = 0.0
    while t <= t_end_h + 1e-9:
        c = prefactor * (math.exp(-ke_per_h * t) - math.exp(-ka_per_h * t))
        c = max(0.0, c)
        times.append(round(t, 6))
        concs.append(c)
        t += dt_h

    return times, concs


def _compute_auc_trapz(times: list[float], concs: list[float]) -> float:
    """Manual trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1])
    return auc


def simulate_gut_wall_metabolism(
    drug_name: str,
    dose_mg: float,
    fa: float = 0.9,
    clint_gut_mL_per_min: float = 20.0,
    fug: float = 1.0,
    q_villous_mL_per_min: float = 18.0,
    clint_hep_mL_per_min: float = 100.0,
    fup: float = 0.1,
    q_hepatic_mL_per_min: float = 1500.0,
    ka_per_h: float = 1.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> GutWallMetabolismResult:
    """Simulate gut wall metabolism PBPK using Pang-Rowland model.

    Args:
        drug_name: Name of the drug.
        dose_mg: Oral dose in mg. Must be > 0.
        fa: Fraction absorbed from GI lumen (default 0.9). Must be in (0, 1].
        clint_gut_mL_per_min: Gut wall intrinsic clearance (mL/min). Must be >= 0.
        fug: Unbound fraction in gut wall (default 1.0).
        q_villous_mL_per_min: Villous blood flow (mL/min). Must be > 0.
        clint_hep_mL_per_min: Hepatic intrinsic clearance (mL/min).
        fup: Unbound fraction in plasma.
        q_hepatic_mL_per_min: Hepatic blood flow (mL/min).
        ka_per_h: Oral absorption rate constant (per h).
        vd_L: Volume of distribution (L).
        t_end_h: Simulation end time (h).
        dt_h: Time step (h).

    Returns:
        GutWallMetabolismResult with bioavailability components and PK profile.
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if fa <= 0 or fa > 1:
        raise ValueError("fa must be in (0, 1]")
    if clint_gut_mL_per_min < 0:
        raise ValueError("clint_gut_mL_per_min must be >= 0")
    if q_villous_mL_per_min <= 0:
        raise ValueError("q_villous_mL_per_min must be > 0")

    # Compute gut wall and hepatic availability
    fg = _compute_fg(clint_gut_mL_per_min, fug, q_villous_mL_per_min)
    fh = _compute_fh(clint_hep_mL_per_min, fup, q_hepatic_mL_per_min)
    f = fa * fg * fh

    gut_er = 1.0 - fg
    hep_er = 1.0 - fh

    if gut_er > hep_er:
        extraction_dominant = "gut"
    elif hep_er > gut_er:
        extraction_dominant = "hepatic"
    else:
        extraction_dominant = "equal"

    effective_dose_mg = dose_mg * f

    # Compute elimination rate constant
    # CL_systemic from hepatic clearance: CL = Q * EH (well-stirred)
    q_hep_L_per_h = q_hepatic_mL_per_min * 60.0 / 1000.0
    cl_int_hep_L_per_h = clint_hep_mL_per_min * 60.0 / 1000.0
    fup_cl = fup * cl_int_hep_L_per_h
    cl_systemic_L_per_h = q_hep_L_per_h * (fup_cl / (q_hep_L_per_h + fup_cl))
    cl_systemic_L_per_h = max(cl_systemic_L_per_h, 0.001)  # prevent zero

    ke = cl_systemic_L_per_h / vd_L

    # Simulate 1-cpt oral PK
    times, concs = _simulate_1cpt_oral(effective_dose_mg, ka_per_h, ke, vd_L, t_end_h, dt_h)

    # Find Cmax and Tmax
    cmax = 0.0
    tmax = 0.0
    for i, c in enumerate(concs):
        if c > cmax:
            cmax = c
            tmax = times[i]

    auc = _compute_auc_trapz(times, concs)

    notes = (
        f"Pang-Rowland gut wall model. "
        f"Fg={fg:.3f}, Fh={fh:.3f}, F={f:.3f}. "
        f"Dominant extraction: {extraction_dominant}."
    )

    return GutWallMetabolismResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        fa=fa,
        clint_gut_mL_per_min=clint_gut_mL_per_min,
        clint_hep_mL_per_min=clint_hep_mL_per_min,
        fg_gut_availability=fg,
        fh_hepatic_availability=fh,
        f_bioavailability=f,
        effective_dose_mg=effective_dose_mg,
        times_h=times,
        c_plasma_mg_L=concs,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        gut_extraction_ratio=gut_er,
        hepatic_extraction_ratio=hep_er,
        extraction_dominant=extraction_dominant,
        notes=notes,
    )


def sensitivity_gut_clint(
    drug_name: str,
    dose_mg: float,
    clint_gut_values: list[float],
    fa: float = 0.9,
    fup: float = 0.1,
) -> list[GutWallMetabolismResult]:
    """Run sensitivity analysis over gut CLint values.

    Args:
        drug_name: Name of the drug.
        dose_mg: Oral dose in mg.
        clint_gut_values: List of gut CLint values (mL/min) to evaluate.
        fa: Fraction absorbed (default 0.9).
        fup: Plasma unbound fraction (default 0.1).

    Returns:
        List of GutWallMetabolismResult sorted by f_bioavailability descending.
    """
    results = []
    for clint_gut in clint_gut_values:
        result = simulate_gut_wall_metabolism(
            drug_name=drug_name,
            dose_mg=dose_mg,
            fa=fa,
            clint_gut_mL_per_min=clint_gut,
            fup=fup,
        )
        results.append(result)

    results.sort(key=lambda r: r.f_bioavailability, reverse=True)
    return results
