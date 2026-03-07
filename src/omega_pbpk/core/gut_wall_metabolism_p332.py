"""Phase 332 — Gut Wall Metabolism Model.

Models first-pass gut wall metabolism (CYP3A4 in enterocytes)
affecting oral bioavailability using the well-stirred model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log


@dataclass(frozen=True)
class GutWallMetabolismResult:
    """Result of gut wall metabolism calculation (frozen, no lists)."""

    drug_name: str
    dose_mg: float
    f_abs: float
    fg_gut_wall: float
    fh_hepatic: float
    f_oral: float
    gut_extraction_ratio: float
    hepatic_extraction_ratio: float
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    t_half_h: float
    cl_oral_apparent_L_per_h: float
    high_gut_metabolism: bool
    notes: str


def _trapezoidal_auc(x: list, y: list) -> float:
    """Manual trapezoidal AUC."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_gut_wall_metabolism(
    drug_name: str,
    dose_mg: float,
    f_abs: float,
    cl_gut_wall_L_per_h: float,
    cl_hepatic_L_per_h: float,
    vd_L: float,
    t_end_h: float,
    dt_h: float,
    q_gut_L_per_h: float = 18.0,
    q_hepatic_L_per_h: float = 90.0,
    fu_plasma: float = 1.0,
    fu_gut: float = 1.0,
) -> GutWallMetabolismResult:
    """Simulate gut wall first-pass metabolism and oral PK.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose in mg.
    f_abs:
        Fraction absorbed from gut lumen (0, 1].
    cl_gut_wall_L_per_h:
        Gut wall intrinsic clearance (L/h), >= 0.
    cl_hepatic_L_per_h:
        Hepatic clearance (L/h), > 0.
    vd_L:
        Volume of distribution (L), > 0.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).
    q_gut_L_per_h:
        Gut blood flow (L/h), default 18.0.
    q_hepatic_L_per_h:
        Hepatic blood flow (L/h), default 90.0.
    fu_plasma:
        Fraction unbound in plasma, default 1.0.
    fu_gut:
        Fraction unbound in gut wall, default 1.0.

    Returns
    -------
    GutWallMetabolismResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not (0 < f_abs <= 1):
        raise ValueError("f_abs must be in (0, 1]")
    if cl_gut_wall_L_per_h < 0:
        raise ValueError("cl_gut_wall_L_per_h must be >= 0")
    if cl_hepatic_L_per_h <= 0:
        raise ValueError("cl_hepatic_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0 or dt_h > t_end_h:
        raise ValueError("dt_h must be > 0 and <= t_end_h")
    if q_gut_L_per_h <= 0:
        raise ValueError("q_gut_L_per_h must be > 0")
    if q_hepatic_L_per_h <= 0:
        raise ValueError("q_hepatic_L_per_h must be > 0")
    if not (0 < fu_plasma <= 1):
        raise ValueError("fu_plasma must be in (0, 1]")
    if not (0 < fu_gut <= 1):
        raise ValueError("fu_gut must be in (0, 1]")

    # Well-stirred gut wall extraction ratio
    # Eg = 1 - Q_gut / (Q_gut + CL_gut_wall * fu_gut)
    gut_extraction_ratio = 1.0 - q_gut_L_per_h / (q_gut_L_per_h + cl_gut_wall_L_per_h * fu_gut)
    fg_gut_wall = 1.0 - gut_extraction_ratio

    # Well-stirred hepatic extraction ratio
    # Eh = 1 - Q_hepatic / (Q_hepatic + CL_hepatic * fu_plasma)
    hepatic_extraction_ratio = 1.0 - q_hepatic_L_per_h / (
        q_hepatic_L_per_h + cl_hepatic_L_per_h * fu_plasma
    )
    fh_hepatic = 1.0 - hepatic_extraction_ratio

    # Overall oral bioavailability
    f_oral = f_abs * fg_gut_wall * fh_hepatic

    # Simulation parameters
    ka = 1.0  # absorption rate constant (1/h)
    ke = cl_hepatic_L_per_h / vd_L  # elimination rate constant (1/h)

    # Initial conditions
    # Amount in gut compartment = f_oral * dose_mg
    A_gut = f_oral * dose_mg
    C_sys = 0.0

    # Storage
    times_h: list = []
    c_sys_list: list = []

    n_steps = int(round(t_end_h / dt_h))

    for step in range(n_steps + 1):
        t = step * dt_h
        times_h.append(t)
        c_sys_list.append(C_sys)

        if step < n_steps:
            # dA_gut/dt = -ka * A_gut
            dA_gut = -ka * A_gut
            # dC/dt = ka * A_gut / vd - ke * C
            dC_sys = ka * A_gut / vd_L - ke * C_sys

            A_gut += dA_gut * dt_h
            if A_gut < 0.0:
                A_gut = 0.0
            C_sys += dC_sys * dt_h
            if C_sys < 0.0:
                C_sys = 0.0

    # Derived PK metrics
    auc = _trapezoidal_auc(times_h, c_sys_list)
    cmax = max(c_sys_list)
    tmax_idx = c_sys_list.index(cmax)
    tmax = times_h[tmax_idx]

    # Half-life from elimination rate constant
    if ke > 0:
        t_half = log(2.0) / ke
    else:
        t_half = float("inf")

    # Apparent oral clearance: dose_oral / AUC
    if auc > 0:
        cl_oral_apparent = dose_mg / auc
    else:
        cl_oral_apparent = float("inf")

    high_gut_metabolism = gut_extraction_ratio > 0.3

    notes = (
        f"Gut wall metabolism for {drug_name}. "
        f"Eg={gut_extraction_ratio:.3f}, Eh={hepatic_extraction_ratio:.3f}, "
        f"Fg={fg_gut_wall:.3f}, Fh={fh_hepatic:.3f}, F_oral={f_oral:.3f}. "
        f"{'High gut extraction.' if high_gut_metabolism else 'Low gut extraction.'}"
    )

    return GutWallMetabolismResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        f_abs=f_abs,
        fg_gut_wall=fg_gut_wall,
        fh_hepatic=fh_hepatic,
        f_oral=f_oral,
        gut_extraction_ratio=gut_extraction_ratio,
        hepatic_extraction_ratio=hepatic_extraction_ratio,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        t_half_h=t_half,
        cl_oral_apparent_L_per_h=cl_oral_apparent,
        high_gut_metabolism=high_gut_metabolism,
        notes=notes,
    )
