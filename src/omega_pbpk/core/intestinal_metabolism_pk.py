"""Phase 728 — Intestinal Metabolism PK.

Models gut-wall first-pass metabolism via CYP3A4 in enterocytes.

Simplified 3-pool model:
  A_gut    (mg): lumen + enterocyte combined
  A_portal (mg): portal vein
  C_plasma (mg/L): systemic plasma

fg (gut wall bioavailability) captures intestinal extraction:
  fg = 1 / (1 + fu_gut * CLint_gut / Q_gut)  [for assess_gut_wall_extraction]

ODEs (forward Euler):
  dA_gut/dt    = -ka * A_gut
  dA_portal/dt = ka * fg * A_gut - ke_portal * A_portal
  dC_plasma/dt = ke_portal * A_portal / Vd - ke * C_plasma

Overall oral bioavailability: F = f_abs * fg * fh
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class IntestinalMetabPKResult:
    """Results from intestinal metabolism PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    a_gut_mg: list
    a_portal_mg: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    f_oral: float
    fg: float
    fh: float
    t_half_h: float
    notes: str


def _validate_params(
    dose_mg: float,
    fg: float,
    fh: float,
    cl_L_per_h: float,
    vd_L: float,
) -> None:
    """Validate simulation parameters."""
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (0 < fg <= 1):
        raise ValueError(f"fg must be in (0, 1], got {fg}")
    if not (0 < fh <= 1):
        raise ValueError(f"fh must be in (0, 1], got {fh}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")


def simulate_intestinal_metabolism_pk(
    drug_name: str = "drug",
    dose_mg: float = 1.0,
    ka_per_h: float = 1.0,
    fg: float = 0.7,
    fh: float = 0.7,
    f_abs: float = 1.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    ke_portal_per_h: float = 5.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> IntestinalMetabPKResult:
    """Simulate oral PK with intestinal (gut-wall) metabolism.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose (mg).
    ka_per_h:
        Absorption rate constant from gut lumen (1/h).
    fg:
        Gut wall bioavailability (fraction escaping intestinal metabolism).
        fg = 1 means no gut-wall extraction.
    fh:
        Hepatic bioavailability (fraction escaping hepatic first-pass).
    f_abs:
        Fraction of dose absorbed from GI lumen.
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    ke_portal_per_h:
        Portal-to-systemic transfer rate constant (1/h). Default 5.0/h.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    IntestinalMetabPKResult
    """
    _validate_params(dose_mg, fg, fh, cl_L_per_h, vd_L)

    ke = cl_L_per_h / vd_L  # systemic elimination rate (1/h)

    # Initial conditions: dose enters gut lumen
    a_gut = dose_mg * f_abs
    a_portal = 0.0
    c_plasma = 0.0

    times_h: list = []
    c_plasma_list: list = []
    a_gut_list: list = []
    a_portal_list: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        a_gut_list.append(a_gut)
        a_portal_list.append(a_portal)

        # ODEs
        da_gut = -ka_per_h * a_gut
        # gut wall metabolism: only fg fraction reaches portal vein
        # hepatic first-pass: only fh fraction reaches systemic
        da_portal = ka_per_h * fg * a_gut - ke_portal_per_h * a_portal
        dc_plasma = ke_portal_per_h * fh * a_portal / vd_L - ke * c_plasma

        # Forward Euler update
        a_gut = max(0.0, a_gut + da_gut * dt_h)
        a_portal = max(0.0, a_portal + da_portal * dt_h)
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)

        t = round(t + dt_h, 10)

    # PK metrics
    cmax_mg_L = max(c_plasma_list)
    tmax_idx = c_plasma_list.index(cmax_mg_L)
    tmax_h = times_h[tmax_idx]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma_list[i - 1] + c_plasma_list[i]) * (times_h[i] - times_h[i - 1])

    t_half_h = math.log(2.0) / ke
    f_oral = f_abs * fg * fh

    notes = (
        f"Intestinal metabolism PK for {drug_name}. "
        f"fg (gut wall BA): {fg:.2f}, fh (hepatic BA): {fh:.2f}, "
        f"f_abs: {f_abs:.2f}, F_oral: {f_oral:.3f}."
    )

    return IntestinalMetabPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        a_gut_mg=a_gut_list,
        a_portal_mg=a_portal_list,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        f_oral=f_oral,
        fg=fg,
        fh=fh,
        t_half_h=t_half_h,
        notes=notes,
    )


def assess_gut_wall_extraction(
    ka_per_h: float,
    clint_gut_L_per_h: float,
    fu_gut: float,
    q_gut_L_per_h: float = 18.0,
) -> dict:
    """Assess gut wall extraction ratio and fg.

    Uses the well-stirred gut model:
        fg = Q_gut / (Q_gut + fu_gut * CLint_gut)
    Equivalently:
        gut_extraction_ratio = fu_gut * CLint_gut / (Q_gut + fu_gut * CLint_gut)

    Parameters
    ----------
    ka_per_h:
        Luminal absorption rate constant (1/h). Informational; not used
        in the well-stirred model calculation but included for context.
    clint_gut_L_per_h:
        Intrinsic gut-wall metabolic clearance (L/h).
    fu_gut:
        Unbound fraction in gut tissue.
    q_gut_L_per_h:
        Intestinal blood flow rate (L/h). Default 18 L/h.

    Returns
    -------
    dict with keys:
        fg, gut_extraction_ratio, q_gut_L_per_h, fu_gut,
        clint_gut_L_per_h, ka_per_h, notes
    """
    if clint_gut_L_per_h < 0:
        raise ValueError(f"clint_gut_L_per_h must be >= 0, got {clint_gut_L_per_h}")
    if not (0.0 < fu_gut <= 1.0):
        raise ValueError(f"fu_gut must be in (0, 1], got {fu_gut}")
    if q_gut_L_per_h <= 0:
        raise ValueError(f"q_gut_L_per_h must be > 0, got {q_gut_L_per_h}")

    fu_cl = fu_gut * clint_gut_L_per_h
    fg = q_gut_L_per_h / (q_gut_L_per_h + fu_cl)
    gut_extraction_ratio = 1.0 - fg

    if gut_extraction_ratio < 0.1:
        risk = "low"
    elif gut_extraction_ratio < 0.5:
        risk = "moderate"
    else:
        risk = "high"

    notes = (
        f"Gut wall extraction assessment. "
        f"fg: {fg:.3f}, extraction ratio: {gut_extraction_ratio:.3f} ({risk} gut extraction). "
        f"Q_gut={q_gut_L_per_h:.1f} L/h, fu_gut={fu_gut:.3f}, CLint_gut={clint_gut_L_per_h:.2f} L/h."
    )

    return {
        "fg": fg,
        "gut_extraction_ratio": gut_extraction_ratio,
        "q_gut_L_per_h": q_gut_L_per_h,
        "fu_gut": fu_gut,
        "clint_gut_L_per_h": clint_gut_L_per_h,
        "ka_per_h": ka_per_h,
        "notes": notes,
    }


__all__ = [
    "IntestinalMetabPKResult",
    "simulate_intestinal_metabolism_pk",
    "assess_gut_wall_extraction",
]
