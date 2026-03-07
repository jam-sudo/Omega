"""Phase 301: Subcutaneous (SC) absorption PK model with lymphatic pathway.

Models SC drug absorption through two parallel pathways:
  1. Direct blood capillary absorption (predominant for small molecules)
  2. Lymphatic absorption (important for large molecules/biologics, >3 kDa)

ODE system (Forward Euler):
  dA_depot/dt  = -ka * A_depot
  dA_lymph/dt  = ka * lymph_fraction * A_depot - k_lymph * A_lymph
  dA_plasma/dt = ka * (1 - lymph_fraction) * A_depot + k_lymph * A_lymph
                  - (cl/vd) * A_plasma
  C_plasma = A_plasma / vd

References
----------
- Supersaxo A et al., Pharm Res. 1990;7(2):167-9 (lymphatic absorption)
- Kagan L et al., J Pharmacokinet Pharmacodyn. 2007;34(5):681-706 (SC PBPK)
- Charman WN et al., J Pharm Sci. 1986;75(8):745-53 (lymph vs blood)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "SubcutaneousPKResult",
    "simulate_sc_pk",
    "compare_formulations",
]

_K_LYMPH_PER_H = 0.1  # lymph-to-systemic drainage rate constant (1/h)


@dataclass
class SubcutaneousPKResult:
    """Results from subcutaneous PK simulation (Phase 301).

    Attributes
    ----------
    drug_name : Drug identifier.
    dose_mg : Administered SC dose (mg).
    ka_per_h : Absorption rate constant from SC depot (1/h).
    f_sc : Bioavailability fraction from SC site (0-1).
    mw_kDa : Molecular weight in kDa.
    lymph_fraction : Fraction absorbed via lymphatics (0-1).
    times_h : Simulation time points (h).
    c_plasma_mg_L : Plasma concentration time-course (mg/L).
    depot_remaining_pct : Percent of dose remaining in depot at each time.
    auc_plasma_mg_h_per_L : AUC from 0 to t_end (mg*h/L), trapezoidal.
    cmax_plasma_mg_L : Peak plasma concentration (mg/L).
    tmax_plasma_h : Time to Cmax (h).
    t_half_absorption_h : Absorption half-life = ln(2)/ka (h).
    lymphatic_contribution_pct : lymph_fraction * 100 (%).
    notes : Simulation summary.
    """

    drug_name: str
    dose_mg: float
    ka_per_h: float
    f_sc: float
    mw_kDa: float
    lymph_fraction: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    depot_remaining_pct: list[float]
    auc_plasma_mg_h_per_L: float
    cmax_plasma_mg_L: float
    tmax_plasma_h: float
    t_half_absorption_h: float
    lymphatic_contribution_pct: float
    notes: str


def _auto_lymph_fraction(mw_kDa: float) -> float:
    """Auto-compute lymph fraction from molecular weight.

    Parameters
    ----------
    mw_kDa : Molecular weight in kDa.

    Returns
    -------
    float
        Lymphatic fraction: 0.0 for mw <= 3 kDa, min(0.8, mw/30) for mw > 3 kDa.
    """
    if mw_kDa <= 3.0:
        return 0.0
    return min(0.8, mw_kDa / 30.0)


def simulate_sc_pk(
    drug_name: str,
    dose_mg: float,
    ka_per_h: float,
    f_sc: float,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float = 48.0,
    dt_h: float = 0.05,
    mw_kDa: float = 0.5,
    lymph_fraction: float | None = None,
) -> SubcutaneousPKResult:
    """Simulate subcutaneous drug absorption with lymphatic and capillary pathways.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : SC dose (mg, must be > 0).
    ka_per_h : Absorption rate constant from SC depot (1/h, must be > 0).
    f_sc : SC bioavailability (0 < f_sc <= 1).
    cl_L_per_h : Systemic clearance (L/h, must be > 0).
    vd_L : Volume of distribution (L, must be > 0).
    t_end_h : Simulation end time (h, must be > 0).
    dt_h : Forward Euler time step (h).
    mw_kDa : Molecular weight in kDa (must be > 0). Used for auto lymph_fraction.
    lymph_fraction : Lymphatic absorption fraction (0-1). If None, auto-computed
        from mw_kDa: 0.0 for mw <= 3 kDa, min(0.8, mw/30) for mw > 3 kDa.

    Returns
    -------
    SubcutaneousPKResult

    Raises
    ------
    ValueError : If any parameter is invalid.
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if ka_per_h <= 0:
        raise ValueError(f"ka_per_h must be > 0, got {ka_per_h}")
    if not (0 < f_sc <= 1):
        raise ValueError(f"f_sc must be in (0, 1], got {f_sc}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if mw_kDa <= 0:
        raise ValueError(f"mw_kDa must be > 0, got {mw_kDa}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")

    if lymph_fraction is None:
        lf = _auto_lymph_fraction(mw_kDa)
    else:
        if not (0 <= lymph_fraction <= 1):
            raise ValueError(f"lymph_fraction must be in [0, 1], got {lymph_fraction}")
        lf = float(lymph_fraction)

    ke = cl_L_per_h / vd_L

    # Initial conditions
    a_depot = dose_mg * f_sc  # effective absorbed dose in depot
    a_lymph = 0.0
    a_plasma = 0.0

    n_steps = max(int(round(t_end_h / dt_h)), 1)
    dt_actual = t_end_h / n_steps

    times: list[float] = []
    c_plasma_arr: list[float] = []
    depot_pct_arr: list[float] = []

    for i in range(n_steps + 1):
        t = i * dt_actual
        times.append(t)
        c_val = a_plasma / vd_L if vd_L > 0 else 0.0
        c_plasma_arr.append(max(0.0, c_val))
        pct = (a_depot / dose_mg) * 100.0 if dose_mg > 0 else 0.0
        depot_pct_arr.append(max(0.0, pct))

        if i < n_steps:
            # ODE rates
            d_depot = -ka_per_h * a_depot
            d_lymph = ka_per_h * lf * a_depot - _K_LYMPH_PER_H * a_lymph
            d_plasma = ka_per_h * (1.0 - lf) * a_depot + _K_LYMPH_PER_H * a_lymph - ke * a_plasma

            a_depot = max(0.0, a_depot + d_depot * dt_actual)
            a_lymph = max(0.0, a_lymph + d_lymph * dt_actual)
            a_plasma = max(0.0, a_plasma + d_plasma * dt_actual)

    # PK metrics
    cmax = max(c_plasma_arr)
    tmax_idx = c_plasma_arr.index(cmax)
    tmax = times[tmax_idx]

    # Trapezoidal AUC
    auc = sum(
        0.5 * (c_plasma_arr[i] + c_plasma_arr[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    t_half_abs = math.log(2.0) / ka_per_h if ka_per_h > 0 else float("inf")

    notes = (
        f"SC PK (Phase 301): {drug_name}, {dose_mg} mg. "
        f"f_sc={f_sc:.3f}, ka={ka_per_h:.3f}/h, MW={mw_kDa} kDa. "
        f"Lymph fraction={lf:.3f} ({lf * 100:.1f}%). "
        f"Cmax={cmax:.4g} mg/L at {tmax:.2f} h. AUC={auc:.4g} mg*h/L. "
        f"t1/2_abs={t_half_abs:.2f} h."
    )

    return SubcutaneousPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        ka_per_h=ka_per_h,
        f_sc=f_sc,
        mw_kDa=mw_kDa,
        lymph_fraction=lf,
        times_h=times,
        c_plasma_mg_L=c_plasma_arr,
        depot_remaining_pct=depot_pct_arr,
        auc_plasma_mg_h_per_L=auc,
        cmax_plasma_mg_L=cmax,
        tmax_plasma_h=tmax,
        t_half_absorption_h=t_half_abs,
        lymphatic_contribution_pct=lf * 100.0,
        notes=notes,
    )


def compare_formulations(
    drug_name: str,
    dose_mg: float,
    ka_list: list[float],
    f_sc_list: list[float],
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float = 48.0,
    dt_h: float = 0.05,
    mw_kDa: float = 0.5,
    lymph_fraction: float | None = None,
) -> list[SubcutaneousPKResult]:
    """Compare multiple SC formulations ranked by AUC descending.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : SC dose (mg).
    ka_list : List of absorption rate constants (1/h), one per formulation.
    f_sc_list : List of SC bioavailability fractions (0-1), parallel to ka_list.
    cl_L_per_h : Systemic clearance (L/h).
    vd_L : Volume of distribution (L).
    t_end_h : Simulation end time (h).
    dt_h : Forward Euler time step (h).
    mw_kDa : Molecular weight in kDa.
    lymph_fraction : Lymphatic fraction override (None = auto from mw_kDa).

    Returns
    -------
    list[SubcutaneousPKResult]
        Results sorted by auc_plasma_mg_h_per_L descending (highest AUC first).

    Raises
    ------
    ValueError : If ka_list and f_sc_list have different lengths or are empty.
    """
    if len(ka_list) != len(f_sc_list):
        raise ValueError("ka_list and f_sc_list must have the same length")
    if not ka_list:
        raise ValueError("ka_list must not be empty")

    results: list[SubcutaneousPKResult] = []
    for ka, f_sc in zip(ka_list, f_sc_list):
        r = simulate_sc_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            ka_per_h=ka,
            f_sc=f_sc,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
            mw_kDa=mw_kDa,
            lymph_fraction=lymph_fraction,
        )
        results.append(r)

    results.sort(key=lambda r: r.auc_plasma_mg_h_per_L, reverse=True)
    return results
