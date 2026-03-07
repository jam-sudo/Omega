"""Intramuscular depot pharmacokinetics simulation (Phase 1121).

Models drug absorption from an IM depot injection (oil-based, microsphere,
or aqueous suspension) using a 2-compartment forward Euler ODE:

  dA_depot/dt  = -k_rel * A_depot
  dC_plasma/dt =  k_rel * A_depot / Vd  -  (CL/Vd) * C_plasma

Supports flip-flop kinetics (absorption rate-limiting) typical of long-acting
depot formulations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_DEPOT_PARAMS: dict[str, dict[str, float]] = {
    "oil": {"k_rel_per_h": 0.005, "f_bioavailable": 0.95},
    "microsphere": {"k_rel_per_h": 0.02, "f_bioavailable": 0.90},
    "suspension": {"k_rel_per_h": 0.1, "f_bioavailable": 0.85},
    "aqueous": {"k_rel_per_h": 0.5, "f_bioavailable": 1.00},
}


@dataclass
class IMDepotPKResult:
    """Result of an IM depot PK simulation."""

    drug_name: str
    dose_mg: float
    depot_type: str
    times_h: list
    c_plasma_mg_L: list
    a_depot_mg: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    t_half_apparent_h: float
    depot_depleted_pct: float
    flip_flop: bool
    notes: str


def simulate_im_depot_pk(
    drug_name: str,
    dose_mg: float,
    depot_type: str = "microsphere",
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 720.0,
    dt_h: float = 1.0,
) -> IMDepotPKResult:
    """Simulate IM depot pharmacokinetics.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose in milligrams.
    depot_type:
        One of ``"oil"``, ``"microsphere"``, ``"suspension"``, ``"aqueous"``.
    cl_L_per_h:
        Systemic clearance in L/h.
    vd_L:
        Volume of distribution in litres.
    t_end_h:
        Simulation end time in hours (default 720 h = 30 days).
    dt_h:
        Forward Euler time step in hours.

    Returns
    -------
    IMDepotPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if depot_type not in _DEPOT_PARAMS:
        raise ValueError(f"depot_type must be one of {sorted(_DEPOT_PARAMS)}, got '{depot_type}'")

    params = _DEPOT_PARAMS[depot_type]
    k_rel: float = params["k_rel_per_h"]
    f_bio: float = params["f_bioavailable"]
    ke: float = cl_L_per_h / vd_L

    # Initial conditions
    a_depot: float = dose_mg * f_bio
    c_plasma: float = 0.0

    n_steps = int(round(t_end_h / dt_h)) + 1
    times_h: list[float] = []
    c_plasma_list: list[float] = []
    a_depot_list: list[float] = []

    t = 0.0
    for _ in range(n_steps):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        a_depot_list.append(a_depot)

        d_depot = -k_rel * a_depot
        d_plasma = k_rel * a_depot / vd_L - ke * c_plasma

        a_depot = max(0.0, a_depot + d_depot * dt_h)
        c_plasma = max(0.0, c_plasma + d_plasma * dt_h)
        t += dt_h

    # Derived metrics
    cmax_mg_L = max(c_plasma_list)
    tmax_h = times_h[c_plasma_list.index(cmax_mg_L)]

    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma_list[i - 1] + c_plasma_list[i]) * (times_h[i] - times_h[i - 1])

    t_half_apparent_h = math.log(2.0) / k_rel

    depot_depleted_pct = (
        (1.0 - a_depot_list[-1] / (dose_mg * f_bio)) * 100.0 if dose_mg * f_bio > 0 else 100.0
    )

    flip_flop = bool(k_rel < ke)

    notes = (
        f"k_rel={k_rel:.4f}/h, ke={ke:.4f}/h, f_bio={f_bio:.2f}; "
        f"t½_apparent={t_half_apparent_h:.1f}h; "
        f"{'flip-flop kinetics (absorption rate-limiting)' if flip_flop else 'normal kinetics (elimination rate-limiting)'}"
    )

    return IMDepotPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        depot_type=depot_type,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        a_depot_mg=a_depot_list,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        t_half_apparent_h=t_half_apparent_h,
        depot_depleted_pct=depot_depleted_pct,
        flip_flop=flip_flop,
        notes=notes,
    )


def compare_depot_types(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
) -> list[IMDepotPKResult]:
    """Simulate all 4 depot types and return results sorted by AUC (descending).

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose in milligrams.
    cl_L_per_h:
        Systemic clearance in L/h.
    vd_L:
        Volume of distribution in litres.

    Returns
    -------
    list[IMDepotPKResult]
        Four results, one per depot type, sorted by ``auc_mg_h_per_L`` descending.
    """
    results = []
    for depot in _DEPOT_PARAMS:
        r = simulate_im_depot_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            depot_type=depot,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
        )
        results.append(r)
    results.sort(key=lambda x: x.auc_mg_h_per_L, reverse=True)
    return results
