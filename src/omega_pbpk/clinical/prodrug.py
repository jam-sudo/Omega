"""Prodrug activation model — enzymatic conversion of prodrug to active drug.

2-compartment ODE system:
  Prodrug plasma (C_p) and Active drug plasma (C_a)

Oral route (ka absorption):
  dA_depot/dt = -ka * A_depot
  dC_p/dt = (ka * A_depot * f_oral / Vd_p) - (CL_p/Vd_p)*C_p - k_convert*C_p
  dC_a/dt = f_convert * k_convert * Vd_p/Vd_a * C_p - (CL_a/Vd_a)*C_a

IV route (instantaneous dose into prodrug plasma):
  C_p(0) = dose_mg / Vd_p
  dC_p/dt = -(CL_p/Vd_p)*C_p - k_convert*C_p
  dC_a/dt = f_convert * k_convert * Vd_p/Vd_a * C_p - (CL_a/Vd_a)*C_a

References
----------
- Stella VJ (2010) Prodrugs: challenges and rewards. Drugs 70(Suppl 1):1-28
- Rautio J et al. (2008) Prodrugs: design and clinical applications. Nat Rev Drug Discov 7:255-270
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from omega_pbpk._compat import np_trapz

__all__ = [
    "ProdrugResult",
    "simulate_prodrug",
    "compare_prodrug_doses",
]


@dataclass
class ProdrugResult:
    """Results from a prodrug activation simulation."""

    drug_name: str
    prodrug_name: str
    dose_mg: float
    route: str
    times_h: np.ndarray
    c_prodrug_mg_L: np.ndarray
    c_active_mg_L: np.ndarray
    cmax_prodrug: float
    cmax_active: float
    tmax_prodrug_h: float
    tmax_active_h: float
    auc_prodrug: float
    auc_active: float
    activation_efficiency: float
    notes: list[str] = field(default_factory=list)


def simulate_prodrug(
    drug_name: str,
    prodrug_name: str,
    dose_mg: float,
    route: str,
    ka_per_h: float,
    k_convert_per_h: float,
    f_convert: float,
    cl_prodrug_L_per_h: float,
    cl_active_L_per_h: float,
    vd_prodrug_L: float,
    vd_active_L: float,
    f_oral: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> ProdrugResult:
    """Simulate prodrug conversion to active drug using Forward Euler integration.

    Parameters
    ----------
    drug_name:
        Name of the active drug.
    prodrug_name:
        Name of the prodrug.
    dose_mg:
        Administered dose in mg. Must be > 0.
    route:
        Administration route: 'oral' or 'iv'.
    ka_per_h:
        First-order absorption rate constant for oral prodrug (h^-1).
    k_convert_per_h:
        First-order enzymatic conversion rate of prodrug to active drug (h^-1).
    f_convert:
        Fraction of converted prodrug that becomes active drug. Must be in [0, 1].
    cl_prodrug_L_per_h:
        Total clearance of prodrug (L/h). Must be > 0.
    cl_active_L_per_h:
        Total clearance of active drug (L/h). Must be > 0.
    vd_prodrug_L:
        Volume of distribution of prodrug (L). Must be > 0.
    vd_active_L:
        Volume of distribution of active drug (L). Must be > 0.
    f_oral:
        Oral bioavailability fraction (0-1). Default 1.0.
    t_end_h:
        Simulation end time in hours. Default 24 h.
    dt_h:
        Forward Euler time step in hours. Default 0.05 h.

    Returns
    -------
    ProdrugResult
    """
    # --- Input validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if k_convert_per_h < 0:
        raise ValueError(f"k_convert_per_h must be >= 0, got {k_convert_per_h}")
    if not (0.0 <= f_convert <= 1.0):
        raise ValueError(f"f_convert must be in [0, 1], got {f_convert}")
    if cl_prodrug_L_per_h <= 0:
        raise ValueError(f"cl_prodrug_L_per_h must be > 0, got {cl_prodrug_L_per_h}")
    if cl_active_L_per_h <= 0:
        raise ValueError(f"cl_active_L_per_h must be > 0, got {cl_active_L_per_h}")
    if vd_prodrug_L <= 0:
        raise ValueError(f"vd_prodrug_L must be > 0, got {vd_prodrug_L}")
    if vd_active_L <= 0:
        raise ValueError(f"vd_active_L must be > 0, got {vd_active_L}")

    route_lower = route.lower()
    if route_lower not in ("oral", "iv"):
        raise ValueError(f"route must be 'oral' or 'iv', got {route!r}")

    notes: list[str] = []

    # --- Setup ODE rates ---
    ke_prodrug = cl_prodrug_L_per_h / vd_prodrug_L  # elimination rate of prodrug
    ke_active = cl_active_L_per_h / vd_active_L  # elimination rate of active drug
    # Transfer rate from prodrug to active drug compartment (concentration basis)
    k_transfer = f_convert * k_convert_per_h * vd_prodrug_L / vd_active_L

    n_steps = int(round(t_end_h / dt_h)) + 1
    times = np.linspace(0.0, t_end_h, n_steps)

    c_prod = np.zeros(n_steps)
    c_act = np.zeros(n_steps)

    if route_lower == "iv":
        # Instantaneous IV bolus
        c_prod[0] = dose_mg / vd_prodrug_L
        c_act[0] = 0.0
        a_depot = 0.0
    else:
        # Oral: drug in depot (mg)
        a_depot = dose_mg * f_oral
        c_prod[0] = 0.0
        c_act[0] = 0.0

    for i in range(n_steps - 1):
        cp = c_prod[i]
        ca = c_act[i]

        if route_lower == "oral":
            abs_rate = (
                ka_per_h * a_depot / vd_prodrug_L
            )  # rate of change in prodrug conc from absorption
            a_depot_new = a_depot - ka_per_h * a_depot * dt_h
            a_depot = max(0.0, a_depot_new)
        else:
            abs_rate = 0.0

        dcp = abs_rate - (ke_prodrug + k_convert_per_h) * cp
        dca = k_transfer * cp - ke_active * ca

        c_prod[i + 1] = max(0.0, cp + dcp * dt_h)
        c_act[i + 1] = max(0.0, ca + dca * dt_h)

    # --- NCA metrics ---
    cmax_prod = float(np.max(c_prod))
    cmax_act = float(np.max(c_act))
    tmax_prod = float(times[int(np.argmax(c_prod))])
    tmax_act = float(times[int(np.argmax(c_act))])
    auc_prod = float(np_trapz(c_prod, times))
    auc_act = float(np_trapz(c_act, times))

    # Activation efficiency: ratio of active drug AUC to theoretical maximum
    theoretical_max = dose_mg * f_convert / vd_active_L if vd_active_L > 0 else 1.0
    activation_efficiency = auc_act / theoretical_max if theoretical_max > 0 else 0.0

    if f_convert == 0.0:
        notes.append("f_convert=0: no active drug formed.")
    if k_convert_per_h == 0.0:
        notes.append("k_convert_per_h=0: prodrug not converted.")

    return ProdrugResult(
        drug_name=drug_name,
        prodrug_name=prodrug_name,
        dose_mg=dose_mg,
        route=route_lower,
        times_h=times,
        c_prodrug_mg_L=c_prod,
        c_active_mg_L=c_act,
        cmax_prodrug=cmax_prod,
        cmax_active=cmax_act,
        tmax_prodrug_h=tmax_prod,
        tmax_active_h=tmax_act,
        auc_prodrug=auc_prod,
        auc_active=auc_act,
        activation_efficiency=activation_efficiency,
        notes=notes,
    )


def compare_prodrug_doses(
    drug_name: str,
    prodrug_name: str,
    doses_mg: list[float],
    route: str,
    ka_per_h: float,
    k_convert_per_h: float,
    f_convert: float,
    cl_prodrug_L_per_h: float,
    cl_active_L_per_h: float,
    vd_prodrug_L: float,
    vd_active_L: float,
    f_oral: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> list[ProdrugResult]:
    """Compare prodrug simulations across multiple doses, sorted by active drug AUC.

    Parameters
    ----------
    drug_name:
        Name of the active drug.
    prodrug_name:
        Name of the prodrug.
    doses_mg:
        List of doses (mg) to simulate.
    route:
        Administration route: 'oral' or 'iv'.
    ka_per_h:
        First-order absorption rate constant (h^-1).
    k_convert_per_h:
        Enzymatic conversion rate (h^-1).
    f_convert:
        Fraction converted to active drug.
    cl_prodrug_L_per_h:
        Prodrug clearance (L/h).
    cl_active_L_per_h:
        Active drug clearance (L/h).
    vd_prodrug_L:
        Prodrug volume of distribution (L).
    vd_active_L:
        Active drug volume of distribution (L).
    f_oral:
        Oral bioavailability fraction. Default 1.0.
    t_end_h:
        Simulation end time in hours. Default 24 h.
    dt_h:
        Forward Euler step size. Default 0.05 h.

    Returns
    -------
    list[ProdrugResult]
        Results sorted by auc_active (ascending).
    """
    results = [
        simulate_prodrug(
            drug_name=drug_name,
            prodrug_name=prodrug_name,
            dose_mg=d,
            route=route,
            ka_per_h=ka_per_h,
            k_convert_per_h=k_convert_per_h,
            f_convert=f_convert,
            cl_prodrug_L_per_h=cl_prodrug_L_per_h,
            cl_active_L_per_h=cl_active_L_per_h,
            vd_prodrug_L=vd_prodrug_L,
            vd_active_L=vd_active_L,
            f_oral=f_oral,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        for d in doses_mg
    ]
    return sorted(results, key=lambda r: r.auc_active)
