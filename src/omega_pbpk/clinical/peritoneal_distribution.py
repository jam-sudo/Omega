"""Peritoneal fluid drug distribution model.

Models drug distribution in peritoneal fluid, relevant for intraperitoneal
chemotherapy (HIPEC) and peritoneal dialysis drug removal.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PeritonealDistributionResult",
    "simulate_peritoneal_distribution",
    "compare_ip_routes",
]


@dataclass(frozen=True)
class PeritonealDistributionResult:
    drug_name: str
    dose_mg: float
    route: str
    times_h: tuple[float, ...]
    c_peritoneal_mg_L: tuple[float, ...]
    c_plasma_mg_L: tuple[float, ...]
    cmax_peritoneal: float
    tmax_peritoneal_h: float
    auc_peritoneal: float
    cmax_plasma: float
    auc_plasma: float
    peritoneal_plasma_ratio: float
    peritoneal_advantage_score: float
    notes: str


def _trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1])
    return auc


def simulate_peritoneal_distribution(
    drug_name: str,
    dose_mg: float,
    route: str = "intraperitoneal",
    v_peritoneal_L: float = 1.5,
    k_absorption_per_h: float = 0.3,
    cl_sys_L_per_h: float = 10.0,
    vd_sys_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> PeritonealDistributionResult:
    """Simulate peritoneal and plasma drug concentrations over time.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose administered in mg. Must be > 0.
    route:
        Administration route: "intraperitoneal" or "systemic".
    v_peritoneal_L:
        Volume of peritoneal fluid in litres. Must be > 0.
    k_absorption_per_h:
        First-order peritoneal absorption rate constant (h^-1).
    cl_sys_L_per_h:
        Systemic clearance in L/h. Must be > 0.
    vd_sys_L:
        Volume of distribution for systemic compartment in litres.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward Euler time step in hours.

    Returns
    -------
    PeritonealDistributionResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if v_peritoneal_L <= 0:
        raise ValueError(f"v_peritoneal_L must be > 0, got {v_peritoneal_L}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if route not in {"intraperitoneal", "systemic"}:
        raise ValueError(f"route must be 'intraperitoneal' or 'systemic', got {route!r}")

    ke_sys = cl_sys_L_per_h / vd_sys_L  # systemic elimination rate constant (h^-1)
    k_ip_transfer = 0.05  # passive transfer plasma -> peritoneal (h^-1)

    # Initial conditions
    if route == "intraperitoneal":
        a_peri = dose_mg
        a_plasma = 0.0
    else:
        a_peri = 0.0
        a_plasma = dose_mg

    times: list[float] = []
    c_peri_list: list[float] = []
    c_plasma_list: list[float] = []

    t = 0.0
    n_steps = int(t_end_h / dt_h) + 1

    for _ in range(n_steps):
        c_peri = a_peri / v_peritoneal_L
        c_plasma_val = a_plasma / vd_sys_L

        times.append(t)
        c_peri_list.append(c_peri)
        c_plasma_list.append(c_plasma_val)

        # Derivatives (Forward Euler)
        if route == "intraperitoneal":
            da_peri = -k_absorption_per_h * a_peri
            da_plasma = k_absorption_per_h * a_peri - ke_sys * a_plasma
        else:
            da_plasma = -ke_sys * a_plasma - k_ip_transfer * a_plasma
            da_peri = k_ip_transfer * a_plasma - k_absorption_per_h * a_peri

        a_peri = max(0.0, a_peri + da_peri * dt_h)
        a_plasma = max(0.0, a_plasma + da_plasma * dt_h)
        t += dt_h

    # PK metrics
    auc_peri = _trapz(times, c_peri_list)
    auc_plasma = _trapz(times, c_plasma_list)

    cmax_peri = max(c_peri_list)
    tmax_peri_h = times[c_peri_list.index(cmax_peri)]
    cmax_plasma = max(c_plasma_list)

    peritoneal_plasma_ratio = auc_peri / auc_plasma if auc_plasma > 0 else 0.0

    raw_score = peritoneal_plasma_ratio * 10.0
    peritoneal_advantage_score = max(0.0, min(100.0, raw_score))

    # Notes
    if route == "intraperitoneal" and peritoneal_plasma_ratio > 5:
        notes = (
            "IP advantage demonstrated — high local peritoneal exposure "
            "suitable for HIPEC-type therapy"
        )
    elif peritoneal_plasma_ratio > 1:
        notes = "Moderate peritoneal advantage"
    else:
        notes = "No peritoneal advantage over systemic route"

    return PeritonealDistributionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=tuple(times),
        c_peritoneal_mg_L=tuple(c_peri_list),
        c_plasma_mg_L=tuple(c_plasma_list),
        cmax_peritoneal=cmax_peri,
        tmax_peritoneal_h=tmax_peri_h,
        auc_peritoneal=auc_peri,
        cmax_plasma=cmax_plasma,
        auc_plasma=auc_plasma,
        peritoneal_plasma_ratio=peritoneal_plasma_ratio,
        peritoneal_advantage_score=peritoneal_advantage_score,
        notes=notes,
    )


def compare_ip_routes(
    drug_name: str,
    dose_mg: float,
    **kwargs,
) -> list[PeritonealDistributionResult]:
    """Compare intraperitoneal and systemic routes for the same drug and dose.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in mg.
    **kwargs:
        Additional keyword arguments forwarded to simulate_peritoneal_distribution.

    Returns
    -------
    List of 2 PeritonealDistributionResult (IP and systemic), sorted by
    auc_peritoneal descending.
    """
    ip_result = simulate_peritoneal_distribution(
        drug_name=drug_name, dose_mg=dose_mg, route="intraperitoneal", **kwargs
    )
    sys_result = simulate_peritoneal_distribution(
        drug_name=drug_name, dose_mg=dose_mg, route="systemic", **kwargs
    )
    results = [ip_result, sys_result]
    results.sort(key=lambda r: r.auc_peritoneal, reverse=True)
    return results
