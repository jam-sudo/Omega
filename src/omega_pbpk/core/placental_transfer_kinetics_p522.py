"""Phase 522 — Bidirectional placental drug transfer model.

Simulates maternal-fetal drug exposure via passive diffusion and active P-gp efflux.
Uses Forward Euler ODE integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PlacentalTransferResult:
    """Result from placental transfer kinetics simulation."""

    drug_name: str
    dose_mg: float
    gestational_week: int
    ps_placenta_L_per_h: float
    p_gp_efflux: float
    fup_maternal: float
    fup_fetal: float
    times_h: list[float] = field(default_factory=list)
    c_maternal_mg_L: list[float] = field(default_factory=list)
    c_fetal_mg_L: list[float] = field(default_factory=list)
    cmax_maternal: float = 0.0
    tmax_maternal_h: float = 0.0
    auc_maternal: float = 0.0
    cmax_fetal: float = 0.0
    tmax_fetal_h: float = 0.0
    auc_fetal: float = 0.0
    fetal_maternal_auc_ratio: float = 0.0
    fetal_exposure_concern: bool = False
    t_half_maternal_h: float = 0.0
    notes: str = ""


def _trapezoidal_auc(x: list[float], y: list[float]) -> float:
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_placental_transfer(
    drug_name: str,
    dose_mg: float,
    cl_maternal_L_per_h: float,
    vd_maternal_L: float,
    ps_placenta_L_per_h: float = 0.1,
    p_gp_efflux: float = 0.0,
    fup_maternal: float = 0.1,
    fup_fetal: float = 0.15,
    vd_fetal_L: float = 2.0,
    gestational_week: int = 36,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> PlacentalTransferResult:
    """Simulate bidirectional placental drug transfer.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        IV bolus dose in mg.
    cl_maternal_L_per_h:
        Maternal total clearance (L/h).
    vd_maternal_L:
        Maternal volume of distribution (L).
    ps_placenta_L_per_h:
        Placental permeability-surface area product (L/h).
    p_gp_efflux:
        P-gp efflux factor (0 = no efflux, 1 = complete block of M→F transfer).
    fup_maternal:
        Unbound fraction in maternal plasma.
    fup_fetal:
        Unbound fraction in fetal plasma.
    vd_fetal_L:
        Fetal volume of distribution (L).
    gestational_week:
        Gestational age in weeks (20–42).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Integration step size (h).

    Returns
    -------
    PlacentalTransferResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_maternal_L_per_h <= 0:
        raise ValueError("cl_maternal_L_per_h must be > 0")
    if vd_maternal_L <= 0:
        raise ValueError("vd_maternal_L must be > 0")
    if not (0.0 <= p_gp_efflux <= 1.0):
        raise ValueError("p_gp_efflux must be between 0 and 1")
    if not (20 <= gestational_week <= 42):
        raise ValueError("gestational_week must be between 20 and 42")

    # Derived constants
    k_el_m = cl_maternal_L_per_h / vd_maternal_L  # maternal elimination [1/h]
    # Fetal elimination: 20% of maternal CL/Vd
    cl_fetal = 0.20 * cl_maternal_L_per_h
    k_el_f = cl_fetal / vd_fetal_L

    # Initial conditions
    c_m = dose_mg / vd_maternal_L
    c_f = 0.0

    times_h: list[float] = []
    c_maternal_list: list[float] = []
    c_fetal_list: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for _step in range(n_steps):
        times_h.append(t)
        c_maternal_list.append(c_m)
        c_fetal_list.append(c_f)

        # Fluxes
        flux_mf = ps_placenta_L_per_h * c_m * fup_maternal * (1.0 - p_gp_efflux)
        flux_fm = ps_placenta_L_per_h * c_f * fup_fetal

        # ODEs
        dc_m = -k_el_m * c_m - flux_mf / vd_maternal_L + flux_fm / vd_maternal_L
        dc_f = flux_mf / vd_fetal_L - flux_fm / vd_fetal_L - k_el_f * c_f

        c_m = max(0.0, c_m + dc_m * dt_h)
        c_f = max(0.0, c_f + dc_f * dt_h)
        t += dt_h

    # PK metrics — maternal
    cmax_m = max(c_maternal_list)
    tmax_m = times_h[c_maternal_list.index(cmax_m)]
    auc_m = _trapezoidal_auc(times_h, c_maternal_list)

    # PK metrics — fetal
    cmax_f = max(c_fetal_list)
    tmax_f = times_h[c_fetal_list.index(cmax_f)]
    auc_f = _trapezoidal_auc(times_h, c_fetal_list)

    ratio = auc_f / auc_m if auc_m > 0.0 else 0.0

    # Maternal half-life
    t_half_m = math.log(2.0) / k_el_m if k_el_m > 0 else float("inf")

    notes = (
        f"k_el_m={k_el_m:.4f}/h; k_el_f={k_el_f:.4f}/h; "
        f"AUC_ratio={ratio:.4f}; GW={gestational_week}"
    )

    return PlacentalTransferResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        gestational_week=gestational_week,
        ps_placenta_L_per_h=ps_placenta_L_per_h,
        p_gp_efflux=p_gp_efflux,
        fup_maternal=fup_maternal,
        fup_fetal=fup_fetal,
        times_h=times_h,
        c_maternal_mg_L=c_maternal_list,
        c_fetal_mg_L=c_fetal_list,
        cmax_maternal=cmax_m,
        tmax_maternal_h=tmax_m,
        auc_maternal=auc_m,
        cmax_fetal=cmax_f,
        tmax_fetal_h=tmax_f,
        auc_fetal=auc_f,
        fetal_maternal_auc_ratio=ratio,
        fetal_exposure_concern=(ratio > 0.1),
        t_half_maternal_h=t_half_m,
        notes=notes,
    )


def compare_pgp_efflux(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_maternal_L: float,
    efflux_values: list[float],
) -> list[PlacentalTransferResult]:
    """Compare fetal exposure across different P-gp efflux levels.

    Returns results sorted by fetal_maternal_auc_ratio ascending
    (lower ratio = more fetal protection).
    """
    results = [
        simulate_placental_transfer(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_maternal_L_per_h=cl_L_per_h,
            vd_maternal_L=vd_maternal_L,
            p_gp_efflux=efflux,
        )
        for efflux in efflux_values
    ]
    return sorted(results, key=lambda r: r.fetal_maternal_auc_ratio)
