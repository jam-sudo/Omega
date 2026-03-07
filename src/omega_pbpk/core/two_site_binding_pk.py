"""
Phase 950 — Two-Site Receptor Binding PK Model
Drug binds to two receptor populations (high-affinity/low-capacity site 1,
low-affinity/high-capacity site 2) with Forward Euler ODE integration.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TwoSiteBindingResult", "simulate_two_site_binding", "optimize_selectivity"]


@dataclass
class TwoSiteBindingResult:
    drug_name: str
    dose_mg: float
    times_h: list
    c_free_nM: list
    c_bound_r1_nM: list
    c_bound_r2_nM: list
    peak_free_nM: float
    peak_r1_occupancy: float
    peak_r2_occupancy: float
    selectivity_index: float
    auc_free_nM_h: float
    time_above_kd1_h: float
    notes: str


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def simulate_two_site_binding(
    drug_name: str,
    dose_mg: float,
    mw_Da: float = 300.0,
    kd1_nM: float = 1.0,
    kd2_nM: float = 100.0,
    r1_total_nM: float = 10.0,
    r2_total_nM: float = 1000.0,
    kon1_per_nM_per_h: float = 0.1,
    kon2_per_nM_per_h: float = 0.01,
    cl_free_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> TwoSiteBindingResult:
    """Simulate two-site receptor binding PK via forward Euler integration."""

    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if kd1_nM <= 0:
        raise ValueError("kd1_nM must be > 0")
    if kd2_nM <= 0:
        raise ValueError("kd2_nM must be > 0")
    if cl_free_L_per_h <= 0:
        raise ValueError("cl_free_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    # Derived rate constants
    koff1 = kon1_per_nM_per_h * kd1_nM  # /h
    koff2 = kon2_per_nM_per_h * kd2_nM  # /h

    # Elimination rate constant (first-order from free drug)
    ke = cl_free_L_per_h / vd_L  # /h

    # Initial free concentration (IV bolus): convert dose_mg to nM
    # dose_mg [mg] / mw_Da [g/mol] = mmol; * 1e6 -> nmol; / vd_L [L] -> nM
    c_free_init = (dose_mg / mw_Da) * 1e6 / vd_L  # nM

    # State variables
    c_free = c_free_init
    c_r1 = 0.0
    c_r2 = 0.0

    times = []
    c_free_list = []
    c_r1_list = []
    c_r2_list = []

    n_steps = max(1, int(round(t_end_h / dt_h)))
    t = 0.0

    for i in range(n_steps + 1):
        times.append(t)
        c_free_list.append(c_free)
        c_r1_list.append(c_r1)
        c_r2_list.append(c_r2)

        if i < n_steps:
            # Free receptor concentrations
            r1_free = max(0.0, r1_total_nM - c_r1)
            r2_free = max(0.0, r2_total_nM - c_r2)

            # ODEs
            dcfree_dt = (
                -kon1_per_nM_per_h * c_free * r1_free
                + koff1 * c_r1
                - kon2_per_nM_per_h * c_free * r2_free
                + koff2 * c_r2
                - ke * c_free
            )
            dcr1_dt = kon1_per_nM_per_h * c_free * r1_free - koff1 * c_r1
            dcr2_dt = kon2_per_nM_per_h * c_free * r2_free - koff2 * c_r2

            # Forward Euler
            c_free = max(0.0, c_free + dcfree_dt * dt_h)
            c_r1 = _clamp(c_r1 + dcr1_dt * dt_h, 0.0, r1_total_nM)
            c_r2 = _clamp(c_r2 + dcr2_dt * dt_h, 0.0, r2_total_nM)
            t += dt_h

    # Metrics
    peak_free_nM = max(c_free_list)
    peak_r1_raw = max(c_r1_list)
    peak_r2_raw = max(c_r2_list)
    peak_r1_occupancy = _clamp(peak_r1_raw / r1_total_nM, 0.0, 1.0)
    peak_r2_occupancy = _clamp(peak_r2_raw / r2_total_nM, 0.0, 1.0)
    selectivity_index = peak_r1_occupancy / (peak_r2_occupancy + 0.001)

    # Trapezoidal AUC of free drug
    auc = 0.0
    for j in range(1, len(times)):
        auc += 0.5 * (c_free_list[j - 1] + c_free_list[j]) * (times[j] - times[j - 1])

    # Time above kd1
    t_above = sum(dt_h for j in range(len(c_free_list)) if c_free_list[j] > kd1_nM)

    notes = (
        f"IV bolus {dose_mg} mg; C0={c_free_init:.2f} nM; "
        f"ke={ke:.3f}/h; selectivity={selectivity_index:.2f}; "
        f"peak_R1_occ={peak_r1_occupancy:.3f}; peak_R2_occ={peak_r2_occupancy:.3f}"
    )

    return TwoSiteBindingResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_free_nM=c_free_list,
        c_bound_r1_nM=c_r1_list,
        c_bound_r2_nM=c_r2_list,
        peak_free_nM=peak_free_nM,
        peak_r1_occupancy=peak_r1_occupancy,
        peak_r2_occupancy=peak_r2_occupancy,
        selectivity_index=selectivity_index,
        auc_free_nM_h=auc,
        time_above_kd1_h=t_above,
        notes=notes,
    )


def optimize_selectivity(
    drug_name: str,
    mw_Da: float,
    dose_range_mg: list | None = None,
    kd1_nM: float = 1.0,
    kd2_nM: float = 100.0,
    r1_total_nM: float = 10.0,
    r2_total_nM: float = 1000.0,
    kon1_per_nM_per_h: float = 0.1,
    kon2_per_nM_per_h: float = 0.01,
    cl_free_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> list[TwoSiteBindingResult]:
    """Simulate across dose range and sort by selectivity_index descending."""
    if dose_range_mg is None:
        dose_range_mg = [1.0, 5.0, 10.0, 50.0, 100.0]

    results = []
    for dose in dose_range_mg:
        res = simulate_two_site_binding(
            drug_name=drug_name,
            dose_mg=dose,
            mw_Da=mw_Da,
            kd1_nM=kd1_nM,
            kd2_nM=kd2_nM,
            r1_total_nM=r1_total_nM,
            r2_total_nM=r2_total_nM,
            kon1_per_nM_per_h=kon1_per_nM_per_h,
            kon2_per_nM_per_h=kon2_per_nM_per_h,
            cl_free_L_per_h=cl_free_L_per_h,
            vd_L=vd_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(res)

    results.sort(key=lambda r: r.selectivity_index, reverse=True)
    return results
