"""Multidrug Resistance (MDR) Modulation — Phase 168.

Model the effect of MDR efflux pump inhibitors on drug intracellular
accumulation using a two-compartment (plasma/extracellular + intracellular)
exchange model solved with the Euler method.

MDR1/P-gp efflux reduces intracellular drug accumulation.  Inhibitors block
efflux and increase intracellular concentration, potentially restoring
sensitivity in resistant cells.

ODE system
----------
dc_plasma/dt = -(cl / vd) * c_plasma
               - k_in * c_plasma
               + (k_out + mdr_efflux * (1 - inhibitor_fraction)) * c_intra

dc_intra/dt  = k_in * c_plasma
               - (k_out + mdr_efflux * (1 - inhibitor_fraction)) * c_intra
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class MDRResult:
    """Results from MDR modulation simulation.

    Attributes
    ----------
    drug_name : Name of the drug.
    dose_mg : Administered dose (mg).
    inhibitor_fraction : Fraction of MDR pump inhibited (0-1).
    times_h : Stored simulation time points (h).
    c_plasma_mg_L : Plasma/extracellular concentration over time (mg/L).
    c_intracellular_mg_L : Intracellular concentration over time (mg/L).
    cmax_plasma : Peak plasma concentration (mg/L).
    cmax_intracellular : Peak intracellular concentration (mg/L).
    auc_plasma : AUC of plasma concentration (mg*h/L).
    auc_intracellular : AUC of intracellular concentration (mg*h/L).
    intracellular_to_plasma_ratio : AUC_intracellular / AUC_plasma.
    mdr_fold_change : Ratio of AUC_intracellular vs no-inhibitor baseline.
    notes : Summary notes.
    """

    drug_name: str
    dose_mg: float
    inhibitor_fraction: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_intracellular_mg_L: list[float]
    cmax_plasma: float
    cmax_intracellular: float
    auc_plasma: float
    auc_intracellular: float
    intracellular_to_plasma_ratio: float
    mdr_fold_change: float
    notes: str


# ── internal solver ─────────────────────────────────────────────────────────


def _run_euler(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    mdr_efflux_rate_per_h: float,
    k_in_per_h: float,
    k_out_per_h: float,
    inhibitor_fraction: float,
    t_end_h: float,
) -> tuple[list[float], list[float], list[float]]:
    """Euler integration returning (times, c_plasma, c_intra) stored every 0.1 h."""
    dt = 0.01
    n_steps = int(round(t_end_h / dt))
    ke = cl_L_per_h / vd_L
    effective_efflux = mdr_efflux_rate_per_h * (1.0 - inhibitor_fraction)

    c_plasma = dose_mg / vd_L
    c_intra = 0.0

    times: list[float] = []
    cp_out: list[float] = []
    ci_out: list[float] = []

    for i in range(n_steps + 1):
        if i % 10 == 0:
            times.append(round(i * dt, 6))
            cp_out.append(c_plasma)
            ci_out.append(c_intra)

        if i < n_steps:
            dc_plasma = (
                -ke * c_plasma - k_in_per_h * c_plasma + (k_out_per_h + effective_efflux) * c_intra
            )
            dc_intra = k_in_per_h * c_plasma - (k_out_per_h + effective_efflux) * c_intra
            c_plasma = max(0.0, c_plasma + dc_plasma * dt)
            c_intra = max(0.0, c_intra + dc_intra * dt)

    return times, cp_out, ci_out


# ── validation ──────────────────────────────────────────────────────────────


def _validate_common(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    mdr_efflux_rate_per_h: float,
    k_in_per_h: float,
    k_out_per_h: float,
    inhibitor_fraction: float,
    t_end_h: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if mdr_efflux_rate_per_h < 0:
        raise ValueError("mdr_efflux_rate_per_h must be >= 0")
    if k_in_per_h < 0:
        raise ValueError("k_in_per_h must be >= 0")
    if k_out_per_h < 0:
        raise ValueError("k_out_per_h must be >= 0")
    if not (0.0 <= inhibitor_fraction <= 1.0):
        raise ValueError("inhibitor_fraction must be between 0 and 1")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")


# ── public API ──────────────────────────────────────────────────────────────


def simulate_mdr_effect(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    mdr_efflux_rate_per_h: float = 0.5,
    k_in_per_h: float = 1.0,
    k_out_per_h: float = 0.3,
    inhibitor_fraction: float = 0.0,
    cell_volume_fraction: float = 0.7,  # noqa: ARG001 – reserved
    t_end_h: float = 24.0,
) -> MDRResult:
    """Simulate MDR efflux effect on intracellular drug accumulation.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose administered (mg); must be > 0.
    cl_L_per_h : float
        Systemic clearance (L/h); must be > 0.
    vd_L : float
        Volume of distribution (L); must be > 0.
    mdr_efflux_rate_per_h : float
        P-gp efflux rate constant (1/h); default 0.5.
    k_in_per_h : float
        Passive entry rate into cell (1/h); default 1.0.
    k_out_per_h : float
        Passive exit rate from cell (1/h); default 0.3.
    inhibitor_fraction : float
        Fraction of MDR pump inhibited (0-1); default 0.0.
    cell_volume_fraction : float
        Reserved parameter (unused); default 0.7.
    t_end_h : float
        Simulation duration (h); default 24.0.

    Returns
    -------
    MDRResult

    Raises
    ------
    ValueError
        If any numeric constraint is violated.
    """
    _validate_common(
        dose_mg,
        cl_L_per_h,
        vd_L,
        mdr_efflux_rate_per_h,
        k_in_per_h,
        k_out_per_h,
        inhibitor_fraction,
        t_end_h,
    )

    times, cp, ci = _run_euler(
        dose_mg,
        cl_L_per_h,
        vd_L,
        mdr_efflux_rate_per_h,
        k_in_per_h,
        k_out_per_h,
        inhibitor_fraction,
        t_end_h,
    )

    t_arr = np.asarray(times)
    cp_arr = np.asarray(cp)
    ci_arr = np.asarray(ci)

    auc_plasma = float(np_trapz(cp_arr, t_arr))
    auc_intra = float(np_trapz(ci_arr, t_arr))

    # Fold-change vs uninhibited baseline
    if inhibitor_fraction == 0.0:
        mdr_fold_change = 1.0
    else:
        _, _, ci_base = _run_euler(
            dose_mg,
            cl_L_per_h,
            vd_L,
            mdr_efflux_rate_per_h,
            k_in_per_h,
            k_out_per_h,
            0.0,
            t_end_h,
        )
        base_times, _, _ = _run_euler(
            dose_mg,
            cl_L_per_h,
            vd_L,
            mdr_efflux_rate_per_h,
            k_in_per_h,
            k_out_per_h,
            0.0,
            t_end_h,
        )
        auc_base = float(np_trapz(np.asarray(ci_base), np.asarray(base_times)))
        mdr_fold_change = auc_intra / auc_base if auc_base > 0 else float("inf")

    ratio = auc_intra / auc_plasma if auc_plasma > 0 else 0.0

    notes = (
        f"MDR modulation for {drug_name}: "
        f"inhibitor_fraction={inhibitor_fraction:.2f}, "
        f"fold_change={mdr_fold_change:.2f}."
    )

    return MDRResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        inhibitor_fraction=inhibitor_fraction,
        times_h=times,
        c_plasma_mg_L=cp,
        c_intracellular_mg_L=ci,
        cmax_plasma=float(cp_arr.max()),
        cmax_intracellular=float(ci_arr.max()),
        auc_plasma=auc_plasma,
        auc_intracellular=auc_intra,
        intracellular_to_plasma_ratio=ratio,
        mdr_fold_change=mdr_fold_change,
        notes=notes,
    )


def screen_mdr_inhibitors(
    drug_name: str,
    dose_mg: float,
    inhibitors: list[dict],
    cl_L_per_h: float = 5.0,
    vd_L: float = 70.0,
    mdr_efflux_rate_per_h: float = 0.5,
    k_in_per_h: float = 1.0,
    k_out_per_h: float = 0.3,
    t_end_h: float = 24.0,
) -> list[MDRResult]:
    """Screen a panel of MDR inhibitors for their effect on intracellular accumulation.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose administered (mg); must be > 0.
    inhibitors : list[dict]
        Each dict must have ``inhibitor_name`` (str) and ``inhibitor_fraction``
        (float, 0-1).
    cl_L_per_h, vd_L, mdr_efflux_rate_per_h, k_in_per_h, k_out_per_h, t_end_h
        PK / model parameters forwarded to :func:`simulate_mdr_effect`.

    Returns
    -------
    list[MDRResult]
        Sorted by ``intracellular_to_plasma_ratio`` descending.

    Raises
    ------
    ValueError
        If *inhibitors* is empty or any numeric constraint is violated.
    """
    if not inhibitors:
        raise ValueError("inhibitors list must not be empty")

    results: list[MDRResult] = []
    for inh in inhibitors:
        res = simulate_mdr_effect(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            mdr_efflux_rate_per_h=mdr_efflux_rate_per_h,
            k_in_per_h=k_in_per_h,
            k_out_per_h=k_out_per_h,
            inhibitor_fraction=inh["inhibitor_fraction"],
            t_end_h=t_end_h,
        )
        results.append(res)

    results.sort(key=lambda r: r.intracellular_to_plasma_ratio, reverse=True)
    return results
