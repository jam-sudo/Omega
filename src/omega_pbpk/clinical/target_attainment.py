"""PK/PD target attainment analysis — MIC-based PTA for antibiotics and other drugs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class PKPDIndex:
    """PK/PD index type."""

    name: str  # "fT>MIC", "AUC/MIC", "Cmax/MIC"
    target: float  # breakpoint value for efficacy
    description: str


# Standard PK/PD indices (literature breakpoints)
T_GREATER_MIC = PKPDIndex("fT>MIC", 40.0, "% time free drug > MIC (beta-lactams: 40-70%)")
AUC_MIC = PKPDIndex("AUC/MIC", 400.0, "24h fAUC/MIC ratio (fluoroquinolones: ≥400)")
CMAX_MIC = PKPDIndex("Cmax/MIC", 10.0, "Peak/MIC ratio (aminoglycosides: ≥10)")


@dataclass(frozen=True)
class TargetAttainmentResult:
    """Result of PK/PD target attainment analysis."""

    drug_name: str
    dose_mg: float
    interval_h: float
    mic_mg_L: float
    pkpd_index: str  # "fT>MIC", "AUC/MIC", "Cmax/MIC"
    index_value: float  # computed index value
    index_target: float  # required target
    target_attained: bool
    pta_percent: float  # probability of target attainment (from MIC distribution)
    f_t_mic_pct: float  # % dosing interval free drug > MIC
    auc24_mg_h_L: float  # 24h AUC (total)
    fauc24_mg_h_L: float  # 24h fAUC (unbound)
    cmax_mg_L: float
    fcmax_mg_L: float  # free Cmax
    times_h: list[float]
    conc_mg_L: list[float]
    free_conc_mg_L: list[float]


@dataclass(frozen=True)
class PTACurveResult:
    """PTA curve across a range of MIC values."""

    drug_name: str
    dose_mg: float
    interval_h: float
    mic_values: list[float]
    pta_values: list[float]  # PTA at each MIC (0–100%)
    pkpd_index: str
    mic90_breakpoint: float | None  # MIC where PTA drops below 90%


def _simulate_ss_profile(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    interval_h: float,
    route: str,
    fup: float,
    n_doses: int = 5,
    dt_h: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate steady-state concentration profile over one dosing interval."""
    # Simulate n_doses to reach steady state, then extract last interval
    t_total = interval_h * n_doses
    n_steps = max(int(t_total / dt_h), 1)
    t_sim = np.linspace(0.0, t_total, n_steps + 1)
    ke = cl_L_per_h / vd_L
    cp = np.zeros(n_steps + 1)

    a_central = 0.0
    a_gut = 0.0
    is_oral = route.lower() == "oral"

    dose_times = [i * interval_h for i in range(n_doses)]
    next_dose_idx = 0

    for i in range(n_steps):
        t_now = t_sim[i]
        # Administer dose at scheduled times
        if next_dose_idx < len(dose_times) and t_now >= dose_times[next_dose_idx] - dt_h / 2:
            if is_oral:
                a_gut += dose_mg
            else:
                a_central += dose_mg
            next_dose_idx += 1

        if is_oral and ka_per_h > 0:
            absorbed = ka_per_h * a_gut * dt_h
            a_gut -= absorbed
            a_central += absorbed
            a_gut = max(a_gut, 0.0)

        a_central -= ke * a_central * dt_h
        a_central = max(a_central, 0.0)
        cp[i + 1] = a_central / vd_L

    # Extract last dosing interval
    last_dose_start = (n_doses - 1) * interval_h
    mask = t_sim >= last_dose_start - dt_h / 2
    t_last = t_sim[mask] - last_dose_start
    cp_last = cp[mask]

    # Trim to one interval
    end_mask = t_last <= interval_h + dt_h
    return t_last[end_mask], cp_last[end_mask]


def compute_target_attainment(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    mic_mg_L: float,
    fup: float = 1.0,
    ka_per_h: float = 1.0,
    route: str = "oral",
    interval_h: float = 24.0,
    pkpd_index: str = "fT>MIC",
    index_target: float | None = None,
    dt_h: float = 0.05,
) -> TargetAttainmentResult:
    """Compute PK/PD target attainment for a given dose and MIC.

    Parameters
    ----------
    drug_name : str
        Drug name.
    dose_mg : float
        Single dose (mg).
    cl_L_per_h : float
        Clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    mic_mg_L : float
        Minimum inhibitory concentration (mg/L).
    fup : float
        Fraction unbound in plasma.
    ka_per_h : float
        Absorption rate (1/h), oral only.
    route : str
        "oral" or "iv".
    interval_h : float
        Dosing interval (h).
    pkpd_index : str
        "fT>MIC", "AUC/MIC", or "Cmax/MIC".
    index_target : float | None
        Required index value. None uses literature defaults.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mic_mg_L <= 0:
        raise ValueError("mic_mg_L must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    fup = float(np.clip(fup, 1e-6, 1.0))

    # Steady-state profile
    t_ss, cp_ss = _simulate_ss_profile(
        dose_mg, cl_L_per_h, vd_L, ka_per_h, interval_h, route, fup, dt_h=dt_h
    )

    # Free concentration
    cp_free = cp_ss * fup

    # PK metrics
    cmax = float(np.max(cp_ss))
    fcmax = cmax * fup
    auc24 = float(np_trapz(cp_ss, t_ss)) * (24.0 / interval_h)
    fauc24 = auc24 * fup

    # % time free drug > MIC
    n_above = int(np.sum(cp_free > mic_mg_L))
    f_t_mic = 100.0 * n_above / max(len(cp_free), 1)

    # Compute index value
    if pkpd_index == "fT>MIC":
        index_value = f_t_mic
        default_target = T_GREATER_MIC.target
    elif pkpd_index == "AUC/MIC":
        index_value = fauc24 / mic_mg_L if mic_mg_L > 0 else 0.0
        default_target = AUC_MIC.target
    elif pkpd_index == "Cmax/MIC":
        index_value = fcmax / mic_mg_L if mic_mg_L > 0 else 0.0
        default_target = CMAX_MIC.target
    else:
        raise ValueError(f"Unknown pkpd_index: {pkpd_index}. Use 'fT>MIC', 'AUC/MIC', 'Cmax/MIC'")

    target = index_target if index_target is not None else default_target
    attained = index_value >= target

    # PTA (simplified: logistic probability based on index_value/target)
    ratio = index_value / target if target > 0 else 0.0
    pta = 100.0 / (1.0 + math.exp(-5.0 * (ratio - 1.0)))

    return TargetAttainmentResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        interval_h=interval_h,
        mic_mg_L=mic_mg_L,
        pkpd_index=pkpd_index,
        index_value=index_value,
        index_target=target,
        target_attained=attained,
        pta_percent=pta,
        f_t_mic_pct=f_t_mic,
        auc24_mg_h_L=auc24,
        fauc24_mg_h_L=fauc24,
        cmax_mg_L=cmax,
        fcmax_mg_L=fcmax,
        times_h=t_ss.tolist(),
        conc_mg_L=cp_ss.tolist(),
        free_conc_mg_L=cp_free.tolist(),
    )


def compute_pta_curve(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    fup: float = 1.0,
    ka_per_h: float = 1.0,
    route: str = "oral",
    interval_h: float = 24.0,
    pkpd_index: str = "fT>MIC",
    index_target: float | None = None,
    mic_range: list[float] | None = None,
) -> PTACurveResult:
    """Compute PTA curve across a range of MIC values.

    Parameters
    ----------
    mic_range : list[float] | None
        MIC values to evaluate. Default: [0.06, 0.12, 0.25, 0.5, 1, 2, 4, 8, 16, 32] mg/L.
    """
    if mic_range is None:
        mic_range = [0.06, 0.12, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

    pta_values = []
    for mic in mic_range:
        result = compute_target_attainment(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            mic_mg_L=mic,
            fup=fup,
            ka_per_h=ka_per_h,
            route=route,
            interval_h=interval_h,
            pkpd_index=pkpd_index,
            index_target=index_target,
        )
        pta_values.append(result.pta_percent)

    # Find MIC90 breakpoint (MIC where PTA drops below 90%)
    mic90 = None
    for mic, pta in zip(mic_range, pta_values, strict=True):
        if pta >= 90.0:
            mic90 = mic

    return PTACurveResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        interval_h=interval_h,
        mic_values=list(mic_range),
        pta_values=pta_values,
        pkpd_index=pkpd_index,
        mic90_breakpoint=mic90,
    )


__all__ = [
    "PKPDIndex",
    "T_GREATER_MIC",
    "AUC_MIC",
    "CMAX_MIC",
    "TargetAttainmentResult",
    "PTACurveResult",
    "compute_target_attainment",
    "compute_pta_curve",
]
