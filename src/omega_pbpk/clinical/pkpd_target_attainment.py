"""
Phase 407 — PK/PD Target Attainment Analysis

Calculate probability of target attainment (PTA) for antimicrobials
against various MIC distributions.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TargetAttainmentResult:
    """Result of PK/PD target attainment assessment."""
    drug_name: str
    pkpd_index: str          # "fT>MIC", "AUC/MIC", "Cmax/MIC"
    target_value: float      # target threshold (e.g., 40% fT>MIC, AUC/MIC=400)
    dose_mg: float
    dosing_interval_h: float
    mic_mg_L: float
    auc_24h_mg_h_per_L: float
    cmax_mg_L: float
    f_time_above_mic_pct: float  # % of dosing interval above MIC
    achieved_value: float        # actual achieved index value
    target_attained: bool
    margin_pct: float            # % above/below target (positive = achieved)
    static_breakpoint_mg_L: float  # MIC at which target is just met
    notes: str


_VALID_INDICES = {"fT>MIC", "AUC/MIC", "Cmax/MIC"}


def _validate_inputs(dose_mg, cl_L_per_h, vd_L, mic_mg_L, pkpd_index,
                     target_value, dosing_interval_h):
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if mic_mg_L <= 0:
        raise ValueError("mic_mg_L must be > 0")
    if pkpd_index not in _VALID_INDICES:
        raise ValueError(f"pkpd_index must be one of {_VALID_INDICES}")
    if target_value <= 0:
        raise ValueError("target_value must be > 0")
    if dosing_interval_h <= 0:
        raise ValueError("dosing_interval_h must be > 0")


def _compute_concentration(t, dose_mg, vd_L, ka_per_h, ke, f_oral):
    """1-compartment oral concentration at time t (single dose)."""
    if abs(ka_per_h - ke) < 1e-9:
        # Degenerate case: ka == ke, use approximation
        return f_oral * dose_mg / vd_L * ka_per_h * t * math.exp(-ke * t)
    return (f_oral * dose_mg / vd_L * ka_per_h / (ka_per_h - ke)
            * (math.exp(-ke * t) - math.exp(-ka_per_h * t)))


def _compute_cmax_tmax(dose_mg, vd_L, ka_per_h, ke, f_oral):
    """Compute Cmax and tmax analytically for 1-compartment oral."""
    if abs(ka_per_h - ke) < 1e-9:
        tmax = 1.0 / ke
    else:
        tmax = math.log(ka_per_h / ke) / (ka_per_h - ke)
    tmax = max(tmax, 0.0)
    cmax = _compute_concentration(tmax, dose_mg, vd_L, ka_per_h, ke, f_oral)
    return cmax, tmax


def _compute_f_time_above_mic(dose_mg, vd_L, ka_per_h, ke, f_oral,
                               mic_mg_L, dosing_interval_h, dt=0.01):
    """
    Compute fraction of dosing interval where C(t) >= MIC using
    numerical integration with fine time steps.
    """
    t = 0.0
    t_above = 0.0
    while t < dosing_interval_h:
        c = _compute_concentration(t, dose_mg, vd_L, ka_per_h, ke, f_oral)
        if c >= mic_mg_L:
            t_above += dt
        t += dt
    f_time_pct = t_above / dosing_interval_h * 100.0
    return min(f_time_pct, 100.0)


def _find_breakpoint_ft_mic(dose_mg, vd_L, ka_per_h, ke, f_oral,
                             dosing_interval_h, target_pct, dt=0.01):
    """
    Binary search for the MIC at which f_time_above_mic_pct = target_pct.
    """
    # Determine feasible range
    cmax, _ = _compute_cmax_tmax(dose_mg, vd_L, ka_per_h, ke, f_oral)
    if cmax <= 0:
        return 0.0

    lo = 1e-6
    hi = cmax * 2.0

    # Check if target is achievable at all
    ft_lo = _compute_f_time_above_mic(dose_mg, vd_L, ka_per_h, ke, f_oral,
                                      lo, dosing_interval_h, dt)
    if ft_lo < target_pct:
        # Even at very low MIC, target not met
        return 0.0

    ft_hi = _compute_f_time_above_mic(dose_mg, vd_L, ka_per_h, ke, f_oral,
                                      hi, dosing_interval_h, dt)
    if ft_hi >= target_pct:
        return hi

    # Binary search
    for _ in range(50):
        mid = (lo + hi) / 2.0
        ft_mid = _compute_f_time_above_mic(dose_mg, vd_L, ka_per_h, ke, f_oral,
                                           mid, dosing_interval_h, dt)
        if ft_mid >= target_pct:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < 1e-6:
            break
    return (lo + hi) / 2.0


def assess_target_attainment(
    drug_name: str,
    dose_mg: float,
    dosing_interval_h: float,
    cl_L_per_h: float,
    vd_L: float,
    mic_mg_L: float,
    pkpd_index: str = "fT>MIC",
    target_value: float = 40.0,
    ka_per_h: float = 1.5,
    f_oral: float = 1.0,
) -> TargetAttainmentResult:
    """
    Assess PK/PD target attainment for a given dose and MIC.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    dosing_interval_h : float
    cl_L_per_h : float
    vd_L : float
    mic_mg_L : float
    pkpd_index : str  "fT>MIC", "AUC/MIC", or "Cmax/MIC"
    target_value : float  threshold (40 for fT>MIC%, 400 for AUC/MIC, etc.)
    ka_per_h : float  absorption rate constant
    f_oral : float  oral bioavailability fraction

    Returns
    -------
    TargetAttainmentResult
    """
    _validate_inputs(dose_mg, cl_L_per_h, vd_L, mic_mg_L, pkpd_index,
                     target_value, dosing_interval_h)

    ke = cl_L_per_h / vd_L

    # AUC and Cmax
    auc_single = f_oral * dose_mg / cl_L_per_h
    auc_24h = auc_single * (24.0 / dosing_interval_h)

    cmax, _ = _compute_cmax_tmax(dose_mg, vd_L, ka_per_h, ke, f_oral)

    # fT>MIC
    f_time_pct = _compute_f_time_above_mic(
        dose_mg, vd_L, ka_per_h, ke, f_oral,
        mic_mg_L, dosing_interval_h
    )

    # Achieved value
    if pkpd_index == "fT>MIC":
        achieved_value = f_time_pct
    elif pkpd_index == "AUC/MIC":
        achieved_value = auc_24h / mic_mg_L
    else:  # Cmax/MIC
        achieved_value = cmax / mic_mg_L

    # Target attainment
    if pkpd_index == "fT>MIC":
        target_attained = f_time_pct >= target_value
    else:
        target_attained = achieved_value >= target_value

    margin_pct = (achieved_value - target_value) / target_value * 100.0

    # Static breakpoint
    if pkpd_index == "AUC/MIC":
        breakpoint = auc_24h / target_value
    elif pkpd_index == "Cmax/MIC":
        breakpoint = cmax / target_value
    else:  # fT>MIC
        breakpoint = _find_breakpoint_ft_mic(
            dose_mg, vd_L, ka_per_h, ke, f_oral,
            dosing_interval_h, target_value
        )

    notes = (
        f"ke={ke:.4f}/h, AUC_single={auc_single:.2f} mg*h/L, "
        f"AUC_24h={auc_24h:.2f} mg*h/L, Cmax={cmax:.3f} mg/L"
    )

    return TargetAttainmentResult(
        drug_name=drug_name,
        pkpd_index=pkpd_index,
        target_value=target_value,
        dose_mg=dose_mg,
        dosing_interval_h=dosing_interval_h,
        mic_mg_L=mic_mg_L,
        auc_24h_mg_h_per_L=auc_24h,
        cmax_mg_L=cmax,
        f_time_above_mic_pct=f_time_pct,
        achieved_value=achieved_value,
        target_attained=target_attained,
        margin_pct=margin_pct,
        static_breakpoint_mg_L=breakpoint,
        notes=notes,
    )


def screen_mic_range(
    drug_name: str,
    dose_mg: float,
    dosing_interval_h: float,
    cl_L_per_h: float,
    vd_L: float,
    pkpd_index: str = "fT>MIC",
    target_value: float = 40.0,
    mic_values: list[float] | None = None,
) -> list[TargetAttainmentResult]:
    """
    Screen a range of MIC values and return TargetAttainmentResult for each.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    dosing_interval_h : float
    cl_L_per_h : float
    vd_L : float
    pkpd_index : str
    target_value : float
    mic_values : list of float or None (default = standard doubling dilutions)

    Returns
    -------
    list of TargetAttainmentResult sorted by mic_mg_L ascending
    """
    if mic_values is None:
        mic_values = [0.03, 0.06, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]

    results = []
    for mic in sorted(mic_values):
        result = assess_target_attainment(
            drug_name=drug_name,
            dose_mg=dose_mg,
            dosing_interval_h=dosing_interval_h,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            mic_mg_L=mic,
            pkpd_index=pkpd_index,
            target_value=target_value,
        )
        results.append(result)

    return results
