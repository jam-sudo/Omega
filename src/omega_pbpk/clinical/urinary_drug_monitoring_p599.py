"""Phase 599 — Urinary Drug Monitoring.

Model drug and metabolite concentrations in urine for therapeutic drug
monitoring using a 1-compartment oral PK model with renal excretion.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UrinaryDrugMonitoringResult:
    """Result of urinary drug monitoring simulation."""

    drug_name: str
    dose_mg: float
    times_h: list  # collection time points (h)
    urine_conc_mg_L: list  # drug concentration in urine at each time (mg/L)
    cumulative_excreted_mg: list  # cumulative drug excreted (mg)
    urine_flow_mL_per_h: float  # urine flow rate used (mL/h)
    fe_urine: float  # fraction excreted unchanged
    t_detection_h: float  # first time urine conc >= LOD (0.001 mg/L)
    t_clearance_h: float  # last time urine conc >= LOD (0.001 mg/L)
    peak_urine_conc_mg_L: float
    tpeak_urine_h: float
    auc_urine_mg_h_per_L: float
    renal_cl_L_per_h: float
    notes: str


def _trapz(x: list, y: list) -> float:
    """Manual trapezoidal integration."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_urinary_monitoring(
    drug_name: str,
    dose_mg: float,
    cl_renal_L_per_h: float,
    vd_L: float,
    ka_per_h: float = 1.2,
    f_oral: float = 0.85,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
    urine_flow_mL_per_h: float = 60.0,
) -> UrinaryDrugMonitoringResult:
    """Simulate drug concentration in urine over time.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Oral dose in mg.
    cl_renal_L_per_h : float
        Renal clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    ka_per_h : float
        First-order absorption rate constant (per h).
    f_oral : float
        Oral bioavailability fraction (0, 1].
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Forward Euler step size (h).
    urine_flow_mL_per_h : float
        Urine flow rate (mL/h), used to compute urine concentration.
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_renal_L_per_h <= 0:
        raise ValueError("cl_renal_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if urine_flow_mL_per_h <= 0:
        raise ValueError("urine_flow_mL_per_h must be > 0")
    if not (0 < f_oral <= 1.0):
        raise ValueError("f_oral must be in (0, 1]")

    # Total clearance: assume renal fraction is 40%
    # cl_total = cl_renal / renal_fraction  with renal_fraction = 0.4
    # clamp: cl_total >= cl_renal
    renal_fraction = 0.4
    cl_total = cl_renal_L_per_h / renal_fraction
    cl_total = max(cl_total, cl_renal_L_per_h)

    # Urine flow in L/h for concentration calculation
    urine_flow_L_per_h = urine_flow_mL_per_h / 1000.0

    n = max(int(round(t_end_h / dt_h)), 1)
    times = [i * dt_h for i in range(n + 1)]

    # State: A_gut (mg absorbed, pre-systemic), C_plasma (mg/L)
    a_gut = dose_mg * f_oral
    c_plasma = 0.0

    c_plasma_arr = [0.0] * (n + 1)
    c_plasma_arr[0] = c_plasma

    for i in range(n):
        abs_rate = ka_per_h * a_gut
        a_gut = max(a_gut - abs_rate * dt_h, 0.0)
        dc = abs_rate / vd_L - (cl_total / vd_L) * c_plasma_arr[i]
        c_plasma_arr[i + 1] = max(c_plasma_arr[i] + dc * dt_h, 0.0)

    # Urine concentration at each time point:
    # C_urine [mg/L] = CL_renal [L/h] * C_plasma [mg/L] / urine_flow [L/h]
    LOD = 0.001  # mg/L
    urine_conc = [cl_renal_L_per_h * c_plasma_arr[i] / urine_flow_L_per_h for i in range(n + 1)]

    # Cumulative excreted: running sum of CL_renal * C_plasma * dt
    cumulative = [0.0] * (n + 1)
    for i in range(1, n + 1):
        delta = cl_renal_L_per_h * c_plasma_arr[i] * dt_h
        cumulative[i] = cumulative[i - 1] + delta

    fe_urine = min(cumulative[-1] / dose_mg, 1.0)

    # Detection/clearance times
    t_detection_h = t_end_h  # default: never detected
    t_clearance_h = 0.0  # default: never above LOD

    for i in range(n + 1):
        if urine_conc[i] >= LOD:
            if times[i] < t_detection_h:
                t_detection_h = times[i]
            t_clearance_h = times[i]

    # Peak urine concentration
    peak_urine_conc = max(urine_conc)
    tpeak_idx = urine_conc.index(peak_urine_conc)
    tpeak_urine_h = times[tpeak_idx]

    auc_urine = _trapz(times, urine_conc)

    notes = (
        f"CL_total={cl_total:.3f} L/h (renal fraction assumed 40%). "
        f"CL_renal={cl_renal_L_per_h:.3f} L/h. "
        f"Detection window: {t_detection_h:.1f}–{t_clearance_h:.1f} h. "
        f"fe_urine={fe_urine:.3f}."
    )

    return UrinaryDrugMonitoringResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        urine_conc_mg_L=urine_conc,
        cumulative_excreted_mg=cumulative,
        urine_flow_mL_per_h=urine_flow_mL_per_h,
        fe_urine=fe_urine,
        t_detection_h=t_detection_h,
        t_clearance_h=t_clearance_h,
        peak_urine_conc_mg_L=peak_urine_conc,
        tpeak_urine_h=tpeak_urine_h,
        auc_urine_mg_h_per_L=auc_urine,
        renal_cl_L_per_h=cl_renal_L_per_h,
        notes=notes,
    )


def detection_window(
    drug_name: str,
    dose_mg: float,
    cl_renal_L_per_h: float,
    vd_L: float,
) -> dict:
    """Return detection window metrics for a drug in urine.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    cl_renal_L_per_h : float
    vd_L : float

    Returns
    -------
    dict with keys: t_detection_h, t_clearance_h, window_h
    """
    result = simulate_urinary_monitoring(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_renal_L_per_h=cl_renal_L_per_h,
        vd_L=vd_L,
    )
    window_h = max(result.t_clearance_h - result.t_detection_h, 0.0)
    return {
        "t_detection_h": result.t_detection_h,
        "t_clearance_h": result.t_clearance_h,
        "window_h": window_h,
    }


__all__ = [
    "UrinaryDrugMonitoringResult",
    "simulate_urinary_monitoring",
    "detection_window",
]
