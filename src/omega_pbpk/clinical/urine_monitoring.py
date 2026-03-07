"""Urine drug monitoring: urinary excretion simulation and TDM."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UrineDrugResult:
    """Results from urinary excretion simulation."""

    drug_name: str
    dose_mg: float
    route: str
    cl_renal_L_per_h: float
    fe: float  # fraction excreted unchanged = cl_renal / cl_total
    times_h: list[float]
    c_plasma: list[float]  # plasma concentration (mg/L)
    c_urine_mg_L: list[float]  # instantaneous urine concentration (mg/L)
    cumulative_excreted_mg: list[float]  # cumulative drug excreted to urine (mg)
    total_excreted_mg: float
    fraction_recovered: float  # total_excreted_mg / dose_mg
    peak_urine_conc_mg_L: float
    notes: str


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_urine_drug(
    drug_name: str,
    dose_mg: float,
    cl_renal_L_per_h: float,
    cl_total_L_per_h: float,
    vd_L: float,
    urine_volume_mL_per_h: float = 60.0,
    ka_per_h: float = 1.5,
    route: str = "oral",
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> UrineDrugResult:
    """Simulate plasma PK and urinary excretion of a drug using a 1-compartment model.

    Plasma ODE (Forward Euler):
        Oral:
            dA_gut/dt  = -ka * A_gut
            dC_plasma/dt = (ka * A_gut) / Vd - (CL_total / Vd) * C_plasma
        IV:
            dC_plasma/dt = -(CL_total / Vd) * C_plasma

    Urine accumulation:
        dA_urine/dt = CL_renal * C_plasma   [mg/h]

    Urine concentration at each step:
        c_urine(t) = (CL_renal * C_plasma) / (urine_volume_mL_per_h * 1e-3)
        i.e. mg/h excreted / (L/h urine) = mg/L

    Parameters
    ----------
    drug_name : str
        Drug name.
    dose_mg : float
        Administered dose (mg). Must be > 0.
    cl_renal_L_per_h : float
        Renal clearance (L/h). Must be >= 0 and <= cl_total_L_per_h.
    cl_total_L_per_h : float
        Total systemic clearance (L/h). Must be > 0.
    vd_L : float
        Volume of distribution (L). Must be > 0.
    urine_volume_mL_per_h : float
        Urine production rate (mL/h). Must be > 0. Default 60 mL/h (1 mL/min).
    ka_per_h : float
        First-order oral absorption rate constant (1/h). Used only for oral route.
    route : str
        "oral" or "iv".
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Integration step size (h).

    Returns
    -------
    UrineDrugResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_renal_L_per_h < 0:
        raise ValueError("cl_renal_L_per_h must be >= 0")
    if cl_total_L_per_h <= 0:
        raise ValueError("cl_total_L_per_h must be > 0")
    if cl_renal_L_per_h > cl_total_L_per_h:
        raise ValueError("cl_renal_L_per_h must be <= cl_total_L_per_h")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if urine_volume_mL_per_h <= 0:
        raise ValueError("urine_volume_mL_per_h must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    route_lc = route.lower()
    if route_lc not in ("oral", "iv"):
        raise ValueError(f"route must be 'oral' or 'iv', got '{route}'")

    ke = cl_total_L_per_h / vd_L  # elimination rate constant (1/h)
    urine_vol_L_per_h = urine_volume_mL_per_h / 1000.0

    n_steps = max(int(t_end_h / dt_h), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)

    c_plasma = np.zeros(n_steps + 1)
    c_urine = np.zeros(n_steps + 1)
    cum_excreted = np.zeros(n_steps + 1)

    # Initial conditions
    if route_lc == "iv":
        c_plasma[0] = dose_mg / vd_L  # mg/L = mg / L
        a_gut = 0.0
    else:
        c_plasma[0] = 0.0
        a_gut = dose_mg  # mg in gut

    # Forward Euler integration
    for i in range(n_steps):
        cp = c_plasma[i]

        if route_lc == "oral":
            d_a_gut = -ka_per_h * a_gut * dt_h
            absorption_flux = ka_per_h * a_gut / vd_L  # mg/h per L
            a_gut = max(a_gut + d_a_gut, 0.0)
        else:
            absorption_flux = 0.0

        dcp = (absorption_flux - ke * cp) * dt_h
        c_plasma[i + 1] = max(cp + dcp, 0.0)

        # Urine excretion rate (mg/h)
        excretion_rate = cl_renal_L_per_h * cp
        cum_excreted[i + 1] = cum_excreted[i] + excretion_rate * dt_h

        # Urine concentration: excretion rate / urine flow rate
        if urine_vol_L_per_h > 0:
            c_urine[i + 1] = excretion_rate / urine_vol_L_per_h
        else:
            c_urine[i + 1] = 0.0

    # Set t=0 urine concentration from plasma at t=0
    excretion_rate_0 = cl_renal_L_per_h * c_plasma[0]
    c_urine[0] = excretion_rate_0 / urine_vol_L_per_h if urine_vol_L_per_h > 0 else 0.0

    total_excreted = float(cum_excreted[-1])
    fraction_recovered = total_excreted / dose_mg
    fe = cl_renal_L_per_h / cl_total_L_per_h
    peak_urine_conc = float(np.max(c_urine))

    notes = (
        f"1-cpt {route_lc} model; fe={fe:.3f}; "
        f"total excreted {total_excreted:.2f} mg of {dose_mg:.2f} mg dose "
        f"({fraction_recovered:.1%} recovered in urine); "
        f"peak urine conc {peak_urine_conc:.4f} mg/L."
    )

    return UrineDrugResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route_lc,
        cl_renal_L_per_h=cl_renal_L_per_h,
        fe=fe,
        times_h=times.tolist(),
        c_plasma=c_plasma.tolist(),
        c_urine_mg_L=c_urine.tolist(),
        cumulative_excreted_mg=cum_excreted.tolist(),
        total_excreted_mg=total_excreted,
        fraction_recovered=fraction_recovered,
        peak_urine_conc_mg_L=peak_urine_conc,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def urine_concentration_window(
    times_h: list[float],
    c_urine_mg_L: list[float],
    collection_window_h: tuple[float, float],
) -> dict:
    """Extract urine concentration statistics within a collection window.

    Parameters
    ----------
    times_h : list[float]
        Time points (h) from simulation.
    c_urine_mg_L : list[float]
        Urine drug concentrations (mg/L).
    collection_window_h : tuple[float, float]
        (start_h, end_h) collection window.

    Returns
    -------
    dict with keys: mean, min, max, creatinine_normalized_mg_g, n_points,
        window_start_h, window_end_h, concentrations, times.
    """
    t_arr = np.asarray(times_h, dtype=float)
    c_arr = np.asarray(c_urine_mg_L, dtype=float)

    t_start, t_end = collection_window_h
    if t_start >= t_end:
        raise ValueError("collection_window_h[0] must be < collection_window_h[1]")

    mask = (t_arr >= t_start) & (t_arr <= t_end)
    c_win = c_arr[mask]
    t_win = t_arr[mask]

    if len(c_win) == 0:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "creatinine_normalized_mg_g": 0.0,
            "n_points": 0,
            "window_start_h": t_start,
            "window_end_h": t_end,
            "concentrations": [],
            "times": [],
        }

    mean_c = float(np.mean(c_win))
    min_c = float(np.min(c_win))
    max_c = float(np.max(c_win))

    # Creatinine normalization: assume 1 g creatinine per 24 h = 1 mg creatinine/mL
    # => creatinine in window (mg) ~ (window_duration_h / 24) * 1000 mg
    # normalized = mean drug conc (mg/L) / creatinine conc in window (g/L)
    # creatinine conc = (1000 mg / 24 h) / (60 mL/h) ~ 0.6944 mg/mL = 0.6944 g/L
    creatinine_conc_g_L = (1000.0 / 24.0) / 60.0  # g/L  (1g/24h over 60 mL/h)
    creatinine_normalized = mean_c / creatinine_conc_g_L if creatinine_conc_g_L > 0 else 0.0

    return {
        "mean": mean_c,
        "min": min_c,
        "max": max_c,
        "creatinine_normalized_mg_g": creatinine_normalized,
        "n_points": int(np.sum(mask)),
        "window_start_h": t_start,
        "window_end_h": t_end,
        "concentrations": c_win.tolist(),
        "times": t_win.tolist(),
    }


def therapeutic_drug_monitoring_urine(
    drug_name: str,
    c_urine_mg_L: float,
    reference_range: tuple[float, float],
) -> dict:
    """Compare observed urine drug concentration to reference range.

    Parameters
    ----------
    drug_name : str
        Drug name.
    c_urine_mg_L : float
        Observed urine drug concentration (mg/L).
    reference_range : tuple[float, float]
        (lower, upper) reference range (mg/L).

    Returns
    -------
    dict with keys: drug_name, c_urine_mg_L, reference_low, reference_high,
        in_range, interpretation, notes.
    """
    ref_low, ref_high = reference_range
    if ref_low >= ref_high:
        raise ValueError("reference_range[0] must be < reference_range[1]")
    if c_urine_mg_L < 0:
        raise ValueError("c_urine_mg_L must be >= 0")

    in_range = bool(ref_low <= c_urine_mg_L <= ref_high)

    if c_urine_mg_L < ref_low:
        interpretation = "below_range"
        notes = (
            f"{drug_name} urine concentration ({c_urine_mg_L:.4f} mg/L) is below the reference "
            f"range ({ref_low:.4f}–{ref_high:.4f} mg/L). Possible sub-therapeutic dosing, "
            "poor compliance, or rapid metabolism."
        )
    elif c_urine_mg_L > ref_high:
        interpretation = "above_range"
        notes = (
            f"{drug_name} urine concentration ({c_urine_mg_L:.4f} mg/L) is above the reference "
            f"range ({ref_low:.4f}–{ref_high:.4f} mg/L). Possible excess dose, renal impairment, "
            "or reduced non-renal clearance."
        )
    else:
        interpretation = "within_range"
        notes = (
            f"{drug_name} urine concentration ({c_urine_mg_L:.4f} mg/L) is within the reference "
            f"range ({ref_low:.4f}–{ref_high:.4f} mg/L). Consistent with expected exposure."
        )

    return {
        "drug_name": drug_name,
        "c_urine_mg_L": c_urine_mg_L,
        "reference_low": ref_low,
        "reference_high": ref_high,
        "in_range": in_range,
        "interpretation": interpretation,
        "notes": notes,
    }


def detection_window(
    drug_name: str,
    dose_mg: float,
    cl_renal_L_per_h: float,
    cl_total_L_per_h: float,
    vd_L: float,
    lloq_urine_mg_L: float = 0.001,
) -> float:
    """Estimate how long a drug remains detectable in urine.

    Uses a 1-compartment IV-equivalent model (simplified):
        C_plasma(t) = (dose_mg / vd_L) * exp(-ke * t)
        c_urine(t)  = CL_renal * C_plasma(t) / urine_flow_rate_L_per_h

    Detection window = time when c_urine < LLOQ.

    Parameters
    ----------
    drug_name : str
        Drug name (used for validation message only).
    dose_mg : float
        Administered dose (mg). Must be > 0.
    cl_renal_L_per_h : float
        Renal clearance (L/h). Must be >= 0.
    cl_total_L_per_h : float
        Total clearance (L/h). Must be > 0.
    vd_L : float
        Volume of distribution (L). Must be > 0.
    lloq_urine_mg_L : float
        Lower limit of quantification for urine assay (mg/L). Must be > 0.

    Returns
    -------
    float
        Detection window in hours. Returns 0.0 if drug is not detectable at t=0.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_renal_L_per_h < 0:
        raise ValueError("cl_renal_L_per_h must be >= 0")
    if cl_total_L_per_h <= 0:
        raise ValueError("cl_total_L_per_h must be > 0")
    if cl_renal_L_per_h > cl_total_L_per_h:
        raise ValueError("cl_renal_L_per_h must be <= cl_total_L_per_h")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if lloq_urine_mg_L <= 0:
        raise ValueError("lloq_urine_mg_L must be > 0")

    # Default urine flow: 60 mL/h
    urine_flow_L_per_h = 0.060

    ke = cl_total_L_per_h / vd_L  # elimination rate constant (1/h)

    # At t=0: c_urine_0 = CL_renal * (dose/Vd) / urine_flow
    c_urine_0 = cl_renal_L_per_h * (dose_mg / vd_L) / urine_flow_L_per_h

    if c_urine_0 <= lloq_urine_mg_L:
        return 0.0

    # Solve: c_urine_0 * exp(-ke * t) = lloq  => t = ln(c_urine_0 / lloq) / ke
    if ke <= 0:
        return float("inf")

    t_detect = math.log(c_urine_0 / lloq_urine_mg_L) / ke
    return float(max(t_detect, 0.0))


__all__ = [
    "UrineDrugResult",
    "simulate_urine_drug",
    "urine_concentration_window",
    "therapeutic_drug_monitoring_urine",
    "detection_window",
]
