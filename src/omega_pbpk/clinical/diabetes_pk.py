"""PK simulation for diabetes drugs: insulin, metformin, and GLP-1 agonists."""

from __future__ import annotations

from dataclasses import dataclass

from omega_pbpk._compat import np_trapz

# Insulin conversion: 1 unit of insulin ≈ 0.0347 mg (human insulin, MW ~5808 Da)
_UNITS_TO_MG = 0.035

# Insulin pharmacokinetic profiles (absorption rate constants per h)
_INSULIN_KA: dict[str, float] = {
    "rapid": 6.0,    # lispro/aspart: peak ~15 min
    "regular": 2.0,  # regular human: peak ~30-60 min
    "NPH": 0.5,      # intermediate-acting: peak ~4-8 h
    "glargine": 0.2, # long-acting: flat profile
}

# Threshold for "active" insulin in plasma (mU/L)
_THRESHOLD_MU_L = 0.5


@dataclass
class InsulinPKResult:
    """Result from insulin PK simulation."""

    insulin_type: str
    dose_units: float
    times_h: list
    c_plasma_mU_L: list  # mU/L
    cmax_mU_L: float
    tmax_h: float
    duration_above_threshold_h: float
    notes: str


@dataclass
class MetforminPKResult:
    """Result from metformin multi-dose PK simulation."""

    dose_mg: float
    n_doses: int
    times_h: list
    c_plasma_mg_L: list
    cmax_ss: float
    trough_ss: float
    auc_tau: float
    notes: str


def simulate_insulin_pk(
    dose_units: float,
    insulin_type: str = "rapid",
    cl_L_per_h: float = 0.8,
    vd_L: float = 12.0,
    t_end_h: float = 8.0,
    dt_h: float = 0.05,
) -> InsulinPKResult:
    """Simulate subcutaneous insulin PK using a depot-plasma 1-compartment model.

    Parameters
    ----------
    dose_units : float
        Dose in insulin units (IU). Converted to mg internally (1 IU ≈ 0.035 mg).
    insulin_type : str
        One of 'rapid', 'regular', 'NPH', 'glargine'.
    cl_L_per_h : float
        Plasma clearance (L/h). Default 0.8 L/h for typical insulin.
    vd_L : float
        Volume of distribution (L). Default 12 L.
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Time step (h).

    Returns
    -------
    InsulinPKResult
        Contains plasma concentration (mU/L), Cmax, Tmax, and duration above threshold.
    """
    if dose_units <= 0:
        raise ValueError("dose_units must be > 0")
    if insulin_type not in _INSULIN_KA:
        raise ValueError(
            f"insulin_type must be one of {list(_INSULIN_KA.keys())}, got '{insulin_type}'"
        )
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0 or dt_h > t_end_h:
        raise ValueError("dt_h must be > 0 and <= t_end_h")

    ka = _INSULIN_KA[insulin_type]
    ke = cl_L_per_h / vd_L

    # Convert dose: units → mg → IU equivalent for mU/L output
    # Plasma concentration reported as mU/L (milli-units per litre)
    # 1 unit = 1000 mU; dose_units * 1000 mU
    dose_mU = dose_units * 1000.0  # total dose in milli-units

    n_steps = max(int(round(t_end_h / dt_h)), 1)
    times_h: list[float] = []
    c_plasma: list[float] = []

    # Forward Euler: depot compartment A_depot (mU), plasma C (mU/L)
    a_depot = dose_mU
    c = 0.0

    for i in range(n_steps + 1):
        t = i * dt_h
        times_h.append(t)
        c_plasma.append(max(c, 0.0))
        if i < n_steps:
            # dA_depot/dt = -ka * A_depot
            # dC/dt = (ka * A_depot) / Vd - ke * C
            da = -ka * a_depot
            dc = (ka * a_depot) / vd_L - ke * c
            a_depot = max(a_depot + da * dt_h, 0.0)
            c = max(c + dc * dt_h, 0.0)

    cmax_mU_L = max(c_plasma)
    tmax_h = times_h[c_plasma.index(cmax_mU_L)]

    # Duration above threshold
    above = sum(1 for ci in c_plasma if ci >= _THRESHOLD_MU_L)
    duration_h = above * dt_h

    notes = (
        f"SC injection, 1-cpt depot model. ka={ka}/h, ke={ke:.3f}/h. "
        f"Threshold={_THRESHOLD_MU_L} mU/L."
    )

    return InsulinPKResult(
        insulin_type=insulin_type,
        dose_units=dose_units,
        times_h=times_h,
        c_plasma_mU_L=c_plasma,
        cmax_mU_L=cmax_mU_L,
        tmax_h=tmax_h,
        duration_above_threshold_h=duration_h,
        notes=notes,
    )


def simulate_metformin_pk(
    dose_mg: float,
    cl_L_per_h: float = 30.0,
    vd_L: float = 350.0,
    interval_h: float = 12.0,
    n_doses: int = 5,
    dt_h: float = 0.1,
) -> MetforminPKResult:
    """Simulate oral metformin PK using a 1-compartment multi-dose model.

    Parameters
    ----------
    dose_mg : float
        Dose in mg (typical: 500–2000 mg).
    cl_L_per_h : float
        Apparent oral clearance (L/h). Default 30 L/h.
    vd_L : float
        Apparent volume of distribution (L). Default 350 L (metformin distributes widely).
    interval_h : float
        Dosing interval (h). Default 12 h (twice daily).
    n_doses : int
        Number of doses to simulate.
    dt_h : float
        Time step (h).

    Returns
    -------
    MetforminPKResult
        Contains multi-dose plasma concentrations and steady-state metrics.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if interval_h <= 0:
        raise ValueError("interval_h must be > 0")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    # Metformin parameters
    ka = 0.5       # absorption rate constant /h
    f_oral = 0.55  # bioavailability (low due to absorption window)

    ke = cl_L_per_h / vd_L
    absorbed_dose_mg = dose_mg * f_oral

    t_end_h = n_doses * interval_h
    n_steps = max(int(round(t_end_h / dt_h)), 1)

    times_h: list[float] = []
    c_plasma: list[float] = []

    # Schedule dosing events
    dose_times = [i * interval_h for i in range(n_doses)]

    # Forward Euler with multiple depot boluses
    a_depot = 0.0
    c = 0.0
    dose_idx = 0

    for i in range(n_steps + 1):
        t = i * dt_h
        # Add dose if this is a dosing time (within half a step)
        if dose_idx < len(dose_times) and abs(t - dose_times[dose_idx]) < dt_h * 0.5:
            a_depot += absorbed_dose_mg
            dose_idx += 1

        times_h.append(t)
        c_plasma.append(max(c, 0.0))

        if i < n_steps:
            da = -ka * a_depot
            dc = (ka * a_depot) / vd_L - ke * c
            a_depot = max(a_depot + da * dt_h, 0.0)
            c = max(c + dc * dt_h, 0.0)

    # Steady-state metrics: use last dosing interval
    last_dose_start = (n_doses - 1) * interval_h
    last_interval_idx = [
        j for j, th in enumerate(times_h) if last_dose_start <= th <= last_dose_start + interval_h
    ]

    if last_interval_idx:
        c_last = [c_plasma[j] for j in last_interval_idx]
        t_last = [times_h[j] for j in last_interval_idx]
        cmax_ss = max(c_last)
        trough_ss = min(c_last)
        auc_tau = float(
            np_trapz(c_last, t_last)
        )
    else:
        cmax_ss = max(c_plasma)
        trough_ss = min(c_plasma)
        auc_tau = 0.0

    notes = (
        f"Oral metformin, ka={ka}/h, F={f_oral}. "
        f"ke={ke:.4f}/h, {n_doses} doses every {interval_h}h."
    )

    return MetforminPKResult(
        dose_mg=dose_mg,
        n_doses=n_doses,
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        cmax_ss=cmax_ss,
        trough_ss=trough_ss,
        auc_tau=auc_tau,
        notes=notes,
    )


__all__ = [
    "InsulinPKResult",
    "MetforminPKResult",
    "simulate_insulin_pk",
    "simulate_metformin_pk",
]
