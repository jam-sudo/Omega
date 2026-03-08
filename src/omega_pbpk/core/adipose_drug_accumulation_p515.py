"""Phase 515 — Drug Accumulation in Adipose (Extended).

Extended adipose accumulation model with washout kinetics and obesity effects.
3-compartment: plasma, lean tissue, adipose — Forward Euler ODE.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AdiposeAccumulationResult:
    """Result of adipose drug accumulation simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    weight_kg: float
    f_adipose: float

    kp_adipose: float
    kp_lean: float

    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_lean_mg_L: list = field(default_factory=list)
    c_adipose_mg_L: list = field(default_factory=list)

    cmax_plasma: float = 0.0
    cmax_lean: float = 0.0
    cmax_adipose: float = 0.0

    auc_plasma: float = 0.0
    auc_lean: float = 0.0
    auc_adipose: float = 0.0

    adipose_to_plasma_auc_ratio: float = 0.0
    accumulation_concern: bool = False

    washout_t50_h: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Helper: trapezoidal AUC
# ---------------------------------------------------------------------------


def _trapz(x: list, y: list) -> float:
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_adipose_accumulation(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_plasma_L_per_h: float,
    vd_plasma_L: float,
    vd_lean_L: float = 30.0,
    vd_adipose_L: float = 15.0,
    weight_kg: float = 70.0,
    f_adipose: float = 0.25,
    t_end_h: float = 72.0,
    dt_h: float = 0.2,
) -> AdiposeAccumulationResult:
    """Simulate adipose drug accumulation with washout kinetics.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        IV bolus dose in mg.
    logP:
        Octanol-water partition coefficient.
    cl_plasma_L_per_h:
        Total plasma clearance (L/h).
    vd_plasma_L:
        Volume of distribution for plasma compartment (L).
    vd_lean_L:
        Volume of lean tissue compartment (L).
    vd_adipose_L:
        Volume of adipose compartment (L).
    weight_kg:
        Body weight in kg.
    f_adipose:
        Adipose fraction of body weight (default 0.25).
    t_end_h:
        Simulation duration in hours.
    dt_h:
        Forward Euler time step in hours.

    Returns
    -------
    AdiposeAccumulationResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_plasma_L_per_h <= 0:
        raise ValueError("cl_plasma_L_per_h must be positive")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be positive")

    # Partition coefficients
    kp_adipose = 2.0 ** (logP - 1.0) if logP > 1.0 else 1.0
    kp_adipose = max(0.5, min(100.0, kp_adipose))
    kp_lean = 0.5 + 0.1 * max(0.0, logP)

    # Transfer rate constants (h^-1)
    k_el = cl_plasma_L_per_h / vd_plasma_L
    k_12 = 0.5  # plasma → lean
    k_21 = 0.5 / kp_lean  # lean → plasma
    k_13 = 0.2  # plasma → adipose
    k_31 = 0.2 / kp_adipose  # adipose → plasma

    # State variables: amounts in mg
    a_plasma = dose_mg
    a_lean = 0.0
    a_adipose = 0.0

    # Collect time series
    n_steps = int(round(t_end_h / dt_h)) + 1
    times: list = []
    cp: list = []
    cl_ts: list = []
    ca: list = []

    t = 0.0
    for _ in range(n_steps):
        times.append(t)
        cp.append(a_plasma / vd_plasma_L)
        cl_ts.append(a_lean / vd_lean_L)
        ca.append(a_adipose / vd_adipose_L)

        # Forward Euler
        da_plasma = -(k_el + k_12 + k_13) * a_plasma + k_21 * a_lean + k_31 * a_adipose
        da_lean = k_12 * a_plasma - k_21 * a_lean
        da_adipose = k_13 * a_plasma - k_31 * a_adipose

        a_plasma = max(0.0, a_plasma + da_plasma * dt_h)
        a_lean = max(0.0, a_lean + da_lean * dt_h)
        a_adipose = max(0.0, a_adipose + da_adipose * dt_h)

        t += dt_h

    # PK metrics
    cmax_plasma = max(cp)
    cmax_lean = max(cl_ts)
    cmax_adipose = max(ca)

    auc_plasma = _trapz(times, cp)
    auc_lean = _trapz(times, cl_ts)
    auc_adipose = _trapz(times, ca)

    adipose_to_plasma_auc_ratio = auc_adipose / auc_plasma if auc_plasma > 0 else 0.0
    accumulation_concern = kp_adipose > 5.0

    # Washout t50: time for adipose concentration to drop to 50% of Cmax after peak
    washout_t50_h = 0.0
    if cmax_adipose > 0:
        idx_peak = ca.index(cmax_adipose)
        half_target = cmax_adipose * 0.5
        for i in range(idx_peak + 1, len(ca)):
            if ca[i] <= half_target:
                # Linear interpolation
                frac = (ca[i - 1] - half_target) / (ca[i - 1] - ca[i] + 1e-30)
                washout_t50_h = times[i - 1] + frac * (times[i] - times[i - 1]) - times[idx_peak]
                break

    notes = (
        f"kp_adipose={kp_adipose:.2f}, kp_lean={kp_lean:.2f}; "
        f"accumulation_concern={'Yes' if accumulation_concern else 'No'}"
    )

    return AdiposeAccumulationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        weight_kg=weight_kg,
        f_adipose=f_adipose,
        kp_adipose=kp_adipose,
        kp_lean=kp_lean,
        times_h=times,
        c_plasma_mg_L=cp,
        c_lean_mg_L=cl_ts,
        c_adipose_mg_L=ca,
        cmax_plasma=cmax_plasma,
        cmax_lean=cmax_lean,
        cmax_adipose=cmax_adipose,
        auc_plasma=auc_plasma,
        auc_lean=auc_lean,
        auc_adipose=auc_adipose,
        adipose_to_plasma_auc_ratio=adipose_to_plasma_auc_ratio,
        accumulation_concern=accumulation_concern,
        washout_t50_h=washout_t50_h,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------


def compare_logp_adipose(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_plasma_L: float,
    logp_values: list,
) -> list:
    """Compare adipose accumulation across different logP values.

    Returns a list of :class:`AdiposeAccumulationResult` sorted by
    *adipose_to_plasma_auc_ratio* in descending order.
    """
    results = [
        simulate_adipose_accumulation(
            drug_name=drug_name,
            dose_mg=dose_mg,
            logP=lp,
            cl_plasma_L_per_h=cl_L_per_h,
            vd_plasma_L=vd_plasma_L,
        )
        for lp in logp_values
    ]
    results.sort(key=lambda r: r.adipose_to_plasma_auc_ratio, reverse=True)
    return results
