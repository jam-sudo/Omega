"""Phase 262 — Adipose tissue drug accumulation PK model.

Models drug accumulation in adipose tissue over long-term dosing using
a 2-compartment (plasma + adipose) model with Forward Euler integration.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "AdiposeAccumulationResult",
    "simulate_adipose_accumulation",
    "compare_logp_accumulation",
]


def _trapz(times: list, values: list) -> float:
    """Trapezoidal integration."""
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


@dataclass
class AdiposeAccumulationResult:
    """Result of adipose drug accumulation simulation."""

    drug_name: str
    dose_mg: float
    logp: float
    dosing_interval_h: float
    n_doses: int
    times_h: list
    c_plasma_mg_L: list
    c_adipose_mg_L: list
    cmax_plasma_steady: float
    css_plasma_avg: float
    cmax_adipose_steady: float
    css_adipose_avg: float
    accumulation_ratio_plasma: float
    accumulation_ratio_adipose: float
    washout_t50_h: float
    notes: str


def simulate_adipose_accumulation(
    drug_name: str,
    dose_mg: float,
    logp: float,
    dosing_interval_h: float,
    n_doses: int,
    cl_L_per_h: float,
    dt_h: float = 0.1,
) -> AdiposeAccumulationResult:
    """Simulate drug accumulation in adipose tissue over multiple doses.

    Uses a 2-compartment Forward Euler ODE:
        dC_plasma/dt = -(ke + k_pa)*C_plasma + k_ap*C_adipose*V_adi/V_plasma
        dC_adipose/dt = k_pa*C_plasma*V_plasma/V_adi - k_ap*C_adipose

    Dosing events add dose/V_plasma to C_plasma at t = i * dosing_interval_h.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in mg. Must be > 0.
    logp:
        logP value (octanol-water). Determines adipose volume and transfer rate.
    dosing_interval_h:
        Time between doses in hours. Must be > 0.
    n_doses:
        Number of doses. Must be >= 1.
    cl_L_per_h:
        Plasma clearance (L/h). Must be > 0.
    dt_h:
        Integration step size (hours).

    Returns
    -------
    AdiposeAccumulationResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")
    if dosing_interval_h <= 0:
        raise ValueError("dosing_interval_h must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")

    # Compartment volumes
    v_plasma = 3.0  # L
    logp_pos = max(0.0, logp)
    v_adi = max(5.0, 10.0 * logp_pos * 0.5)  # L; lipophilic drugs → larger

    # Rate constants
    logp_factor = max(0.1, logp)
    k_pa = logp_factor * 0.1  # plasma → adipose (/h)
    k_ap = 0.02  # adipose → plasma (/h); slow release
    ke = cl_L_per_h / v_plasma  # elimination (/h)

    # Washout half-life for adipose
    washout_t50_h = 0.693 / k_ap

    # Simulation
    t_end = n_doses * dosing_interval_h
    n_steps = int(t_end / dt_h) + 1

    times_h: list = []
    c_plasma_mg_L: list = []
    c_adipose_mg_L: list = []

    c_p = 0.0  # mg/L plasma concentration
    c_a = 0.0  # mg/L adipose concentration

    # Pre-compute dose event times (indices)
    dose_times = set()
    for i in range(n_doses):
        idx = round(i * dosing_interval_h / dt_h)
        dose_times.add(idx)

    t = 0.0
    for step in range(n_steps):
        # Apply dose at the start of each dosing interval
        if step in dose_times:
            c_p += dose_mg / v_plasma

        times_h.append(t)
        c_plasma_mg_L.append(c_p)
        c_adipose_mg_L.append(c_a)

        # Forward Euler
        dc_p = -(ke + k_pa) * c_p + k_ap * c_a * (v_adi / v_plasma)
        dc_a = k_pa * c_p * (v_plasma / v_adi) - k_ap * c_a

        c_p = max(0.0, c_p + dc_p * dt_h)
        c_a = max(0.0, c_a + dc_a * dt_h)
        t += dt_h

    # First-dose Cmax (plasma): max over first interval
    n_first_interval = min(int(dosing_interval_h / dt_h), len(c_plasma_mg_L))
    cmax_plasma_dose1 = (
        max(c_plasma_mg_L[:n_first_interval]) if n_first_interval > 0 else c_plasma_mg_L[0]
    )

    # Steady-state metrics: last dosing interval
    last_start_idx = round((n_doses - 1) * dosing_interval_h / dt_h)
    last_end_idx = min(len(c_plasma_mg_L), last_start_idx + int(dosing_interval_h / dt_h) + 1)

    ss_plasma = c_plasma_mg_L[last_start_idx:last_end_idx]
    ss_adipose = c_adipose_mg_L[last_start_idx:last_end_idx]
    ss_times = times_h[last_start_idx:last_end_idx]

    cmax_plasma_steady = max(ss_plasma) if ss_plasma else 0.0
    cmax_adipose_steady = max(ss_adipose) if ss_adipose else 0.0

    if len(ss_times) >= 2 and (ss_times[-1] - ss_times[0]) > 0:
        css_plasma_avg = _trapz(ss_times, ss_plasma) / (ss_times[-1] - ss_times[0])
        css_adipose_avg = _trapz(ss_times, ss_adipose) / (ss_times[-1] - ss_times[0])
    else:
        css_plasma_avg = ss_plasma[0] if ss_plasma else 0.0
        css_adipose_avg = ss_adipose[0] if ss_adipose else 0.0

    # Accumulation ratios
    if cmax_plasma_dose1 > 0:
        accumulation_ratio_plasma = cmax_plasma_steady / cmax_plasma_dose1
    else:
        accumulation_ratio_plasma = 1.0

    cmax_adipose_dose1 = max(c_adipose_mg_L[:n_first_interval]) if n_first_interval > 0 else 0.0
    if cmax_adipose_dose1 > 0:
        accumulation_ratio_adipose = cmax_adipose_steady / cmax_adipose_dose1
    else:
        # If no accumulation yet in first interval, use ratio relative to dose 2 or fallback
        accumulation_ratio_adipose = max(1.0, cmax_adipose_steady)

    notes = (
        f"logP={logp:.2f}, V_plasma={v_plasma:.1f}L, V_adipose={v_adi:.1f}L. "
        f"k_pa={k_pa:.4f}/h, k_ap={k_ap:.4f}/h, ke={ke:.4f}/h. "
        f"Adipose washout t50={washout_t50_h:.1f}h. "
        f"Plasma accumulation ratio={accumulation_ratio_plasma:.2f}, "
        f"Adipose accumulation ratio={accumulation_ratio_adipose:.2f}."
    )

    return AdiposeAccumulationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logp=logp,
        dosing_interval_h=dosing_interval_h,
        n_doses=n_doses,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_mg_L,
        c_adipose_mg_L=c_adipose_mg_L,
        cmax_plasma_steady=cmax_plasma_steady,
        css_plasma_avg=css_plasma_avg,
        cmax_adipose_steady=cmax_adipose_steady,
        css_adipose_avg=css_adipose_avg,
        accumulation_ratio_plasma=accumulation_ratio_plasma,
        accumulation_ratio_adipose=accumulation_ratio_adipose,
        washout_t50_h=washout_t50_h,
        notes=notes,
    )


def compare_logp_accumulation(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    logp_values: list | None = None,
) -> list[AdiposeAccumulationResult]:
    """Compare adipose accumulation across different logP values.

    Parameters
    ----------
    drug_name:
        Base name for the drug.
    dose_mg:
        Dose in mg.
    cl_L_per_h:
        Plasma clearance (L/h).
    logp_values:
        List of logP values to compare. Defaults to [1.0, 2.0, 3.0, 4.0, 5.0].

    Returns
    -------
    list[AdiposeAccumulationResult] sorted by accumulation_ratio_adipose descending.
    """
    if logp_values is None:
        logp_values = [1.0, 2.0, 3.0, 4.0, 5.0]

    results = []
    for lp in logp_values:
        res = simulate_adipose_accumulation(
            drug_name=f"{drug_name}_logP{lp:.1f}",
            dose_mg=dose_mg,
            logp=lp,
            dosing_interval_h=24.0,
            n_doses=10,
            cl_L_per_h=cl_L_per_h,
        )
        results.append(res)

    return sorted(results, key=lambda r: r.accumulation_ratio_adipose, reverse=True)
