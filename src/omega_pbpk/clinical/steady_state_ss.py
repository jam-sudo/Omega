"""Multi-day repeated dosing simulation to reach pharmacokinetic steady state.

Simulates iterative ODE-based repeat dosing to detect convergence, using
1-compartment forward Euler integration. More explicit than superposition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "SteadyStateResult",
    "simulate_to_steady_state",
    "steady_state_metrics",
]


@dataclass
class SteadyStateResult:
    """Result of multi-day steady-state simulation."""

    drug_name: str
    dose_mg: float
    interval_h: float
    route: str
    n_cycles_to_ss: int
    css_max_mg_L: float
    css_min_mg_L: float
    css_avg_mg_L: float
    accumulation_ratio: float
    fluctuation_pct: float
    times_last3_cycles: np.ndarray
    c_last3_cycles: np.ndarray
    converged: bool
    t_half_h: float
    notes: str


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration compatible with older numpy."""
    try:
        return float(np.trapz(y, x))
    except AttributeError:
        return float(np.trapezoid(y, x))


def _validate_ss_inputs(
    dose_mg: float,
    interval_h: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    max_cycles: int,
) -> None:
    """Validate inputs for simulate_to_steady_state."""
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if interval_h <= 0:
        raise ValueError(f"interval_h must be > 0, got {interval_h}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if ka_per_h <= 0:
        raise ValueError(f"ka_per_h must be > 0, got {ka_per_h}")
    if max_cycles < 3:
        raise ValueError(f"max_cycles must be >= 3, got {max_cycles}")


def simulate_to_steady_state(
    drug_name: str,
    dose_mg: float,
    interval_h: float,
    cl_L_per_h: float,
    vd_L: float,
    route: str = "oral",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    max_cycles: int = 100,
    convergence_tol: float = 0.01,
    t_per_cycle_pts: int = 200,
) -> SteadyStateResult:
    """Simulate repeated dosing until pharmacokinetic steady state is reached.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Dose per administration in mg. Must be > 0.
    interval_h:
        Dosing interval (tau) in hours. Must be > 0.
    cl_L_per_h:
        Clearance in L/h. Must be > 0.
    vd_L:
        Volume of distribution in L. Must be > 0.
    route:
        "oral" or "iv".
    ka_per_h:
        First-order oral absorption rate constant in 1/h. Must be > 0.
    f_oral:
        Oral bioavailability fraction (0 to 1).
    max_cycles:
        Maximum number of dosing cycles before declaring non-convergence.
        Must be >= 3.
    convergence_tol:
        Fractional tolerance for Cmax/Cmin between consecutive cycles.
        Converged when relative change < convergence_tol.
    t_per_cycle_pts:
        Number of time points per dosing cycle for ODE integration.

    Returns
    -------
    SteadyStateResult
    """
    _validate_ss_inputs(dose_mg, interval_h, cl_L_per_h, vd_L, ka_per_h, max_cycles)

    ke = cl_L_per_h / vd_L  # elimination rate constant [1/h]
    t_half_h = math.log(2.0) / ke

    # Time step within a cycle
    dt_h = interval_h / t_per_cycle_pts

    # Bioavailable dose per cycle
    dose_abs = dose_mg * f_oral if route == "oral" else dose_mg

    # State: central concentration [mg/L], depot amount [mg] (oral)
    c_central = 0.0
    depot = 0.0

    # Track per-cycle metrics
    prev_cmax = None
    prev_cmin = None
    n_cycles_to_ss = 1
    converged = False

    # Storage for last 3 cycles
    last3_times: list[np.ndarray] = []
    last3_conc: list[np.ndarray] = []

    # Track single-dose Cmax (first cycle, no inversion carryover)
    single_dose_cmax: float | None = None

    for cycle_idx in range(max_cycles):
        # Apply dose at start of this cycle
        if route == "oral":
            depot = depot + dose_abs
        else:
            c_central = c_central + dose_abs / vd_L

        # Store time-concentration for this cycle
        cycle_offset = cycle_idx * interval_h
        cycle_times = np.linspace(0.0, interval_h, t_per_cycle_pts + 1)
        cycle_conc = np.zeros(t_per_cycle_pts + 1)
        cycle_conc[0] = c_central

        for i in range(t_per_cycle_pts):
            cc = cycle_conc[i]

            if route == "oral":
                abs_flux = ka_per_h * depot  # mg/h from depot
                depot_new = max(0.0, depot - abs_flux * dt_h)
                abs_conc_flux = abs_flux / vd_L  # mg/L/h into central
                depot = depot_new
            else:
                abs_conc_flux = 0.0

            dcc_dt = abs_conc_flux - ke * cc
            cycle_conc[i + 1] = max(0.0, cc + dcc_dt * dt_h)
            c_central = cycle_conc[i + 1]

        cmax_cycle = float(np.max(cycle_conc))
        cmin_cycle = float(cycle_conc[-1])  # trough = end of cycle

        if cycle_idx == 0:
            single_dose_cmax = cmax_cycle

        # Check convergence (starting from cycle 2 to have a previous value)
        if prev_cmax is not None:
            rel_cmax = abs(cmax_cycle - prev_cmax) / max(prev_cmax, 1e-12)
            rel_cmin = abs(cmin_cycle - prev_cmin) / max(prev_cmin, 1e-12)  # type: ignore[arg-type]
            if rel_cmax < convergence_tol and rel_cmin < convergence_tol:
                n_cycles_to_ss = cycle_idx + 1
                converged = True
                # Store current cycle and break
                last3_times.append(cycle_offset + cycle_times)
                last3_conc.append(cycle_conc.copy())
                break

        prev_cmax = cmax_cycle
        prev_cmin = cmin_cycle

        # Keep rolling window of last 3 cycles
        last3_times.append(cycle_offset + cycle_times)
        last3_conc.append(cycle_conc.copy())
        if len(last3_times) > 3:
            last3_times.pop(0)
            last3_conc.pop(0)

    if not converged:
        n_cycles_to_ss = max_cycles

    # Use last complete cycle for SS metrics
    final_conc = last3_conc[-1]
    final_times = last3_times[-1]

    css_max = float(np.max(final_conc))
    css_min = float(final_conc[-1])
    auc_tau = _trapz(final_conc, final_times - final_times[0])
    css_avg = auc_tau / max(interval_h, 1e-12)
    accum_ratio = css_max / max(single_dose_cmax or 1e-12, 1e-12)
    accum_ratio = max(1.0, accum_ratio)  # accumulation ratio >= 1
    fluctuation_pct = 100.0 * (css_max - css_min) / max(css_avg, 1e-12)
    fluctuation_pct = max(0.0, fluctuation_pct)

    # Concatenate last 3 cycles
    times_all = np.concatenate([t for t in last3_times])
    conc_all = np.concatenate([c for c in last3_conc])

    notes_parts = [
        f"t_half={t_half_h:.2f} h",
        f"ke={ke:.4f} 1/h",
        f"Converged: {converged}",
        f"Cycles to SS: {n_cycles_to_ss}",
    ]
    notes = "; ".join(notes_parts)

    return SteadyStateResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        interval_h=interval_h,
        route=route,
        n_cycles_to_ss=n_cycles_to_ss,
        css_max_mg_L=css_max,
        css_min_mg_L=css_min,
        css_avg_mg_L=css_avg,
        accumulation_ratio=accum_ratio,
        fluctuation_pct=fluctuation_pct,
        times_last3_cycles=times_all,
        c_last3_cycles=conc_all,
        converged=converged,
        t_half_h=t_half_h,
        notes=notes,
    )


def steady_state_metrics(
    cl_L_per_h: float,
    vd_L: float,
    dose_mg: float,
    interval_h: float,
    f_oral: float = 1.0,
) -> dict:
    """Analytical predictions of steady-state PK parameters.

    Parameters
    ----------
    cl_L_per_h:
        Clearance in L/h.
    vd_L:
        Volume of distribution in L.
    dose_mg:
        Dose per administration in mg.
    interval_h:
        Dosing interval (tau) in hours.
    f_oral:
        Oral bioavailability fraction.

    Returns
    -------
    dict
        Dictionary with keys: Css_avg_mg_L, t_half_h, accumulation_ratio,
        AUC_tau, n_half_lives_to_ss.
    """
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if interval_h <= 0:
        raise ValueError(f"interval_h must be > 0, got {interval_h}")

    ke = cl_L_per_h / vd_L
    t_half_h = math.log(2.0) / ke

    # Css_avg = F * dose / (CL * tau)
    css_avg = f_oral * dose_mg / (cl_L_per_h * interval_h)

    # AUC over one dosing interval at steady state
    auc_tau = css_avg * interval_h

    # Accumulation ratio: R = 1 / (1 - exp(-ke*tau))
    accum_ratio = 1.0 / (1.0 - math.exp(-ke * interval_h))

    # Number of half-lives to 95% steady state
    n_hl_to_ss = math.log(1.0 / (1.0 - 0.95)) / math.log(2.0)

    return {
        "Css_avg_mg_L": css_avg,
        "t_half_h": t_half_h,
        "accumulation_ratio": accum_ratio,
        "AUC_tau": auc_tau,
        "n_half_lives_to_ss": n_hl_to_ss,
    }
