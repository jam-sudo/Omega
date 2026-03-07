"""Multiple dosing PK simulation with flexible regimen specification."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["MultipleDosePKResult", "simulate_multiple_dosing", "simulate_qd_bid_tid"]


@dataclass
class MultipleDosePKResult:
    """Result of multiple dosing PK simulation."""

    drug_name: str
    n_doses: int
    total_dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    css_max: float
    css_min: float
    css_avg: float
    accumulation_ratio: float
    auc_0_inf: float
    t_half_h: float
    notes: str = field(default="")


def simulate_multiple_dosing(
    drug_name: str,
    doses_mg: list,
    dose_times_h: list,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    ka_per_h: float = 1.0,
    f_oral: float = 0.8,
    t_end_h: float | None = None,
    dt_h: float = 0.1,
) -> MultipleDosePKResult:
    """Simulate multiple dosing PK using Forward Euler integration.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    doses_mg:
        List of dose amounts in mg for each dosing event.
    dose_times_h:
        List of dose times in hours (must be sorted ascending).
    cl_L_per_h:
        Total clearance in L/h.
    vd_L:
        Volume of distribution in L.
    ka_per_h:
        Absorption rate constant in 1/h.
    f_oral:
        Oral bioavailability fraction.
    t_end_h:
        End time of simulation; defaults to last_dose + max(5*t_half, 24).
    dt_h:
        Integration time step in hours.

    Returns
    -------
    MultipleDosePKResult
    """
    # Validate inputs
    if len(doses_mg) != len(dose_times_h):
        raise ValueError("doses_mg and dose_times_h must have the same length")

    for i in range(len(dose_times_h) - 1):
        if dose_times_h[i] >= dose_times_h[i + 1]:
            raise ValueError("dose_times_h must be sorted in strictly ascending order")

    for d in doses_mg:
        if d <= 0:
            raise ValueError("All doses must be positive (> 0)")

    ke = cl_L_per_h / vd_L
    t_half_h = math.log(2.0) / ke

    if t_end_h is None:
        t_end_h = dose_times_h[-1] + max(5.0 * t_half_h, 24.0)

    # Forward Euler simulation
    n_steps = int(round(t_end_h / dt_h)) + 1
    times = [i * dt_h for i in range(n_steps)]

    # Ensure we don't exceed t_end_h
    times = [t for t in times if t <= t_end_h + 1e-9]

    # State: A_gut (amount in gut), C (plasma concentration)
    A_gut = 0.0
    C = 0.0

    dose_idx = 0
    n_doses_total = len(doses_mg)

    c_plasma = []
    times_out = []

    for _i, t in enumerate(times):
        # Apply doses scheduled at this time step
        while dose_idx < n_doses_total and dose_times_h[dose_idx] <= t + dt_h * 0.5:
            A_gut += doses_mg[dose_idx] * f_oral
            dose_idx += 1

        times_out.append(t)
        c_plasma.append(C)

        # Forward Euler update
        dA_gut = -ka_per_h * A_gut
        dC = ka_per_h * A_gut / vd_L - ke * C

        A_gut += dA_gut * dt_h
        C += dC * dt_h
        if C < 0.0:
            C = 0.0
        if A_gut < 0.0:
            A_gut = 0.0

    # Global Cmax and tmax
    cmax_mg_L = max(c_plasma)
    tmax_h = times_out[c_plasma.index(cmax_mg_L)]

    # Steady-state metrics from last dosing interval
    if n_doses_total >= 2:
        last_dose_t = dose_times_h[-1]
        prev_dose_t = dose_times_h[-2]
        interval = last_dose_t - prev_dose_t
        ss_times = [t for t in times_out if t >= last_dose_t]
        ss_concs = [c_plasma[times_out.index(t)] for t in ss_times]
        if len(ss_concs) > 0:
            css_max = max(ss_concs)
            css_min = min(ss_concs)
            # Trapezoidal AUC over last interval
            ss_end = last_dose_t + interval
            ss_window = [
                (t, c) for t, c in zip(times_out, c_plasma) if last_dose_t <= t <= ss_end + 1e-9
            ]
            if len(ss_window) >= 2:
                auc_ss = 0.0
                for j in range(1, len(ss_window)):
                    dt_j = ss_window[j][0] - ss_window[j - 1][0]
                    auc_ss += 0.5 * (ss_window[j][1] + ss_window[j - 1][1]) * dt_j
                css_avg = auc_ss / interval if interval > 0 else css_max
            else:
                css_avg = css_max
        else:
            css_max = cmax_mg_L
            css_min = 0.0
            css_avg = cmax_mg_L
    else:
        # Single dose: css is same as max
        css_max = cmax_mg_L
        css_min = min(c_plasma)
        css_avg = cmax_mg_L

    # Single-dose Cmax for accumulation ratio
    single_cmax = _single_dose_cmax(doses_mg[0], f_oral, ka_per_h, ke, vd_L)
    accumulation_ratio = css_max / single_cmax if single_cmax > 0 else 1.0

    # Total AUC (trapezoidal)
    auc_0_inf = 0.0
    for j in range(1, len(times_out)):
        dt_j = times_out[j] - times_out[j - 1]
        auc_0_inf += 0.5 * (c_plasma[j] + c_plasma[j - 1]) * dt_j

    total_dose_mg = sum(doses_mg)
    notes = f"Forward Euler simulation; ke={ke:.4f}/h; t_half={t_half_h:.2f}h"

    return MultipleDosePKResult(
        drug_name=drug_name,
        n_doses=n_doses_total,
        total_dose_mg=total_dose_mg,
        times_h=times_out,
        c_plasma_mg_L=c_plasma,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        css_max=css_max,
        css_min=css_min,
        css_avg=css_avg,
        accumulation_ratio=accumulation_ratio,
        auc_0_inf=auc_0_inf,
        t_half_h=t_half_h,
        notes=notes,
    )


def _single_dose_cmax(
    dose_mg: float,
    f_oral: float,
    ka: float,
    ke: float,
    vd: float,
) -> float:
    """Analytical single-dose Cmax for 1-cpt oral model."""
    if abs(ka - ke) < 1e-9:
        ka = ka * 1.0001
    tmax = math.log(ka / ke) / (ka - ke)
    if tmax < 0:
        tmax = 0.0
    cmax = (
        (dose_mg * f_oral * ka) / (vd * (ka - ke)) * (math.exp(-ke * tmax) - math.exp(-ka * tmax))
    )
    return max(cmax, 0.0)


def simulate_qd_bid_tid(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    f_oral: float = 0.8,
    n_days: int = 7,
) -> dict:
    """Compare QD, BID, TID regimens with same total daily dose.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Total daily dose in mg.
    cl_L_per_h:
        Clearance in L/h.
    vd_L:
        Volume of distribution in L.
    ka_per_h:
        Absorption rate constant in 1/h.
    f_oral:
        Oral bioavailability.
    n_days:
        Number of days to simulate.

    Returns
    -------
    dict with keys 'qd', 'bid', 'tid', 'notes'
    """
    t_end = n_days * 24.0

    # QD: full dose once daily
    qd_doses = [dose_mg] * n_days
    qd_times = [i * 24.0 for i in range(n_days)]

    # BID: half dose every 12h
    bid_doses = [dose_mg / 2.0] * (n_days * 2)
    bid_times = [i * 12.0 for i in range(n_days * 2)]

    # TID: third dose every 8h
    tid_doses = [dose_mg / 3.0] * (n_days * 3)
    tid_times = [i * 8.0 for i in range(n_days * 3)]

    qd_result = simulate_multiple_dosing(
        drug_name=drug_name,
        doses_mg=qd_doses,
        dose_times_h=qd_times,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        ka_per_h=ka_per_h,
        f_oral=f_oral,
        t_end_h=t_end,
    )
    bid_result = simulate_multiple_dosing(
        drug_name=drug_name,
        doses_mg=bid_doses,
        dose_times_h=bid_times,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        ka_per_h=ka_per_h,
        f_oral=f_oral,
        t_end_h=t_end,
    )
    tid_result = simulate_multiple_dosing(
        drug_name=drug_name,
        doses_mg=tid_doses,
        dose_times_h=tid_times,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        ka_per_h=ka_per_h,
        f_oral=f_oral,
        t_end_h=t_end,
    )

    notes = (
        f"QD={dose_mg}mg q24h, BID={dose_mg / 2:.1f}mg q12h, TID={dose_mg / 3:.2f}mg q8h; "
        f"same daily dose={dose_mg}mg"
    )

    return {"qd": qd_result, "bid": bid_result, "tid": tid_result, "notes": notes}
