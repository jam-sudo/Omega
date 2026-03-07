"""Drug washout time calculator — Phase 684.

Estimates time to reach target residual drug levels after stopping treatment.
Supports mono-exponential decay, simulation, and multi-dose superposition.

References
----------
- Guidance for Industry: Residual Drug in Biopharmaceutics. FDA, 2001.
- Rowland M, Tozer TN. Clinical Pharmacokinetics and Pharmacodynamics. 4th ed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "WashoutResult",
    "WashoutSimResult",
    "estimate_washout_time",
    "simulate_washout_profile",
    "multi_dose_washout",
]


@dataclass(frozen=True)
class WashoutResult:
    """Result from :func:`estimate_washout_time`.

    Attributes
    ----------
    t_half_h : float
        Elimination half-life (h).
    target_fraction : float
        Target residual fraction (0 < target_fraction < 1).
    n_half_lives : float
        Number of half-lives required to reach target.
    washout_time_h : float
        Total washout time in hours.
    residual_fraction : float
        Fraction of dose remaining at washout_time_h (equals target_fraction).
    five_half_life_time_h : float
        Conventional "5 half-lives" washout rule time (h).
    notes : str
    """

    t_half_h: float
    target_fraction: float
    n_half_lives: float
    washout_time_h: float
    residual_fraction: float
    five_half_life_time_h: float
    notes: str


@dataclass
class WashoutSimResult:
    """Result from :func:`simulate_washout_profile`.

    Attributes
    ----------
    t_half_h : float
    dose_mg : float
    vd_L : float
    times_h : list[float]
    c_plasma_mg_L : list[float]
    c0_mg_L : float
        Initial concentration at t=0 (dose_mg / vd_L).
    time_to_50pct_h : float
        Time to reach 50% of C0 (h).
    time_to_10pct_h : float
    time_to_5pct_h : float
    time_to_1pct_h : float
    notes : str
    """

    t_half_h: float
    dose_mg: float
    vd_L: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c0_mg_L: float = 0.0
    time_to_50pct_h: float = 0.0
    time_to_10pct_h: float = 0.0
    time_to_5pct_h: float = 0.0
    time_to_1pct_h: float = 0.0
    notes: str = ""


def estimate_washout_time(
    t_half_h: float,
    target_fraction_remaining: float = 0.01,
    dose_mg: float = 100.0,
) -> WashoutResult:
    """Estimate washout time using mono-exponential decay.

    Parameters
    ----------
    t_half_h : float
        Elimination half-life (h). Must be > 0.
    target_fraction_remaining : float
        Fraction of dose that must remain at washout end (0 < x < 1).
    dose_mg : float
        Reference dose (mg). Must be > 0. Used only for residual mass note.

    Returns
    -------
    WashoutResult
    """
    if t_half_h <= 0:
        raise ValueError("t_half_h must be > 0")
    if not (0.0 < target_fraction_remaining < 1.0):
        raise ValueError("target_fraction_remaining must be in (0, 1)")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")

    # n = log(1/target) / log(2) = -log(target) / log(2)
    n_half_lives = math.log(1.0 / target_fraction_remaining) / math.log(2.0)
    washout_time_h = n_half_lives * t_half_h
    five_half_life_time_h = 5.0 * t_half_h
    residual_mg = target_fraction_remaining * dose_mg

    notes = (
        f"t½={t_half_h:.2f} h; target={target_fraction_remaining:.4f} "
        f"({target_fraction_remaining * 100:.2f}%); "
        f"n_half_lives={n_half_lives:.3f}; washout={washout_time_h:.2f} h; "
        f"residual≈{residual_mg:.4f} mg (from {dose_mg} mg dose). "
        f"5×t½ rule: {five_half_life_time_h:.2f} h."
    )

    return WashoutResult(
        t_half_h=t_half_h,
        target_fraction=target_fraction_remaining,
        n_half_lives=n_half_lives,
        washout_time_h=washout_time_h,
        residual_fraction=target_fraction_remaining,
        five_half_life_time_h=five_half_life_time_h,
        notes=notes,
    )


def simulate_washout_profile(
    t_half_h: float,
    dose_mg: float = 100.0,
    vd_L: float = 50.0,
    t_end_h: float | None = None,
    dt_h: float = 0.5,
) -> WashoutSimResult:
    """Simulate mono-exponential drug washout profile.

    C(t) = C0 × exp(-ke × t),  C0 = dose_mg / vd_L,  ke = ln(2) / t_half_h

    Parameters
    ----------
    t_half_h : float  (> 0)
    dose_mg : float  (> 0)
    vd_L : float  (> 0)
    t_end_h : float | None
        End time; defaults to 7 × t_half_h if None.
    dt_h : float
        Time step (h).

    Returns
    -------
    WashoutSimResult
    """
    if t_half_h <= 0:
        raise ValueError("t_half_h must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    if t_end_h is None:
        t_end_h = 7.0 * t_half_h

    ke = math.log(2.0) / t_half_h
    c0 = dose_mg / vd_L

    times: list[float] = []
    concs: list[float] = []

    t = 0.0
    while t <= t_end_h + 1e-9:
        c = c0 * math.exp(-ke * t)
        times.append(t)
        concs.append(c)
        t += dt_h

    # Find time to reach threshold fractions
    thresholds = {0.50: 0.0, 0.10: 0.0, 0.05: 0.0, 0.01: 0.0}
    found = {k: False for k in thresholds}
    for ti, ci in zip(times, concs):
        for frac in thresholds:
            if not found[frac] and ci <= frac * c0:
                thresholds[frac] = ti
                found[frac] = True

    # If not found in simulation range, extrapolate analytically
    for frac in thresholds:
        if not found[frac]:
            # t = -ln(frac) / ke
            thresholds[frac] = -math.log(frac) / ke

    notes = (
        f"Mono-exponential washout; t½={t_half_h:.2f} h; "
        f"C0={c0:.4f} mg/L; ke={ke:.4f} h⁻¹; "
        f"t(50%)={thresholds[0.50]:.2f} h, "
        f"t(10%)={thresholds[0.10]:.2f} h, "
        f"t(5%)={thresholds[0.05]:.2f} h, "
        f"t(1%)={thresholds[0.01]:.2f} h."
    )

    result = WashoutSimResult(
        t_half_h=t_half_h,
        dose_mg=dose_mg,
        vd_L=vd_L,
        times_h=times,
        c_plasma_mg_L=concs,
        c0_mg_L=c0,
        time_to_50pct_h=thresholds[0.50],
        time_to_10pct_h=thresholds[0.10],
        time_to_5pct_h=thresholds[0.05],
        time_to_1pct_h=thresholds[0.01],
        notes=notes,
    )
    return result


def multi_dose_washout(
    doses_mg: list[float],
    dose_times_h: list[float],
    t_half_h: float,
    vd_L: float = 50.0,
    target_fraction: float = 0.01,
    t_eval_start_h: float | None = None,
) -> dict:
    """Multi-dose washout using superposition principle.

    C(t) = Σ_i (D_i/Vd) × exp(-ke × (t − t_i))  for t ≥ t_i

    Parameters
    ----------
    doses_mg : list[float]
        Dose amounts (mg). All must be > 0.
    dose_times_h : list[float]
        Dosing times (h). Must be same length as doses_mg.
    t_half_h : float  (> 0)
    vd_L : float  (> 0)
    target_fraction : float
        Target residual fraction relative to max_C0 = sum(D)/Vd (0 < x < 1).
    t_eval_start_h : float | None
        Start of washout evaluation; defaults to last dose time.

    Returns
    -------
    dict with keys:
        c_at_start : float  Concentration at t_eval_start_h.
        washout_time_h : float  Time (absolute) when C drops below threshold.
        notes : str
    """
    if t_half_h <= 0:
        raise ValueError("t_half_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (0.0 < target_fraction < 1.0):
        raise ValueError("target_fraction must be in (0, 1)")
    if len(doses_mg) == 0:
        raise ValueError("doses_mg must not be empty")
    if len(doses_mg) != len(dose_times_h):
        raise ValueError("doses_mg and dose_times_h must have equal length")
    for d in doses_mg:
        if d <= 0:
            raise ValueError("All doses must be > 0")

    ke = math.log(2.0) / t_half_h

    if t_eval_start_h is None:
        t_eval_start_h = max(dose_times_h)

    total_dose = sum(doses_mg)
    c_threshold = target_fraction * (total_dose / vd_L)

    def _total_conc(t: float) -> float:
        c = 0.0
        for d, td in zip(doses_mg, dose_times_h):
            if t >= td:
                c += (d / vd_L) * math.exp(-ke * (t - td))
        return c

    c_at_start = _total_conc(t_eval_start_h)

    # Scan forward from t_eval_start_h in 0.5h steps
    dt_scan = 0.5
    washout_time_h = t_eval_start_h
    t_scan = t_eval_start_h
    found = False

    # Max scan: 20 × t_half beyond start
    t_max_scan = t_eval_start_h + 20.0 * t_half_h
    while t_scan <= t_max_scan:
        if _total_conc(t_scan) <= c_threshold:
            washout_time_h = t_scan
            found = True
            break
        t_scan += dt_scan

    if not found:
        washout_time_h = t_max_scan

    notes = (
        f"Multi-dose washout; {len(doses_mg)} dose(s); "
        f"total_dose={total_dose:.1f} mg; t½={t_half_h:.2f} h; "
        f"threshold_C={c_threshold:.4f} mg/L; "
        f"washout_time={washout_time_h:.2f} h from t=0."
    )

    return {
        "c_at_start": c_at_start,
        "washout_time_h": washout_time_h,
        "notes": notes,
    }
