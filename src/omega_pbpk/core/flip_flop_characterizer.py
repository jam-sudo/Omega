"""Phase 644 — Flip-flop pharmacokinetics characterizer.

Detects and characterizes flip-flop PK (ka < ke), where absorption is rate-
limiting and the terminal slope reflects ka rather than ke.

This module provides characterization via simulation and data-driven detection.
For simple detection utilities, see flip_flop_pk.py.

References
----------
- Riegelman S, Collier P. J Pharmacokinet Biopharm. 1980;8(5):509-34.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FlipFlopResult:
    drug_name: str
    dose_mg: float
    ka_true_per_h: float
    ke_true_per_h: float
    is_flip_flop: bool
    apparent_terminal_slope: float
    true_ke_per_h: float
    true_t_half_h: float
    apparent_t_half_h: float
    tmax_h: float
    cmax_mg_L: float
    auc_mg_h_per_L: float
    flip_flop_ratio: float
    notes: str


def _fit_log_linear_slope(times: list[float], concs: list[float]) -> float:
    """Least-squares log-linear slope; returns magnitude (rate constant)."""
    pairs = [(t, c) for t, c in zip(times, concs) if c > 1e-12]
    if len(pairs) < 2:
        return 0.0

    n = len(pairs)
    sum_t = sum(p[0] for p in pairs)
    sum_lnc = sum(math.log(p[1]) for p in pairs)
    sum_t2 = sum(p[0] ** 2 for p in pairs)
    sum_t_lnc = sum(p[0] * math.log(p[1]) for p in pairs)

    denom = n * sum_t2 - sum_t**2
    if abs(denom) < 1e-12:
        return 0.0

    slope = (n * sum_t_lnc - sum_t * sum_lnc) / denom
    return max(0.0, -slope)  # slope is negative for decay; return magnitude


def simulate_flip_flop_pk(
    drug_name: str,
    dose_mg: float,
    ka_per_h: float,
    ke_per_h: float,
    vd_L: float = 50.0,
    f_oral: float = 1.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.05,
) -> FlipFlopResult:
    """Simulate oral 1-cpt PK and characterize flip-flop condition.

    ODE (forward Euler):
        dA_gut/dt = -ka * A_gut
        dC/dt     =  ka * f_oral * A_gut / Vd  -  ke * C
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if ka_per_h <= 0:
        raise ValueError(f"ka_per_h must be > 0, got {ka_per_h}")
    if ke_per_h <= 0:
        raise ValueError(f"ke_per_h must be > 0, got {ke_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if not (0.0 < f_oral <= 1.0):
        raise ValueError(f"f_oral must be in (0, 1], got {f_oral}")

    # Forward Euler
    n_steps = int(t_end_h / dt_h) + 1
    times: list[float] = []
    concs: list[float] = []

    a_gut = dose_mg
    c = 0.0
    t = 0.0

    for _ in range(n_steps):
        times.append(t)
        concs.append(c)
        da_gut = -ka_per_h * a_gut
        dc = (ka_per_h * f_oral * a_gut / vd_L) - ke_per_h * c
        a_gut = max(0.0, a_gut + da_gut * dt_h)
        c = max(0.0, c + dc * dt_h)
        t = round(t + dt_h, 10)

    # Cmax and Tmax
    cmax_mg_L = max(concs)
    tmax_h = times[concs.index(cmax_mg_L)]

    # AUC (trapezoidal)
    auc_mg_h_per_L = sum(
        (concs[i] + concs[i + 1]) / 2.0 * (times[i + 1] - times[i]) for i in range(len(times) - 1)
    )

    # Terminal slope: last 33% of time points
    n_total = len(times)
    terminal_start = int(n_total * 0.67)
    terminal_times = times[terminal_start:]
    terminal_concs = concs[terminal_start:]

    apparent_slope = _fit_log_linear_slope(terminal_times, terminal_concs)
    if apparent_slope <= 0.0:
        apparent_slope = min(ka_per_h, ke_per_h)

    is_flip_flop = ka_per_h < ke_per_h
    true_t_half_h = math.log(2.0) / ke_per_h
    apparent_t_half_h = math.log(2.0) / apparent_slope
    flip_flop_ratio = ke_per_h / ka_per_h

    note_parts = []
    if is_flip_flop:
        note_parts.append(
            f"Flip-flop PK detected: ka={ka_per_h:.3f}/h < ke={ke_per_h:.3f}/h. "
            "Terminal slope reflects absorption, not elimination."
        )
        note_parts.append(
            f"True t\u00bd={true_t_half_h:.2f}h; apparent t\u00bd={apparent_t_half_h:.2f}h."
        )
    else:
        note_parts.append(
            f"Normal PK: ka={ka_per_h:.3f}/h >= ke={ke_per_h:.3f}/h. "
            "Terminal slope reflects elimination."
        )
    notes = " ".join(note_parts)

    return FlipFlopResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        ka_true_per_h=ka_per_h,
        ke_true_per_h=ke_per_h,
        is_flip_flop=is_flip_flop,
        apparent_terminal_slope=apparent_slope,
        true_ke_per_h=ke_per_h,
        true_t_half_h=true_t_half_h,
        apparent_t_half_h=apparent_t_half_h,
        tmax_h=tmax_h,
        cmax_mg_L=cmax_mg_L,
        auc_mg_h_per_L=auc_mg_h_per_L,
        flip_flop_ratio=flip_flop_ratio,
        notes=notes,
    )


def detect_flip_flop(
    times_h: list[float],
    concs_mg_L: list[float],
    iv_ke_per_h: float | None = None,
) -> dict:
    """Detect flip-flop from observed concentration-time data.

    Returns dict with keys: is_flip_flop, apparent_ke, confidence, notes.
    If iv_ke_per_h provided: is_flip_flop = apparent_ke < iv_ke_per_h * 0.7.
    """
    if len(times_h) < 3 or len(concs_mg_L) < 3:
        return {
            "is_flip_flop": False,
            "apparent_ke": 0.0,
            "confidence": "low",
            "notes": "Insufficient data points for reliable flip-flop detection.",
        }

    n = len(times_h)
    terminal_start = max(int(n * 0.70), n - 10)
    terminal_start = min(terminal_start, n - 2)

    terminal_times = times_h[terminal_start:]
    terminal_concs = concs_mg_L[terminal_start:]

    apparent_ke = _fit_log_linear_slope(terminal_times, terminal_concs)

    is_flip_flop = False
    confidence = "moderate"
    note_parts: list[str] = []

    if iv_ke_per_h is not None:
        is_flip_flop = apparent_ke < iv_ke_per_h * 0.7
        ratio = apparent_ke / iv_ke_per_h if iv_ke_per_h > 0.0 else 1.0
        if ratio < 0.5:
            confidence = "high"
            note_parts.append(
                f"Apparent ke ({apparent_ke:.4f}/h) << IV ke ({iv_ke_per_h:.4f}/h): "
                "high-confidence flip-flop."
            )
        elif ratio < 0.7:
            confidence = "moderate"
            note_parts.append(
                f"Apparent ke ({apparent_ke:.4f}/h) < 0.7 \u00d7 IV ke ({iv_ke_per_h:.4f}/h): "
                "moderate flip-flop evidence."
            )
        else:
            confidence = "low"
            note_parts.append(
                f"Apparent ke ({apparent_ke:.4f}/h) similar to IV ke ({iv_ke_per_h:.4f}/h): "
                "no flip-flop detected."
            )
    else:
        n_terminal = len(terminal_times)
        confidence = "moderate" if n_terminal >= 5 else "low"
        note_parts.append(
            f"Apparent terminal ke={apparent_ke:.4f}/h from {n_terminal} points. "
            "Provide iv_ke_per_h for definitive detection."
        )

    notes = " ".join(note_parts) if note_parts else "Analysis complete."
    return {
        "is_flip_flop": is_flip_flop,
        "apparent_ke": apparent_ke,
        "confidence": confidence,
        "notes": notes,
    }
