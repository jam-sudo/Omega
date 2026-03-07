"""Flip-flop pharmacokinetics — Phase 561.

In flip-flop PK, absorption is slower than elimination (ka < ke).
The apparent terminal half-life reflects absorption, not elimination.
This is common with controlled-release formulations or drugs with very
rapid distribution/elimination.

References
----------
- Riegelman S, Collier P. J Pharmacokinet Biopharm. 1980;8(5):509-34.
- Gabrielsson J, Weiner D. Pharmacokinetic and Pharmacodynamic Data Analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Core detection / ratio helpers
# ---------------------------------------------------------------------------


def detect_flip_flop(ka_per_h: float, ke_per_h: float) -> bool:
    """Return True if flip-flop condition holds (ka < ke).

    Parameters
    ----------
    ka_per_h:
        Absorption rate constant (h^-1). Must be > 0.
    ke_per_h:
        Elimination rate constant (h^-1). Must be > 0.

    Returns
    -------
    bool
        True when ka < ke (flip-flop pharmacokinetics).
    """
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if ke_per_h <= 0:
        raise ValueError("ke_per_h must be > 0")
    return ka_per_h < ke_per_h


def flip_flop_ratio(ka_per_h: float, ke_per_h: float) -> float:
    """Return ka / ke.  A value < 1 indicates flip-flop pharmacokinetics.

    Parameters
    ----------
    ka_per_h:
        Absorption rate constant (h^-1). Must be > 0.
    ke_per_h:
        Elimination rate constant (h^-1). Must be > 0.

    Returns
    -------
    float
        ka / ke.
    """
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if ke_per_h <= 0:
        raise ValueError("ke_per_h must be > 0")
    return ka_per_h / ke_per_h


def true_half_life_from_iv(ke_per_h: float) -> float:
    """Return the true elimination half-life (ln2 / ke) measured from IV data.

    Parameters
    ----------
    ke_per_h:
        Elimination rate constant (h^-1). Must be > 0.

    Returns
    -------
    float
        True elimination half-life in hours.
    """
    if ke_per_h <= 0:
        raise ValueError("ke_per_h must be > 0")
    return math.log(2.0) / ke_per_h


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass
class FlipFlopResult:
    """Result from a flip-flop PK simulation."""

    drug_name: str
    times_h: list
    c_plasma_mg_L: list
    cmax: float
    tmax_h: float
    auc: float  # Trapezoidal AUC_0-t (mg*h/L)
    t_half_apparent_h: float  # ln2 / min(ka, ke) — governs terminal slope
    is_flip_flop: bool
    notes: str


def simulate_flip_flop_pk(
    drug_name: str,
    dose_mg: float,
    ka_per_h: float,
    ke_per_h: float,
    vd_L: float,
    f_oral: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> dict:
    """Simulate one-compartment oral PK with possible flip-flop kinetics.

    Uses forward Euler integration of:
        dA_gut/dt = -ka * A_gut
        dCp/dt    =  ka * A_gut / Vd - ke * Cp

    Parameters
    ----------
    drug_name:
        Name of the drug / formulation.
    dose_mg:
        Administered oral dose (mg). Must be > 0.
    ka_per_h:
        First-order absorption rate constant (h^-1). Must be > 0.
    ke_per_h:
        First-order elimination rate constant (h^-1). Must be > 0.
    vd_L:
        Volume of distribution (L). Must be > 0.
    f_oral:
        Oral bioavailability fraction (0, 1]. Default 1.0.
    t_end_h:
        Simulation end time (h). Must be > 0.
    dt_h:
        Integration step (h). Must be > 0.

    Returns
    -------
    dict with keys:
        times_h, c_plasma_mg_L, cmax, tmax_h, auc,
        t_half_apparent_h, is_flip_flop, notes
    """
    # --- validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if ke_per_h <= 0:
        raise ValueError("ke_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (0 < f_oral <= 1.0):
        raise ValueError("f_oral must be in (0, 1]")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    # --- setup ---
    n_steps = int(t_end_h / dt_h) + 1
    times = [i * dt_h for i in range(n_steps)]
    c_plasma = [0.0] * n_steps

    a_gut = dose_mg * f_oral  # initial gut amount (mg)
    cp = 0.0

    for i in range(1, n_steps):
        d_agut = -ka_per_h * a_gut
        d_cp = ka_per_h * a_gut / vd_L - ke_per_h * cp

        a_gut = max(0.0, a_gut + d_agut * dt_h)
        cp = max(0.0, cp + d_cp * dt_h)
        c_plasma[i] = cp

    # --- metrics ---
    cmax = max(c_plasma)
    tmax_h = times[c_plasma.index(cmax)]

    # Trapezoidal AUC
    auc = 0.0
    for j in range(1, n_steps):
        auc += 0.5 * (c_plasma[j - 1] + c_plasma[j]) * dt_h

    # In flip-flop, the apparent terminal half-life is governed by the slower
    # (rate-limiting) step, which is absorption (ka < ke).
    rate_limiting = min(ka_per_h, ke_per_h)
    t_half_apparent_h = math.log(2.0) / rate_limiting

    is_ff = detect_flip_flop(ka_per_h, ke_per_h)

    if is_ff:
        notes = (
            f"{drug_name}: FLIP-FLOP kinetics detected (ka={ka_per_h:.3f} < "
            f"ke={ke_per_h:.3f} h^-1). Terminal slope reflects absorption, "
            f"not elimination. True t1/2_elim = {true_half_life_from_iv(ke_per_h):.2f} h; "
            f"apparent t1/2 = {t_half_apparent_h:.2f} h."
        )
    else:
        notes = (
            f"{drug_name}: Normal PK (ka={ka_per_h:.3f} >= ke={ke_per_h:.3f} h^-1). "
            f"Terminal slope reflects elimination. "
            f"t1/2_apparent = {t_half_apparent_h:.2f} h."
        )

    return {
        "times_h": times,
        "c_plasma_mg_L": c_plasma,
        "cmax": cmax,
        "tmax_h": tmax_h,
        "auc": auc,
        "t_half_apparent_h": t_half_apparent_h,
        "is_flip_flop": is_ff,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Formulation comparison
# ---------------------------------------------------------------------------


def compare_formulations(
    drug_name: str,
    dose_mg: float,
    ke_per_h: float,
    vd_L: float,
    ka_values: list,
    t_end_h: float = 24.0,
) -> list:
    """Compare multiple formulations with different ka values.

    Parameters
    ----------
    drug_name:
        Drug / compound name.
    dose_mg:
        Dose (mg). Must be > 0.
    ke_per_h:
        Elimination rate constant (h^-1). Must be > 0.
    vd_L:
        Volume of distribution (L). Must be > 0.
    ka_values:
        List of absorption rate constants (h^-1) to evaluate.
    t_end_h:
        Simulation duration (h). Default 24.0.

    Returns
    -------
    list[dict]
        Sorted by ka ascending. Each dict contains:
        ka, is_flip_flop, cmax, tmax_h, auc, t_half_apparent_h.
    """
    if not ka_values:
        return []

    results = []
    for ka in sorted(ka_values):
        res = simulate_flip_flop_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            ka_per_h=ka,
            ke_per_h=ke_per_h,
            vd_L=vd_L,
            t_end_h=t_end_h,
        )
        results.append(
            {
                "ka": ka,
                "is_flip_flop": res["is_flip_flop"],
                "cmax": res["cmax"],
                "tmax_h": res["tmax_h"],
                "auc": res["auc"],
                "t_half_apparent_h": res["t_half_apparent_h"],
            }
        )
    return results


__all__ = [
    "detect_flip_flop",
    "flip_flop_ratio",
    "true_half_life_from_iv",
    "simulate_flip_flop_pk",
    "compare_formulations",
    "FlipFlopResult",
]
