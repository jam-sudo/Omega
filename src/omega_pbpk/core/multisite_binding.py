"""
Phase 557 — Multi-site receptor binding model.

Pure Python, no numpy/scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def hill_binding(concentration: float, kd: float, n_hill: float = 1.0) -> float:
    """Return fractional receptor occupancy via the Hill equation.

    occupancy = C^n / (Kd^n + C^n)

    Parameters
    ----------
    concentration:
        Free drug concentration (same units as kd). Must be >= 0.
    kd:
        Equilibrium dissociation constant. Must be > 0.
    n_hill:
        Hill coefficient. Must be > 0.
    """
    if concentration < 0:
        raise ValueError("concentration must be >= 0")
    if kd <= 0:
        raise ValueError("kd must be > 0")
    if n_hill <= 0:
        raise ValueError("n_hill must be > 0")

    if concentration == 0.0:
        return 0.0

    c_n = concentration**n_hill
    kd_n = kd**n_hill
    return c_n / (kd_n + c_n)


def two_site_binding(
    concentration: float,
    kd1: float,
    bmax1: float,
    kd2: float,
    bmax2: float,
) -> dict:
    """Compute binding to two independent receptor populations.

    Bound_i = Bmax_i * C / (Kd_i + C)

    Parameters
    ----------
    concentration:
        Free drug concentration. Must be >= 0.
    kd1, kd2:
        Equilibrium dissociation constants for each site. Must be > 0.
    bmax1, bmax2:
        Maximum binding capacities for each site. Must be > 0.

    Returns
    -------
    dict with keys:
        bound_site1, bound_site2, total_bound,
        occupancy_site1, occupancy_site2
    """
    if concentration < 0:
        raise ValueError("concentration must be >= 0")
    for name, val in [("kd1", kd1), ("kd2", kd2)]:
        if val <= 0:
            raise ValueError(f"{name} must be > 0")
    for name, val in [("bmax1", bmax1), ("bmax2", bmax2)]:
        if val <= 0:
            raise ValueError(f"{name} must be > 0")

    b1 = bmax1 * concentration / (kd1 + concentration) if concentration > 0 else 0.0
    b2 = bmax2 * concentration / (kd2 + concentration) if concentration > 0 else 0.0

    occ1 = b1 / bmax1
    occ2 = b2 / bmax2

    return {
        "bound_site1": b1,
        "bound_site2": b2,
        "total_bound": b1 + b2,
        "occupancy_site1": occ1,
        "occupancy_site2": occ2,
    }


def receptor_occupancy_curve(
    kd: float,
    n_hill: float = 1.0,
    c_range_min: float = 0.001,
    c_range_max: float = 1000.0,
    n_points: int = 50,
) -> list[dict]:
    """Generate a receptor occupancy curve with logarithmically spaced concentrations.

    Parameters
    ----------
    kd:
        Equilibrium dissociation constant. Must be > 0.
    n_hill:
        Hill coefficient. Must be > 0.
    c_range_min, c_range_max:
        Concentration range endpoints. Must be > 0 with min < max.
    n_points:
        Number of points. Must be >= 2.

    Returns
    -------
    List of dicts with keys: concentration, occupancy
    """
    if kd <= 0:
        raise ValueError("kd must be > 0")
    if n_hill <= 0:
        raise ValueError("n_hill must be > 0")
    if c_range_min <= 0 or c_range_max <= 0:
        raise ValueError("concentration range endpoints must be > 0")
    if c_range_min >= c_range_max:
        raise ValueError("c_range_min must be < c_range_max")
    if n_points < 2:
        raise ValueError("n_points must be >= 2")

    log_min = math.log10(c_range_min)
    log_max = math.log10(c_range_max)
    step = (log_max - log_min) / (n_points - 1)

    result = []
    for i in range(n_points):
        c = 10 ** (log_min + i * step)
        occ = hill_binding(c, kd, n_hill)
        result.append({"concentration": c, "occupancy": occ})

    return result


def competitive_binding_ki(
    ic50: float,
    kd_radioligand: float,
    conc_radioligand: float,
) -> float:
    """Calculate Ki from IC50 using the Cheng-Prusoff equation.

    Ki = IC50 / (1 + [L] / Kd_radioligand)

    Parameters
    ----------
    ic50:
        Observed IC50 of the competing drug. Must be > 0.
    kd_radioligand:
        Kd of the radioligand. Must be > 0.
    conc_radioligand:
        Concentration of the radioligand used. Must be >= 0.

    Returns
    -------
    float
        Ki value.
    """
    if ic50 <= 0:
        raise ValueError("ic50 must be > 0")
    if kd_radioligand <= 0:
        raise ValueError("kd_radioligand must be > 0")
    if conc_radioligand < 0:
        raise ValueError("conc_radioligand must be >= 0")

    return ic50 / (1.0 + conc_radioligand / kd_radioligand)


def receptor_reserve(ec50_receptor: float, ec50_response: float) -> float:
    """Estimate receptor reserve (spare receptor fraction, %).

    receptor_reserve = (ec50_receptor - ec50_response) / ec50_receptor * 100

    Returns 0 if ec50_response >= ec50_receptor (no spare receptors).

    Parameters
    ----------
    ec50_receptor:
        EC50 at the receptor binding level. Must be > 0.
    ec50_response:
        EC50 at the functional/response level. Must be > 0.

    Returns
    -------
    float
        Receptor reserve in percent (0–100).
    """
    if ec50_receptor <= 0:
        raise ValueError("ec50_receptor must be > 0")
    if ec50_response <= 0:
        raise ValueError("ec50_response must be > 0")

    if ec50_response >= ec50_receptor:
        return 0.0

    return (ec50_receptor - ec50_response) / ec50_receptor * 100.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BindingResult:
    kd: float
    n_hill: float
    ec50: float
    bmax: float
    r_squared: float
    notes: str


# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------


def fit_binding_curve(
    concentrations: list[float],
    occupancies: list[float],
) -> BindingResult:
    """Fit a Hill binding curve by grid search.

    Parameters
    ----------
    concentrations:
        List of free drug concentrations. Must be > 0.
    occupancies:
        Observed fractional occupancies (0–1). Must have same length as concentrations.

    Returns
    -------
    BindingResult
    """
    if len(concentrations) != len(occupancies):
        raise ValueError("concentrations and occupancies must have the same length")
    if len(concentrations) < 2:
        raise ValueError("at least 2 data points required")
    if any(c <= 0 for c in concentrations):
        raise ValueError("all concentrations must be > 0")
    if any(not (0.0 <= o <= 1.0) for o in occupancies):
        raise ValueError("all occupancies must be in [0, 1]")

    # Grid parameters
    n_kd = 100
    n_hill_pts = 10
    kd_log_min = math.log10(0.001)
    kd_log_max = math.log10(10000.0)
    hill_values = [0.5 + (3.0 - 0.5) * i / (n_hill_pts - 1) for i in range(n_hill_pts)]

    best_sse = float("inf")
    best_kd = 1.0
    best_n = 1.0

    for ki in range(n_kd):
        kd_try = 10 ** (kd_log_min + (kd_log_max - kd_log_min) * ki / (n_kd - 1))
        for n_try in hill_values:
            sse = 0.0
            for c, obs in zip(concentrations, occupancies):
                pred = hill_binding(c, kd_try, n_try)
                sse += (pred - obs) ** 2
            if sse < best_sse:
                best_sse = sse
                best_kd = kd_try
                best_n = n_try

    # Compute R²
    mean_occ = sum(occupancies) / len(occupancies)
    ss_tot = sum((o - mean_occ) ** 2 for o in occupancies)
    r_sq = 1.0 - best_sse / ss_tot if ss_tot > 0 else 1.0
    r_sq = max(0.0, min(1.0, r_sq))

    return BindingResult(
        kd=best_kd,
        n_hill=best_n,
        ec50=best_kd,  # EC50 = Kd for Hill equation
        bmax=1.0,  # normalized
        r_squared=r_sq,
        notes="Grid search over log-spaced Kd and Hill coefficient",
    )
