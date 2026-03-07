"""Two-site receptor binding model -- Phase 274.

Models ligand binding to a receptor with two distinct binding sites:
a primary orthosteric site (high affinity) and a secondary allosteric site.
Includes cooperativity between sites.

References
----------
- Kenakin T, Pharmacol Rev. 2004;56(3):331-42
- Ehlert FJ, Mol Pharmacol. 1988;33(2):187-94
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["TwoSiteBindingResult", "two_site_occupancy", "evaluate_two_site_binding"]


@dataclass(frozen=True)
class TwoSiteBindingResult:
    """Result from two-site binding model evaluation."""

    drug_name: str
    concentrations: list[float]
    site1_occupancy: list[float]  # Primary site occupancy (0-1)
    site2_occupancy: list[float]  # Secondary site occupancy (0-1)
    total_effect: list[float]  # Combined pharmacological effect
    ec50_apparent: float  # Apparent EC50 for total effect
    hill_apparent: float  # Apparent Hill coefficient
    notes: str


def two_site_occupancy(
    c: float,
    kd1: float,
    kd2: float,
    alpha: float = 1.0,
) -> tuple[float, float]:
    """Compute occupancy at two binding sites for concentration c.

    Simple model: site 1 (primary orthosteric) and site 2 (allosteric).
    Alpha = cooperativity factor (>1 = positive, <1 = negative cooperativity).

    Parameters
    ----------
    c : Drug concentration. Must be >= 0.
    kd1 : Dissociation constant for site 1. Must be > 0.
    kd2 : Dissociation constant for site 2. Must be > 0.
    alpha : Cooperativity factor. Must be > 0.

    Returns (occ1, occ2)
    """
    if c < 0:
        raise ValueError("c must be >= 0")
    if kd1 <= 0:
        raise ValueError("kd1 must be > 0")
    if kd2 <= 0:
        raise ValueError("kd2 must be > 0")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    if c == 0:
        return 0.0, 0.0

    # Site 1: primary orthosteric
    occ1 = c / (kd1 + c)

    # Site 2: allosteric, modulated by site 1 occupancy (cooperativity)
    kd2_eff = kd2 / (1.0 + alpha * occ1)
    occ2 = c / (kd2_eff + c)

    return occ1, occ2


def evaluate_two_site_binding(
    drug_name: str,
    kd1: float,
    kd2: float,
    emax1: float = 0.6,
    emax2: float = 0.4,
    alpha: float = 1.0,
    c_min: float = 1e-4,
    c_max: float = 100.0,
    n_points: int = 50,
) -> TwoSiteBindingResult:
    """Evaluate two-site binding model over a concentration range.

    Parameters
    ----------
    drug_name : Drug name.
    kd1 : Dissociation constant for site 1 (same units as c). Must be > 0.
    kd2 : Dissociation constant for site 2. Must be > 0.
    emax1 : Maximum effect via site 1 (0-1). Must be in (0, 1].
    emax2 : Maximum effect via site 2 (0-1). Must be in [0, 1].
    alpha : Cooperativity (>1 positive, <1 negative). Must be > 0.
    c_min : Minimum concentration for curve. Must be > 0.
    c_max : Maximum concentration. Must be > c_min.
    n_points : Number of points. Must be >= 2.

    Returns
    -------
    TwoSiteBindingResult
    """
    if kd1 <= 0:
        raise ValueError("kd1 must be > 0")
    if kd2 <= 0:
        raise ValueError("kd2 must be > 0")
    if not (0 < emax1 <= 1):
        raise ValueError("emax1 must be in (0, 1]")
    if not (0 <= emax2 <= 1):
        raise ValueError("emax2 must be in [0, 1]")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if c_min <= 0:
        raise ValueError("c_min must be > 0")
    if c_max <= c_min:
        raise ValueError("c_max must be > c_min")
    if n_points < 2:
        raise ValueError("n_points must be >= 2")

    # Log-spaced concentrations
    log_min = math.log10(c_min)
    log_max = math.log10(c_max)
    step = (log_max - log_min) / (n_points - 1)
    concs = [10 ** (log_min + i * step) for i in range(n_points)]

    occ1_list = []
    occ2_list = []
    total_eff = []

    for c in concs:
        o1, o2 = two_site_occupancy(c, kd1, kd2, alpha)
        occ1_list.append(o1)
        occ2_list.append(o2)
        total_eff.append(emax1 * o1 + emax2 * o2)

    # Apparent EC50: interpolate where total_effect = 0.5 * max_effect
    max_eff = max(total_eff)
    ec50_app = c_max  # default
    for i in range(1, n_points):
        if total_eff[i] >= 0.5 * max_eff:
            # Linear interpolation
            frac = (0.5 * max_eff - total_eff[i - 1]) / (total_eff[i] - total_eff[i - 1])
            ec50_app = concs[i - 1] + frac * (concs[i] - concs[i - 1])
            break

    # Apparent Hill: slope at EC50 using log-linear interpolation
    # Approximate: n_app = log(81) / log(C90/C10)
    target_10 = 0.1 * max_eff
    target_90 = 0.9 * max_eff
    c10, c90 = c_min, c_max
    for i in range(1, n_points):
        if total_eff[i] >= target_10 and total_eff[i - 1] < target_10:
            frac = (target_10 - total_eff[i - 1]) / (total_eff[i] - total_eff[i - 1])
            c10 = concs[i - 1] + frac * (concs[i] - concs[i - 1])
        if total_eff[i] >= target_90 and total_eff[i - 1] < target_90:
            frac = (target_90 - total_eff[i - 1]) / (total_eff[i] - total_eff[i - 1])
            c90 = concs[i - 1] + frac * (concs[i] - concs[i - 1])
    if c10 > 0 and c90 > c10:
        hill_app = math.log(81) / math.log(c90 / c10)
    else:
        hill_app = 1.0

    notes = (
        f"{drug_name}: Kd1={kd1:.3g}, Kd2={kd2:.3g}, alpha={alpha:.2f}. "
        f"EC50_apparent={ec50_app:.3g}, Hill_apparent={hill_app:.2f}."
    )

    return TwoSiteBindingResult(
        drug_name=drug_name,
        concentrations=concs,
        site1_occupancy=occ1_list,
        site2_occupancy=occ2_list,
        total_effect=total_eff,
        ec50_apparent=ec50_app,
        hill_apparent=hill_app,
        notes=notes,
    )
