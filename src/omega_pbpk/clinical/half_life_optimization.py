"""Drug half-life optimization for desired dosing regimens.

Focuses on the DESIGN aspects: how to modify a compound's t½ to achieve
a target dosing regimen using structure-property relationships.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "HalfLifeTargetResult",
    "OptimalRegimenResult",
    "estimate_t_half",
    "t_half_from_targets",
    "metabolic_stability_t_half_target",
    "design_t_half_strategies",
    "optimal_dosing_regimen",
]

# Candidate dosing intervals (hours) evaluated in order
_CANDIDATE_INTERVALS_H: list[float] = [8.0, 12.0, 24.0, 48.0]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HalfLifeTargetResult:
    """Recommended half-life range for drug design."""

    route: str
    dosing_frequency: str
    recommended_t_half_min_h: float
    recommended_t_half_max_h: float
    rationale: str


@dataclass(frozen=True)
class OptimalRegimenResult:
    """Result of optimal dosing regimen search."""

    t_half_h: float
    best_interval_h: float
    predicted_cmin_mg_L: float
    predicted_cmax_mg_L: float
    in_window: bool
    accumulation_ratio: float
    notes: str


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def estimate_t_half(cl_L_per_h: float, vd_L: float) -> float:
    """Estimate half-life from clearance and volume of distribution.

    t_half = 0.693 * Vd / CL

    Parameters
    ----------
    cl_L_per_h:
        Total body clearance in L/h. Must be > 0.
    vd_L:
        Volume of distribution in L. Must be > 0.

    Returns
    -------
    float
        Estimated half-life in hours.
    """
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")

    return 0.693 * vd_L / cl_L_per_h


def t_half_from_targets(
    target_dosing_interval_h: float,
    desired_accumulation_ratio: float = 2.0,
) -> float:
    """Back-calculate required t_half from desired dosing interval and accumulation ratio.

    Uses: R = 1 / (1 - exp(-ke * tau))
    Rearranged: ke = -ln(1 - 1/R) / tau
    Then: t_half = 0.693 / ke

    Parameters
    ----------
    target_dosing_interval_h:
        Desired dosing interval tau in hours (e.g., 24 for once-daily). Must be > 0.
    desired_accumulation_ratio:
        Target steady-state accumulation ratio R. Typical range 1.5–3x. Must be > 1.

    Returns
    -------
    float
        Required half-life in hours to achieve the desired accumulation.
    """
    if target_dosing_interval_h <= 0:
        raise ValueError(f"target_dosing_interval_h must be > 0, got {target_dosing_interval_h}")
    if desired_accumulation_ratio <= 1.0:
        raise ValueError(
            f"desired_accumulation_ratio must be > 1, got {desired_accumulation_ratio}"
        )

    tau = target_dosing_interval_h
    R = desired_accumulation_ratio
    ke = -math.log(1.0 - 1.0 / R) / tau
    return 0.693 / ke


# Lookup table for route/frequency → (min_h, max_h, rationale)
_T_HALF_TARGETS: dict[tuple[str, str], tuple[float, float, str]] = {
    ("oral", "once_daily"): (
        12.0,
        24.0,
        "Once-daily oral dosing requires t½ 12–24 h to maintain adequate trough levels "
        "and limit peak/trough fluctuation.",
    ),
    ("oral", "twice_daily"): (
        6.0,
        12.0,
        "Twice-daily oral dosing (q12h) benefits from t½ 6–12 h to control fluctuation "
        "while reaching steady state within 2–3 days.",
    ),
    ("iv", "once_daily"): (
        18.0,
        30.0,
        "Once-daily IV dosing typically requires a longer t½ (18–30 h) because there is "
        "no absorption phase to blunt peak concentrations.",
    ),
    ("long_acting_injection", "once_daily"): (
        120.0,
        336.0,
        "Long-acting injectable formulations target t½ of 5–14 days (120–336 h) to "
        "provide sustained systemic exposure between monthly or bi-weekly injections.",
    ),
    ("long_acting_injection", "once_weekly"): (
        120.0,
        336.0,
        "Long-acting injectable formulations target t½ of 5–14 days (120–336 h) to "
        "sustain exposure between weekly doses.",
    ),
}

_DEFAULT_T_HALF_TARGET = (
    12.0,
    24.0,
    "Default recommendation for standard oral dosing: t½ 12–24 h.",
)


def metabolic_stability_t_half_target(
    route: str = "oral",
    dosing_frequency: str = "once_daily",
) -> HalfLifeTargetResult:
    """Return recommended half-life range for drug design given route and frequency.

    Parameters
    ----------
    route:
        Administration route: 'oral', 'iv', 'long_acting_injection'.
    dosing_frequency:
        Target dosing frequency: 'once_daily', 'twice_daily', 'once_weekly'.

    Returns
    -------
    HalfLifeTargetResult
    """
    key = (route.lower(), dosing_frequency.lower())
    min_h, max_h, rationale = _T_HALF_TARGETS.get(key, _DEFAULT_T_HALF_TARGET)

    return HalfLifeTargetResult(
        route=route,
        dosing_frequency=dosing_frequency,
        recommended_t_half_min_h=min_h,
        recommended_t_half_max_h=max_h,
        rationale=rationale,
    )


def design_t_half_strategies(
    current_t_half_h: float,
    target_t_half_h: float,
) -> list[str]:
    """Generate medicinal chemistry strategies to adjust half-life toward target.

    Parameters
    ----------
    current_t_half_h:
        Current observed half-life in hours. Must be > 0.
    target_t_half_h:
        Desired target half-life in hours. Must be > 0.

    Returns
    -------
    list[str]
        List of design strategy recommendations.
    """
    if current_t_half_h <= 0:
        raise ValueError(f"current_t_half_h must be > 0, got {current_t_half_h}")
    if target_t_half_h <= 0:
        raise ValueError(f"target_t_half_h must be > 0, got {target_t_half_h}")

    strategies: list[str] = []

    if target_t_half_h > current_t_half_h:
        # Need to extend half-life
        ratio = target_t_half_h / current_t_half_h
        strategies.append(
            f"Target t½ ({target_t_half_h:.1f} h) > current t½ ({current_t_half_h:.1f} h) — "
            f"need ~{ratio:.1f}x extension."
        )
        strategies.append(
            "Reduce metabolic clearance: block metabolic soft spots, introduce fluorine or "
            "deuterium at sites of CYP oxidation, add steric shielding groups."
        )
        strategies.append(
            "Increase volume of distribution: increase lipophilicity (logP), add basic nitrogen "
            "to promote tissue sequestration, exploit albumin or alpha-1-acid glycoprotein binding."
        )
        strategies.append(
            "Reduce renal clearance: decrease polarity and hydrogen-bond donor count "
            "(PSA < 75 Å²), increase molecular weight above GFR filtration cut-off "
            "(~40 kDa for biologics)."
        )
        if ratio > 5:
            strategies.append(
                "Consider PEGylation or Fc fusion (biologics): PEG chains extend t½ 10–100x via "
                "FcRn recycling and reduced renal filtration."
            )
        if ratio > 10:
            strategies.append(
                "Explore depot/prodrug formulation: sustained-release or long-acting injectable "
                "can decouple pharmacokinetic t½ from dosing interval."
            )
    elif target_t_half_h < current_t_half_h:
        # Need to shorten half-life
        ratio = current_t_half_h / target_t_half_h
        strategies.append(
            f"Target t½ ({target_t_half_h:.1f} h) < current t½ ({current_t_half_h:.1f} h) — "
            f"need ~{ratio:.1f}x reduction."
        )
        strategies.append(
            "Increase metabolic clearance: introduce metabolic soft spots (unblocked "
            "para-position on aromatic ring), reduce steric shielding, add oxidisable groups."
        )
        strategies.append(
            "Reduce volume of distribution: decrease lipophilicity (logP), increase hydrophilicity "
            "(add polar groups, reduce aromatic ring count), limit tissue partitioning."
        )
        strategies.append(
            "Increase renal clearance: add ionisable group at physiological pH, reduce molecular "
            "weight below tubular reabsorption threshold."
        )
        if ratio > 3:
            strategies.append(
                "Consider soft-drug design: introduce an ester or other labile bond that undergoes "
                "controlled hydrolysis to an inactive metabolite."
            )
    else:
        strategies.append(
            f"Current t½ ({current_t_half_h:.1f} h) is already at target — no structural "
            "modification required."
        )

    return strategies


def _steady_state_cmin_cmax(
    t_half_h: float,
    dose_mg: float,
    vd_L: float,
    cl_L_per_h: float,
    interval_h: float,
    route: str,
    ka_per_h: float,
    n_points: int = 500,
) -> tuple[float, float, float]:
    """Compute steady-state Cmin, Cmax, and accumulation ratio using Forward Euler.

    Returns (Cmin_mg_L, Cmax_mg_L, accumulation_ratio).
    """
    ke = 0.693 / t_half_h
    dt = interval_h / n_points

    if route.lower() == "iv":
        # IV bolus: C(t) = (Dose/Vd) * exp(-ke*t)
        # Steady-state: Cmax_ss = (Dose/Vd) / (1 - exp(-ke*tau))
        c0 = dose_mg / vd_L
        r = math.exp(-ke * interval_h)
        cmax = c0 / (1.0 - r)
        cmin = cmax * r
        accum = cmax / c0
        return cmin, cmax, accum

    # Oral: 1-compartment forward Euler simulation
    # Simulate enough multidose cycles to reach steady state (~10 half-lives)
    n_cycles_ss = max(10, int(10.0 * t_half_h / interval_h) + 1)

    c_gut = 0.0
    c_plasma = 0.0
    # Run to steady state
    for _ in range(n_cycles_ss):
        c_gut += dose_mg / vd_L  # dose in "gut" units (mg/L equivalent)
        for _ in range(n_points):
            dc_gut = -ka_per_h * c_gut * dt
            dc_plasma = ka_per_h * c_gut * dt - ke * c_plasma * dt
            c_gut += dc_gut
            c_plasma += dc_plasma

    # Record one final cycle for Cmin/Cmax
    c_gut += dose_mg / vd_L
    cmax = c_plasma
    cmin = c_plasma

    for _ in range(n_points):
        dc_gut = -ka_per_h * c_gut * dt
        dc_plasma = ka_per_h * c_gut * dt - ke * c_plasma * dt
        c_gut += dc_gut
        c_plasma += dc_plasma
        if c_plasma > cmax:
            cmax = c_plasma
        if c_plasma < cmin:
            cmin = c_plasma

    # Accumulation ratio: Cmax_ss / Cmax_single_dose
    # Single dose Cmax (analytical approximation for 1-cpt oral)
    if abs(ka_per_h - ke) > 1e-9:
        tmax_single = math.log(ka_per_h / ke) / (ka_per_h - ke)
        cmax_single = (dose_mg / vd_L) * (ka_per_h / (ka_per_h - ke)) * (
            math.exp(-ke * tmax_single) - math.exp(-ka_per_h * tmax_single)
        )
        accum = cmax / cmax_single if cmax_single > 0 else 1.0
    else:
        cmax_single = (dose_mg / vd_L) * math.exp(-1.0)
        accum = cmax / cmax_single if cmax_single > 0 else 1.0

    return max(0.0, cmin), max(0.0, cmax), max(1.0, accum)


def optimal_dosing_regimen(
    t_half_h: float,
    therapeutic_window: tuple[float, float],
    dose_mg: float,
    vd_L: float,
    cl_L_per_h: float,
    route: str = "oral",
    ka_per_h: float = 1.5,
) -> OptimalRegimenResult:
    """Find the dosing interval that keeps steady-state concentrations within therapeutic window.

    Evaluates candidate intervals (q8h, q12h, q24h, q48h) and selects the one whose
    predicted steady-state Cmin/Cmax best fit the therapeutic window.

    Parameters
    ----------
    t_half_h:
        Drug half-life in hours. Must be > 0.
    therapeutic_window:
        (Cmin_target_mg_L, Cmax_target_mg_L) therapeutic window. Must have min < max.
    dose_mg:
        Dose in mg.
    vd_L:
        Volume of distribution in L. Must be > 0.
    cl_L_per_h:
        Clearance in L/h. Must be > 0.
    route:
        Route of administration: 'oral' or 'iv'.
    ka_per_h:
        Oral absorption rate constant in h⁻¹ (ignored for IV).

    Returns
    -------
    OptimalRegimenResult
    """
    if t_half_h <= 0:
        raise ValueError(f"t_half_h must be > 0, got {t_half_h}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if therapeutic_window[0] >= therapeutic_window[1]:
        raise ValueError(
            f"therapeutic_window lower bound must be < upper bound, got {therapeutic_window}"
        )

    c_min_target, c_max_target = therapeutic_window

    best_interval: float = _CANDIDATE_INTERVALS_H[0]
    best_cmin: float = 0.0
    best_cmax: float = 0.0
    best_accum: float = 1.0
    best_in_window = False
    best_score: float = float("inf")

    for interval_h in _CANDIDATE_INTERVALS_H:
        cmin, cmax, accum = _steady_state_cmin_cmax(
            t_half_h=t_half_h,
            dose_mg=dose_mg,
            vd_L=vd_L,
            cl_L_per_h=cl_L_per_h,
            interval_h=interval_h,
            route=route,
            ka_per_h=ka_per_h,
        )
        in_window = (cmin >= c_min_target) and (cmax <= c_max_target)

        # Score: penalise out-of-window concentrations; prefer in-window intervals
        score = 0.0
        if cmin < c_min_target:
            score += (c_min_target - cmin) / max(c_min_target, 1e-9)
        if cmax > c_max_target:
            score += (cmax - c_max_target) / max(c_max_target, 1e-9)

        if in_window and not best_in_window:
            best_in_window = True
            best_interval = interval_h
            best_cmin = cmin
            best_cmax = cmax
            best_accum = accum
            best_score = score
        elif in_window == best_in_window and score < best_score:
            best_interval = interval_h
            best_cmin = cmin
            best_cmax = cmax
            best_accum = accum
            best_score = score

    notes_parts: list[str] = [f"Best interval: q{int(best_interval)}h"]
    if best_in_window:
        notes_parts.append("Predicted SS concentrations within therapeutic window.")
    else:
        notes_parts.append(
            "No tested interval achieves concentrations fully within therapeutic window; "
            "consider dose adjustment."
        )

    return OptimalRegimenResult(
        t_half_h=t_half_h,
        best_interval_h=best_interval,
        predicted_cmin_mg_L=best_cmin,
        predicted_cmax_mg_L=best_cmax,
        in_window=best_in_window,
        accumulation_ratio=best_accum,
        notes="; ".join(notes_parts),
    )
