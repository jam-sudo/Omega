"""Enteral nutrition - drug interaction PK model.

Models how tube feeding (enteral nutrition) affects drug absorption
through protein binding, divalent cation chelation, and pH changes.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["EnteralNutritionPKResult", "simulate_en_pk", "compare_feeding_modes"]

_VALID_CATEGORIES = {
    "high_interaction",
    "moderate_interaction",
    "low_interaction",
    "no_interaction",
}

_VALID_FEEDING_MODES = {"continuous", "cyclic", "bolus_feeds", "fasted"}

# Interaction factors: fraction of normal absorption LOST due to EN
_INTERACTION_FACTORS = {
    "high_interaction": 0.65,  # 35% of normal absorbed
    "moderate_interaction": 0.35,  # 65% of normal absorbed
    "low_interaction": 0.10,  # 90% of normal absorbed
    "no_interaction": 0.0,  # 100% of normal absorbed
}

# Cyclic EN: 16h on, 8h off; drug given at start of off period
_CYCLIC_ON_H = 16.0
_CYCLIC_PERIOD_H = 24.0


@dataclass
class EnteralNutritionPKResult:
    """Result of enteral nutrition PK simulation."""

    drug_name: str
    interaction_category: str
    feeding_mode: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    fa_effective: float
    auc_ratio_vs_fasted: float
    interaction_factor: float
    notes: str


def _en_active_at(t: float, feeding_mode: str) -> bool:
    """Return True if EN is currently active (interacting with drug) at time t."""
    if feeding_mode == "fasted":
        return False
    if feeding_mode == "continuous":
        return True
    if feeding_mode == "cyclic":
        # 16h on, 8h off; drug given at start of 8h off window
        # off period: [0..8h] within each 24h cycle
        t_mod = t % _CYCLIC_PERIOD_H
        return t_mod >= 8.0  # EN on from hour 8 to 24 (16h)
    if feeding_mode == "bolus_feeds":
        # Bolus feeds: EN given 4x/day for 30 min each → 2h/day active
        # Simplified: EN active at 0,6,12,18h for 0.5h windows
        t_mod = t % 6.0
        return t_mod < 0.5
    return False


def _effective_fa_factor(t: float, interaction_category: str, feeding_mode: str) -> float:
    """Return the absorption multiplier at time t (1 = full, <1 = reduced by EN)."""
    base_factor = _INTERACTION_FACTORS[interaction_category]
    if _en_active_at(t, feeding_mode):
        return 1.0 - base_factor
    return 1.0


def _simulate_pk(
    dose_mg: float,
    interaction_category: str,
    feeding_mode: str,
    ka_per_h: float,
    cl_L_per_h: float,
    vd_L: float,
    dosing_interval_h: float,
    n_doses: int,
    t_end_h: float,
    dt_h: float,
) -> tuple:
    """Run forward Euler 1-cpt PK simulation. Returns (times, concentrations)."""
    ka = ka_per_h
    ke = cl_L_per_h / vd_L

    n_steps = int(t_end_h / dt_h) + 1
    times = [i * dt_h for i in range(n_steps)]

    # Build dosing schedule
    dose_times = [i * dosing_interval_h for i in range(n_doses)]

    # State: gut amount (mg) and plasma concentration (mg/L)
    a_gut = 0.0
    c_plasma = 0.0
    concentrations = []

    for step in range(n_steps):
        t = times[step]

        # Apply doses
        for dt_dose in dose_times:
            if abs(t - dt_dose) < dt_h * 0.5:
                a_gut += dose_mg

        concentrations.append(c_plasma)

        # Effective absorption at this time
        fa_factor = _effective_fa_factor(t, interaction_category, feeding_mode)

        # ODE: forward Euler
        # dA_gut/dt = -ka * A_gut  (gut empties regardless; fa_factor applied to how much reaches plasma)
        # dC/dt = ka * fa_factor * A_gut / Vd - ke * C
        dA_gut = -ka * a_gut * dt_h
        dC = (ka * fa_factor * a_gut / vd_L - ke * c_plasma) * dt_h

        a_gut = max(0.0, a_gut + dA_gut)
        c_plasma = max(0.0, c_plasma + dC)

    return times, concentrations


def _compute_auc(times: list, concentrations: list) -> float:
    """Trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (concentrations[i - 1] + concentrations[i]) * (times[i] - times[i - 1])
    return auc


def _compute_fa_effective(
    interaction_category: str, feeding_mode: str, t_end_h: float, dt_h: float
) -> float:
    """Compute time-averaged effective fa fraction over simulation period."""
    n_steps = int(t_end_h / dt_h) + 1
    total_factor = sum(
        _effective_fa_factor(i * dt_h, interaction_category, feeding_mode) for i in range(n_steps)
    )
    return total_factor / n_steps


def simulate_en_pk(
    drug_name: str,
    dose_mg: float,
    interaction_category: str = "moderate_interaction",
    feeding_mode: str = "continuous",
    ka_per_h: float = 1.0,
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    dosing_interval_h: float = 12.0,
    n_doses: int = 3,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> EnteralNutritionPKResult:
    """Simulate drug PK with enteral nutrition interaction.

    Parameters
    ----------
    drug_name:
        Name of the drug being simulated.
    dose_mg:
        Dose in milligrams.
    interaction_category:
        One of high_interaction, moderate_interaction, low_interaction, no_interaction.
    feeding_mode:
        One of continuous, cyclic, bolus_feeds, fasted.
    ka_per_h:
        First-order absorption rate constant (1/h).
    cl_L_per_h:
        Clearance (L/h).
    vd_L:
        Volume of distribution (L).
    dosing_interval_h:
        Interval between doses (h).
    n_doses:
        Number of doses.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Integration step size (h).

    Returns
    -------
    EnteralNutritionPKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be positive")
    if vd_L <= 0:
        raise ValueError("vd_L must be positive")
    if interaction_category not in _VALID_CATEGORIES:
        raise ValueError(f"interaction_category must be one of {_VALID_CATEGORIES}")
    if feeding_mode not in _VALID_FEEDING_MODES:
        raise ValueError(f"feeding_mode must be one of {_VALID_FEEDING_MODES}")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")

    times, concentrations = _simulate_pk(
        dose_mg,
        interaction_category,
        feeding_mode,
        ka_per_h,
        cl_L_per_h,
        vd_L,
        dosing_interval_h,
        n_doses,
        t_end_h,
        dt_h,
    )

    auc = _compute_auc(times, concentrations)
    cmax = max(concentrations) if concentrations else 0.0
    tmax = times[concentrations.index(cmax)] if concentrations else 0.0

    # Compute fasted AUC for ratio
    times_fasted, conc_fasted = _simulate_pk(
        dose_mg,
        interaction_category,
        "fasted",
        ka_per_h,
        cl_L_per_h,
        vd_L,
        dosing_interval_h,
        n_doses,
        t_end_h,
        dt_h,
    )
    auc_fasted = _compute_auc(times_fasted, conc_fasted)
    auc_ratio = auc / auc_fasted if auc_fasted > 0 else 1.0

    fa_effective = _compute_fa_effective(interaction_category, feeding_mode, t_end_h, dt_h)
    interaction_factor = _INTERACTION_FACTORS[interaction_category]

    # Build notes
    notes = (
        f"{drug_name}: {interaction_category.replace('_', ' ')} with {feeding_mode} EN. "
        f"Effective absorption {fa_effective * 100:.1f}% of normal. "
        f"AUC ratio vs fasted = {auc_ratio:.2f}."
    )

    return EnteralNutritionPKResult(
        drug_name=drug_name,
        interaction_category=interaction_category,
        feeding_mode=feeding_mode,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=concentrations,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        fa_effective=fa_effective,
        auc_ratio_vs_fasted=auc_ratio,
        interaction_factor=interaction_factor,
        notes=notes,
    )


def compare_feeding_modes(
    drug_name: str,
    dose_mg: float,
    interaction_category: str = "moderate_interaction",
    ka_per_h: float = 1.0,
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    dosing_interval_h: float = 12.0,
    n_doses: int = 3,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> list[EnteralNutritionPKResult]:
    """Compare all 4 feeding modes and return results sorted by AUC descending.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams.
    interaction_category:
        One of high_interaction, moderate_interaction, low_interaction, no_interaction.

    Returns
    -------
    List of EnteralNutritionPKResult sorted by AUC (highest first).
    """
    results = []
    for mode in _VALID_FEEDING_MODES:
        result = simulate_en_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            interaction_category=interaction_category,
            feeding_mode=mode,
            ka_per_h=ka_per_h,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            dosing_interval_h=dosing_interval_h,
            n_doses=n_doses,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_mg_h_per_L, reverse=True)
    return results
