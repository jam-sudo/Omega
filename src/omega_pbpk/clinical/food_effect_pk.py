"""Phase 468 — Food Effect on Drug PK.

1-compartment oral PK model with food-effect multipliers on ka, F, and lag time.
BCS class determines sensitivity to food effects.
"""

from __future__ import annotations

from dataclasses import dataclass

# Meal type multipliers: (ka_mult, fa_mult_high_sol, fa_mult_low_sol, lag_h)
# fa_mult differs by BCS class: BCS I/II (high perm) use one set; III/IV use another
_MEAL_KA_MULT: dict[str, float] = {
    "fasted": 1.0,
    "high_fat": 0.5,
    "low_fat": 0.7,
    "standard": 0.8,
}

_MEAL_FA_MULT_HIGH: dict[str, float] = {
    "fasted": 1.0,
    "high_fat": 1.3,
    "low_fat": 1.1,
    "standard": 1.05,
}

_MEAL_FA_MULT_LOW: dict[str, float] = {
    "fasted": 1.0,
    "high_fat": 0.7,
    "low_fat": 1.1,
    "standard": 1.05,
}

_MEAL_LAG_H: dict[str, float] = {
    "fasted": 0.0,
    "high_fat": 1.0,
    "low_fat": 0.5,
    "standard": 0.3,
}

_VALID_MEAL_TYPES = {"fasted", "high_fat", "low_fat", "standard"}
_VALID_BCS_CLASSES = {1, 2, 3, 4}


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Manual trapezoidal AUC."""
    return sum(
        0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def _clinical_significance(pct_change: float) -> str:
    """Classify food effect magnitude."""
    abs_change = abs(pct_change)
    if abs_change < 10.0:
        return "none"
    if abs_change < 25.0:
        return "minor"
    if abs_change < 50.0:
        return "moderate"
    return "major"


def _simulate_single(
    dose_mg: float,
    ka: float,
    f_adj: float,
    k_el: float,
    vd_sys_L: float,
    lag_h: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[list[float], list[float]]:
    """Forward Euler 1-cpt oral simulation with lag time.

    M_gut absorbed only after lag time.
    Returns (times, c_plasma_list).
    """
    # Initial gut mass (full dose, absorption starts after lag)
    m_gut = dose_mg  # mg (not yet adjusted by F; F applied at absorption step)
    c_plasma = 0.0

    times: list[float] = [0.0]
    c_plasma_list: list[float] = [0.0]

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps):
        if t >= lag_h:
            dm_gut = -ka * m_gut
            dc_plasma = ka * m_gut * f_adj / vd_sys_L - k_el * c_plasma
        else:
            dm_gut = 0.0
            dc_plasma = -k_el * c_plasma

        m_gut = max(0.0, m_gut + dm_gut * dt_h)
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
        t = round(t + dt_h, 10)

        times.append(t)
        c_plasma_list.append(c_plasma)

    return times, c_plasma_list


@dataclass
class FoodEffectResult:
    """Result of a food-effect PK simulation."""

    drug_name: str
    dose_mg: float
    bcs_class: int
    meal_type: str
    times_h: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    f_adjusted: float
    ka_adjusted_per_h: float
    food_effect_on_cmax_pct: float
    food_effect_on_auc_pct: float
    fasted_cmax_mg_L: float
    fasted_auc_mg_h_per_L: float
    clinical_significance: str
    notes: str


def simulate_food_effect_pk(
    drug_name: str,
    dose_mg: float,
    bcs_class: int,
    meal_type: str,
    ka_fasted_per_h: float = 1.0,
    f_fasted: float = 0.7,
    cl_sys_L_per_h: float = 5.0,
    vd_sys_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> FoodEffectResult:
    """Simulate 1-compartment oral PK with food effect.

    Food effect multipliers applied to ka, F (bioavailability), and lag time
    based on meal_type. BCS class determines fa multiplier for high_fat meal.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if bcs_class not in _VALID_BCS_CLASSES:
        raise ValueError(f"bcs_class must be one of {_VALID_BCS_CLASSES}")
    if meal_type not in _VALID_MEAL_TYPES:
        raise ValueError(f"meal_type must be one of {_VALID_MEAL_TYPES}")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if not (0 < f_fasted <= 1):
        raise ValueError("f_fasted must be in (0, 1]")

    k_el = cl_sys_L_per_h / vd_sys_L
    ka_mult = _MEAL_KA_MULT[meal_type]
    lag_h = _MEAL_LAG_H[meal_type]

    # BCS I/II: high solubility or high permeability (1=high sol+perm, 2=low sol+high perm)
    # Food increases absorption for BCS I/II, decreases for BCS III/IV
    if bcs_class in (1, 2):
        fa_mult = _MEAL_FA_MULT_HIGH[meal_type]
    else:
        fa_mult = _MEAL_FA_MULT_LOW[meal_type]

    ka_adj = ka_fasted_per_h * ka_mult
    f_adj = min(1.0, f_fasted * fa_mult)

    # Simulate current meal condition
    times, c_plasma = _simulate_single(
        dose_mg=dose_mg,
        ka=ka_adj,
        f_adj=f_adj,
        k_el=k_el,
        vd_sys_L=vd_sys_L,
        lag_h=lag_h,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    cmax = max(c_plasma)
    tmax = times[c_plasma.index(cmax)]
    auc = _trapezoidal_auc(times, c_plasma)

    # Simulate fasted as reference
    ka_fasted_adj = ka_fasted_per_h * _MEAL_KA_MULT["fasted"]
    f_fasted_adj = min(1.0, f_fasted * _MEAL_FA_MULT_HIGH["fasted"])
    lag_fasted = _MEAL_LAG_H["fasted"]

    times_f, c_plasma_f = _simulate_single(
        dose_mg=dose_mg,
        ka=ka_fasted_adj,
        f_adj=f_fasted_adj,
        k_el=k_el,
        vd_sys_L=vd_sys_L,
        lag_h=lag_fasted,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )
    fasted_cmax = max(c_plasma_f)
    fasted_auc = _trapezoidal_auc(times_f, c_plasma_f)

    if fasted_cmax > 0:
        cmax_pct = (cmax - fasted_cmax) / fasted_cmax * 100.0
    else:
        cmax_pct = 0.0

    if fasted_auc > 0:
        auc_pct = (auc - fasted_auc) / fasted_auc * 100.0
    else:
        auc_pct = 0.0

    significance = _clinical_significance(auc_pct)

    notes = (
        f"BCS class {bcs_class}; ka_adj={ka_adj:.3f} h^-1; "
        f"f_adj={f_adj:.3f}; lag={lag_h:.2f} h; k_el={k_el:.4f} h^-1"
    )

    return FoodEffectResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        bcs_class=bcs_class,
        meal_type=meal_type,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        f_adjusted=f_adj,
        ka_adjusted_per_h=ka_adj,
        food_effect_on_cmax_pct=cmax_pct,
        food_effect_on_auc_pct=auc_pct,
        fasted_cmax_mg_L=fasted_cmax,
        fasted_auc_mg_h_per_L=fasted_auc,
        clinical_significance=significance,
        notes=notes,
    )


def compare_meal_types(
    drug_name: str,
    dose_mg: float,
    bcs_class: int,
    ka_fasted_per_h: float = 1.0,
    f_fasted: float = 0.7,
    cl_sys_L_per_h: float = 5.0,
    vd_sys_L: float = 50.0,
) -> list[FoodEffectResult]:
    """Compare all meal types for a drug, sorted by AUC descending."""
    results = []
    for meal_type in _VALID_MEAL_TYPES:
        result = simulate_food_effect_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            bcs_class=bcs_class,
            meal_type=meal_type,
            ka_fasted_per_h=ka_fasted_per_h,
            f_fasted=f_fasted,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_sys_L=vd_sys_L,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_mg_h_per_L, reverse=True)
    return results
