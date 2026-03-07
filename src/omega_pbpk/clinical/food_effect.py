"""Food effect model for oral drug PK — Phase 257.

Models how food (high-fat meal, low-fat meal, fasting) alters drug absorption
kinetics for BCS Class II/IV drugs. Based on FDA 2002 food-effect guidance.

References
----------
- FDA CDER. Food-Effect Bioavailability and Fed Bioequivalence Studies. 2002.
- Fleisher D et al., Clin Pharmacokinet. 1999;36(3):233-54
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FoodEffectResult:
    """Result from food effect simulation."""

    drug_name: str
    fed_state: str  # "fasted", "low_fat", "high_fat"
    dose_mg: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    cmax: float
    tmax_h: float
    auc: float
    f_relative: float  # AUC_fed / AUC_fasted (1.0 for fasted)
    cmax_ratio: float  # Cmax_fed / Cmax_fasted
    food_effect_category: str  # "none", "moderate", "significant"
    recommendation: str
    notes: str


# Food state PK multipliers (ka, F, lag_time additions)
_FOOD_PARAMS: dict[str, dict] = {
    "fasted": {"ka_mult": 1.0, "f_mult": 1.0, "lag_h": 0.25},
    "low_fat": {"ka_mult": 0.7, "f_mult": 1.2, "lag_h": 0.75},
    "high_fat": {"ka_mult": 0.5, "f_mult": 1.5, "lag_h": 1.5},
}


def simulate_food_effect(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_base_per_h: float = 1.0,
    f_oral_fasted: float = 0.5,
    bcs_class: str = "II",
    fed_state: str = "fasted",
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> FoodEffectResult:
    """Simulate oral PK under different fed states.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Dose (mg). Must be > 0.
    cl_L_per_h : Systemic clearance (L/h). Must be > 0.
    vd_L : Volume of distribution (L). Must be > 0.
    ka_base_per_h : Base oral absorption rate constant (fasted, h^-1). Must be > 0.
    f_oral_fasted : Oral bioavailability in fasted state (0, 1]. Must be in (0, 1].
    bcs_class : BCS class ('I', 'II', 'III', 'IV'). Determines food sensitivity.
    fed_state : 'fasted', 'low_fat', or 'high_fat'.
    t_end_h : Simulation end time (h). Must be > 0.
    dt_h : Integration time step (h).

    Returns
    -------
    FoodEffectResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if ka_base_per_h <= 0:
        raise ValueError("ka_base_per_h must be > 0")
    if not (0 < f_oral_fasted <= 1):
        raise ValueError("f_oral_fasted must be in (0, 1]")
    if fed_state not in _FOOD_PARAMS:
        raise ValueError(f"fed_state must be one of {list(_FOOD_PARAMS.keys())}")
    if bcs_class not in ("I", "II", "III", "IV"):
        raise ValueError("bcs_class must be 'I', 'II', 'III', or 'IV'")

    params = _FOOD_PARAMS[fed_state]

    # BCS class modulates food sensitivity
    # BCS I: food has minimal effect; BCS II: dissolution-limited, food enhances
    # BCS III: permeability-limited; BCS IV: both
    bcs_f_mult_scale = {"I": 0.3, "II": 1.0, "III": 0.5, "IV": 0.8}[bcs_class]

    ka_eff = ka_base_per_h * params["ka_mult"]
    f_eff = min(1.0, f_oral_fasted * (1.0 + (params["f_mult"] - 1.0) * bcs_f_mult_scale))
    lag_h = params["lag_h"]

    ke = cl_L_per_h / vd_L

    n_steps = int(t_end_h / dt_h) + 1
    times = [i * dt_h for i in range(n_steps)]
    c_plasma = [0.0] * n_steps
    a_gut = f_eff * dose_mg

    for i in range(1, n_steps):
        t_cur = times[i - 1]
        cp = c_plasma[i - 1]

        if t_cur >= lag_h:
            abs_rate = ka_eff * a_gut
        else:
            abs_rate = 0.0

        elim = ke * cp * vd_L
        dc = (abs_rate - elim) / vd_L * dt_h
        a_gut = max(0.0, a_gut - abs_rate * dt_h)
        c_plasma[i] = max(0.0, cp + dc)

    cmax = max(c_plasma)
    tmax = times[c_plasma.index(cmax)]
    auc = sum(0.5 * (c_plasma[j - 1] + c_plasma[j]) * dt_h for j in range(1, n_steps))

    # Relative metrics (fasted = reference)
    # Approximate fasted AUC for ratio: AUC proportional to F * dose / CL
    auc_fasted_approx = f_oral_fasted * dose_mg / cl_L_per_h
    auc_current = f_eff * dose_mg / cl_L_per_h
    f_relative = auc_current / auc_fasted_approx if auc_fasted_approx > 0 else 1.0

    # Cmax ratio: Cmax proportional to ka * F
    cmax_fasted_approx_ratio = ka_eff / ka_base_per_h * (f_eff / f_oral_fasted)
    cmax_ratio = cmax_fasted_approx_ratio if fed_state != "fasted" else 1.0

    # Food effect category
    f_diff_pct = abs(f_relative - 1.0) * 100
    if f_diff_pct < 20:
        category = "none"
    elif f_diff_pct < 40:
        category = "moderate"
    else:
        category = "significant"

    if category == "none":
        recommendation = (
            "No clinically meaningful food effect. Can be taken without regard to food."
        )
    elif category == "moderate":
        recommendation = (
            "Moderate food effect. Consider standardizing administration "
            "(consistently fed or fasted)."
        )
    else:
        recommendation = (
            f"Significant food effect (AUC ratio {f_relative:.2f}x). "
            "Specify fasting/fed conditions in labeling; consider formulation optimization."
        )

    notes = (
        f"{drug_name} [{fed_state}]: AUC ratio={f_relative:.2f}, "
        f"Cmax ratio={cmax_ratio:.2f}. "
        f"Food effect: {category} (BCS class {bcs_class})."
    )

    return FoodEffectResult(
        drug_name=drug_name,
        fed_state=fed_state,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        cmax=cmax,
        tmax_h=tmax,
        auc=auc,
        f_relative=f_relative,
        cmax_ratio=cmax_ratio,
        food_effect_category=category,
        recommendation=recommendation,
        notes=notes,
    )


def compare_fed_states(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    **kwargs: object,
) -> list[FoodEffectResult]:
    """Simulate PK in all three fed states for comparison.

    Returns list of [fasted, low_fat, high_fat] results.
    """
    return [
        simulate_food_effect(drug_name, dose_mg, cl_L_per_h, vd_L, fed_state=state, **kwargs)
        for state in ("fasted", "low_fat", "high_fat")
    ]


__all__ = ["FoodEffectResult", "simulate_food_effect", "compare_fed_states"]
