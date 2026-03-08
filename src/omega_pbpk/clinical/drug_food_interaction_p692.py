"""Phase 692 — Drug-Food Interaction Kinetics model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DrugFoodInteractionResult:
    """Result of drug-food interaction simulation."""

    drug_name: str
    dose_mg: float
    food_state: str
    times_h: list = field(default_factory=list)
    c_plasma_fasted_mg_L: list = field(default_factory=list)
    c_plasma_fed_mg_L: list = field(default_factory=list)
    cmax_fasted_mg_L: float = 0.0
    cmax_fed_mg_L: float = 0.0
    tmax_fasted_h: float = 0.0
    tmax_fed_h: float = 0.0
    auc_fasted_mg_h_per_L: float = 0.0
    auc_fed_mg_h_per_L: float = 0.0
    food_effect_cmax: float = 0.0
    food_effect_auc: float = 0.0
    bcs_class: str = ""
    recommendation: str = ""
    notes: str = ""


_VALID_FOOD_STATES = {"high_fat", "low_fat", "fasted"}


def _classify_bcs(solubility_mg_mL: float, logP: float) -> str:
    """Classify drug by BCS system."""
    high_sol = solubility_mg_mL >= 1.0
    high_perm = logP > 1.0
    if high_sol and high_perm:
        return "BCS I"
    if not high_sol and high_perm:
        return "BCS II"
    if high_sol and not high_perm:
        return "BCS III"
    return "BCS IV"


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Manual trapezoidal AUC."""
    return sum(
        0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def _simulate_1cpt(
    ka: float, a_gut_0: float, ke: float, vd: float, t_end_h: float, dt_h: float
) -> tuple:
    """Run 1-compartment Forward Euler simulation. Returns (times, concs, cmax, tmax)."""
    n_steps = int(t_end_h / dt_h) + 1
    times = []
    concs = []
    a_gut = a_gut_0
    a_central = 0.0
    cmax = 0.0
    tmax = 0.0

    for i in range(n_steps):
        t = i * dt_h
        c = a_central / vd
        if c < 0:
            c = 0.0
        times.append(round(t, 6))
        concs.append(c)
        if c > cmax:
            cmax = c
            tmax = t

        da_gut = -ka * a_gut
        da_central = ka * a_gut - ke * a_central
        a_gut += da_gut * dt_h
        a_central += da_central * dt_h

    return times, concs, cmax, tmax


def simulate_drug_food_interaction(
    drug_name: str,
    dose_mg: float,
    logP: float,
    solubility_mg_mL: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    food_state: str = "high_fat",
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> DrugFoodInteractionResult:
    """Simulate pharmacokinetic drug-food interactions."""

    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be positive")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be positive")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be positive")
    if food_state not in _VALID_FOOD_STATES:
        raise ValueError(
            f"food_state must be one of {sorted(_VALID_FOOD_STATES)}, got '{food_state}'"
        )

    # BCS classification
    bcs_class = _classify_bcs(solubility_mg_mL, logP)

    # Elimination rate
    ke = cl_sys_L_per_h / vd_sys_L

    # Fasted parameters
    ka_fasted = 1.0  # /h
    f_fasted = min(1.0, max(0.1, 0.5 + logP * 0.05))
    a_gut_fasted = dose_mg * 0.85 * f_fasted

    # Fasted simulation
    times, c_fasted, cmax_fasted, tmax_fasted = _simulate_1cpt(
        ka_fasted, a_gut_fasted, ke, vd_sys_L, t_end_h, dt_h
    )
    auc_fasted = _trapezoidal_auc(times, c_fasted)

    # Fed parameters
    if food_state == "fasted":
        # No food effect
        c_fed = list(c_fasted)
        cmax_fed = cmax_fasted
        tmax_fed = tmax_fasted
        auc_fed = auc_fasted
    else:
        # Determine food effect magnitude
        if food_state == "high_fat":
            effect_scale = 1.0
        else:  # low_fat
            effect_scale = 0.5

        # ka reduction (slower gastric emptying)
        ka_reduction = 0.4 + (1.0 - 0.4) * (1.0 - effect_scale)
        ka_fed = ka_fasted * ka_reduction

        # Solubility enhancement
        sol_enhancement = 1.0 + max(0.0, logP) * 2.0 * effect_scale
        solubility_fed = solubility_mg_mL * sol_enhancement

        # Bioavailability
        f_fed = min(1.0, f_fasted * (solubility_fed / solubility_mg_mL) ** 0.3)

        a_gut_fed = dose_mg * 0.85 * f_fed

        _, c_fed, cmax_fed, tmax_fed = _simulate_1cpt(
            ka_fed, a_gut_fed, ke, vd_sys_L, t_end_h, dt_h
        )
        auc_fed = _trapezoidal_auc(times, c_fed)

    # Food effect ratios
    food_effect_cmax = cmax_fed / cmax_fasted if cmax_fasted > 0 else 1.0
    food_effect_auc = auc_fed / auc_fasted if auc_fasted > 0 else 1.0

    # Recommendation
    if food_effect_auc > 1.25:
        recommendation = "Take with food"
    elif food_effect_auc < 0.8:
        recommendation = "Take on empty stomach"
    else:
        recommendation = "May be taken with or without food"

    notes_parts = [
        f"BCS: {bcs_class}",
        f"Food state: {food_state}",
        f"F_fasted: {f_fasted:.3f}",
        f"Food effect on AUC: {food_effect_auc:.2f}x",
    ]

    return DrugFoodInteractionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        food_state=food_state,
        times_h=times,
        c_plasma_fasted_mg_L=c_fasted,
        c_plasma_fed_mg_L=c_fed,
        cmax_fasted_mg_L=cmax_fasted,
        cmax_fed_mg_L=cmax_fed,
        tmax_fasted_h=tmax_fasted,
        tmax_fed_h=tmax_fed,
        auc_fasted_mg_h_per_L=auc_fasted,
        auc_fed_mg_h_per_L=auc_fed,
        food_effect_cmax=food_effect_cmax,
        food_effect_auc=food_effect_auc,
        bcs_class=bcs_class,
        recommendation=recommendation,
        notes="; ".join(notes_parts),
    )
