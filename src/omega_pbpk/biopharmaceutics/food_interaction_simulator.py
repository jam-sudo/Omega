"""Mechanistic food-drug interaction simulator — Phase 314.

Models how co-administration with food alters oral drug PK via:
- Bile salt enhancement of solubility (especially BCS II/IV)
- Delayed gastric emptying → lower Cmax, prolonged Tmax
- Altered intestinal transit affecting absorption rate
- pH changes affecting ionisation and permeability

References
----------
- FDA Guidance: Food-Effect Bioavailability and Fed Bioequivalence Studies. 2002.
- Singh BN. Effects of food on clinical pharmacokinetics. Clin Pharmacokinet. 1999;37(3):213-55.
- Fleisher D et al. Drug, meal and formulation interactions influencing drug
  absorption after oral administration. Clin Pharmacokinet. 1999;36(3):233-54.
- Pouton CW. Formulation of poorly water-soluble drugs for oral administration.
  Eur J Pharm Sci. 2006;29(3-4):278-87.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fasted reference GI parameters
_FASTED_BILE_mM = 1.0
_FASTED_GE_T50_H = 0.5

# BCS class permeability thresholds (rough Papp cut-offs)
_HIGH_PERM_THRESHOLD = 1e-6  # cm/s (> this → high permeability, BCS I or II)

# Bile micellar solubility enhancement factor per mM bile salt
# Lipid-soluble (BCS II/IV) drugs get ~5–15x enhancement at 8 mM bile
# Hydrophilic (BCS I/III) drugs get minimal enhancement
_BILE_ENHANCEMENT_PER_mM = {
    1: 0.05,   # BCS I: minimal
    2: 1.5,    # BCS II: large enhancement
    3: 0.05,   # BCS III: minimal
    4: 1.0,    # BCS IV: moderate
}

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MealComposition:
    """Nutritional and biopharmaceutic properties of a meal type."""

    meal_type: str
    fat_g: float
    protein_g: float
    carb_g: float
    total_kcal: float
    gastric_emptying_t50_h: float
    bile_conc_mM: float


@dataclass(frozen=True)
class FoodInteractionResult:
    """Result of a mechanistic food-drug interaction simulation."""

    drug_name: str
    dose_mg: float
    bcs_class: int
    meal_type: str
    auc_ratio: float          # AUC_fed / AUC_fasted
    cmax_ratio: float         # Cmax_fed / Cmax_fasted
    tmax_fed_h: float
    tmax_fasted_h: float
    food_effect_class: str    # none / weak / moderate / strong
    times_h: tuple[float, ...]
    c_plasma_fed: tuple[float, ...]
    c_plasma_fasted: tuple[float, ...]
    notes: str


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_dose(dose_mg: float) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")


def _validate_bcs(bcs_class: int) -> None:
    if bcs_class not in (1, 2, 3, 4):
        raise ValueError(f"bcs_class must be 1, 2, 3, or 4, got {bcs_class}")


def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def meal_composition(meal_type: str) -> MealComposition:
    """Return biopharmaceutic properties of a standard meal type.

    Parameters
    ----------
    meal_type : str
        One of 'fasted', 'low_fat', 'standard_fda', 'high_fat_fda'.

    Returns
    -------
    MealComposition

    Raises
    ------
    ValueError
        If meal_type is not recognised.
    """
    params: dict[str, dict[str, float]] = {
        "fasted": {
            "fat_g": 0.0,
            "protein_g": 0.0,
            "carb_g": 0.0,
            "total_kcal": 0.0,
            "gastric_emptying_t50_h": 0.5,
            "bile_conc_mM": 1.0,
        },
        "low_fat": {
            "fat_g": 5.0,
            "protein_g": 15.0,
            "carb_g": 60.0,
            "total_kcal": 300.0,
            "gastric_emptying_t50_h": 1.5,
            "bile_conc_mM": 3.0,
        },
        "standard_fda": {
            "fat_g": 20.0,
            "protein_g": 20.0,
            "carb_g": 40.0,
            "total_kcal": 400.0,
            "gastric_emptying_t50_h": 2.0,
            "bile_conc_mM": 5.0,
        },
        "high_fat_fda": {
            "fat_g": 50.0,
            "protein_g": 25.0,
            "carb_g": 58.0,
            "total_kcal": 1000.0,
            "gastric_emptying_t50_h": 3.0,
            "bile_conc_mM": 8.0,
        },
    }
    if meal_type not in params:
        raise ValueError(
            f"Unknown meal_type {meal_type!r}. "
            f"Supported: {sorted(params)}"
        )
    p = params[meal_type]
    return MealComposition(
        meal_type=meal_type,
        fat_g=p["fat_g"],
        protein_g=p["protein_g"],
        carb_g=p["carb_g"],
        total_kcal=p["total_kcal"],
        gastric_emptying_t50_h=p["gastric_emptying_t50_h"],
        bile_conc_mM=p["bile_conc_mM"],
    )


def _effective_solubility(
    solubility_water_mg_mL: float,
    bile_conc_mM: float,
    bcs_class: int,
) -> float:
    """Estimate effective GI solubility including bile micellar enhancement (mg/mL)."""
    enhancement_rate = _BILE_ENHANCEMENT_PER_mM[bcs_class]
    # Solubility = S_water * (1 + enhancement_rate * bile_conc_mM)
    return solubility_water_mg_mL * (1.0 + enhancement_rate * bile_conc_mM)


def _ka_from_solubility_papp(
    effective_sol_mg_mL: float,
    papp_cm_s: float,
    dose_mg: float,
    bcs_class: int,
) -> float:
    """Estimate first-order absorption rate constant (per h) from Papp and solubility.

    Uses a simplified approach:
    - If solubility-limited (BCS II/IV): ka scales with effective solubility
    - If permeability-limited (BCS III): ka scales with Papp
    - BCS I: governed by Papp (high solubility, high permeability)

    Papp → ka via intestinal surface model:
        ka ≈ 2 * Papp * (surface factor) / r_intestine
    where r_intestine ~ 1.25 cm, surface factor ~ 500 (villi), t_transit ~ 3.5 h.

    Simplified: ka_perm = Papp (cm/s) * 3600 (s/h) * 2 / 1.25 (cm)
              = Papp * 5760 (1/h)
    """
    # Permeability-limited ka (1/h)
    ka_perm = papp_cm_s * 5760.0  # converts cm/s → h^-1

    # Dissolution rate constant (simplified Noyes-Whitney)
    # kd ∝ effective_sol / dose_mg (if not dissolved fast, rate limits)
    # Normalise: 1 mg/mL solubility with 100 mg dose → kd ≈ 1/h
    kd_norm = (effective_sol_mg_mL * 100.0) / max(dose_mg, 1.0)  # dimensionless scale
    kd = kd_norm  # proxy for dissolution rate in h^-1

    if bcs_class == 1:
        # Both high — limited by permeability (ka typically 1-3/h)
        ka = min(ka_perm, 3.0)
    elif bcs_class == 2:
        # Low solubility: dissolution limited, enhanced by bile
        ka = min(kd, ka_perm)
    elif bcs_class == 3:
        # High solubility, low permeability → permeability limited
        ka = min(ka_perm, kd * 2.0)
    else:
        # BCS IV: both low → most conservative
        ka = min(kd, ka_perm) * 0.5

    # Practical bounds
    return float(np.clip(ka, 0.05, 5.0))


def _simulate_1cpt_oral(
    dose_mg: float,
    ka: float,
    cl_L_per_h: float,
    vd_L: float,
    bioavailability: float,
    t_lag_h: float,
    times_h: np.ndarray,
) -> np.ndarray:
    """Simulate 1-compartment oral PK using Forward Euler.

    Parameters
    ----------
    dose_mg : float
    ka : float  Absorption rate (h^-1)
    cl_L_per_h : float
    vd_L : float
    bioavailability : float  (0-1)
    t_lag_h : float  Lag time before absorption begins
    times_h : np.ndarray

    Returns
    -------
    np.ndarray  Plasma concentration (mg/L)
    """
    ke = cl_L_per_h / vd_L  # elimination rate constant (h^-1)

    dt = times_h[1] - times_h[0] if len(times_h) > 1 else 0.05

    # State: [A_gut (mg), A_central (mg)]
    a_gut = bioavailability * dose_mg
    a_central = 0.0
    conc = np.zeros(len(times_h))

    t_sim = 0.0

    # Use fine internal dt for accuracy
    dt_internal = min(dt / 10.0, 0.005)
    t_targets = list(times_h)
    target_idx = 0

    while target_idx < len(t_targets):
        # Record at target time
        if abs(t_sim - t_targets[target_idx]) < dt_internal * 0.5:
            conc[target_idx] = max(0.0, a_central / vd_L)
            target_idx += 1
            if target_idx >= len(t_targets):
                break

        # Forward Euler step
        if t_sim >= t_lag_h and a_gut > 0.0:
            d_gut = -ka * a_gut * dt_internal
            d_central = ka * a_gut * dt_internal - ke * a_central * dt_internal
        else:
            d_gut = 0.0
            d_central = -ke * a_central * dt_internal

        a_gut = max(0.0, a_gut + d_gut)
        a_central = max(0.0, a_central + d_central)
        t_sim += dt_internal

    # Fill any remaining indices (should not happen normally)
    for i in range(target_idx, len(t_targets)):
        conc[i] = max(0.0, a_central / vd_L)

    return conc


def _bioavailability_from_bcs(
    bcs_class: int,
    effective_sol_mg_mL: float,
    dose_mg: float,
    papp_cm_s: float,
) -> float:
    """Estimate oral bioavailability fraction from BCS class + solubility."""
    # Dose number Do: Do = dose_mg / (solubility * 250 mL)
    # High if Do < 1; low if Do > 1
    gfr_vol_mL = 250.0
    dose_number = dose_mg / max(effective_sol_mg_mL * gfr_vol_mL, 1e-9)

    # Absorption number An: An = Papp * t_transit / R_intestine
    # simplified: if Papp > 1e-6 cm/s → high perm → An > 1
    perm_factor = min(1.0, papp_cm_s / _HIGH_PERM_THRESHOLD)

    if bcs_class == 1:
        # High sol, high perm → F~0.7–0.95
        f = 0.85 * perm_factor
    elif bcs_class == 2:
        # Low sol, high perm → F limited by dissolution
        dissolution_f = min(1.0, 1.0 / max(dose_number, 0.1))
        f = dissolution_f * perm_factor * 0.80
    elif bcs_class == 3:
        # High sol, low perm → F limited by permeability
        f = perm_factor * 0.60
    else:
        # BCS IV: both low → poorest absorption
        dissolution_f = min(1.0, 1.0 / max(dose_number, 0.1))
        f = dissolution_f * perm_factor * 0.40

    return float(np.clip(f, 0.01, 1.0))


def _nca_metrics(
    times_h: np.ndarray,
    conc: np.ndarray,
) -> tuple[float, float, float]:
    """Return (Cmax, Tmax, AUC_last) from concentration-time profile."""
    cmax_idx = int(np.argmax(conc))
    cmax = float(conc[cmax_idx])
    tmax = float(times_h[cmax_idx])
    auc = float(np.trapz(conc, times_h))
    return cmax, tmax, auc


def food_effect_on_pk(
    drug_name: str,
    dose_mg: float,
    bcs_class: int,
    logP: float,
    pka: float,
    solubility_water_mg_mL: float,
    cl_L_per_h: float,
    vd_L: float,
    meal_type: str = "high_fat_fda",
    papp_cm_s: float = 5e-6,
) -> FoodInteractionResult:
    """Simulate food effect on oral drug PK via mechanistic model.

    Computes plasma concentration-time profiles in fasted and fed states and
    derives AUC ratio, Cmax ratio, and food effect classification.

    Parameters
    ----------
    drug_name : str
    dose_mg : float  Oral dose in mg (> 0)
    bcs_class : int  BCS class (1, 2, 3, or 4)
    logP : float  Octanol-water partition coefficient
    pka : float  Acid dissociation constant (use most acidic/basic pKa)
    solubility_water_mg_mL : float  Aqueous solubility at pH 7.4 (mg/mL, > 0)
    cl_L_per_h : float  Systemic clearance (L/h, > 0)
    vd_L : float  Volume of distribution (L, > 0)
    meal_type : str  One of 'fasted', 'low_fat', 'standard_fda', 'high_fat_fda'
    papp_cm_s : float  Caco-2/MDCK apparent permeability (cm/s, > 0)

    Returns
    -------
    FoodInteractionResult
    """
    _validate_dose(dose_mg)
    _validate_bcs(bcs_class)
    _validate_positive(solubility_water_mg_mL, "solubility_water_mg_mL")
    _validate_positive(cl_L_per_h, "cl_L_per_h")
    _validate_positive(vd_L, "vd_L")
    _validate_positive(papp_cm_s, "papp_cm_s")

    meal = meal_composition(meal_type)
    fasted_meal = meal_composition("fasted")

    # Simulation time grid
    t_end = max(24.0, 6.0 * vd_L / cl_L_per_h)
    dt = 0.1
    times_h = np.arange(0.0, t_end + dt, dt)

    # ----- Fasted state -----
    sol_fasted = _effective_solubility(
        solubility_water_mg_mL, fasted_meal.bile_conc_mM, bcs_class
    )
    ka_fasted = _ka_from_solubility_papp(sol_fasted, papp_cm_s, dose_mg, bcs_class)
    f_fasted = _bioavailability_from_bcs(bcs_class, sol_fasted, dose_mg, papp_cm_s)
    # Gastric lag: fasted ~ 0.25 h (rapid emptying)
    lag_fasted = fasted_meal.gastric_emptying_t50_h * 0.5

    conc_fasted = _simulate_1cpt_oral(
        dose_mg=dose_mg,
        ka=ka_fasted,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        bioavailability=f_fasted,
        t_lag_h=lag_fasted,
        times_h=times_h,
    )

    # ----- Fed state -----
    sol_fed = _effective_solubility(
        solubility_water_mg_mL, meal.bile_conc_mM, bcs_class
    )
    # Delayed gastric emptying slows effective ka
    ge_delay_factor = fasted_meal.gastric_emptying_t50_h / meal.gastric_emptying_t50_h
    ka_fed_base = _ka_from_solubility_papp(sol_fed, papp_cm_s, dose_mg, bcs_class)
    # Apply GE delay: slower emptying reduces effective absorption rate
    ka_fed = ka_fed_base * (0.5 + 0.5 * ge_delay_factor)

    f_fed = _bioavailability_from_bcs(bcs_class, sol_fed, dose_mg, papp_cm_s)
    # Fed lag time: 0.5 * gastric_emptying_t50
    lag_fed = meal.gastric_emptying_t50_h * 0.5

    conc_fed = _simulate_1cpt_oral(
        dose_mg=dose_mg,
        ka=ka_fed,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        bioavailability=f_fed,
        t_lag_h=lag_fed,
        times_h=times_h,
    )

    # ----- NCA metrics -----
    cmax_fasted, tmax_fasted, auc_fasted = _nca_metrics(times_h, conc_fasted)
    cmax_fed, tmax_fed, auc_fed = _nca_metrics(times_h, conc_fed)

    auc_ratio = auc_fed / max(auc_fasted, 1e-12)
    cmax_ratio = cmax_fed / max(cmax_fasted, 1e-12)

    food_class = classify_food_effect(auc_ratio, cmax_ratio)

    # Build notes
    notes_parts: list[str] = []
    if bcs_class == 2:
        notes_parts.append(
            f"BCS II: bile enhances solubility {sol_fed / max(sol_fasted, 1e-12):.1f}x"
        )
    elif bcs_class == 4:
        notes_parts.append("BCS IV: both low solubility and permeability")
    if meal.gastric_emptying_t50_h > 1.0:
        notes_parts.append(
            f"gastric emptying delayed to t50={meal.gastric_emptying_t50_h:.1f} h"
        )
    notes = "; ".join(notes_parts) if notes_parts else "standard food interaction"

    return FoodInteractionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        bcs_class=bcs_class,
        meal_type=meal_type,
        auc_ratio=round(auc_ratio, 4),
        cmax_ratio=round(cmax_ratio, 4),
        tmax_fed_h=round(tmax_fed, 3),
        tmax_fasted_h=round(tmax_fasted, 3),
        food_effect_class=food_class,
        times_h=tuple(float(t) for t in times_h),
        c_plasma_fed=tuple(float(c) for c in conc_fed),
        c_plasma_fasted=tuple(float(c) for c in conc_fasted),
        notes=notes,
    )


def classify_food_effect(auc_ratio: float, cmax_ratio: float) -> str:
    """Classify food effect magnitude per EMA guidance.

    Parameters
    ----------
    auc_ratio : float
        AUC_fed / AUC_fasted
    cmax_ratio : float
        Cmax_fed / Cmax_fasted

    Returns
    -------
    str
        'none', 'weak', 'moderate', or 'strong'
    """
    # Strong: AUC ratio < 0.5 or > 2.0
    if auc_ratio < 0.5 or auc_ratio > 2.0:
        return "strong"

    # Moderate: AUC ratio 0.5-0.8 or 1.5-2.0
    if auc_ratio < 0.8 or auc_ratio > 1.5:
        return "moderate"

    # Weak: AUC ratio 0.8-1.5 with Cmax outside 0.8-1.25
    if cmax_ratio < 0.8 or cmax_ratio > 2.0:
        return "moderate"  # Cmax also matters for classification

    if 0.8 <= auc_ratio <= 1.25 and 0.8 <= cmax_ratio <= 1.25:
        return "none"

    # AUC within 0.8-1.5 and Cmax within 0.5-2.0 → weak
    if cmax_ratio >= 0.5:
        return "weak"

    return "moderate"
