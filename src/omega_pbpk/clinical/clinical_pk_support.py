"""Clinical PK decision support tools — Phase 504."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TherapeuticWindowResult:
    """Result of therapeutic window assessment."""

    cmin_mg_L: float
    cmax_mg_L: float
    mec_mg_L: float
    mtc_mg_L: float
    therapeutic_range_pct: float
    subtherapeutic: bool
    supratherapeutic: bool
    in_window: bool
    fluctuation_pct: float
    recommendation: str
    notes: str


def assess_therapeutic_window(
    cmin_mg_L: float,
    cmax_mg_L: float,
    mec_mg_L: float,
    mtc_mg_L: float,
) -> TherapeuticWindowResult:
    """Assess whether drug concentrations fall within the therapeutic window.

    Args:
        cmin_mg_L: Trough concentration (mg/L).
        cmax_mg_L: Peak concentration (mg/L).
        mec_mg_L: Minimum effective concentration (mg/L).
        mtc_mg_L: Maximum tolerated concentration (mg/L).

    Returns:
        TherapeuticWindowResult with assessment.

    Raises:
        ValueError: If inputs are invalid.
    """
    if cmin_mg_L < 0.0:
        raise ValueError(f"cmin_mg_L must be >= 0, got {cmin_mg_L}")
    if cmax_mg_L < 0.0:
        raise ValueError(f"cmax_mg_L must be >= 0, got {cmax_mg_L}")
    if cmax_mg_L < cmin_mg_L:
        raise ValueError(
            f"cmax_mg_L ({cmax_mg_L}) must be >= cmin_mg_L ({cmin_mg_L})"
        )
    if mec_mg_L < 0.0:
        raise ValueError(f"mec_mg_L must be >= 0, got {mec_mg_L}")
    if mtc_mg_L <= mec_mg_L:
        raise ValueError(
            f"mtc_mg_L ({mtc_mg_L}) must be > mec_mg_L ({mec_mg_L})"
        )

    subtherapeutic = cmax_mg_L < mec_mg_L
    supratherapeutic = cmin_mg_L > mtc_mg_L
    in_window = (cmin_mg_L >= mec_mg_L) and (cmax_mg_L <= mtc_mg_L)

    # Fluctuation: (Cmax - Cmin) / Cavg * 100, Cavg = (Cmax + Cmin) / 2
    if (cmax_mg_L + cmin_mg_L) > 0.0:
        cavg = (cmax_mg_L + cmin_mg_L) / 2.0
        fluctuation_pct = (cmax_mg_L - cmin_mg_L) / cavg * 100.0
    else:
        fluctuation_pct = 0.0

    # Estimate therapeutic_range_pct: fraction of dosing interval in window.
    # Approximation: linear interpolation between Cmin and Cmax within [MEC, MTC].
    total_range = cmax_mg_L - cmin_mg_L
    if total_range <= 0.0:
        # Constant concentration
        if mec_mg_L <= cmin_mg_L <= mtc_mg_L:
            therapeutic_range_pct = 100.0
        else:
            therapeutic_range_pct = 0.0
    else:
        # Fraction of [Cmin, Cmax] that overlaps with [MEC, MTC]
        overlap_low = max(cmin_mg_L, mec_mg_L)
        overlap_high = min(cmax_mg_L, mtc_mg_L)
        overlap = max(0.0, overlap_high - overlap_low)
        therapeutic_range_pct = (overlap / total_range) * 100.0

    # Build recommendation
    if in_window:
        recommendation = "Current regimen is within the therapeutic window. Continue."
    elif subtherapeutic:
        recommendation = (
            "Concentrations are subtherapeutic. Consider increasing dose or "
            "shortening dosing interval."
        )
    elif supratherapeutic:
        recommendation = (
            "Concentrations exceed the maximum tolerated concentration. "
            "Reduce dose or extend dosing interval."
        )
    else:
        # Partially overlapping — Cmin < MEC or Cmax > MTC but not both extremes
        if cmin_mg_L < mec_mg_L and cmax_mg_L <= mtc_mg_L:
            recommendation = (
                "Trough below MEC. Consider shortening dosing interval."
            )
        elif cmin_mg_L >= mec_mg_L and cmax_mg_L > mtc_mg_L:
            recommendation = (
                "Peak exceeds MTC. Consider reducing dose or extending interval."
            )
        else:
            recommendation = (
                "Concentrations partially overlap therapeutic window. "
                "Review dosing regimen."
            )

    note_parts = []
    if fluctuation_pct > 100.0:
        note_parts.append("high fluctuation (>100%) — consider modified-release formulation")
    if (mtc_mg_L / mec_mg_L) < 2.0:
        note_parts.append("narrow therapeutic index — close monitoring recommended")
    notes = "; ".join(note_parts) if note_parts else "no special flags"

    return TherapeuticWindowResult(
        cmin_mg_L=cmin_mg_L,
        cmax_mg_L=cmax_mg_L,
        mec_mg_L=mec_mg_L,
        mtc_mg_L=mtc_mg_L,
        therapeutic_range_pct=therapeutic_range_pct,
        subtherapeutic=subtherapeutic,
        supratherapeutic=supratherapeutic,
        in_window=in_window,
        fluctuation_pct=fluctuation_pct,
        recommendation=recommendation,
        notes=notes,
    )


def recommend_dose_timing(
    t_half_h: float,
    target_trough_mg_L: float,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    interval_options_h: list[float] | None = None,
) -> dict:
    """Recommend optimal dosing interval based on PK parameters.

    Uses 1-compartment model at steady state to evaluate interval options.

    Args:
        t_half_h: Half-life in hours.
        target_trough_mg_L: Target trough concentration (mg/L).
        dose_mg: Dose in mg.
        cl_L_per_h: Clearance in L/h.
        vd_L: Volume of distribution in L.
        interval_options_h: List of intervals to evaluate (default: [6, 8, 12, 24]).

    Returns:
        dict with recommended_interval_h, predicted_trough_mg_L, predicted_peak_mg_L,
        interval_evaluations (list of dicts), rationale.

    Raises:
        ValueError: If inputs are invalid.
    """
    if t_half_h <= 0.0:
        raise ValueError(f"t_half_h must be > 0, got {t_half_h}")
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0.0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0.0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if target_trough_mg_L < 0.0:
        raise ValueError(f"target_trough_mg_L must be >= 0, got {target_trough_mg_L}")

    if interval_options_h is None:
        interval_options_h = [6.0, 8.0, 12.0, 24.0]

    ke = math.log(2.0) / t_half_h  # elimination rate constant (1/h)

    evaluations = []
    best_interval = None
    best_score = float("inf")

    for tau in interval_options_h:
        if tau <= 0.0:
            continue
        # Steady-state 1-cpt IV bolus approximation
        # Css_peak = (Dose/Vd) / (1 - exp(-ke*tau))
        # Css_trough = Css_peak * exp(-ke*tau)
        exp_ke_tau = math.exp(-ke * tau)
        css_peak = (dose_mg / vd_L) / (1.0 - exp_ke_tau)
        css_trough = css_peak * exp_ke_tau

        # Score: absolute distance of trough from target
        score = abs(css_trough - target_trough_mg_L)

        evaluations.append(
            {
                "interval_h": tau,
                "predicted_peak_mg_L": css_peak,
                "predicted_trough_mg_L": css_trough,
                "trough_meets_target": css_trough >= target_trough_mg_L,
            }
        )

        if score < best_score:
            best_score = score
            best_interval = tau

    if best_interval is None:
        best_interval = interval_options_h[0]

    # Get best evaluation
    best_eval = next(e for e in evaluations if e["interval_h"] == best_interval)

    return {
        "recommended_interval_h": best_interval,
        "predicted_trough_mg_L": best_eval["predicted_trough_mg_L"],
        "predicted_peak_mg_L": best_eval["predicted_peak_mg_L"],
        "interval_evaluations": evaluations,
        "rationale": (
            f"Interval of {best_interval}h yields trough "
            f"{best_eval['predicted_trough_mg_L']:.3f} mg/L closest to target "
            f"{target_trough_mg_L} mg/L."
        ),
    }


def bayesian_dose_update(
    prior_cl_L_per_h: float,
    prior_vd_L: float,
    observed_conc_mg_L: float,
    observed_time_h: float,
    dose_mg: float,
    route: str = "iv",
    n_iter: int = 3,
) -> dict:
    """Update PK parameter estimates using simplified Bayesian approach.

    Uses 1-compartment analytical solution.
    For IV: C(t) = (Dose/Vd) * exp(-CL/Vd * t)
    Update rule: CL_new = CL_prior * (C_predicted / C_observed)^0.5 (damped)

    Args:
        prior_cl_L_per_h: Prior clearance estimate (L/h).
        prior_vd_L: Prior volume of distribution (L).
        observed_conc_mg_L: Observed concentration (mg/L).
        observed_time_h: Time of observation post-dose (h).
        dose_mg: Dose given (mg).
        route: 'iv' (default) — only IV supported currently.
        n_iter: Number of Bayesian update iterations.

    Returns:
        dict with posterior_cl_L_per_h, posterior_vd_L, predicted_conc_mg_L,
        observed_conc_mg_L, fold_change_cl, n_iterations, route.

    Raises:
        ValueError: If inputs are invalid.
    """
    if prior_cl_L_per_h <= 0.0:
        raise ValueError(f"prior_cl_L_per_h must be > 0, got {prior_cl_L_per_h}")
    if prior_vd_L <= 0.0:
        raise ValueError(f"prior_vd_L must be > 0, got {prior_vd_L}")
    if observed_conc_mg_L <= 0.0:
        raise ValueError(f"observed_conc_mg_L must be > 0, got {observed_conc_mg_L}")
    if observed_time_h <= 0.0:
        raise ValueError(f"observed_time_h must be > 0, got {observed_time_h}")
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    if route not in ("iv",):
        raise ValueError(f"route must be 'iv', got '{route}'")

    cl = prior_cl_L_per_h
    vd = prior_vd_L

    for _ in range(n_iter):
        # Predicted concentration using current estimates
        c_pred = (dose_mg / vd) * math.exp(-cl / vd * observed_time_h)

        if c_pred <= 0.0:
            break

        # Damped Bayesian update: CL_new = CL * (C_pred / C_obs)^0.5
        ratio = c_pred / observed_conc_mg_L
        cl = cl * (ratio ** 0.5)

    # Final predicted concentration with posterior CL
    c_pred_final = (dose_mg / vd) * math.exp(-cl / vd * observed_time_h)

    return {
        "posterior_cl_L_per_h": cl,
        "posterior_vd_L": vd,
        "predicted_conc_mg_L": c_pred_final,
        "observed_conc_mg_L": observed_conc_mg_L,
        "fold_change_cl": cl / prior_cl_L_per_h,
        "n_iterations": n_iter,
        "route": route,
    }


def population_adjustment(
    cl_typical: float,
    vd_typical: float,
    weight_kg: float = 70.0,
    age_years: float = 35.0,
    sex: str = "male",
    creatinine_umol_L: float = 80.0,
) -> dict:
    """Adjust typical PK parameters for individual patient characteristics.

    Uses allometric scaling for weight, age-based hepatic/renal decline,
    sex correction, and creatinine-based renal function.

    Args:
        cl_typical: Typical population clearance (L/h).
        vd_typical: Typical population volume of distribution (L).
        weight_kg: Patient weight (kg).
        age_years: Patient age (years).
        sex: 'male' or 'female'.
        creatinine_umol_L: Serum creatinine (umol/L).

    Returns:
        dict with adjusted_cl_L_per_h, adjusted_vd_L, weight_factor,
        age_factor, sex_factor, renal_factor, combined_factor, notes.

    Raises:
        ValueError: If inputs are invalid.
    """
    if cl_typical <= 0.0:
        raise ValueError(f"cl_typical must be > 0, got {cl_typical}")
    if vd_typical <= 0.0:
        raise ValueError(f"vd_typical must be > 0, got {vd_typical}")
    if weight_kg <= 0.0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if age_years < 0.0:
        raise ValueError(f"age_years must be >= 0, got {age_years}")
    if sex not in ("male", "female"):
        raise ValueError(f"sex must be 'male' or 'female', got '{sex}'")
    if creatinine_umol_L <= 0.0:
        raise ValueError(f"creatinine_umol_L must be > 0, got {creatinine_umol_L}")

    # Allometric weight scaling (exponent 0.75 for CL, 1.0 for Vd)
    ref_weight = 70.0
    weight_factor_cl = (weight_kg / ref_weight) ** 0.75
    weight_factor_vd = weight_kg / ref_weight

    # Age correction: linear decline above 40 years (1% per year, max 30%)
    if age_years > 40.0:
        age_decline = min(0.30, (age_years - 40.0) * 0.01)
        age_factor = 1.0 - age_decline
    else:
        age_factor = 1.0

    # Sex correction: females have ~15% lower CL (CYP3A4 activity)
    if sex == "female":
        sex_factor = 0.85
    else:
        sex_factor = 1.0

    # Renal function from serum creatinine (Cockcroft-Gault-inspired simplified)
    # GFR_norm ~ (140 - age) * weight / (creatinine_umol_L) — simplified
    # Normalize to typical GFR of ~100 mL/min
    if sex == "female":
        cr_factor_sex = 0.85
    else:
        cr_factor_sex = 1.0

    # Simplified: renal_factor = (140 - age) * weight * cr_factor_sex / (creatinine_umol_L * 0.815)
    # Clamp age contribution: min age contribution = 20 years equivalent
    age_contrib = max(20.0, 140.0 - age_years)
    # Typical GFR reference: 140-35 (ref age) * 70 * 1.0 / (80 * 0.815) ≈ 113.5
    gfr_estimate = age_contrib * weight_kg * cr_factor_sex / creatinine_umol_L
    # Normalize to typical (age=35, wt=70, cr=80, male)
    typical_gfr = (140.0 - 35.0) * 70.0 * 1.0 / 80.0  # = 91.875
    renal_factor = gfr_estimate / typical_gfr
    renal_factor = max(0.1, min(2.0, renal_factor))  # clamp

    # Combined factors
    combined_cl = cl_typical * weight_factor_cl * age_factor * sex_factor * renal_factor
    combined_vd = vd_typical * weight_factor_vd

    note_parts = []
    if weight_kg > 100.0:
        note_parts.append("obese patient — allometric scaling applied")
    if age_years > 65.0:
        note_parts.append("elderly patient — age-based CL reduction applied")
    if sex == "female":
        note_parts.append("female sex — 15% CL reduction applied")
    if creatinine_umol_L > 120.0:
        note_parts.append("elevated creatinine — renal impairment adjustment applied")
    notes = "; ".join(note_parts) if note_parts else "standard adjustment"

    return {
        "adjusted_cl_L_per_h": combined_cl,
        "adjusted_vd_L": combined_vd,
        "weight_factor": weight_factor_cl,
        "age_factor": age_factor,
        "sex_factor": sex_factor,
        "renal_factor": renal_factor,
        "combined_factor": combined_cl / cl_typical,
        "notes": notes,
    }
