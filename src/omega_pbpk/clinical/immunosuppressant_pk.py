"""Phase 582 — Immunosuppressant PK modeling.

Narrow therapeutic index drugs: tacrolimus, cyclosporine.
Pure Python, no numpy/scipy.
"""

from __future__ import annotations


def tacrolimus_trough_target(
    indication: str,
    months_post_transplant: int,
) -> dict:
    """Return tacrolimus trough target range.

    Parameters
    ----------
    indication : str
        'kidney', 'liver', or 'heart'
    months_post_transplant : int
        Months since transplant

    Returns
    -------
    dict with target_low_ng_mL, target_high_ng_mL, indication, months, notes
    """
    valid_indications = {"kidney", "liver", "heart"}
    if indication not in valid_indications:
        raise ValueError(f"indication must be one of {valid_indications}")
    if months_post_transplant < 0:
        raise ValueError("months_post_transplant must be >= 0")

    # Targets (ng/mL) by indication and time phase
    targets = {
        "kidney": {
            "early": (10.0, 15.0),
            "middle": (8.0, 12.0),
            "late": (5.0, 10.0),
        },
        "liver": {
            "early": (8.0, 12.0),
            "middle": (5.0, 10.0),
            "late": (3.0, 8.0),
        },
        "heart": {
            "early": (12.0, 18.0),
            "middle": (10.0, 15.0),
            "late": (8.0, 12.0),
        },
    }

    if months_post_transplant <= 3:
        phase = "early"
        phase_label = "0-3 months post-transplant"
    elif months_post_transplant <= 12:
        phase = "middle"
        phase_label = "3-12 months post-transplant"
    else:
        phase = "late"
        phase_label = ">12 months post-transplant"

    low, high = targets[indication][phase]

    return {
        "target_low_ng_mL": low,
        "target_high_ng_mL": high,
        "indication": indication,
        "months": months_post_transplant,
        "notes": (
            f"Tacrolimus trough target for {indication} transplant, {phase_label}. "
            f"Target: {low}-{high} ng/mL."
        ),
    }


def cyclosporine_dose_adjustment(
    current_trough_ng_mL: float,
    target_trough_ng_mL: float,
    current_dose_mg: float,
) -> dict:
    """Proportional cyclosporine dose adjustment based on trough level.

    Parameters
    ----------
    current_trough_ng_mL : float
        Measured trough concentration (ng/mL)
    target_trough_ng_mL : float
        Target trough concentration (ng/mL)
    current_dose_mg : float
        Current dose (mg)

    Returns
    -------
    dict with current_dose_mg, new_dose_mg, adjustment_pct, warning
    """
    if current_trough_ng_mL <= 0:
        raise ValueError("current_trough_ng_mL must be > 0")
    if target_trough_ng_mL <= 0:
        raise ValueError("target_trough_ng_mL must be > 0")
    if current_dose_mg <= 0:
        raise ValueError("current_dose_mg must be > 0")

    raw_new_dose = current_dose_mg * (target_trough_ng_mL / current_trough_ng_mL)
    adjustment_pct = ((raw_new_dose - current_dose_mg) / current_dose_mg) * 100.0

    warning = None

    # Cap adjustment at ±50%
    if adjustment_pct > 50.0:
        new_dose_mg = current_dose_mg * 1.5
        actual_adjustment_pct = 50.0
        warning = (
            "Calculated dose increase exceeded 50%; capped at +50%. "
            "Reassess trough after adjustment."
        )
    elif adjustment_pct < -50.0:
        new_dose_mg = current_dose_mg * 0.5
        actual_adjustment_pct = -50.0
        warning = (
            "Calculated dose decrease exceeded 50%; capped at -50%. "
            "Monitor for rejection or toxicity."
        )
    else:
        new_dose_mg = raw_new_dose
        actual_adjustment_pct = adjustment_pct

    return {
        "current_dose_mg": current_dose_mg,
        "new_dose_mg": round(new_dose_mg, 2),
        "adjustment_pct": round(actual_adjustment_pct, 2),
        "warning": warning,
    }


def simulate_nti_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    f_oral: float,
    therapeutic_low: float,
    therapeutic_high: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> dict:
    """Simulate 1-compartment oral PK for a narrow therapeutic index drug.

    Uses forward Euler integration with a simple first-order absorption model
    (ka = 1.0 /h assumed internally, dose depot → central).

    Parameters
    ----------
    drug_name : str
        Drug name (informational)
    dose_mg : float
        Oral dose (mg)
    cl_L_per_h : float
        Clearance (L/h)
    vd_L : float
        Volume of distribution (L)
    f_oral : float
        Oral bioavailability (0-1)
    therapeutic_low : float
        Therapeutic window lower bound (mg/L, same units as output)
    therapeutic_high : float
        Therapeutic window upper bound (mg/L)
    t_end_h : float
        Simulation end time (h)
    dt_h : float
        Time step (h)

    Returns
    -------
    dict with times_h, c_plasma_mg_L, cmax, trough, time_in_window_h,
         fraction_in_window, is_subtherapeutic, is_supratherapeutic
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (0.0 < f_oral <= 1.0):
        raise ValueError("f_oral must be in (0, 1]")
    if therapeutic_low >= therapeutic_high:
        raise ValueError("therapeutic_low must be < therapeutic_high")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")

    ka = 1.0  # /h — fixed absorption rate constant
    ke = cl_L_per_h / vd_L  # elimination rate constant (/h)

    # Initial conditions: drug in GI depot, zero in central compartment
    depot = f_oral * dose_mg  # mg in GI
    c_central = 0.0  # mg/L

    times_h = []
    c_plasma_mg_L = []
    time_in_window_h = 0.0

    t = 0.0
    n_steps = max(1, int(round(t_end_h / dt_h)))

    for _ in range(n_steps + 1):
        times_h.append(round(t, 6))
        c_plasma_mg_L.append(c_central)

        # Check therapeutic window at this time point
        if therapeutic_low <= c_central <= therapeutic_high:
            time_in_window_h += dt_h if _ > 0 else 0.0

        # Forward Euler
        d_depot = -ka * depot
        d_c = (ka * depot / vd_L) - ke * c_central

        depot += d_depot * dt_h
        depot = max(0.0, depot)
        c_central += d_c * dt_h
        c_central = max(0.0, c_central)
        t += dt_h

    cmax = max(c_plasma_mg_L)
    trough = c_plasma_mg_L[-1]
    total_time = t_end_h
    fraction_in_window = min(1.0, time_in_window_h / total_time) if total_time > 0 else 0.0

    is_subtherapeutic = trough < therapeutic_low
    is_supratherapeutic = cmax > therapeutic_high

    return {
        "times_h": times_h,
        "c_plasma_mg_L": c_plasma_mg_L,
        "cmax": cmax,
        "trough": trough,
        "time_in_window_h": round(time_in_window_h, 4),
        "fraction_in_window": round(fraction_in_window, 4),
        "is_subtherapeutic": is_subtherapeutic,
        "is_supratherapeutic": is_supratherapeutic,
    }


def tdm_dose_recommendation(
    trough_measured_ng_mL: float,
    target_range: tuple,
    current_dose_mg: float,
    drug: str = "tacrolimus",
) -> dict:
    """Recommend dose adjustment based on measured trough and target range.

    Parameters
    ----------
    trough_measured_ng_mL : float
        Measured trough concentration (ng/mL)
    target_range : tuple[float, float]
        (target_low_ng_mL, target_high_ng_mL)
    current_dose_mg : float
        Current dose (mg)
    drug : str
        Drug name (informational)

    Returns
    -------
    dict with recommendation, new_dose_mg, rationale
    """
    if trough_measured_ng_mL <= 0:
        raise ValueError("trough_measured_ng_mL must be > 0")
    if current_dose_mg <= 0:
        raise ValueError("current_dose_mg must be > 0")
    if len(target_range) != 2:
        raise ValueError("target_range must be a 2-element tuple (low, high)")

    target_low, target_high = target_range
    if target_low >= target_high:
        raise ValueError("target_range[0] must be < target_range[1]")

    def _round_to_half(x: float) -> float:
        return round(x * 2.0) / 2.0

    if trough_measured_ng_mL < target_low:
        # Proportional increase to mid-target
        target_mid = (target_low + target_high) / 2.0
        raw_new_dose = current_dose_mg * (target_mid / trough_measured_ng_mL)
        new_dose_mg = _round_to_half(raw_new_dose)
        recommendation = "increase"
        rationale = (
            f"Trough {trough_measured_ng_mL:.1f} ng/mL is below target "
            f"{target_low}-{target_high} ng/mL. "
            f"Increase dose from {current_dose_mg} mg to {new_dose_mg} mg."
        )
    elif trough_measured_ng_mL > target_high:
        # Proportional decrease to mid-target
        target_mid = (target_low + target_high) / 2.0
        raw_new_dose = current_dose_mg * (target_mid / trough_measured_ng_mL)
        new_dose_mg = _round_to_half(raw_new_dose)
        recommendation = "decrease"
        rationale = (
            f"Trough {trough_measured_ng_mL:.1f} ng/mL is above target "
            f"{target_low}-{target_high} ng/mL. "
            f"Decrease dose from {current_dose_mg} mg to {new_dose_mg} mg."
        )
    else:
        new_dose_mg = _round_to_half(current_dose_mg)
        recommendation = "maintain"
        rationale = (
            f"Trough {trough_measured_ng_mL:.1f} ng/mL is within target "
            f"{target_low}-{target_high} ng/mL. Maintain current dose."
        )

    return {
        "recommendation": recommendation,
        "new_dose_mg": new_dose_mg,
        "rationale": rationale,
    }


__all__ = [
    "tacrolimus_trough_target",
    "cyclosporine_dose_adjustment",
    "simulate_nti_pk",
    "tdm_dose_recommendation",
]
