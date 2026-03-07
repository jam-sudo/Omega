"""
Phase 570 — Therapeutic Index and Safety Margin Calculator

Functions for computing TI, CSF, safety margins, PK/PD safety windows,
NTI flags, and dose ranges from Hill equation models.
"""


def therapeutic_index(td50: float, ed50: float) -> float:
    """Compute the Therapeutic Index (TI = TD50 / ED50).

    Parameters
    ----------
    td50 : float
        Dose producing toxicity in 50% of subjects (must be > 0).
    ed50 : float
        Dose producing efficacy in 50% of subjects (must be > 0).

    Returns
    -------
    float
        TI = TD50 / ED50
    """
    if td50 <= 0:
        raise ValueError(f"td50 must be > 0, got {td50}")
    if ed50 <= 0:
        raise ValueError(f"ed50 must be > 0, got {ed50}")
    return td50 / ed50


def certain_safety_factor(ld1: float, ed99: float) -> float:
    """Compute the Certain Safety Factor (CSF = LD1 / ED99).

    More conservative than TI because it compares the lowest lethal dose
    with the highest effective dose.

    Parameters
    ----------
    ld1 : float
        Dose lethal to 1% of subjects (must be > 0).
    ed99 : float
        Dose effective in 99% of subjects (must be > 0).

    Returns
    -------
    float
        CSF = LD1 / ED99
    """
    if ld1 <= 0:
        raise ValueError(f"ld1 must be > 0, got {ld1}")
    if ed99 <= 0:
        raise ValueError(f"ed99 must be > 0, got {ed99}")
    return ld1 / ed99


def safety_margin(noael_mg_kg: float, efficacy_dose_mg_kg: float) -> dict:
    """Compute the safety margin between NOAEL and the efficacy dose.

    Parameters
    ----------
    noael_mg_kg : float
        No-Observed-Adverse-Effect Level in mg/kg (must be > 0).
    efficacy_dose_mg_kg : float
        Minimum efficacious dose in mg/kg (must be > 0).

    Returns
    -------
    dict with keys:
        safety_margin_pct : float  (percentage above efficacy dose)
        classification     : str   ('adequate', 'narrow', or 'inadequate')
    """
    if noael_mg_kg <= 0:
        raise ValueError(f"noael_mg_kg must be > 0, got {noael_mg_kg}")
    if efficacy_dose_mg_kg <= 0:
        raise ValueError(f"efficacy_dose_mg_kg must be > 0, got {efficacy_dose_mg_kg}")

    ratio = noael_mg_kg / efficacy_dose_mg_kg
    margin_pct = (noael_mg_kg - efficacy_dose_mg_kg) / efficacy_dose_mg_kg * 100.0

    if ratio >= 10.0:
        classification = "adequate"
    elif ratio >= 2.0:
        classification = "narrow"
    else:
        classification = "inadequate"

    return {
        "safety_margin_pct": margin_pct,
        "classification": classification,
    }


def pk_pd_safety_window(
    cmax_efficacy: float,
    cmax_toxicity: float,
    auc_efficacy: float,
    auc_toxicity: float,
) -> dict:
    """Compute PK/PD safety window from efficacy and toxicity exposures.

    Parameters
    ----------
    cmax_efficacy : float
        Cmax at minimum efficacious exposure (must be > 0).
    cmax_toxicity : float
        Cmax at toxicity threshold (must be > 0).
    auc_efficacy : float
        AUC at minimum efficacious exposure (must be > 0).
    auc_toxicity : float
        AUC at toxicity threshold (must be > 0).

    Returns
    -------
    dict with keys: cmax_window, auc_window, therapeutic_window, risk_level
    """
    for name, val in [
        ("cmax_efficacy", cmax_efficacy),
        ("cmax_toxicity", cmax_toxicity),
        ("auc_efficacy", auc_efficacy),
        ("auc_toxicity", auc_toxicity),
    ]:
        if val <= 0:
            raise ValueError(f"{name} must be > 0, got {val}")

    cmax_window = cmax_toxicity / cmax_efficacy
    auc_window = auc_toxicity / auc_efficacy
    therapeutic_window = min(cmax_window, auc_window)

    if therapeutic_window < 2.0:
        risk_level = "high"
    elif therapeutic_window <= 5.0:
        risk_level = "moderate"
    else:
        risk_level = "low"

    return {
        "cmax_window": cmax_window,
        "auc_window": auc_window,
        "therapeutic_window": therapeutic_window,
        "risk_level": risk_level,
    }


def nti_flag(ti: float, threshold: float = 2.0) -> bool:
    """Flag whether a drug is a Narrow Therapeutic Index (NTI) drug.

    Parameters
    ----------
    ti : float
        Therapeutic index value.
    threshold : float
        TI threshold below which (inclusive) a drug is classified as NTI.
        Default is 2.0.

    Returns
    -------
    bool
        True if TI <= threshold (NTI drug).
    """
    return ti <= threshold


def dose_range_to_achieve_window(
    ed50: float,
    hill: float,
    emax: float,
    min_effect: float,
    max_safe_effect: float,
) -> dict:
    """Find concentrations achieving min_effect and max_safe_effect via Hill equation.

    Hill equation: E = Emax * C^n / (EC50^n + C^n)
    Inverted:      C = EC50 * (E / (Emax - E))^(1/n)

    Parameters
    ----------
    ed50 : float
        EC50 / ED50 — concentration at half-maximum effect (mg/L, must be > 0).
    hill : float
        Hill coefficient n (must be > 0).
    emax : float
        Maximum achievable effect (must be > 0).
    min_effect : float
        Minimum required effect level (must be > 0 and < emax).
    max_safe_effect : float
        Maximum tolerated effect level (must be > min_effect and <= emax).

    Returns
    -------
    dict with keys: c_min_mg_L, c_max_mg_L, therapeutic_range_fold, notes
    """
    if ed50 <= 0:
        raise ValueError(f"ed50 must be > 0, got {ed50}")
    if hill <= 0:
        raise ValueError(f"hill must be > 0, got {hill}")
    if emax <= 0:
        raise ValueError(f"emax must be > 0, got {emax}")
    if min_effect <= 0 or min_effect >= emax:
        raise ValueError(
            f"min_effect must be in (0, emax). Got min_effect={min_effect}, emax={emax}"
        )
    if max_safe_effect <= min_effect or max_safe_effect > emax:
        raise ValueError(
            f"max_safe_effect must be in (min_effect, emax]. "
            f"Got max_safe_effect={max_safe_effect}, min_effect={min_effect}, emax={emax}"
        )

    def _invert_hill(effect: float) -> float:
        ratio = effect / (emax - effect)
        return ed50 * (ratio ** (1.0 / hill))

    c_min = _invert_hill(min_effect)

    # If max_safe_effect equals emax we cannot invert (division by zero → infinity)
    if max_safe_effect >= emax:
        c_max = float("inf")
        notes = "max_safe_effect equals Emax; upper concentration is unbounded"
        therapeutic_range_fold = float("inf")
    else:
        c_max = _invert_hill(max_safe_effect)
        therapeutic_range_fold = c_max / c_min if c_min > 0 else float("inf")
        notes = "Concentrations derived from Hill equation inversion"

    return {
        "c_min_mg_L": c_min,
        "c_max_mg_L": c_max,
        "therapeutic_range_fold": therapeutic_range_fold,
        "notes": notes,
    }
