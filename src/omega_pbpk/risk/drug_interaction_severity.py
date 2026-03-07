"""
Phase 564 — Drug-drug interaction severity classification.

Pure Python, no numpy/scipy.
"""



def classify_ddi_severity(
    aucr: float,
    cmax_ratio: float = None,
) -> dict:
    """Classify DDI severity per FDA 2020 guidance.

    Parameters
    ----------
    aucr : AUC ratio (victim AUC with / without perpetrator)
    cmax_ratio : Cmax ratio (optional, same thresholds applied)

    Returns
    -------
    dict with keys: aucr, aucr_class, cmax_ratio, cmax_class (or None),
                    clinical_action, notes
    """

    def _classify(ratio: float) -> str:
        if ratio >= 5.0:
            return "strong"
        elif ratio >= 2.0:
            return "moderate"
        elif ratio >= 1.25:
            return "weak"
        else:
            return "negligible"

    _clinical_actions = {
        "strong": "Contraindicated or dose reduction required; consult prescribing information.",
        "moderate": "Monitor for adverse effects; consider dose adjustment.",
        "weak": "Monitor; dose adjustment generally not required.",
        "negligible": "No clinically relevant DDI expected.",
    }

    aucr_class = _classify(aucr)
    clinical_action = _clinical_actions[aucr_class]

    cmax_class = None
    if cmax_ratio is not None:
        cmax_class = _classify(cmax_ratio)

    notes = f"AUCR={aucr:.3f} classified as '{aucr_class}'." + (
        f" Cmax ratio={cmax_ratio:.3f} classified as '{cmax_class}'."
        if cmax_ratio is not None
        else ""
    )

    return {
        "aucr": aucr,
        "aucr_class": aucr_class,
        "cmax_ratio": cmax_ratio,
        "cmax_class": cmax_class,
        "clinical_action": clinical_action,
        "notes": notes,
    }


def static_r1_value(inhibitor_conc_uM: float, ki_uM: float) -> float:
    """Basic static R1 model for hepatic CYP inhibition.

    R1 = 1 + (I / Ki)

    Parameters
    ----------
    inhibitor_conc_uM : inhibitor concentration at enzyme site (uM)
    ki_uM : inhibition constant (uM)

    Returns
    -------
    float : R1 value (>= 1.0)
    """
    if ki_uM <= 0:
        raise ValueError("ki_uM must be positive (> 0)")
    if inhibitor_conc_uM < 0:
        raise ValueError("inhibitor_conc_uM must be non-negative")

    return 1.0 + inhibitor_conc_uM / ki_uM


def static_r1_gut(
    dose_mg: float,
    mw: float,
    ki_uM: float,
    fa: float = 1.0,
) -> float:
    """Static R1 model for gut CYP inhibition.

    I_gut = (dose_mg / mw * 1e6) / 250e3  [uM, assuming 250 mL gut volume]
    R1_gut = 1 + Fa * I_gut / Ki

    Parameters
    ----------
    dose_mg : oral dose in mg
    mw : molecular weight in g/mol (Da)
    ki_uM : inhibition constant (uM)
    fa : fraction absorbed (default 1.0)

    Returns
    -------
    float : R1_gut value (>= 1.0)
    """
    if ki_uM <= 0:
        raise ValueError("ki_uM must be positive")
    if dose_mg < 0:
        raise ValueError("dose_mg must be non-negative")
    if mw <= 0:
        raise ValueError("mw must be positive")
    if fa < 0 or fa > 1:
        raise ValueError("fa must be between 0 and 1")

    # Convert dose to micromoles, then to uM in 250 mL
    dose_umol = (dose_mg / mw) * 1e3  # dose in mg / (g/mol) * 1000 -> umol
    i_gut_uM = dose_umol / 0.250  # uM = umol / L (250 mL = 0.250 L)

    return 1.0 + fa * i_gut_uM / ki_uM


def mbi_r2_value(
    kinact_per_h: float,
    ki_MBI_uM: float,
    inhibitor_conc_uM: float,
    kdeg_per_h: float = 0.03,
) -> float:
    """Mechanism-based inhibitor R2 value.

    R2 = kdeg / (kdeg + kinact * I / (KI + I))

    Parameters
    ----------
    kinact_per_h : maximum inactivation rate constant (1/h)
    ki_MBI_uM : concentration for half-maximal inactivation (uM)
    inhibitor_conc_uM : inhibitor concentration (uM)
    kdeg_per_h : enzyme degradation rate constant (1/h, default 0.03)

    Returns
    -------
    float : R2 value (0 < R2 <= 1.0)
    """
    if kinact_per_h < 0:
        raise ValueError("kinact_per_h must be non-negative")
    if ki_MBI_uM <= 0:
        raise ValueError("ki_MBI_uM must be positive")
    if inhibitor_conc_uM < 0:
        raise ValueError("inhibitor_conc_uM must be non-negative")
    if kdeg_per_h <= 0:
        raise ValueError("kdeg_per_h must be positive")

    inactivation_rate = kinact_per_h * inhibitor_conc_uM / (ki_MBI_uM + inhibitor_conc_uM)
    return kdeg_per_h / (kdeg_per_h + inactivation_rate)


def induction_fold_change(
    emax_fold: float,
    ec50_uM: float,
    inducer_conc_uM: float,
) -> float:
    """Predict fold-induction of enzyme activity.

    Fold-induction = 1 + Emax * C / (EC50 + C)

    Parameters
    ----------
    emax_fold : maximum fold-induction (>= 0)
    ec50_uM : concentration for half-maximal induction (uM, > 0)
    inducer_conc_uM : inducer concentration (uM, >= 0)

    Returns
    -------
    float : fold-induction (>= 1.0)
    """
    if emax_fold < 0:
        raise ValueError("emax_fold must be non-negative")
    if ec50_uM <= 0:
        raise ValueError("ec50_uM must be positive")
    if inducer_conc_uM < 0:
        raise ValueError("inducer_conc_uM must be non-negative")

    return 1.0 + emax_fold * inducer_conc_uM / (ec50_uM + inducer_conc_uM)


def net_ddi_aucr(
    r1: float,
    r2: float,
    induction_fold: float = 1.0,
) -> float:
    """Calculate net AUCR from inhibition and induction components.

    Net AUCR = R1 / R2

    Parameters
    ----------
    r1 : reversible inhibition factor (>= 1.0)
    r2 : MBI factor (0 < R2 <= 1.0)
    induction_fold : fold-induction (unused in final ratio per FDA guidance)

    Returns
    -------
    float : net AUCR (R1 / R2)
    """
    if r2 <= 0:
        raise ValueError("r2 must be positive")
    return r1 / r2


__all__ = [
    "classify_ddi_severity",
    "static_r1_value",
    "static_r1_gut",
    "mbi_r2_value",
    "induction_fold_change",
    "net_ddi_aucr",
]
