"""Phase 1128: Mechanistic oral bioavailability prediction.

Predicts oral bioavailability F = fa * fg * fh from drug physicochemical
properties and pharmacokinetic parameters.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OralBioavailabilityResult:
    drug_name: str
    dose_mg: float
    solubility_mg_mL: float
    logP: float
    fu_plasma: float
    bcs_class: str
    do_value: float
    peff_cm_per_s: float
    fa: float
    fg: float
    fh: float
    f_oral_predicted: float
    f_oral_pct: float
    bioavailability_class: str
    limiting_factor: str
    notes: str


def _determine_bcs_class(do_value: float, peff_cm_per_s: float) -> str:
    """Classify drug using BCS system."""
    high_sol = do_value <= 1.0  # Do <= 1 means high solubility
    high_perm = peff_cm_per_s > 1e-4

    if high_sol and high_perm:
        return "I"
    elif not high_sol and high_perm:
        return "II"
    elif high_sol and not high_perm:
        return "III"
    else:
        return "IV"


def _compute_fa(bcs_class: str, do_value: float, peff_cm_per_s: float) -> float:
    """Compute fraction absorbed (fa)."""
    if bcs_class in ("I", "II"):
        fa = min(1.0, 0.9 / (1.0 + 0.1 * do_value))
    else:
        # BCS III or IV: low Peff
        fa = 0.3

    # Override: high Peff boosts absorption
    if peff_cm_per_s > 5e-4:
        fa = min(1.0, fa * 1.5)

    return fa


def _compute_fg(logP: float) -> float:
    """Compute gut wall fraction (fg = 1 - EG)."""
    # EG = logP-based gut wall extraction
    if logP < 1.0:
        eg = 0.0
    elif logP > 9.0:
        eg = 0.8
    else:
        eg = max(0.0, min(0.8, 0.1 * logP - 0.1))
    return 1.0 - eg


def _compute_fh(cl_L_per_h: float, fu_plasma: float) -> float:
    """Compute hepatic first-pass fraction (fh = 1 - EH)."""
    QH = 90.0  # L/h hepatic blood flow
    # Simplified: if CL_total/fu > Qh → high extraction
    if (cl_L_per_h / fu_plasma) > QH:
        fh = 0.2
    else:
        fh = 1.0 - cl_L_per_h / QH
    return max(0.0, min(1.0, fh))


def _bioavailability_class(f: float) -> str:
    if f < 0.30:
        return "low"
    elif f <= 0.70:
        return "moderate"
    else:
        return "high"


def _limiting_factor(fa: float, fg: float, fh: float) -> str:
    """Identify the most limiting step."""
    factors = {"absorption": fa, "gut_extraction": fg, "hepatic_extraction": fh}
    return min(factors, key=lambda k: factors[k])


def predict_oral_bioavailability(
    drug_name: str,
    dose_mg: float,
    solubility_mg_mL: float,
    logP: float,
    pka: float,
    ionization_type: str,
    fu_plasma: float,
    cl_L_per_h: float,
    mw_Da: float,
) -> OralBioavailabilityResult:
    """Predict oral bioavailability from drug properties.

    Args:
        drug_name: Name of the drug.
        dose_mg: Dose in mg.
        solubility_mg_mL: Aqueous solubility (mg/mL).
        logP: Lipophilicity (octanol-water partition coefficient).
        pka: pKa value.
        ionization_type: "acid", "base", or "neutral".
        fu_plasma: Unbound fraction in plasma (0, 1].
        cl_L_per_h: Total plasma clearance (L/h).
        mw_Da: Molecular weight (Da).

    Returns:
        OralBioavailabilityResult with F prediction and mechanistic breakdown.
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")
    if not (0 < fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be in (0, 1]")
    if not (-5 <= logP <= 10):
        raise ValueError("logP must be in [-5, 10]")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")

    # Dose number: Do = dose / (solubility * 250 mL)
    do_value = dose_mg / (solubility_mg_mL * 250.0)

    # Peff estimation from logP (simplified empirical relationship)
    # Peff = 10^(-4.36 + 0.11 * logP) cm/s
    # At logP=0: ~4.4e-5; at logP=4: ~1.2e-4 (high perm threshold 1e-4)
    peff_cm_per_s = 10.0 ** (-4.36 + 0.11 * logP)

    # BCS classification
    bcs_class = _determine_bcs_class(do_value, peff_cm_per_s)

    # Mechanistic components
    fa = _compute_fa(bcs_class, do_value, peff_cm_per_s)
    fg = _compute_fg(logP)
    fh = _compute_fh(cl_L_per_h, fu_plasma)

    # Bioavailability
    f_oral = fa * fg * fh
    f_oral_pct = f_oral * 100.0

    bio_class = _bioavailability_class(f_oral)
    limiting = _limiting_factor(fa, fg, fh)

    notes = (
        f"BCS {bcs_class}: Do={do_value:.3f}, Peff={peff_cm_per_s:.2e} cm/s; "
        f"fa={fa:.3f}, fg={fg:.3f}, fh={fh:.3f}; "
        f"pka={pka}, ionization={ionization_type}, mw={mw_Da}"
    )

    return OralBioavailabilityResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        solubility_mg_mL=solubility_mg_mL,
        logP=logP,
        fu_plasma=fu_plasma,
        bcs_class=bcs_class,
        do_value=do_value,
        peff_cm_per_s=peff_cm_per_s,
        fa=fa,
        fg=fg,
        fh=fh,
        f_oral_predicted=f_oral,
        f_oral_pct=f_oral_pct,
        bioavailability_class=bio_class,
        limiting_factor=limiting,
        notes=notes,
    )


def sensitivity_analysis(
    drug_name: str,
    dose_mg: float,
    solubility_mg_mL: float,
    logP: float,
    pka: float,
    ionization_type: str,
    fu_plasma: float,
    cl_L_per_h: float,
    mw_Da: float,
) -> dict:
    """Sensitivity analysis: ±50% changes in each input parameter.

    Returns a dict with baseline values and sensitivity results showing
    how ±50% perturbation in each parameter affects F.
    """
    base = predict_oral_bioavailability(
        drug_name,
        dose_mg,
        solubility_mg_mL,
        logP,
        pka,
        ionization_type,
        fu_plasma,
        cl_L_per_h,
        mw_Da,
    )

    result = {
        "fa": base.fa,
        "fg": base.fg,
        "fh": base.fh,
        "F": base.f_oral_predicted,
        "sensitivity": {},
    }

    # Parameters to perturb and their bounds
    perturbations = {
        "dose_mg": (dose_mg, 0.0, None),
        "solubility_mg_mL": (solubility_mg_mL, 1e-9, None),
        "logP": (logP, -5.0, 10.0),
        "fu_plasma": (fu_plasma, 1e-6, 1.0),
        "cl_L_per_h": (cl_L_per_h, 1e-9, None),
    }

    def _clamp(val, lo, hi):
        if lo is not None and val < lo:
            val = lo
        if hi is not None and val > hi:
            val = hi
        return val

    for param, (base_val, lo, hi) in perturbations.items():
        low_val = _clamp(base_val * 0.5, lo, hi)
        high_val = _clamp(base_val * 1.5, lo, hi)

        kwargs = {
            "drug_name": drug_name,
            "dose_mg": dose_mg,
            "solubility_mg_mL": solubility_mg_mL,
            "logP": logP,
            "pka": pka,
            "ionization_type": ionization_type,
            "fu_plasma": fu_plasma,
            "cl_L_per_h": cl_L_per_h,
            "mw_Da": mw_Da,
        }

        kwargs_low = dict(kwargs)
        kwargs_low[param] = low_val
        kwargs_high = dict(kwargs)
        kwargs_high[param] = high_val

        try:
            f_low = predict_oral_bioavailability(**kwargs_low).f_oral_predicted
        except ValueError:
            f_low = base.f_oral_predicted

        try:
            f_high = predict_oral_bioavailability(**kwargs_high).f_oral_predicted
        except ValueError:
            f_high = base.f_oral_predicted

        result["sensitivity"][param] = {
            "low_50pct": f_low,
            "high_50pct": f_high,
            "delta_low": f_low - base.f_oral_predicted,
            "delta_high": f_high - base.f_oral_predicted,
        }

    return result
