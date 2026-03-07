"""Bioavailability prediction from in vitro ADME data using IVIVE scaling.

Predicts oral bioavailability F = fa * fg * fh from in vitro measurements:
  fa - fraction absorbed (Peff/BCS approach)
  fg - gut-wall availability (gut CYP3A4 CLint)
  fh - hepatic first-pass availability (well-stirred model)

Reference: Sun et al. (2004), Rowland & Tozer Clinical PK 5th Ed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BioavailabilityPredictionResult:
    """Result of bioavailability prediction from in vitro data."""

    compound_name: str

    # Input in vitro data
    fa_predicted: float  # Fraction absorbed (from Peff/BCS)
    fg_predicted: float  # Gut availability (from CLint_gut)
    fh_predicted: float  # Hepatic availability (from CLint_hep)

    # Components
    f_oral_predicted: float  # = fa * fg * fh

    # Confidence
    fa_basis: str  # "permeability", "solubility_limited", "unknown"
    fg_basis: str  # "cyp3a4_clint", "assumed_high"
    fh_basis: str  # "well_stirred", "parallel_tube", "assumed"

    # Comparison
    f_oral_literature: float | None  # if provided
    prediction_accuracy_fold: float | None  # |pred/lit|, None if no literature

    # Uncertainty
    f_oral_low: float  # Conservative estimate (5th pct)
    f_oral_high: float  # Optimistic estimate (95th pct)

    # Additional metrics
    cl_predicted_L_per_h: float  # Predicted systemic CL
    t_half_predicted_h: float  # Predicted half-life

    recommendation: str
    notes: str


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def _predict_fa(
    peff_cm_per_s: float,
    solubility_mg_mL: float,
    dose_mg: float,
    dose_volume_mL: float,
) -> tuple[float, str]:
    """Predict fraction absorbed and basis string."""
    dose_number = (
        dose_mg / (solubility_mg_mL * dose_volume_mL) if solubility_mg_mL > 0 else float("inf")
    )

    if peff_cm_per_s >= 2e-4:
        if dose_number < 1.0:
            fa = 0.95
            basis = "permeability"
        else:
            fa = _clamp(0.95 / dose_number, 0.0, 0.95)
            basis = "solubility_limited"
    elif peff_cm_per_s < 1e-4:
        fa = _clamp(0.3 * peff_cm_per_s / 1e-4, 0.0, 0.95)
        basis = "permeability"
    else:
        # Linear interpolation between 1e-4 and 2e-4 cm/s
        frac = (peff_cm_per_s - 1e-4) / (2e-4 - 1e-4)
        fa_low = _clamp(0.3 * peff_cm_per_s / 1e-4, 0.0, 0.95)
        fa_high = 0.95 if dose_number < 1.0 else _clamp(0.95 / dose_number, 0.0, 0.95)
        fa = fa_low + frac * (fa_high - fa_low)
        basis = "permeability" if dose_number < 1.0 else "solubility_limited"

    return _clamp(fa, 0.0, 1.0), basis


def _predict_fg(
    clint_gut_uL_per_min_per_mg: float | None,
    fu_plasma: float,
    hepatic_blood_flow_L_per_h: float,
) -> tuple[float, str]:
    """Predict gut-wall availability and basis string."""
    if clint_gut_uL_per_min_per_mg is not None:
        # Scale: uL/min/mg protein * 60 min/h / 1000 uL/mL * 20 mg_protein/g * 1000 mL gut
        # = uL/min/mg * 60/1000 * 20  [net factor: *1.2 L/h per unit]
        clint_gut_L_per_h = clint_gut_uL_per_min_per_mg * 60.0 / 1000.0 * 20.0
        q_gut = hepatic_blood_flow_L_per_h * 0.3  # portal blood flow
        denom = q_gut + fu_plasma * clint_gut_L_per_h
        fg = q_gut / denom if denom > 0 else 1.0
        fg = _clamp(fg, 0.0, 1.0)
        basis = "cyp3a4_clint"
    else:
        fg = 0.9
        basis = "assumed_high"

    return fg, basis


def _predict_fh(
    clint_hep_uL_per_min_per_mg: float | None,
    fu_plasma: float,
    hepatic_blood_flow_L_per_h: float,
) -> tuple[float, float, str]:
    """Predict hepatic availability and basis string. Returns (fh, cl_hep, basis)."""
    if clint_hep_uL_per_min_per_mg is not None:
        # Brown 2007 scaling: uL/min/mg * 60/1000 * 45 mg/g * 1.5 (liver factor)
        clint_hep_L_per_h = clint_hep_uL_per_min_per_mg * 60.0 / 1000.0 * 45.0 * 1.5
        qh = hepatic_blood_flow_L_per_h
        cl_hep = qh * fu_plasma * clint_hep_L_per_h / (qh + fu_plasma * clint_hep_L_per_h)
        fh = 1.0 - cl_hep / qh
        fh = _clamp(fh, 0.0, 1.0)
        cl_hep = _clamp(cl_hep, 0.0, qh)
        basis = "well_stirred"
    else:
        fh = 0.85
        cl_hep = (1.0 - fh) * hepatic_blood_flow_L_per_h
        basis = "assumed"

    return fh, cl_hep, basis


def _build_recommendation(f_oral: float) -> str:
    """Return recommendation based on predicted F."""
    if f_oral >= 0.7:
        return "Excellent oral bioavailability predicted. Suitable candidate for oral dosing."
    elif f_oral >= 0.4:
        return "Moderate oral bioavailability predicted. May require formulation optimization."
    elif f_oral >= 0.2:
        return "Low oral bioavailability predicted. Consider prodrug strategy or parenteral route."
    else:
        return "Very low oral bioavailability predicted. Oral route likely not feasible."


def predict_bioavailability(
    compound_name: str,
    peff_cm_per_s: float,
    solubility_mg_mL: float,
    dose_mg: float,
    mw: float,
    logP: float,
    fu_plasma: float,
    clint_hep_uL_per_min_per_mg: float | None = None,
    clint_gut_uL_per_min_per_mg: float | None = None,
    vd_L: float | None = None,
    hepatic_blood_flow_L_per_h: float = 90.0,
    f_oral_literature: float | None = None,
    dose_volume_mL: float = 250.0,
) -> BioavailabilityPredictionResult:
    """Predict oral bioavailability from in vitro ADME data.

    Parameters
    ----------
    compound_name : str
        Name of the compound.
    peff_cm_per_s : float
        Effective intestinal permeability (cm/s).
    solubility_mg_mL : float
        Aqueous solubility (mg/mL).
    dose_mg : float
        Oral dose (mg).
    mw : float
        Molecular weight (g/mol).
    logP : float
        Lipophilicity.
    fu_plasma : float
        Fraction unbound in plasma (0-1).
    clint_hep_uL_per_min_per_mg : float, optional
        Hepatic microsomal intrinsic clearance (uL/min/mg protein).
    clint_gut_uL_per_min_per_mg : float, optional
        Gut microsomal intrinsic clearance (uL/min/mg protein).
    vd_L : float, optional
        Volume of distribution (L). Estimated from logP if not provided.
    hepatic_blood_flow_L_per_h : float
        Hepatic blood flow (L/h). Default 90 L/h.
    f_oral_literature : float, optional
        Observed bioavailability for comparison.
    dose_volume_mL : float
        Volume of GI fluid for dose number calculation (mL).

    Returns
    -------
    BioavailabilityPredictionResult
    """
    # Validate inputs
    if fu_plasma <= 0.0 or fu_plasma > 1.0:
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if solubility_mg_mL <= 0.0:
        raise ValueError(f"solubility_mg_mL must be positive, got {solubility_mg_mL}")
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if mw <= 0.0:
        raise ValueError(f"mw must be positive, got {mw}")

    fa, fa_basis = _predict_fa(peff_cm_per_s, solubility_mg_mL, dose_mg, dose_volume_mL)
    fg, fg_basis = _predict_fg(clint_gut_uL_per_min_per_mg, fu_plasma, hepatic_blood_flow_L_per_h)
    fh, cl_hep, fh_basis = _predict_fh(
        clint_hep_uL_per_min_per_mg, fu_plasma, hepatic_blood_flow_L_per_h
    )

    f_oral = fa * fg * fh
    f_oral = _clamp(f_oral, 0.0, 1.0)

    # Volume of distribution estimate
    if vd_L is not None:
        vd_used = vd_L
    else:
        vd_used = _clamp((0.5 + 0.4 * logP) * 70.0, 1.0, 500.0)

    # Half-life
    cl_predicted = cl_hep  # primary clearance organ
    if cl_predicted > 0:
        t_half = 0.693 * vd_used / cl_predicted
    else:
        t_half = float("inf")

    # Uncertainty bounds (±30% on each component)
    fa_low = fa * 0.7
    fg_low = fg * 0.7
    fh_low = fh * 0.7
    fa_high = fa * 1.3
    fg_high = fg * 1.3
    fh_high = fh * 1.3

    f_oral_low = _clamp(fa_low * fg_low * fh_low, 0.0, 1.0)
    f_oral_high = _clamp(fa_high * fg_high * fh_high, 0.0, 1.0)

    # Accuracy fold vs literature
    if f_oral_literature is not None and f_oral_literature > 0 and f_oral > 0:
        prediction_accuracy_fold = f_oral / f_oral_literature
    else:
        prediction_accuracy_fold = None

    recommendation = _build_recommendation(f_oral)

    dose_number = dose_mg / (solubility_mg_mL * dose_volume_mL)
    notes = (
        f"Dose number (Dn) = {dose_number:.2f}. "
        f"fa basis: {fa_basis}, fg basis: {fg_basis}, fh basis: {fh_basis}. "
        f"Estimated Vd = {vd_used:.1f} L."
    )

    return BioavailabilityPredictionResult(
        compound_name=compound_name,
        fa_predicted=fa,
        fg_predicted=fg,
        fh_predicted=fh,
        f_oral_predicted=f_oral,
        fa_basis=fa_basis,
        fg_basis=fg_basis,
        fh_basis=fh_basis,
        f_oral_literature=f_oral_literature,
        prediction_accuracy_fold=prediction_accuracy_fold,
        f_oral_low=f_oral_low,
        f_oral_high=f_oral_high,
        cl_predicted_L_per_h=cl_predicted,
        t_half_predicted_h=t_half,
        recommendation=recommendation,
        notes=notes,
    )


def screen_bioavailability(
    compounds_list: list[dict],
) -> list[BioavailabilityPredictionResult]:
    """Screen a list of compounds for oral bioavailability.

    Parameters
    ----------
    compounds_list : list of dict
        Each dict with keys: name, peff, solubility, dose_mg, mw, logP, fu_plasma.
        Optional keys: clint_hep, clint_gut, vd_L, f_oral_literature.

    Returns
    -------
    list of BioavailabilityPredictionResult sorted by f_oral_predicted descending.
    """
    results = []
    for c in compounds_list:
        result = predict_bioavailability(
            compound_name=c["name"],
            peff_cm_per_s=c["peff"],
            solubility_mg_mL=c["solubility"],
            dose_mg=c["dose_mg"],
            mw=c["mw"],
            logP=c["logP"],
            fu_plasma=c["fu_plasma"],
            clint_hep_uL_per_min_per_mg=c.get("clint_hep"),
            clint_gut_uL_per_min_per_mg=c.get("clint_gut"),
            vd_L=c.get("vd_L"),
            f_oral_literature=c.get("f_oral_literature"),
        )
        results.append(result)

    results.sort(key=lambda r: r.f_oral_predicted, reverse=True)
    return results
