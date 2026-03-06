"""IV-to-oral bridging: convert IV PK data to predict oral PK parameters."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["IVOralBridgeResult", "bridge_iv_to_oral", "predict_oral_bioavailability"]

_VALID_BCS = {"I", "II", "III", "IV"}


@dataclass(frozen=True)
class IVOralBridgeResult:
    drug_name: str
    # IV-derived PK
    cl_iv_L_per_h: float
    vd_iv_L: float
    t_half_iv_h: float
    auc_iv_mg_L_h: float
    # Oral predicted PK
    f_oral_predicted: float
    ka_predicted_per_h: float
    tmax_predicted_h: float
    cmax_predicted_mg_L: float
    auc_oral_predicted_mg_L_h: float
    dose_mg: float
    # Comparison (if oral AUC provided)
    f_observed: float
    prediction_error_pct: float
    fa_predicted: float
    fg_predicted: float
    fh_predicted: float
    notes: str


def _compute_fa(bcs_class: str, psa: float, logP: float) -> float:
    if bcs_class == "I":
        return 0.95
    elif bcs_class == "II":
        return max(0.3, 0.95 - max(0.0, psa - 40.0) * 0.01)
    elif bcs_class == "III":
        return 0.85
    elif bcs_class == "IV":
        return max(0.1, 0.5 - logP * 0.05)
    else:
        raise ValueError(f"Invalid BCS class '{bcs_class}'. Must be one of I, II, III, IV.")


def bridge_iv_to_oral(
    drug_name: str,
    iv_dose_mg: float,
    iv_auc_mg_L_h: float,
    cl_iv_L_per_h: float,
    vd_iv_L: float,
    oral_dose_mg: float,
    bcs_class: str = "I",
    psa: float = 60.0,
    logP: float = 2.0,
    oral_auc_observed: float = None,
    qh_L_per_h: float = 80.0,
) -> IVOralBridgeResult:
    """Bridge IV PK data to predict oral PK parameters.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    iv_dose_mg:
        IV dose in mg (must be > 0).
    iv_auc_mg_L_h:
        AUC from IV dose in mg/L*h (must be > 0).
    cl_iv_L_per_h:
        IV clearance in L/h (must be > 0).
    vd_iv_L:
        Volume of distribution from IV in L (must be > 0).
    oral_dose_mg:
        Oral dose in mg (must be > 0).
    bcs_class:
        BCS class (I, II, III, IV).
    psa:
        Polar surface area in A^2.
    logP:
        Octanol-water partition coefficient.
    oral_auc_observed:
        Optional observed oral AUC for validation (mg/L*h).
    qh_L_per_h:
        Hepatic blood flow in L/h (default 80).

    Returns
    -------
    IVOralBridgeResult
    """
    # Validate inputs
    if iv_dose_mg <= 0:
        raise ValueError("iv_dose_mg must be > 0")
    if iv_auc_mg_L_h <= 0:
        raise ValueError("iv_auc_mg_L_h must be > 0")
    if cl_iv_L_per_h <= 0:
        raise ValueError("cl_iv_L_per_h must be > 0")
    if vd_iv_L <= 0:
        raise ValueError("vd_iv_L must be > 0")
    if oral_dose_mg <= 0:
        raise ValueError("oral_dose_mg must be > 0")
    if bcs_class not in _VALID_BCS:
        raise ValueError(f"Invalid BCS class '{bcs_class}'. Must be one of I, II, III, IV.")

    # IV-derived PK
    t_half_iv = 0.693 * vd_iv_L / cl_iv_L_per_h

    # Hepatic extraction and availability
    eh = min(0.99, cl_iv_L_per_h / qh_L_per_h)
    fh = 1.0 - eh

    # Fraction absorbed from BCS class + physicochemistry
    fa = _compute_fa(bcs_class, psa, logP)

    # Gut wall availability
    fg = max(0.3, 1.0 - max(0.0, logP - 3.0) * 0.1)

    # Oral bioavailability
    f_oral = fa * fg * fh

    # Ka prediction from logP (Abraham 2001 simplified)
    ka = max(0.1, 0.5 + logP * 0.2)

    # Predicted oral AUC
    auc_oral_pred = oral_dose_mg * f_oral / cl_iv_L_per_h

    # Tmax: ln(ka/ke) / (ka - ke)
    ke = cl_iv_L_per_h / vd_iv_L
    if abs(ka - ke) < 1e-6:
        ka += 1e-6
    tmax = math.log(ka / ke) / (ka - ke)
    tmax = max(0.1, tmax)

    # Cmax: analytical oral 1-cpt
    cmax = (oral_dose_mg * f_oral * ka) / (vd_iv_L * (ka - ke)) * (
        math.exp(-ke * tmax) - math.exp(-ka * tmax)
    )
    cmax = abs(cmax)

    # Prediction error vs observed (if provided)
    if oral_auc_observed is not None:
        f_obs = oral_auc_observed * cl_iv_L_per_h / oral_dose_mg
        pred_err = (f_oral - f_obs) / f_obs * 100.0
    else:
        f_obs = float("nan")
        pred_err = float("nan")

    notes = (
        f"EH={eh:.3f}, FH={fh:.3f}, fa={fa:.3f}, fg={fg:.3f}; "
        f"ka={ka:.3f} h^-1, ke={ke:.3f} h^-1"
    )

    return IVOralBridgeResult(
        drug_name=drug_name,
        cl_iv_L_per_h=cl_iv_L_per_h,
        vd_iv_L=vd_iv_L,
        t_half_iv_h=t_half_iv,
        auc_iv_mg_L_h=iv_auc_mg_L_h,
        f_oral_predicted=f_oral,
        ka_predicted_per_h=ka,
        tmax_predicted_h=tmax,
        cmax_predicted_mg_L=cmax,
        auc_oral_predicted_mg_L_h=auc_oral_pred,
        dose_mg=oral_dose_mg,
        f_observed=f_obs,
        prediction_error_pct=pred_err,
        fa_predicted=fa,
        fg_predicted=fg,
        fh_predicted=fh,
        notes=notes,
    )


def predict_oral_bioavailability(
    drug_name: str,
    cl_iv_L_per_h: float,
    vd_iv_L: float,
    bcs_class: str = "I",
    logP: float = 2.0,
    psa: float = 60.0,
    qh_L_per_h: float = 80.0,
) -> dict:
    """Predict oral bioavailability components from IV PK parameters.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    cl_iv_L_per_h:
        IV clearance in L/h.
    vd_iv_L:
        Volume of distribution from IV in L.
    bcs_class:
        BCS class (I, II, III, IV).
    logP:
        Octanol-water partition coefficient.
    psa:
        Polar surface area in A^2.
    qh_L_per_h:
        Hepatic blood flow in L/h (default 80).

    Returns
    -------
    dict with fa, fg, fh, f_oral, eh, ka, ke, t_half_iv_h, drug_name
    """
    if bcs_class not in _VALID_BCS:
        raise ValueError(f"Invalid BCS class '{bcs_class}'. Must be one of I, II, III, IV.")

    eh = min(0.99, cl_iv_L_per_h / qh_L_per_h)
    fh = 1.0 - eh
    fa = _compute_fa(bcs_class, psa, logP)
    fg = max(0.3, 1.0 - max(0.0, logP - 3.0) * 0.1)
    f_oral = fa * fg * fh
    ka = max(0.1, 0.5 + logP * 0.2)
    ke = cl_iv_L_per_h / vd_iv_L
    t_half_iv = 0.693 * vd_iv_L / cl_iv_L_per_h

    return {
        "drug_name": drug_name,
        "fa": fa,
        "fg": fg,
        "fh": fh,
        "f_oral": f_oral,
        "eh": eh,
        "ka_per_h": ka,
        "ke_per_h": ke,
        "t_half_iv_h": t_half_iv,
        "bcs_class": bcs_class,
    }
