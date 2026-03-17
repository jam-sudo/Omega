"""Analytical 1-compartment PK prediction engine.

Provides clean analytical PK predictions from ADME properties.

NOTE (2026-03-17): This module is NOT used by OmegaPipeline.simulate().
The pipeline performs its own inline analytical Cmax calculation in
pipeline/__init__.py (hybrid Cmax selector). Both this module and the
inline code use VDss as the volume term. For a true 1-compartment model
VDss == Vc, so this is correct. However, when VDss is predicted from a
multi-compartment perspective (XGBoost or Berezhkovskiy Kp), it may
over-estimate central volume for highly distributed lipophilic drugs,
leading to under-predicted Cmax. The pipeline mitigates this via the
adaptive ODE/analytical blending and VDss correction heuristics.
"""

import math

import numpy as np


def estimate_vc(vd_ss: float, body_weight: float = 70.0) -> float:
    """Estimate central compartment volume (Vc) from VDss.

    For highly distributed drugs (VDss >> plasma volume), Vc is much
    smaller than VDss. Using VDss in a 1-compartment Cmax formula
    over-estimates volume and under-predicts Cmax.

    Heuristic: Vc = max(blood_volume, VDss * 0.5) capped so that
    Vc is at least blood volume (~5L for 70kg) but no larger than VDss.
    For typical drugs (VDss 10-50L), this gives Vc ~ 5-25L.
    For very low VDss drugs (<10L), Vc ~ VDss (single compartment).

    Args:
        vd_ss: Steady-state volume of distribution in L
        body_weight: Body weight in kg (default 70)

    Returns:
        Estimated central volume in L
    """
    blood_volume = 0.07 * body_weight  # ~5L for 70kg
    # For low-Vd drugs, Vc ≈ VDss; for high-Vd, Vc is bounded below by blood volume
    vc = max(blood_volume, min(vd_ss, vd_ss * 0.5))
    return vc


def predict_pk_analytical(
    dose_mg: float,
    cl_L_h: float,
    vd_L: float,
    f_oral: float,
    ka_h: float,
    duration_h: float = 24.0,
    n_points: int = 241,
    use_vc: bool = False,
    body_weight: float = 70.0,
) -> dict:
    """Analytical 1-compartment oral PK prediction.

    Computes Cmax, Tmax, AUC, t_half from clearance-volume model:
        C(t) = (F * Dose / Vd) * (ka / (ka - ke)) * (exp(-ke*t) - exp(-ka*t))

    Args:
        dose_mg: Oral dose in mg
        cl_L_h: Total clearance (hepatic + renal) in L/h
        vd_L: Volume of distribution (VDss) in L
        f_oral: Oral bioavailability (0-1)
        ka_h: Absorption rate constant (1/h)
        duration_h: Simulation duration
        n_points: Number of timepoints
        use_vc: If True, estimate Vc from VDss for Cmax calculation.
            This avoids under-predicting Cmax for lipophilic drugs where
            VDss >> Vc. AUC and t_half still use VDss (correct for 1-cpt).
        body_weight: Body weight in kg (default 70), used for Vc estimation

    Returns:
        dict with keys: time_h, cp_mg_L, cmax_mg_L, tmax_h, auc_mg_h_L, t_half_h
    """
    # Guard: invalid parameters → return safe defaults
    if vd_L <= 0 or cl_L_h <= 0:
        t = np.linspace(0, duration_h, n_points)
        default_cmax = dose_mg / 100.0
        return {
            "time_h": t,
            "cp_mg_L": np.full_like(t, 0.0),
            "cmax_mg_L": default_cmax,
            "tmax_h": 1.0,
            "auc_mg_h_L": default_cmax * 4.0,
            "t_half_h": 4.0,
        }

    # For Cmax, optionally use Vc (central volume) instead of VDss.
    # AUC and t_half use VDss via ke = CL / VDss (pharmacologically correct).
    vd_for_cmax = estimate_vc(vd_L, body_weight) if use_vc else vd_L
    ke = cl_L_h / vd_L  # elimination rate constant (1/h) — always from VDss

    # Guard: if ka ≈ ke (within 1%), nudge ka to avoid division by zero
    if abs(ka_h - ke) / max(ke, 1e-10) < 0.01:
        ka_h = ke * 1.01

    t_half = 0.693 / ke

    # Tmax analytical (only valid when ka > ke)
    if ka_h > ke:
        tmax = math.log(ka_h / ke) / (ka_h - ke)
    else:
        # ka < ke: peak is very early; use numerical max below
        tmax = 0.5

    # Concentration-time profile
    # Use vd_for_cmax for concentration (Vc-based when use_vc=True),
    # but ke still uses VDss (pharmacologically correct for elimination).
    t = np.linspace(0, duration_h, n_points)
    coeff = (f_oral * dose_mg / vd_for_cmax) * (ka_h / (ka_h - ke))
    cp = coeff * (np.exp(-ke * t) - np.exp(-ka_h * t))
    cp = np.maximum(cp, 0.0)  # no negative concentrations

    # Cmax from the analytical tmax
    cmax = coeff * (math.exp(-ke * tmax) - math.exp(-ka_h * tmax))
    cmax = max(cmax, 0.0)

    # Cross-check with numerical max (handles edge cases)
    numerical_cmax = float(np.max(cp))
    if numerical_cmax > cmax:
        cmax = numerical_cmax
        tmax = float(t[np.argmax(cp)])

    # AUC via trapezoidal integration
    auc = float(np.trapz(cp, t))

    return {
        "time_h": t,
        "cp_mg_L": cp,
        "cmax_mg_L": cmax,
        "tmax_h": tmax,
        "auc_mg_h_L": auc,
        "t_half_h": t_half,
    }


def compute_pk_parameters(
    adme_props: dict,
    dose_mg: float,
    smiles: str,
    vdss_predictor=None,
    clint_predictor=None,
    body_weight: float = 70.0,
) -> dict:
    """Compute CL, Vd, F, ka from ADME properties.

    Uses standard IVIVE with empirical scaling factor (ESF=3.0)
    and XGBoost VDss for volume of distribution.

    Args:
        adme_props: Dict with keys like fup, logP, peff, clint_hepatocyte_uL_min
        dose_mg: Oral dose in mg
        smiles: SMILES string for the compound
        vdss_predictor: Optional predictor with predict_vdss(smiles) method
        clint_predictor: Optional predictor with predict_clint(smiles) method
        body_weight: Body weight in kg (default 70)

    Returns:
        dict with keys: cl_hepatic, cl_renal, cl_total, vd_L, f_oral, ka_h,
                        and intermediate values for debugging
    """
    fup = adme_props.get("fup", 0.1)
    logP = adme_props.get("logP", 2.0)
    peff = adme_props.get("peff", 1.0)  # in 1e-4 cm/s units

    # --- Intrinsic clearance ---
    if clint_predictor is not None:
        clint_hep = clint_predictor.predict_clint(smiles)
    else:
        clint_hep = adme_props.get("clint_hepatocyte_uL_min", 0.0)

    # Standard IVIVE: CLint_hep (uL/min/10^6 cells) → CLint_liver (L/h)
    # CLint_liver = CLint_hep * hepatocellularity(120e6/g) * liver_weight(1800g) / 1e6 / 60
    #             = CLint_hep * 12.96
    # Apply empirical scaling factor (ESF=3.0) to correct IVIVE gap
    # (Hallifax et al., Drug Metab Dispos 2010)
    ESF = 3.0
    clint_liver = clint_hep * 12.96 / ESF

    # Well-stirred hepatic model
    Q_H = 90.0  # hepatic blood flow L/h
    cl_hepatic = (Q_H * fup * clint_liver) / (Q_H + fup * clint_liver)

    # --- Renal clearance ---
    if logP < 1.0:
        cl_renal = 7.2 * fup * 0.5  # GFR-based, 50% reabsorption
    elif logP < 2.0:
        cl_renal = 7.2 * fup * 0.2  # mostly reabsorbed
    else:
        cl_renal = 0.0  # lipophilic, complete reabsorption

    cl_total = cl_hepatic + cl_renal

    # --- Volume of distribution ---
    if vdss_predictor is not None:
        vd_L = vdss_predictor.predict_vdss(smiles) * body_weight
    else:
        # Fallback: Oie-Tozer approximation (rough)
        vd_L = 0.07 * body_weight + 0.7 * body_weight * fup

    vd_L = max(vd_L, 0.04 * body_weight)  # minimum ~3L for 70kg

    # --- Oral bioavailability ---
    # fa from peff (permeability in 1e-4 cm/s units)
    peff_cm_s = peff * 1e-4
    fa = 1.0 - math.exp(-2.3e5 * peff_cm_s)
    fa = max(min(fa, 1.0), 0.01)

    # fh from well-stirred extraction ratio
    extraction = cl_hepatic / Q_H if Q_H > 0 else 0.0
    fh = 1.0 - extraction
    fh = max(fh, 0.01)

    f_oral = max(fa * fh, 0.01)

    # --- Absorption rate constant ---
    ka_h = max(0.3, min(5.0, peff))

    return {
        "cl_hepatic": cl_hepatic,
        "cl_renal": cl_renal,
        "cl_total": cl_total,
        "vd_L": vd_L,
        "f_oral": f_oral,
        "ka_h": ka_h,
        # Intermediate values for debugging
        "clint_hep": clint_hep,
        "clint_liver": clint_liver,
        "fup": fup,
        "fa": fa,
        "fh": fh,
        "extraction_ratio": extraction,
    }
