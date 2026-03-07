"""Drug metabolism rate prediction from physicochemical properties.

Predicts intrinsic hepatic clearance (CLint) and dominant CYP/UGT/MAO
metabolic pathway using empirical QSAR relationships.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Microsomal protein scaling: 45 mg/g liver × 1500 g liver = 67,500 mg protein
# But standard approach: CLint_total per whole liver = clint_µL/min/mg × 45 mg/g × 1500 g
_MICROSOMAL_PROTEIN_PER_G_LIVER = 45.0  # mg/g
_LIVER_WEIGHT_G = 1500.0  # g
_HEPATIC_BLOOD_FLOW_L_PER_H = 90.0  # L/h
_DEFAULT_VD_L = 70.0  # L (used for t½ estimate)
_MINUTES_PER_HOUR = 60.0
_UL_PER_ML = 1.0  # µL vs µL consistent units
_L_PER_UL = 1e-6  # 1 L = 1e6 µL


# ---------------------------------------------------------------------------
# Primary function: CLint prediction
# ---------------------------------------------------------------------------


def predict_clint_hepatic(
    logP: float,
    mw: float,
    n_nitrogen: int,
    n_oxygen: int,
    has_primary_amine: bool = False,
    has_phenol: bool = False,
) -> dict:
    """Predict intrinsic hepatic clearance from physicochemical descriptors.

    Parameters
    ----------
    logP:
        Octanol-water partition coefficient.
    mw:
        Molecular weight (Da).
    n_nitrogen:
        Number of nitrogen atoms (metabolic soft spots).
    n_oxygen:
        Number of oxygen atoms.
    has_primary_amine:
        True if molecule contains a primary amine.
    has_phenol:
        True if molecule contains a phenol group.

    Returns
    -------
    dict with: clint_uL_min_mg, clint_L_per_h, clh_L_per_h,
    extraction_ratio, t_half_min, metabolic_rate
    """
    if mw <= 0:
        raise ValueError("mw must be positive")
    if n_nitrogen < 0:
        raise ValueError("n_nitrogen must be >= 0")
    if n_oxygen < 0:
        raise ValueError("n_oxygen must be >= 0")

    # --- Empirical CLint model (µL/min/mg microsomal protein) ---
    base = 10.0

    # Lipophilicity: more lipophilic → faster CYP metabolism
    if logP > 3.0:
        base += (logP - 3.0) * 5.0

    # Molecular weight penalty: large molecules slower
    if mw > 500.0:
        base -= (mw - 500.0) * 0.02

    # Nitrogen soft spots (N-demethylation, N-oxidation)
    base += n_nitrogen * 3.0

    # Oxygen soft spots (O-demethylation, epoxidation)
    base += n_oxygen * 1.0

    # Functional group boosts
    if has_primary_amine:
        base += 15.0  # N-hydroxylation, deamination via MAO
    if has_phenol:
        base += 10.0  # glucuronidation / sulfation

    # Clamp to physiologically plausible range
    clint_uL_min_mg = max(1.0, min(300.0, base))

    # --- Scale to whole-liver CLint ---
    # Total microsomal protein: 45 mg/g × 1500 g = 67,500 mg
    total_mg_protein = _MICROSOMAL_PROTEIN_PER_G_LIVER * _LIVER_WEIGHT_G  # 67,500

    # CLint_total in µL/min
    clint_total_uL_min = clint_uL_min_mg * total_mg_protein

    # Convert to L/h
    clint_L_per_h = clint_total_uL_min * _L_PER_UL * _MINUTES_PER_HOUR

    # --- Well-stirred model: CLh = Qh * CLint / (Qh + CLint) ---
    Qh = _HEPATIC_BLOOD_FLOW_L_PER_H
    clh_L_per_h = (Qh * clint_L_per_h) / (Qh + clint_L_per_h)

    # Extraction ratio
    extraction_ratio = clh_L_per_h / Qh

    # Half-life estimate using CLh and default Vd
    # t½ = (Vd × ln2) / CLh  (in hours → convert to minutes)
    if clh_L_per_h > 0:
        t_half_h = (_DEFAULT_VD_L * 0.693147) / clh_L_per_h
        t_half_min = t_half_h * _MINUTES_PER_HOUR
    else:
        t_half_min = float("inf")

    # Metabolic rate category
    if clint_uL_min_mg < 1.0:
        metabolic_rate = "low"
    elif clint_uL_min_mg <= 30.0:
        metabolic_rate = "moderate"
    else:
        metabolic_rate = "high"

    return {
        "clint_uL_min_mg": clint_uL_min_mg,
        "clint_L_per_h": clint_L_per_h,
        "clh_L_per_h": clh_L_per_h,
        "extraction_ratio": extraction_ratio,
        "t_half_min": t_half_min,
        "metabolic_rate": metabolic_rate,
    }


# ---------------------------------------------------------------------------
# Pathway prediction
# ---------------------------------------------------------------------------


def predict_metabolic_pathway(
    logP: float,
    mw: float,
    n_nitrogen: int,
    n_oxygen: int,
    has_primary_amine: bool,
    has_phenol: bool,
    n_aromatic_rings: int,
) -> dict:
    """Predict dominant metabolic pathway.

    Uses a rule-based decision tree reflecting known CYP substrate preferences.

    Returns
    -------
    dict with: primary_pathway, secondary_pathway, confidence
    """
    # Priority order: structural features trump lipophilicity defaults
    if has_primary_amine:
        primary = "MAO"
        secondary = "CYP2D6" if n_nitrogen > 1 else "CYP3A4"
        confidence = "high"
    elif has_phenol:
        primary = "UGT"
        secondary = "CYP2C9"
        confidence = "high"
    elif n_nitrogen > 2:
        primary = "CYP2D6"
        secondary = "CYP3A4"
        confidence = "moderate"
    elif logP > 3.0 and mw < 500.0:
        primary = "CYP3A4"
        secondary = "CYP2C9"
        confidence = "moderate"
    else:
        primary = "CYP2C9"
        secondary = "CYP3A4"
        confidence = "low"

    return {
        "primary_pathway": primary,
        "secondary_pathway": secondary,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


def clint_sensitivity_analysis(
    logP_range: list[float],
    mw: float,
    n_nitrogen: int,
    n_oxygen: int,
) -> list[dict]:
    """Vary logP and return CLint metrics for each value.

    Parameters
    ----------
    logP_range:
        List of logP values to evaluate.
    mw:
        Molecular weight (Da).
    n_nitrogen:
        Number of nitrogen atoms.
    n_oxygen:
        Number of oxygen atoms.

    Returns
    -------
    List of dicts: {logP, clint_uL_min_mg, clh_L_per_h, extraction_ratio}
    """
    results = []
    for lp in logP_range:
        pred = predict_clint_hepatic(
            logP=lp,
            mw=mw,
            n_nitrogen=n_nitrogen,
            n_oxygen=n_oxygen,
        )
        results.append(
            {
                "logP": lp,
                "clint_uL_min_mg": pred["clint_uL_min_mg"],
                "clh_L_per_h": pred["clh_L_per_h"],
                "extraction_ratio": pred["extraction_ratio"],
            }
        )
    return results
