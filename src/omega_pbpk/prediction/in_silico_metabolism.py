"""
In silico prediction of metabolic pathways.

Phase 584: Rule-based (no rdkit) prediction of CYP enzymes, phase 1 products,
intrinsic clearance, and CYP substrate probabilities.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    """Logistic sigmoid: 1 / (1 + exp(-x))."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_metabolic_sites(
    smarts_alerts: list[str],
    logP: float,
    mw: float,
) -> dict:
    """
    Predict metabolic sites using physicochemical properties (no SMARTS parsing).

    Rules
    -----
    - High logP (>3) and MW 300–500  → likely CYP3A4 substrate (phase 1)
    - Low logP (<1) and MW < 300     → likely phase 2 (glucuronidation)
    - Otherwise                      → mixed

    Returns
    -------
    dict with keys:
        likely_enzymes  : list[str]
        metabolism_class: 'phase1' | 'phase2' | 'mixed'
        probability_phase1: float  in [0, 1]
    """
    likely_enzymes: list[str] = []
    metabolism_class: str
    probability_phase1: float

    high_logp = logP > 3.0
    mid_mw = 300.0 <= mw <= 500.0
    low_logp = logP < 1.0
    small_mw = mw < 300.0

    if high_logp and mid_mw:
        # Predominantly CYP3A4-mediated phase 1
        likely_enzymes = ["CYP3A4"]
        if logP > 4.0:
            likely_enzymes.append("CYP2C9")
        metabolism_class = "phase1"
        probability_phase1 = _clamp(0.5 + 0.1 * (logP - 3.0), 0.5, 0.95)

    elif low_logp and small_mw:
        # Predominantly phase 2
        likely_enzymes = ["UGT", "SULT"]
        metabolism_class = "phase2"
        probability_phase1 = _clamp(0.1 + 0.05 * logP, 0.0, 0.4)

    else:
        # Mixed metabolism
        likely_enzymes = ["CYP3A4", "CYP2D6", "UGT"]
        metabolism_class = "mixed"
        # Interpolate based on logP
        probability_phase1 = _clamp(0.3 + 0.05 * logP, 0.2, 0.8)

    return {
        "likely_enzymes": likely_enzymes,
        "metabolism_class": metabolism_class,
        "probability_phase1": probability_phase1,
    }


def predict_phase1_products(
    drug_name: str,
    logP: float,
    mw: float,
    molecular_formula_estimate: str = "CxHyOz",
) -> list[dict]:
    """
    Predict likely phase 1 metabolic products using simplified rules.

    Reactions modelled
    ------------------
    1. Hydroxylation      : product_mw = mw + 16
    2. N-dealkylation     : product_mw = mw - 14
    3. O-demethylation    : product_mw = mw - 14
    4. Oxidation          : product_mw = mw + 16

    Returns
    -------
    list of dict with keys: reaction, product_mw_estimate, likely_enzyme, notes
    """
    reactions = [
        {
            "reaction": "hydroxylation",
            "delta_mw": 16.0,
            "likely_enzyme": "CYP3A4" if logP > 2.0 else "CYP2C9",
            "notes": "Aromatic or aliphatic hydroxylation; adds one oxygen atom",
        },
        {
            "reaction": "N-dealkylation",
            "delta_mw": -14.0,
            "likely_enzyme": "CYP3A4",
            "notes": "Removal of N-alkyl group (loss of CH2)",
        },
        {
            "reaction": "O-demethylation",
            "delta_mw": -14.0,
            "likely_enzyme": "CYP2D6",
            "notes": "Removal of O-methyl group (loss of CH2)",
        },
        {
            "reaction": "oxidation",
            "delta_mw": 16.0,
            "likely_enzyme": "CYP1A2" if logP < 2.0 else "CYP3A4",
            "notes": "General oxidation; adds one oxygen atom",
        },
    ]

    products = []
    for rxn in reactions:
        products.append(
            {
                "reaction": rxn["reaction"],
                "product_mw_estimate": mw + rxn["delta_mw"],
                "likely_enzyme": rxn["likely_enzyme"],
                "notes": rxn["notes"],
            }
        )

    return products


def clint_from_structure(
    logP: float,
    mw: float,
    psa: float,
    n_hba: int,
) -> dict:
    """
    Empirical prediction of microsomal intrinsic clearance (CLint).

    Formula
    -------
    CLint = 10^(0.3*logP - 0.01*psa + 0.002*mw - 0.1*n_hba)
    Clamped to [0.1, 1000] μL/min/mg protein.

    IVIVE estimate uses simplified well-stirred model:
    CLint_u = CLint / fu_mic  (fu_mic = 0.1 default)
    Hepatic CL ≈ Qh * CLint_u / (Qh + CLint_u)  with Qh = 90 L/h (human liver blood flow)

    Returns
    -------
    dict: clint_uL_min_mg, clint_unbound, hepatic_cl_estimate_L_per_h,
          prediction_confidence
    """
    exponent = 0.3 * logP - 0.01 * psa + 0.002 * mw - 0.1 * n_hba
    clint_raw = 10.0**exponent
    clint = _clamp(clint_raw, 0.1, 1000.0)

    fu_mic = 0.1
    clint_unbound = clint / fu_mic  # μL/min/mg

    # Convert μL/min/mg to L/h: multiply by 60 min/h and microsomal protein scaling
    # Human liver microsomal protein ~ 45 mg/g liver; liver weight ~ 1500 g
    # Scale: 45 mg/g * 1500 g = 67500 mg protein; * 60 / 1e6 = L/h factor
    scaling_factor = 45.0 * 1500.0 * 60.0 / 1e6  # ≈ 4.05
    clint_scaled_L_h = clint_unbound * scaling_factor  # L/h

    # Well-stirred hepatic extraction
    Qh = 90.0  # L/h
    hepatic_cl = (Qh * clint_scaled_L_h) / (Qh + clint_scaled_L_h)

    # Confidence: higher when within the interior of the clamp range
    if clint_raw < 0.1 or clint_raw > 1000.0:
        confidence = "low"
    elif 1.0 <= clint <= 500.0:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "clint_uL_min_mg": clint,
        "clint_unbound": clint_unbound,
        "hepatic_cl_estimate_L_per_h": hepatic_cl,
        "prediction_confidence": confidence,
    }


def cyp_substrate_probability(
    logP: float,
    mw: float,
    psa: float,
) -> dict:
    """
    Estimate CYP substrate probability using logistic scoring.

    Formulas
    --------
    CYP3A4: p = sigmoid(0.3*logP + 0.001*(mw-400) - 0.005*psa - 0.5)
    CYP2D6: p = sigmoid(0.2*logP - 0.008*psa + 0.5) * 0.6
    CYP2C9: p = sigmoid(0.1*logP - 0.004*psa - 0.2)

    Returns
    -------
    dict: cyp3a4_prob, cyp2d6_prob, cyp2c9_prob, dominant_cyp
    """
    cyp3a4 = _sigmoid(0.3 * logP + 0.001 * (mw - 400.0) - 0.005 * psa - 0.5)
    cyp2d6 = _sigmoid(0.2 * logP - 0.008 * psa + 0.5) * 0.6
    cyp2c9 = _sigmoid(0.1 * logP - 0.004 * psa - 0.2)

    probs = {
        "CYP3A4": cyp3a4,
        "CYP2D6": cyp2d6,
        "CYP2C9": cyp2c9,
    }
    dominant_cyp = max(probs, key=lambda k: probs[k])

    return {
        "cyp3a4_prob": cyp3a4,
        "cyp2d6_prob": cyp2d6,
        "cyp2c9_prob": cyp2c9,
        "dominant_cyp": dominant_cyp,
    }
