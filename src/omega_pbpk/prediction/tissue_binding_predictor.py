"""
Phase 569 — Tissue Binding Predictor

Predicts non-specific tissue binding parameters (fu_tissue, Kp_tissue)
from physicochemical properties using empirical lipid/phospholipid fractions.
"""

# Empirical lipid fractions per tissue (neutral lipid %)
_NEUTRAL_LIPID = {
    "muscle": 0.01,
    "adipose": 0.80,
    "liver": 0.04,
    "brain": 0.04,
    "kidney": 0.02,
    "lung": 0.01,
    "heart": 0.01,
}

# Phospholipid fractions per tissue
_PHOSPHOLIPID = {
    "muscle": 0.07,
    "adipose": 0.002,
    "liver": 0.10,
    "brain": 0.35,
    "kidney": 0.10,
    "lung": 0.07,
    "heart": 0.10,
}

_VALID_TISSUES = set(_NEUTRAL_LIPID.keys())
_VALID_DRUG_TYPES = {"neutral", "acid", "base", "strong_base", "zwitterion"}


def predict_fu_tissue(logP: float, drug_type: str, tissue: str) -> dict:
    """Predict unbound fraction in tissue using empirical Kp model.

    Parameters
    ----------
    logP : float
        Octanol-water partition coefficient (log scale).
    drug_type : str
        One of: 'neutral', 'acid', 'base', 'strong_base', 'zwitterion'.
    tissue : str
        Target tissue name.

    Returns
    -------
    dict with keys: tissue, drug_type, kp_tissue, fu_tissue, distribution_category
    """
    if tissue not in _VALID_TISSUES:
        raise ValueError(f"Invalid tissue '{tissue}'. Valid options: {sorted(_VALID_TISSUES)}")
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(
            f"Invalid drug_type '{drug_type}'. Valid options: {sorted(_VALID_DRUG_TYPES)}"
        )

    P = 10.0**logP
    lipid_frac = _NEUTRAL_LIPID[tissue]
    phospholipid_frac = _PHOSPHOLIPID[tissue]

    kp_tissue = P * lipid_frac + (P**0.3) * phospholipid_frac + 0.9

    # Lysosomal trapping for strong bases
    if drug_type == "strong_base":
        kp_tissue *= 2.0

    fu_tissue = 1.0 / (1.0 + kp_tissue)

    if kp_tissue > 10.0:
        distribution_category = "extensive"
    elif kp_tissue >= 1.0:
        distribution_category = "moderate"
    else:
        distribution_category = "low"

    return {
        "tissue": tissue,
        "drug_type": drug_type,
        "kp_tissue": kp_tissue,
        "fu_tissue": fu_tissue,
        "distribution_category": distribution_category,
    }


def kp_ratio(kp_tissue: float, fu_plasma: float) -> float:
    """Compute Kp ratio: Kp_tissue * fu_plasma.

    If Kp_ratio > 1, the drug accumulates in the tissue relative to plasma.

    Parameters
    ----------
    kp_tissue : float
        Tissue-to-plasma partition coefficient.
    fu_plasma : float
        Unbound fraction in plasma (0–1).

    Returns
    -------
    float
        Kp_ratio = Kp_tissue * fu_plasma
    """
    return kp_tissue * fu_plasma


def vd_from_tissue_binding(fu_plasma: float, tissue_kp_list: list) -> float:
    """Estimate apparent volume of distribution from tissue Kp values.

    Vd = Vp + sum(Vt_i * Kp_i)

    Parameters
    ----------
    fu_plasma : float
        Unbound fraction in plasma (not used directly in the simplified formula,
        but kept for API consistency).
    tissue_kp_list : list of (volume_L, kp) tuples
        Each tuple contains the tissue volume (L) and its Kp_tissue.

    Returns
    -------
    float
        Estimated Vd in litres.
    """
    Vp = 3.0  # plasma volume in L
    vd = Vp + sum(vt * kp for vt, kp in tissue_kp_list)
    return vd


def screen_tissue_distribution(logP: float, drug_type: str, fu_plasma: float) -> list:
    """Evaluate tissue distribution across all 7 tissues.

    Parameters
    ----------
    logP : float
        Octanol-water partition coefficient.
    drug_type : str
        Drug ionisation type.
    fu_plasma : float
        Unbound fraction in plasma.

    Returns
    -------
    list of dict, sorted by kp_tissue descending.
        Each entry: tissue, kp_tissue, fu_tissue, accumulation (bool).
    """
    results = []
    for tissue in _VALID_TISSUES:
        pred = predict_fu_tissue(logP, drug_type, tissue)
        accumulation = (pred["kp_tissue"] * fu_plasma) > 1.0
        results.append(
            {
                "tissue": tissue,
                "kp_tissue": pred["kp_tissue"],
                "fu_tissue": pred["fu_tissue"],
                "accumulation": accumulation,
            }
        )
    results.sort(key=lambda x: x["kp_tissue"], reverse=True)
    return results
