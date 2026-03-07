"""Volume of distribution prediction using multiple methods (Oie-Tozer and QSAR)."""

from __future__ import annotations

import math


def oie_tozer_vss(
    fu_plasma: float,
    fu_tissue: float,
    vp: float = 3.0,
    ve: float = 7.6,
    r: float = 0.55,
) -> float:
    """Predict Vss using the Oie-Tozer method.

    Uses simplified practical form: Vss = Vp + Vt * fu_plasma / fu_tissue
    where Vt = 38 L (total tissue water volume for 70 kg human).

    Args:
        fu_plasma: Fraction unbound in plasma (0, 1]
        fu_tissue: Fraction unbound in tissue (0, 1]
        vp: Plasma volume in L (default 3.0 L/70 kg)
        ve: Extracellular fluid volume excluding plasma in L (default 7.6 L)
        r: Ratio of erythrocyte water to plasma water (default 0.55)

    Returns:
        Vss in L for a 70 kg human
    """
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if not (0.0 < fu_tissue <= 1.0):
        raise ValueError(f"fu_tissue must be in (0, 1], got {fu_tissue}")

    # Total tissue volume (non-plasma body water ~38 L for 70 kg)
    vt = 38.0
    # Oie-Tozer practical: Vss = Vp + Vt * (fu_p / fu_t)
    vss = vp + vt * (fu_plasma / fu_tissue)
    return vss


def predict_vd_from_logP(
    logP: float,
    pka: float | None = None,
    drug_type: str = "neutral",
    fu_plasma: float = 0.1,
    mw: float = 400.0,
) -> dict:
    """Predict volume of distribution from logP using empirical QSAR.

    Args:
        logP: Lipophilicity (log partition coefficient octanol/water)
        pka: pKa of the drug (optional, for ionization context)
        drug_type: One of 'neutral', 'weak_acid', 'weak_base'
        fu_plasma: Fraction unbound in plasma (0, 1]
        mw: Molecular weight in g/mol

    Returns:
        dict with vd_L, vd_L_per_kg, logP, drug_type, notes
    """
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")

    valid_types = ("neutral", "weak_acid", "weak_base")
    if drug_type not in valid_types:
        raise ValueError(f"drug_type must be one of {valid_types}, got {drug_type!r}")

    # Empirical QSAR: base Vd from logP by drug type
    if drug_type == "neutral":
        base_vd = math.exp(0.5 * logP + 1.5)
    elif drug_type == "weak_acid":
        base_vd = math.exp(0.3 * logP + 1.0)
    else:  # weak_base
        base_vd = math.exp(0.7 * logP + 2.0)

    # fu correction: higher fu → lower Vd
    fu_correction = (1.0 / fu_plasma) ** 0.5
    vd = base_vd * fu_correction

    # Clamp to physiologically plausible range
    vd = max(3.0, min(3000.0, vd))

    notes = f"QSAR estimate for {drug_type} compound (logP={logP:.1f}, fu={fu_plasma:.3f})"
    if pka is not None:
        notes += f", pKa={pka:.1f}"

    return {
        "vd_L": vd,
        "vd_L_per_kg": vd / 70.0,
        "logP": logP,
        "drug_type": drug_type,
        "notes": notes,
    }


def predict_vd_ensemble(
    logP: float,
    fu_plasma: float,
    fu_tissue: float,
    pka: float | None = None,
    drug_type: str = "neutral",
    mw: float = 400.0,
) -> dict:
    """Predict Vss using an ensemble of Oie-Tozer and QSAR methods.

    Args:
        logP: Lipophilicity
        fu_plasma: Fraction unbound in plasma (0, 1]
        fu_tissue: Fraction unbound in tissue (0, 1]
        pka: pKa (optional)
        drug_type: One of 'neutral', 'weak_acid', 'weak_base'
        mw: Molecular weight in g/mol

    Returns:
        dict with vd_oie_tozer_L, vd_qsar_L, vd_ensemble_L, vd_ensemble_L_per_kg,
        confidence ('high' if within 2x, 'low' otherwise)
    """
    vd_ot = oie_tozer_vss(fu_plasma, fu_tissue)
    qsar_result = predict_vd_from_logP(
        logP, pka=pka, drug_type=drug_type, fu_plasma=fu_plasma, mw=mw
    )
    vd_qsar = qsar_result["vd_L"]

    vd_ensemble = (vd_ot + vd_qsar) / 2.0

    # Confidence: high if both predictions are within 2x of each other
    if vd_ot > 0 and vd_qsar > 0:
        ratio = max(vd_ot, vd_qsar) / min(vd_ot, vd_qsar)
        confidence = "high" if ratio <= 2.0 else "low"
    else:
        confidence = "low"

    return {
        "vd_oie_tozer_L": vd_ot,
        "vd_qsar_L": vd_qsar,
        "vd_ensemble_L": vd_ensemble,
        "vd_ensemble_L_per_kg": vd_ensemble / 70.0,
        "confidence": confidence,
    }
