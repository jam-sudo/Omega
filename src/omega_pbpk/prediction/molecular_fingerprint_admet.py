"""
Phase 1006 — Molecular Fingerprint-based ADMET Predictor
Simplified ECFP-style predictions from comma-separated structural feature flags.
No external dependencies (stdlib only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Feature contribution table: feature -> (delta_mw, delta_logp, delta_hbd, delta_hba, delta_rings)
_FEATURE_CONTRIBUTIONS: dict[str, tuple[float, float, int, int, int]] = {
    "aromatic_ring": (20.0, 1.0, 0, 2, 1),
    "aliphatic_chain_C4": (56.0, 2.0, 0, 0, 0),
    "hydroxyl": (17.0, -1.0, 1, 1, 0),
    "carboxylic_acid": (45.0, -1.5, 1, 2, 0),
    "amine_primary": (16.0, -0.5, 2, 1, 0),
    "amine_tertiary": (14.0, 0.5, 0, 1, 0),
    "ester": (44.0, 0.5, 0, 2, 0),
    "halogen_F": (19.0, 0.4, 0, 0, 0),
    "halogen_Cl": (35.0, 0.6, 0, 0, 0),
}

_KNOWN_FEATURES = set(_FEATURE_CONTRIBUTIONS.keys())


@dataclass(frozen=True)
class FingerprintADMETResult:
    """Result of fingerprint-based ADMET prediction."""

    drug_name: str
    smiles_fragment_count: int
    mw_predicted: float
    logp_predicted: float
    hbd_count: int
    hba_count: int
    aromatic_ring_count: int
    predicted_caco2_papp: float
    predicted_bioavailability: float
    predicted_t_half_h: float
    predicted_vd_L_per_kg: float
    bbb_penetration_probability: float
    drug_likeness_score: float
    lipinski_pass: bool
    notes: str


def predict_admet_from_features(
    drug_name: str,
    feature_string: str,
    base_mw: float = 100.0,
) -> FingerprintADMETResult:
    """Predict ADMET properties from structural feature flags.

    Parameters
    ----------
    drug_name:
        Name of the compound.
    feature_string:
        Comma-separated list of feature flags (e.g. "aromatic_ring,hydroxyl").
    base_mw:
        Base molecular weight before feature contributions (default 100 Da).

    Returns
    -------
    FingerprintADMETResult
    """
    # Parse features
    if feature_string.strip():
        raw_features = [f.strip() for f in feature_string.split(",")]
    else:
        raw_features = []

    fragment_count = len(raw_features)

    # Accumulate contributions from known features (unknown silently ignored)
    mw = base_mw
    logp = 0.0
    hbd = 0
    hba = 0
    ring_count = 0

    for feat in raw_features:
        if feat in _FEATURE_CONTRIBUTIONS:
            d_mw, d_logp, d_hbd, d_hba, d_rings = _FEATURE_CONTRIBUTIONS[feat]
            mw += d_mw
            logp += d_logp
            hbd += d_hbd
            hba += d_hba
            ring_count += d_rings

    # Clamp values
    logp = max(-3.0, min(8.0, logp))
    hbd = max(0, min(20, hbd))
    hba = max(0, min(20, hba))

    # PSA estimate
    psa_estimate = 20.0 * hbd + 10.0 * hba

    # Caco-2 Papp (cm/s) from permeability model
    log_papp = -0.90 - 0.011 * psa_estimate - 0.24 * hbd + 0.27 * logp - 0.0015 * mw
    predicted_caco2_papp = 10.0**log_papp

    # Oral bioavailability
    f_bio = 1.0 / (1.0 + math.exp(-2.0 * (logp - 2.0))) * 0.8
    f_bio = min(0.95, max(0.0, f_bio))

    # t½ (hours)
    logp_above_2 = 1 if logp > 2 else 0
    mw_below_300 = 1 if mw < 300 else 0
    t_half_min = 450.0 / max(2.0, 5.0 + 30.0 * logp_above_2 + 25.0 * mw_below_300)
    t_half_h = t_half_min / 60.0

    # Vd (L/kg)
    vd = max(0.1, (10.0 ** (0.3 * logp)) * (1.0 / (1.0 + hbd * 0.1)))

    # BBB penetration probability
    bbb = 1.0 / (1.0 + math.exp(-(logp - 1.5 - 0.02 * psa_estimate)))

    # Lipinski rule of 5
    lipinski_pass = mw < 500 and logp < 5 and hbd < 5 and hba < 10

    # Drug-likeness score (0–100)
    dl_score = 0.0
    dl_score += 40.0 if lipinski_pass else 0.0
    dl_score += 30.0 if (0.0 <= logp <= 5.0) else 0.0
    dl_score += 30.0 if (150.0 <= mw <= 500.0) else 0.0

    unknown_feats = [f for f in raw_features if f not in _KNOWN_FEATURES and f != ""]
    notes_parts = [
        f"Features detected: {fragment_count}",
        f"MW={mw:.1f}, logP={logp:.2f}, HBD={hbd}, HBA={hba}, rings={ring_count}",
    ]
    if unknown_feats:
        notes_parts.append(f"Unknown features ignored: {', '.join(unknown_feats)}")

    return FingerprintADMETResult(
        drug_name=drug_name,
        smiles_fragment_count=fragment_count,
        mw_predicted=mw,
        logp_predicted=logp,
        hbd_count=hbd,
        hba_count=hba,
        aromatic_ring_count=ring_count,
        predicted_caco2_papp=predicted_caco2_papp,
        predicted_bioavailability=f_bio,
        predicted_t_half_h=t_half_h,
        predicted_vd_L_per_kg=vd,
        bbb_penetration_probability=bbb,
        drug_likeness_score=dl_score,
        lipinski_pass=lipinski_pass,
        notes="; ".join(notes_parts),
    )


def screen_admet_features(compounds: list[dict]) -> list[FingerprintADMETResult]:
    """Screen a list of compounds and return results sorted by drug_likeness_score.

    Parameters
    ----------
    compounds:
        List of dicts, each with keys:
        - drug_name (str)
        - feature_string (str)
        - base_mw (float, optional, default 100.0)

    Returns
    -------
    List of FingerprintADMETResult sorted by drug_likeness_score descending.
    """
    results = []
    for cmpd in compounds:
        name = cmpd["drug_name"]
        features = cmpd["feature_string"]
        base_mw = float(cmpd.get("base_mw", 100.0))
        result = predict_admet_from_features(name, features, base_mw)
        results.append(result)

    results.sort(key=lambda r: r.drug_likeness_score, reverse=True)
    return results
