"""CSF (cerebrospinal fluid) penetration predictor — Phase 794.

Predicts the extent to which a drug penetrates the blood-brain barrier
and reaches the CSF, an important consideration in CNS drug development.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "CSFPenetrationResult",
    "predict_csf_penetration",
    "screen_csf_penetration",
]


def _logistic(x: float) -> float:
    """Standard logistic function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    # Numerically stable for large negative x
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class CSFPenetrationResult:
    """Result of CSF penetration prediction."""

    compound_name: str
    logp: float
    mw_da: float
    psa_A2: float
    fup: float
    pka_basic: float | None
    predicted_kp_uu_csf: float  # unbound CSF / unbound plasma
    predicted_kp_total_csf: float  # total CSF / total plasma
    bbb_permeability: str  # "high", "moderate", "low"
    csf_exposure_class: str  # "excellent", "good", "limited", "poor"
    efflux_risk: str  # "high" or "low"
    pgp_substrate_probability: float  # 0-1
    mdr_mdr_score: float  # MDR1 efflux score (0-100)
    notes: str


def predict_csf_penetration(
    compound_name: str,
    logp: float,
    mw_da: float,
    psa_A2: float,
    fup: float,
    hbd_count: int = 0,
    pka_basic: float | None = None,
    is_pgp_substrate: bool | None = None,
) -> CSFPenetrationResult:
    """Predict CSF penetration for a drug candidate.

    Parameters
    ----------
    compound_name : str
        Name of the compound.
    logp : float
        Octanol/water partition coefficient (log scale).
    mw_da : float
        Molecular weight in Daltons (must be > 0).
    psa_A2 : float
        Polar surface area in square Angstroms.
    fup : float
        Fraction unbound in plasma (0 < fup <= 1).
    hbd_count : int
        Number of hydrogen bond donors.
    pka_basic : float or None
        Basic pKa value (optional; not used in current model but stored).
    is_pgp_substrate : bool or None
        If True/False, overrides the predicted P-gp substrate probability.
        If None, probability is predicted from physicochemical features.

    Returns
    -------
    CSFPenetrationResult
    """
    if mw_da <= 0:
        raise ValueError(f"mw_da must be > 0, got {mw_da}")
    if not (0.0 < fup <= 1.0):
        raise ValueError(f"fup must be in (0, 1], got {fup}")

    # --- BBB permeability (log Papp, cm/s) ---
    # logPapp = -5 + 0.6*logP - 0.01*PSA - 0.001*MW - 0.15*HBD
    log_papp = -5.0 + 0.6 * logp - 0.01 * psa_A2 - 0.001 * mw_da - 0.15 * hbd_count
    papp = 10.0**log_papp  # cm/s

    # Classify BBB permeability
    if log_papp > -5.0:
        bbb_permeability = "high"
    elif log_papp > -6.0:
        bbb_permeability = "moderate"
    else:
        bbb_permeability = "low"

    # --- Kp_uu_csf (before P-gp adjustment) ---
    # Kp_uu_csf = 1 / (1 + exp(-3 * log10(Papp / 1e-6))) * 0.8
    # papp is in cm/s; reference = 1e-6 cm/s
    papp_ratio = papp / 1e-6
    # Avoid log(0) — papp is always > 0 (10^something)
    if papp_ratio <= 0:
        papp_ratio = 1e-30
    log10_ratio = math.log10(papp_ratio)
    kp_uu_raw = (1.0 / (1.0 + math.exp(-3.0 * log10_ratio))) * 0.8
    kp_uu_csf = _clamp(kp_uu_raw, 0.01, 0.8)

    # --- P-gp substrate probability ---
    if is_pgp_substrate is True:
        pgp_prob = 0.99
    elif is_pgp_substrate is False:
        pgp_prob = 0.01
    else:
        # Predict: logistic(0.5*logP + 0.3*hbd - 0.02*psa + 0.01*mw - 2.0)
        pgp_score = 0.5 * logp + 0.3 * hbd_count - 0.02 * psa_A2 + 0.01 * mw_da - 2.0
        pgp_prob = _clamp(_logistic(pgp_score), 0.01, 0.99)

    # --- Apply P-gp efflux correction ---
    if pgp_prob > 0.5:
        kp_uu_csf = _clamp(kp_uu_csf * 0.70, 0.01, 0.8)
        efflux_risk = "high"
    else:
        efflux_risk = "low"

    # --- Kp_total_csf ---
    kp_total_csf = kp_uu_csf * fup

    # --- MDR score ---
    mdr_score = pgp_prob * 100.0

    # --- CSF exposure class ---
    if kp_uu_csf >= 0.3:
        csf_exposure_class = "excellent"
    elif kp_uu_csf >= 0.1:
        csf_exposure_class = "good"
    elif kp_uu_csf >= 0.05:
        csf_exposure_class = "limited"
    else:
        csf_exposure_class = "poor"

    notes = (
        f"logPapp={log_papp:.2f}, Kp_uu={kp_uu_csf:.3f}, "
        f"P-gp_prob={pgp_prob:.2f}, BBB={bbb_permeability}, "
        f"CSF_class={csf_exposure_class}"
    )

    return CSFPenetrationResult(
        compound_name=compound_name,
        logp=logp,
        mw_da=mw_da,
        psa_A2=psa_A2,
        fup=fup,
        pka_basic=pka_basic,
        predicted_kp_uu_csf=kp_uu_csf,
        predicted_kp_total_csf=kp_total_csf,
        bbb_permeability=bbb_permeability,
        csf_exposure_class=csf_exposure_class,
        efflux_risk=efflux_risk,
        pgp_substrate_probability=pgp_prob,
        mdr_mdr_score=mdr_score,
        notes=notes,
    )


def screen_csf_penetration(compounds: list[dict]) -> list[CSFPenetrationResult]:
    """Screen a list of compounds for CSF penetration.

    Each dict must contain at minimum: compound_name, logp, mw_da, psa_A2, fup.
    Optional keys: hbd_count, pka_basic, is_pgp_substrate.

    Returns results sorted by predicted_kp_uu_csf descending.
    """
    results = []
    for cpd in compounds:
        result = predict_csf_penetration(
            compound_name=cpd["compound_name"],
            logp=cpd["logp"],
            mw_da=cpd["mw_da"],
            psa_A2=cpd["psa_A2"],
            fup=cpd["fup"],
            hbd_count=cpd.get("hbd_count", 0),
            pka_basic=cpd.get("pka_basic", None),
            is_pgp_substrate=cpd.get("is_pgp_substrate", None),
        )
        results.append(result)

    results.sort(key=lambda r: r.predicted_kp_uu_csf, reverse=True)
    return results
