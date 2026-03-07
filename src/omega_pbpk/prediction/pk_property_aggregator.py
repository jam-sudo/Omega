"""Phase 673 — PK Property Aggregator.

Aggregate and summarize multiple physicochemical and PK predictions into a
unified drug profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DrugProfile:
    compound_name: str
    logP: float
    mw: float
    psa: float
    # Predicted PK properties
    predicted_t_half_h: float
    predicted_vd_L_per_kg: float
    predicted_cl_L_per_h_per_kg: float
    predicted_f_oral: float
    predicted_fup: float
    predicted_logD_74: float
    predicted_solubility_mg_mL: float
    predicted_psa_permeability: str
    # Risk assessments
    bbb_penetration_likely: bool
    herg_risk_flag: bool
    metabolic_stability_class: str
    # Scores
    lipinski_score: int
    admet_profile_grade: str
    notes: str


def _predict_logD_74(logP: float, pka_acid: float | None, pka_base: float | None) -> float:
    pH = 7.4
    if pka_acid is not None and pka_base is None:
        # Acid: ionised fraction increases above pKa
        return logP - math.log10(1.0 + 10 ** (pH - pka_acid))
    elif pka_base is not None and pka_acid is None:
        # Base: ionised fraction increases below pKa
        return logP - math.log10(1.0 + 10 ** (pka_base - pH))
    elif pka_acid is not None and pka_base is not None:
        # Amphoteric: apply both corrections
        corr_acid = math.log10(1.0 + 10 ** (pH - pka_acid))
        corr_base = math.log10(1.0 + 10 ** (pka_base - pH))
        return logP - corr_acid - corr_base
    return logP


def _predict_fup(logP: float, n_hba: int) -> float:
    val = 1.0 / (1.0 + logP * 0.3 + n_hba * 0.05)
    return max(0.01, min(1.0, val))


def _predict_t_half_h(logP: float, clint: float) -> float:
    return max(0.5, 6.0 * (1.0 + logP) / (clint * 0.1 + 1.0))


def _predict_vd(logP: float, psa: float, fup: float) -> float:
    raw = 0.7 + logP * 0.5 - psa * 0.005
    val = max(0.1, raw) / max(0.01, fup)
    return max(0.1, min(50.0, val))


def _predict_f_oral(logP: float, psa: float) -> float:
    if logP < -1.0 or psa > 140.0:
        return 0.2
    if logP > 5.0:
        return 0.4
    return 0.85 * (1.0 - psa / 200.0)


def _predict_solubility(logP: float, psa: float, mw: float, melting_point_c: float) -> float:
    exponent = 0.5 - 0.01 * (melting_point_c - 25.0) - logP + psa * 0.008
    val = (10.0**exponent) * mw / 1000.0
    return max(0.001, val)


def _psa_permeability(psa: float) -> str:
    if psa < 60.0:
        return "high"
    if psa <= 120.0:
        return "moderate"
    return "low"


def _metabolic_stability_class(clint: float) -> str:
    # in vitro t½ = ln(2)*1000 / clint  (clint in µL/min/mg → min-equivalent)
    if clint <= 0.0:
        return "stable"
    t_half_invitro_min = math.log(2.0) * 1000.0 / clint
    if t_half_invitro_min > 120.0:
        return "stable"
    if t_half_invitro_min >= 30.0:
        return "moderate"
    return "labile"


def _lipinski_score(mw: float, logP: float, n_hbd: int, n_hba: int) -> int:
    score = 0
    if mw < 500.0:
        score += 1
    if logP < 5.0:
        score += 1
    if n_hbd <= 5:
        score += 1
    if n_hba <= 10:
        score += 1
    return score


def _admet_grade(
    lipinski: int,
    permeability: str,
    stability: str,
    herg_flag: bool,
) -> str:
    if lipinski >= 4 and stability == "stable" and permeability == "high":
        return "A"
    if lipinski >= 3 and stability in ("stable", "moderate") and not herg_flag:
        return "B"
    if lipinski >= 2:
        return "C"
    return "D"


def _build_notes(
    bbb: bool,
    herg: bool,
    stability: str,
    permeability: str,
    lipinski: int,
) -> str:
    parts: list[str] = []
    if bbb:
        parts.append("CNS penetration likely.")
    else:
        parts.append("CNS penetration unlikely.")
    if herg:
        parts.append("hERG risk: cationic lipophilic compound.")
    if stability == "labile":
        parts.append("Labile metabolic stability.")
    elif stability == "moderate":
        parts.append("Moderate metabolic stability.")
    else:
        parts.append("Stable metabolically.")
    parts.append(f"Lipinski score: {lipinski}/4.")
    parts.append(f"PSA permeability: {permeability}.")
    return " ".join(parts)


def build_drug_profile(
    compound_name: str,
    logP: float,
    mw: float,
    psa: float,
    n_hbd: int = 2,
    n_hba: int = 4,
    clint_uL_per_min_per_mg: float = 10.0,
    melting_point_c: float = 150.0,
    pka_acid: float | None = None,
    pka_base: float | None = None,
    measured_fup: float | None = None,
    measured_t_half_h: float | None = None,
) -> DrugProfile:
    """Build a unified drug profile from physicochemical inputs."""
    # Validation
    if mw <= 0.0:
        raise ValueError("mw must be positive")
    if psa < 0.0:
        raise ValueError("psa must be non-negative")
    if n_hbd < 0:
        raise ValueError("n_hbd must be non-negative")
    if n_hba < 0:
        raise ValueError("n_hba must be non-negative")
    if clint_uL_per_min_per_mg < 0.0:
        raise ValueError("clint_uL_per_min_per_mg must be non-negative")

    logD_74 = _predict_logD_74(logP, pka_acid, pka_base)

    pred_fup = _predict_fup(logP, n_hba)
    fup = measured_fup if measured_fup is not None else pred_fup

    pred_t_half = _predict_t_half_h(logP, clint_uL_per_min_per_mg)
    t_half = measured_t_half_h if measured_t_half_h is not None else pred_t_half

    vd = _predict_vd(logP, psa, fup)
    cl = math.log(2.0) * vd / t_half
    f_oral = _predict_f_oral(logP, psa)
    sol = _predict_solubility(logP, psa, mw, melting_point_c)
    permeability = _psa_permeability(psa)
    bbb = (1.0 <= logP <= 3.0) and (psa < 90.0) and (mw < 450.0)
    herg = (logP > 4.0) and (pka_base is not None) and (pka_base > 7.0)
    stability = _metabolic_stability_class(clint_uL_per_min_per_mg)
    lipinski = _lipinski_score(mw, logP, n_hbd, n_hba)
    grade = _admet_grade(lipinski, permeability, stability, herg)
    notes = _build_notes(bbb, herg, stability, permeability, lipinski)

    return DrugProfile(
        compound_name=compound_name,
        logP=logP,
        mw=mw,
        psa=psa,
        predicted_t_half_h=t_half,
        predicted_vd_L_per_kg=vd,
        predicted_cl_L_per_h_per_kg=cl,
        predicted_f_oral=f_oral,
        predicted_fup=fup,
        predicted_logD_74=logD_74,
        predicted_solubility_mg_mL=sol,
        predicted_psa_permeability=permeability,
        bbb_penetration_likely=bbb,
        herg_risk_flag=herg,
        metabolic_stability_class=stability,
        lipinski_score=lipinski,
        admet_profile_grade=grade,
        notes=notes,
    )


def compare_drug_profiles(
    profiles: list[DrugProfile], metric: str = "predicted_f_oral"
) -> list[DrugProfile]:
    """Return profiles sorted by the given metric descending."""
    return sorted(profiles, key=lambda p: getattr(p, metric), reverse=True)


def profile_similarity(profile1: DrugProfile, profile2: DrugProfile) -> float:
    """Compute similarity score (0-1) between two profiles based on key physicochemical props."""
    # Normalised Euclidean distance on [logP, mw/500, psa/150]
    d_logP = profile1.logP - profile2.logP
    d_mw = (profile1.mw - profile2.mw) / 500.0
    d_psa = (profile1.psa - profile2.psa) / 150.0
    dist = math.sqrt(d_logP**2 + d_mw**2 + d_psa**2)
    similarity = 1.0 - dist / math.sqrt(3.0)
    return max(0.0, min(1.0, similarity))
