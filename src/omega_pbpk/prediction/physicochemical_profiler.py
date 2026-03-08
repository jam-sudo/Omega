"""Phase 1040 — Comprehensive physicochemical property profiler.

Combines Lipinski, Veber, Egan, CNS MPO, BCS, and Rule-of-3 rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

_VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class PhysicochemicalProfile:
    drug_name: str
    mw_Da: float
    logp: float
    hbd: int
    hba: int
    psa_A2: float
    rotatable_bonds: int
    # Calculated
    logd_74: float
    pka: float
    ionization_type: str
    # Rule-based flags
    lipinski_pass: bool
    veber_pass: bool
    egan_pass: bool
    cns_mpo_score: float
    bcs_class: str
    ro3_pass: bool
    # Summary
    drug_likeness_score: float
    development_risk: str
    notes: str


def _calc_logd(logp: float, pka: float, ionization_type: str, ph: float = 7.4) -> float:
    """Calculate logD at a given pH."""
    if ionization_type == "acid":
        logd = logp - math.log10(1.0 + 10 ** (ph - pka))
    elif ionization_type == "base":
        logd = logp - math.log10(1.0 + 10 ** (pka - ph))
    else:
        logd = logp
    return max(-5.0, min(10.0, logd))


def _calc_cns_mpo(
    mw_Da: float,
    logp: float,
    logd: float,
    hbd: int,
    psa_A2: float,
    pka: float,
    ionization_type: str,
) -> float:
    """Compute CNS MPO score (0–6)."""
    # 1. MW
    if mw_Da <= 360.0:
        s_mw = 1.0
    elif mw_Da <= 500.0:
        s_mw = (500.0 - mw_Da) / 140.0
    else:
        s_mw = 0.0

    # 2. logP
    if logp <= 3.0:
        s_logp = 1.0
    elif logp <= 5.0:
        s_logp = (5.0 - logp) / 2.0
    else:
        s_logp = 0.0

    # 3. logD
    if -2.0 <= logd <= 5.0:
        s_logd = 1.0
    else:
        s_logd = 0.0

    # 4. HBD
    if hbd == 0:
        s_hbd = 1.0
    elif hbd == 1:
        s_hbd = 0.75
    elif hbd == 2:
        s_hbd = 0.5
    elif hbd == 3:
        s_hbd = 0.25
    else:
        s_hbd = 0.0

    # 5. PSA
    if psa_A2 <= 60.0:
        s_psa = 1.0
    elif psa_A2 <= 90.0:
        s_psa = (90.0 - psa_A2) / 30.0
    else:
        s_psa = 0.0

    # 6. pKa (basic)
    if ionization_type == "base":
        if pka <= 8.0:
            s_pka = 1.0
        elif pka <= 10.0:
            s_pka = (10.0 - pka) / 2.0
        else:
            s_pka = 0.0
    elif ionization_type == "neutral":
        s_pka = 1.0
    else:  # acid
        s_pka = 0.5

    score = s_mw + s_logp + s_logd + s_hbd + s_psa + s_pka
    return max(0.0, min(6.0, score))


def _calc_bcs(logp: float, mw_Da: float) -> str:
    """Simplified BCS classification."""
    high_solubility = logp <= 2.0
    high_permeability = logp >= 1.0 and mw_Da <= 450.0
    if high_solubility and high_permeability:
        return "BCS I"
    elif not high_solubility and high_permeability:
        return "BCS II"
    elif high_solubility and not high_permeability:
        return "BCS III"
    else:
        return "BCS IV"


def profile_drug(
    drug_name: str,
    mw_Da: float,
    logp: float,
    hbd: int,
    hba: int,
    psa_A2: float,
    rotatable_bonds: int,
    pka: float = 7.0,
    ionization_type: str = "neutral",
) -> PhysicochemicalProfile:
    """Generate a comprehensive physicochemical profile for a drug candidate.

    Parameters
    ----------
    drug_name:
        Name of the compound.
    mw_Da:
        Molecular weight in Daltons (must be > 0).
    logp:
        Octanol-water partition coefficient.
    hbd:
        Number of hydrogen-bond donors (must be >= 0).
    hba:
        Number of hydrogen-bond acceptors (must be >= 0).
    psa_A2:
        Polar surface area in Ångström² (must be >= 0).
    rotatable_bonds:
        Number of rotatable bonds (must be >= 0).
    pka:
        Acid/base dissociation constant (must be in [0, 14]).
    ionization_type:
        One of "acid", "base", "neutral".

    Returns
    -------
    PhysicochemicalProfile
    """
    # --- Validation ---
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if hbd < 0:
        raise ValueError(f"hbd must be >= 0, got {hbd}")
    if hba < 0:
        raise ValueError(f"hba must be >= 0, got {hba}")
    if psa_A2 < 0:
        raise ValueError(f"psa_A2 must be >= 0, got {psa_A2}")
    if rotatable_bonds < 0:
        raise ValueError(f"rotatable_bonds must be >= 0, got {rotatable_bonds}")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION_TYPES}, got {ionization_type!r}"
        )
    if not (0.0 <= pka <= 14.0):
        raise ValueError(f"pka must be in [0, 14], got {pka}")

    # --- Derived properties ---
    logd_74 = _calc_logd(logp, pka, ionization_type, ph=7.4)

    # --- Rules ---
    lipinski_pass = mw_Da < 500.0 and logp < 5.0 and hbd <= 5 and hba <= 10
    veber_pass = rotatable_bonds <= 10 and psa_A2 <= 140.0
    egan_pass = logp <= 5.88 and psa_A2 <= 131.6
    ro3_pass = mw_Da < 300.0 and logp < 3.0 and hbd <= 3 and hba <= 3

    cns_mpo_score = _calc_cns_mpo(mw_Da, logp, logd_74, hbd, psa_A2, pka, ionization_type)
    bcs_class = _calc_bcs(logp, mw_Da)

    # --- Drug-likeness score (0–100) ---
    lipinski_pts = 30.0 if lipinski_pass else 0.0
    veber_pts = 20.0 if veber_pass else 0.0
    egan_pts = 20.0 if egan_pass else 0.0
    cns_pts = (cns_mpo_score / 6.0) * 30.0
    drug_likeness_score = max(0.0, min(100.0, lipinski_pts + veber_pts + egan_pts + cns_pts))

    # --- Development risk ---
    if drug_likeness_score >= 70.0:
        development_risk = "low"
    elif drug_likeness_score >= 40.0:
        development_risk = "moderate"
    else:
        development_risk = "high"

    # --- Notes ---
    notes_parts: list[str] = []
    if lipinski_pass:
        notes_parts.append("Lipinski Ro5: PASS.")
    else:
        notes_parts.append("Lipinski Ro5: FAIL.")
    if veber_pass:
        notes_parts.append("Veber: PASS.")
    else:
        notes_parts.append("Veber: FAIL.")
    if egan_pass:
        notes_parts.append("Egan: PASS.")
    else:
        notes_parts.append("Egan: FAIL.")
    notes_parts.append(f"CNS MPO={cns_mpo_score:.2f}/6.0.")
    notes_parts.append(f"BCS class: {bcs_class}.")
    if ro3_pass:
        notes_parts.append("Rule-of-3 (fragment-like): PASS.")
    notes = " ".join(notes_parts)

    return PhysicochemicalProfile(
        drug_name=drug_name,
        mw_Da=mw_Da,
        logp=logp,
        hbd=hbd,
        hba=hba,
        psa_A2=psa_A2,
        rotatable_bonds=rotatable_bonds,
        logd_74=logd_74,
        pka=pka,
        ionization_type=ionization_type,
        lipinski_pass=lipinski_pass,
        veber_pass=veber_pass,
        egan_pass=egan_pass,
        cns_mpo_score=cns_mpo_score,
        bcs_class=bcs_class,
        ro3_pass=ro3_pass,
        drug_likeness_score=drug_likeness_score,
        development_risk=development_risk,
        notes=notes,
    )


def compare_candidates(candidates: list[dict[str, Any]]) -> list[PhysicochemicalProfile]:
    """Profile multiple drug candidates and return sorted by drug_likeness_score descending.

    Each dict should contain keys matching the parameters of :func:`profile_drug`.

    Parameters
    ----------
    candidates:
        List of dicts with profiling parameters.

    Returns
    -------
    List of PhysicochemicalProfile sorted by drug_likeness_score descending.
    """
    profiles = [profile_drug(**c) for c in candidates]
    return sorted(profiles, key=lambda p: p.drug_likeness_score, reverse=True)
