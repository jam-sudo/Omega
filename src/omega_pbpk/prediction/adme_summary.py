"""Comprehensive ADME summary prediction from physicochemical properties.

Predicts absorption, distribution, metabolism, and excretion characteristics
from molecular descriptors using established pharmacokinetic relationships.

References:
- Lipinski et al. (2001) Rule of Five
- Oie-Tozer Vd prediction
- BCS classification (FDA 2000)
- Poulin & Theil CNS penetration criteria
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ADMESummary:
    """Comprehensive ADME summary from physicochemical properties."""

    # Input descriptors
    drug_name: str
    mw: float
    logP: float
    psa: float
    n_hbd: int
    n_hba: int

    # Absorption
    bcs_class: str  # "I", "II", "III", "IV"
    lipinski_pass: bool
    absorption_class: str  # "high" or "low"
    absorption_notes: str

    # Distribution
    vd_estimated_L_per_kg: float
    ppb_pct: float  # plasma protein binding %
    cns_penetration: bool

    # Metabolism & Excretion
    primary_cyp: str
    excretion_route: str  # "renal", "hepatic", "mixed"
    t_half_estimated_h: float

    # Risk & scoring
    risk_flags: tuple[str, ...]
    overall_adme_score: float  # 0-100, higher = better
    notes: str


def predict_adme_summary(
    drug_name: str,
    mw: float,
    logP: float,
    psa: float,
    n_hbd: int,
    n_hba: int,
    pka: float = 7.0,
    molecule_type: str = "neutral",
    fu_plasma: float = 0.1,
    logD_pH74: float | None = None,
) -> ADMESummary:
    """Generate comprehensive ADME summary from physicochemical properties.

    Parameters
    ----------
    drug_name : str
        Drug name (non-empty).
    mw : float
        Molecular weight (g/mol); must be > 0.
    logP : float
        Lipophilicity (octanol-water partition coefficient).
    psa : float
        Polar surface area (Å²); must be >= 0.
    n_hbd : int
        Number of hydrogen bond donors; must be >= 0.
    n_hba : int
        Number of hydrogen bond acceptors; must be >= 0.
    pka : float
        Most basic pKa (used for ionisation state at pH 7.4).
    molecule_type : str
        "neutral", "acid", "base", or "zwitterion".
    fu_plasma : float
        Fraction unbound in plasma (0, 1]; must be > 0.
    logD_pH74 : float or None
        LogD at pH 7.4; if None, estimated from logP and pKa.

    Returns
    -------
    ADMESummary
    """
    # --- Input validation ---
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if psa < 0:
        raise ValueError("psa must be >= 0")
    if n_hbd < 0:
        raise ValueError("n_hbd must be >= 0")
    if n_hba < 0:
        raise ValueError("n_hba must be >= 0")
    if fu_plasma <= 0 or fu_plasma > 1.0:
        raise ValueError("fu_plasma must be in (0, 1]")
    if molecule_type not in ("neutral", "acid", "base", "zwitterion"):
        raise ValueError("molecule_type must be 'neutral', 'acid', 'base', or 'zwitterion'")

    # --- Effective logD at pH 7.4 ---
    if logD_pH74 is None:
        if molecule_type == "base":
            # Henderson-Hasselbalch: logD = logP - log(1 + 10^(pKa - pH))
            logD_pH74 = logP - math.log10(1.0 + 10.0 ** (pka - 7.4))
        elif molecule_type == "acid":
            logD_pH74 = logP - math.log10(1.0 + 10.0 ** (7.4 - pka))
        else:
            logD_pH74 = logP

    # =========================================================================
    # ABSORPTION
    # =========================================================================

    # Permeability proxy: high if PSA < 140 and logP > -1
    high_permeability = psa < 140.0 and logP > -1.0

    # Solubility proxy: high if logP < 2 and MW < 500
    high_solubility = logP < 2.0 and mw < 500.0

    # BCS class
    if high_permeability and high_solubility:
        bcs_class = "I"
    elif high_permeability and not high_solubility:
        bcs_class = "II"
    elif not high_permeability and high_solubility:
        bcs_class = "III"
    else:
        bcs_class = "IV"

    absorption_class = "high" if high_permeability else "low"

    absorption_notes_parts = [f"BCS Class {bcs_class}"]
    if not high_permeability:
        absorption_notes_parts.append(f"low permeability (PSA={psa:.0f} Å², logP={logP:.1f})")
    if not high_solubility:
        absorption_notes_parts.append(f"low solubility proxy (logP={logP:.1f}, MW={mw:.0f})")
    if high_permeability and high_solubility:
        absorption_notes_parts.append("good absorption expected")
    absorption_notes = "; ".join(absorption_notes_parts)

    # Lipinski Rule of Five
    lipinski_pass = mw <= 500.0 and logP <= 5.0 and n_hbd <= 5 and n_hba <= 10

    # =========================================================================
    # DISTRIBUTION
    # =========================================================================

    # Oie-Tozer simplified Vd prediction (L/kg)
    # Vd = Vp + Vt * (fu_plasma / fu_tissue)
    # Simplified: Vd correlates with lipophilicity and protein binding
    # Using empirical: logVd ~ 0.5 * logP - log(fu_plasma) - 0.5
    # Clamp to physiological range [0.04, 20] L/kg
    vp = 0.04  # plasma volume (L/kg)
    # Tissue binding estimated from logP
    fu_tissue = 1.0 / (1.0 + 0.5 * (10.0 ** max(-2.0, min(logP, 4.0))))
    vt = 0.38  # tissue volume fraction (L/kg, ~38% body weight)
    vd_estimated = vp + vt * (fu_plasma / fu_tissue)
    vd_estimated = max(0.04, min(20.0, vd_estimated))

    # Plasma protein binding
    # High logP and low fu → high PPB
    ppb_pct = (1.0 - fu_plasma) * 100.0

    # CNS penetration: logP 1-3, PSA < 90, MW < 450, P-gp not substrate (basic amine with pKa)
    cns_logP_ok = 1.0 <= logP <= 3.0
    cns_psa_ok = psa < 90.0
    cns_mw_ok = mw < 450.0
    cns_penetration = cns_logP_ok and cns_psa_ok and cns_mw_ok

    # =========================================================================
    # METABOLISM
    # =========================================================================

    # Primary CYP assignment from logP and molecule type
    is_basic_amine = molecule_type == "base" and pka > 7.0

    if logP > 3.0:
        primary_cyp = "CYP3A4"
    elif logP < 0.0:
        primary_cyp = "renal"  # predominantly renal elimination
    elif is_basic_amine:
        primary_cyp = "CYP2D6"
    elif molecule_type == "acid":
        primary_cyp = "CYP2C9"
    else:
        primary_cyp = "CYP3A4"  # most common for neutral lipophilic drugs

    # =========================================================================
    # EXCRETION
    # =========================================================================

    # Renal vs hepatic dominant
    # High hydrophilicity (logP < 1) → renal; high lipophilicity → hepatic
    if logP < 0.0 or primary_cyp == "renal":
        excretion_route = "renal"
    elif logP > 2.0:
        excretion_route = "hepatic"
    else:
        excretion_route = "mixed"

    # Estimated half-life (h) from clearance proxies
    # Lipophilic, high protein binding → longer t½; hydrophilic → shorter
    # Very rough empirical estimate: t½ ~ Vd(L/kg) * 70(kg) * ln(2) / CL_est(L/h)
    body_weight_kg = 70.0
    # CL rough estimate: 0.1-50 L/h based on logP
    cl_est = 2.0 + 5.0 * max(0.0, logP)  # L/h, rough
    cl_est = max(0.5, min(60.0, cl_est))
    t_half_estimated_h = vd_estimated * body_weight_kg * math.log(2.0) / cl_est
    t_half_estimated_h = max(0.5, min(200.0, t_half_estimated_h))

    # =========================================================================
    # RISK FLAGS
    # =========================================================================

    risk_flags_list: list[str] = []

    # Poor absorption: BCS III or IV, or Lipinski violation
    if not high_permeability or not lipinski_pass:
        risk_flags_list.append("poor_absorption")

    # High metabolic clearance: high logP → extensive CYP metabolism
    if logP > 4.0 and fu_plasma < 0.1:
        risk_flags_list.append("high_metabolic_clearance")

    # Narrow TI risk: extreme lipophilicity / protein binding / CNS active
    if logP > 5.0 or (ppb_pct > 99.0 and cns_penetration):
        risk_flags_list.append("narrow_TI_risk")

    # Poor CNS penetration (relevant for CNS-targeted drugs)
    if not cns_penetration:
        risk_flags_list.append("poor_CNS_penetration")

    # =========================================================================
    # OVERALL ADME SCORE (0-100, higher = better)
    # =========================================================================

    score = 100.0

    # Penalize BCS class
    bcs_penalty = {"I": 0, "II": 15, "III": 20, "IV": 35}
    score -= bcs_penalty[bcs_class]

    # Penalize Lipinski violation
    if not lipinski_pass:
        score -= 15.0

    # Penalize risk flags (each flag costs points)
    score -= len(risk_flags_list) * 8.0

    # Reward CNS penetration (not penalized if drug intended for CNS)
    # Neutral: CNS penetration is neither rewarded nor penalized here

    # Penalize extreme molecular weight
    if mw > 600:
        score -= 10.0
    elif mw < 150:
        score -= 5.0

    # Reward good distribution
    if 0.3 <= vd_estimated <= 2.0:
        score += 5.0

    score = max(0.0, min(100.0, score))

    # =========================================================================
    # NOTES
    # =========================================================================

    notes_parts = [
        f"MW={mw:.0f}, logP={logP:.2f}, PSA={psa:.0f} Å²",
        f"BCS {bcs_class}, {'Lipinski pass' if lipinski_pass else 'Lipinski FAIL'}",
        f"Vd~{vd_estimated:.2f} L/kg, PPB={ppb_pct:.0f}%",
        f"Primary CYP: {primary_cyp}",
        f"Excretion: {excretion_route}",
        f"t½~{t_half_estimated_h:.1f} h",
    ]
    notes = "; ".join(notes_parts)

    return ADMESummary(
        drug_name=drug_name,
        mw=mw,
        logP=logP,
        psa=psa,
        n_hbd=n_hbd,
        n_hba=n_hba,
        bcs_class=bcs_class,
        lipinski_pass=lipinski_pass,
        absorption_class=absorption_class,
        absorption_notes=absorption_notes,
        vd_estimated_L_per_kg=vd_estimated,
        ppb_pct=ppb_pct,
        cns_penetration=cns_penetration,
        primary_cyp=primary_cyp,
        excretion_route=excretion_route,
        t_half_estimated_h=t_half_estimated_h,
        risk_flags=tuple(risk_flags_list),
        overall_adme_score=score,
        notes=notes,
    )


__all__ = ["ADMESummary", "predict_adme_summary"]
