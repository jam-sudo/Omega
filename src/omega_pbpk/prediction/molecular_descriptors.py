"""Phase 575 — Molecular descriptor calculation from structural features."""

from __future__ import annotations


def lipinski_ro5(mw: float, logP: float, n_hbd: int, n_hba: int) -> dict:
    """Lipinski Rule of 5 assessment.

    Args:
        mw: Molecular weight (Da)
        logP: Octanol-water partition coefficient
        n_hbd: Number of hydrogen bond donors
        n_hba: Number of hydrogen bond acceptors

    Returns:
        dict with violations (list), passes_ro5 (bool), drug_like_score (0-4)
    """
    violations = []
    if mw > 500:
        violations.append("MW > 500")
    if logP > 5:
        violations.append("logP > 5")
    if n_hbd > 5:
        violations.append("HBD > 5")
    if n_hba > 10:
        violations.append("HBA > 10")

    drug_like_score = 4 - len(violations)
    passes_ro5 = len(violations) == 0

    return {
        "violations": violations,
        "passes_ro5": passes_ro5,
        "drug_like_score": drug_like_score,
    }


def veber_rules(psa: float, n_rotatable_bonds: int) -> dict:
    """Veber oral bioavailability rules.

    Args:
        psa: Polar surface area (Angstrom^2)
        n_rotatable_bonds: Number of rotatable bonds

    Returns:
        dict with passes_veber (bool), violations (list)
    """
    violations = []
    if psa > 140:
        violations.append("PSA > 140 A^2")
    if n_rotatable_bonds > 10:
        violations.append("Rotatable bonds > 10")

    passes_veber = len(violations) == 0

    return {
        "passes_veber": passes_veber,
        "violations": violations,
    }


def ghose_filter(mw: float, logP: float, n_atoms: int, molar_refractivity: float) -> dict:
    """Ghose drug-likeness filter.

    Args:
        mw: Molecular weight (Da)
        logP: Octanol-water partition coefficient
        n_atoms: Number of heavy atoms
        molar_refractivity: Molar refractivity (cm^3/mol)

    Returns:
        dict with passes_ghose (bool), violations (list)
    """
    violations = []
    if mw < 160 or mw > 480:
        violations.append("MW not in [160, 480]")
    if logP < -0.4 or logP > 5.6:
        violations.append("logP not in [-0.4, 5.6]")
    if n_atoms < 20 or n_atoms > 70:
        violations.append("n_atoms not in [20, 70]")
    if molar_refractivity < 40 or molar_refractivity > 130:
        violations.append("MR not in [40, 130]")

    passes_ghose = len(violations) == 0

    return {
        "passes_ghose": passes_ghose,
        "violations": violations,
    }


def drug_likeness_score(
    mw: float,
    logP: float,
    n_hbd: int,
    n_hba: int,
    psa: float,
    n_rotatable_bonds: int,
) -> dict:
    """Combined drug-likeness score (0-100).

    Scoring:
        - Lipinski: 25 pts (4 rules, each worth 6.25)
        - Veber: 25 pts (2 rules, each worth 12.5)
        - logP penalty: up to -10 if logP > 5
        - MW penalty: up to -10 if MW > 500
        - PSA bonus: +5 if 50 < PSA < 120

    Args:
        mw: Molecular weight (Da)
        logP: Octanol-water partition coefficient
        n_hbd: Number of hydrogen bond donors
        n_hba: Number of hydrogen bond acceptors
        psa: Polar surface area (Angstrom^2)
        n_rotatable_bonds: Number of rotatable bonds

    Returns:
        dict with score_0_100, ro5_result, veber_result, category
    """
    ro5_result = lipinski_ro5(mw, logP, n_hbd, n_hba)
    veber_result = veber_rules(psa, n_rotatable_bonds)

    n_ro5_violations = len(ro5_result["violations"])
    n_veber_violations = len(veber_result["violations"])

    lipinski_pts = 25.0 - n_ro5_violations * 6.25
    veber_pts = 25.0 - n_veber_violations * 12.5

    score = lipinski_pts + veber_pts

    # logP penalty: up to -10 if logP > 5
    if logP > 5:
        excess = logP - 5.0
        logp_penalty = min(10.0, excess * 5.0)
        score -= logp_penalty

    # MW penalty: up to -10 if MW > 500
    if mw > 500:
        excess = mw - 500.0
        mw_penalty = min(10.0, excess * 0.05)
        score -= mw_penalty

    # PSA bonus: +5 if 50 < PSA < 120
    if 50.0 < psa < 120.0:
        score += 5.0

    score = max(0.0, min(100.0, score))

    # Max achievable score: Lipinski(25) + Veber(25) + PSA bonus(5) = 55
    # Category thresholds scaled to achievable maximum:
    if score >= 50:
        category = "excellent"
    elif score >= 35:
        category = "good"
    elif score >= 20:
        category = "moderate"
    else:
        category = "poor"

    return {
        "score_0_100": round(score, 2),
        "ro5_result": ro5_result,
        "veber_result": veber_result,
        "category": category,
    }


def molar_refractivity_estimate(mw: float, logP: float) -> float:
    """Estimate molar refractivity from MW and logP.

    Empirical approximation: MR ≈ 0.2 * MW + 5 * logP + 2

    Args:
        mw: Molecular weight (Da)
        logP: Octanol-water partition coefficient

    Returns:
        Estimated molar refractivity (cm^3/mol)
    """
    return 0.2 * mw + 5.0 * logP + 2.0
