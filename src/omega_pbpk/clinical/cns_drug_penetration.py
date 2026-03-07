"""CNS drug penetration and target engagement prediction — Phase 508."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BBBPenetrationResult:
    """Result of BBB penetration prediction."""

    drug_name: str
    logP: float
    psa_A2: float
    bbb_score: float
    kp_uu_brain: float
    predicted_cns_exposure: str
    pgp_efflux_concern: bool
    notes: str


@dataclass(frozen=True)
class CNSEfficacyResult:
    """Result of CNS efficacy likelihood analysis."""

    predicted_occupancy_at_cmax_pct: float
    is_efficacious: bool
    mw_concern: bool
    psa_concern: bool
    optimal_logP: bool
    cns_mpo_score: float
    notes: str


def _logP_score(logP: float) -> float:
    """Score logP contribution (optimal 1-4). Returns 0-1."""
    if 1.0 <= logP <= 4.0:
        return 1.0
    elif logP < 0.0:
        return max(0.0, 1.0 + logP * 0.2)
    elif logP < 1.0:
        return 0.7 + logP * 0.3
    elif logP <= 5.0:
        return 1.0 - (logP - 4.0) * 0.3
    else:
        return max(0.0, 0.7 - (logP - 5.0) * 0.15)


def _psa_score(psa_A2: float) -> float:
    """Score PSA contribution (lower is better for BBB). Returns 0-1."""
    if psa_A2 <= 60.0:
        return 1.0
    elif psa_A2 <= 90.0:
        return 1.0 - (psa_A2 - 60.0) / 60.0
    else:
        return max(0.0, 1.0 - (psa_A2 - 60.0) / 120.0)


def calculate_kp_uu_brain(
    logP: float,
    mw: float,
    psa_A2: float,
    pgp_substrate: bool = False,
    fu_plasma: float = 0.1,
) -> float:
    """Calculate unbound brain-to-plasma ratio (Kp,uu).

    Parameters
    ----------
    logP:
        Lipophilicity (log partition coefficient).
    mw:
        Molecular weight in Da.
    psa_A2:
        Polar surface area in Angstrom squared.
    pgp_substrate:
        Whether the drug is a P-gp substrate.
    fu_plasma:
        Fraction unbound in plasma (0-1).

    Returns
    -------
    float
        Kp,uu,brain clamped to [0.01, 5.0].
    """
    if mw <= 0:
        raise ValueError("mw must be positive.")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be in (0, 1].")

    lp_score = _logP_score(logP)
    psa_pen = max(0.0, psa_A2 - 60.0)
    mw_pen = max(0.0, mw - 400.0)

    # Kp_uu model: exponential with logP, PSA, and MW penalties
    log_kp = 0.5 * lp_score - 0.02 * psa_pen - 0.005 * mw_pen

    kp_uu = math.exp(log_kp)

    # P-gp efflux: strong reduction
    if pgp_substrate:
        kp_uu *= 0.2

    return max(0.01, min(5.0, kp_uu))


def _exposure_category(kp_uu: float) -> str:
    """Classify CNS exposure level from Kp,uu."""
    if kp_uu >= 0.5:
        return "high"
    elif kp_uu >= 0.2:
        return "moderate"
    elif kp_uu >= 0.05:
        return "low"
    else:
        return "minimal"


def predict_bbb_penetration(
    drug_name: str,
    logP: float,
    mw: float,
    psa_A2: float,
    n_hbd: int,
    pgp_substrate: bool = False,
    fu_plasma: float = 0.1,
) -> BBBPenetrationResult:
    """Predict BBB penetration for a drug.

    Parameters
    ----------
    drug_name:
        Name identifier for the drug.
    logP:
        Lipophilicity.
    mw:
        Molecular weight in Da.
    psa_A2:
        Polar surface area in Angstrom squared.
    n_hbd:
        Number of hydrogen bond donors.
    pgp_substrate:
        Whether the drug is a P-gp substrate.
    fu_plasma:
        Fraction unbound in plasma.

    Returns
    -------
    BBBPenetrationResult
        BBB penetration prediction result.
    """
    if mw <= 0:
        raise ValueError("mw must be positive.")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be in (0, 1].")
    if n_hbd < 0:
        raise ValueError("n_hbd must be non-negative.")

    kp_uu = calculate_kp_uu_brain(logP, mw, psa_A2, pgp_substrate, fu_plasma)
    exposure = _exposure_category(kp_uu)

    # BBB score (0-100): composite of logP, PSA, MW, HBD, P-gp, Kp,uu
    lp_s = _logP_score(logP) * 25.0  # 0-25
    psa_s = _psa_score(psa_A2) * 25.0  # 0-25
    mw_s = max(0.0, 25.0 - max(0.0, mw - 400.0) * 0.05)  # 0-25
    hbd_s = max(0.0, 15.0 - n_hbd * 3.0)  # 0-15 (up to 5 HBD)
    pgp_pen = 10.0 if pgp_substrate else 0.0

    bbb_score = lp_s + psa_s + mw_s + hbd_s - pgp_pen
    bbb_score = max(0.0, min(100.0, bbb_score))

    # Notes
    notes_parts = []
    if 1.0 <= logP <= 4.0:
        notes_parts.append("logP in optimal CNS range (1-4).")
    elif logP > 5.0:
        notes_parts.append("High logP may cause non-specific binding.")
    elif logP < 0.0:
        notes_parts.append("Low logP reduces passive membrane permeability.")
    if psa_A2 > 90:
        notes_parts.append("High PSA (>90 A2): poor passive BBB penetration expected.")
    if mw > 450:
        notes_parts.append("MW >450 Da: reduced BBB penetration.")
    if pgp_substrate:
        notes_parts.append("P-gp substrate: significant efflux at BBB expected.")
    if not notes_parts:
        notes_parts.append("Physicochemical properties favor CNS penetration.")
    notes = " ".join(notes_parts)

    return BBBPenetrationResult(
        drug_name=drug_name,
        logP=logP,
        psa_A2=psa_A2,
        bbb_score=bbb_score,
        kp_uu_brain=kp_uu,
        predicted_cns_exposure=exposure,
        pgp_efflux_concern=pgp_substrate,
        notes=notes,
    )


def predict_cns_target_engagement(
    kp_uu: float,
    fu_plasma: float,
    plasma_conc_mg_L: float,
    mw: float,
    ec50_nM: float,
) -> float:
    """Predict CNS receptor occupancy (% of maximal).

    Uses the Hill equation with n=1 (Langmuir binding):
        occupancy = C_free_brain / (EC50 + C_free_brain)

    Parameters
    ----------
    kp_uu:
        Unbound brain-to-plasma ratio.
    fu_plasma:
        Fraction unbound in plasma.
    plasma_conc_mg_L:
        Total plasma concentration in mg/L.
    mw:
        Molecular weight in Da.
    ec50_nM:
        EC50 at the CNS target in nM.

    Returns
    -------
    float
        Percent receptor occupancy (0-100).
    """
    if kp_uu <= 0:
        raise ValueError("kp_uu must be positive.")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be in (0, 1].")
    if plasma_conc_mg_L < 0:
        raise ValueError("plasma_conc_mg_L must be non-negative.")
    if mw <= 0:
        raise ValueError("mw must be positive.")
    if ec50_nM <= 0:
        raise ValueError("ec50_nM must be positive.")

    # Convert mg/L to nM
    plasma_conc_nM = (plasma_conc_mg_L / mw) * 1e6

    # Unbound plasma concentration
    cu_plasma_nM = plasma_conc_nM * fu_plasma

    # Unbound brain concentration
    cu_brain_nM = cu_plasma_nM * kp_uu

    # Receptor occupancy via Langmuir binding
    occupancy_pct = 100.0 * cu_brain_nM / (ec50_nM + cu_brain_nM)
    return min(100.0, max(0.0, occupancy_pct))


def _cns_mpo_score(
    logP: float, mw: float, psa_A2: float, n_hbd: int, pka: float = 8.0, logD: float | None = None
) -> float:
    """Calculate simplified CNS MPO score (max 6).

    Properties scored (each 0 or 1 with partial credit):
    1. logP <= 5
    2. logD (approximated) <= 3
    3. MW <= 360
    4. PSA <= 90
    5. HBD <= 3
    6. pKa <= 8

    Reference: Wager et al. ACS Chem Neurosci 2010.
    """
    score = 0.0

    # 1. logP
    if logP <= 3.0:
        score += 1.0
    elif logP <= 5.0:
        score += 1.0 - (logP - 3.0) / 2.0
    # else 0

    # 2. logD (use logP as proxy if not provided)
    ld = logD if logD is not None else logP
    if ld <= 2.0:
        score += 1.0
    elif ld <= 4.0:
        score += 1.0 - (ld - 2.0) / 2.0

    # 3. MW
    if mw <= 300.0:
        score += 1.0
    elif mw <= 450.0:
        score += 1.0 - (mw - 300.0) / 150.0

    # 4. PSA
    if psa_A2 <= 60.0:
        score += 1.0
    elif psa_A2 <= 90.0:
        score += 1.0 - (psa_A2 - 60.0) / 30.0

    # 5. HBD
    if n_hbd == 0:
        score += 1.0
    elif n_hbd <= 3:
        score += 1.0 - n_hbd / 3.0

    # 6. pKa
    if pka <= 8.0:
        score += 1.0
    elif pka <= 10.0:
        score += 1.0 - (pka - 8.0) / 2.0

    return round(min(6.0, max(0.0, score)), 3)


def cns_efficacy_likelihood(
    logP: float,
    mw: float,
    psa_A2: float,
    n_hbd: int,
    ec50_nM: float,
    expected_plasma_cmax_nM: float,
) -> CNSEfficacyResult:
    """Predict likelihood of CNS efficacy based on physicochemical properties.

    Parameters
    ----------
    logP:
        Lipophilicity.
    mw:
        Molecular weight in Da.
    psa_A2:
        Polar surface area in Angstrom squared.
    n_hbd:
        Number of hydrogen bond donors.
    ec50_nM:
        EC50 at the CNS target in nM.
    expected_plasma_cmax_nM:
        Expected plasma Cmax in nM.

    Returns
    -------
    CNSEfficacyResult
    """
    if mw <= 0:
        raise ValueError("mw must be positive.")
    if ec50_nM <= 0:
        raise ValueError("ec50_nM must be positive.")
    if expected_plasma_cmax_nM < 0:
        raise ValueError("expected_plasma_cmax_nM must be non-negative.")
    if n_hbd < 0:
        raise ValueError("n_hbd must be non-negative.")

    # Default fu_plasma (typical for moderate lipophilicity)
    fu_plasma = max(0.01, min(0.5, 0.1 + logP * 0.02))

    kp_uu = calculate_kp_uu_brain(logP, mw, psa_A2, pgp_substrate=False, fu_plasma=fu_plasma)

    # Unbound brain Cmax in nM
    cu_brain_nM = expected_plasma_cmax_nM * fu_plasma * kp_uu

    # Occupancy at Cmax
    occupancy = 100.0 * cu_brain_nM / (ec50_nM + cu_brain_nM)
    occupancy = min(100.0, max(0.0, occupancy))

    mpo = _cns_mpo_score(logP, mw, psa_A2, n_hbd)

    mw_concern = mw > 450
    psa_concern = psa_A2 > 90
    optimal_lp = 1.0 <= logP <= 4.0
    is_efficacious = occupancy > 50.0

    notes_parts = [f"CNS MPO score: {mpo:.2f}/6."]
    if is_efficacious:
        notes_parts.append(f"Predicted occupancy {occupancy:.1f}% > 50% — likely efficacious.")
    else:
        notes_parts.append(
            f"Predicted occupancy {occupancy:.1f}% < 50% — insufficient target engagement."
        )
    if mw_concern:
        notes_parts.append("MW >450 Da limits BBB penetration.")
    if psa_concern:
        notes_parts.append("PSA >90 A2 limits passive penetration.")
    if not optimal_lp:
        notes_parts.append(f"logP {logP:.1f} outside optimal CNS range (1-4).")

    return CNSEfficacyResult(
        predicted_occupancy_at_cmax_pct=round(occupancy, 2),
        is_efficacious=is_efficacious,
        mw_concern=mw_concern,
        psa_concern=psa_concern,
        optimal_logP=optimal_lp,
        cns_mpo_score=mpo,
        notes=" ".join(notes_parts),
    )
