"""Blood-brain barrier (BBB) distribution predictor — Phase 186.

Predicts Kp,uu,brain (unbound brain/plasma ratio) and logBB using
physicochemical descriptors and P-gp efflux considerations.

Reference models
----------------
logBB = 0.2*logP - 0.012*PSA - 0.3*HBD + intercept  (Young et al. 1988,
        updated with Kelder 1999 PSA correction)
Kpu,brain is derived from fu_plasma and a P-gp efflux correction.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "BBBRatioResult",
    "predict_bbb_ratio",
    "screen_cns_penetration",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INTERCEPT = 0.15  # logBB intercept (empirical)
_PGP_EFFLUX_FACTOR_SUB = 8.0  # fold-reduction in Kpu for P-gp substrates
_PGP_EFFLUX_FACTOR_NON = 1.0
_HIGH_PSA_THRESHOLD = 90.0  # Å² — strong CNS barrier
_HIGH_MW_THRESHOLD = 450.0  # Da  — reduced CNS penetration
_MW_PENALTY = 0.0003  # logBB reduction per Da above threshold
_PSA_PENALTY_EXTRA = 0.004  # additional PSA penalty above 90 Å²


# ---------------------------------------------------------------------------
# Result dataclass (frozen — no list fields)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BBBRatioResult:
    """Result of BBB ratio prediction for a single compound."""

    drug_name: str
    logP: float
    mw_Da: float
    psa_A2: float
    hbd: int
    fu_plasma: float
    pgp_substrate: bool
    logBB: float
    kp_uu_brain: float
    cns_penetration_class: str  # "high", "moderate", or "low"
    pgp_efflux_factor: float
    bbb_score: float  # 0–100
    notes: str


# ---------------------------------------------------------------------------
# Core predictor
# ---------------------------------------------------------------------------


def predict_bbb_ratio(
    drug_name: str,
    logP: float,
    mw_Da: float,
    psa_A2: float,
    hbd: int,
    fu_plasma: float,
    pgp_substrate: bool = False,
    *,
    notes_override: str | None = None,
) -> BBBRatioResult:
    """Predict BBB distribution parameters for a drug.

    Parameters
    ----------
    drug_name:
        Identifier for the compound.
    logP:
        Octanol-water partition coefficient (log units).
    mw_Da:
        Molecular weight (Da).  Must be > 0.
    psa_A2:
        Polar surface area (Å²).  Must be ≥ 0.
    hbd:
        Number of hydrogen-bond donors.
    fu_plasma:
        Fraction unbound in plasma (0 < fu_plasma ≤ 1).
    pgp_substrate:
        Whether the compound is a P-glycoprotein substrate.
    notes_override:
        Optional note string.  If None, notes are auto-generated.

    Returns
    -------
    BBBRatioResult
    """
    # --- validation ---
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if psa_A2 < 0:
        raise ValueError("psa_A2 must be >= 0")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be in (0, 1]")

    # --- logBB calculation ---
    logBB = _INTERCEPT + 0.2 * logP - 0.012 * psa_A2 - 0.3 * hbd

    # High PSA penalty (above 90 Å²)
    if psa_A2 > _HIGH_PSA_THRESHOLD:
        logBB -= _PSA_PENALTY_EXTRA * (psa_A2 - _HIGH_PSA_THRESHOLD)

    # High MW penalty (above 450 Da)
    if mw_Da > _HIGH_MW_THRESHOLD:
        logBB -= _MW_PENALTY * (mw_Da - _HIGH_MW_THRESHOLD)

    # --- CNS penetration class ---
    if logBB > 0.0:
        cns_class = "high"
    elif logBB > -1.0:
        cns_class = "moderate"
    else:
        cns_class = "low"

    # --- Kpu,brain (unbound) ---
    pgp_factor = _PGP_EFFLUX_FACTOR_SUB if pgp_substrate else _PGP_EFFLUX_FACTOR_NON
    # BB (total) from logBB; Kpu = BB * (fu_brain / fu_plasma)
    # Simplified: fu_brain ≈ 1 / (1 + 0.5 * logP) clipped to (0.01, 1)
    fu_brain_est = 1.0 / (1.0 + max(0.0, 0.5 * logP))
    fu_brain_est = max(0.01, min(1.0, fu_brain_est))

    BB = 10.0**logBB
    kp_uu_brain = BB * (fu_brain_est / fu_plasma) / pgp_factor
    kp_uu_brain = max(0.0, kp_uu_brain)

    # --- BBB score (0–100, higher = better CNS penetration) ---
    # Map logBB from [-2, 1] → [0, 100]
    score_raw = (logBB + 2.0) / 3.0 * 100.0
    # Penalise P-gp substrates
    if pgp_substrate:
        score_raw -= 20.0
    bbb_score = max(0.0, min(100.0, score_raw))

    # --- notes ---
    if notes_override is not None:
        notes = notes_override
    else:
        note_parts = []
        if psa_A2 > _HIGH_PSA_THRESHOLD:
            note_parts.append("High PSA limits CNS penetration")
        if mw_Da > _HIGH_MW_THRESHOLD:
            note_parts.append("High MW reduces CNS penetration")
        if pgp_substrate:
            note_parts.append(f"P-gp efflux reduces Kpu,brain {pgp_factor:.0f}x")
        if cns_class == "high":
            note_parts.append("Good CNS penetration predicted")
        notes = "; ".join(note_parts) if note_parts else "Standard BBB prediction"

    return BBBRatioResult(
        drug_name=drug_name,
        logP=logP,
        mw_Da=mw_Da,
        psa_A2=psa_A2,
        hbd=hbd,
        fu_plasma=fu_plasma,
        pgp_substrate=pgp_substrate,
        logBB=logBB,
        kp_uu_brain=kp_uu_brain,
        cns_penetration_class=cns_class,
        pgp_efflux_factor=pgp_factor,
        bbb_score=bbb_score,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Screening helper
# ---------------------------------------------------------------------------


def screen_cns_penetration(drugs: list) -> list:
    """Screen a list of drugs for CNS penetration potential.

    Parameters
    ----------
    drugs:
        List of dicts, each containing keyword arguments for
        :func:`predict_bbb_ratio` (drug_name, logP, mw_Da, psa_A2, hbd,
        fu_plasma; optionally pgp_substrate).

    Returns
    -------
    list[BBBRatioResult] sorted by logBB descending.
    """
    results = []
    for d in drugs:
        res = predict_bbb_ratio(**d)
        results.append(res)
    results.sort(key=lambda r: r.logBB, reverse=True)
    return results
