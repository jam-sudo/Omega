"""Drug-target binding affinity prediction using QSAR-based models.

Predicts drug-target Ki from physicochemical descriptors for common target
classes using linear QSAR models tuned per target class.

Target classes
--------------
- kinase
- gpcr
- ion_channel
- nuclear_receptor
- protease

Model
-----
pKi = intercept + w_logP*logP + w_psa*psa + w_mw*mw + w_hbd*n_hbd + charge_bonus
Ki  = 10^(-pKi) * 1e6  (converted to µM)

References
----------
- Gleeson MP et al., J Med Chem. 2011;54(14):4968-77 (physicochemical properties)
- Paolini GV et al., Nat Biotechnol. 2006;24(7):805-15 (target selectivity)
- Lipinski CA, Lombardo F, Dominy BW, Feeney PJ, Adv Drug Deliv Rev. 2001
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Target-class QSAR parameters
# ---------------------------------------------------------------------------

# Each entry: (intercept, w_logP, w_psa, w_mw, w_hbd)
_TARGET_PARAMS: dict[str, tuple[float, float, float, float, float]] = {
    "kinase": (6.5, 0.3, -0.01, -0.002, 0.1),
    "gpcr": (7.0, 0.4, -0.008, -0.001, 0.05),
    "ion_channel": (5.5, 0.5, -0.005, -0.0015, 0.0),
    "nuclear_receptor": (7.5, 0.5, -0.012, -0.003, 0.15),
    "protease": (6.0, 0.2, -0.015, -0.001, 0.2),
}

VALID_TARGETS = set(_TARGET_PARAMS.keys())

# Charge bonus for basic drugs (pKa_base present) per target
_CHARGE_BONUS: dict[str, float] = {
    "kinase": 0.1,
    "gpcr": 0.3,
    "ion_channel": 0.2,
    "nuclear_receptor": 0.05,
    "protease": 0.15,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrugTargetAffinityResult:
    """Predicted drug-target binding affinity.

    Attributes
    ----------
    compound_name : Compound identifier.
    target_class : Target class ('kinase', 'gpcr', etc.).
    logP : Log partition coefficient.
    mw : Molecular weight (Da).
    psa : Polar surface area (A^2).
    predicted_pKi : Predicted pKi (-log10(Ki/M)).
    predicted_Ki_uM : Predicted Ki in µM.
    predicted_IC50_uM : Approximate IC50 (≈ 2*Ki for competitive inhibition).
    selectivity_score : 0-10, higher = more likely selective for this target class.
    potency_class : 'very_potent' (<10 nM), 'potent' (10-100 nM),
                   'moderate' (100-1000 nM), 'weak' (>1 µM).
    drugability_score : 0-100, combining affinity and ADME flags.
    notes : Summary text.
    """

    compound_name: str
    target_class: str
    logP: float
    mw: float
    psa: float
    predicted_pKi: float
    predicted_Ki_uM: float
    predicted_IC50_uM: float
    selectivity_score: float
    potency_class: str
    drugability_score: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _potency_class(ki_um: float) -> str:
    """Classify potency from Ki (µM)."""
    if ki_um < 0.01:
        return "very_potent"
    if ki_um < 0.1:
        return "potent"
    if ki_um <= 1.0:
        return "moderate"
    return "weak"


def _selectivity_score(target_class: str, logP: float, mw: float, psa: float) -> float:
    """Compute 0-10 selectivity score based on known target preferences."""
    score = 5.0
    if target_class == "kinase":
        # Kinases favour MW > 300, moderate logP
        if mw > 300:
            score += 1.5
        if 1.0 <= logP <= 5.0:
            score += 1.5
        if psa < 120:
            score += 1.0
        if mw > 700:
            score -= 2.0
    elif target_class == "gpcr":
        # GPCRs favour logP 2-5, moderate MW
        if 2.0 <= logP <= 5.0:
            score += 2.0
        if 250 <= mw <= 550:
            score += 1.5
        if psa < 90:
            score += 0.5
    elif target_class == "ion_channel":
        # Ion channels favour lipophilic, low PSA
        if logP >= 2.0:
            score += 1.5
        if psa < 60:
            score += 2.0
        if mw < 500:
            score += 0.5
    elif target_class == "nuclear_receptor":
        # Nuclear receptors favour logP > 3, moderate PSA
        if logP > 3.0:
            score += 2.5
        if 50 < psa < 140:
            score += 1.0
        if 300 <= mw <= 600:
            score += 0.5
    elif target_class == "protease":
        # Proteases favour H-bond donors, moderate logP
        if 0.0 <= logP <= 4.0:
            score += 1.5
        if psa > 60:
            score += 1.5
        if mw < 700:
            score += 1.0
    return _clamp(score, 0.0, 10.0)


def _drugability_score(
    ki_um: float,
    logP: float,
    mw: float,
    psa: float,
    n_hbd: int,
    n_hba: int,
) -> float:
    """Compute drugability score 0-100.

    Combines affinity quality and ADME drug-likeness (Lipinski-inspired).
    """
    score = 50.0

    # Affinity contribution (up to +30)
    if ki_um < 0.01:
        score += 30.0
    elif ki_um < 0.1:
        score += 20.0
    elif ki_um < 1.0:
        score += 10.0
    else:
        score -= 10.0

    # Lipinski Rule of 5 ADME bonuses / penalties
    # MW: favoured 150-500
    if mw <= 500:
        score += 8.0
    elif mw <= 700:
        score += 0.0
    else:
        score -= 15.0  # heavy MW penalty

    # logP: favoured < 5
    if logP <= 5.0:
        score += 5.0
    else:
        score -= 5.0

    # PSA: favoured < 140
    if psa <= 140:
        score += 3.0
    else:
        score -= 3.0

    # HBD: favoured <= 5
    if n_hbd <= 5:
        score += 2.0
    else:
        score -= 2.0

    # HBA: favoured <= 10
    if n_hba <= 10:
        score += 2.0
    else:
        score -= 2.0

    return _clamp(score, 0.0, 100.0)


def _predict_pki(
    target_class: str,
    logP: float,
    mw: float,
    psa: float,
    n_hbd: int,
    charge_bonus: float,
) -> float:
    """Compute pKi from linear QSAR model."""
    intercept, w_logP, w_psa, w_mw, w_hbd = _TARGET_PARAMS[target_class]
    pki = intercept + w_logP * logP + w_psa * psa + w_mw * mw + w_hbd * n_hbd + charge_bonus
    # Clamp to physically meaningful range (pKi 3-12)
    return _clamp(pki, 3.0, 12.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_drug_target_affinity(
    compound_name: str,
    target_class: str,
    logP: float,
    mw: float,
    psa: float,
    n_hbd: int = 2,
    n_hba: int = 4,
    pka_base: float | None = None,
) -> DrugTargetAffinityResult:
    """Predict drug-target binding affinity (Ki) from molecular descriptors.

    Parameters
    ----------
    compound_name : Compound identifier.
    target_class : One of 'kinase', 'gpcr', 'ion_channel', 'nuclear_receptor', 'protease'.
    logP : Octanol-water partition coefficient.
    mw : Molecular weight (Da). Must be > 0.
    psa : Polar surface area (A^2). Must be >= 0.
    n_hbd : Number of H-bond donors. Must be >= 0.
    n_hba : Number of H-bond acceptors. Must be >= 0.
    pka_base : pKa of basic nitrogen (if present); activates charge bonus.

    Returns
    -------
    DrugTargetAffinityResult

    Raises
    ------
    ValueError
        If target_class is unrecognised or physicochemical constraints violated.
    """
    if target_class not in VALID_TARGETS:
        raise ValueError(
            f"target_class must be one of {sorted(VALID_TARGETS)}, got '{target_class}'"
        )
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if psa < 0:
        raise ValueError(f"psa must be >= 0, got {psa}")
    if n_hbd < 0:
        raise ValueError(f"n_hbd must be >= 0, got {n_hbd}")
    if n_hba < 0:
        raise ValueError(f"n_hba must be >= 0, got {n_hba}")

    # Charge bonus: basic drugs may benefit from ionic interactions
    charge_bonus = 0.0
    if pka_base is not None and pka_base > 6.0:
        # Ionised fraction at physiological pH 7.4
        ionised_frac = 1.0 / (1.0 + 10.0 ** (7.4 - pka_base))
        charge_bonus = _CHARGE_BONUS.get(target_class, 0.1) * ionised_frac

    pki = _predict_pki(target_class, logP, mw, psa, n_hbd, charge_bonus)

    # Ki in M, then convert to µM
    ki_m = 10.0 ** (-pki)
    ki_um = ki_m * 1e6
    ic50_um = 2.0 * ki_um  # competitive inhibition approximation

    sel_score = _selectivity_score(target_class, logP, mw, psa)
    pot_class = _potency_class(ki_um)
    drug_score = _drugability_score(ki_um, logP, mw, psa, n_hbd, n_hba)

    notes = (
        f"{compound_name} vs {target_class}: pKi={pki:.2f}, Ki={ki_um:.4f} µM, "
        f"IC50≈{ic50_um:.4f} µM, potency='{pot_class}', "
        f"selectivity={sel_score:.1f}/10, drugability={drug_score:.0f}/100."
    )

    return DrugTargetAffinityResult(
        compound_name=compound_name,
        target_class=target_class,
        logP=logP,
        mw=mw,
        psa=psa,
        predicted_pKi=pki,
        predicted_Ki_uM=ki_um,
        predicted_IC50_uM=ic50_um,
        selectivity_score=sel_score,
        potency_class=pot_class,
        drugability_score=drug_score,
        notes=notes,
    )


def screen_against_targets(
    compound_name: str,
    logP: float,
    mw: float,
    psa: float,
    n_hbd: int = 2,
    n_hba: int = 4,
    targets: list[str] | None = None,
) -> list[DrugTargetAffinityResult]:
    """Screen a compound against all (or specified) target classes.

    Parameters
    ----------
    compound_name : Compound identifier.
    logP : Octanol-water partition coefficient.
    mw : Molecular weight (Da).
    psa : Polar surface area (A^2).
    n_hbd : Number of H-bond donors.
    n_hba : Number of H-bond acceptors.
    targets : List of target classes to screen. Defaults to all 5 classes.

    Returns
    -------
    list[DrugTargetAffinityResult]
        Sorted by predicted_Ki_uM ascending (most potent first).

    Raises
    ------
    ValueError
        If any target_class is unrecognised, or physicochemical constraints violated.
    """
    if targets is None:
        targets = list(VALID_TARGETS)

    results = []
    for tc in targets:
        result = predict_drug_target_affinity(
            compound_name=compound_name,
            target_class=tc,
            logP=logP,
            mw=mw,
            psa=psa,
            n_hbd=n_hbd,
            n_hba=n_hba,
        )
        results.append(result)

    results.sort(key=lambda r: r.predicted_Ki_uM)
    return results
