"""
Phase 572 — Salt form selection and solubility enhancement prediction.
"""

from __future__ import annotations

# ------------------------------------------------------------------
# Counterion databases
# ------------------------------------------------------------------
_ACID_COUNTERIONS = {
    "sodium": 14.0,
    "potassium": 14.0,
    "calcium": 12.0,
    "meglumine": 9.5,
}

_BASE_COUNTERIONS = {
    "HCl": -7.0,
    "maleate": 1.9,
    "tartrate": 3.0,
    "mesylate": -1.5,
}


def salt_solubility_enhancement(
    pka_drug: float,
    pka_counterion: float,
    intrinsic_solubility_mg_L: float,
    ph: float = 7.4,
    drug_type: str = "acid",
) -> dict:
    """
    Predict salt form solubility using the pH-solubility product approach.

    For an *acid* drug:
        S_salt = S0 * (1 + 10^(pH - pKa_drug))
    For a *base* drug:
        S_salt = S0 * (1 + 10^(pKa_drug - pH))

    Parameters
    ----------
    pka_drug : float                Drug pKa.
    pka_counterion : float          Counterion pKa.
    intrinsic_solubility_mg_L : float  Intrinsic (unionised) solubility (mg/L).
    ph : float                      pH of interest (default 7.4).
    drug_type : str                 'acid' or 'base'.

    Returns
    -------
    dict with keys:
        ph, drug_type, s_salt_mg_L, s_intrinsic_mg_L,
        enhancement_fold, recommendation
    """
    if intrinsic_solubility_mg_L < 0:
        raise ValueError("intrinsic_solubility_mg_L must be >= 0")
    drug_type = drug_type.lower()
    if drug_type not in ("acid", "base"):
        raise ValueError("drug_type must be 'acid' or 'base'")

    if drug_type == "acid":
        s_salt = intrinsic_solubility_mg_L * (1.0 + 10 ** (ph - pka_drug))
    else:  # base
        s_salt = intrinsic_solubility_mg_L * (1.0 + 10 ** (pka_drug - ph))

    # Enhancement fold must be >= 1 (salt can't reduce solubility below intrinsic)
    enhancement_fold = s_salt / intrinsic_solubility_mg_L if intrinsic_solubility_mg_L > 0 else 1.0

    if enhancement_fold >= 10:
        recommendation = "Salt form strongly recommended — >10-fold solubility gain."
    elif enhancement_fold >= 2:
        recommendation = "Salt form recommended — significant solubility improvement."
    else:
        recommendation = "Salt form offers marginal benefit at this pH."

    return {
        "ph": ph,
        "drug_type": drug_type,
        "s_salt_mg_L": s_salt,
        "s_intrinsic_mg_L": intrinsic_solubility_mg_L,
        "enhancement_fold": enhancement_fold,
        "recommendation": recommendation,
    }


def select_optimal_salt(
    drug_name: str,
    pka: float,
    drug_type: str,
    intrinsic_solubility_mg_L: float,
    target_ph: float = 7.4,
) -> list[dict]:
    """
    Evaluate common counterions and return them sorted by predicted salt
    solubility (descending).

    Parameters
    ----------
    drug_name : str
    pka : float              Drug pKa.
    drug_type : str          'acid' or 'base'.
    intrinsic_solubility_mg_L : float
    target_ph : float        pH for evaluation (default 7.4).

    Returns
    -------
    list of dict, each with keys:
        drug_name, counterion, pka_counterion, s_salt_mg_L,
        enhancement_fold, stability
    """
    drug_type = drug_type.lower()
    if drug_type not in ("acid", "base"):
        raise ValueError("drug_type must be 'acid' or 'base'")

    counterions = _ACID_COUNTERIONS if drug_type == "acid" else _BASE_COUNTERIONS

    results = []
    for name, pka_ci in counterions.items():
        enh = salt_solubility_enhancement(
            pka_drug=pka,
            pka_counterion=pka_ci,
            intrinsic_solubility_mg_L=intrinsic_solubility_mg_L,
            ph=target_ph,
            drug_type=drug_type,
        )
        stab = salt_stability_flag(pka_drug=pka, pka_counterion=pka_ci, drug_type=drug_type)
        results.append(
            {
                "drug_name": drug_name,
                "counterion": name,
                "pka_counterion": pka_ci,
                "s_salt_mg_L": enh["s_salt_mg_L"],
                "enhancement_fold": enh["enhancement_fold"],
                "stability": stab,
            }
        )

    results.sort(key=lambda r: r["s_salt_mg_L"], reverse=True)
    return results


def common_ion_effect(
    salt_solubility_mg_L: float,
    counterion_conc_mM: float,
    ksp_estimate: float | None = None,
) -> dict:
    """
    Estimate effective solubility when excess counterion is present.

    S_effective = salt_solubility / (1 + counterion_conc_mM / 100)

    Parameters
    ----------
    salt_solubility_mg_L : float
    counterion_conc_mM : float   Excess counterion concentration (mM).
    ksp_estimate : float | None  Not used in simplified model (reserved).

    Returns
    -------
    dict with keys: s_effective_mg_L, reduction_factor, notes
    """
    if salt_solubility_mg_L < 0:
        raise ValueError("salt_solubility_mg_L must be >= 0")
    if counterion_conc_mM < 0:
        raise ValueError("counterion_conc_mM must be >= 0")

    reduction_factor = 1.0 + counterion_conc_mM / 100.0
    s_effective = salt_solubility_mg_L / reduction_factor

    notes = (
        f"Common-ion reduction factor: {reduction_factor:.3f}. "
        f"Effective solubility reduced from {salt_solubility_mg_L:.2f} to "
        f"{s_effective:.2f} mg/L at {counterion_conc_mM} mM counterion."
    )

    return {
        "s_effective_mg_L": s_effective,
        "reduction_factor": reduction_factor,
        "notes": notes,
    }


def salt_stability_flag(
    pka_drug: float,
    pka_counterion: float,
    drug_type: str,
) -> str:
    """
    Predict salt stability based on the pKa difference rule.

    delta_pKa = |pKa_counterion - pKa_drug|
    >= 3 → 'stable'
    >= 1 → 'moderate stability'
    else → 'unstable — salt may convert to free form'
    """
    drug_type = drug_type.lower()
    if drug_type not in ("acid", "base"):
        raise ValueError("drug_type must be 'acid' or 'base'")

    delta_pka = abs(pka_counterion - pka_drug)

    if delta_pka >= 3:
        return "stable"
    elif delta_pka >= 1:
        return "moderate stability"
    else:
        return "unstable — salt may convert to free form"
