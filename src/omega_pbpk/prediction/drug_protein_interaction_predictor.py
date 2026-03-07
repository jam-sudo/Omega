"""Phase 275 — Drug-protein interaction predictor.

Predicts binding constants (Kd) and bound fractions for five key plasma/
tissue proteins using a simplified physicochemical regression model.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DrugProteinInteractionResult",
    "predict_drug_protein_interaction",
    "screen_protein_binding",
]

# ---------------------------------------------------------------------------
# Target protein lookup table
# ---------------------------------------------------------------------------

_PROTEIN_TABLE: dict[str, dict] = {
    "albumin": {
        "typical_ku_nM": 10000.0,
        "binding_sites": 2,
        "logp_threshold": 2.0,
        "c_protein_nM": 45000.0,  # ~45 uM typical plasma concentration
    },
    "alpha1_acid_glycoprotein": {
        "typical_ku_nM": 5000.0,
        "binding_sites": 1,
        "logp_threshold": 1.5,
        "c_protein_nM": 10000.0,
    },
    "lipoprotein": {
        "typical_ku_nM": 50000.0,
        "binding_sites": 5,
        "logp_threshold": 3.0,
        "c_protein_nM": 10000.0,
    },
    "cytochrome_p450_3A4": {
        "typical_ku_nM": 1000.0,
        "binding_sites": 1,
        "logp_threshold": 0.5,
        "c_protein_nM": 10000.0,
    },
    "p_glycoprotein": {
        "typical_ku_nM": 2000.0,
        "binding_sites": 1,
        "logp_threshold": 1.0,
        "c_protein_nM": 10000.0,
    },
}

_VALID_PROTEINS = set(_PROTEIN_TABLE.keys())


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrugProteinInteractionResult:
    drug_name: str
    target_protein: str
    mw: float
    logp: float
    psa_A2: float
    predicted_kd_nM: float
    bound_fraction_at_1uM: float
    fu_protein: float
    interaction_strength: str
    selectivity_index: float
    binding_risk: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _predict_kd(
    logp: float,
    psa_A2: float,
    mw: float,
    typical_ku_nM: float,
    logp_threshold: float,
) -> float:
    """Compute predicted Kd (nM) using physicochemical corrections."""
    # LogP effect: higher logP → lower Kd (stronger binding)
    logp_factor = 1.0 + max(0.0, logp - logp_threshold) * 0.5
    kd = typical_ku_nM / logp_factor

    # PSA penalty: higher PSA → weaker binding (higher Kd)
    kd *= 1.0 + psa_A2 / 200.0

    # MW penalty: larger MW → weaker binding
    kd *= 1.0 + max(0.0, mw - 300.0) / 500.0

    return max(kd, 1e-6)


def _bound_fraction(kd_nM: float, c_protein_nM: float) -> float:
    """One-site binding: B = C_protein / (Kd + C_protein)."""
    return c_protein_nM / (kd_nM + c_protein_nM)


def _interaction_strength(kd_nM: float) -> str:
    if kd_nM < 100.0:
        return "strong"
    if kd_nM <= 10000.0:
        return "moderate"
    return "weak"


def _binding_risk(fu: float, target_protein: str) -> str:
    """Binding risk is defined for albumin; others default to 'low'."""
    if target_protein != "albumin":
        return "low"
    if fu < 0.01:
        return "high"
    if fu < 0.1:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_drug_protein_interaction(
    drug_name: str,
    mw: float,
    logp: float,
    psa_A2: float,
    target_protein: str,
) -> DrugProteinInteractionResult:
    """Predict drug-protein binding for a single target protein.

    Parameters
    ----------
    drug_name : str
    mw : float  Molecular weight (Da); must be > 0
    logp : float  Lipophilicity; must be in (-5, 10)
    psa_A2 : float  Polar surface area (Angstrom^2); must be >= 0
    target_protein : str  One of the five supported proteins

    Returns
    -------
    DrugProteinInteractionResult
    """
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if logp <= -5 or logp >= 10:
        raise ValueError("logp must be in (-5, 10)")
    if psa_A2 < 0:
        raise ValueError("psa_A2 must be >= 0")
    if target_protein not in _VALID_PROTEINS:
        raise ValueError(
            f"target_protein must be one of {sorted(_VALID_PROTEINS)}, got '{target_protein}'"
        )

    params = _PROTEIN_TABLE[target_protein]
    kd_nM = _predict_kd(
        logp=logp,
        psa_A2=psa_A2,
        mw=mw,
        typical_ku_nM=params["typical_ku_nM"],
        logp_threshold=params["logp_threshold"],
    )

    c_protein_nM = params["c_protein_nM"]
    bf = _bound_fraction(kd_nM, c_protein_nM)
    fu = 1.0 - bf

    strength = _interaction_strength(kd_nM)
    sel_idx = 1000.0 / kd_nM
    risk = _binding_risk(fu, target_protein)

    notes = (
        f"Kd={kd_nM:.1f} nM; C_protein={c_protein_nM:.0f} nM; "
        f"bound_fraction={bf:.4f}; fu={fu:.4f}; "
        f"sites={params['binding_sites']}; strength={strength}"
    )

    return DrugProteinInteractionResult(
        drug_name=drug_name,
        target_protein=target_protein,
        mw=mw,
        logp=logp,
        psa_A2=psa_A2,
        predicted_kd_nM=kd_nM,
        bound_fraction_at_1uM=bf,
        fu_protein=fu,
        interaction_strength=strength,
        selectivity_index=sel_idx,
        binding_risk=risk,
        notes=notes,
    )


def screen_protein_binding(
    drug_name: str,
    mw: float,
    logp: float,
    psa_A2: float,
) -> list[DrugProteinInteractionResult]:
    """Screen all five proteins and sort by bound_fraction descending (tightest first).

    Parameters
    ----------
    drug_name : str
    mw : float
    logp : float
    psa_A2 : float

    Returns
    -------
    list[DrugProteinInteractionResult] sorted by bound_fraction_at_1uM descending
    """
    results = []
    for protein in _VALID_PROTEINS:
        r = predict_drug_protein_interaction(
            drug_name=drug_name,
            mw=mw,
            logp=logp,
            psa_A2=psa_A2,
            target_protein=protein,
        )
        results.append(r)

    results.sort(key=lambda r: r.bound_fraction_at_1uM, reverse=True)
    return results
