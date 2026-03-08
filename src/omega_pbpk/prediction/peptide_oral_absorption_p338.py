"""Phase 338 — Peptide Oral Absorption Predictor.

Predicts oral bioavailability of peptide drugs based on sequence properties.
Uses Caco-2-based Peff prediction, proteolytic stability, and Lipinski-style
rule violations adapted for peptides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeptideAbsorptionResult:
    """Result of peptide oral absorption prediction.

    Frozen dataclass — no list fields (all primitive/str).
    """

    drug_name: str
    n_residues: int
    mw_Da: float
    peff_cm_per_s: float
    fa: float
    proteolytic_stability: float
    f_oral: float
    lipinski_violations: int
    bcs_class: str
    formulation_recommendation: str
    absorption_enhancer_needed: bool
    notes: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_BCS_CLASSES = {"I", "II", "III", "IV"}
_PEFF_REFERENCE = 0.02  # cm/s reference permeability for fa calculation


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------


def predict_peptide_oral_absorption(
    drug_name: str,
    n_residues: int,
    mw_Da: float,
    logP: float,
    psa_angstrom2: float,
    n_hbd: int,
    n_hba: int,
    has_cyclic_structure: bool,
    has_n_methylation: bool,
    charge_at_ph7: int,
) -> PeptideAbsorptionResult:
    """Predict oral absorption of a peptide drug.

    Parameters
    ----------
    drug_name:
        Name of the peptide drug.
    n_residues:
        Number of amino acid residues (1–50).
    mw_Da:
        Molecular weight in Daltons (> 0).
    logP:
        Octanol-water partition coefficient.
    psa_angstrom2:
        Polar surface area in Angstrom^2.
    n_hbd:
        Number of hydrogen bond donors.
    n_hba:
        Number of hydrogen bond acceptors.
    has_cyclic_structure:
        True if the peptide has a cyclic backbone.
    has_n_methylation:
        True if N-methylation is present on backbone amides.
    charge_at_ph7:
        Net molecular charge at pH 7 (integer).

    Returns
    -------
    PeptideAbsorptionResult
    """
    # --- Input validation ---
    if not isinstance(n_residues, int) or not (1 <= n_residues <= 50):
        raise ValueError(f"n_residues must be an integer in [1, 50], got {n_residues}")
    if mw_Da <= 0.0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if not isinstance(charge_at_ph7, int):
        raise ValueError(f"charge_at_ph7 must be an integer, got {type(charge_at_ph7).__name__}")

    # --- Peff prediction (Caco-2-based, log scale) ---
    log_peff = -0.4 - 0.012 * psa_angstrom2 - 0.1 * n_hbd - 0.05 * n_hba + 0.3 * logP
    if has_cyclic_structure:
        log_peff += 0.5
    if has_n_methylation:
        log_peff += 0.3

    peff = 10.0**log_peff  # cm/s

    # --- fa prediction ---
    fa_raw = 1.0 / (1.0 + (_PEFF_REFERENCE / peff) ** 1.5)
    fa = max(0.001, min(0.999, fa_raw))

    # --- Proteolytic stability ---
    stab_raw = math.exp(-0.15 * n_residues)
    if has_cyclic_structure:
        stab_raw *= 1.3
    if has_n_methylation:
        stab_raw *= 1.2
    proteolytic_stability = max(0.01, min(1.0, stab_raw))

    # --- F_oral ---
    f_oral_raw = fa * proteolytic_stability
    f_oral = max(0.0, min(1.0, f_oral_raw))

    # --- Lipinski violations (peptide-adapted) ---
    violations = 0
    if mw_Da > 500.0:
        violations += 1
    if logP > 5.0 or logP < -5.0:
        violations += 1
    if n_hbd > 5:
        violations += 1
    if n_hba > 10:
        violations += 1
    if psa_angstrom2 > 140.0:
        violations += 1

    # --- BCS class for peptides (based on fa) ---
    if fa > 0.9:
        bcs_class = "I"
    elif fa >= 0.5:
        bcs_class = "III"
    else:
        bcs_class = "IV"

    # --- Formulation recommendation ---
    formulation_rec = _recommend_formulation(f_oral, fa, proteolytic_stability)

    # --- Absorption enhancer needed ---
    absorption_enhancer_needed = f_oral < 0.1

    # --- Notes ---
    notes = (
        f"Peff={peff:.2e} cm/s, fa={fa:.3f}, proteolytic stability="
        f"{proteolytic_stability:.3f}, F_oral={f_oral:.3f}. "
        f"Lipinski violations: {violations}/5. BCS class: {bcs_class}."
    )

    return PeptideAbsorptionResult(
        drug_name=drug_name,
        n_residues=n_residues,
        mw_Da=mw_Da,
        peff_cm_per_s=round(peff, 8),
        fa=round(fa, 6),
        proteolytic_stability=round(proteolytic_stability, 6),
        f_oral=round(f_oral, 6),
        lipinski_violations=violations,
        bcs_class=bcs_class,
        formulation_recommendation=formulation_rec,
        absorption_enhancer_needed=absorption_enhancer_needed,
        notes=notes,
    )


def _recommend_formulation(
    f_oral: float,
    fa: float,
    proteolytic_stability: float,
) -> str:
    """Generate a formulation recommendation based on predicted oral bioavailability."""
    if f_oral >= 0.5:
        return "Conventional oral tablet or capsule; good oral bioavailability expected."
    elif f_oral >= 0.2:
        if proteolytic_stability < 0.3:
            return (
                "Enteric-coated formulation to protect against GI proteolysis; "
                "consider enzyme inhibitor co-formulation."
            )
        return (
            "Modified-release formulation to maximize absorption window; "
            "consider permeation enhancers."
        )
    elif f_oral >= 0.1:
        return (
            "Nanoparticle or self-emulsifying drug delivery system (SEDDS) to "
            "enhance permeability and protect against enzymatic degradation."
        )
    else:
        return (
            "Absorption enhancer required (e.g., bile salts, fatty acids, SNAC); "
            "consider alternative routes (SC, nasal, buccal) or prodrug strategies."
        )
