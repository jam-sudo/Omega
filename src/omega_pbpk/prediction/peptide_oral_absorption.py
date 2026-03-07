"""Phase 853 — Peptide/Protein Oral Absorption model."""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_PEPTIDE_TYPES = {"small_peptide", "large_peptide", "protein"}


def _sigmoid(x: float) -> float:
    """Logistic sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _lipophilicity_factor(logp: float) -> float:
    """Sigmoid centred at logp=1.5; moderate logp is optimal."""
    # Score peaks near logp 1.5; use a bell-like approximation via product of sigmoids
    # Rising edge: sigmoid(logp - 0.0)
    # Falling edge: 1 - sigmoid(logp - 3.0)
    f_low = _sigmoid((logp - 0.0) * 2.0)  # increases from low logp
    f_high = 1.0 - _sigmoid((logp - 3.0) * 2.0)  # decreases at high logp
    # Combine and normalise so that maximum ≈ 1.0 (at logp ≈ 1.5)
    raw = f_low * f_high
    # At logp=1.5: f_low=sigmoid(3.0)≈0.953, f_high=1-sigmoid(-3.0)≈0.953 → raw≈0.908
    # Normalise to 1.0 at the peak
    peak = _sigmoid(3.0) * (1.0 - _sigmoid(-3.0))
    if peak <= 0.0:
        return raw
    return min(raw / peak, 1.0)


def _gi_stability_factor(gi_stability_pct: float, peptide_type: str) -> float:
    """Apply peptide_type correction to gi_stability_pct."""
    base = gi_stability_pct / 100.0
    if peptide_type == "large_peptide":
        base *= 0.7
    elif peptide_type == "protein":
        base *= 0.3
    # small_peptide: no correction
    return min(max(base, 0.0), 1.0)


def _absorption_class(f: float) -> str:
    if f < 0.01:
        return "negligible"
    if f < 0.1:
        return "poor"
    if f < 0.3:
        return "moderate"
    return "good"


@dataclass(frozen=True)
class PeptideOralAbsorptionResult:
    drug_name: str
    mw_da: float
    peptide_type: str
    gi_stability_pct: float
    fa_oral: float
    bioavailability_f: float
    absorbed_dose_mg: float
    dose_mg: float
    f_stable: float
    f_size: float
    f_lip: float
    f_pgp: float
    absorption_class: str
    notes: str


def predict_peptide_oral_absorption(
    drug_name: str,
    mw_da: float,
    n_hbond_donors: int,
    logp: float,
    dose_mg: float,
    gi_stability_pct: float = 50.0,
    p_gp_substrate: bool = False,
    peptide_type: str = "small_peptide",
) -> PeptideOralAbsorptionResult:
    """Predict oral absorption of a peptide/protein drug.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    mw_da : float
        Molecular weight in Daltons (must be > 0).
    n_hbond_donors : int
        Number of H-bond donors (informational only).
    logp : float
        Logarithm of the octanol-water partition coefficient.
    dose_mg : float
        Oral dose in mg (must be > 0).
    gi_stability_pct : float
        Percentage of drug surviving GI proteolysis (0-100).
    p_gp_substrate : bool
        Whether the compound is a P-gp substrate.
    peptide_type : str
        One of "small_peptide", "large_peptide", "protein".

    Returns
    -------
    PeptideOralAbsorptionResult
    """
    if mw_da <= 0:
        raise ValueError("mw_da must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not (0.0 <= gi_stability_pct <= 100.0):
        raise ValueError("gi_stability_pct must be in [0, 100]")
    if peptide_type not in _VALID_PEPTIDE_TYPES:
        raise ValueError(f"peptide_type must be one of {sorted(_VALID_PEPTIDE_TYPES)}")

    f_stable = _gi_stability_factor(gi_stability_pct, peptide_type)
    f_size = min(math.exp(-mw_da / 500.0), 1.0)
    f_lip = _lipophilicity_factor(logp)
    f_pgp = 0.3 if p_gp_substrate else 1.0

    fa = f_stable * f_size * f_lip * f_pgp
    fa = min(max(fa, 0.0), 1.0)

    # Simplified: assume FH = 1 for peptides
    bioavailability_f = fa
    absorbed_dose_mg = bioavailability_f * dose_mg

    abs_class = _absorption_class(bioavailability_f)

    notes_parts = []
    if mw_da > 5000:
        notes_parts.append("High MW severely limits membrane permeability.")
    if f_stable < 0.2:
        notes_parts.append("Low GI stability is a major absorption barrier.")
    if p_gp_substrate:
        notes_parts.append("P-gp efflux reduces effective absorption.")
    if not notes_parts:
        notes_parts.append("Standard peptide absorption model applied.")
    notes = " ".join(notes_parts)

    return PeptideOralAbsorptionResult(
        drug_name=drug_name,
        mw_da=mw_da,
        peptide_type=peptide_type,
        gi_stability_pct=gi_stability_pct,
        fa_oral=fa,
        bioavailability_f=bioavailability_f,
        absorbed_dose_mg=absorbed_dose_mg,
        dose_mg=dose_mg,
        f_stable=f_stable,
        f_size=f_size,
        f_lip=f_lip,
        f_pgp=f_pgp,
        absorption_class=abs_class,
        notes=notes,
    )


def screen_peptide_modifications(
    drug_name: str,
    mw_da: float,
    logp: float,
    dose_mg: float,
    modifications: list[dict],
) -> list[PeptideOralAbsorptionResult]:
    """Screen a list of peptide modifications for oral absorption.

    Each modification dict must contain:
      - "name" (str)
      - "gi_stability_pct" (float)
      - "p_gp_substrate" (bool)
      - "logp_delta" (float): added to the base logp

    Results are sorted by bioavailability_f descending.
    """
    results = []
    for mod in modifications:
        name = mod.get("name", drug_name)
        gi_stability = float(mod["gi_stability_pct"])
        p_gp = bool(mod["p_gp_substrate"])
        logp_adj = logp + float(mod.get("logp_delta", 0.0))

        result = predict_peptide_oral_absorption(
            drug_name=name,
            mw_da=mw_da,
            n_hbond_donors=0,  # not tracked per modification
            logp=logp_adj,
            dose_mg=dose_mg,
            gi_stability_pct=gi_stability,
            p_gp_substrate=p_gp,
            peptide_type="small_peptide",
        )
        results.append(result)

    results.sort(key=lambda r: r.bioavailability_f, reverse=True)
    return results
