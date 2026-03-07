"""Plasma protein binding affinity prediction using physicochemical QSAR.

Predicts Kd, Bmax, Hill coefficient, and fu for albumin, alpha1-AGP, and lipoprotein.
This module is distinct from plasma_protein_binding.py (Phase 105), which predicts
fraction unbound (fup). This module focuses on binding AFFINITY parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

# Protein concentrations in plasma (mg/L)
_PROTEIN_CONC_MG_L = {
    "albumin": 45000.0,
    "alpha1_agp": 1000.0,
    "lipoprotein": 8000.0,
}

# Bmax in µmol/L for each protein
_BMAX_UMOL_L = {
    "albumin": 600.0,
    "alpha1_agp": 20.0,
    "lipoprotein": 50.0,
}

_VALID_PROTEINS = {"albumin", "alpha1_agp", "lipoprotein"}


@dataclass(frozen=True)
class PlasmaProteinAffinityResult:
    compound_name: str
    protein: str
    predicted_kd_mg_L: float
    predicted_bmax_umol_L: float
    hill_coefficient: float
    fu_at_therapeutic_conc: float
    fu_at_high_conc: float
    saturable_binding: bool
    dominant_binding_site: str
    binding_class: str
    notes: str


def _compute_fu(kd_mg_L: float, protein_conc_mg_L: float) -> float:
    """Compute fraction unbound using simple competitive binding: fu = Kd / (Kd + protein_conc)."""
    return kd_mg_L / (kd_mg_L + protein_conc_mg_L)


def _binding_class(kd_mg_L: float) -> str:
    if kd_mg_L < 0.1:
        return "strong"
    if kd_mg_L < 1.0:
        return "moderate"
    if kd_mg_L <= 10.0:
        return "weak"
    return "negligible"


def _dominant_binding_site(protein: str, pka_acid: float | None, pka_base: float | None) -> str:
    """Determine dominant albumin binding site based on ionization."""
    if protein != "albumin":
        return "non_specific"
    if pka_acid is not None and pka_acid < 5:
        return "site_i"  # warfarin site: acidic drugs
    if pka_base is not None and pka_base > 7:
        return "site_ii"  # benzodiazepine site: basic/neutral drugs
    return "non_specific"


def _predict_kd_albumin(
    logP: float,
    mw: float,
    psa: float,
    pka_acid: float | None,
    pka_base: float | None,
) -> float:
    pKd = 5.0 + logP * 0.4 - psa * 0.008 - mw * 0.002
    if pka_acid is not None and pka_acid < 5:
        pKd += 0.8
    if pka_base is not None and pka_base > 8:
        pKd -= 0.3
    kd = 10 ** (-pKd + 3)
    return max(0.001, kd)


def _predict_kd_alpha1_agp(
    logP: float,
    psa: float,
    pka_base: float | None,
) -> float:
    pKd = 4.5 + logP * 0.3 - psa * 0.005
    if pka_base is not None and pka_base > 7:
        pKd += 1.0
    kd = 10 ** (-pKd + 3)
    return max(0.001, kd)


def _predict_kd_lipoprotein(logP: float, mw: float) -> float:
    pKd = 3.0 + logP * 0.5 - mw * 0.001
    kd = 10 ** (-pKd + 3)
    return max(0.001, kd)


def predict_plasma_protein_affinity(
    compound_name: str,
    logP: float,
    mw: float,
    psa: float,
    n_hbd: int = 2,
    n_hba: int = 4,
    pka_acid: float | None = None,
    pka_base: float | None = None,
    protein: str = "albumin",
    therapeutic_conc_mg_L: float = 1.0,
) -> PlasmaProteinAffinityResult:
    """Predict plasma protein binding affinity for a given protein."""
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if psa < 0:
        raise ValueError("psa must be >= 0")
    if protein not in _VALID_PROTEINS:
        raise ValueError(f"protein must be one of {sorted(_VALID_PROTEINS)}, got '{protein}'")
    if n_hbd < 0:
        raise ValueError("n_hbd must be >= 0")
    if n_hba < 0:
        raise ValueError("n_hba must be >= 0")

    # Predict Kd
    if protein == "albumin":
        kd = _predict_kd_albumin(logP, mw, psa, pka_acid, pka_base)
    elif protein == "alpha1_agp":
        kd = _predict_kd_alpha1_agp(logP, psa, pka_base)
    else:  # lipoprotein
        kd = _predict_kd_lipoprotein(logP, mw)

    bmax = _BMAX_UMOL_L[protein]
    protein_conc = _PROTEIN_CONC_MG_L[protein]

    # fu at therapeutic concentration and 10x therapeutic
    fu_therapeutic = _compute_fu(kd, protein_conc)
    fu_high = _compute_fu(kd, protein_conc)

    # Both use protein_conc as dominant term; saturation effect modeled
    # At higher drug concentration the protein sites become saturated → fu increases
    # Use a simple occupancy model: fu_high = (Kd + 10*C) / (Kd + protein_conc + 10*C)
    # but spec says: fu = Kd / (Kd + protein_conc), saturable when fu_high > 1.5 * fu_therapeutic
    # To model saturation effect: at high conc, bound fraction saturates
    # Use: bound_frac = Bmax * C / (Kd + C) style → fu = 1 - bound_frac / C
    # Simpler: model with effective protein reduction at high concentration
    # fu_high represents fu at 10× therapeutic — use protein_conc reduced by occupancy
    c_high_mg_L = 10.0 * therapeutic_conc_mg_L
    # Estimate occupancy at high concentration reduces effective free protein
    # f_occupied = C / (Kd + C) where C is in mg/L (rough estimate)
    occupancy = c_high_mg_L / (kd + c_high_mg_L)
    effective_protein_high = protein_conc * (1.0 - occupancy)
    if effective_protein_high < 0.0:
        effective_protein_high = 0.0
    fu_high = _compute_fu(kd, effective_protein_high) if effective_protein_high > 0 else 1.0

    # fu must be in (0, 1]
    fu_therapeutic = max(1e-6, min(1.0, fu_therapeutic))
    fu_high = max(1e-6, min(1.0, fu_high))

    saturable = fu_high > 1.5 * fu_therapeutic

    binding_site = _dominant_binding_site(protein, pka_acid, pka_base)
    bc = _binding_class(kd)

    notes = (
        f"{protein} binding: Kd={kd:.4f} mg/L ({bc}); "
        f"fu_therapeutic={fu_therapeutic:.4f}; "
        f"fu_high={fu_high:.4f}; "
        f"saturable={saturable}"
    )

    return PlasmaProteinAffinityResult(
        compound_name=compound_name,
        protein=protein,
        predicted_kd_mg_L=kd,
        predicted_bmax_umol_L=bmax,
        hill_coefficient=1.0,
        fu_at_therapeutic_conc=fu_therapeutic,
        fu_at_high_conc=fu_high,
        saturable_binding=saturable,
        dominant_binding_site=binding_site,
        binding_class=bc,
        notes=notes,
    )


def screen_protein_affinity(
    compound_name: str,
    logP: float,
    mw: float,
    psa: float,
    n_hbd: int = 2,
    n_hba: int = 4,
    proteins: list[str] | None = None,
) -> list[PlasmaProteinAffinityResult]:
    """Screen compound against all plasma proteins, sorted by Kd ascending (tightest binding first)."""
    if proteins is None:
        proteins = ["albumin", "alpha1_agp", "lipoprotein"]

    results = []
    for protein in proteins:
        result = predict_plasma_protein_affinity(
            compound_name=compound_name,
            logP=logP,
            mw=mw,
            psa=psa,
            n_hbd=n_hbd,
            n_hba=n_hba,
            protein=protein,
        )
        results.append(result)

    # Sort by Kd ascending (tightest binding = lowest Kd first)
    results.sort(key=lambda r: r.predicted_kd_mg_L)
    return results
