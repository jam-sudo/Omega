"""Drug-protein interaction energy prediction from physicochemical properties.

Estimates drug-protein binding affinity (Kd), binding free energy (deltaG),
and fraction unbound using empirical correlations without 3D structure.

Proteins supported:
  - albumin (human serum albumin, HSA)
  - alpha1_agp (alpha-1-acid glycoprotein)
  - cyp3a4 (cytochrome P450 3A4 active site)
  - p_glycoprotein (P-gp efflux transporter)
  - plasma_generic (weighted albumin + AGP composite)

Reference: Colmenarejo (2003) Med Res Rev; Kratochwil (2002) Pharm Res.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_PROTEINS = frozenset(
    {"albumin", "alpha1_agp", "cyp3a4", "p_glycoprotein", "plasma_generic"}
)

_R = 8.314e-3  # kJ/(mol·K)
_T = 310.0     # K (body temperature)


@dataclass(frozen=True)
class ProteinInteractionResult:
    """Result of drug-protein interaction prediction."""

    compound_name: str
    protein_name: str  # "albumin", "alpha1_agp", "cyp3a4", "p_glycoprotein", "plasma_generic"

    # Predicted binding
    predicted_kd_uM: float               # Dissociation constant (µM)
    predicted_delta_g_kJ_per_mol: float  # Binding free energy
    predicted_fu: float                  # Fraction unbound

    # Interaction analysis
    hydrophobic_score: float       # Contribution from logP
    polar_score: float             # Contribution from PSA/HBD/HBA
    charge_score: float            # Based on pKa
    total_interaction_score: float

    # Selectivity
    likely_binding_mode: str    # "hydrophobic_pocket", "ionic", "mixed", "surface"
    displacement_risk: str      # "low", "moderate", "high"

    notes: str


def _kd_from_pkd(pkd: float) -> float:
    """Convert pKd to Kd in µM (10^(-pKd))."""
    return 10.0 ** (-pkd)


def _delta_g(kd_uM: float) -> float:
    """Compute deltaG = R*T*ln(Kd_M) in kJ/mol."""
    kd_M = kd_uM * 1e-6
    return _R * _T * math.log(kd_M)


def _fu_from_kd(kd_uM: float, protein_conc_uM: float) -> float:
    """Simplified fraction unbound using 1-site binding: fu = 1/(1 + [P]/Kd)."""
    fu = 1.0 / (1.0 + protein_conc_uM / kd_uM)
    return max(1e-6, min(1.0, fu))


def _binding_mode(logP: float, psa: float, charge_score: float) -> str:
    """Determine likely binding mode."""
    if logP > 3 and psa < 60:
        if charge_score > 0:
            return "mixed"
        return "hydrophobic_pocket"
    if charge_score > 0:
        return "ionic"
    return "surface"


def _displacement_risk(kd_uM: float) -> str:
    """Classify displacement risk based on Kd."""
    if kd_uM < 1.0:
        return "high"
    if kd_uM < 10.0:
        return "moderate"
    return "low"


def _predict_albumin(
    logP: float,
    mw: float,
    psa: float,
    pka_acid: float | None,
) -> tuple[float, float, float, float]:
    """Return (pkd, protein_conc_uM, fu, notes_fragment)."""
    pkd = 0.7 * logP - 0.01 * psa + 0.01 * mw / 100.0
    if pka_acid is not None and pka_acid < 5.0:
        pkd += 1.0  # acidic drugs bind more strongly at Site I
    kd = _kd_from_pkd(pkd)
    protein_conc_uM = 50.0  # ~50 µM albumin in plasma
    fu = _fu_from_kd(kd, protein_conc_uM)
    return pkd, kd, protein_conc_uM, fu


def _predict_alpha1_agp(
    logP: float,
    psa: float,
    pka_base: float | None,
) -> tuple[float, float, float, float]:
    """Return (pkd, kd, protein_conc_uM, fu)."""
    pkd = 0.5 * logP - 0.005 * psa
    if pka_base is not None and pka_base > 7.0:
        pkd += 1.5  # basic drugs bind strongly to AGP
    kd = _kd_from_pkd(pkd)
    protein_conc_uM = 10.0  # ~10 µM AGP in plasma
    fu = _fu_from_kd(kd, protein_conc_uM)
    return pkd, kd, protein_conc_uM, fu


def _predict_cyp3a4(
    logP: float,
    psa: float,
    mw: float,
) -> tuple[float, float]:
    """Return (pkd, kd) for CYP3A4 active site binding."""
    pkd = 0.4 * logP - 0.003 * psa - 0.001 * mw
    kd = _kd_from_pkd(pkd)
    return pkd, kd


def _predict_pgp(
    logP: float,
    n_hbd: float,
    n_hba: float,
) -> tuple[float, float]:
    """Return (pkd, kd) for P-glycoprotein binding."""
    if (n_hbd + n_hba) < 5:
        pkd = 0.3 * logP + 0.5
    else:
        pkd = 0.3 * logP
    kd = _kd_from_pkd(pkd)
    return pkd, kd


def predict_protein_binding(
    compound_name: str,
    protein_name: str,
    logP: float,
    mw: float,
    psa: float,
    n_hbd: int,
    n_hba: int,
    pka_acid: float | None = None,
    pka_base: float | None = None,
) -> ProteinInteractionResult:
    """Predict drug-protein binding from physicochemical properties.

    Parameters
    ----------
    compound_name : str
        Name of the compound.
    protein_name : str
        One of: 'albumin', 'alpha1_agp', 'cyp3a4', 'p_glycoprotein', 'plasma_generic'.
    logP : float
        Lipophilicity.
    mw : float
        Molecular weight (g/mol).
    psa : float
        Polar surface area (A^2).
    n_hbd : int
        Number of hydrogen bond donors.
    n_hba : int
        Number of hydrogen bond acceptors.
    pka_acid : float, optional
        pKa of the most acidic group.
    pka_base : float, optional
        pKa of the most basic group.

    Returns
    -------
    ProteinInteractionResult
    """
    if protein_name not in _VALID_PROTEINS:
        raise ValueError(
            f"protein_name must be one of {sorted(_VALID_PROTEINS)}, got {protein_name!r}"
        )
    if mw <= 0:
        raise ValueError(f"mw must be positive, got {mw}")
    if n_hbd < 0:
        raise ValueError(f"n_hbd must be >= 0, got {n_hbd}")
    if n_hba < 0:
        raise ValueError(f"n_hba must be >= 0, got {n_hba}")
    if psa < 0:
        raise ValueError(f"psa must be >= 0, got {psa}")

    # Interaction scores (protein-independent)
    hydrophobic_score = logP * 0.5
    polar_score = -(psa / 100.0 + n_hbd * 0.3)
    if (pka_acid is not None and pka_acid < 5.0) or (
        pka_base is not None and pka_base > 8.0
    ):
        charge_score = 1.0
    else:
        charge_score = 0.0
    total_interaction_score = hydrophobic_score + polar_score + charge_score

    # Protein-specific prediction
    notes_parts: list[str] = []
    predicted_fu = 1.0

    if protein_name == "albumin":
        pkd, kd, conc, fu = _predict_albumin(logP, mw, psa, pka_acid)
        predicted_kd_uM = kd
        predicted_fu = fu
        notes_parts.append(f"Albumin pKd={pkd:.2f}, conc={conc} µM.")

    elif protein_name == "alpha1_agp":
        pkd, kd, conc, fu = _predict_alpha1_agp(logP, psa, pka_base)
        predicted_kd_uM = kd
        predicted_fu = fu
        notes_parts.append(f"AGP pKd={pkd:.2f}, conc={conc} µM.")

    elif protein_name == "cyp3a4":
        pkd, kd = _predict_cyp3a4(logP, psa, mw)
        predicted_kd_uM = kd
        predicted_fu = 1.0  # not a PPB context
        notes_parts.append(f"CYP3A4 active-site pKd={pkd:.2f}. fu not applicable (metabolic binding).")

    elif protein_name == "p_glycoprotein":
        pkd, kd = _predict_pgp(logP, n_hbd, n_hba)
        predicted_kd_uM = kd
        predicted_fu = 1.0  # not a PPB context
        notes_parts.append(f"P-gp pKd={pkd:.2f}. Substrate likelihood based on HBD+HBA count.")

    else:  # plasma_generic: weighted average of albumin and AGP
        _, kd_alb, conc_alb, fu_alb = _predict_albumin(logP, mw, psa, pka_acid)
        _, kd_agp, conc_agp, fu_agp = _predict_alpha1_agp(logP, psa, pka_base)
        # Weighted by protein concentration contribution
        w_alb, w_agp = 0.7, 0.3
        predicted_kd_uM = w_alb * kd_alb + w_agp * kd_agp
        predicted_fu = w_alb * fu_alb + w_agp * fu_agp
        predicted_fu = max(1e-6, min(1.0, predicted_fu))
        notes_parts.append(
            f"Plasma generic: albumin Kd={kd_alb:.2f} µM, AGP Kd={kd_agp:.2f} µM (70/30 blend)."
        )

    # Ensure Kd is positive
    predicted_kd_uM = max(1e-6, predicted_kd_uM)

    predicted_delta_g = _delta_g(predicted_kd_uM)

    likely_binding_mode = _binding_mode(logP, psa, charge_score)
    displacement_risk = _displacement_risk(predicted_kd_uM)

    notes_parts.append(
        f"Binding mode: {likely_binding_mode}. "
        f"Displacement risk: {displacement_risk}."
    )
    notes = " ".join(notes_parts)

    return ProteinInteractionResult(
        compound_name=compound_name,
        protein_name=protein_name,
        predicted_kd_uM=predicted_kd_uM,
        predicted_delta_g_kJ_per_mol=predicted_delta_g,
        predicted_fu=predicted_fu,
        hydrophobic_score=hydrophobic_score,
        polar_score=polar_score,
        charge_score=charge_score,
        total_interaction_score=total_interaction_score,
        likely_binding_mode=likely_binding_mode,
        displacement_risk=displacement_risk,
        notes=notes,
    )


def screen_for_protein_interactions(
    compounds: list[dict],
    proteins: list[str] | None = None,
) -> dict[str, dict[str, ProteinInteractionResult]]:
    """Screen compounds against multiple proteins.

    Parameters
    ----------
    compounds : list of dict
        Each dict with keys: name, logP, mw, psa, n_hbd, n_hba.
        Optional: pka_acid, pka_base.
    proteins : list of str, optional
        Protein names to screen against. Defaults to all 5.

    Returns
    -------
    dict mapping compound_name -> {protein_name -> ProteinInteractionResult},
    ordered by total_interaction_score descending per compound.
    """
    if proteins is None:
        proteins = sorted(_VALID_PROTEINS)

    output: dict[str, dict[str, ProteinInteractionResult]] = {}
    for c in compounds:
        name = c["name"]
        protein_results: dict[str, ProteinInteractionResult] = {}
        for prot in proteins:
            result = predict_protein_binding(
                compound_name=name,
                protein_name=prot,
                logP=c["logP"],
                mw=c["mw"],
                psa=c["psa"],
                n_hbd=c["n_hbd"],
                n_hba=c["n_hba"],
                pka_acid=c.get("pka_acid"),
                pka_base=c.get("pka_base"),
            )
            protein_results[prot] = result
        # Sort by total_interaction_score descending
        protein_results = dict(
            sorted(
                protein_results.items(),
                key=lambda kv: kv[1].total_interaction_score,
                reverse=True,
            )
        )
        output[name] = protein_results

    return output
