"""Phase 1024 — Protein Binding Kinetics (prediction module).

Models drug binding to plasma proteins (albumin, AGP) using equilibrium
quadratic binding equations derived from single-site 1:1 binding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["ProteinBindingResult", "predict_protein_binding", "screen_protein_binding"]

_VALID_PROTEINS = {"albumin", "AGP"}
_VALID_IONIZATION = {"acid", "base", "neutral"}

# Protein plasma concentrations
_PROTEIN_CONC_uM: dict[str, float] = {
    "albumin": 600.0,  # ~40 g/L, MW=66500
    "AGP": 20.0,  # ~1 g/L, MW=41000
}


@dataclass(frozen=True)
class ProteinBindingResult:
    drug_name: str
    protein: str
    total_drug_mg_L: float
    bound_drug_mg_L: float
    free_drug_mg_L: float
    fu: float  # unbound fraction at equilibrium
    kd_uM: float  # dissociation constant (µM)
    binding_capacity_mg_L: float  # Bmax * protein_conc
    percent_bound: float  # 0-100
    saturation_fraction: float  # fraction of protein binding sites occupied
    notes: str


def _predict_kd_uM(
    logp: float,
    ionization_type: str,
    protein: str,
) -> float:
    """Predict Kd (µM) from logP and ionization type."""
    base = 10.0 ** (0.5 - 0.3 * logp)
    kd = base
    if protein == "albumin":
        if ionization_type == "acid":
            kd *= 0.5
        elif ionization_type == "base":
            kd *= 2.0
    else:  # AGP
        if ionization_type == "base":
            kd *= 0.3
        elif ionization_type == "acid":
            kd *= 3.0
    return max(0.01, kd)


def predict_protein_binding(
    drug_name: str,
    total_drug_mg_L: float,
    mw_Da: float,
    logp: float,
    pka: float = 7.0,
    ionization_type: str = "neutral",
    protein: str = "albumin",
) -> ProteinBindingResult:
    """Predict equilibrium plasma protein binding using quadratic binding model.

    Single binding site (1:1) model solved analytically via quadratic equation.
    Kd is predicted from logP and ionization type.
    """
    # --- Validation ---
    if total_drug_mg_L <= 0:
        raise ValueError(f"total_drug_mg_L must be > 0, got {total_drug_mg_L}")
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if protein not in _VALID_PROTEINS:
        raise ValueError(f"protein must be one of {_VALID_PROTEINS}, got {protein!r}")
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION}, got {ionization_type!r}"
        )

    protein_conc_uM = _PROTEIN_CONC_uM[protein]
    kd_uM = _predict_kd_uM(logp, ionization_type, protein)

    # Binding capacity: protein_conc_uM * mw_Da / 1000  (mg/L)
    # µmol/L * Da = µmol/L * g/mol → g/L * (1000 mg/g) / (1e6 µmol/mol) * 1e6 = mg/L
    # = protein_conc_uM * mw_Da / 1000
    binding_capacity_mg_L = protein_conc_uM * mw_Da / 1000.0

    # Convert total drug to µM
    total_uM = total_drug_mg_L * 1000.0 / mw_Da

    # Quadratic solution for bound (µM):
    # B² - (total_uM + protein_conc_uM + kd_uM)*B + total_uM * protein_conc_uM = 0
    s = total_uM + protein_conc_uM + kd_uM
    discriminant = s * s - 4.0 * total_uM * protein_conc_uM
    discriminant = max(0.0, discriminant)
    bound_uM = (s - math.sqrt(discriminant)) / 2.0

    # Clamp to physically valid range
    bound_uM = max(0.0, min(bound_uM, min(total_uM, protein_conc_uM)))
    free_uM = total_uM - bound_uM

    fu = free_uM / total_uM if total_uM > 0 else 1.0
    fu = max(0.0, min(1.0, fu))

    bound_drug_mg_L = bound_uM * mw_Da / 1000.0
    free_drug_mg_L = total_drug_mg_L - bound_drug_mg_L
    percent_bound = (1.0 - fu) * 100.0
    saturation_fraction = min(1.0, bound_uM / protein_conc_uM) if protein_conc_uM > 0 else 0.0

    notes = (
        f"Protein: {protein}; kd={kd_uM:.3f} µM; "
        f"logP={logp:.1f}; ionization={ionization_type}; "
        f"fu={fu:.3f}; percent_bound={percent_bound:.1f}%"
    )

    return ProteinBindingResult(
        drug_name=drug_name,
        protein=protein,
        total_drug_mg_L=total_drug_mg_L,
        bound_drug_mg_L=bound_drug_mg_L,
        free_drug_mg_L=free_drug_mg_L,
        fu=fu,
        kd_uM=kd_uM,
        binding_capacity_mg_L=binding_capacity_mg_L,
        percent_bound=percent_bound,
        saturation_fraction=saturation_fraction,
        notes=notes,
    )


def screen_protein_binding(
    drug_name: str,
    total_drug_mg_L: float,
    mw_Da: float,
    logp: float,
    pka: float = 7.0,
    ionization_type: str = "neutral",
) -> list:
    """Screen drug binding against both albumin and AGP.

    Returns [albumin_result, AGP_result] sorted by fu ascending (most bound first).
    """
    results = []
    for protein in ("albumin", "AGP"):
        r = predict_protein_binding(
            drug_name=drug_name,
            total_drug_mg_L=total_drug_mg_L,
            mw_Da=mw_Da,
            logp=logp,
            pka=pka,
            ionization_type=ionization_type,
            protein=protein,
        )
        results.append(r)
    results.sort(key=lambda r: r.fu)
    return results
