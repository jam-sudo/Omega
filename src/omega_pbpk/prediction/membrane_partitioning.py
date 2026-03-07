"""Predict drug partitioning into lipid bilayer membranes.

Models membrane:water partition coefficients (Km/w) for neutral and ionized
species, pH-dependent effective partitioning, and passive permeability.

References:
- Osterberg & Norinder (2001) DMPC bilayer calibration
- Abraham model for PAMPA/Caco-2 permeability correlations
- Herbette/Avdeef model for ionized species correction
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_MEMBRANE_TYPES = {"DMPC", "DPPC", "mixed"}


@dataclass(frozen=True)
class MembranePartitionResult:
    compound_name: str
    logP: float
    mw: float
    psa: float
    n_hbd: int

    # Membrane:water partition coefficients
    log_km_w_neutral: float
    km_w_neutral: float
    log_km_w_ionized: float
    km_w_ionized: float

    # pH-dependent effective Km/w
    log_km_w_effective: float
    km_w_effective: float

    # Permeability predictions
    log_perm_cm_per_s: float
    perm_cm_per_s: float

    # Lipid accumulation
    lipid_accumulation_factor: float

    # Classification
    membrane_permeability_class: str
    cns_membrane_potential: bool

    notes: str


def predict_membrane_partitioning(
    compound_name: str,
    logP: float,
    mw: float,
    psa: float,
    n_hbd: int,
    pka_acid: float | None = None,
    pka_base: float | None = None,
    membrane_type: str = "DMPC",
) -> MembranePartitionResult:
    """Predict membrane partitioning and permeability for a compound.

    Parameters
    ----------
    compound_name:
        Identifier for the compound.
    logP:
        Octanol-water partition coefficient (log scale).
    mw:
        Molecular weight in Da.
    psa:
        Polar surface area in A^2 (>= 0).
    n_hbd:
        Number of hydrogen bond donors (>= 0).
    pka_acid:
        Acidic pKa (if compound is an acid or zwitterion).
    pka_base:
        Basic pKa (if compound is a base or zwitterion).
    membrane_type:
        Bilayer type: "DMPC", "DPPC", or "mixed".

    Returns
    -------
    MembranePartitionResult
    """
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if psa < 0:
        raise ValueError(f"psa must be >= 0, got {psa}")
    if n_hbd < 0:
        raise ValueError(f"n_hbd must be >= 0, got {n_hbd}")
    if membrane_type not in _VALID_MEMBRANE_TYPES:
        raise ValueError(
            f"membrane_type must be one of {_VALID_MEMBRANE_TYPES}, got {membrane_type!r}"
        )

    # Km/w neutral species (Osterberg 2001, DMPC calibration)
    log_km_w_neutral = 1.09 * logP - 0.01 * psa - 0.15 * n_hbd
    km_w_neutral = 10.0 ** log_km_w_neutral

    # Ionized species: 1000x less membrane-permeable
    log_km_w_ionized = log_km_w_neutral - 3.0
    km_w_ionized = 10.0 ** log_km_w_ionized

    # Ionized fraction at pH 7.4
    fi = 0.0
    if pka_acid is not None:
        fi = 1.0 / (1.0 + 10.0 ** (pka_acid - 7.4))
    elif pka_base is not None:
        fi = 1.0 / (1.0 + 10.0 ** (7.4 - pka_base))

    # Effective Km/w (linear weighted average, then back to log)
    km_w_linear = km_w_neutral * (1.0 - fi) + km_w_ionized * fi
    # Guard against zero/negative (shouldn't happen but be safe)
    if km_w_linear <= 0.0:
        km_w_linear = 1e-30
    log_km_w_effective = math.log10(km_w_linear)
    km_w_effective = km_w_linear

    # Permeability from Abraham model (PAMPA/Caco-2 correlations)
    log_perm_cm_per_s = (
        0.76 * log_km_w_effective - 0.56 * n_hbd - 0.0086 * mw - 4.0
    )
    perm_cm_per_s = 10.0 ** log_perm_cm_per_s

    # Lipid accumulation (30% membrane lipid content)
    lipid_accumulation_factor = km_w_effective * 0.3 / 100.0

    # Classification
    if perm_cm_per_s > 1e-5:
        membrane_permeability_class = "high"
    elif perm_cm_per_s > 1e-6:
        membrane_permeability_class = "moderate"
    else:
        membrane_permeability_class = "low"

    cns_membrane_potential = km_w_effective > 100.0

    notes_parts = []
    if membrane_type != "DMPC":
        notes_parts.append(f"Calibrated for DMPC; {membrane_type} may differ slightly.")
    if fi > 0.5:
        notes_parts.append(f"Predominantly ionized at pH 7.4 (fi={fi:.2f}); reduced partitioning.")
    elif fi > 0.0:
        notes_parts.append(f"Partially ionized at pH 7.4 (fi={fi:.2f}).")
    notes = " ".join(notes_parts) if notes_parts else "Neutral compound at physiological pH."

    return MembranePartitionResult(
        compound_name=compound_name,
        logP=logP,
        mw=mw,
        psa=psa,
        n_hbd=n_hbd,
        log_km_w_neutral=log_km_w_neutral,
        km_w_neutral=km_w_neutral,
        log_km_w_ionized=log_km_w_ionized,
        km_w_ionized=km_w_ionized,
        log_km_w_effective=log_km_w_effective,
        km_w_effective=km_w_effective,
        log_perm_cm_per_s=log_perm_cm_per_s,
        perm_cm_per_s=perm_cm_per_s,
        lipid_accumulation_factor=lipid_accumulation_factor,
        membrane_permeability_class=membrane_permeability_class,
        cns_membrane_potential=cns_membrane_potential,
        notes=notes,
    )


def screen_membrane_partitioning(
    compounds: list[dict],
) -> list[MembranePartitionResult]:
    """Screen a list of compounds for membrane partitioning.

    Each compound dict must contain:
        "name": str, "logP": float, "mw": float, "psa": float, "n_hbd": int

    Optional keys: "pka_acid", "pka_base", "membrane_type"

    Returns a list sorted by km_w_effective descending (highest membrane
    affinity first).
    """
    results = []
    for cpd in compounds:
        result = predict_membrane_partitioning(
            compound_name=cpd["name"],
            logP=float(cpd["logP"]),
            mw=float(cpd["mw"]),
            psa=float(cpd["psa"]),
            n_hbd=int(cpd["n_hbd"]),
            pka_acid=cpd.get("pka_acid"),
            pka_base=cpd.get("pka_base"),
            membrane_type=cpd.get("membrane_type", "DMPC"),
        )
        results.append(result)

    results.sort(key=lambda r: r.km_w_effective, reverse=True)
    return results
