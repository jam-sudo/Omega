"""
Phase 976 — Drug Membrane Partitioning Prediction

Predicts drug partitioning into biological membranes (phospholipid bilayer).
Based on Osterberg & Norinder (2001) logP-based model with ionization corrections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}

# Membrane fraction (volume fraction of membranes in a typical in vitro system)
_MEMBRANE_FRACTION = 0.01

# Microsomal membrane fraction coefficient
_MIC_COEFF = 0.005

# Maximum kp_mem cap
_KP_MEM_MAX = 10_000.0
_KP_MEM_MIN = 0.01

# MIC binding factor cap
_MIC_FACTOR_MAX = 10.0

# Lysosomal trapping factor cap
_TRAPPED_FACTOR_MAX = 1000.0


@dataclass(frozen=True)
class MembranePartitioningResult:
    """Predicted drug-membrane partitioning properties."""

    drug_name: str
    logp: float
    pka: float
    ionization_type: str
    kp_mem: float
    log_kp_mem: float
    f_unbound_membrane: float
    mic_binding_factor: float
    lysosomal_trapping_risk: bool
    accumulation_risk: str
    notes: str


def predict_membrane_partitioning(
    drug_name: str,
    logp: float,
    pka: float,
    ionization_type: str = "neutral",
    ph_lysosome: float = 5.0,
) -> MembranePartitioningResult:
    """
    Predict membrane partitioning properties for a drug.

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    logp : float
        Octanol-water partition coefficient (log scale). Must be -5 to 10.
    pka : float
        pKa of the drug. Must be 0 to 14.
    ionization_type : str
        One of "acid", "base", or "neutral".
    ph_lysosome : float
        Lysosomal pH (default 5.0).

    Returns
    -------
    MembranePartitioningResult
    """
    # --- Validation ---
    if pka < 0.0 or pka > 14.0:
        raise ValueError(f"pka must be between 0 and 14, got {pka}")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION_TYPES}, got '{ionization_type}'"
        )
    if logp < -5.0 or logp > 10.0:
        raise ValueError(f"logp must be between -5 and 10, got {logp}")

    # --- Kp_mem calculation ---
    # Neutral lipid partition (Osterberg & Norinder 2001)
    log_kp_mem_neutral = 0.73 * logp - 0.84

    if ionization_type == "base":
        # Lysosomal amine trapping correction
        delta = pka - ph_lysosome
        if delta > 0.0:
            trapped_factor = min(10.0**delta, _TRAPPED_FACTOR_MAX)
        else:
            trapped_factor = 1.0
        log_kp_mem = log_kp_mem_neutral + 0.3 * math.log10(trapped_factor)
    else:
        log_kp_mem = log_kp_mem_neutral

    # Convert to linear scale with bounds
    kp_mem_raw = 10.0**log_kp_mem
    kp_mem = min(max(kp_mem_raw, _KP_MEM_MIN), _KP_MEM_MAX)

    # Recalculate log after clamping for consistency
    log_kp_mem_final = math.log10(kp_mem)

    # --- Unbound fraction in membrane ---
    f_unbound_membrane = 1.0 / (1.0 + kp_mem * _MEMBRANE_FRACTION)

    # --- Microsomal binding factor ---
    mic_binding_factor = min(1.0 + kp_mem * _MIC_COEFF, _MIC_FACTOR_MAX)

    # --- Lysosomal trapping risk ---
    lysosomal_trapping_risk = ionization_type == "base" and logp > 1.0

    # --- Accumulation risk ---
    if kp_mem > 1000.0:
        accumulation_risk = "high"
    elif kp_mem > 100.0:
        accumulation_risk = "moderate"
    else:
        accumulation_risk = "low"

    # --- Notes ---
    notes_parts: list[str] = [
        f"log_kp_mem_neutral={log_kp_mem_neutral:.3f}",
        f"kp_mem={kp_mem:.2f}",
        f"f_unbound={f_unbound_membrane:.4f}",
    ]
    if ionization_type == "base" and lysosomal_trapping_risk:
        notes_parts.append(f"Lysosomal trapping predicted (pKa={pka}, pH_lys={ph_lysosome})")
    notes = "; ".join(notes_parts)

    return MembranePartitioningResult(
        drug_name=drug_name,
        logp=logp,
        pka=pka,
        ionization_type=ionization_type,
        kp_mem=kp_mem,
        log_kp_mem=log_kp_mem_final,
        f_unbound_membrane=f_unbound_membrane,
        mic_binding_factor=mic_binding_factor,
        lysosomal_trapping_risk=lysosomal_trapping_risk,
        accumulation_risk=accumulation_risk,
        notes=notes,
    )


def compare_membrane_partitioning(drugs: list) -> list:
    """
    Compare membrane partitioning for a list of drug parameter dicts.

    Parameters
    ----------
    drugs : list of dict
        Each dict must have keys: drug_name, logp, pka.
        Optional keys: ionization_type, ph_lysosome.

    Returns
    -------
    list of MembranePartitioningResult
        Sorted by kp_mem descending.
    """
    results: list[MembranePartitioningResult] = []
    for drug in drugs:
        result = predict_membrane_partitioning(
            drug_name=drug["drug_name"],
            logp=drug["logp"],
            pka=drug["pka"],
            ionization_type=drug.get("ionization_type", "neutral"),
            ph_lysosome=drug.get("ph_lysosome", 5.0),
        )
        results.append(result)

    results.sort(key=lambda r: r.kp_mem, reverse=True)
    return results
