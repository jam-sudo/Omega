"""
Phase 960 — Drug Binding to Multiple Plasma Proteins

Predicts binding to HSA, AGP, and lipoproteins.
Pure Python only (stdlib math, dataclasses). Frozen dataclass for result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "ProteinBindingSitesResult",
    "predict_protein_binding_sites",
    "screen_protein_binding",
]

# ---------------------------------------------------------------------------
# Result dataclass (frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProteinBindingSitesResult:
    drug_name: str
    logp: float
    pka: float
    ionization_type: str
    fraction_bound_hsa: float
    fraction_bound_agp: float
    fraction_bound_lipo: float
    total_fraction_bound: float
    fu_plasma: float
    dominant_binding_protein: str
    binding_category: str
    displacement_risk: str
    notes: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _scatchard_fraction(ka: float, conc_uM: float, n_sites: float = 1.0) -> float:
    """Scatchard single-site binding fraction.

    f_bound = n * Ka * [P] / (1 + Ka * [P])
    where Ka is in M^-1 and conc_uM is in µM (converted to M).
    """
    conc_M = conc_uM * 1e-6
    f = n_sites * ka * conc_M / (1.0 + ka * conc_M)
    return _clamp(f, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------


def predict_protein_binding_sites(
    drug_name: str,
    logp: float,
    pka: float,
    ionization_type: str,
    drug_conc_uM: float = 1.0,  # not used in population-level model, reserved
) -> ProteinBindingSitesResult:
    """Predict multi-protein plasma binding from physicochemical properties."""

    if ionization_type not in {"acid", "base", "neutral"}:
        raise ValueError("ionization_type must be 'acid', 'base', or 'neutral'")

    # --- HSA binding affinity ---
    if ionization_type == "acid":
        ka_hsa = (10 ** (0.8 * logp)) * 1000.0
    elif ionization_type == "base":
        ka_hsa = (10 ** (0.3 * logp)) * 100.0
    else:  # neutral
        ka_hsa = (10 ** (0.5 * logp)) * 200.0
    ka_hsa = _clamp(ka_hsa, 100.0, 1e8)

    # --- AGP binding affinity ---
    # AGP is the primary binding protein for basic drugs.
    # Use higher intrinsic affinity for bases to reflect their strong AGP interaction.
    if ionization_type == "base":
        ka_agp = (10 ** (0.6 * logp)) * 5e5  # strong base-AGP affinity
    else:
        ka_agp = (10 ** (0.2 * logp)) * 1000.0
    ka_agp = _clamp(ka_agp, 100.0, 1e8)

    # --- Lipoprotein binding affinity ---
    if logp > 3.0:
        ka_lipo = (10 ** (0.7 * logp)) * 100.0
    else:
        ka_lipo = 10.0
    ka_lipo = _clamp(ka_lipo, 0.0, 1e8)

    # --- Protein concentrations (µM) ---
    conc_hsa_uM = 600.0  # µM
    conc_agp_uM = 15.0  # µM
    conc_lipo_uM = 10.0  # µM

    # --- Fraction bound to each protein ---
    f_hsa = _scatchard_fraction(ka_hsa, conc_hsa_uM)
    f_agp = _scatchard_fraction(ka_agp, conc_agp_uM)
    f_lipo = _scatchard_fraction(ka_lipo, conc_lipo_uM)

    # Normalize total bound (cap at 1)
    total_bound = min(1.0, f_hsa + f_agp + f_lipo)
    fu = 1.0 - total_bound

    # --- Dominant binding protein ---
    fracs = {"HSA": f_hsa, "AGP": f_agp, "lipoprotein": f_lipo}
    dominant = max(fracs, key=lambda k: fracs[k])

    # --- Binding category ---
    if fu < 0.1:
        binding_category = "high"
    elif fu < 0.5:
        binding_category = "moderate"
    else:
        binding_category = "low"

    # --- Displacement risk ---
    displacement_risk = "high" if f_hsa > 0.70 else "low"

    notes = (
        f"HSA Ka={ka_hsa:.2e} M-1 | AGP Ka={ka_agp:.2e} M-1 | "
        f"Lipo Ka={ka_lipo:.2e} M-1 | "
        f"f_hsa={f_hsa:.3f} f_agp={f_agp:.3f} f_lipo={f_lipo:.3f} | "
        f"fu={fu:.3f} ({binding_category} binding)"
    )

    return ProteinBindingSitesResult(
        drug_name=drug_name,
        logp=logp,
        pka=pka,
        ionization_type=ionization_type,
        fraction_bound_hsa=f_hsa,
        fraction_bound_agp=f_agp,
        fraction_bound_lipo=f_lipo,
        total_fraction_bound=total_bound,
        fu_plasma=fu,
        dominant_binding_protein=dominant,
        binding_category=binding_category,
        displacement_risk=displacement_risk,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Screening function
# ---------------------------------------------------------------------------


def screen_protein_binding(
    compounds: list[dict[str, Any]],
) -> list[ProteinBindingSitesResult]:
    """Screen a list of compounds sorted by fu_plasma ascending (most bound first).

    Each dict should have keys: drug_name, logp, pka, ionization_type.
    Optional key: drug_conc_uM.
    """
    results = []
    for cpd in compounds:
        r = predict_protein_binding_sites(
            drug_name=cpd["drug_name"],
            logp=cpd["logp"],
            pka=cpd["pka"],
            ionization_type=cpd["ionization_type"],
            drug_conc_uM=cpd.get("drug_conc_uM", 1.0),
        )
        results.append(r)
    results.sort(key=lambda r: r.fu_plasma)
    return results
