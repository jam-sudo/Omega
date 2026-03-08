"""Phase 585 - Drug Solubility in Simulated Intestinal Fluids.

Predict drug solubility in fasted (FaSSIF), fed (FeSSIF), and gastric (FaSSGF)
simulated intestinal fluids, accounting for bile salt micellization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Fluid parameters: pH, bile_salt_conc_mM, lecithin_conc_mM
FLUID_PARAMS: dict[str, dict[str, float]] = {
    "FaSSIF": {"pH": 6.5, "bile_salt_mM": 3.0, "lecithin_mM": 0.75},
    "FeSSIF": {"pH": 5.0, "bile_salt_mM": 15.0, "lecithin_mM": 3.75},
    "FaSSGF": {"pH": 1.6, "bile_salt_mM": 0.08, "lecithin_mM": 0.02},
}

VALID_FLUID_TYPES = frozenset(FLUID_PARAMS.keys())


@dataclass(frozen=True)
class SIFSolubilityResult:
    """Result of simulated intestinal fluid solubility prediction."""

    drug_name: str
    logP: float
    intrinsic_solubility_mg_mL: float
    fluid_type: str
    solubility_sif_mg_mL: float
    enhancement_ratio: float
    micelle_solubilization_factor: float
    ph_correction_factor: float
    absorption_potential: float
    biorelevant_class: str
    notes: str


def predict_sif_solubility(
    drug_name: str,
    logP: float,
    pKa: float,
    drug_type: str,
    intrinsic_solubility_mg_mL: float,
    fluid_type: str = "FaSSIF",
    peff_cm_per_s: float = 1e-4,
) -> SIFSolubilityResult:
    """Predict drug solubility in a simulated intestinal fluid.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    logP : float
        Partition coefficient (log scale).
    pKa : float
        Acid dissociation constant.
    drug_type : str
        One of "acid", "base", or "neutral".
    intrinsic_solubility_mg_mL : float
        Intrinsic aqueous solubility in mg/mL (must be > 0).
    fluid_type : str
        Simulated fluid type: "FaSSIF", "FeSSIF", or "FaSSGF".
    peff_cm_per_s : float
        Effective permeability in cm/s (must be > 0).

    Returns
    -------
    SIFSolubilityResult
    """
    if fluid_type not in VALID_FLUID_TYPES:
        raise ValueError(
            f"Invalid fluid_type '{fluid_type}'. Must be one of {sorted(VALID_FLUID_TYPES)}"
        )
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError("intrinsic_solubility_mg_mL must be > 0")
    if peff_cm_per_s <= 0:
        raise ValueError("peff_cm_per_s must be > 0")

    params = FLUID_PARAMS[fluid_type]
    fluid_ph = params["pH"]
    bile_salt_mM = params["bile_salt_mM"]

    # pH correction via Henderson-Hasselbalch
    drug_type_lower = drug_type.lower()
    if drug_type_lower == "acid":
        ionized_frac = 1.0 / (1.0 + 10.0 ** (pKa - fluid_ph))
    elif drug_type_lower == "base":
        ionized_frac = 1.0 / (1.0 + 10.0 ** (fluid_ph - pKa))
    else:
        ionized_frac = 0.0

    ph_correction_factor = 1.0 + ionized_frac * 2.0

    # Micelle solubilization (log-linear model)
    kmc_raw = 10.0 ** (0.6 * logP - 1.5)
    kmc = max(1.0, min(1000.0, kmc_raw))

    micelle_factor_raw = 1.0 + (kmc / 1000.0) * bile_salt_mM
    micelle_factor = max(1.0, min(100.0, micelle_factor_raw))

    # Solubility in SIF
    solubility_sif = intrinsic_solubility_mg_mL * ph_correction_factor * micelle_factor

    # Enhancement ratio
    enhancement_ratio = solubility_sif / intrinsic_solubility_mg_mL

    # Absorption potential
    absorption_potential = math.log10(max(1e-10, solubility_sif * peff_cm_per_s))

    # Biorelevant classification
    if solubility_sif < 0.1:
        biorelevant_class = "solubility_limited"
    elif peff_cm_per_s < 1e-5:
        biorelevant_class = "permeability_limited"
    else:
        biorelevant_class = "not_limited"

    notes = (
        f"Fluid={fluid_type}, pH={fluid_ph}, "
        f"bile_salt={bile_salt_mM}mM, "
        f"Kmc={kmc:.2f}, ionized_frac={ionized_frac:.4f}"
    )

    return SIFSolubilityResult(
        drug_name=drug_name,
        logP=logP,
        intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        fluid_type=fluid_type,
        solubility_sif_mg_mL=solubility_sif,
        enhancement_ratio=enhancement_ratio,
        micelle_solubilization_factor=micelle_factor,
        ph_correction_factor=ph_correction_factor,
        absorption_potential=absorption_potential,
        biorelevant_class=biorelevant_class,
        notes=notes,
    )


def compare_fluids(
    drug_name: str,
    logP: float,
    pKa: float,
    drug_type: str,
    intrinsic_solubility_mg_mL: float,
    peff_cm_per_s: float = 1e-4,
) -> list[SIFSolubilityResult]:
    """Compare drug solubility across all three simulated fluids.

    Returns list sorted by solubility_sif_mg_mL descending.
    """
    results = []
    for ft in FLUID_PARAMS:
        r = predict_sif_solubility(
            drug_name=drug_name,
            logP=logP,
            pKa=pKa,
            drug_type=drug_type,
            intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
            fluid_type=ft,
            peff_cm_per_s=peff_cm_per_s,
        )
        results.append(r)
    results.sort(key=lambda r: r.solubility_sif_mg_mL, reverse=True)
    return results
