"""Drug solubility enhancement strategies for poorly soluble drugs.

Models three pharmaceutical solubility enhancement approaches:
1. Cyclodextrin (HP-beta-CD) complexation
2. Co-solvent solubilization (Yalkowsky equation)
3. Surfactant micellar solubilization

References:
  Brewster ME & Loftsson T (2007) Cyclodextrins as pharmaceutical solubilizers.
    Adv Drug Deliv Rev 59:645-666.
  Yalkowsky SH & Roseman TJ (1981) Solubilization of drugs by cosolvents.
    Techniques of Solubilization of Drugs, Marcel Dekker.
  Rosen MJ (2004) Surfactants and Interfacial Phenomena. 3rd ed. Wiley.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclasses (frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolubilityComparisonResult:
    """Comparison of solubility enhancement strategies for a single drug."""

    drug_name: str
    solubility_water: float  # intrinsic aqueous solubility (mg/mL)
    logP: float
    strategies: dict[str, float]  # strategy label -> total solubility (mg/mL)
    max_enhancement_strategy: str  # label of strategy with highest solubility
    max_enhancement_fold: float  # fold-increase vs. water
    notes: str


# ---------------------------------------------------------------------------
# Core solubility functions
# ---------------------------------------------------------------------------


def cyclodextrin_solubility(
    c_free_mg_mL: float,
    k11_M_inv: float,
    cd_conc_mM: float,
    drug_mw: float,
    cd_mw: float = 1135.0,
) -> float:
    """Compute total drug solubility in the presence of HP-beta-cyclodextrin.

    Uses the 1:1 inclusion complex model (Higuchi-Connors AL-type phase diagram):
        S_total = S_free * (1 + K11 * [CD])

    where [CD] is the free cyclodextrin concentration in mol/L.

    Args:
        c_free_mg_mL: Intrinsic (free) drug solubility in mg/mL (must be > 0).
        k11_M_inv: 1:1 binding constant in M^-1 (1/mol/L).
        cd_conc_mM: Cyclodextrin concentration in mM (millimolar, >= 0).
        drug_mw: Drug molecular weight in g/mol (for unit consistency, not used
            in this simplified model but required for API completeness).
        cd_mw: Cyclodextrin molecular weight in g/mol (default 1135 for HP-beta-CD).

    Returns:
        Total drug solubility (free + complexed) in mg/mL.

    Raises:
        ValueError: If c_free_mg_mL <= 0 or cd_conc_mM < 0.
    """
    if c_free_mg_mL <= 0:
        raise ValueError(f"c_free_mg_mL must be > 0, got {c_free_mg_mL}")
    if cd_conc_mM < 0:
        raise ValueError(f"cd_conc_mM must be >= 0, got {cd_conc_mM}")
    if drug_mw <= 0:
        raise ValueError(f"drug_mw must be > 0, got {drug_mw}")
    if cd_mw <= 0:
        raise ValueError(f"cd_mw must be > 0, got {cd_mw}")

    # Convert mM to mol/L
    cd_conc_M = cd_conc_mM * 1e-3

    # Total solubility: AL-type phase diagram
    s_total = c_free_mg_mL * (1.0 + k11_M_inv * cd_conc_M)
    return s_total


def cosolvent_solubility(
    solubility_water_mg_mL: float,
    logP: float,
    cosolvent_fraction: float,
    sigma_cosolvent: float = 3.0,
) -> float:
    """Predict solubility in a co-solvent/water mixture using the Yalkowsky equation.

    Log-linear model:
        log10(S_mix) = log10(S_water) + sigma * f_cosolvent

    where sigma is the solubilization power of the co-solvent (characteristic
    slope, dimensionless, typically 2-4 for pharmaceutical co-solvents).

    Args:
        solubility_water_mg_mL: Aqueous solubility in mg/mL (must be > 0).
        logP: Drug octanol-water partition coefficient (used for context).
        cosolvent_fraction: Volume fraction of co-solvent in 0 to <1.
        sigma_cosolvent: Solubilization power of co-solvent (default 3.0 for
            ethanol; PEG400 ~ 2.5, propylene glycol ~ 2.0).

    Returns:
        Predicted solubility in mg/mL.

    Raises:
        ValueError: If inputs are out of valid range.
    """
    if solubility_water_mg_mL <= 0:
        raise ValueError(f"solubility_water_mg_mL must be > 0, got {solubility_water_mg_mL}")
    if not (0.0 <= cosolvent_fraction < 1.0):
        raise ValueError(f"cosolvent_fraction must be in [0, 1), got {cosolvent_fraction}")

    log_s_mix = math.log10(solubility_water_mg_mL) + sigma_cosolvent * cosolvent_fraction
    return 10.0**log_s_mix


def surfactant_solubilization(
    solubility_water_mg_mL: float,
    logP: float,
    surfactant_conc_mM: float,
    cmc_mM: float = 1.0,
    ksp: float = 100.0,
) -> float:
    """Predict drug solubility in surfactant solution via micellar solubilization.

    Above the critical micelle concentration (CMC), drug partitions into micelles:
        S_total = S_water + Ksp_eff * [micelles]

    where [micelles] = max(0, surf_conc - CMC) in mM
    and Ksp_eff = ksp * exp(0.5 * logP)  (logP-dependent partitioning into micelles)

    Args:
        solubility_water_mg_mL: Intrinsic aqueous solubility in mg/mL (must be > 0).
        logP: Drug octanol-water partition coefficient.
        surfactant_conc_mM: Total surfactant concentration in mM.
        cmc_mM: Critical micelle concentration in mM (must be > 0).
        ksp: Base micellar partitioning coefficient (mL/mM·mg/mL units, > 0).

    Returns:
        Total drug solubility in mg/mL.

    Raises:
        ValueError: If inputs violate constraints.
    """
    if solubility_water_mg_mL <= 0:
        raise ValueError(f"solubility_water_mg_mL must be > 0, got {solubility_water_mg_mL}")
    if cmc_mM <= 0:
        raise ValueError(f"cmc_mM must be > 0, got {cmc_mM}")
    if ksp <= 0:
        raise ValueError(f"ksp must be > 0, got {ksp}")
    if surfactant_conc_mM < 0:
        raise ValueError(f"surfactant_conc_mM must be >= 0, got {surfactant_conc_mM}")

    micelle_conc_mM = max(0.0, surfactant_conc_mM - cmc_mM)

    # logP-dependent effective Ksp
    ksp_eff = ksp * math.exp(0.5 * logP)

    s_total = solubility_water_mg_mL + ksp_eff * micelle_conc_mM
    return s_total


def compare_enhancement_strategies(
    drug_name: str,
    solubility_water_mg_mL: float,
    logP: float,
    drug_mw: float,
) -> SolubilityComparisonResult:
    """Compare multiple solubility enhancement strategies for a drug.

    Strategies evaluated:
      - water:           Neat water (baseline)
      - cd_10mM:         HP-beta-CD at 10 mM (K11 = 5000 M^-1)
      - cd_50mM:         HP-beta-CD at 50 mM (K11 = 5000 M^-1)
      - cosolvent_10pct: 10% ethanol (sigma=3.0)
      - cosolvent_30pct: 30% ethanol (sigma=3.0)
      - surfactant_5mM:  Polysorbate 80 at 5 mM (CMC=0.05 mM)
      - surfactant_20mM: Polysorbate 80 at 20 mM (CMC=0.05 mM)

    Args:
        drug_name: Drug identifier string.
        solubility_water_mg_mL: Intrinsic aqueous solubility in mg/mL (> 0).
        logP: Drug logP value.
        drug_mw: Drug molecular weight (g/mol, > 0).

    Returns:
        SolubilityComparisonResult with enhancement factors for each strategy.

    Raises:
        ValueError: If solubility_water_mg_mL <= 0 or drug_mw <= 0.
    """
    if solubility_water_mg_mL <= 0:
        raise ValueError(f"solubility_water_mg_mL must be > 0, got {solubility_water_mg_mL}")
    if drug_mw <= 0:
        raise ValueError(f"drug_mw must be > 0, got {drug_mw}")

    # Typical HP-beta-CD binding constant
    k11 = 5000.0  # M^-1 (midrange for lipophilic drugs)

    # Polysorbate 80 CMC ~ 0.05 mM; ksp reference value
    ps80_cmc = 0.05
    ps80_ksp = 50.0

    strategies: dict[str, float] = {
        "water": solubility_water_mg_mL,
        "cd_10mM": cyclodextrin_solubility(solubility_water_mg_mL, k11, 10.0, drug_mw),
        "cd_50mM": cyclodextrin_solubility(solubility_water_mg_mL, k11, 50.0, drug_mw),
        "cosolvent_10pct": cosolvent_solubility(solubility_water_mg_mL, logP, 0.10),
        "cosolvent_30pct": cosolvent_solubility(solubility_water_mg_mL, logP, 0.30),
        "surfactant_5mM": surfactant_solubilization(
            solubility_water_mg_mL, logP, 5.0, cmc_mM=ps80_cmc, ksp=ps80_ksp
        ),
        "surfactant_20mM": surfactant_solubilization(
            solubility_water_mg_mL, logP, 20.0, cmc_mM=ps80_cmc, ksp=ps80_ksp
        ),
    }

    # Find strategy with maximum solubility (excluding water as baseline)
    non_water = {k: v for k, v in strategies.items() if k != "water"}
    max_strategy = max(non_water, key=lambda k: non_water[k])
    max_sol = non_water[max_strategy]
    max_fold = max_sol / solubility_water_mg_mL

    notes_parts = [
        f"logP={logP:.2f}",
        f"MW={drug_mw:.1f} g/mol",
        f"Best strategy: {max_strategy} ({max_fold:.1f}x)",
    ]

    return SolubilityComparisonResult(
        drug_name=drug_name,
        solubility_water=solubility_water_mg_mL,
        logP=logP,
        strategies=strategies,
        max_enhancement_strategy=max_strategy,
        max_enhancement_fold=round(max_fold, 4),
        notes="; ".join(notes_parts),
    )


__all__ = [
    "SolubilityComparisonResult",
    "cyclodextrin_solubility",
    "cosolvent_solubility",
    "surfactant_solubilization",
    "compare_enhancement_strategies",
]
