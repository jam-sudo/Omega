"""Bioavailability enhancement strategies for BCS II/IV drugs."""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BioavailEnhancementResult:
    """Results from comparing bioavailability enhancement strategies."""

    drug_name: str
    bcs_class: int
    logP: float
    base_f_estimated: float
    nanosizing_fold: float
    asd_fold: float
    sedds_fold: float
    smedds_fold: float
    predicted_f_nanosized: float
    predicted_f_asd: float
    predicted_f_sedds: float
    predicted_f_smedds: float
    recommended_strategy: str
    notes: str


# ---------------------------------------------------------------------------
# Individual strategy functions
# ---------------------------------------------------------------------------

def nanosizing_enhancement(
    solubility_bulk_mg_mL: float,
    particle_size_um: float,
    target_size_nm: float = 200.0,
    drug_density_g_mL: float = 1.2,
) -> float:
    """Estimate solubility fold-increase from nanosizing (Ostwald-Freundlich simplified).

    Uses empirical relation:
        fold_increase = (particle_size_um * 1000 / target_size_nm) ^ 0.3

    Clamped to [1.0, 10.0].

    Parameters
    ----------
    solubility_bulk_mg_mL : float
        Bulk crystalline solubility (mg/mL). Must be > 0.
    particle_size_um : float
        Starting bulk particle size in micrometres. Must be > 0.
    target_size_nm : float
        Target nano-particle size in nanometres (default 200 nm).
    drug_density_g_mL : float
        Drug density (g/mL); retained for API completeness (not used in simplified model).

    Returns
    -------
    float
        Fold-increase in apparent solubility after nanosizing.
    """
    if solubility_bulk_mg_mL <= 0:
        raise ValueError("solubility_bulk_mg_mL must be > 0")
    if particle_size_um <= 0:
        raise ValueError("particle_size_um must be > 0")
    if target_size_nm <= 0:
        raise ValueError("target_size_nm must be > 0")
    if drug_density_g_mL <= 0:
        raise ValueError("drug_density_g_mL must be > 0")

    bulk_nm = particle_size_um * 1000.0
    fold = (bulk_nm / target_size_nm) ** 0.3
    return float(min(max(fold, 1.0), 10.0))


def amorphous_enhancement(
    tm_C: float,
    tg_C: float,
    temperature_C: float = 37.0,
) -> float:
    """Estimate solubility advantage of amorphous form vs crystalline.

    Simplified free-energy model:
        fold = exp(0.005 * (tm_C - tg_C))

    Clamped to [1.0, 50.0].

    Parameters
    ----------
    tm_C : float
        Melting temperature of crystalline drug (°C). Must be > tg_C.
    tg_C : float
        Glass transition temperature of amorphous form (°C). Must be > 0.
    temperature_C : float
        Storage/body temperature (°C); retained for API completeness.

    Returns
    -------
    float
        Fold-increase in apparent solubility from amorphous dispersion.
    """
    if tg_C <= 0:
        raise ValueError("tg_C must be > 0")
    if tm_C <= tg_C:
        raise ValueError("tm_C must be > tg_C")

    fold = math.exp(0.005 * (tm_C - tg_C))
    return float(min(max(fold, 1.0), 50.0))


def lipid_formulation_enhancement(
    logP: float,
    meal_fat_g: float = 20.0,
    formulation_type: str = "SEDDS",
) -> float:
    """Estimate solubility/absorption fold-increase from lipid-based formulations.

    Model:
        fold = exp(0.3 * logP) * (meal_fat_g / 10) ^ 0.5 * formulation_factor

    Formulation factors:
        solution: 1.5, SEDDS: 2.0, SMEDDS: 3.0, SNEDDS: 4.0

    Clamped to [1.0, 20.0].

    Parameters
    ----------
    logP : float
        Octanol-water partition coefficient.
    meal_fat_g : float
        Fat content of co-administered meal (g).
    formulation_type : str
        One of "solution", "SEDDS", "SMEDDS", "SNEDDS".

    Returns
    -------
    float
        Fold-increase in apparent absorption.
    """
    if meal_fat_g < 0:
        raise ValueError("meal_fat_g must be >= 0")

    _factors: dict[str, float] = {
        "SOLUTION": 1.5,
        "SEDDS": 2.0,
        "SMEDDS": 3.0,
        "SNEDDS": 4.0,
    }
    ft = formulation_type.upper()
    if ft not in _factors:
        valid = ["solution", "SEDDS", "SMEDDS", "SNEDDS"]
        raise ValueError(
            f"formulation_type must be one of {valid}, got '{formulation_type}'"
        )

    ff = _factors[ft]
    fat_factor = (meal_fat_g / 10.0) ** 0.5 if meal_fat_g > 0 else 0.0
    fold = math.exp(0.3 * logP) * fat_factor * ff
    return float(min(max(fold, 1.0), 20.0))


# ---------------------------------------------------------------------------
# Comparison function
# ---------------------------------------------------------------------------

_BCS_BASE_F: dict[int, float] = {1: 0.90, 2: 0.20, 3: 0.50, 4: 0.10}


def compare_enhancement_strategies_ba(
    drug_name: str,
    bcs_class: int,
    logP: float,
    solubility_bulk_mg_mL: float,
    tm_C: float,
    tg_C: float,
    particle_size_um: float = 100.0,
    meal_fat_g: float = 20.0,
) -> BioavailEnhancementResult:
    """Compare nanosizing, ASD, and lipid formulation strategies for oral bioavailability.

    Base F is estimated from BCS class:
        I → 0.90, II → 0.20, III → 0.50, IV → 0.10

    Predicted F = min(base_F * fold_increase, 1.0).

    The recommended strategy is the one yielding highest predicted F.

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    bcs_class : int
        BCS classification (1, 2, 3, or 4).
    logP : float
        Octanol-water partition coefficient.
    solubility_bulk_mg_mL : float
        Bulk crystalline solubility (mg/mL). Must be > 0.
    tm_C : float
        Melting temperature (°C). Must be > tg_C > 0.
    tg_C : float
        Glass transition temperature of amorphous form (°C). Must be > 0.
    particle_size_um : float
        Starting bulk particle size (µm) for nanosizing estimate.
    meal_fat_g : float
        Co-administered meal fat content (g) for lipid formulation estimate.

    Returns
    -------
    BioavailEnhancementResult
    """
    if bcs_class not in (1, 2, 3, 4):
        raise ValueError(f"bcs_class must be 1, 2, 3, or 4; got {bcs_class}")
    if solubility_bulk_mg_mL <= 0:
        raise ValueError("solubility_bulk_mg_mL must be > 0")
    if tg_C <= 0:
        raise ValueError("tg_C must be > 0")
    if tm_C <= tg_C:
        raise ValueError("tm_C must be > tg_C")

    base_f = _BCS_BASE_F[bcs_class]

    nano_fold = nanosizing_enhancement(
        solubility_bulk_mg_mL=solubility_bulk_mg_mL,
        particle_size_um=particle_size_um,
    )
    asd_fold = amorphous_enhancement(tm_C=tm_C, tg_C=tg_C)
    sedds_fold = lipid_formulation_enhancement(
        logP=logP, meal_fat_g=meal_fat_g, formulation_type="SEDDS"
    )
    smedds_fold = lipid_formulation_enhancement(
        logP=logP, meal_fat_g=meal_fat_g, formulation_type="SMEDDS"
    )

    pred_nano = float(min(base_f * nano_fold, 1.0))
    pred_asd = float(min(base_f * asd_fold, 1.0))
    pred_sedds = float(min(base_f * sedds_fold, 1.0))
    pred_smedds = float(min(base_f * smedds_fold, 1.0))

    strategy_scores = {
        "nanosizing": pred_nano,
        "amorphous_solid_dispersion": pred_asd,
        "SEDDS": pred_sedds,
        "SMEDDS": pred_smedds,
    }
    recommended = max(strategy_scores, key=lambda k: strategy_scores[k])

    # Build human-readable notes
    notes_parts: list[str] = [
        f"BCS class {bcs_class} drug; base F estimated at {base_f:.0%}.",
        f"Nanosizing ({particle_size_um} µm → 200 nm): {nano_fold:.2f}x → F={pred_nano:.2%}.",
        f"ASD (tm={tm_C}°C, tg={tg_C}°C): {asd_fold:.2f}x → F={pred_asd:.2%}.",
        f"SEDDS: {sedds_fold:.2f}x → F={pred_sedds:.2%}.",
        f"SMEDDS: {smedds_fold:.2f}x → F={pred_smedds:.2%}.",
        f"Recommended strategy: {recommended}.",
    ]
    notes = " ".join(notes_parts)

    return BioavailEnhancementResult(
        drug_name=drug_name,
        bcs_class=bcs_class,
        logP=logP,
        base_f_estimated=base_f,
        nanosizing_fold=nano_fold,
        asd_fold=asd_fold,
        sedds_fold=sedds_fold,
        smedds_fold=smedds_fold,
        predicted_f_nanosized=pred_nano,
        predicted_f_asd=pred_asd,
        predicted_f_sedds=pred_sedds,
        predicted_f_smedds=pred_smedds,
        recommended_strategy=recommended,
        notes=notes,
    )


__all__ = [
    "BioavailEnhancementResult",
    "nanosizing_enhancement",
    "amorphous_enhancement",
    "lipid_formulation_enhancement",
    "compare_enhancement_strategies_ba",
]
