"""Phase 1114 — Drug liposome loading efficiency prediction.

Predicts encapsulation efficiency of drugs into liposomes based on
physicochemical properties and lipid composition.
"""

import math
from dataclasses import dataclass

_VALID_IONIZATION = {"acid", "base", "neutral"}
_VALID_LIPIDS = {"DPPC", "DSPC", "POPC", "DOPC", "DOPG"}


@dataclass(frozen=True)
class LiposomeLoadingResult:
    """Result from liposome drug loading prediction."""

    drug_name: str
    logP: float
    pka: float
    ionization_type: str
    lipid_type: str
    ph_internal: float
    ph_external: float
    loading_efficiency_pct: float  # 0-100
    trapping_mechanism: str  # "passive", "active_pH_gradient", "passive_partitioning"
    drug_to_lipid_ratio: float
    encapsulation_class: str  # "poor"(<30), "moderate"(30-70), "high"(>70)
    recommended_for: str
    notes: str


def _logistic(x: float, midpoint: float, steepness: float) -> float:
    """Logistic sigmoid function."""
    return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))


def _compute_efficiency(
    logP: float,
    pka: float,
    ionization_type: str,
    lipid_type: str,
    ph_internal: float,
    ph_external: float,
) -> tuple:
    """Compute loading efficiency and mechanism.

    Returns (efficiency_pct, mechanism).
    """
    # Passive partition efficiency (0-60%)
    partition_efficiency = _logistic(logP, midpoint=2.0, steepness=1.5) * 60.0

    # Ionization boost (0-35%)
    ionization_boost = 0.0
    mechanism = "passive_partitioning"

    if ionization_type == "base":
        # pH gradient trapping for bases: protonated form trapped in acidic core
        ph_diff = ph_external - ph_internal
        if ph_diff > 0:
            loading_factor = 1.0 + 10.0**ph_diff
            # Normalize: factor at standard 7.4/4.0 ≈ 2512, cap boost at 35%
            max_factor = 1.0 + 10.0 ** (7.4 - 4.0)  # ~2512
            ionization_boost = min(35.0, (loading_factor / max_factor) * 35.0)
            mechanism = "active_pH_gradient"
        else:
            mechanism = "passive_partitioning"

    elif ionization_type == "acid":
        # Acids prefer neutral interior; beneficial if ph_internal > pka
        if ph_internal > pka:
            # Fraction ionized at internal pH
            fraction_ionized = 1.0 / (1.0 + 10.0 ** (pka - ph_internal))
            ionization_boost = fraction_ionized * 35.0
            mechanism = "active_pH_gradient"
        else:
            mechanism = "passive_partitioning"

    # Base efficiency
    base_efficiency = partition_efficiency + ionization_boost

    # Lipid correction
    lipid_correction = 0.0
    if lipid_type in ("DPPC", "DSPC"):
        # Crystalline phase — best for hydrophobic drugs
        if logP > 2:
            lipid_correction = 10.0
    elif lipid_type == "DOPG":
        # Anionic — better for cationic bases
        if ionization_type == "base":
            lipid_correction = 15.0
    elif lipid_type == "DOPC":
        # Most fluid — slightly lower retention
        lipid_correction = -5.0
    # POPC: no correction (moderate)

    efficiency = min(95.0, base_efficiency + lipid_correction)
    efficiency = max(0.0, efficiency)

    return efficiency, mechanism


def _recommended_for(lipid_type: str, ionization_type: str, logP: float) -> str:
    descriptions = {
        "DPPC": "crystalline neutral lipid; optimal for hydrophobic drugs (logP>2)",
        "DSPC": "high-Tm neutral lipid; thermostable bilayers for hydrophobic drugs",
        "POPC": "fluid neutral lipid; general-purpose moderate loading",
        "DOPC": "highly fluid neutral lipid; amphiphilic or small molecules",
        "DOPG": "anionic lipid; electrostatically attracts cationic bases",
    }
    base = descriptions.get(lipid_type, lipid_type)
    if ionization_type == "base" and lipid_type == "DOPG":
        return base + "; best choice for cationic drugs with pH gradient"
    return base


def predict_loading_efficiency(
    drug_name: str,
    logP: float,
    pka: float,
    ionization_type: str,
    lipid_type: str,
    ph_internal: float = 4.0,
    ph_external: float = 7.4,
    drug_to_lipid_ratio: float = 0.1,
) -> LiposomeLoadingResult:
    """Predict liposome drug loading efficiency.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    logP:
        Octanol-water partition coefficient.
    pka:
        Drug pKa value.
    ionization_type:
        One of "acid", "base", or "neutral".
    lipid_type:
        One of "DPPC", "DSPC", "POPC", "DOPC", "DOPG".
    ph_internal:
        Internal liposome pH (default 4.0 for active loading).
    ph_external:
        External (buffer) pH (default 7.4).
    drug_to_lipid_ratio:
        Molar drug-to-lipid ratio (0, 1].

    Returns
    -------
    LiposomeLoadingResult
    """
    # --- Validation ---
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION}, got '{ionization_type}'"
        )
    if lipid_type not in _VALID_LIPIDS:
        raise ValueError(f"lipid_type must be one of {_VALID_LIPIDS}, got '{lipid_type}'")
    if not (0 < drug_to_lipid_ratio <= 1):
        raise ValueError("drug_to_lipid_ratio must be in (0, 1]")
    if not (0 <= ph_internal <= 14):
        raise ValueError("ph_internal must be in [0, 14]")
    if not (0 <= ph_external <= 14):
        raise ValueError("ph_external must be in [0, 14]")
    if not (-5 <= logP <= 10):
        raise ValueError("logP must be in [-5, 10]")

    efficiency, mechanism = _compute_efficiency(
        logP, pka, ionization_type, lipid_type, ph_internal, ph_external
    )

    # Encapsulation class
    if efficiency < 30.0:
        encap_class = "poor"
    elif efficiency <= 70.0:
        encap_class = "moderate"
    else:
        encap_class = "high"

    rec = _recommended_for(lipid_type, ionization_type, logP)

    notes = (
        f"logP={logP:.2f}, pKa={pka:.2f}, {ionization_type}. "
        f"Partition efficiency={_logistic(logP, 2.0, 1.5) * 60:.1f}%. "
        f"Mechanism: {mechanism}."
    )

    return LiposomeLoadingResult(
        drug_name=drug_name,
        logP=logP,
        pka=pka,
        ionization_type=ionization_type,
        lipid_type=lipid_type,
        ph_internal=ph_internal,
        ph_external=ph_external,
        loading_efficiency_pct=round(efficiency, 2),
        trapping_mechanism=mechanism,
        drug_to_lipid_ratio=drug_to_lipid_ratio,
        encapsulation_class=encap_class,
        recommended_for=rec,
        notes=notes,
    )


def screen_lipid_types(
    drug_name: str,
    logP: float,
    pka: float,
    ionization_type: str,
    ph_internal: float = 4.0,
    ph_external: float = 7.4,
    drug_to_lipid_ratio: float = 0.1,
) -> list:
    """Screen all lipid types for a given drug and return sorted results.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    logP:
        Octanol-water partition coefficient.
    pka:
        Drug pKa value.
    ionization_type:
        One of "acid", "base", or "neutral".
    ph_internal:
        Internal liposome pH.
    ph_external:
        External buffer pH.
    drug_to_lipid_ratio:
        Molar drug-to-lipid ratio.

    Returns
    -------
    list of LiposomeLoadingResult sorted by loading_efficiency_pct descending.
    """
    results = []
    for lipid in sorted(_VALID_LIPIDS):
        result = predict_loading_efficiency(
            drug_name=drug_name,
            logP=logP,
            pka=pka,
            ionization_type=ionization_type,
            lipid_type=lipid,
            ph_internal=ph_internal,
            ph_external=ph_external,
            drug_to_lipid_ratio=drug_to_lipid_ratio,
        )
        results.append(result)

    results.sort(key=lambda r: r.loading_efficiency_pct, reverse=True)
    return results
