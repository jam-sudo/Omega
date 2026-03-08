"""Phase 598 — Drug Salivary Excretion

Predicts drug concentrations in saliva based on plasma-salivary partitioning.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SalivaryDrugExcretionResult:
    """Result of salivary drug excretion prediction."""

    drug_name: str
    plasma_conc_mg_L: float
    saliva_conc_mg_L: float
    saliva_plasma_ratio: float  # S/P ratio
    fu_plasma: float
    fu_saliva: float
    salivary_clearance_mL_per_min: float
    salivary_flow_mL_per_min: float  # used in calculation
    is_salivary_monitor_viable: bool  # True if S/P ratio is consistent (0.5–2.0)
    ionization_correction: float  # pH effect factor
    excretion_class: str  # "low"(<0.5), "similar"(0.5–2.0), "concentrated"(>2.0)
    notes: str


def predict_salivary_excretion(
    drug_name: str,
    plasma_conc_mg_L: float,
    fu_plasma: float,
    pka: float = None,
    drug_type: str = "neutral",
    salivary_ph: float = 6.8,
    plasma_ph: float = 7.4,
    salivary_flow_mL_per_min: float = 0.5,
) -> SalivaryDrugExcretionResult:
    """Predict salivary drug excretion from plasma concentration.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    plasma_conc_mg_L : float
        Plasma drug concentration in mg/L.
    fu_plasma : float
        Unbound fraction in plasma (0 < fu_plasma <= 1).
    pka : float or None
        Acid/base dissociation constant. Used for ionization correction.
    drug_type : str
        "neutral", "acid", or "base".
    salivary_ph : float
        Salivary pH (default 6.8).
    plasma_ph : float
        Plasma pH (default 7.4).
    salivary_flow_mL_per_min : float
        Salivary flow rate in mL/min (default 0.5).
    """
    # --- Validation ---
    if plasma_conc_mg_L <= 0:
        raise ValueError(f"plasma_conc_mg_L must be > 0, got {plasma_conc_mg_L}")
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if salivary_flow_mL_per_min <= 0:
        raise ValueError(f"salivary_flow_mL_per_min must be > 0, got {salivary_flow_mL_per_min}")
    if plasma_ph <= 0:
        raise ValueError(f"plasma_ph must be > 0, got {plasma_ph}")
    if salivary_ph <= 0:
        raise ValueError(f"salivary_ph must be > 0, got {salivary_ph}")
    if drug_type not in ("neutral", "acid", "base"):
        raise ValueError(f"drug_type must be 'neutral', 'acid', or 'base', got {drug_type!r}")

    # --- Ionization correction ---
    if drug_type == "neutral":
        ion_factor = 1.0
    elif drug_type == "acid":
        if pka is not None:
            numerator = 1.0 + 10 ** (plasma_ph - pka)
            denominator = 1.0 + 10 ** (salivary_ph - pka)
            ion_factor = numerator / denominator if denominator != 0 else 1.0
        else:
            ion_factor = 1.0
    else:  # base
        if pka is not None:
            numerator = 1.0 + 10 ** (pka - plasma_ph)
            denominator = 1.0 + 10 ** (pka - salivary_ph)
            ion_factor = numerator / denominator if denominator != 0 else 1.0
        else:
            ion_factor = 1.0

    # --- S/P ratio and concentrations ---
    sp_ratio = fu_plasma * ion_factor
    saliva_conc_mg_L = plasma_conc_mg_L * sp_ratio

    # Saliva has little protein binding
    fu_saliva = 1.0

    # Salivary clearance
    salivary_clearance_mL_per_min = salivary_flow_mL_per_min * sp_ratio

    # Monitoring viability
    is_viable = 0.5 <= sp_ratio <= 2.0

    # Excretion class
    if sp_ratio < 0.5:
        excretion_class = "low"
    elif sp_ratio <= 2.0:
        excretion_class = "similar"
    else:
        excretion_class = "concentrated"

    notes = f"drug_type={drug_type}, pka={pka}, ion_factor={ion_factor:.4f}, S/P={sp_ratio:.4f}"

    return SalivaryDrugExcretionResult(
        drug_name=drug_name,
        plasma_conc_mg_L=plasma_conc_mg_L,
        saliva_conc_mg_L=saliva_conc_mg_L,
        saliva_plasma_ratio=sp_ratio,
        fu_plasma=fu_plasma,
        fu_saliva=fu_saliva,
        salivary_clearance_mL_per_min=salivary_clearance_mL_per_min,
        salivary_flow_mL_per_min=salivary_flow_mL_per_min,
        is_salivary_monitor_viable=is_viable,
        ionization_correction=ion_factor,
        excretion_class=excretion_class,
        notes=notes,
    )


def saliva_monitoring_assessment(
    drug_name: str,
    plasma_conc_mg_L: float,
    fu_plasma: float,
    pka: float = None,
    drug_type: str = "neutral",
) -> dict:
    """Assess whether salivary monitoring is viable for TDM.

    Parameters
    ----------
    drug_name : str
        Drug name.
    plasma_conc_mg_L : float
        Plasma drug concentration in mg/L.
    fu_plasma : float
        Unbound fraction in plasma.
    pka : float or None
        pKa value.
    drug_type : str
        "neutral", "acid", or "base".

    Returns
    -------
    dict with keys: viable (bool), ratio (float), recommendation (str)
    """
    result = predict_salivary_excretion(
        drug_name=drug_name,
        plasma_conc_mg_L=plasma_conc_mg_L,
        fu_plasma=fu_plasma,
        pka=pka,
        drug_type=drug_type,
    )

    ratio = result.saliva_plasma_ratio
    viable = result.is_salivary_monitor_viable

    if viable:
        recommendation = (
            f"Salivary monitoring is viable for {drug_name}. "
            f"S/P ratio={ratio:.3f} is within acceptable range (0.5–2.0). "
            "Saliva can be used as a non-invasive surrogate for plasma."
        )
    elif ratio < 0.5:
        recommendation = (
            f"Salivary monitoring is NOT recommended for {drug_name}. "
            f"S/P ratio={ratio:.3f} is too low (<0.5); saliva underrepresents plasma levels."
        )
    else:
        recommendation = (
            f"Salivary monitoring may be feasible for {drug_name} with caution. "
            f"S/P ratio={ratio:.3f} is elevated (>2.0); saliva concentrates the drug."
        )

    return {
        "viable": viable,
        "ratio": ratio,
        "recommendation": recommendation,
    }
