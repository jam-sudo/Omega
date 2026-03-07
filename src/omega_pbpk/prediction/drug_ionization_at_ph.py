"""Phase 1030 — Drug Ionization at Physiological pH.

Predicts ionization state, solubility fold-change, permeability, and absorption
class for a drug at a given physiological pH using Henderson-Hasselbalch equations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IonizationAtPhResult:
    drug_name: str
    ph: float
    pka: float
    ionization_type: str
    fraction_ionized: float  # 0-1
    fraction_unionized: float  # 0-1
    charge_state: str  # "neutral", "cationic", "anionic", "zwitterionic"
    solubility_fold_change: float  # relative to neutral form
    permeability_factor: float  # 0-1
    absorption_class: str  # "high", "moderate", "low"
    notes: str


def predict_ionization_at_ph(
    drug_name: str,
    pka: float,
    ionization_type: str,  # "acid", "base", "neutral", "zwitterion"
    ph: float = 7.4,
    logp_neutral: float = 2.0,
    mw_Da: float = 300.0,
) -> IonizationAtPhResult:
    """Predict ionization state at a single physiological pH.

    Args:
        drug_name: Name of the drug.
        pka: Acid dissociation constant (0-14).
        ionization_type: One of "acid", "base", "neutral", "zwitterion".
        ph: Target pH value (0-14).
        logp_neutral: LogP of the neutral form.
        mw_Da: Molecular weight in Daltons (must be > 0).

    Returns:
        IonizationAtPhResult with ionization metrics.
    """
    # Validation
    if not (0.0 <= pka <= 14.0):
        raise ValueError(f"pka must be in [0, 14], got {pka}")
    if not (0.0 <= ph <= 14.0):
        raise ValueError(f"ph must be in [0, 14], got {ph}")
    if ionization_type not in {"acid", "base", "neutral", "zwitterion"}:
        raise ValueError(
            f"ionization_type must be 'acid', 'base', 'neutral', or 'zwitterion', "
            f"got '{ionization_type}'"
        )
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")

    # Henderson-Hasselbalch ionization
    if ionization_type == "acid":
        # fraction_ionized = 1 / (1 + 10^(pka - ph))
        exponent = pka - ph
        fraction_ionized = 1.0 / (1.0 + 10.0**exponent)
    elif ionization_type == "base":
        # fraction_ionized = 1 / (1 + 10^(ph - pka))
        exponent = ph - pka
        fraction_ionized = 1.0 / (1.0 + 10.0**exponent)
    elif ionization_type == "neutral":
        fraction_ionized = 0.0
    else:  # zwitterion — use acid formula (simplified)
        exponent = pka - ph
        fraction_ionized = 1.0 / (1.0 + 10.0**exponent)

    fraction_unionized = 1.0 - fraction_ionized

    # Charge state determination
    if ionization_type == "neutral":
        charge_state = "neutral"
    elif ionization_type == "zwitterion":
        charge_state = "zwitterionic"
    elif ionization_type == "acid" and fraction_ionized > 0.5:
        charge_state = "anionic"
    elif ionization_type == "base" and fraction_ionized > 0.5:
        charge_state = "cationic"
    else:
        charge_state = "neutral"

    # Solubility fold change relative to neutral form (Henderson-Hasselbalch)
    if ionization_type == "acid":
        # S/S0 = 1 + 10^(ph - pka)
        raw_sfc = 1.0 + 10.0 ** (ph - pka)
    elif ionization_type == "base":
        # S/S0 = 1 + 10^(pka - ph)
        raw_sfc = 1.0 + 10.0 ** (pka - ph)
    else:
        # neutral / zwitterion
        raw_sfc = 1.0

    # Clamp to [0.1, 1000.0]
    solubility_fold_change = max(0.1, min(1000.0, raw_sfc))

    # Permeability factor: unionized fraction corrected for logP
    logp_correction = min(1.0, 10.0 ** (logp_neutral / 5.0))
    raw_perm = fraction_unionized * logp_correction
    permeability_factor = max(0.0, min(1.0, raw_perm))

    # Absorption class
    if permeability_factor > 0.5:
        absorption_class = "high"
    elif permeability_factor > 0.2:
        absorption_class = "moderate"
    else:
        absorption_class = "low"

    notes = (
        f"ionization_type={ionization_type}, pKa={pka}, pH={ph}, "
        f"fi={fraction_ionized:.4f}, logP={logp_neutral}, "
        f"perm_correction={logp_correction:.3f}"
    )

    return IonizationAtPhResult(
        drug_name=drug_name,
        ph=ph,
        pka=pka,
        ionization_type=ionization_type,
        fraction_ionized=fraction_ionized,
        fraction_unionized=fraction_unionized,
        charge_state=charge_state,
        solubility_fold_change=solubility_fold_change,
        permeability_factor=permeability_factor,
        absorption_class=absorption_class,
        notes=notes,
    )


def compare_ph_conditions(
    drug_name: str,
    pka: float,
    ionization_type: str,
    ph_values: list,
    logp_neutral: float = 2.0,
    mw_Da: float = 300.0,
) -> list:
    """Compare ionization at multiple pH values.

    Args:
        drug_name: Name of the drug.
        pka: Acid dissociation constant.
        ionization_type: Ionization type ("acid", "base", "neutral", "zwitterion").
        ph_values: List of pH floats to compare.
        logp_neutral: LogP of the neutral form.
        mw_Da: Molecular weight in Daltons.

    Returns:
        List of IonizationAtPhResult in the same order as ph_values.
    """
    return [
        predict_ionization_at_ph(
            drug_name=drug_name,
            pka=pka,
            ionization_type=ionization_type,
            ph=ph,
            logp_neutral=logp_neutral,
            mw_Da=mw_Da,
        )
        for ph in ph_values
    ]
