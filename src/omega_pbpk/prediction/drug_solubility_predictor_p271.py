"""Phase 271: Drug solubility predictor using General Solubility Equation (GSE).

Predicts aqueous drug solubility from physicochemical properties including
pH-dependent solubility via Henderson-Hasselbalch equation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SolubilityPredictionResult:
    """Result of drug solubility prediction."""

    drug_name: str
    mw: float
    logp: float
    mp_C: float
    pka: float | None
    ionization_type: str
    sw_intrinsic_mg_mL: float
    sw_gastric_mg_mL: float
    sw_intestinal_mg_mL: float
    sw_colon_mg_mL: float
    solubility_class: str
    bcs_dose_number: float
    bcs_class: str
    notes: str


def _gse_log_sw(logp: float, mp_C: float) -> float:
    """General Solubility Equation: log10(Sw in mol/L).

    Based on Yalkowsky GSE for solids.
    """
    return 0.5 - logp - 0.01 * (mp_C - 25.0)


def _sw_mol_per_L(log_sw: float) -> float:
    """Convert log10(Sw) to Sw in mol/L."""
    return 10.0**log_sw


def _sw_mg_per_mL(sw_mol_per_L: float, mw: float) -> float:
    """Convert mol/L to mg/mL (= g/L)."""
    return sw_mol_per_L * mw / 1000.0


def _ph_dependent_solubility(
    sw_intrinsic: float, ph: float, pka: float | None, ionization_type: str
) -> float:
    """Compute pH-dependent solubility via Henderson-Hasselbalch.

    Args:
        sw_intrinsic: Intrinsic solubility in mg/mL (unionized form).
        ph: Target pH.
        pka: Drug pKa value.
        ionization_type: "acid", "base", or "neutral".

    Returns:
        Solubility in mg/mL at the given pH.
    """
    if pka is None or ionization_type == "neutral":
        return sw_intrinsic

    if ionization_type == "acid":
        # Acidic drug: ionizes above pKa, more soluble at high pH
        if ph > pka:
            return sw_intrinsic * (1.0 + 10.0 ** (ph - pka))
        else:
            return sw_intrinsic

    elif ionization_type == "base":
        # Basic drug: ionizes below pKa, more soluble at low pH
        if ph < pka:
            return sw_intrinsic * (1.0 + 10.0 ** (pka - ph))
        else:
            return sw_intrinsic

    return sw_intrinsic


def _classify_solubility(sw_intestinal_mg_mL: float) -> str:
    """Classify solubility per BCS guidance."""
    if sw_intestinal_mg_mL >= 1.0:
        return "highly_soluble"
    elif sw_intestinal_mg_mL >= 0.1:
        return "moderately_soluble"
    else:
        return "poorly_soluble"


def predict_solubility(
    drug_name: str,
    mw: float,
    logp: float,
    mp_C: float,
    pka: float | None = None,
    ionization_type: str = "neutral",
    dose_mg: float = 100.0,
) -> SolubilityPredictionResult:
    """Predict aqueous drug solubility from physicochemical properties.

    Args:
        drug_name: Name of the drug.
        mw: Molecular weight in g/mol.
        logp: Octanol-water partition coefficient.
        mp_C: Melting point in Celsius.
        pka: Drug pKa (optional).
        ionization_type: "acid", "base", or "neutral".
        dose_mg: Dose in mg (used for BCS dose number calculation).

    Returns:
        SolubilityPredictionResult with solubility predictions at multiple pH values.

    Raises:
        ValueError: If input parameters are invalid.
    """
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if not (-5.0 <= logp <= 10.0):
        raise ValueError(f"logp must be in [-5, 10], got {logp}")
    if not (-100.0 <= mp_C <= 300.0):
        raise ValueError(f"mp_C must be in [-100, 300], got {mp_C}")
    if ionization_type not in {"acid", "base", "neutral"}:
        raise ValueError(
            f"ionization_type must be 'acid', 'base', or 'neutral', got '{ionization_type}'"
        )

    # Compute intrinsic solubility using GSE (at pH where drug is fully unionized)
    log_sw = _gse_log_sw(logp, mp_C)
    sw_mol_L = _sw_mol_per_L(log_sw)
    sw_intrinsic = _sw_mg_per_mL(sw_mol_L, mw)

    # BCS-relevant compartment pH values
    pH_gastric = 1.2  # Fasted stomach
    pH_intestinal = 6.8  # Small intestine (fasted)
    pH_colon = 7.4  # Colon / systemic

    sw_gastric = _ph_dependent_solubility(sw_intrinsic, pH_gastric, pka, ionization_type)
    sw_intestinal = _ph_dependent_solubility(sw_intrinsic, pH_intestinal, pka, ionization_type)
    sw_colon = _ph_dependent_solubility(sw_intrinsic, pH_colon, pka, ionization_type)
    # Intrinsic at pH 7.4
    sw_at_74 = _ph_dependent_solubility(sw_intrinsic, 7.4, pka, ionization_type)

    solubility_class = _classify_solubility(sw_intestinal)

    # BCS Dose Number: Do = dose / (sw_intestinal * 250 mL)
    bcs_dose_number = dose_mg / (sw_intestinal * 250.0) if sw_intestinal > 0 else float("inf")
    bcs_class = "I/III" if bcs_dose_number < 1.0 else "II/IV"

    notes_parts = []
    if sw_intrinsic < 0.01:
        notes_parts.append("Very low intrinsic solubility — formulation strategy needed.")
    if bcs_dose_number >= 1.0:
        notes_parts.append("BCS dose number >= 1 suggests dissolution-limited absorption.")
    if ionization_type != "neutral" and pka is not None:
        notes_parts.append(
            f"pH-dependent solubility modeled via Henderson-Hasselbalch (pKa={pka:.1f}, {ionization_type})."
        )
    notes = " ".join(notes_parts) if notes_parts else "Solubility prediction complete."

    return SolubilityPredictionResult(
        drug_name=drug_name,
        mw=mw,
        logp=logp,
        mp_C=mp_C,
        pka=pka,
        ionization_type=ionization_type,
        sw_intrinsic_mg_mL=sw_at_74,
        sw_gastric_mg_mL=sw_gastric,
        sw_intestinal_mg_mL=sw_intestinal,
        sw_colon_mg_mL=sw_colon,
        solubility_class=solubility_class,
        bcs_dose_number=bcs_dose_number,
        bcs_class=bcs_class,
        notes=notes,
    )


def screen_solubility_conditions(
    drug_name: str,
    mw: float,
    logp: float,
    mp_C: float,
    pka: float | None = None,
    ionization_type: str = "neutral",
) -> dict:
    """Screen drug solubility across GI compartment conditions.

    Args:
        drug_name: Name of the drug.
        mw: Molecular weight in g/mol.
        logp: Octanol-water partition coefficient.
        mp_C: Melting point in Celsius.
        pka: Drug pKa (optional).
        ionization_type: "acid", "base", or "neutral".

    Returns:
        Dict with keys "gastric", "intestinal", "colon", "plasma" mapping to
        solubility in mg/mL at each physiological condition.
    """
    # Compute intrinsic solubility
    log_sw = _gse_log_sw(logp, mp_C)
    sw_mol_L = _sw_mol_per_L(log_sw)
    sw_intrinsic = _sw_mg_per_mL(sw_mol_L, mw)

    # Physiological pH values
    ph_map = {
        "gastric": 1.2,
        "intestinal": 6.8,
        "colon": 7.4,
        "plasma": 7.4,
    }

    return {
        compartment: _ph_dependent_solubility(sw_intrinsic, ph, pka, ionization_type)
        for compartment, ph in ph_map.items()
    }
