"""
Phase 536 — Drug Excretion into Breast Milk

Predict drug concentration in breast milk for infant exposure assessment
using analytical milk/plasma ratio models.
"""

from __future__ import annotations

from dataclasses import dataclass

_MILK_FAT_FRACTION = 0.04  # 4% fat in mature milk
_PH_MILK = 7.0
_PH_PLASMA = 7.4
_MILK_INTAKE_ML_PER_KG_PER_DAY = 150.0
_VALID_DRUG_TYPES = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class BreastMilkExcretionResult:
    """Result of breast milk drug excretion prediction."""

    drug_name: str
    logP: float
    pKa: float
    drug_type: str
    fu_plasma: float
    c_plasma_mg_L: float
    mp_ratio: float
    c_milk_mg_L: float
    infant_daily_dose_mg_per_kg: float
    rid_pct: float
    lactation_safety: str
    lipid_trapping: bool
    ion_trapping: bool
    notes: str


def _compute_kp_lipid(logP: float) -> float:
    """Lipid:water partition coefficient from logP."""
    if logP <= 1.0:
        return 1.0
    return 10.0 ** (0.8 * logP - 0.5)


def _lipid_mp_ratio(fu: float, logP: float) -> float:
    """M:P ratio via lipid partitioning."""
    kp_lipid = _compute_kp_lipid(logP)
    return fu * (1.0 + _MILK_FAT_FRACTION * (kp_lipid - 1.0))


def _ion_trapping_factor_base(pKa: float) -> float:
    """Ion trapping factor for basic drug (milk pH < plasma pH)."""
    # Factor = (1 + 10^(pKa - pH_milk)) / (1 + 10^(pKa - pH_plasma))
    numerator = 1.0 + 10.0 ** (pKa - _PH_MILK)
    denominator = 1.0 + 10.0 ** (pKa - _PH_PLASMA)
    if denominator == 0.0:
        return 1.0
    return numerator / denominator


def predict_breast_milk_excretion(
    drug_name: str,
    logP: float,
    pKa: float,
    drug_type: str,
    fu_plasma: float,
    c_plasma_mg_L: float,
    maternal_dose_mg: float,
    infant_weight_kg: float = 5.0,
    maternal_weight_kg: float = 60.0,
) -> BreastMilkExcretionResult:
    """
    Predict drug concentration in breast milk and infant daily exposure.

    Parameters
    ----------
    drug_name : str
    logP : float
    pKa : float
    drug_type : str
        One of "acid", "base", "neutral"
    fu_plasma : float
        Unbound fraction in plasma (0, 1]
    c_plasma_mg_L : float
        Maternal plasma concentration (mg/L)
    maternal_dose_mg : float
    infant_weight_kg : float, optional
    maternal_weight_kg : float, optional

    Returns
    -------
    BreastMilkExcretionResult
    """
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(f"drug_type must be one of {_VALID_DRUG_TYPES}")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be in (0, 1]")
    if c_plasma_mg_L <= 0.0:
        raise ValueError("c_plasma_mg_L must be > 0")
    if maternal_dose_mg <= 0.0:
        raise ValueError("maternal_dose_mg must be > 0")

    # Compute M:P ratio
    if drug_type == "base":
        trap = _ion_trapping_factor_base(pKa)
        # Combined for basic lipophilic: take max of lipid and ion trapping
        raw_mp = fu_plasma * max(
            1.0 + _MILK_FAT_FRACTION * (_compute_kp_lipid(logP) - 1.0),
            trap,
        )
    elif drug_type == "acid":
        # Acids: slightly less ion trapping in milk (milk more acidic — ionized stays in plasma)
        raw_mp = fu_plasma * (1.0 + _MILK_FAT_FRACTION * (_compute_kp_lipid(logP) - 1.0))
    else:  # neutral
        raw_mp = fu_plasma * (1.0 + _MILK_FAT_FRACTION * (_compute_kp_lipid(logP) - 1.0))

    mp_ratio = max(0.01, min(20.0, raw_mp))

    # Milk concentration
    c_milk_mg_L = c_plasma_mg_L * mp_ratio

    # Infant daily dose
    # c_milk in mg/L; intake = 150 mL/kg/day = 0.150 L/kg/day
    infant_daily_dose_mg_per_kg = c_milk_mg_L * _MILK_INTAKE_ML_PER_KG_PER_DAY / 1000.0

    # RID: relative infant dose (%)
    # RID = (infant_dose_mg/kg * infant_weight_kg) / maternal_dose_mg * 100
    rid_pct = infant_daily_dose_mg_per_kg * infant_weight_kg / maternal_dose_mg * 100.0

    # Lactation safety classification
    if rid_pct < 10.0:
        lactation_safety = "compatible"
    elif rid_pct <= 25.0:
        lactation_safety = "use_with_caution"
    else:
        lactation_safety = "avoid"

    # Lipid trapping flag
    kp_lipid = _compute_kp_lipid(logP)
    lipid_contribution = fu_plasma * _MILK_FAT_FRACTION * (kp_lipid - 1.0)
    lipid_trapping = (
        logP > 2.0 and (lipid_contribution / mp_ratio > 0.20) if mp_ratio > 0 else False
    )

    # Ion trapping flag
    ion_trapping = drug_type == "base" and pKa > 7.5

    # Build notes
    notes_parts: list[str] = []
    notes_parts.append(f"M:P ratio = {mp_ratio:.3f}")
    notes_parts.append(f"RID = {rid_pct:.2f}%")
    if lipid_trapping:
        notes_parts.append("Significant lipid trapping in milk")
    if ion_trapping:
        notes_parts.append("Ion trapping (basic drug in acidic milk)")
    notes_parts.append(f"Safety: {lactation_safety}")
    notes = "; ".join(notes_parts)

    return BreastMilkExcretionResult(
        drug_name=drug_name,
        logP=logP,
        pKa=pKa,
        drug_type=drug_type,
        fu_plasma=fu_plasma,
        c_plasma_mg_L=c_plasma_mg_L,
        mp_ratio=mp_ratio,
        c_milk_mg_L=c_milk_mg_L,
        infant_daily_dose_mg_per_kg=infant_daily_dose_mg_per_kg,
        rid_pct=rid_pct,
        lactation_safety=lactation_safety,
        lipid_trapping=lipid_trapping,
        ion_trapping=ion_trapping,
        notes=notes,
    )


def screen_lactation_safety(compounds: list) -> list:
    """
    Screen a list of compounds for lactation safety, sorted by RID ascending
    (safest first).

    Parameters
    ----------
    compounds : list of dict
        Each dict contains keyword arguments for predict_breast_milk_excretion.

    Returns
    -------
    list of BreastMilkExcretionResult sorted by rid_pct ascending
    """
    results = []
    for compound in compounds:
        result = predict_breast_milk_excretion(**compound)
        results.append(result)
    results.sort(key=lambda r: r.rid_pct)
    return results
