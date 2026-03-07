"""Biologic (protein/mAb) stability predictor — Phase 456."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ThermalStabilityResult:
    """Predicted thermal stability of a biologic."""

    tm_C_predicted: float
    stability_category: str  # 'high', 'moderate', 'low'
    notes: str


@dataclass(frozen=True)
class ChemicalStabilityResult:
    """Predicted chemical stability of a biologic."""

    deamidation_risk: str  # 'low', 'moderate', 'high'
    oxidation_risk: str  # 'low', 'moderate', 'high'
    hydrolysis_risk: str  # 'low', 'moderate', 'high'
    overall_chemical_risk: str  # 'low', 'moderate', 'high'
    notes: str


@dataclass(frozen=True)
class ColloidalStabilityResult:
    """Predicted colloidal stability of a biologic."""

    zeta_potential_mV: float
    colloidal_stability: str  # 'stable', 'borderline', 'unstable'
    notes: str


# ---------------------------------------------------------------------------
# Thermal stability
# ---------------------------------------------------------------------------


def predict_thermal_stability(
    mw_kDa: float,
    isoelectric_point: float,
    n_disulfide_bonds: int,
    glycosylated: bool = False,
) -> ThermalStabilityResult:
    """Predict melting temperature and thermal stability category.

    Model assumptions:
    - Base Tm ≈ 60 °C for a generic IgG-like protein.
    - Each disulfide bond contributes +1.5 °C (conformational constraint).
    - Glycosylation adds +3 °C (shielding / solvation benefit).
    - Larger MW slightly increases Tm via cooperative folding (+0.05 °C per kDa
      above 10 kDa, capped at +10 °C).
    - Extreme pI (< 5 or > 9) reduces Tm by 4 °C (electrostatic strain).

    Args:
        mw_kDa: Molecular weight in kDa. Must be > 0.
        isoelectric_point: Isoelectric point (pH units). Must be in [1, 14].
        n_disulfide_bonds: Number of disulfide bonds. Must be >= 0.
        glycosylated: Whether the molecule is glycosylated.

    Returns:
        ThermalStabilityResult with predicted Tm and category.

    Raises:
        ValueError: If any argument is out of range.
    """
    if mw_kDa <= 0:
        raise ValueError(f"mw_kDa must be > 0, got {mw_kDa}")
    if not (1.0 <= isoelectric_point <= 14.0):
        raise ValueError(f"isoelectric_point must be in [1, 14], got {isoelectric_point}")
    if n_disulfide_bonds < 0:
        raise ValueError(f"n_disulfide_bonds must be >= 0, got {n_disulfide_bonds}")

    tm = 60.0  # base Tm °C

    # Disulfide contribution
    tm += n_disulfide_bonds * 1.5

    # Glycosylation benefit
    if glycosylated:
        tm += 3.0

    # MW contribution (cooperative folding)
    mw_excess = max(0.0, mw_kDa - 10.0)
    tm += min(10.0, mw_excess * 0.05)

    # pI penalty for extremes
    if isoelectric_point < 5.0 or isoelectric_point > 9.0:
        tm -= 4.0

    # Category
    if tm >= 70.0:
        category = "high"
    elif tm >= 60.0:
        category = "moderate"
    else:
        category = "low"

    notes_parts = [
        f"Predicted Tm = {tm:.1f} °C.",
        f"Disulfide bonds: {n_disulfide_bonds} (+{n_disulfide_bonds * 1.5:.1f} °C).",
    ]
    if glycosylated:
        notes_parts.append("Glycosylation contributes +3.0 °C.")
    if isoelectric_point < 5.0 or isoelectric_point > 9.0:
        notes_parts.append(f"Extreme pI ({isoelectric_point:.1f}) reduces Tm by 4.0 °C.")

    return ThermalStabilityResult(
        tm_C_predicted=round(tm, 2),
        stability_category=category,
        notes=" ".join(notes_parts),
    )


# ---------------------------------------------------------------------------
# Chemical stability
# ---------------------------------------------------------------------------

_RISK_LEVELS = ("low", "moderate", "high")


def _risk_from_count(count: int, low_thresh: int, high_thresh: int) -> str:
    """Map a residue count to a risk level string."""
    if count <= low_thresh:
        return "low"
    if count <= high_thresh:
        return "moderate"
    return "high"


def predict_chemical_stability(
    n_asp: int,
    n_asn: int,
    n_met: int,
    n_trp: int,
    pH: float = 7.4,
    temperature_C: float = 25.0,
) -> ChemicalStabilityResult:
    """Predict chemical degradation risks for a biologic.

    Degradation pathways:
    - Deamidation: driven by Asn (and to a lesser extent Asp) count + low/high pH.
    - Oxidation: driven by Met and Trp count + elevated temperature.
    - Hydrolysis: driven by Asp count + acidic pH.

    Risk thresholds (residue count at neutral pH, 25 °C):
        Deamidation (Asn): low ≤ 2, moderate ≤ 5, high > 5.
        Oxidation (Met+Trp): low ≤ 2, moderate ≤ 5, high > 5.
        Hydrolysis (Asp): low ≤ 3, moderate ≤ 7, high > 7.

    pH and temperature adjust risk upward:
        - pH < 5 or > 8 → deamidation risk shifted up one level.
        - Temperature > 40 °C → oxidation risk shifted up one level.
        - pH < 5 → hydrolysis risk shifted up one level.

    Args:
        n_asp: Number of aspartate residues.
        n_asn: Number of asparagine residues.
        n_met: Number of methionine residues.
        n_trp: Number of tryptophan residues.
        pH: Formulation pH. Default 7.4.
        temperature_C: Storage temperature (°C). Default 25.

    Returns:
        ChemicalStabilityResult with individual and overall risk levels.

    Raises:
        ValueError: If counts are negative or pH / temperature out of range.
    """
    for name, val in [
        ("n_asp", n_asp),
        ("n_asn", n_asn),
        ("n_met", n_met),
        ("n_trp", n_trp),
    ]:
        if val < 0:
            raise ValueError(f"{name} must be >= 0, got {val}")

    if not (0.0 <= pH <= 14.0):
        raise ValueError(f"pH must be in [0, 14], got {pH}")
    if temperature_C < -273.15:
        raise ValueError(f"temperature_C is unphysical, got {temperature_C}")

    # --- Deamidation ---
    deam_idx = _RISK_LEVELS.index(_risk_from_count(n_asn, 2, 5))
    if pH < 5.0 or pH > 8.0:
        deam_idx = min(2, deam_idx + 1)
    deamidation_risk = _RISK_LEVELS[deam_idx]

    # --- Oxidation ---
    ox_count = n_met + n_trp
    ox_idx = _RISK_LEVELS.index(_risk_from_count(ox_count, 2, 5))
    if temperature_C > 40.0:
        ox_idx = min(2, ox_idx + 1)
    oxidation_risk = _RISK_LEVELS[ox_idx]

    # --- Hydrolysis (Asp-Pro bonds are hotspots; use Asp as proxy) ---
    hyd_idx = _RISK_LEVELS.index(_risk_from_count(n_asp, 3, 7))
    if pH < 5.0:
        hyd_idx = min(2, hyd_idx + 1)
    hydrolysis_risk = _RISK_LEVELS[hyd_idx]

    # --- Overall: worst of the three ---
    overall_idx = max(
        _RISK_LEVELS.index(deamidation_risk),
        _RISK_LEVELS.index(oxidation_risk),
        _RISK_LEVELS.index(hydrolysis_risk),
    )
    overall_chemical_risk = _RISK_LEVELS[overall_idx]

    notes = (
        f"Deamidation (Asn={n_asn}): {deamidation_risk}. "
        f"Oxidation (Met={n_met}, Trp={n_trp}): {oxidation_risk}. "
        f"Hydrolysis (Asp={n_asp}): {hydrolysis_risk}. "
        f"pH={pH:.1f}, T={temperature_C:.1f} °C."
    )

    return ChemicalStabilityResult(
        deamidation_risk=deamidation_risk,
        oxidation_risk=oxidation_risk,
        hydrolysis_risk=hydrolysis_risk,
        overall_chemical_risk=overall_chemical_risk,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Colloidal stability
# ---------------------------------------------------------------------------


def predict_colloidal_stability(
    mw_kDa: float,
    isoelectric_point: float,
    pH: float = 7.4,
    ionic_strength_mM: float = 150.0,
) -> ColloidalStabilityResult:
    """Predict colloidal stability via estimated zeta potential.

    Model:
    - Effective surface charge ∝ (pH - pI).  A molecule is positively charged
      below pI and negatively charged above pI.
    - Zeta potential (mV) ≈ base_scale * (pH - pI), capped at ±80 mV.
      base_scale = 8 mV / pH-unit (approximate for mAbs).
    - Debye screening by ionic strength reduces |zeta| via:
        screening_factor = exp(-ionic_strength_mM / 500.0)
      (empirical; 500 mM is the "screening decay" constant).
    - |zeta| >= 25 mV → stable; 15–25 mV → borderline; < 15 mV → unstable.

    Args:
        mw_kDa: Molecular weight in kDa. Must be > 0.
        isoelectric_point: Isoelectric point. Must be in [1, 14].
        pH: Solution pH. Default 7.4.
        ionic_strength_mM: Ionic strength in mM. Must be >= 0.

    Returns:
        ColloidalStabilityResult with zeta potential and stability category.

    Raises:
        ValueError: If arguments are out of range.
    """
    if mw_kDa <= 0:
        raise ValueError(f"mw_kDa must be > 0, got {mw_kDa}")
    if not (1.0 <= isoelectric_point <= 14.0):
        raise ValueError(f"isoelectric_point must be in [1, 14], got {isoelectric_point}")
    if not (0.0 <= pH <= 14.0):
        raise ValueError(f"pH must be in [0, 14], got {pH}")
    if ionic_strength_mM < 0:
        raise ValueError(f"ionic_strength_mM must be >= 0, got {ionic_strength_mM}")

    # Convention: below pI → positive charge; above pI → negative charge.
    # Zeta is proportional to (pI - pH): positive when pH < pI, negative when pH > pI.
    base_scale = 8.0  # mV per pH unit
    raw_zeta = base_scale * (isoelectric_point - pH)

    # Cap at ±80 mV
    raw_zeta = max(-80.0, min(80.0, raw_zeta))

    # Debye screening (higher ionic strength → more screening → lower |zeta|)
    screening_factor = math.exp(-ionic_strength_mM / 500.0)
    zeta = raw_zeta * screening_factor

    abs_zeta = abs(zeta)
    if abs_zeta >= 25.0:
        stability = "stable"
    elif abs_zeta >= 15.0:
        stability = "borderline"
    else:
        stability = "unstable"

    notes = (
        f"Zeta potential \u2248 {zeta:.1f} mV (|{abs_zeta:.1f}| mV). "
        f"pH={pH:.1f}, pI={isoelectric_point:.1f} (delta pI-pH={isoelectric_point - pH:.1f}), "
        f"ionic strength={ionic_strength_mM:.0f} mM. "
        f"Screening factor={screening_factor:.3f}."
    )

    return ColloidalStabilityResult(
        zeta_potential_mV=round(zeta, 2),
        colloidal_stability=stability,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Overall stability assessment
# ---------------------------------------------------------------------------


def assess_overall_stability(
    mw_kDa: float,
    isoelectric_point: float,
    n_disulfide_bonds: int,
    n_asp: int,
    n_asn: int,
    n_met: int,
    n_trp: int,
    glycosylated: bool = False,
    pH: float = 7.4,
) -> dict:
    """Assess thermal, chemical, and colloidal stability in one call.

    Args:
        mw_kDa: Molecular weight in kDa.
        isoelectric_point: Isoelectric point.
        n_disulfide_bonds: Number of disulfide bonds.
        n_asp: Number of aspartate residues.
        n_asn: Number of asparagine residues.
        n_met: Number of methionine residues.
        n_trp: Number of tryptophan residues.
        glycosylated: Whether the molecule is glycosylated.
        pH: Solution / formulation pH.

    Returns:
        Dict with keys:
            - 'thermal': ThermalStabilityResult
            - 'chemical': ChemicalStabilityResult
            - 'colloidal': ColloidalStabilityResult
            - 'overall_risk': str ('low', 'moderate', 'high')
    """
    thermal = predict_thermal_stability(
        mw_kDa=mw_kDa,
        isoelectric_point=isoelectric_point,
        n_disulfide_bonds=n_disulfide_bonds,
        glycosylated=glycosylated,
    )
    chemical = predict_chemical_stability(
        n_asp=n_asp,
        n_asn=n_asn,
        n_met=n_met,
        n_trp=n_trp,
        pH=pH,
    )
    colloidal = predict_colloidal_stability(
        mw_kDa=mw_kDa,
        isoelectric_point=isoelectric_point,
        pH=pH,
    )

    # Map thermal stability category to risk level (inverted)
    _thermal_to_risk = {"high": "low", "moderate": "moderate", "low": "high"}
    thermal_risk = _thermal_to_risk[thermal.stability_category]

    overall_idx = max(
        _RISK_LEVELS.index(thermal_risk),
        _RISK_LEVELS.index(chemical.overall_chemical_risk),
        _RISK_LEVELS.index(
            "low"
            if colloidal.colloidal_stability == "stable"
            else "moderate"
            if colloidal.colloidal_stability == "borderline"
            else "high"
        ),
    )
    overall_risk = _RISK_LEVELS[overall_idx]

    return {
        "thermal": thermal,
        "chemical": chemical,
        "colloidal": colloidal,
        "overall_risk": overall_risk,
    }
