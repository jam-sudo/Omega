"""Phase 488 — Drug Precipitation in GI Tract.

Models pH-dependent drug precipitation and supersaturation in the GI tract
for ionizable drugs.  High doses of poorly soluble drugs may exceed local
solubility → precipitation → reduced bioavailable fraction.

Reference: Henderson-Hasselbalch pH-solubility relationship.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# GI segment definitions
# --------------------------------------------------------------------------- #

_GI_SEGMENTS = {
    "stomach": {"pH": 1.5, "volume_mL": 250.0},
    "duodenum": {"pH": 6.0, "volume_mL": 50.0},
    "jejunum": {"pH": 6.5, "volume_mL": 300.0},
    "ileum": {"pH": 7.2, "volume_mL": 400.0},
}

_VALID_IONIZATION = {"acid", "base", "neutral"}


# --------------------------------------------------------------------------- #
# Result dataclass
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GIPrecipitationResult:
    """Result of GI precipitation modelling for a single drug dose."""

    drug_name: str
    dose_mg: float
    pKa: float
    ionization_type: str
    intrinsic_solubility_mg_L: float

    dissolved_stomach_mg: float
    dissolved_duodenum_mg: float
    dissolved_jejunum_mg: float
    dissolved_ileum_mg: float

    precipitated_total_mg: float
    total_dissolved_mg: float
    predicted_fa: float  # total_dissolved / dose
    precipitation_risk: str  # none | low | moderate | high
    dose_normalized_dissolution: float  # total_dissolved / dose
    supersaturation_index_jejunum: float
    notes: str


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _ph_solubility(sintr: float, ionization_type: str, pKa: float, pH: float) -> float:
    """Calculate pH-dependent solubility (mg/L) using Henderson-Hasselbalch.

    Parameters
    ----------
    sintr:
        Intrinsic (unionised form) solubility, mg/L.
    ionization_type:
        'acid', 'base', or 'neutral'.
    pKa:
        Acid dissociation constant.
    pH:
        pH of the GI segment.
    """
    if ionization_type == "base":
        return sintr * (1.0 + 10.0 ** (pKa - pH))
    if ionization_type == "acid":
        return sintr * (1.0 + 10.0 ** (pH - pKa))
    # neutral
    return sintr


def _segment_dissolution(
    dose_fraction_mg: float,
    sintr: float,
    ionization_type: str,
    pKa: float,
    segment_pH: float,
    volume_mL: float,
) -> tuple[float, float]:
    """Return (dissolved_mg, precipitated_mg) for one GI segment.

    Parameters
    ----------
    dose_fraction_mg:
        Drug mass entering this segment (mg).
    sintr, ionization_type, pKa:
        Solubility parameters.
    segment_pH:
        pH of the segment.
    volume_mL:
        Fluid volume of the segment (mL).
    """
    s_mg_L = _ph_solubility(sintr, ionization_type, pKa, segment_pH)
    # Capacity = solubility × volume (convert mL → L)
    capacity_mg = s_mg_L * volume_mL / 1000.0
    dissolved_mg = min(dose_fraction_mg, capacity_mg)
    precipitated_mg = dose_fraction_mg - dissolved_mg
    return dissolved_mg, precipitated_mg


def _classify_risk(precipitated_total_mg: float, dose_mg: float) -> str:
    pct = precipitated_total_mg / dose_mg * 100.0
    if pct < 5.0:
        return "none"
    if pct < 20.0:
        return "low"
    if pct < 50.0:
        return "moderate"
    return "high"


# --------------------------------------------------------------------------- #
# Public functions
# --------------------------------------------------------------------------- #


def predict_gi_precipitation(
    drug_name: str,
    dose_mg: float,
    pKa: float,
    ionization_type: str,
    intrinsic_solubility_mg_L: float,
) -> GIPrecipitationResult:
    """Predict GI precipitation for a single drug dose.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Administered dose (mg).
    pKa:
        Ionization constant (used for Henderson-Hasselbalch).
    ionization_type:
        'acid', 'base', or 'neutral'.
    intrinsic_solubility_mg_L:
        Solubility of the unionised form (mg/L).

    Returns
    -------
    GIPrecipitationResult
    """
    # ----- Validation -----
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if intrinsic_solubility_mg_L <= 0.0:
        raise ValueError(f"intrinsic_solubility_mg_L must be > 0, got {intrinsic_solubility_mg_L}")
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION}, got '{ionization_type}'"
        )

    seg_names = ["stomach", "duodenum", "jejunum", "ileum"]
    n_segs = len(seg_names)
    dose_per_seg = dose_mg / n_segs  # equal-fraction distribution

    dissolved_per_seg: dict[str, float] = {}
    precipitated_total = 0.0

    for seg in seg_names:
        info = _GI_SEGMENTS[seg]
        d_mg, p_mg = _segment_dissolution(
            dose_per_seg,
            intrinsic_solubility_mg_L,
            ionization_type,
            pKa,
            info["pH"],
            info["volume_mL"],
        )
        dissolved_per_seg[seg] = d_mg
        precipitated_total += p_mg

    total_dissolved = sum(dissolved_per_seg.values())
    predicted_fa = total_dissolved / dose_mg

    # Supersaturation index at jejunum
    jej_info = _GI_SEGMENTS["jejunum"]
    s_jej = _ph_solubility(intrinsic_solubility_mg_L, ionization_type, pKa, jej_info["pH"])
    conc_jej = dose_per_seg / (jej_info["volume_mL"] / 1000.0)  # mg/L in-segment
    si_jej = conc_jej / s_jej if s_jej > 0.0 else 0.0

    # Clamp predicted_fa to [0, 1]
    predicted_fa = max(0.0, min(1.0, predicted_fa))

    precipitation_risk = _classify_risk(precipitated_total, dose_mg)

    notes = (
        f"Intrinsic solubility {intrinsic_solubility_mg_L:.2f} mg/L, "
        f"pKa {pKa:.2f}, {ionization_type}. "
        f"Dose {dose_mg:.1f} mg distributed equally across {n_segs} segments. "
        f"Total dissolved: {total_dissolved:.2f} mg ({predicted_fa * 100:.1f}% of dose). "
        f"Precipitation risk: {precipitation_risk}. "
        f"Jejunum supersaturation index: {si_jej:.2f}."
    )

    return GIPrecipitationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        pKa=pKa,
        ionization_type=ionization_type,
        intrinsic_solubility_mg_L=intrinsic_solubility_mg_L,
        dissolved_stomach_mg=dissolved_per_seg["stomach"],
        dissolved_duodenum_mg=dissolved_per_seg["duodenum"],
        dissolved_jejunum_mg=dissolved_per_seg["jejunum"],
        dissolved_ileum_mg=dissolved_per_seg["ileum"],
        precipitated_total_mg=precipitated_total,
        total_dissolved_mg=total_dissolved,
        predicted_fa=predicted_fa,
        precipitation_risk=precipitation_risk,
        dose_normalized_dissolution=total_dissolved / dose_mg,
        supersaturation_index_jejunum=si_jej,
        notes=notes,
    )


def screen_dose_range(
    drug_name: str,
    pKa: float,
    ionization_type: str,
    intrinsic_solubility_mg_L: float,
    doses_mg: list,
) -> list:
    """Screen a list of doses and return results sorted by predicted_fa descending.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    pKa:
        Ionization constant.
    ionization_type:
        'acid', 'base', or 'neutral'.
    intrinsic_solubility_mg_L:
        Solubility of the unionised form (mg/L).
    doses_mg:
        List of doses to screen (mg).

    Returns
    -------
    list of GIPrecipitationResult, sorted by predicted_fa descending.
    """
    results = [
        predict_gi_precipitation(drug_name, dose, pKa, ionization_type, intrinsic_solubility_mg_L)
        for dose in doses_mg
    ]
    results.sort(key=lambda r: r.predicted_fa, reverse=True)
    return results
