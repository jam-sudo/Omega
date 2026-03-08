"""Phase 393 - Drug Absorption Window Simulator.

Simulates the time available for drug absorption in each GI segment
accounting for transit time, dissolution, and pH-dependent solubility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class AbsorptionWindowResult:
    """Results from GI absorption window simulation."""

    drug_name: str
    dose_mg: float
    segments: list = field(default_factory=list)
    transit_times_h: list = field(default_factory=list)
    dissolved_fractions: list = field(default_factory=list)
    absorbed_fractions: list = field(default_factory=list)
    cumulative_absorbed_pct: float = 0.0
    absorption_complete_segment: str = "incomplete"
    rate_limiting_step: str = "transit"
    bcs_class: str = "I"
    fa_predicted: float = 0.0
    notes: str = ""


_SEGMENTS = {
    "stomach": {"ph": 1.5, "volume_mL": 250, "transit_h": 0.5, "surface_cm2": 100},
    "duodenum": {"ph": 6.0, "volume_mL": 100, "transit_h": 0.08, "surface_cm2": 800},
    "jejunum": {"ph": 6.5, "volume_mL": 200, "transit_h": 1.5, "surface_cm2": 100000},
    "ileum": {"ph": 7.2, "volume_mL": 200, "transit_h": 2.0, "surface_cm2": 80000},
    "colon": {"ph": 7.4, "volume_mL": 300, "transit_h": 16.0, "surface_cm2": 2000},
}

_SEGMENT_ORDER = ["stomach", "duodenum", "jejunum", "ileum", "colon"]


def _ph_adjusted_solubility(
    solubility_mg_mL: float,
    pka: float,
    ph: float,
    drug_type: str,
) -> float:
    """Compute pH-adjusted solubility using Henderson-Hasselbalch equation."""
    if drug_type == "acid":
        s_eff = solubility_mg_mL * (1.0 + 10.0 ** (ph - pka))
    elif drug_type == "base":
        s_eff = solubility_mg_mL * (1.0 + 10.0 ** (pka - ph))
    else:
        s_eff = solubility_mg_mL
    # Cap at 1000x intrinsic solubility
    return min(s_eff, 1000.0 * solubility_mg_mL)


def _classify_bcs(
    dose_mg: float,
    solubility_mg_mL: float,
    permeability_cm_s: float,
) -> str:
    """Classify drug into BCS class I-IV."""
    do = dose_mg / (solubility_mg_mL * 250.0)
    pn = permeability_cm_s / 2.0e-4
    if do <= 1.0 and pn >= 1.0:
        return "I"
    elif do > 1.0 and pn >= 1.0:
        return "II"
    elif do <= 1.0 and pn < 1.0:
        return "III"
    else:
        return "IV"


def _rate_limiting_step(bcs_class: str) -> str:
    """Determine rate-limiting step from BCS class."""
    mapping = {
        "I": "transit",
        "II": "dissolution",
        "III": "permeability",
        "IV": "solubility",
    }
    return mapping[bcs_class]


def simulate_absorption_window(
    drug_name: str,
    dose_mg: float,
    pka: float,
    logp: float,
    mw_da: float,
    solubility_mg_mL: float,
    permeability_cm_s: float,
    drug_type: str = "neutral",
) -> AbsorptionWindowResult:
    """Simulate drug absorption window across GI segments.

    Args:
        drug_name: Name of the drug.
        dose_mg: Oral dose in mg.
        pka: Acid/base dissociation constant.
        logp: Log octanol-water partition coefficient.
        mw_da: Molecular weight in Daltons.
        solubility_mg_mL: Aqueous solubility in mg/mL.
        permeability_cm_s: Effective intestinal permeability in cm/s.
        drug_type: Ionization type - "acid", "base", or "neutral".

    Returns:
        AbsorptionWindowResult with segment-by-segment absorption data.

    Raises:
        ValueError: If invalid parameters are provided.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if solubility_mg_mL <= 0:
        raise ValueError(f"solubility_mg_mL must be > 0, got {solubility_mg_mL}")
    if permeability_cm_s <= 0:
        raise ValueError(f"permeability_cm_s must be > 0, got {permeability_cm_s}")
    if mw_da <= 0:
        raise ValueError(f"mw_da must be > 0, got {mw_da}")

    # BCS classification
    bcs_class = _classify_bcs(dose_mg, solubility_mg_mL, permeability_cm_s)
    rate_limit = _rate_limiting_step(bcs_class)

    # Track drug state through GI tract
    undissolved_mg = dose_mg
    dissolved_mg = 0.0
    total_absorbed_mg = 0.0

    segment_names = []
    transit_times_h = []
    dissolved_fractions_list = []
    absorbed_fractions_list = []

    absorption_complete_segment = "incomplete"

    for seg_name in _SEGMENT_ORDER:
        seg = _SEGMENTS[seg_name]
        ph = seg["ph"]
        volume_mL = seg["volume_mL"]
        transit_h = seg["transit_h"]
        surface_cm2 = seg["surface_cm2"]

        # pH-adjusted solubility
        s_eff = _ph_adjusted_solubility(solubility_mg_mL, pka, ph, drug_type)

        # Max dissolvable in this segment
        m_dissolve_max = s_eff * volume_mL

        # Dissolution in this segment
        if undissolved_mg <= m_dissolve_max:
            newly_dissolved = undissolved_mg
            undissolved_mg = 0.0
        else:
            newly_dissolved = m_dissolve_max
            undissolved_mg -= m_dissolve_max

        dissolved_in_seg = dissolved_mg + newly_dissolved

        # Absorption rate constant for this segment (1/h)
        ka_seg = permeability_cm_s * surface_cm2 * 3600.0 / volume_mL

        # Fraction absorbed from dissolved drug in this segment
        fa_seg = 1.0 - math.exp(-ka_seg * transit_h)

        # Amount absorbed from this segment
        absorbed_from_seg = dissolved_in_seg * fa_seg

        # Carry forward unabsorbed dissolved drug
        dissolved_mg = dissolved_in_seg - absorbed_from_seg

        total_absorbed_mg += absorbed_from_seg

        # Track fractions
        dissolved_fraction = min(1.0, (dose_mg - undissolved_mg) / dose_mg)
        absorbed_fraction = min(1.0, absorbed_from_seg / dose_mg)

        segment_names.append(seg_name)
        transit_times_h.append(transit_h)
        dissolved_fractions_list.append(dissolved_fraction)
        absorbed_fractions_list.append(absorbed_fraction)

        # Check if 90% absorption achieved
        cumulative_pct = min(100.0, total_absorbed_mg / dose_mg * 100.0)
        if cumulative_pct >= 90.0 and absorption_complete_segment == "incomplete":
            absorption_complete_segment = seg_name

    cumulative_absorbed_pct = min(100.0, total_absorbed_mg / dose_mg * 100.0)
    fa_predicted = cumulative_absorbed_pct / 100.0

    notes_parts = [
        f"BCS Class {bcs_class}.",
        f"Rate-limiting step: {rate_limit}.",
        f"Predicted fa: {fa_predicted:.3f}.",
        f"Absorption complete by: {absorption_complete_segment}.",
    ]
    notes = " ".join(notes_parts)

    return AbsorptionWindowResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        segments=segment_names,
        transit_times_h=transit_times_h,
        dissolved_fractions=dissolved_fractions_list,
        absorbed_fractions=absorbed_fractions_list,
        cumulative_absorbed_pct=cumulative_absorbed_pct,
        absorption_complete_segment=absorption_complete_segment,
        rate_limiting_step=rate_limit,
        bcs_class=bcs_class,
        fa_predicted=fa_predicted,
        notes=notes,
    )


def compare_formulation_solubilities(
    drug_name: str,
    dose_mg: float,
    pka: float,
    logp: float,
    mw_da: float,
    permeability_cm_s: float,
    solubility_values_mg_mL: list = None,
) -> list:
    """Compare absorption across different formulation solubilities.

    Args:
        drug_name: Name of the drug.
        dose_mg: Oral dose in mg.
        pka: pKa value.
        logp: Log octanol-water partition coefficient.
        mw_da: Molecular weight in Daltons.
        permeability_cm_s: Effective intestinal permeability in cm/s.
        solubility_values_mg_mL: List of solubility values to compare.
            Defaults to [0.001, 0.01, 0.1, 1.0, 10.0].

    Returns:
        List of AbsorptionWindowResult objects sorted by fa_predicted descending.
    """
    if solubility_values_mg_mL is None:
        solubility_values_mg_mL = [0.001, 0.01, 0.1, 1.0, 10.0]

    results = []
    for sol in solubility_values_mg_mL:
        result = simulate_absorption_window(
            drug_name=drug_name,
            dose_mg=dose_mg,
            pka=pka,
            logp=logp,
            mw_da=mw_da,
            solubility_mg_mL=sol,
            permeability_cm_s=permeability_cm_s,
        )
        results.append(result)

    results.sort(key=lambda r: r.fa_predicted, reverse=True)
    return results
