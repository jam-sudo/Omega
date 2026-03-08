"""Phase 575 - Drug Precipitation Kinetics in GI Tract.

Models drug precipitation from supersaturated solution in the GI tract,
competing with absorption.
"""

import math
from dataclasses import dataclass


@dataclass
class GIPrecipitationResult:
    """Result of GI precipitation kinetics simulation."""

    drug_name: str
    dose_mg: float
    intrinsic_solubility_mg_mL: float
    gi_volume_mL: float
    times_min: list
    c_dissolved_mg_mL: list
    m_precipitated_mg: list
    dissolved_pct: list
    fa_effective: float
    precipitation_onset_min: float
    t_critical_min: float
    inhibition_factor: float
    notes: str


def simulate_gi_precipitation(
    drug_name: str,
    dose_mg: float,
    intrinsic_solubility_mg_mL: float,
    gi_volume_mL: float = 250.0,
    ka_abs_per_min: float = 0.01,
    k_precip_per_min: float = 0.05,
    polymer_present: bool = False,
    t_end_min: float = 120.0,
    dt_min: float = 0.5,
) -> GIPrecipitationResult:
    """Simulate drug precipitation from supersaturated GI solution.

    Uses Forward Euler integration to model competing absorption and
    precipitation kinetics.
    """
    c_initial = dose_mg / gi_volume_mL
    c_sol = intrinsic_solubility_mg_mL
    inhibition_factor = 0.5 if polymer_present else 1.0

    n_steps = int(t_end_min / dt_min)
    times_min = [i * dt_min for i in range(n_steps + 1)]

    if c_initial <= c_sol:
        # No supersaturation - simple absorption, no precipitation
        c_dissolved = [c_initial * math.exp(-ka_abs_per_min * t) for t in times_min]
        m_precipitated = [0.0] * len(times_min)
        dissolved_pct = [
            (c * gi_volume_mL / dose_mg) * 100.0 if dose_mg > 0 else 0.0 for c in c_dissolved
        ]
        fa_effective = 1.0 - math.exp(-ka_abs_per_min * t_end_min * 60.0)
        fa_effective = max(0.0, min(1.0, fa_effective))

        return GIPrecipitationResult(
            drug_name=drug_name,
            dose_mg=dose_mg,
            intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
            gi_volume_mL=gi_volume_mL,
            times_min=times_min,
            c_dissolved_mg_mL=c_dissolved,
            m_precipitated_mg=m_precipitated,
            dissolved_pct=dissolved_pct,
            fa_effective=fa_effective,
            precipitation_onset_min=float("inf"),
            t_critical_min=float("inf"),
            inhibition_factor=inhibition_factor,
            notes="Below solubility - no precipitation expected",
        )

    # Supersaturated case - Forward Euler
    k_p_eff = k_precip_per_min * inhibition_factor

    c_dissolved = [0.0] * len(times_min)
    m_precipitated = [0.0] * len(times_min)

    c_dissolved[0] = c_initial
    m_precipitated[0] = 0.0

    for i in range(n_steps):
        c = c_dissolved[i]
        m_p = m_precipitated[i]

        supersaturation = max(0.0, c - c_sol)
        dc_dt = -ka_abs_per_min * c - k_p_eff * supersaturation
        dm_dt = k_p_eff * supersaturation * gi_volume_mL

        c_new = c + dc_dt * dt_min
        m_new = m_p + dm_dt * dt_min

        c_dissolved[i + 1] = max(0.0, c_new)
        m_precipitated[i + 1] = max(0.0, m_new)

    dissolved_pct = [
        (c * gi_volume_mL / dose_mg) * 100.0 if dose_mg > 0 else 0.0 for c in c_dissolved
    ]

    m_precip_final = m_precipitated[-1]
    fa_effective = max(0.0, min(1.0, (dose_mg - m_precip_final) / dose_mg))

    # Find precipitation onset (first time M_precip > 1% of dose)
    precipitation_onset_min = float("inf")
    threshold_onset = 0.01 * dose_mg
    for i, m_p in enumerate(m_precipitated):
        if m_p > threshold_onset:
            precipitation_onset_min = times_min[i]
            break

    # Find t_critical (first time M_precip > 50% of dose)
    t_critical_min = float("inf")
    threshold_critical = 0.5 * dose_mg
    for i, m_p in enumerate(m_precipitated):
        if m_p > threshold_critical:
            t_critical_min = times_min[i]
            break

    return GIPrecipitationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        gi_volume_mL=gi_volume_mL,
        times_min=times_min,
        c_dissolved_mg_mL=c_dissolved,
        m_precipitated_mg=m_precipitated,
        dissolved_pct=dissolved_pct,
        fa_effective=fa_effective,
        precipitation_onset_min=precipitation_onset_min,
        t_critical_min=t_critical_min,
        inhibition_factor=inhibition_factor,
        notes="Supersaturated - precipitation kinetics modeled",
    )


def compare_polymer_effect(
    drug_name: str,
    dose_mg: float,
    solubility: float,
) -> list:
    """Compare precipitation with and without polymer inhibitor.

    Returns list [no_polymer_result, polymer_result] sorted by
    fa_effective descending.
    """
    no_polymer = simulate_gi_precipitation(
        drug_name=drug_name,
        dose_mg=dose_mg,
        intrinsic_solubility_mg_mL=solubility,
        polymer_present=False,
    )
    with_polymer = simulate_gi_precipitation(
        drug_name=drug_name,
        dose_mg=dose_mg,
        intrinsic_solubility_mg_mL=solubility,
        polymer_present=True,
    )
    results = [no_polymer, with_polymer]
    results.sort(key=lambda r: r.fa_effective, reverse=True)
    return results
