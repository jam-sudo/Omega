"""Phase 277 — Pulmonary Drug Deposition Predictor.

Predicts regional lung deposition (extrathoracic, tracheobronchial, alveolar)
based on particle aerodynamics using ICRP 1994-style empirical equations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PulmonaryDepositionResult:
    """Result of pulmonary deposition prediction."""

    drug_name: str
    mmad_um: float
    gsd: float
    flow_rate_L_min: float
    extrathoracic_fraction: float
    tracheobronchial_fraction: float
    alveolar_fraction: float
    total_deposited_fraction: float
    fine_particle_fraction: float
    impaction_parameter: float
    deposition_class: str
    notes: str


def _validate_inputs(
    mmad_um: float,
    gsd: float,
    flow_rate_L_min: float,
    breathing_frequency_per_min: float,
    tidal_volume_mL: float,
) -> None:
    """Validate input parameters."""
    if mmad_um <= 0:
        raise ValueError(f"mmad_um must be > 0, got {mmad_um}")
    if gsd < 1.0:
        raise ValueError(f"gsd must be >= 1.0, got {gsd}")
    if flow_rate_L_min <= 0:
        raise ValueError(f"flow_rate_L_min must be > 0, got {flow_rate_L_min}")
    if breathing_frequency_per_min <= 0:
        raise ValueError(
            f"breathing_frequency_per_min must be > 0, got {breathing_frequency_per_min}"
        )
    if tidal_volume_mL <= 0:
        raise ValueError(f"tidal_volume_mL must be > 0, got {tidal_volume_mL}")


def predict_pulmonary_deposition(
    drug_name: str,
    mmad_um: float,
    gsd: float,
    flow_rate_L_min: float,
    breathing_frequency_per_min: float,
    tidal_volume_mL: float,
) -> PulmonaryDepositionResult:
    """Predict regional lung deposition fractions.

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    mmad_um : float
        Mass median aerodynamic diameter in micrometers.
    gsd : float
        Geometric standard deviation of particle size distribution (>= 1.0).
    flow_rate_L_min : float
        Inhalation flow rate in L/min.
    breathing_frequency_per_min : float
        Breathing frequency in breaths per minute.
    tidal_volume_mL : float
        Tidal volume in mL.

    Returns
    -------
    PulmonaryDepositionResult
        Predicted deposition fractions and classification.
    """
    _validate_inputs(mmad_um, gsd, flow_rate_L_min, breathing_frequency_per_min, tidal_volume_mL)

    # Impaction parameter: IP = flow_rate * mmad^2 (L/min * um^2)
    ip = flow_rate_L_min * (mmad_um**2)

    # Extrathoracic fraction (ICRP-style)
    if ip > 0:
        et = min(0.95, 0.0587 * math.sqrt(ip))
    else:
        et = 0.0

    # Tracheobronchial fraction
    tb = max(0.0, 0.2 * (1.0 - et) * math.exp(-0.05 * mmad_um))

    # Alveolar fraction
    al = max(0.0, (1.0 - et - tb) * math.exp(-0.1 * mmad_um))

    # Total deposited fraction (capped at 1.0)
    total = min(1.0, et + tb + al)

    # Fine particle fraction: approximate lognormal fraction below 5 um
    fpf = max(0.0, 1.0 - mmad_um / 10.0)

    # Determine deposition class based on dominant fraction
    fractions = {
        "upper_airway": et,
        "bronchial": tb,
        "alveolar": al,
    }
    deposition_class = max(fractions, key=lambda k: fractions[k])

    # Build notes
    notes_parts = []
    if mmad_um < 1.0:
        notes_parts.append("Very fine particles may be exhaled")
    elif mmad_um > 10.0:
        notes_parts.append("Coarse particles dominate upper airway deposition")
    if gsd > 2.5:
        notes_parts.append("Broad size distribution (polydisperse aerosol)")
    if not notes_parts:
        notes_parts.append("Standard aerosol deposition profile")
    notes = "; ".join(notes_parts)

    return PulmonaryDepositionResult(
        drug_name=drug_name,
        mmad_um=mmad_um,
        gsd=gsd,
        flow_rate_L_min=flow_rate_L_min,
        extrathoracic_fraction=et,
        tracheobronchial_fraction=tb,
        alveolar_fraction=al,
        total_deposited_fraction=total,
        fine_particle_fraction=fpf,
        impaction_parameter=ip,
        deposition_class=deposition_class,
        notes=notes,
    )


def compare_formulations(
    drug_name: str,
    mmad_list: list[float],
    gsd_list: list[float],
    flow_rate_L_min: float = 30.0,
    breathing_frequency_per_min: float = 15.0,
    tidal_volume_mL: float = 500.0,
) -> list[PulmonaryDepositionResult]:
    """Compare multiple formulations and rank by alveolar deposition.

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    mmad_list : list of float
        List of MMAD values (um) for each formulation.
    gsd_list : list of float
        List of GSD values for each formulation.
    flow_rate_L_min : float
        Inhalation flow rate in L/min (default 30).
    breathing_frequency_per_min : float
        Breathing frequency in breaths/min (default 15).
    tidal_volume_mL : float
        Tidal volume in mL (default 500).

    Returns
    -------
    list of PulmonaryDepositionResult
        Results sorted by alveolar_fraction descending.
    """
    if len(mmad_list) != len(gsd_list):
        raise ValueError("mmad_list and gsd_list must have the same length")

    results = []
    for mmad, gsd in zip(mmad_list, gsd_list):
        result = predict_pulmonary_deposition(
            drug_name=drug_name,
            mmad_um=mmad,
            gsd=gsd,
            flow_rate_L_min=flow_rate_L_min,
            breathing_frequency_per_min=breathing_frequency_per_min,
            tidal_volume_mL=tidal_volume_mL,
        )
        results.append(result)

    results.sort(key=lambda r: r.alveolar_fraction, reverse=True)
    return results
