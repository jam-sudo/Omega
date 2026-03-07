"""Pulmonary drug delivery dosimetry for inhaled therapeutics.

Models regional deposition of inhaled aerosols into:
- Oropharyngeal (swallowed)
- Conducting airways (bronchi/bronchioles — local therapeutic effect)
- Alveolar region (systemic absorption)
- Exhaled fraction

Supports MDI, DPI, nebulizer, and soft-mist devices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

_VALID_DEVICES = {"MDI", "DPI", "nebulizer", "soft_mist"}

_DEVICE_PARAMS: dict[str, dict[str, float]] = {
    "MDI":        {"device_efficiency": 0.80, "oropharyngeal_fraction": 0.50},
    "DPI":        {"device_efficiency": 0.70, "oropharyngeal_fraction": 0.35},
    "nebulizer":  {"device_efficiency": 0.60, "oropharyngeal_fraction": 0.15},
    "soft_mist":  {"device_efficiency": 0.85, "oropharyngeal_fraction": 0.20},
}


@dataclass(frozen=True)
class PulmonaryDosimetryResult:
    """Regional dosimetry result for an inhaled drug formulation."""

    drug_name: str
    device_type: str  # "MDI", "DPI", "nebulizer", "soft_mist"
    nominal_dose_mg: float

    # Emitted dose characteristics
    emitted_dose_mg: float
    fine_particle_dose_mg: float
    fine_particle_fraction: float

    # Regional deposition (mg)
    m_device_loss_mg: float
    m_oropharyngeal_mg: float
    m_conducting_airway_mg: float
    m_alveolar_mg: float
    m_exhaled_mg: float

    # Fractions of nominal dose
    f_lung_total: float
    f_alveolar: float
    f_swallowed: float

    # Systemic exposure
    systemic_dose_mg: float
    f_systemic: float

    # Efficacy metric
    therapeutic_index_local: float

    notes: str


def simulate_pulmonary_dosimetry(
    drug_name: str,
    nominal_dose_mg: float,
    device_type: str = "MDI",
    mmad_um: float = 2.0,
    gsd: float = 2.0,
    inhalation_flow_L_per_min: float = 30.0,
    lung_volume_L: float = 3.0,
    f_oral_bioavailability: float = 0.20,
    f_alveolar_systemic: float = 0.90,
) -> PulmonaryDosimetryResult:
    """Simulate regional pulmonary dosimetry for an inhaled drug.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    nominal_dose_mg:
        Nominal (labeled) dose per actuation or inhalation (mg).
    device_type:
        Inhaler device type: "MDI", "DPI", "nebulizer", or "soft_mist".
    mmad_um:
        Mass median aerodynamic diameter (µm).
    gsd:
        Geometric standard deviation of particle size distribution (> 1).
    inhalation_flow_L_per_min:
        Patient inhalation flow rate (L/min).
    lung_volume_L:
        Functional residual capacity + tidal volume (L).
    f_oral_bioavailability:
        Oral bioavailability of drug swallowed from oropharynx (fraction 0-1).
    f_alveolar_systemic:
        Fraction of alveolar-deposited drug absorbed into systemic circulation.

    Returns
    -------
    PulmonaryDosimetryResult
    """
    # --- Input validation ---
    if nominal_dose_mg <= 0:
        raise ValueError(f"nominal_dose_mg must be > 0, got {nominal_dose_mg}")
    if mmad_um <= 0:
        raise ValueError(f"mmad_um must be > 0, got {mmad_um}")
    if gsd <= 1:
        raise ValueError(f"gsd must be > 1, got {gsd}")
    if inhalation_flow_L_per_min <= 0:
        raise ValueError(
            f"inhalation_flow_L_per_min must be > 0, got {inhalation_flow_L_per_min}"
        )
    if device_type not in _VALID_DEVICES:
        raise ValueError(
            f"device_type must be one of {_VALID_DEVICES}, got '{device_type}'"
        )

    params = _DEVICE_PARAMS[device_type]
    device_efficiency: float = params["device_efficiency"]
    oropharyngeal_fraction: float = params["oropharyngeal_fraction"]

    # --- Emitted dose ---
    emitted_dose_mg = nominal_dose_mg * device_efficiency
    m_device_loss_mg = nominal_dose_mg - emitted_dose_mg

    # --- Fine particle fraction (log-normal below 5 µm) ---
    ln_5 = np.log(5.0)
    ln_mmad = np.log(mmad_um)
    ln_gsd = np.log(gsd)
    fpf = float(norm.cdf((ln_5 - ln_mmad) / ln_gsd))
    fine_particle_dose_mg = emitted_dose_mg * fpf

    # --- Regional deposition ---
    m_oropharyngeal_mg = emitted_dose_mg * oropharyngeal_fraction

    # Lung deposition: fine particles that are NOT intercepted in the oropharynx.
    # Apply partial overlap correction (oropharyngeal captures some fine particles too).
    m_lung_uncorrected = fine_particle_dose_mg * (1.0 - oropharyngeal_fraction * 0.3)

    # Mass available for lung + exhaled = emitted - oropharyngeal
    m_available_for_lung = max(0.0, emitted_dose_mg - m_oropharyngeal_mg)

    # Cap lung deposition to available mass (enforces mass conservation)
    m_lung_mg = min(m_lung_uncorrected, m_available_for_lung)

    m_exhaled_mg = max(0.0, m_available_for_lung - m_lung_mg)

    # --- Regional split within lung ---
    # Smaller particles (< 1 µm) have higher alveolar deposition
    fraction_alveolar = max(0.30, 0.60 - (mmad_um - 1.0) * 0.10)
    fraction_conducting = 1.0 - fraction_alveolar

    m_alveolar_mg = m_lung_mg * fraction_alveolar
    m_conducting_airway_mg = m_lung_mg * fraction_conducting

    # --- Fractions of nominal dose ---
    f_lung_total = (m_conducting_airway_mg + m_alveolar_mg) / nominal_dose_mg
    f_alveolar = m_alveolar_mg / nominal_dose_mg
    f_swallowed = m_oropharyngeal_mg / nominal_dose_mg

    # --- Systemic exposure ---
    systemic_from_alveolar = m_alveolar_mg * f_alveolar_systemic
    systemic_from_swallowed = m_oropharyngeal_mg * f_oral_bioavailability
    systemic_dose_mg = systemic_from_alveolar + systemic_from_swallowed
    f_systemic = systemic_dose_mg / nominal_dose_mg

    # --- Efficacy metric (local, conducting airways) ---
    therapeutic_index_local = m_conducting_airway_mg / nominal_dose_mg

    notes = (
        f"{drug_name} | {device_type} | MMAD={mmad_um} µm | GSD={gsd:.2f} | "
        f"FPF={fpf:.1%} | lung={f_lung_total:.1%} | systemic={f_systemic:.1%}"
    )

    return PulmonaryDosimetryResult(
        drug_name=drug_name,
        device_type=device_type,
        nominal_dose_mg=nominal_dose_mg,
        emitted_dose_mg=emitted_dose_mg,
        fine_particle_dose_mg=fine_particle_dose_mg,
        fine_particle_fraction=fpf,
        m_device_loss_mg=m_device_loss_mg,
        m_oropharyngeal_mg=m_oropharyngeal_mg,
        m_conducting_airway_mg=m_conducting_airway_mg,
        m_alveolar_mg=m_alveolar_mg,
        m_exhaled_mg=m_exhaled_mg,
        f_lung_total=f_lung_total,
        f_alveolar=f_alveolar,
        f_swallowed=f_swallowed,
        systemic_dose_mg=systemic_dose_mg,
        f_systemic=f_systemic,
        therapeutic_index_local=therapeutic_index_local,
        notes=notes,
    )


def compare_devices(
    drug_name: str,
    nominal_dose_mg: float,
    **kwargs,
) -> dict:
    """Simulate all 4 device types and compare pulmonary dosimetry.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    nominal_dose_mg:
        Nominal dose (mg).
    **kwargs:
        Additional keyword arguments passed to :func:`simulate_pulmonary_dosimetry`.

    Returns
    -------
    dict
        Keys are device names ("MDI", "DPI", "nebulizer", "soft_mist") mapping to
        :class:`PulmonaryDosimetryResult`, plus "best_lung_delivery" (device name
        with highest ``f_lung_total``).
    """
    results: dict[str, PulmonaryDosimetryResult] = {}
    for device in _VALID_DEVICES:
        results[device] = simulate_pulmonary_dosimetry(
            drug_name=drug_name,
            nominal_dose_mg=nominal_dose_mg,
            device_type=device,
            **kwargs,
        )

    best_device = max(results, key=lambda d: results[d].f_lung_total)
    return {**results, "best_lung_delivery": best_device}
