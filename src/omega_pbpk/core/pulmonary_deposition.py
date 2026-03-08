"""Phase 568 - Pulmonary Drug Deposition Model.

Models drug deposition in lung regions (oropharynx, tracheobronchial,
alveolar) after inhalation based on particle size and device type.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PulmonaryDepositionResult:
    drug_name: str
    particle_size_um: float
    device_type: str
    f_oropharynx: float
    f_tracheobronchial: float
    f_alveolar: float
    f_exhaled: float
    fine_particle_fraction: float
    lung_deposition_total: float
    systemic_absorption_fraction: float
    oropharyngeal_dose_mg: float
    lung_dose_mg: float
    notes: str


_DEVICE_EFFICIENCY = {
    "MDI": 0.2,
    "DPI": 0.35,
    "nebulizer": 0.5,
}


def predict_pulmonary_deposition(
    drug_name: str,
    dose_mg: float,
    particle_size_um: float,
    device_type: str,
    breathing_pattern: str = "normal",
) -> PulmonaryDepositionResult:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if particle_size_um <= 0:
        raise ValueError("particle_size_um must be > 0")
    if device_type not in _DEVICE_EFFICIENCY:
        raise ValueError(f"device_type must be one of {list(_DEVICE_EFFICIENCY.keys())}")

    efficiency = _DEVICE_EFFICIENCY[device_type]

    # Oropharyngeal deposition fraction
    f_oro_raw = 0.05 * (particle_size_um**1.5)
    f_oro = max(0.02, min(0.8, f_oro_raw))

    # Fine particle fraction (< 5 um)
    fpf = max(0.0, min(1.0, 1.0 - (particle_size_um / 10.0) ** 2))

    # Alveolar and tracheobronchial raw fractions
    f_alv_raw = 0.5 * fpf * (1.0 - f_oro) * efficiency
    f_tb_raw = 0.3 * fpf * (1.0 - f_oro) * efficiency

    # Oropharyngeal adjusted
    f_oro_adj = f_oro * efficiency

    # Exhaled
    f_exhaled_raw = 1.0 - f_oro_adj - f_alv_raw - f_tb_raw

    # Normalize to ensure sum = 1.0
    total = f_oro_adj + f_alv_raw + f_tb_raw + f_exhaled_raw
    if total > 0:
        f_oropharynx = f_oro_adj / total
        f_alveolar = f_alv_raw / total
        f_tracheobronchial = f_tb_raw / total
        f_exhaled = f_exhaled_raw / total
    else:
        f_oropharynx = 0.0
        f_alveolar = 0.0
        f_tracheobronchial = 0.0
        f_exhaled = 1.0

    lung_deposition_total = f_tracheobronchial + f_alveolar
    systemic_absorption = f_alveolar * 0.9 + f_tracheobronchial * 0.5

    oropharyngeal_dose_mg = dose_mg * f_oropharynx
    lung_dose_mg = dose_mg * lung_deposition_total

    notes_parts = []
    if particle_size_um > 10:
        notes_parts.append("large particles: mostly oropharyngeal deposition")
    elif particle_size_um < 1:
        notes_parts.append("very fine particles: high exhaled fraction")
    if lung_deposition_total > 0.3:
        notes_parts.append("good lung deposition")
    notes = "; ".join(notes_parts) if notes_parts else "standard deposition profile"

    return PulmonaryDepositionResult(
        drug_name=drug_name,
        particle_size_um=particle_size_um,
        device_type=device_type,
        f_oropharynx=f_oropharynx,
        f_tracheobronchial=f_tracheobronchial,
        f_alveolar=f_alveolar,
        f_exhaled=f_exhaled,
        fine_particle_fraction=fpf,
        lung_deposition_total=lung_deposition_total,
        systemic_absorption_fraction=systemic_absorption,
        oropharyngeal_dose_mg=oropharyngeal_dose_mg,
        lung_dose_mg=lung_dose_mg,
        notes=notes,
    )


def compare_particle_sizes(
    drug_name: str,
    dose_mg: float,
    device_type: str = "MDI",
) -> list:
    sizes = [1, 2, 5, 10, 20]
    results = []
    for size in sizes:
        r = predict_pulmonary_deposition(
            drug_name=drug_name,
            dose_mg=dose_mg,
            particle_size_um=float(size),
            device_type=device_type,
        )
        results.append(r)
    results.sort(key=lambda x: x.lung_deposition_total, reverse=True)
    return results
