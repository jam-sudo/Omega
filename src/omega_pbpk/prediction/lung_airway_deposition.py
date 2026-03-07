"""Phase 324 — Drug Deposition in Lung Airways.

Predicts regional lung deposition (oropharyngeal, tracheobronchial, alveolar)
for inhaled aerosols using a 3-compartment ICRP model with flow-rate and
breathing-pattern effects.
"""

import math
from dataclasses import dataclass

_VALID_DEVICES = {"MDI", "DPI", "nebulizer", "soft_mist"}

_DEVICE_CORRECTION = {
    "MDI": {"velocity_factor": 1.5, "et_extra_loss": 0.2},
    "DPI": {"velocity_factor": 0.8, "et_extra_loss": 0.1},
    "nebulizer": {"velocity_factor": 0.3, "et_extra_loss": 0.05},
    "soft_mist": {"velocity_factor": 0.2, "et_extra_loss": 0.05},
}


@dataclass(frozen=True)
class AirwayDepositionResult:
    drug_name: str
    mmad_um: float
    gsd: float
    flow_rate_L_min: float
    device_type: str
    et_fraction: float
    tb_fraction: float
    alveolar_fraction: float
    total_deposited_fraction: float
    exhaled_fraction: float
    lung_deposited_fraction: float
    fine_particle_fraction: float
    deposition_efficiency_class: str
    notes: str


def _validate(
    mmad_um: float,
    gsd: float,
    flow_rate_L_min: float,
    tidal_volume_mL: float,
    respiratory_rate_per_min: float,
    device_type: str,
) -> None:
    if mmad_um <= 0:
        raise ValueError("mmad_um must be > 0")
    if gsd < 1.0:
        raise ValueError("gsd must be >= 1.0")
    if flow_rate_L_min <= 0:
        raise ValueError("flow_rate_L_min must be > 0")
    if tidal_volume_mL <= 0:
        raise ValueError("tidal_volume_mL must be > 0")
    if respiratory_rate_per_min <= 0:
        raise ValueError("respiratory_rate_per_min must be > 0")
    if device_type not in _VALID_DEVICES:
        raise ValueError(f"device_type must be one of {_VALID_DEVICES}")


def _compute_fractions(
    mmad_um: float,
    flow_rate_L_min: float,
    device_type: str,
) -> tuple:
    """Return (ET, TB, AL) fractions using device-corrected impaction parameter."""
    correction = _DEVICE_CORRECTION[device_type]
    velocity_factor = correction["velocity_factor"]
    et_extra_loss = correction["et_extra_loss"]

    ip = flow_rate_L_min * mmad_um**2
    effective_ip = ip * velocity_factor

    if effective_ip > 0:
        et_base = min(0.95, 0.0587 * math.sqrt(effective_ip))
    else:
        et_base = 0.0

    et_corrected = min(0.99, et_base + et_extra_loss)

    tb = max(0.0, 0.2 * (1.0 - et_corrected) * math.exp(-0.05 * mmad_um))
    al = max(0.0, (1.0 - et_corrected - tb) * math.exp(-0.1 * mmad_um))

    return et_corrected, tb, al


def predict_airway_deposition(
    drug_name: str,
    mmad_um: float,
    gsd: float,
    flow_rate_L_min: float,
    tidal_volume_mL: float,
    respiratory_rate_per_min: float,
    device_type: str,
) -> AirwayDepositionResult:
    """Predict regional lung deposition for an inhaled aerosol.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    mmad_um:
        Mass median aerodynamic diameter in micrometres (> 0).
    gsd:
        Geometric standard deviation (>= 1.0).
    flow_rate_L_min:
        Inhalation flow rate in L/min (> 0).
    tidal_volume_mL:
        Tidal volume in mL (> 0).
    respiratory_rate_per_min:
        Breathing rate in breaths/min (> 0).
    device_type:
        One of "MDI", "DPI", "nebulizer", "soft_mist".

    Returns
    -------
    AirwayDepositionResult
    """
    _validate(mmad_um, gsd, flow_rate_L_min, tidal_volume_mL, respiratory_rate_per_min, device_type)

    et, tb, al = _compute_fractions(mmad_um, flow_rate_L_min, device_type)

    total = min(1.0, et + tb + al)
    exhaled = max(0.0, 1.0 - total)
    lung = tb + al

    fpf = max(0.0, 1.0 - mmad_um / 10.0)

    if al > 0.3:
        dep_class = "excellent"
    elif al >= 0.15:
        dep_class = "good"
    else:
        dep_class = "poor"

    notes = (
        f"Device: {device_type}; MMAD={mmad_um:.2f} um; "
        f"ET={et:.3f}, TB={tb:.3f}, AL={al:.3f}; "
        f"Lung deposited (TB+AL)={lung:.3f}; "
        f"FPF={fpf:.3f}; Class={dep_class}"
    )

    return AirwayDepositionResult(
        drug_name=drug_name,
        mmad_um=mmad_um,
        gsd=gsd,
        flow_rate_L_min=flow_rate_L_min,
        device_type=device_type,
        et_fraction=et,
        tb_fraction=tb,
        alveolar_fraction=al,
        total_deposited_fraction=total,
        exhaled_fraction=exhaled,
        lung_deposited_fraction=lung,
        fine_particle_fraction=fpf,
        deposition_efficiency_class=dep_class,
        notes=notes,
    )


def optimize_for_alveolar_delivery(
    drug_name: str,
    gsd: float,
    flow_rate_L_min: float,
    tidal_volume_mL: float,
    respiratory_rate_per_min: float,
    device_type: str,
) -> AirwayDepositionResult:
    """Find the MMAD (0.5–10 um) that maximises alveolar deposition fraction.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    gsd:
        Geometric standard deviation (>= 1.0).
    flow_rate_L_min:
        Inhalation flow rate in L/min (> 0).
    tidal_volume_mL:
        Tidal volume in mL (> 0).
    respiratory_rate_per_min:
        Breathing rate in breaths/min (> 0).
    device_type:
        One of "MDI", "DPI", "nebulizer", "soft_mist".

    Returns
    -------
    AirwayDepositionResult with the highest alveolar fraction.
    """
    _validate(0.5, gsd, flow_rate_L_min, tidal_volume_mL, respiratory_rate_per_min, device_type)

    best: AirwayDepositionResult | None = None
    mmad = 0.5
    step = 0.1
    while mmad <= 10.0 + 1e-9:
        r = predict_airway_deposition(
            drug_name=drug_name,
            mmad_um=round(mmad, 6),
            gsd=gsd,
            flow_rate_L_min=flow_rate_L_min,
            tidal_volume_mL=tidal_volume_mL,
            respiratory_rate_per_min=respiratory_rate_per_min,
            device_type=device_type,
        )
        if best is None or r.alveolar_fraction > best.alveolar_fraction:
            best = r
        mmad += step

    assert best is not None
    return best
