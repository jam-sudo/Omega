"""Phase 323 — Tablet Disintegration Time Predictor.

Predicts tablet disintegration time based on formulation composition and
compression force. Simulates water uptake and swelling kinetics.
"""

from dataclasses import dataclass

_VALID_DISINTEGRANTS = {"SSG", "CCS", "Kollidon_CL", "starch"}
_VALID_BINDERS = {"HPMC", "PVP", "MCC", "starch_binder"}

_DIS_EFFICACY = {"SSG": 0.8, "CCS": 0.9, "Kollidon_CL": 0.7, "starch": 0.4}

_BINDER_RESISTANCE = {"HPMC": 1.5, "PVP": 1.2, "MCC": 0.8, "starch_binder": 1.0}


@dataclass(frozen=True)
class TabletDisintegrationResult:
    drug_name: str
    tablet_weight_mg: float
    compression_force_kN: float
    disintegrant_pct: float
    disintegrant_type: str
    binder_pct: float
    binder_type: str
    porosity_pct: float
    disintegration_time_min: float
    water_uptake_rate_mg_per_min: float
    meets_usp_standard: bool
    disintegration_class: str
    notes: str


def _validate(
    tablet_weight_mg: float,
    compression_force_kN: float,
    disintegrant_pct: float,
    binder_pct: float,
    porosity_pct: float,
    disintegrant_type: str,
    binder_type: str,
) -> None:
    if tablet_weight_mg <= 0:
        raise ValueError("tablet_weight_mg must be > 0")
    if compression_force_kN <= 0:
        raise ValueError("compression_force_kN must be > 0")
    if not (0 < disintegrant_pct <= 20):
        raise ValueError("disintegrant_pct must be in (0, 20]")
    if not (0 < binder_pct <= 30):
        raise ValueError("binder_pct must be in (0, 30]")
    if not (0 < porosity_pct < 60):
        raise ValueError("porosity_pct must be in (0, 60)")
    if disintegrant_type not in _VALID_DISINTEGRANTS:
        raise ValueError(f"disintegrant_type must be one of {_VALID_DISINTEGRANTS}")
    if binder_type not in _VALID_BINDERS:
        raise ValueError(f"binder_type must be one of {_VALID_BINDERS}")


def _compute_disintegration_time(
    compression_force_kN: float,
    porosity_pct: float,
    disintegrant_pct: float,
    disintegrant_type: str,
    binder_type: str,
) -> float:
    cf = 1.0 + compression_force_kN / 10.0
    raw_pf = 2.0 - porosity_pct / 50.0
    pf = max(0.5, min(3.0, raw_pf))
    binder_resistance = _BINDER_RESISTANCE[binder_type]
    efficacy = _DIS_EFFICACY[disintegrant_type]
    t = cf * pf * binder_resistance / (efficacy * disintegrant_pct + 0.001)
    return max(0.1, min(120.0, t))


def predict_disintegration_time(
    drug_name: str,
    tablet_weight_mg: float,
    compression_force_kN: float,
    disintegrant_pct: float,
    disintegrant_type: str,
    binder_pct: float,
    binder_type: str,
    porosity_pct: float,
) -> TabletDisintegrationResult:
    """Predict tablet disintegration time from formulation parameters.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    tablet_weight_mg:
        Total tablet weight in mg (> 0).
    compression_force_kN:
        Compression force applied during tableting in kN (> 0).
    disintegrant_pct:
        Percentage of disintegrant in tablet (0, 20].
    disintegrant_type:
        One of "SSG", "CCS", "Kollidon_CL", "starch".
    binder_pct:
        Percentage of binder in tablet (0, 30].
    binder_type:
        One of "HPMC", "PVP", "MCC", "starch_binder".
    porosity_pct:
        Tablet porosity percentage (0, 60).

    Returns
    -------
    TabletDisintegrationResult
    """
    _validate(
        tablet_weight_mg,
        compression_force_kN,
        disintegrant_pct,
        binder_pct,
        porosity_pct,
        disintegrant_type,
        binder_type,
    )

    dis_time = _compute_disintegration_time(
        compression_force_kN,
        porosity_pct,
        disintegrant_pct,
        disintegrant_type,
        binder_type,
    )

    water_uptake = tablet_weight_mg * porosity_pct / 100.0 * 2.0

    meets_usp = dis_time < 15.0

    if dis_time < 5.0:
        dis_class = "rapid"
    elif dis_time <= 15.0:
        dis_class = "standard"
    else:
        dis_class = "slow"

    note_parts = [
        f"Disintegrant: {disintegrant_type} {disintegrant_pct:.1f}%",
        f"Binder: {binder_type} {binder_pct:.1f}%",
        f"Porosity: {porosity_pct:.1f}%",
        f"Compression: {compression_force_kN:.1f} kN",
    ]
    if not meets_usp:
        note_parts.append("Does not meet USP <701> standard (< 15 min).")
    else:
        note_parts.append("Meets USP <701> standard (< 15 min).")

    return TabletDisintegrationResult(
        drug_name=drug_name,
        tablet_weight_mg=tablet_weight_mg,
        compression_force_kN=compression_force_kN,
        disintegrant_pct=disintegrant_pct,
        disintegrant_type=disintegrant_type,
        binder_pct=binder_pct,
        binder_type=binder_type,
        porosity_pct=porosity_pct,
        disintegration_time_min=dis_time,
        water_uptake_rate_mg_per_min=water_uptake,
        meets_usp_standard=meets_usp,
        disintegration_class=dis_class,
        notes="; ".join(note_parts),
    )


def screen_disintegrant_levels(
    drug_name: str,
    tablet_weight_mg: float,
    compression_force_kN: float,
    binder_pct: float,
    disintegrant_type: str,
    binder_type: str = "MCC",
    porosity_pct: float = 20.0,
) -> list:
    """Screen a range of disintegrant levels and return results sorted by
    disintegration time ascending (fastest first).

    Parameters
    ----------
    drug_name:
        Name of the drug.
    tablet_weight_mg:
        Total tablet weight in mg.
    compression_force_kN:
        Compression force in kN.
    binder_pct:
        Binder percentage.
    disintegrant_type:
        Disintegrant type to screen.
    binder_type:
        Binder type (default "MCC").
    porosity_pct:
        Tablet porosity percentage (default 20.0).

    Returns
    -------
    list[TabletDisintegrationResult] sorted by disintegration_time_min ascending.
    """
    levels = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    results = []
    for lvl in levels:
        try:
            r = predict_disintegration_time(
                drug_name=drug_name,
                tablet_weight_mg=tablet_weight_mg,
                compression_force_kN=compression_force_kN,
                disintegrant_pct=lvl,
                disintegrant_type=disintegrant_type,
                binder_pct=binder_pct,
                binder_type=binder_type,
                porosity_pct=porosity_pct,
            )
            results.append(r)
        except ValueError:
            pass
    results.sort(key=lambda x: x.disintegration_time_min)
    return results
