"""Phase 530 — Drug Clearance in Critical Illness.

PK adjustments during critical illness (ICU patients): sepsis, ARDS,
multi-organ failure, liver failure, and renal failure.
"""

from dataclasses import dataclass

# Multipliers: (cl_mult, vd_mult, fu_mult)
_ILLNESS_PARAMS: dict = {
    "sepsis_early": (1.5, 1.4, 1.3),
    "sepsis_late": (0.5, 1.5, 1.3),
    "ards": (0.7, 1.6, 1.2),
    "liver_failure": (0.3, 1.2, 1.5),
    "renal_failure": (0.4, 1.1, 1.2),
    "multi_organ_failure": (0.2, 1.8, 1.8),
}


@dataclass(frozen=True)
class CriticalIllnessPKResult:
    """Result of critical illness PK assessment."""

    drug_name: str
    illness_type: str
    baseline_cl_L_per_h: float
    baseline_vd_L: float
    cl_sick_L_per_h: float
    vd_sick_L: float
    fu_sick: float
    cl_ratio: float
    vd_ratio: float
    t_half_sick_h: float
    loading_dose_mg: float
    maintenance_dose_ratio: float
    trough_ratio: float
    monitoring_priority: str
    notes: str


def _monitoring_priority(cl_ratio: float) -> str:
    if cl_ratio < 0.3 or cl_ratio > 1.5:
        return "urgent"
    if cl_ratio < 0.5 or cl_ratio > 1.2:
        return "high"
    return "routine"


def assess_critical_illness_pk(
    drug_name: str,
    illness_type: str,
    baseline_cl_L_per_h: float,
    baseline_vd_L: float,
    baseline_dose_mg: float,
    baseline_fu: float = 0.1,
    dosing_interval_h: float = 24.0,
) -> CriticalIllnessPKResult:
    """Assess PK changes during critical illness.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    illness_type:
        One of "sepsis_early", "sepsis_late", "ards", "liver_failure",
        "renal_failure", "multi_organ_failure".
    baseline_cl_L_per_h:
        Baseline total clearance in L/h (must be > 0).
    baseline_vd_L:
        Baseline volume of distribution in L (must be > 0).
    baseline_dose_mg:
        Baseline dose in mg (must be > 0).
    baseline_fu:
        Baseline fraction unbound in plasma.
    dosing_interval_h:
        Dosing interval in hours.

    Returns
    -------
    CriticalIllnessPKResult
    """
    if illness_type not in _ILLNESS_PARAMS:
        raise ValueError(
            f"illness_type must be one of {list(_ILLNESS_PARAMS)}, got {illness_type!r}"
        )
    if baseline_cl_L_per_h <= 0:
        raise ValueError(f"baseline_cl_L_per_h must be > 0, got {baseline_cl_L_per_h}")
    if baseline_vd_L <= 0:
        raise ValueError(f"baseline_vd_L must be > 0, got {baseline_vd_L}")
    if baseline_dose_mg <= 0:
        raise ValueError(f"baseline_dose_mg must be > 0, got {baseline_dose_mg}")

    cl_mult, vd_mult, fu_mult = _ILLNESS_PARAMS[illness_type]

    cl_sick = baseline_cl_L_per_h * cl_mult
    vd_sick = baseline_vd_L * vd_mult
    fu_sick = min(1.0, baseline_fu * fu_mult)

    cl_ratio = cl_sick / baseline_cl_L_per_h
    vd_ratio = vd_sick / baseline_vd_L

    t_half_sick = 0.693 * vd_sick / cl_sick if cl_sick > 0 else float("inf")

    # Loading dose: target Css_avg * Vd_sick
    # Css_avg (baseline) = F * dose / (CL_baseline * tau); F=1 IV
    # Target Css = same exposure as baseline
    target_css = baseline_dose_mg / (baseline_cl_L_per_h * dosing_interval_h)
    loading_dose = target_css * vd_sick

    # Maintenance dose ratio = cl_sick / baseline_cl (same steady-state)
    maintenance_dose_ratio = cl_ratio

    # Trough ratio: Css at same dose with sick PK vs baseline PK
    # Css_avg = dose / (CL * tau); trough ratio = CL_baseline / CL_sick
    trough_ratio = baseline_cl_L_per_h / cl_sick if cl_sick > 0 else float("inf")

    priority = _monitoring_priority(cl_ratio)

    # Build notes
    _note_map = {
        "sepsis_early": (
            "Early sepsis: augmented renal clearance and increased Vd due to "
            "capillary leak. Higher doses may be needed."
        ),
        "sepsis_late": (
            "Late sepsis: organ dysfunction reduces clearance; dose reduction and TDM recommended."
        ),
        "ards": (
            "ARDS: fluid overload increases Vd; lung inflammation impairs hepatic "
            "and renal clearance."
        ),
        "liver_failure": (
            "Liver failure: severely reduced hepatic metabolism and hypoalbuminemia "
            "increase free fraction; significant dose reduction required."
        ),
        "renal_failure": (
            "Renal failure: reduced GFR and tubular secretion decrease clearance; "
            "accumulation risk is high."
        ),
        "multi_organ_failure": (
            "Multi-organ failure: profound PK disruption; urgent dose adjustment "
            "and intensive monitoring required."
        ),
    }
    notes = _note_map[illness_type]

    return CriticalIllnessPKResult(
        drug_name=drug_name,
        illness_type=illness_type,
        baseline_cl_L_per_h=baseline_cl_L_per_h,
        baseline_vd_L=baseline_vd_L,
        cl_sick_L_per_h=cl_sick,
        vd_sick_L=vd_sick,
        fu_sick=fu_sick,
        cl_ratio=cl_ratio,
        vd_ratio=vd_ratio,
        t_half_sick_h=t_half_sick,
        loading_dose_mg=loading_dose,
        maintenance_dose_ratio=maintenance_dose_ratio,
        trough_ratio=trough_ratio,
        monitoring_priority=priority,
        notes=notes,
    )


def compare_illness_states(
    drug_name: str,
    baseline_cl_L_per_h: float,
    baseline_vd_L: float,
    baseline_dose_mg: float,
) -> list:
    """Run all 6 illness types and sort by cl_ratio ascending (most reduced first)."""
    results = [
        assess_critical_illness_pk(
            drug_name, illness_type, baseline_cl_L_per_h, baseline_vd_L, baseline_dose_mg
        )
        for illness_type in _ILLNESS_PARAMS
    ]
    results.sort(key=lambda r: r.cl_ratio)
    return results
