"""Phase 178 — Sick day protocol: PK adjustments during illness."""

from dataclasses import dataclass

_ILLNESS_EFFECTS: dict[str, dict[str, float]] = {
    "fever": {
        "cl_multiplier": 1.3,
        "vd_multiplier": 1.1,
        "f_multiplier": 0.9,
        "ka_multiplier": 0.8,
    },
    "vomiting": {
        "cl_multiplier": 1.0,
        "vd_multiplier": 1.0,
        "f_multiplier": 0.5,
        "ka_multiplier": 0.7,
    },
    "diarrhea": {
        "cl_multiplier": 1.0,
        "vd_multiplier": 0.9,
        "f_multiplier": 0.6,
        "ka_multiplier": 1.5,
    },
    "infection": {
        "cl_multiplier": 0.7,
        "vd_multiplier": 1.2,
        "f_multiplier": 0.85,
        "ka_multiplier": 0.9,
    },
    "dehydration": {
        "cl_multiplier": 0.6,
        "vd_multiplier": 0.8,
        "f_multiplier": 0.8,
        "ka_multiplier": 0.9,
    },
}


@dataclass(frozen=True)
class SickDayPKResult:
    drug_name: str
    illness_type: str
    baseline_dose_mg: float
    sick_day_dose_mg: float
    cl_sick_L_per_h: float
    vd_sick_L: float
    f_sick: float
    ka_sick_per_h: float
    css_avg_baseline_mg_L: float
    css_avg_sick_mg_L: float
    auc_ratio_sick_to_baseline: float
    dose_adjustment_pct: float
    monitoring_required: bool
    clinical_action: str
    notes: str


def _validate_pk_params(
    illness_type: str,
    baseline_dose_mg: float,
    baseline_cl_L_per_h: float,
    baseline_vd_L: float,
    baseline_f: float,
    baseline_ka_per_h: float,
    dosing_interval_h: float,
) -> None:
    if illness_type not in _ILLNESS_EFFECTS:
        raise ValueError(
            f"illness_type must be one of {set(_ILLNESS_EFFECTS.keys())}, got '{illness_type}'"
        )
    if baseline_dose_mg <= 0:
        raise ValueError(f"baseline_dose_mg must be > 0, got {baseline_dose_mg}")
    if baseline_cl_L_per_h <= 0:
        raise ValueError(f"baseline_cl_L_per_h must be > 0, got {baseline_cl_L_per_h}")
    if baseline_vd_L <= 0:
        raise ValueError(f"baseline_vd_L must be > 0, got {baseline_vd_L}")
    if not (0 < baseline_f <= 1):
        raise ValueError(f"baseline_f must be in (0, 1], got {baseline_f}")
    if baseline_ka_per_h <= 0:
        raise ValueError(f"baseline_ka_per_h must be > 0, got {baseline_ka_per_h}")
    if dosing_interval_h <= 0:
        raise ValueError(f"dosing_interval_h must be > 0, got {dosing_interval_h}")


def _css_avg(f: float, dose_mg: float, cl_L_per_h: float, tau_h: float) -> float:
    """1-cpt oral steady-state average concentration: Css_avg = F*D / (CL*tau)."""
    return (f * dose_mg) / (cl_L_per_h * tau_h)


def _determine_clinical_action(
    cl_sick: float,
    cl_baseline: float,
    f_sick: float,
    f_baseline: float,
    auc_ratio: float,
) -> str:
    """Determine clinical action based on sick PK changes."""
    # If absorption severely impaired (vomiting), consider skip
    if f_sick <= 0.5 * f_baseline:
        return "skip_dose"
    if auc_ratio < 0.7:
        return "increase_dose"
    if auc_ratio > 1.3:
        return "decrease_dose"
    return "maintain"


def assess_sick_day_pk(
    drug_name: str,
    illness_type: str,
    baseline_dose_mg: float,
    baseline_cl_L_per_h: float,
    baseline_vd_L: float,
    baseline_f: float,
    baseline_ka_per_h: float,
    dosing_interval_h: float,
) -> SickDayPKResult:
    """Assess PK changes during illness (same dose as baseline)."""
    _validate_pk_params(
        illness_type,
        baseline_dose_mg,
        baseline_cl_L_per_h,
        baseline_vd_L,
        baseline_f,
        baseline_ka_per_h,
        dosing_interval_h,
    )

    effects = _ILLNESS_EFFECTS[illness_type]
    cl_sick = baseline_cl_L_per_h * effects["cl_multiplier"]
    vd_sick = baseline_vd_L * effects["vd_multiplier"]
    f_sick = baseline_f * effects["f_multiplier"]
    ka_sick = baseline_ka_per_h * effects["ka_multiplier"]

    # Steady-state Css_avg
    css_baseline = _css_avg(baseline_f, baseline_dose_mg, baseline_cl_L_per_h, dosing_interval_h)
    # Sick day: same dose
    css_sick = _css_avg(f_sick, baseline_dose_mg, cl_sick, dosing_interval_h)

    auc_ratio = css_sick / css_baseline if css_baseline > 0 else 1.0

    monitoring_required = auc_ratio < 0.7 or auc_ratio > 1.3

    clinical_action = _determine_clinical_action(
        cl_sick, baseline_cl_L_per_h, f_sick, baseline_f, auc_ratio
    )

    notes = (
        f"CL: {baseline_cl_L_per_h:.2f}→{cl_sick:.2f} L/h, "
        f"F: {baseline_f:.2f}→{f_sick:.2f}, "
        f"AUC ratio: {auc_ratio:.3f}"
    )

    return SickDayPKResult(
        drug_name=drug_name,
        illness_type=illness_type,
        baseline_dose_mg=baseline_dose_mg,
        sick_day_dose_mg=baseline_dose_mg,
        cl_sick_L_per_h=round(cl_sick, 4),
        vd_sick_L=round(vd_sick, 4),
        f_sick=round(f_sick, 4),
        ka_sick_per_h=round(ka_sick, 4),
        css_avg_baseline_mg_L=round(css_baseline, 6),
        css_avg_sick_mg_L=round(css_sick, 6),
        auc_ratio_sick_to_baseline=round(auc_ratio, 6),
        dose_adjustment_pct=0.0,
        monitoring_required=monitoring_required,
        clinical_action=clinical_action,
        notes=notes,
    )


def recommend_sick_day_dose(
    drug_name: str,
    illness_type: str,
    baseline_dose_mg: float,
    target_auc_fraction: float = 1.0,
    baseline_cl_L_per_h: float = 5.0,
    baseline_vd_L: float = 50.0,
    baseline_f: float = 0.8,
    baseline_ka_per_h: float = 1.0,
    dosing_interval_h: float = 24.0,
) -> SickDayPKResult:
    """Recommend adjusted dose to maintain target AUC fraction during illness."""
    _validate_pk_params(
        illness_type,
        baseline_dose_mg,
        baseline_cl_L_per_h,
        baseline_vd_L,
        baseline_f,
        baseline_ka_per_h,
        dosing_interval_h,
    )
    if target_auc_fraction <= 0:
        raise ValueError(f"target_auc_fraction must be > 0, got {target_auc_fraction}")

    effects = _ILLNESS_EFFECTS[illness_type]
    cl_sick = baseline_cl_L_per_h * effects["cl_multiplier"]
    vd_sick = baseline_vd_L * effects["vd_multiplier"]
    f_sick = baseline_f * effects["f_multiplier"]
    ka_sick = baseline_ka_per_h * effects["ka_multiplier"]

    css_baseline = _css_avg(baseline_f, baseline_dose_mg, baseline_cl_L_per_h, dosing_interval_h)
    target_css = css_baseline * target_auc_fraction

    # Solve: target_css = f_sick * D_sick / (cl_sick * tau)
    # D_sick = target_css * cl_sick * tau / f_sick
    if f_sick > 0:
        sick_dose_raw = (target_css * cl_sick * dosing_interval_h) / f_sick
    else:
        # Absorption lost — can't dose orally
        sick_dose_raw = baseline_dose_mg * 2.0

    # Safety cap: max 2×, min 0.25× baseline
    sick_dose = max(baseline_dose_mg * 0.25, min(baseline_dose_mg * 2.0, sick_dose_raw))

    # Actual Css with adjusted dose
    css_sick = _css_avg(f_sick, sick_dose, cl_sick, dosing_interval_h)
    auc_ratio = css_sick / css_baseline if css_baseline > 0 else 1.0

    monitoring_required = auc_ratio < 0.7 or auc_ratio > 1.3

    dose_adjustment_pct = (sick_dose - baseline_dose_mg) / baseline_dose_mg * 100.0

    if dose_adjustment_pct > 5.0:
        clinical_action = "increase_dose"
    elif dose_adjustment_pct < -5.0:
        clinical_action = "decrease_dose"
    elif f_sick < 0.3 * baseline_f:
        clinical_action = "skip_dose"
    else:
        clinical_action = "maintain"

    notes = (
        f"Target AUC fraction: {target_auc_fraction:.2f}; "
        f"Raw sick dose: {sick_dose_raw:.2f} mg (capped to {sick_dose:.2f} mg); "
        f"CL: {baseline_cl_L_per_h:.2f}→{cl_sick:.2f} L/h, "
        f"F: {baseline_f:.2f}→{f_sick:.2f}"
    )

    return SickDayPKResult(
        drug_name=drug_name,
        illness_type=illness_type,
        baseline_dose_mg=baseline_dose_mg,
        sick_day_dose_mg=round(sick_dose, 4),
        cl_sick_L_per_h=round(cl_sick, 4),
        vd_sick_L=round(vd_sick, 4),
        f_sick=round(f_sick, 4),
        ka_sick_per_h=round(ka_sick, 4),
        css_avg_baseline_mg_L=round(css_baseline, 6),
        css_avg_sick_mg_L=round(css_sick, 6),
        auc_ratio_sick_to_baseline=round(auc_ratio, 6),
        dose_adjustment_pct=round(dose_adjustment_pct, 4),
        monitoring_required=monitoring_required,
        clinical_action=clinical_action,
        notes=notes,
    )
