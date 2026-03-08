"""Phase 676 — Dialysis Drug Removal Model."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DialysisDrugRemovalResult:
    """Result of dialysis drug removal prediction."""

    drug_name: str
    session_duration_h: float
    blood_flow_mL_min: float
    dialysate_flow_mL_min: float
    sieving_coefficient: float
    dialysis_clearance_L_per_h: float
    total_removed_mg: float
    removal_pct: float
    pre_dialysis_conc_mg_L: float
    post_dialysis_conc_mg_L: float
    rebound_factor: float
    supplemental_dose_needed_mg: float
    dialyzability_class: str
    notes: str


def predict_dialysis_removal(
    drug_name: str,
    pre_dialysis_conc_mg_L: float,
    vd_L: float,
    protein_binding_pct: float,
    mw_Da: float,
    blood_flow_mL_min: float = 300.0,
    dialysate_flow_mL_min: float = 500.0,
    session_duration_h: float = 4.0,
    renal_cl_L_per_h: float = 0.0,
) -> DialysisDrugRemovalResult:
    """Predict drug removal during hemodialysis."""
    if pre_dialysis_conc_mg_L <= 0:
        raise ValueError("pre_dialysis_conc_mg_L must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (0 <= protein_binding_pct <= 100):
        raise ValueError("protein_binding_pct must be between 0 and 100")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if blood_flow_mL_min <= 0:
        raise ValueError("blood_flow_mL_min must be > 0")
    if session_duration_h <= 0:
        raise ValueError("session_duration_h must be > 0")

    fu = 1.0 - protein_binding_pct / 100.0
    SC = fu * min(1.0, 500.0 / mw_Da)
    SC = max(0.0, min(1.0, SC))

    Q_b = blood_flow_mL_min * 60.0 / 1000.0  # L/h
    Q_d = dialysate_flow_mL_min * 60.0 / 1000.0  # L/h

    if Q_b * SC > 0:
        CL_dialysis = Q_b * SC * (1.0 - math.exp(-Q_d * SC / Q_b))
    else:
        CL_dialysis = 0.0
    CL_dialysis = min(CL_dialysis, Q_b * SC)

    CL_total = CL_dialysis + renal_cl_L_per_h
    k_removal = CL_total / vd_L

    post_dialysis_conc = pre_dialysis_conc_mg_L * math.exp(-k_removal * session_duration_h)
    total_removed_mg = (pre_dialysis_conc_mg_L - post_dialysis_conc) * vd_L

    total_body_amount = pre_dialysis_conc_mg_L * vd_L
    removal_pct = (total_removed_mg / total_body_amount * 100.0) if total_body_amount > 0 else 0.0
    removal_pct = max(0.0, min(100.0, removal_pct))

    rebound_factor = min(3.0, 1.0 + max(0.0, (vd_L / 30.0 - 1.0) * 0.2))
    supplemental_dose_needed_mg = total_removed_mg * rebound_factor

    if removal_pct > 50:
        dialyzability_class = "highly_dialyzable"
    elif removal_pct > 20:
        dialyzability_class = "dialyzable"
    elif removal_pct > 5:
        dialyzability_class = "partially_dialyzable"
    else:
        dialyzability_class = "not_dialyzable"

    return DialysisDrugRemovalResult(
        drug_name=drug_name,
        session_duration_h=session_duration_h,
        blood_flow_mL_min=blood_flow_mL_min,
        dialysate_flow_mL_min=dialysate_flow_mL_min,
        sieving_coefficient=SC,
        dialysis_clearance_L_per_h=CL_dialysis,
        total_removed_mg=total_removed_mg,
        removal_pct=removal_pct,
        pre_dialysis_conc_mg_L=pre_dialysis_conc_mg_L,
        post_dialysis_conc_mg_L=post_dialysis_conc,
        rebound_factor=rebound_factor,
        supplemental_dose_needed_mg=supplemental_dose_needed_mg,
        dialyzability_class=dialyzability_class,
        notes=f"Dialysis removal prediction for {drug_name}",
    )
