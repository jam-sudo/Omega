"""Hepatic Impairment Drug Dosing — Phase 543.

Dose adjustment for patients with hepatic impairment using Child-Pugh scoring.
Accounts for portal shunting, changed protein binding, and altered bioavailability
to maintain target Css_avg comparable to healthy subjects.
"""

from __future__ import annotations

__all__ = [
    "HepaticImpairmentDosingResult",
    "adjust_dose_hepatic_impairment",
    "compare_severity_levels",
]

from dataclasses import dataclass

# Child-Pugh severity parameters
_SEVERITY_PARAMS: dict[str, dict] = {
    "child_pugh_a": {
        "cl_multiplier": 0.75,
        "fu_multiplier": 1.2,
        "vd_multiplier": 1.1,
        "shunt_fraction": 0.1,
    },
    "child_pugh_b": {
        "cl_multiplier": 0.5,
        "fu_multiplier": 1.4,
        "vd_multiplier": 1.3,
        "shunt_fraction": 0.3,
    },
    "child_pugh_c": {
        "cl_multiplier": 0.25,
        "fu_multiplier": 1.7,
        "vd_multiplier": 1.6,
        "shunt_fraction": 0.5,
    },
}

_VALID_SEVERITIES = set(_SEVERITY_PARAMS.keys())

# Portal blood flow in healthy subjects (L/h)
_Q_PORTAL_NORMAL_L_PER_H = 81.0  # ~1350 mL/min


@dataclass(frozen=True)
class HepaticImpairmentDosingResult:
    """Result of hepatic impairment dose adjustment calculation."""

    drug_name: str
    severity: str
    baseline_dose_mg: float
    baseline_cl_L_per_h: float
    baseline_fu: float
    cl_impaired_L_per_h: float
    fu_impaired: float
    vd_impaired_L: float
    cl_ratio: float
    fu_ratio: float
    adjusted_dose_mg: float
    dose_adjustment_pct: float
    css_avg_baseline_mg_L: float
    css_avg_impaired_mg_L: float
    css_ratio: float
    monitoring_required: bool
    contraindicated: bool
    notes: str


def _compute_extraction_ratio(
    fup: float,
    clint_L_per_h: float,
    q_L_per_h: float,
) -> float:
    """Well-stirred hepatic extraction ratio E_h = fup*CLint / (Q + fup*CLint)."""
    numerator = fup * clint_L_per_h
    return numerator / (q_L_per_h + numerator)


def _estimate_clint_from_cl(
    cl_baseline_L_per_h: float,
    fu_baseline: float,
    q_L_per_h: float,
    f_oral: float,
) -> float:
    """Back-calculate CLint from observed CL and F.

    For a well-stirred model:
        CL_hepatic = Q * E_h  where E_h = fup*CLint / (Q + fup*CLint)
        F = 1 - E_h  (first-pass only, assuming complete absorption)

    We assume CL ≈ CL_hepatic here.
    Solve: E_h = CL / Q  -> fup*CLint = Q * E_h / (1 - E_h)
    Clamp E_h to (0, 0.999) to avoid divide-by-zero.
    """
    e_h = min(cl_baseline_L_per_h / q_L_per_h, 0.999)
    clint = (q_L_per_h * e_h) / (fu_baseline * max(1.0 - e_h, 1e-6))
    return max(clint, 1e-6)


def adjust_dose_hepatic_impairment(
    drug_name: str,
    severity: str,
    baseline_dose_mg: float,
    baseline_cl_L_per_h: float,
    baseline_vd_L: float,
    baseline_fu: float = 0.1,
    dosing_interval_h: float = 24.0,
    f_oral: float = 0.8,
) -> HepaticImpairmentDosingResult:
    """Adjust dose for hepatic impairment to maintain target Css_avg.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    severity:
        Child-Pugh classification: "child_pugh_a", "child_pugh_b", or "child_pugh_c".
    baseline_dose_mg:
        Normal (healthy) dose in mg (> 0).
    baseline_cl_L_per_h:
        Healthy-subject total clearance in L/h (> 0).
    baseline_vd_L:
        Healthy-subject volume of distribution in L (> 0).
    baseline_fu:
        Unbound fraction in healthy plasma (0 < fu <= 1, default 0.1).
    dosing_interval_h:
        Dosing interval in hours (default 24).
    f_oral:
        Oral bioavailability in healthy subjects (default 0.8).

    Returns
    -------
    HepaticImpairmentDosingResult
    """
    # --- Validation ---
    if severity not in _VALID_SEVERITIES:
        raise ValueError(f"severity must be one of {sorted(_VALID_SEVERITIES)}, got {severity!r}")
    if baseline_dose_mg <= 0.0:
        raise ValueError(f"baseline_dose_mg must be > 0, got {baseline_dose_mg}")
    if baseline_cl_L_per_h <= 0.0:
        raise ValueError(f"baseline_cl_L_per_h must be > 0, got {baseline_cl_L_per_h}")
    if baseline_vd_L <= 0.0:
        raise ValueError(f"baseline_vd_L must be > 0, got {baseline_vd_L}")
    if not (0.0 < baseline_fu <= 1.0):
        raise ValueError(f"baseline_fu must be in (0, 1], got {baseline_fu}")

    params = _SEVERITY_PARAMS[severity]
    cl_mult = params["cl_multiplier"]
    fu_mult = params["fu_multiplier"]
    vd_mult = params["vd_multiplier"]
    shunt_frac = params["shunt_fraction"]

    # --- Impaired PK parameters ---
    cl_impaired = baseline_cl_L_per_h * cl_mult
    fu_impaired = min(baseline_fu * fu_mult, 1.0)
    vd_impaired = baseline_vd_L * vd_mult

    cl_ratio = cl_impaired / baseline_cl_L_per_h
    fu_ratio = fu_impaired / baseline_fu

    # --- Portal shunting effect on bioavailability ---
    # Q_sick = Q_normal * (1 + shunt_fraction)
    q_normal = _Q_PORTAL_NORMAL_L_PER_H
    q_sick = q_normal * (1.0 + shunt_frac)

    # Back-calculate CLint from baseline CL using well-stirred model
    clint = _estimate_clint_from_cl(baseline_cl_L_per_h, baseline_fu, q_normal, f_oral)

    # Extraction ratio and bioavailability in impaired state
    e_h_sick = _compute_extraction_ratio(fu_impaired, clint, q_sick)
    f_sick = 1.0 - e_h_sick
    f_sick = max(f_sick, 0.01)  # clamp

    # --- Css_avg calculation ---
    # Css_avg = F * dose / (CL * tau)
    css_avg_baseline = (f_oral * baseline_dose_mg) / (baseline_cl_L_per_h * dosing_interval_h)

    # --- Dose adjustment to restore Css_avg ---
    # Target Css_avg_impaired = Css_avg_baseline
    # Css_avg_impaired = F_sick * adjusted_dose / (CL_sick * tau)
    # => adjusted_dose = css_avg_baseline * CL_sick * tau / F_sick
    adjusted_dose_raw = css_avg_baseline * cl_impaired * dosing_interval_h / f_sick

    # Clamp to [0.1, 2.0] * baseline_dose
    adjusted_dose = max(0.1 * baseline_dose_mg, min(2.0 * baseline_dose_mg, adjusted_dose_raw))

    dose_adjustment_pct = (adjusted_dose / baseline_dose_mg - 1.0) * 100.0

    # --- Css_avg with adjusted dose in impaired patient ---
    css_avg_impaired = (f_sick * adjusted_dose) / (cl_impaired * dosing_interval_h)
    css_ratio = css_avg_impaired / css_avg_baseline if css_avg_baseline > 0 else 1.0

    # --- Risk flags ---
    monitoring_required = cl_ratio < 0.5
    contraindicated = (cl_ratio < 0.3) and (baseline_cl_L_per_h > 10.0)

    # --- Notes ---
    label_map = {
        "child_pugh_a": "Child-Pugh A (mild)",
        "child_pugh_b": "Child-Pugh B (moderate)",
        "child_pugh_c": "Child-Pugh C (severe)",
    }
    notes_parts = [
        f"{label_map[severity]}; cl_ratio={cl_ratio:.2f}; fu_ratio={fu_ratio:.2f}",
        f"CLint(estimated)={clint:.1f} L/h; Q_sick={q_sick:.0f} L/h",
        f"E_h_sick={e_h_sick:.3f}; F_sick={f_sick:.3f}",
        f"Adjusted dose {adjusted_dose:.1f} mg vs baseline {baseline_dose_mg:.1f} mg",
    ]
    if monitoring_required:
        notes_parts.append("Enhanced monitoring required")
    if contraindicated:
        notes_parts.append("Consider contraindication — high-CL drug with severe impairment")
    notes = "; ".join(notes_parts)

    return HepaticImpairmentDosingResult(
        drug_name=drug_name,
        severity=severity,
        baseline_dose_mg=baseline_dose_mg,
        baseline_cl_L_per_h=baseline_cl_L_per_h,
        baseline_fu=baseline_fu,
        cl_impaired_L_per_h=cl_impaired,
        fu_impaired=fu_impaired,
        vd_impaired_L=vd_impaired,
        cl_ratio=cl_ratio,
        fu_ratio=fu_ratio,
        adjusted_dose_mg=adjusted_dose,
        dose_adjustment_pct=dose_adjustment_pct,
        css_avg_baseline_mg_L=css_avg_baseline,
        css_avg_impaired_mg_L=css_avg_impaired,
        css_ratio=css_ratio,
        monitoring_required=monitoring_required,
        contraindicated=contraindicated,
        notes=notes,
    )


def compare_severity_levels(
    drug_name: str,
    baseline_dose_mg: float,
    baseline_cl_L_per_h: float,
    baseline_vd_L: float,
) -> list:
    """Run all three Child-Pugh severity levels and sort by cl_ratio ascending.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    baseline_dose_mg:
        Normal dose in mg.
    baseline_cl_L_per_h:
        Healthy clearance in L/h.
    baseline_vd_L:
        Healthy volume of distribution in L.

    Returns
    -------
    list of HepaticImpairmentDosingResult sorted by cl_ratio ascending (most severe first).
    """
    results = []
    for severity in _VALID_SEVERITIES:
        result = adjust_dose_hepatic_impairment(
            drug_name=drug_name,
            severity=severity,
            baseline_dose_mg=baseline_dose_mg,
            baseline_cl_L_per_h=baseline_cl_L_per_h,
            baseline_vd_L=baseline_vd_L,
        )
        results.append(result)

    results.sort(key=lambda r: r.cl_ratio)
    return results
