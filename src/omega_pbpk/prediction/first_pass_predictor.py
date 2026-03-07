"""First-pass metabolism prediction: gut wall + hepatic extraction and oral bioavailability."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FirstPassResult:
    drug_name: str
    fg_gut_wall: float  # fraction surviving gut wall (0-1)
    fh_hepatic: float  # fraction surviving hepatic first pass (0-1)
    fa_absorbed: float  # fraction absorbed from gut
    f_oral: float  # overall F = fa * fg * fh
    f_oral_pct: float  # F as percentage
    er_hepatic: float  # hepatic extraction ratio
    er_gut: float  # gut wall extraction ratio
    cl_gut_mL_per_min: float  # gut wall intrinsic CL
    cl_hepatic_mL_per_min: float
    qh_L_per_h: float  # hepatic blood flow used
    primary_route: str  # "gut_dominant","liver_dominant","balanced","minimal"
    dose_linearity: str  # "linear","nonlinear" (MM saturation)
    notes: str


def predict_first_pass(
    drug_name: str,
    clint_gut_mL_per_min: float,
    clint_hepatic_mL_per_min: float,
    fu_plasma: float = 0.5,
    fu_gut: float = 1.0,
    fa: float = 1.0,
    qh_L_per_h: float = 90.0,
    qgut_L_per_h: float = 18.0,
    dose_mg: float = 100.0,
    km_mg_L: float | None = None,
    vmax_mg_per_h: float | None = None,
) -> FirstPassResult:
    """Predict first-pass extraction and oral bioavailability using well-stirred model."""
    # Validation
    if clint_gut_mL_per_min < 0:
        raise ValueError("clint_gut_mL_per_min must be >= 0")
    if clint_hepatic_mL_per_min < 0:
        raise ValueError("clint_hepatic_mL_per_min must be >= 0")
    if not (0 < fu_plasma <= 1):
        raise ValueError("fu_plasma must be in (0, 1]")
    if not (0 < fu_gut <= 1):
        raise ValueError("fu_gut must be in (0, 1]")
    if not (0 < fa <= 1):
        raise ValueError("fa must be in (0, 1]")
    if qh_L_per_h <= 0:
        raise ValueError("qh_L_per_h must be > 0")
    if qgut_L_per_h <= 0:
        raise ValueError("qgut_L_per_h must be > 0")

    # Convert flows from L/h to mL/min
    qh_mLmin = qh_L_per_h * 1000.0 / 60.0
    qgut_mLmin = qgut_L_per_h * 1000.0 / 60.0

    # Determine effective CLint values (MM saturation if applicable)
    effective_clint_gut = clint_gut_mL_per_min
    effective_clint_hep = clint_hepatic_mL_per_min
    dose_linearity = "linear"

    if km_mg_L is not None and vmax_mg_per_h is not None:
        # Approximate gut concentration: dose dissolved in 250 mL
        c_gut_mg_L = dose_mg / 0.25  # mg / 0.25 L = mg/L
        # Effective CLint = Vmax(mg/h) / (Km(mg/L) + C(mg/L)) in L/h then convert to mL/min
        vmax_L_h = vmax_mg_per_h  # Vmax numerically (mg/h)
        c_total_mg_L = km_mg_L + c_gut_mg_L
        if c_total_mg_L > 0:
            # Effective CLint in L/h then convert
            effective_clint_hep_Lh = vmax_L_h / c_total_mg_L
            effective_clint_hep = effective_clint_hep_Lh * 1000.0 / 60.0
        else:
            effective_clint_hep = clint_hepatic_mL_per_min

        if clint_hepatic_mL_per_min > 0 and effective_clint_hep < clint_hepatic_mL_per_min * 0.8:
            dose_linearity = "nonlinear"

    # Well-stirred gut wall extraction ratio
    gut_denom = qgut_mLmin + fu_gut * effective_clint_gut
    if gut_denom > 0:
        er_gut = (fu_gut * effective_clint_gut) / gut_denom
    else:
        er_gut = 0.0

    fg = 1.0 - er_gut

    # Well-stirred hepatic extraction ratio
    hep_denom = qh_mLmin + fu_plasma * effective_clint_hep
    if hep_denom > 0:
        er_hep = (fu_plasma * effective_clint_hep) / hep_denom
    else:
        er_hep = 0.0

    fh = 1.0 - er_hep

    # Overall oral bioavailability
    f_oral = fa * fg * fh
    f_oral_pct = f_oral * 100.0

    # Determine primary elimination route
    if er_gut < 0.1 and er_hep < 0.1:
        primary_route = "minimal"
    elif er_gut > 2.0 * er_hep:
        primary_route = "gut_dominant"
    elif er_hep > 2.0 * er_gut:
        primary_route = "liver_dominant"
    else:
        primary_route = "balanced"

    # Build notes
    notes_parts = []
    if f_oral < 0.2:
        notes_parts.append("low oral bioavailability (<20%)")
    elif f_oral > 0.8:
        notes_parts.append("high oral bioavailability (>80%)")
    else:
        notes_parts.append(f"moderate oral bioavailability ({f_oral_pct:.1f}%)")

    if dose_linearity == "nonlinear":
        notes_parts.append("Michaelis-Menten saturation detected at this dose")

    notes = "; ".join(notes_parts) if notes_parts else "standard first-pass extraction"

    return FirstPassResult(
        drug_name=drug_name,
        fg_gut_wall=fg,
        fh_hepatic=fh,
        fa_absorbed=fa,
        f_oral=f_oral,
        f_oral_pct=f_oral_pct,
        er_hepatic=er_hep,
        er_gut=er_gut,
        cl_gut_mL_per_min=clint_gut_mL_per_min,
        cl_hepatic_mL_per_min=clint_hepatic_mL_per_min,
        qh_L_per_h=qh_L_per_h,
        primary_route=primary_route,
        dose_linearity=dose_linearity,
        notes=notes,
    )


def scan_clint_range(
    drug_name: str,
    clint_hepatic_values: list,
    fu_plasma: float = 0.5,
    fa: float = 1.0,
) -> list:
    """Vary CLint_hepatic and return list of FirstPassResult for sensitivity analysis."""
    results = []
    for clint_hep in clint_hepatic_values:
        result = predict_first_pass(
            drug_name=drug_name,
            clint_gut_mL_per_min=0.0,
            clint_hepatic_mL_per_min=clint_hep,
            fu_plasma=fu_plasma,
            fa=fa,
        )
        results.append(result)
    return results
