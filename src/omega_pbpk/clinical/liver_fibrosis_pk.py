"""
PK alterations in liver fibrosis and cirrhosis.

Provides functions for:
- Clearance scaling by Child-Pugh score
- Albumin binding changes in fibrosis
- Portal hypertension effect on bioavailability
- Dose recommendations for cirrhotic patients
- 1-compartment PK simulation with cirrhosis adjustments
"""



def fibrosis_cl_scaling(cl_normal_L_per_h: float, child_pugh_score: int) -> dict:
    """
    Scale drug clearance based on Child-Pugh hepatic function score.

    Child-Pugh A (5-6): CL_fibrosis = cl_normal * 0.75
    Child-Pugh B (7-9): CL_fibrosis = cl_normal * 0.50
    Child-Pugh C (10-15): CL_fibrosis = cl_normal * 0.25

    Args:
        cl_normal_L_per_h: Clearance in normal liver (L/h)
        child_pugh_score: Child-Pugh score (5-15)

    Returns:
        dict with cl_fibrosis_L_per_h, cl_ratio, child_pugh_class,
        dose_adjustment_factor, notes
    """
    if cl_normal_L_per_h < 0:
        raise ValueError("cl_normal_L_per_h must be non-negative")
    if not (5 <= child_pugh_score <= 15):
        raise ValueError("child_pugh_score must be between 5 and 15")

    if child_pugh_score <= 6:
        child_pugh_class = "A"
        cl_ratio = 0.75
        notes = "Mild hepatic impairment; monitor for accumulation"
    elif child_pugh_score <= 9:
        child_pugh_class = "B"
        cl_ratio = 0.50
        notes = "Moderate hepatic impairment; dose reduction recommended"
    else:
        child_pugh_class = "C"
        cl_ratio = 0.25
        notes = "Severe hepatic impairment; significant dose reduction required; use with caution"

    cl_fibrosis = cl_normal_L_per_h * cl_ratio
    # Dose adjustment factor: to maintain same exposure, adjust proportionally to CL
    dose_adjustment_factor = cl_ratio

    return {
        "cl_fibrosis_L_per_h": cl_fibrosis,
        "cl_ratio": cl_ratio,
        "child_pugh_class": child_pugh_class,
        "dose_adjustment_factor": dose_adjustment_factor,
        "notes": notes,
    }


def albumin_binding_fibrosis(
    fu_normal: float,
    albumin_g_dL: float,
    albumin_normal_g_dL: float = 4.0,
) -> float:
    """
    Estimate unbound fraction in cirrhosis with reduced albumin.

    Lower albumin in cirrhosis leads to higher unbound fraction.
    fu_fibrosis = fu_normal * (albumin_normal / albumin) ** 0.5

    Returns:
        fu_fibrosis clamped to (0, 1]
    """
    if not (0 < fu_normal <= 1):
        raise ValueError("fu_normal must be in (0, 1]")
    if albumin_g_dL <= 0:
        raise ValueError("albumin_g_dL must be positive")
    if albumin_normal_g_dL <= 0:
        raise ValueError("albumin_normal_g_dL must be positive")

    fu_fibrosis = fu_normal * (albumin_normal_g_dL / albumin_g_dL) ** 0.5
    # Clamp to (0, 1]
    fu_fibrosis = min(fu_fibrosis, 1.0)
    fu_fibrosis = max(fu_fibrosis, 1e-9)
    return fu_fibrosis


def portal_hypertension_bioavailability(
    f_oral_normal: float,
    porto_systemic_shunt_fraction: float = 0.3,
) -> float:
    """
    Estimate oral bioavailability with portal hypertension.

    Porto-systemic shunting allows drug to bypass liver first-pass extraction.
    F_fibrosis = f_oral_normal + (1 - f_oral_normal) * porto_systemic_shunt_fraction

    Returns:
        f_fibrosis clamped to [0, 1]
    """
    if not (0.0 <= f_oral_normal <= 1.0):
        raise ValueError("f_oral_normal must be in [0, 1]")
    if not (0.0 <= porto_systemic_shunt_fraction <= 1.0):
        raise ValueError("porto_systemic_shunt_fraction must be in [0, 1]")

    f_fibrosis = f_oral_normal + (1.0 - f_oral_normal) * porto_systemic_shunt_fraction
    f_fibrosis = max(0.0, min(1.0, f_fibrosis))
    return f_fibrosis


def dose_recommendation_cirrhosis(
    drug_name: str,
    normal_dose_mg: float,
    cl_ratio: float,
    fu_ratio: float,
) -> dict:
    """
    Recommend dose adjustment for cirrhotic patients.

    effective_dose = normal_dose * cl_ratio / fu_ratio

    Args:
        drug_name: Name of the drug
        normal_dose_mg: Standard dose in normal patients (mg)
        cl_ratio: CL_fibrosis / CL_normal
        fu_ratio: fu_fibrosis / fu_normal

    Returns:
        dict with drug_name, recommended_dose_mg, rationale, warnings
    """
    if normal_dose_mg <= 0:
        raise ValueError("normal_dose_mg must be positive")
    if cl_ratio <= 0:
        raise ValueError("cl_ratio must be positive")
    if fu_ratio <= 0:
        raise ValueError("fu_ratio must be positive")

    recommended_dose_mg = normal_dose_mg * cl_ratio / fu_ratio

    rationale = (
        f"CL reduced to {cl_ratio * 100:.0f}% of normal; "
        f"fu increased to {fu_ratio * 100:.0f}% of normal value. "
        f"Recommended dose: {recommended_dose_mg:.1f} mg."
    )

    warnings = []
    if cl_ratio <= 0.25:
        warnings.append("Severe hepatic impairment — monitor closely for toxicity")
    if fu_ratio > 1.5:
        warnings.append("Substantially elevated unbound fraction — risk of toxicity increased")
    if recommended_dose_mg < normal_dose_mg * 0.25:
        warnings.append("Dose reduction >75% — consider alternative therapy")

    return {
        "drug_name": drug_name,
        "recommended_dose_mg": recommended_dose_mg,
        "rationale": rationale,
        "warnings": warnings,
    }


def simulate_cirrhosis_pk(
    drug_name: str,
    dose_mg: float,
    cl_normal_L_per_h: float,
    vd_L: float,
    child_pugh_score: int,
    f_oral_normal: float = 0.8,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> dict:
    """
    Simulate 1-compartment oral PK with cirrhosis adjustments.

    - Scale CL by Child-Pugh class
    - Adjust F for porto-systemic shunting
    - Forward Euler for 1-cpt oral model

    Returns:
        times_h, c_plasma_mg_L, cmax, auc, cl_fibrosis, f_fibrosis, notes
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_normal_L_per_h <= 0:
        raise ValueError("cl_normal_L_per_h must be positive")
    if vd_L <= 0:
        raise ValueError("vd_L must be positive")
    if not (5 <= child_pugh_score <= 15):
        raise ValueError("child_pugh_score must be between 5 and 15")
    if not (0.0 <= f_oral_normal <= 1.0):
        raise ValueError("f_oral_normal must be in [0, 1]")

    # Scale CL
    cl_info = fibrosis_cl_scaling(cl_normal_L_per_h, child_pugh_score)
    cl_fibrosis = cl_info["cl_fibrosis_L_per_h"]

    # Adjust bioavailability for porto-systemic shunting
    # Default shunt fraction = 0.3
    f_fibrosis = portal_hypertension_bioavailability(
        f_oral_normal, porto_systemic_shunt_fraction=0.3
    )

    # 1-compartment oral model
    # Assume ka = 1.0/h (reasonable default absorption rate)
    ka = 1.0  # /h
    ke = cl_fibrosis / vd_L  # elimination rate constant

    # Dose absorbed = f_fibrosis * dose_mg
    dose_absorbed = f_fibrosis * dose_mg

    times_h = []
    c_plasma_mg_L = []

    t = 0.0
    # Amount in gut compartment and central compartment
    a_gut = dose_absorbed
    a_central = 0.0

    while t <= t_end_h + 1e-9:
        times_h.append(t)
        c = a_central / vd_L
        c_plasma_mg_L.append(c)

        # Forward Euler
        da_gut = -ka * a_gut
        da_central = ka * a_gut - ke * a_central

        a_gut = max(a_gut + da_gut * dt_h, 0.0)
        a_central = max(a_central + da_central * dt_h, 0.0)
        t += dt_h

    cmax = max(c_plasma_mg_L)

    # Trapezoidal AUC
    auc = 0.0
    for i in range(len(times_h) - 1):
        dt = times_h[i + 1] - times_h[i]
        auc += 0.5 * (c_plasma_mg_L[i] + c_plasma_mg_L[i + 1]) * dt

    notes = (
        cl_info["notes"]
        + f" Bioavailability increased from {f_oral_normal:.2f} to {f_fibrosis:.2f} due to shunting."
    )

    return {
        "times_h": times_h,
        "c_plasma_mg_L": c_plasma_mg_L,
        "cmax": cmax,
        "auc": auc,
        "cl_fibrosis": cl_fibrosis,
        "f_fibrosis": f_fibrosis,
        "notes": notes,
    }
