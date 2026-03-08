"""Phase 707 — Hepatic Steatosis PK Impact model."""

from dataclasses import dataclass


@dataclass
class HepaticSteatosisPKResult:
    drug_name: str
    dose_mg: float
    steatosis_grade: str
    hepatic_fat_pct: float
    times_h: list
    c_plasma_normal_mg_L: list
    c_plasma_steatosis_mg_L: list
    cmax_normal_mg_L: float
    cmax_steatosis_mg_L: float
    auc_normal_mg_h_per_L: float
    auc_steatosis_mg_h_per_L: float
    cl_ratio: float
    vd_ratio: float
    dose_adjustment_factor: float
    notes: str


_GRADE_FAT_PCT = {
    "mild": 10.0,
    "moderate": 30.0,
    "severe": 60.0,
}


def simulate_hepatic_steatosis_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    steatosis_grade: str = "moderate",
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> HepaticSteatosisPKResult:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be positive")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be positive")
    if steatosis_grade not in _GRADE_FAT_PCT:
        raise ValueError(f"steatosis_grade must be one of {list(_GRADE_FAT_PCT.keys())}")

    hepatic_fat_pct = _GRADE_FAT_PCT[steatosis_grade]

    # Clearance reduction
    cl_reduction = 1.0 - hepatic_fat_pct / 200.0
    cl_steatosis = cl_sys_L_per_h * cl_reduction

    # Vd increase for lipophilic drugs
    fat_fraction_increase = hepatic_fat_pct / 100.0
    vd_increase = 1.0 + fat_fraction_increase * max(0.0, logP) * 0.1
    vd_steatosis = vd_sys_L * vd_increase

    cl_ratio = cl_steatosis / cl_sys_L_per_h
    vd_ratio = vd_steatosis / vd_sys_L

    # 1-compartment oral PK parameters
    ka = 1.2
    bioavail = 0.85

    ke_normal = cl_sys_L_per_h / vd_sys_L
    ke_steatosis = cl_steatosis / vd_steatosis

    # Simulate both in parallel (same time grid)
    n_steps = int(t_end_h / dt_h) + 1
    times_h = []
    c_normal = []
    c_steatosis = []

    A_gut_n = dose_mg * bioavail
    C_n = 0.0
    A_gut_s = dose_mg * bioavail
    C_s = 0.0

    for step in range(n_steps):
        t = step * dt_h
        times_h.append(t)
        c_normal.append(C_n)
        c_steatosis.append(C_s)

        # Forward Euler - normal
        dA_gut_n = -ka * A_gut_n
        dC_n = ka * A_gut_n / vd_sys_L - ke_normal * C_n
        A_gut_n += dA_gut_n * dt_h
        C_n += dC_n * dt_h
        if A_gut_n < 0:
            A_gut_n = 0.0
        if C_n < 0:
            C_n = 0.0

        # Forward Euler - steatosis
        dA_gut_s = -ka * A_gut_s
        dC_s = ka * A_gut_s / vd_steatosis - ke_steatosis * C_s
        A_gut_s += dA_gut_s * dt_h
        C_s += dC_s * dt_h
        if A_gut_s < 0:
            A_gut_s = 0.0
        if C_s < 0:
            C_s = 0.0

    cmax_normal = max(c_normal)
    cmax_steatosis = max(c_steatosis)

    # AUC (trapezoidal)
    auc_normal = sum(
        0.5 * (c_normal[i] + c_normal[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_steatosis = sum(
        0.5 * (c_steatosis[i] + c_steatosis[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Dose adjustment factor
    if auc_steatosis > 0:
        dose_adjustment_factor = max(0.5, min(1.0, auc_normal / auc_steatosis))
    else:
        dose_adjustment_factor = 1.0

    return HepaticSteatosisPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        steatosis_grade=steatosis_grade,
        hepatic_fat_pct=hepatic_fat_pct,
        times_h=times_h,
        c_plasma_normal_mg_L=c_normal,
        c_plasma_steatosis_mg_L=c_steatosis,
        cmax_normal_mg_L=cmax_normal,
        cmax_steatosis_mg_L=cmax_steatosis,
        auc_normal_mg_h_per_L=auc_normal,
        auc_steatosis_mg_h_per_L=auc_steatosis,
        cl_ratio=cl_ratio,
        vd_ratio=vd_ratio,
        dose_adjustment_factor=dose_adjustment_factor,
        notes="Hepatic steatosis PK impact model with grade-dependent clearance reduction and Vd increase",
    )
