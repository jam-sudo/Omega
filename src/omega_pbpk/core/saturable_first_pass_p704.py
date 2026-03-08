"""Phase 704 — Saturable First-Pass Metabolism model."""

from dataclasses import dataclass


@dataclass
class SaturableFirstPassResult:
    drug_name: str
    times_h: list
    c_plasma_mg_L: list
    dose_mg: float
    vmax_mg_per_h: float
    km_mg_L: float
    cl_sys_L_per_h: float
    vd_sys_L: float
    f_bioavailable: float
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    dose_normalized_auc: float
    nonlinearity_index: float
    notes: str


def simulate_saturable_first_pass(
    drug_name: str,
    dose_mg: float,
    vmax_mg_per_h: float,
    km_mg_L: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    ka: float = 1.2,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> SaturableFirstPassResult:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if vmax_mg_per_h <= 0:
        raise ValueError("vmax_mg_per_h must be positive")
    if km_mg_L <= 0:
        raise ValueError("km_mg_L must be positive")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be positive")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be positive")

    V_portal = 1.5  # L
    ke = cl_sys_L_per_h / vd_sys_L
    k_portal_out = 1.0  # 1/h

    # Initial conditions
    A_gut = dose_mg * 0.85
    C_portal = 0.0
    C_plasma = 0.0

    n_steps = int(t_end_h / dt_h) + 1
    times_h = []
    c_plasma_list = []

    cmax = 0.0
    tmax = 0.0

    for step in range(n_steps):
        t = step * dt_h
        times_h.append(t)
        c_plasma_list.append(C_plasma)

        if C_plasma > cmax:
            cmax = C_plasma
            tmax = t

        # Forward Euler
        dA_gut = -ka * A_gut
        mm_extraction = vmax_mg_per_h * C_portal / (km_mg_L + C_portal) if C_portal > 0 else 0.0
        dC_portal = ka * A_gut / V_portal - mm_extraction - k_portal_out * C_portal
        dC_plasma = k_portal_out * C_portal * V_portal / vd_sys_L - ke * C_plasma

        A_gut += dA_gut * dt_h
        C_portal += dC_portal * dt_h
        C_plasma += dC_plasma * dt_h

        if A_gut < 0:
            A_gut = 0.0
        if C_portal < 0:
            C_portal = 0.0
        if C_plasma < 0:
            C_plasma = 0.0

    # AUC (trapezoidal)
    auc = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Bioavailability
    f_bioavailable = min(1.0, auc * cl_sys_L_per_h / dose_mg) if dose_mg > 0 else 0.0

    # Dose-normalized AUC
    dose_normalized_auc = auc / dose_mg

    # Nonlinearity index
    nonlinearity_index = max(0.0, min(1.0, (dose_mg / V_portal) / (km_mg_L + dose_mg / V_portal)))

    return SaturableFirstPassResult(
        drug_name=drug_name,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        dose_mg=dose_mg,
        vmax_mg_per_h=vmax_mg_per_h,
        km_mg_L=km_mg_L,
        cl_sys_L_per_h=cl_sys_L_per_h,
        vd_sys_L=vd_sys_L,
        f_bioavailable=f_bioavailable,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        dose_normalized_auc=dose_normalized_auc,
        nonlinearity_index=nonlinearity_index,
        notes="Saturable first-pass hepatic metabolism with Michaelis-Menten kinetics",
    )
