"""Phase 665 — Drug Cartilage Penetration.

Models drug penetration into avascular cartilage tissue via Fickian diffusion
from synovial fluid through surface and deep cartilage layers.
"""

from dataclasses import dataclass, field


@dataclass
class CartilagePenetrationResult:
    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_synovial_mg_L: list = field(default_factory=list)
    c_cartilage_surface_mg_g: list = field(default_factory=list)
    c_cartilage_deep_mg_g: list = field(default_factory=list)
    diffusion_coefficient_cm2_h: float = 0.0
    penetration_depth_pct: float = 0.0
    steady_state_surface_mg_g: float = 0.0
    steady_state_deep_mg_g: float = 0.0
    surface_to_synovial_ratio: float = 0.0
    time_to_50pct_ss_h: float = 0.0
    cartilage_auc_surface: float = 0.0
    cartilage_auc_deep: float = 0.0
    notes: str = ""


def simulate_cartilage_penetration(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    cartilage_thickness_mm: float = 3.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> CartilagePenetrationResult:
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")

    # Diffusion coefficient
    D = 1e-6 * (1.0 / (1.0 + mw_Da / 200.0)) * max(0.1, 1.0 / (1.0 + 0.5 * abs(logP)))
    D_cm2_h = D * 3600.0
    D_cm2_h = max(1e-5, min(0.1, D_cm2_h))

    # Cartilage layer thickness
    V_surf = 0.5 * cartilage_thickness_mm / 10.0  # cm
    k_diff = D_cm2_h / (V_surf**2)  # /h

    # Synovial/plasma ratio
    r_synovial_plasma = max(0.1, min(2.0, 1.0 - logP * 0.1 + 0.5))

    # Transfer rates
    k_syno_in = D_cm2_h / (0.05**2)
    k_syno_in = max(0.01, min(50.0, k_syno_in))

    # PK parameters
    ka = 1.2  # /h
    k_el = cl_sys_L_per_h / vd_sys_L
    vd = vd_sys_L
    k_bind = 0.01  # /h

    # Initial conditions
    A_gut = dose_mg * 0.85
    C_plasma = 0.0
    C_cart_surf = 0.0
    C_cart_deep = 0.0

    n_steps = int(t_end_h / dt_h) + 1

    times_h = []
    c_synovial_list = []
    c_surf_list = []
    c_deep_list = []

    for i in range(n_steps):
        t = i * dt_h
        C_synovial = C_plasma * r_synovial_plasma

        times_h.append(t)
        c_synovial_list.append(C_synovial)
        c_surf_list.append(C_cart_surf * 0.001)
        c_deep_list.append(C_cart_deep * 0.001)

        # Forward Euler
        dA_gut = -ka * A_gut
        dC_plasma = ka * A_gut / vd - k_el * C_plasma
        dC_cart_surf = k_syno_in * (C_synovial - C_cart_surf) - k_diff * (C_cart_surf - C_cart_deep)
        dC_cart_deep = k_diff * (C_cart_surf - C_cart_deep) - k_bind * C_cart_deep

        A_gut += dA_gut * dt_h
        C_plasma += dC_plasma * dt_h
        C_cart_surf += dC_cart_surf * dt_h
        C_cart_deep += dC_cart_deep * dt_h

        if A_gut < 0:
            A_gut = 0.0
        if C_plasma < 0:
            C_plasma = 0.0
        if C_cart_surf < 0:
            C_cart_surf = 0.0
        if C_cart_deep < 0:
            C_cart_deep = 0.0

    # AUC via manual trapezoidal
    auc_surface = sum(
        0.5 * (c_surf_list[i] + c_surf_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_deep = sum(
        0.5 * (c_deep_list[i] + c_deep_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_synovial = sum(
        0.5 * (c_synovial_list[i] + c_synovial_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Penetration depth
    if c_surf_list[-1] > 0:
        penetration_depth_pct = (c_deep_list[-1] / c_surf_list[-1]) * 100.0
    else:
        penetration_depth_pct = 0.0
    penetration_depth_pct = min(100.0, penetration_depth_pct)

    # Steady state
    ss_surface = c_surf_list[-1]
    ss_deep = c_deep_list[-1]

    # Surface to synovial ratio
    if auc_synovial > 0:
        surface_to_synovial_ratio = auc_surface / auc_synovial
    else:
        surface_to_synovial_ratio = 0.0

    # Time to 50% steady state (surface)
    target = ss_surface * 0.5
    time_to_50 = 0.0
    if target > 0:
        for i in range(len(c_surf_list)):
            if c_surf_list[i] >= target:
                time_to_50 = times_h[i]
                break

    notes = (
        f"Cartilage diffusion model: D={D_cm2_h:.6f} cm2/h, "
        f"k_diff={k_diff:.4f}/h, k_syno_in={k_syno_in:.4f}/h"
    )

    return CartilagePenetrationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_synovial_mg_L=c_synovial_list,
        c_cartilage_surface_mg_g=c_surf_list,
        c_cartilage_deep_mg_g=c_deep_list,
        diffusion_coefficient_cm2_h=D_cm2_h,
        penetration_depth_pct=penetration_depth_pct,
        steady_state_surface_mg_g=ss_surface,
        steady_state_deep_mg_g=ss_deep,
        surface_to_synovial_ratio=surface_to_synovial_ratio,
        time_to_50pct_ss_h=time_to_50,
        cartilage_auc_surface=auc_surface,
        cartilage_auc_deep=auc_deep,
        notes=notes,
    )
