"""Phase 864 — Drug Penetration into Biofilm Model."""

import math
from dataclasses import dataclass, field


@dataclass
class BiofilmPenetrationResult:
    drug_name: str
    dose_mg: float
    biofilm_thickness_um: float = 0.0
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_biofilm_surface_mg_L: list = field(default_factory=list)
    c_biofilm_deep_mg_L: list = field(default_factory=list)
    cmax_plasma_mg_L: float = 0.0
    cmax_surface_mg_L: float = 0.0
    cmax_deep_mg_L: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_surface_mg_h_per_L: float = 0.0
    penetration_ratio: float = 0.0
    mbec_ratio: float = 0.0
    therapeutic_efficacy: str = ""
    notes: str = ""


def simulate_biofilm_penetration(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    biofilm_thickness_um: float = 100.0,
    mbec_mg_L: float = 8.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> BiofilmPenetrationResult:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if biofilm_thickness_um <= 0:
        raise ValueError("biofilm_thickness_um must be > 0")
    if mbec_mg_L <= 0:
        raise ValueError("mbec_mg_L must be > 0")

    # Rate constants
    ka = 1.2
    ke = cl_sys_L_per_h / vd_sys_L

    # Diffusion into surface layer
    k_surface = 0.5 / max(1.0, mw_Da / 300.0) * max(0.1, 1.0 + logP * 0.2)
    k_surface = max(0.01, min(2.0, k_surface))

    # Diffusion into deep layer
    k_deep = k_surface * 50.0 / max(50.0, biofilm_thickness_um)
    k_deep = max(0.001, min(0.5, k_deep))

    k_bio_out = 0.05

    # Adaptive time step
    dt_int = min(dt_h, 0.4 / max(1.2, ke, k_surface, k_deep, k_bio_out))

    # Initial conditions
    a_gut = dose_mg * 0.85
    c_plasma = 0.0
    c_surface = 0.0
    c_deep = 0.0

    times_h = []
    c_plasma_list = []
    c_surface_list = []
    c_deep_list = []

    t = 0.0
    n_steps = int(math.ceil(t_end_h / dt_int))

    for _ in range(n_steps + 1):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        c_surface_list.append(c_surface)
        c_deep_list.append(c_deep)

        # Forward Euler
        da_gut = -ka * a_gut
        dc_plasma = ka * a_gut / vd_sys_L - ke * c_plasma
        dc_surface = k_surface * c_plasma - k_bio_out * c_surface - k_deep * (c_surface - c_deep)
        dc_deep = k_deep * (c_surface - c_deep) - k_bio_out * c_deep * 0.5

        a_gut += da_gut * dt_int
        c_plasma += dc_plasma * dt_int
        c_surface += dc_surface * dt_int
        c_deep += dc_deep * dt_int

        a_gut = max(0.0, a_gut)
        c_plasma = max(0.0, c_plasma)
        c_surface = max(0.0, c_surface)
        c_deep = max(0.0, c_deep)

        t += dt_int

    # Metrics
    cmax_plasma = max(c_plasma_list) if c_plasma_list else 0.0
    cmax_surface = max(c_surface_list) if c_surface_list else 0.0
    cmax_deep = max(c_deep_list) if c_deep_list else 0.0

    # Trapezoidal AUC
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_surface = sum(
        0.5 * (c_surface_list[i] + c_surface_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    penetration_ratio = cmax_deep / cmax_surface if cmax_surface > 0 else 0.0
    mbec_ratio = cmax_deep / mbec_mg_L if mbec_mg_L > 0 else 0.0

    if mbec_ratio >= 1.0:
        therapeutic_efficacy = "effective"
    elif mbec_ratio >= 0.5:
        therapeutic_efficacy = "partially_effective"
    else:
        therapeutic_efficacy = "ineffective"

    return BiofilmPenetrationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        biofilm_thickness_um=biofilm_thickness_um,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_biofilm_surface_mg_L=c_surface_list,
        c_biofilm_deep_mg_L=c_deep_list,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_surface_mg_L=cmax_surface,
        cmax_deep_mg_L=cmax_deep,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_surface_mg_h_per_L=auc_surface,
        penetration_ratio=penetration_ratio,
        mbec_ratio=mbec_ratio,
        therapeutic_efficacy=therapeutic_efficacy,
        notes=f"Biofilm penetration for {drug_name}, thickness={biofilm_thickness_um} um",
    )
