"""Pulmonary drug disposition model.

Models drug disposition in the lungs as a tissue compartment for any
administration route (IV, oral, inhaled).  Tracks lung tissue, epithelial
lining fluid (ELF), and alveolar macrophage concentrations.

Relevant for pulmonary infections, asthma, and lung cancer therapy.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PulmonaryDispositionResult:
    """Result container for pulmonary disposition simulation."""

    drug_name: str
    dose_mg: float
    route: str
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    c_lung_tissue_mg_g: list[float] = field(default_factory=list)
    c_elf_mg_L: list[float] = field(default_factory=list)
    c_macrophage_mg_g: list[float] = field(default_factory=list)
    cmax_plasma_mg_L: float = 0.0
    cmax_lung_mg_g: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_lung_mg_h_per_g: float = 0.0
    lung_to_plasma_auc_ratio: float = 0.0
    elf_to_plasma_ratio: float = 0.0
    macrophage_accumulation_fold: float = 0.0
    kp_lung: float = 0.0
    notes: str = ""


def simulate_pulmonary_disposition(
    drug_name: str,
    dose_mg: float,
    route: str,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    is_base: bool = True,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> PulmonaryDispositionResult:
    """Simulate pulmonary drug disposition for a given route."""

    # --- validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if route not in ("iv", "oral", "inhaled"):
        raise ValueError("route must be 'iv', 'oral', or 'inhaled'")

    # --- lung partitioning ---
    kp_lung = max(0.5, min(20.0, logP * 0.8 + 1.0))
    if is_base:
        kp_lung = min(20.0, kp_lung * 1.5)

    # Macrophage partition
    kp_macro = kp_lung * 5.0 if is_base else kp_lung
    kp_macro = min(100.0, kp_macro)

    # Physiological constants
    v_lung = 1.0  # L (lung tissue volume)
    q_lung = 300.0  # L/h (pulmonary blood flow)
    ka_oral = 1.0  # absorption rate for oral

    k_el = cl_sys_L_per_h / vd_sys_L

    # --- initial conditions ---
    c_plasma = 0.0
    c_lung = 0.0  # mg/g (density 1 g/mL)
    a_gut = 0.0

    if route == "iv":
        c_plasma = dose_mg / vd_sys_L
    elif route == "oral":
        a_gut = dose_mg * 0.85
    elif route == "inhaled":
        c_lung = dose_mg * 0.5 / v_lung  # 50% lung deposition

    n_steps = int(t_end_h / dt_h) + 1

    times_h: list[float] = []
    c_plasma_list: list[float] = []
    c_lung_list: list[float] = []
    c_elf_list: list[float] = []
    c_macro_list: list[float] = []

    for i in range(n_steps):
        t = i * dt_h

        # Derived concentrations
        c_elf = c_lung * kp_lung * 0.3
        c_macrophage = c_lung * kp_macro / kp_lung if kp_lung > 0 else 0.0

        times_h.append(t)
        c_plasma_list.append(c_plasma)
        c_lung_list.append(c_lung)
        c_elf_list.append(c_elf)
        c_macro_list.append(c_macrophage)

        # Oral absorption input
        oral_input = 0.0
        if a_gut > 0:
            oral_input = ka_oral * a_gut / vd_sys_L
            a_gut += (-ka_oral * a_gut) * dt_h

        # ODEs
        # dC_plasma/dt = oral_input - k_el*C_plasma - (Q/Vd)*(C_plasma - C_lung/kp_lung)
        dc_plasma = (
            oral_input - k_el * c_plasma - (q_lung / vd_sys_L) * (c_plasma - c_lung / kp_lung)
        )
        # dC_lung/dt = (Q/V_lung)*(C_plasma - C_lung/kp_lung)
        dc_lung = (q_lung / v_lung) * (c_plasma - c_lung / kp_lung)

        c_plasma += dc_plasma * dt_h
        c_lung += dc_lung * dt_h

        # Clamp to non-negative
        if c_plasma < 0:
            c_plasma = 0.0
        if c_lung < 0:
            c_lung = 0.0

    # --- metrics ---
    cmax_plasma = max(c_plasma_list)
    cmax_lung = max(c_lung_list)

    # Trapezoidal AUC
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * dt_h for i in range(1, len(c_plasma_list))
    )
    auc_lung = sum(
        0.5 * (c_lung_list[i] + c_lung_list[i - 1]) * dt_h for i in range(1, len(c_lung_list))
    )
    auc_elf = sum(
        0.5 * (c_elf_list[i] + c_elf_list[i - 1]) * dt_h for i in range(1, len(c_elf_list))
    )

    lung_to_plasma_auc = auc_lung / auc_plasma if auc_plasma > 0 else 0.0
    elf_to_plasma = auc_elf / auc_plasma if auc_plasma > 0 else 0.0

    max_c_macro = max(c_macro_list)
    macro_fold = max_c_macro / cmax_plasma if cmax_plasma > 0 else 0.0

    notes = f"route={route}, logP={logP}, MW={mw_Da}Da, is_base={is_base}, kp_lung={kp_lung:.2f}"

    return PulmonaryDispositionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_lung_tissue_mg_g=c_lung_list,
        c_elf_mg_L=c_elf_list,
        c_macrophage_mg_g=c_macro_list,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_lung_mg_g=cmax_lung,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_lung_mg_h_per_g=auc_lung,
        lung_to_plasma_auc_ratio=lung_to_plasma_auc,
        elf_to_plasma_ratio=elf_to_plasma,
        macrophage_accumulation_fold=macro_fold,
        kp_lung=kp_lung,
        notes=notes,
    )
