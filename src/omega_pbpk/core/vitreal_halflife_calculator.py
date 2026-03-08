"""Phase 577 - Ocular Drug Vitreal Half-Life Calculator.

Simulates intravitreal drug concentration in vitreous humor, aqueous humor,
and systemic compartments using Forward Euler integration.
"""

from dataclasses import dataclass


@dataclass
class VitrealHalflifeResult:
    drug_name: str
    dose_mg: float
    mw_Da: float
    times_h: list
    c_vitreous_mg_mL: list
    c_aqueous_mg_mL: list
    c_systemic_mg_L: list
    cmax_vitreous_mg_mL: float
    t_half_vitreous_h: float
    auc_vitreous_mg_h_per_mL: float
    cmax_systemic_mg_L: float
    clearance_mechanism: str
    therapeutic_window_h: float
    mec_mg_mL: float
    notes: str


def _trapezoidal_auc(x: list, y: list) -> float:
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_vitreal_halflife(
    drug_name: str,
    dose_mg: float,
    mw_Da: float,
    cl_anterior_per_h: float = 0.02,
    cl_posterior_per_h: float = 0.005,
    mec_mg_mL: float = 0.001,
    t_end_h: float = 720.0,
    dt_h: float = 1.0,
) -> VitrealHalflifeResult:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if mec_mg_mL <= 0:
        raise ValueError("mec_mg_mL must be > 0")

    v_vit = 4.0
    v_aq = 0.25
    v_sys = 50.0  # L

    # Determine rate constants based on MW
    if mw_Da > 50000:
        k_post = cl_posterior_per_h / v_vit
        k_ant = cl_anterior_per_h * 0.1 / v_vit
        mechanism = "posterior"
    elif mw_Da < 1000:
        k_ant = cl_anterior_per_h / v_vit
        k_post = cl_posterior_per_h * 0.1 / v_vit
        mechanism = "anterior"
    else:
        k_ant = cl_anterior_per_h / v_vit
        k_post = cl_posterior_per_h / v_vit
        mechanism = "both"

    t_half = 0.693 / (k_ant + k_post)

    # Initial conditions
    c_vit = dose_mg / v_vit
    c_aq = 0.0
    c_sys = 0.0

    times = [0.0]
    c_vit_list = [c_vit]
    c_aq_list = [c_aq]
    c_sys_list = [c_sys]

    t = 0.0
    n_steps = int(t_end_h / dt_h)

    for _ in range(n_steps):
        dc_vit = -(k_ant + k_post) * c_vit
        dc_aq = k_ant * c_vit * v_vit / v_aq - 0.5 * c_aq
        dc_sys = k_post * c_vit * v_vit / (v_sys * 1000.0) - 0.02 * c_sys

        c_vit += dc_vit * dt_h
        c_aq += dc_aq * dt_h
        c_sys += dc_sys * dt_h

        t += dt_h
        times.append(t)
        c_vit_list.append(c_vit)
        c_aq_list.append(c_aq)
        c_sys_list.append(c_sys)

    cmax_vit = max(c_vit_list)
    cmax_sys = max(c_sys_list)
    auc_vit = _trapezoidal_auc(times, c_vit_list)

    # Therapeutic window: total time C_vit >= mec
    therapeutic_window = sum(dt_h for c in c_vit_list[1:] if c >= mec_mg_mL)

    notes = f"MW={mw_Da:.0f} Da, clearance via {mechanism} route. Vitreous t1/2={t_half:.1f} h."

    return VitrealHalflifeResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw_Da=mw_Da,
        times_h=times,
        c_vitreous_mg_mL=c_vit_list,
        c_aqueous_mg_mL=c_aq_list,
        c_systemic_mg_L=c_sys_list,
        cmax_vitreous_mg_mL=cmax_vit,
        t_half_vitreous_h=t_half,
        auc_vitreous_mg_h_per_mL=auc_vit,
        cmax_systemic_mg_L=cmax_sys,
        clearance_mechanism=mechanism,
        therapeutic_window_h=therapeutic_window,
        mec_mg_mL=mec_mg_mL,
        notes=notes,
    )


def compare_mw_effect(
    drug_name: str,
    dose_mg: float,
    mw_values: list = None,
) -> list:
    if mw_values is None:
        mw_values = [0.3, 1.0, 10.0, 50.0, 150.0]

    results = []
    for mw_kDa in mw_values:
        mw_Da = mw_kDa * 1000.0
        r = simulate_vitreal_halflife(drug_name, dose_mg, mw_Da)
        results.append(r)

    results.sort(key=lambda r: r.t_half_vitreous_h, reverse=True)
    return results
