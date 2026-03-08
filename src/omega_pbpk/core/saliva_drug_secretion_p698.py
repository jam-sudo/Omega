"""Phase 698 — Drug Saliva Secretion."""

from dataclasses import dataclass, field


@dataclass
class SalivaDrugSecretionResult:
    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_saliva_mg_L: list = field(default_factory=list)
    salivary_flow_mL_per_h: float = 0.0
    saliva_to_plasma_ratio_unbound: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    cmax_saliva_mg_L: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_saliva_mg_h_per_L: float = 0.0
    monitoring_feasibility: str = ""
    notes: str = ""


def simulate_saliva_drug_secretion(
    drug_name: str,
    dose_mg: float,
    logP: float,
    pka: float,
    is_acid: bool,
    fu_plasma: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    salivary_flow_mL_per_h: float = 30.0,
    saliva_ph: float = 6.8,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> SalivaDrugSecretionResult:
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be in (0, 1]")
    if salivary_flow_mL_per_h <= 0:
        raise ValueError("salivary_flow_mL_per_h must be > 0")

    plasma_ph = 7.4
    # pH partitioning
    if is_acid:
        ratio = (1.0 + 10.0 ** (saliva_ph - pka)) / (1.0 + 10.0 ** (plasma_ph - pka))
    else:
        ratio = (1.0 + 10.0 ** (pka - saliva_ph)) / (1.0 + 10.0 ** (pka - plasma_ph))
    ratio = max(0.01, min(5.0, ratio))

    # 1-cpt oral model
    ka = 1.2
    f_bio = 0.85
    ke = cl_sys_L_per_h / vd_sys_L

    a_gut = dose_mg * f_bio
    c_plasma = 0.0

    dt_int = min(dt_h, 0.4 / max(ka, ke, 1.0))
    n_steps = int(t_end_h / dt_int) + 1

    times = []
    cp_list = []
    cs_list = []

    t = 0.0
    for _i in range(n_steps):
        c_saliva = fu_plasma * c_plasma * ratio

        times.append(t)
        cp_list.append(c_plasma)
        cs_list.append(max(0.0, c_saliva))

        # Euler step
        da_gut = -ka * a_gut
        dc_plasma = ka * a_gut / vd_sys_L - ke * c_plasma

        a_gut += da_gut * dt_int
        c_plasma += dc_plasma * dt_int
        c_plasma = max(0.0, c_plasma)

        t += dt_int

    # Downsample
    out_times = []
    out_cp = []
    out_cs = []
    sample_interval = max(1, int(dt_h / dt_int + 0.5))
    for i in range(0, len(times), sample_interval):
        out_times.append(times[i])
        out_cp.append(cp_list[i])
        out_cs.append(cs_list[i])
    if out_times[-1] < times[-1]:
        out_times.append(times[-1])
        out_cp.append(cp_list[-1])
        out_cs.append(cs_list[-1])

    # Cmax
    cmax_plasma = max(out_cp) if out_cp else 0.0
    cmax_saliva = max(out_cs) if out_cs else 0.0

    # AUC (trapezoidal)
    auc_plasma = sum(
        0.5 * (out_cp[i] + out_cp[i - 1]) * (out_times[i] - out_times[i - 1])
        for i in range(1, len(out_times))
    )
    auc_saliva = sum(
        0.5 * (out_cs[i] + out_cs[i - 1]) * (out_times[i] - out_times[i - 1])
        for i in range(1, len(out_times))
    )

    # Saliva-to-plasma ratio (unbound)
    if auc_plasma > 0 and fu_plasma > 0:
        saliva_to_plasma_ratio_unbound = auc_saliva / (fu_plasma * auc_plasma)
    else:
        saliva_to_plasma_ratio_unbound = ratio

    # Monitoring feasibility
    if 0.8 <= ratio <= 1.2:
        monitoring_feasibility = "excellent"
    elif 0.5 <= ratio < 0.8 or 1.2 < ratio <= 2.0:
        monitoring_feasibility = "good"
    else:
        monitoring_feasibility = "poor"

    notes_parts = [
        f"pH partition ratio={ratio:.4f}",
        f"is_acid={is_acid}",
        f"saliva_ph={saliva_ph}",
        f"monitoring={monitoring_feasibility}",
    ]

    return SalivaDrugSecretionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=out_times,
        c_plasma_mg_L=out_cp,
        c_saliva_mg_L=out_cs,
        salivary_flow_mL_per_h=salivary_flow_mL_per_h,
        saliva_to_plasma_ratio_unbound=saliva_to_plasma_ratio_unbound,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_saliva_mg_L=cmax_saliva,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_saliva_mg_h_per_L=auc_saliva,
        monitoring_feasibility=monitoring_feasibility,
        notes="; ".join(notes_parts),
    )
