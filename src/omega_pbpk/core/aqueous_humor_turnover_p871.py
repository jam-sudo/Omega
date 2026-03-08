"""Phase 871 — Aqueous Humor Drug Turnover.

Models drug concentration dynamics in aqueous humor accounting for
production and drainage via trabecular meshwork and uveoscleral outflow.
"""

import math
from dataclasses import dataclass, field


@dataclass
class AqueousHumorTurnoverResult:
    drug_name: str
    dose_mg: float
    route: str
    times_h: list[float] = field(default_factory=list)
    c_aqueous_mg_L: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    aq_production_rate_mL_per_h: float = 0.0
    aq_volume_mL: float = 0.0
    cmax_aqueous_mg_L: float = 0.0
    auc_aqueous_mg_h_per_L: float = 0.0
    trabecular_outflow_pct: float = 0.0
    uveoscleral_outflow_pct: float = 0.0
    residence_time_h: float = 0.0
    iop_effect_mmHg: float = 0.0
    notes: str = ""


def simulate_aqueous_humor_turnover(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    route: str = "topical",
    aq_production_mL_per_min: float = 0.003,
    t_end_h: float = 8.0,
    dt_h: float = 0.02,
) -> AqueousHumorTurnoverResult:
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if aq_production_mL_per_min <= 0:
        raise ValueError("aq_production_mL_per_min must be > 0")
    if route not in {"topical", "systemic"}:
        raise ValueError("route must be 'topical' or 'systemic'")

    aq_volume_mL = 0.25
    aq_production_rate_mL_per_h = aq_production_mL_per_min * 60.0
    k_drain = aq_production_mL_per_min * 60.0 / aq_volume_mL  # /h
    trabecular_outflow_pct = 80.0
    uveoscleral_outflow_pct = 20.0
    residence_time_h = 1.0 / k_drain

    # IOP effect
    iop_effect_mmHg = -max(0.0, logP * 0.5)
    iop_effect_mmHg = max(-10.0, min(0.0, iop_effect_mmHg))

    # Route-specific parameters and adaptive dt
    if route == "topical":
        k_corneal_input = 0.05 * max(0.1, logP + 1.0) / max(1.0, mw_Da / 300.0)
        k_corneal_input = max(0.001, min(0.5, k_corneal_input))
        k_pre_drain = 30.0
        c_pre = dose_mg / 0.03  # mg/L (dose in mg, volume 0.03 mL)
        dt_int = min(dt_h, 0.4 / max(k_drain, k_corneal_input, k_pre_drain))
    else:
        ka = 1.2
        a_gut = dose_mg * 0.85
        ke = cl_sys_L_per_h / vd_sys_L
        k_bab = 0.01 * max(0.1, logP) / max(1.0, mw_Da / 300.0)
        k_bab = max(0.001, min(0.1, k_bab))
        dt_int = min(dt_h, 0.4 / max(k_drain, k_bab, ka))

    # Time stepping
    n_steps = int(math.ceil(t_end_h / dt_int))
    dt_int = t_end_h / n_steps

    times = []
    c_aqueous_list = []
    c_plasma_list = []

    c_aq = 0.0

    if route == "topical":
        c_plasma = 0.0
        for i in range(n_steps + 1):
            t = i * dt_int
            times.append(t)
            c_aqueous_list.append(c_aq)
            c_plasma_list.append(c_plasma)

            if i < n_steps:
                dc_pre = -(k_pre_drain + k_corneal_input) * c_pre
                dc_aq = k_corneal_input * c_pre * (0.03 / aq_volume_mL) - k_drain * c_aq
                c_pre += dc_pre * dt_int
                c_aq += dc_aq * dt_int
                if c_pre < 0:
                    c_pre = 0.0
                if c_aq < 0:
                    c_aq = 0.0
    else:
        # Systemic route
        for i in range(n_steps + 1):
            t = i * dt_int
            # 1-compartment oral
            if ka != ke:
                c_plasma = (ka * a_gut / (vd_sys_L * (ka - ke))) * (
                    math.exp(-ke * t) - math.exp(-ka * t)
                )
            else:
                c_plasma = (a_gut / vd_sys_L) * t * math.exp(-ke * t)
            c_plasma = max(0.0, c_plasma)

            times.append(t)
            c_aqueous_list.append(c_aq)
            c_plasma_list.append(c_plasma)

            if i < n_steps:
                dc_aq = k_bab * c_plasma - k_drain * c_aq
                c_aq += dc_aq * dt_int
                if c_aq < 0:
                    c_aq = 0.0

    # Cmax
    cmax_aqueous = max(c_aqueous_list) if c_aqueous_list else 0.0

    # AUC (trapezoidal)
    auc_aqueous = sum(
        0.5 * (c_aqueous_list[i] + c_aqueous_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    return AqueousHumorTurnoverResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times,
        c_aqueous_mg_L=c_aqueous_list,
        c_plasma_mg_L=c_plasma_list,
        aq_production_rate_mL_per_h=aq_production_rate_mL_per_h,
        aq_volume_mL=aq_volume_mL,
        cmax_aqueous_mg_L=cmax_aqueous,
        auc_aqueous_mg_h_per_L=auc_aqueous,
        trabecular_outflow_pct=trabecular_outflow_pct,
        uveoscleral_outflow_pct=uveoscleral_outflow_pct,
        residence_time_h=residence_time_h,
        iop_effect_mmHg=iop_effect_mmHg,
        notes=f"Aqueous humor turnover for {drug_name} via {route} route",
    )
