"""Phase 872 — Skin Appendage Drug Distribution.

Models drug distribution to skin appendages (hair follicle, sebaceous gland,
sweat gland, nail).
"""

import math
from dataclasses import dataclass, field

APPENDAGE_PROPS = {
    "hair_follicle": {"mass_g": 10.0, "blood_flow_L_per_h": 0.1},
    "sebaceous_gland": {"mass_g": 5.0, "blood_flow_L_per_h": 0.05},
    "sweat_gland": {"mass_g": 50.0, "blood_flow_L_per_h": 0.2},
    "nail": {"mass_g": 5.0, "blood_flow_L_per_h": 0.01},
}

VALID_APPENDAGES = set(APPENDAGE_PROPS.keys())


@dataclass
class SkinAppendagePKResult:
    drug_name: str
    dose_mg: float
    appendage: str
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    c_appendage_mg_g: list[float] = field(default_factory=list)
    kp_appendage: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    cmax_appendage_mg_g: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_appendage_mg_h_per_g: float = 0.0
    appendage_to_plasma_ratio: float = 0.0
    t_half_appendage_h: float = 0.0
    notes: str = ""


def _compute_kp(appendage: str, logP: float, mw_Da: float) -> float:
    if appendage == "hair_follicle":
        return max(0.3, min(8.0, (logP + 1.5) * 0.6 / max(1.0, mw_Da / 300.0)))
    elif appendage == "sebaceous_gland":
        return max(0.5, min(15.0, (logP + 2.0) * 1.0 / max(1.0, mw_Da / 300.0)))
    elif appendage == "sweat_gland":
        return max(0.1, min(3.0, 1.5 / (1.0 + max(0.0, logP)) / max(1.0, mw_Da / 300.0)))
    else:  # nail
        return max(0.05, min(2.0, 0.3 / max(1.0, mw_Da / 200.0)))


def simulate_skin_appendage_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    appendage: str = "hair_follicle",
    t_end_h: float = 48.0,
    dt_h: float = 0.05,
) -> SkinAppendagePKResult:
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if appendage not in VALID_APPENDAGES:
        raise ValueError(f"appendage must be one of {VALID_APPENDAGES}")

    props = APPENDAGE_PROPS[appendage]
    mass_g = props["mass_g"]
    q_app = props["blood_flow_L_per_h"]

    kp = _compute_kp(appendage, logP, mw_Da)
    vd_app = max(mass_g / 1000.0, mass_g * kp / 1000.0)

    ka = 1.2
    a_gut = dose_mg * 0.85
    ke = cl_sys_L_per_h / vd_sys_L

    k_in = q_app / vd_sys_L
    k_out = q_app / vd_app
    t_half_appendage = math.log(2) / k_out

    dt_int = min(dt_h, 0.4 / max(ka, ke, k_in, k_out))
    n_steps = int(math.ceil(t_end_h / dt_int))
    dt_int = t_end_h / n_steps

    times = []
    c_plasma_list = []
    c_appendage_list = []

    c_app = 0.0  # concentration in appendage compartment (mg/L equivalent)

    for i in range(n_steps + 1):
        t = i * dt_int

        # Plasma: 1-cpt oral
        if ka != ke:
            c_plasma = (ka * a_gut / (vd_sys_L * (ka - ke))) * (
                math.exp(-ke * t) - math.exp(-ka * t)
            )
        else:
            c_plasma = (a_gut / vd_sys_L) * t * math.exp(-ke * t)
        c_plasma = max(0.0, c_plasma)

        c_app_mg_g = c_app * kp / 1000.0

        times.append(t)
        c_plasma_list.append(c_plasma)
        c_appendage_list.append(c_app_mg_g)

        if i < n_steps:
            dc_app = k_in * c_plasma - k_out * c_app
            c_app += dc_app * dt_int
            if c_app < 0:
                c_app = 0.0

    cmax_plasma = max(c_plasma_list) if c_plasma_list else 0.0
    cmax_appendage = max(c_appendage_list) if c_appendage_list else 0.0

    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )
    auc_appendage = sum(
        0.5 * (c_appendage_list[i] + c_appendage_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    appendage_to_plasma_ratio = auc_appendage / auc_plasma if auc_plasma > 0 else 0.0

    return SkinAppendagePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        appendage=appendage,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_appendage_mg_g=c_appendage_list,
        kp_appendage=kp,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_appendage_mg_g=cmax_appendage,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_appendage_mg_h_per_g=auc_appendage,
        appendage_to_plasma_ratio=appendage_to_plasma_ratio,
        t_half_appendage_h=t_half_appendage,
        notes=f"Skin appendage PK for {drug_name} in {appendage}",
    )
