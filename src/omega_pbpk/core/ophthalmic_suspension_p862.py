"""Phase 862 — Topical Ophthalmic Suspension PK model."""

from dataclasses import dataclass, field


@dataclass
class OphthalmicSuspensionResult:
    drug_name: str
    dose_mg: float
    particle_size_um: float = 0.0
    times_h: list = field(default_factory=list)
    c_suspension_mg_mL: list = field(default_factory=list)
    c_dissolved_mg_mL: list = field(default_factory=list)
    c_cornea_mg_g: list = field(default_factory=list)
    c_aqueous_mg_L: list = field(default_factory=list)
    dissolution_rate_mg_h: list = field(default_factory=list)
    f_dissolved: float = 0.0
    cmax_aqueous_mg_L: float = 0.0
    tmax_aqueous_h: float = 0.0
    auc_aqueous_mg_h_per_L: float = 0.0
    bioavailability_ocular_pct: float = 0.0
    notes: str = ""


def simulate_ophthalmic_suspension(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    solubility_mg_mL: float,
    particle_size_um: float = 10.0,
    cl_sys_L_per_h: float = 5.0,
    vd_sys_L: float = 50.0,
    t_end_h: float = 4.0,
    dt_h: float = 0.02,
) -> OphthalmicSuspensionResult:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if particle_size_um <= 0:
        raise ValueError("particle_size_um must be positive")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be positive")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be positive")

    # Precorneal volume
    vol_pre = 0.03  # mL

    # Drainage
    k_drain = 30.0  # /h

    # Dissolution rate (Noyes-Whitney simplified)
    D_coeff = 0.4e-6  # cm^2/s
    particle_radius_cm = particle_size_um / 1e4 / 2.0
    k_diss_rate = 3 * D_coeff / ((particle_size_um / 1e4) * 10) * 3600
    k_diss_rate = max(0.1, min(100.0, k_diss_rate))

    # Corneal absorption
    k_cornea = 0.02 * max(0.1, logP + 1.0) / max(1.0, mw_Da / 300.0)
    k_cornea = max(0.001, min(0.5, k_cornea))

    # Corneal partition
    kp_cornea = max(0.5, min(5.0, 1.0 + logP * 0.5))

    # Aqueous humor
    vol_aq = 0.25  # mL
    vol_aq_L = vol_aq / 1000.0  # 2.5e-4 L
    k_aq = 0.5  # /h

    # Adaptive time step
    dt_int = min(dt_h, 0.4 / max(k_drain, k_cornea, k_aq, k_diss_rate))

    # Initial conditions
    A_susp = dose_mg  # mg undissolved
    C_dissolved = 0.0  # mg/mL
    C_aq = 0.0  # mg/L

    times = [0.0]
    c_susp_list = [A_susp / vol_pre]
    c_diss_list = [C_dissolved]
    c_cornea_list = [0.0]
    c_aq_list = [C_aq]
    diss_rate_list = [0.0]

    t = 0.0
    while t < t_end_h - 1e-12:
        dt_step = min(dt_int, t_end_h - t)

        # Dissolution rate
        if C_dissolved < solubility_mg_mL and A_susp > 0:
            rate_dissolve = (
                k_diss_rate * A_susp * (1.0 - C_dissolved / solubility_mg_mL)
            )
            rate_dissolve = max(0.0, rate_dissolve)
        else:
            rate_dissolve = 0.0

        # ODEs
        dA_susp = -rate_dissolve
        dC_dissolved = rate_dissolve / vol_pre - k_drain * C_dissolved - k_cornea * C_dissolved
        vol_pre_L = vol_pre / 1000.0
        dC_aq = k_cornea * C_dissolved * vol_pre_L / vol_aq_L - k_aq * C_aq

        A_susp = max(0.0, A_susp + dA_susp * dt_step)
        C_dissolved = max(0.0, C_dissolved + dC_dissolved * dt_step)
        C_aq = max(0.0, C_aq + dC_aq * dt_step)

        c_cornea = C_dissolved * kp_cornea * 0.1

        t += dt_step
        times.append(t)
        c_susp_list.append(A_susp / vol_pre)
        c_diss_list.append(C_dissolved)
        c_cornea_list.append(c_cornea)
        c_aq_list.append(C_aq)
        diss_rate_list.append(max(0.0, rate_dissolve))

    # Metrics
    cmax_aq = max(c_aq_list)
    tmax_aq = times[c_aq_list.index(cmax_aq)]

    auc_aq = sum(
        0.5 * (c_aq_list[i] + c_aq_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    f_dissolved = 1.0 - A_susp / dose_mg

    bioavailability_ocular_pct = min(
        100.0, f_dissolved * k_cornea / (k_cornea + k_drain) * 100
    )

    return OphthalmicSuspensionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        particle_size_um=particle_size_um,
        times_h=times,
        c_suspension_mg_mL=c_susp_list,
        c_dissolved_mg_mL=c_diss_list,
        c_cornea_mg_g=c_cornea_list,
        c_aqueous_mg_L=c_aq_list,
        dissolution_rate_mg_h=diss_rate_list,
        f_dissolved=f_dissolved,
        cmax_aqueous_mg_L=cmax_aq,
        tmax_aqueous_h=tmax_aq,
        auc_aqueous_mg_h_per_L=auc_aq,
        bioavailability_ocular_pct=bioavailability_ocular_pct,
        notes="Ophthalmic suspension: particle dissolution + corneal penetration + aqueous humor PK",
    )
