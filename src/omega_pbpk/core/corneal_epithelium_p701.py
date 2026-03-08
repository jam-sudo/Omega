"""Phase 701 — Corneal Epithelium Barrier model."""

from dataclasses import dataclass, field


@dataclass
class CornealEpitheliumResult:
    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_precorneal_mg_mL: list = field(default_factory=list)
    c_epithelium_mg_g: list = field(default_factory=list)
    c_stroma_mg_mL: list = field(default_factory=list)
    c_endothelium_mg_mL: list = field(default_factory=list)
    c_aqueous_mg_mL: list = field(default_factory=list)
    kp_epithelium: float = 0.0
    kp_stroma: float = 0.0
    cmax_precorneal_mg_mL: float = 0.0
    cmax_aqueous_mg_mL: float = 0.0
    auc_precorneal: float = 0.0
    auc_aqueous: float = 0.0
    ocular_bioavailability_pct: float = 0.0
    notes: str = ""


def simulate_corneal_epithelium(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    t_end_h: float = 4.0,
    dt_h: float = 0.01,
) -> CornealEpitheliumResult:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be positive")

    # Partition coefficients
    kp_epithelium = max(0.5, min(10.0, 1.0 + logP * 1.5))
    kp_stroma = max(0.1, min(5.0, 2.0 / (1.0 + max(0.0, logP))))

    # Permeability coefficients (cm/h)
    P_epi = 0.001 * (1.0 + max(0.0, logP) * 0.5) / max(1.0, mw_Da / 300.0)
    P_str = 0.05 / (1.0 + max(0.0, logP) * 0.3)
    P_end = P_str * 0.5

    # Transfer rates (1/h)
    k_epi_transfer = max(0.01, min(5.0, P_epi * 60.0))
    k_str_transfer = max(0.01, min(5.0, P_str * 20.0))
    k_end_transfer = max(0.01, min(5.0, P_end * 50.0))

    k_drain = 30.0  # /h precorneal drainage
    k_aq_elim = 0.3  # /h aqueous humor turnover (~3h turnover)

    # Adaptive time step for stability
    dt_int = min(
        dt_h,
        0.4 / max(k_drain, k_epi_transfer, k_str_transfer, k_end_transfer, k_aq_elim),
    )

    # Volumes (mL)
    V_pre = 0.03
    V_epi = 0.01
    V_str = 0.05
    V_end = 0.002
    V_aq = 0.25

    # Initial conditions
    C_pre = dose_mg / V_pre
    C_epi = 0.0
    C_str = 0.0
    C_end = 0.0
    C_aq = 0.0

    times = []
    c_pre_list = []
    c_epi_list = []
    c_str_list = []
    c_end_list = []
    c_aq_list = []

    t = 0.0
    next_record = 0.0

    while t <= t_end_h + 1e-12:
        if t >= next_record - 1e-12:
            times.append(round(t, 6))
            c_pre_list.append(C_pre)
            c_epi_list.append(C_epi * kp_epithelium)
            c_str_list.append(C_str)
            c_end_list.append(C_end)
            c_aq_list.append(C_aq)
            next_record += dt_h

        # Fluxes
        flux_pre_to_epi = k_epi_transfer * C_pre
        flux_epi_to_str = k_str_transfer * C_epi
        flux_str_to_end = k_end_transfer * C_str
        flux_end_to_aq = k_end_transfer * 0.5 * C_end

        dC_pre = -k_drain * C_pre - flux_pre_to_epi
        dC_epi = flux_pre_to_epi * (V_pre / V_epi) - flux_epi_to_str
        dC_str = flux_epi_to_str * (V_epi / V_str) - flux_str_to_end
        dC_end = flux_str_to_end * (V_str / V_end) - flux_end_to_aq
        dC_aq = flux_end_to_aq * (V_end / V_aq) - k_aq_elim * C_aq

        C_pre = max(0.0, C_pre + dC_pre * dt_int)
        C_epi = max(0.0, C_epi + dC_epi * dt_int)
        C_str = max(0.0, C_str + dC_str * dt_int)
        C_end = max(0.0, C_end + dC_end * dt_int)
        C_aq = max(0.0, C_aq + dC_aq * dt_int)

        t += dt_int

    # Cmax
    cmax_precorneal = max(c_pre_list) if c_pre_list else 0.0
    cmax_aqueous = max(c_aq_list) if c_aq_list else 0.0

    # Manual trapezoidal AUC
    auc_precorneal = sum(
        0.5 * (c_pre_list[i] + c_pre_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )
    auc_aqueous = sum(
        0.5 * (c_aq_list[i] + c_aq_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    # Ocular bioavailability
    ocular_bioavailability_pct = min(100.0, cmax_aqueous * V_aq / dose_mg * 100)

    return CornealEpitheliumResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_precorneal_mg_mL=c_pre_list,
        c_epithelium_mg_g=c_epi_list,
        c_stroma_mg_mL=c_str_list,
        c_endothelium_mg_mL=c_end_list,
        c_aqueous_mg_mL=c_aq_list,
        kp_epithelium=kp_epithelium,
        kp_stroma=kp_stroma,
        cmax_precorneal_mg_mL=cmax_precorneal,
        cmax_aqueous_mg_mL=cmax_aqueous,
        auc_precorneal=auc_precorneal,
        auc_aqueous=auc_aqueous,
        ocular_bioavailability_pct=ocular_bioavailability_pct,
        notes="5-layer corneal epithelium barrier model with forward Euler",
    )
