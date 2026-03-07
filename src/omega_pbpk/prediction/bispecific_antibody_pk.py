"""Bispecific antibody PK model — two-target TMDD simulation."""

from dataclasses import dataclass


@dataclass
class BispecificAntibodyPKResult:
    drug_name: str
    dose_mg: float
    times_h: list[float]
    c_antibody_mg_L: list[float]
    c_target1_nM: list[float]
    c_target2_nM: list[float]
    c_complex1_nM: list[float]
    c_complex2_nM: list[float]
    cmax_antibody: float
    tmax_antibody_h: float
    auc_antibody: float
    target1_engagement_pct: float
    target2_engagement_pct: float
    t_half_h: float
    notes: str


def simulate_bispecific_antibody_pk(
    drug_name: str,
    dose_mg_per_kg: float,
    body_weight_kg: float = 70.0,
    cl_L_per_day: float = 0.2,
    vd_central_L: float = 3.0,
    target1_baseline_nM: float = 10.0,
    target2_baseline_nM: float = 5.0,
    kon1_per_nM_per_day: float = 0.1,
    koff1_per_day: float = 0.01,
    kon2_per_nM_per_day: float = 0.1,
    koff2_per_day: float = 0.01,
    target_turnover_per_day: float = 0.1,
    mw_kDa: float = 150.0,
    route: str = "iv",
    f_sc: float = 0.7,
    ka_sc_per_day: float = 0.2,
    t_end_days: float = 28.0,
    dt_days: float = 0.1,
) -> BispecificAntibodyPKResult:
    """Simulate bispecific antibody PK with two-target TMDD using forward Euler.

    Parameters
    ----------
    drug_name:
        Name of the bispecific antibody.
    dose_mg_per_kg:
        Dose in mg/kg.
    body_weight_kg:
        Body weight in kg.
    cl_L_per_day:
        Linear clearance in L/day.
    vd_central_L:
        Central volume of distribution in L.
    target1_baseline_nM:
        Baseline concentration of target 1 in nM.
    target2_baseline_nM:
        Baseline concentration of target 2 in nM.
    kon1_per_nM_per_day:
        On-rate for target 1 (nM^-1 day^-1).
    koff1_per_day:
        Off-rate for target 1 (day^-1).
    kon2_per_nM_per_day:
        On-rate for target 2 (nM^-1 day^-1).
    koff2_per_day:
        Off-rate for target 2 (day^-1).
    target_turnover_per_day:
        Target turnover / degradation rate (day^-1); ksyn = turnover * baseline.
    mw_kDa:
        Molecular weight in kDa.
    route:
        Administration route: "iv" or "sc".
    f_sc:
        Subcutaneous bioavailability fraction.
    ka_sc_per_day:
        Absorption rate constant for SC route (day^-1).
    t_end_days:
        Simulation end time in days.
    dt_days:
        Integration time step in days.

    Returns
    -------
    BispecificAntibodyPKResult
    """
    if dose_mg_per_kg <= 0:
        raise ValueError("dose_mg_per_kg must be > 0")
    if body_weight_kg <= 0:
        raise ValueError("body_weight_kg must be > 0")
    if vd_central_L <= 0:
        raise ValueError("vd_central_L must be > 0")
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be > 0")
    if route not in ("iv", "sc"):
        raise ValueError("route must be 'iv' or 'sc'")

    dose_mg = dose_mg_per_kg * body_weight_kg

    # Convert dose_mg to nM in the central compartment
    # dose_mg / mw_kDa gives dose in nmol (since mw_kDa * 1000 g/mol = mw g/mol, 1mg = 1e-3g)
    # nmol / vd_L gives nM concentration
    # dose_mg * 1000 / (mw_kDa * 1000 * vd_L) nM
    dose_nM = (dose_mg * 1000.0) / (mw_kDa * 1000.0 * vd_central_L)

    # Synthesis rates (target turnover maintains baseline at steady state)
    ksyn1 = target_turnover_per_day * target1_baseline_nM
    ksyn2 = target_turnover_per_day * target2_baseline_nM
    kdeg1 = target_turnover_per_day
    kdeg2 = target_turnover_per_day

    # Initial conditions (nM)
    if route == "iv":
        c_ab = dose_nM  # free antibody in central compartment (nM)
        c_depot = 0.0
    else:
        # SC: start with depot, no immediate plasma concentration
        c_ab = 0.0
        c_depot = dose_nM * f_sc  # effective dose via SC absorption

    t1 = target1_baseline_nM  # free target 1
    t2 = target2_baseline_nM  # free target 2
    at1 = 0.0  # drug:target1 complex
    at2 = 0.0  # drug:target2 complex

    # CL in nM/day for antibody: CL_L_per_day * c_ab_nM = flux in nM*L/day
    # Since we track concentration (nM) in Vd: dC/dt terms have /Vd already embedded
    # Actually: dC_ab/dt [nM/day] = -CL/Vd * C_ab - kon1*C_ab*T1 + koff1*AT1 - kon2*C_ab*T2 + koff2*AT2
    ke = cl_L_per_day / vd_central_L  # elimination rate constant (day^-1)

    n_steps = int(t_end_days / dt_days) + 1
    times_h = []
    c_antibody_mg_L = []
    c_target1_nM = []
    c_target2_nM = []
    c_complex1_nM = []
    c_complex2_nM = []

    # Conversion factor nM -> mg/L: c_nM * mw_kDa / 1e6 * 1e9 ...
    # 1 nM = 1e-9 mol/L; mass_conc = 1e-9 * mw_Da g/L = 1e-9 * mw_kDa*1000 g/L = mw_kDa * 1e-6 g/L = mw_kDa * 1e-3 mg/L
    nM_to_mgL = mw_kDa * 1e-3

    for i in range(n_steps):
        t_day = i * dt_days
        times_h.append(t_day * 24.0)
        c_antibody_mg_L.append(c_ab * nM_to_mgL)
        c_target1_nM.append(t1)
        c_target2_nM.append(t2)
        c_complex1_nM.append(at1)
        c_complex2_nM.append(at2)

        # Derivatives
        binding1 = kon1_per_nM_per_day * c_ab * t1
        unbinding1 = koff1_per_day * at1
        binding2 = kon2_per_nM_per_day * c_ab * t2
        unbinding2 = koff2_per_day * at2

        if route == "sc" and c_depot > 0:
            sc_input = ka_sc_per_day * c_depot
        else:
            sc_input = 0.0

        dc_ab = -ke * c_ab - binding1 + unbinding1 - binding2 + unbinding2 + sc_input
        dt1 = ksyn1 - kdeg1 * t1 - binding1 + unbinding1
        dat1 = binding1 - unbinding1 - kdeg1 * at1
        dt2 = ksyn2 - kdeg2 * t2 - binding2 + unbinding2
        dat2 = binding2 - unbinding2 - kdeg2 * at2
        dc_depot = -ka_sc_per_day * c_depot if route == "sc" else 0.0

        # Forward Euler update
        c_ab = max(0.0, c_ab + dc_ab * dt_days)
        t1 = max(0.0, t1 + dt1 * dt_days)
        t2 = max(0.0, t2 + dt2 * dt_days)
        at1 = max(0.0, at1 + dat1 * dt_days)
        at2 = max(0.0, at2 + dat2 * dt_days)
        if route == "sc":
            c_depot = max(0.0, c_depot + dc_depot * dt_days)

    # PK metrics
    cmax_antibody = max(c_antibody_mg_L)
    tmax_idx = c_antibody_mg_L.index(cmax_antibody)
    tmax_antibody_h = times_h[tmax_idx]

    # Trapezoidal AUC (mg/L * h)
    auc_antibody = 0.0
    for i in range(1, len(times_h)):
        dt_h = times_h[i] - times_h[i - 1]
        auc_antibody += 0.5 * (c_antibody_mg_L[i] + c_antibody_mg_L[i - 1]) * dt_h

    # Target engagement at peak complex formation time
    # (peak binding may occur after Cmax, especially for IV bolus)
    total_complex = [c_complex1_nM[i] + c_complex2_nM[i] for i in range(len(c_complex1_nM))]
    peak_complex_idx = total_complex.index(max(total_complex))

    t1_at_peak = c_target1_nM[peak_complex_idx]
    at1_at_peak = c_complex1_nM[peak_complex_idx]
    total_t1 = t1_at_peak + at1_at_peak
    target1_engagement_pct = (at1_at_peak / total_t1 * 100.0) if total_t1 > 0 else 0.0

    t2_at_peak = c_target2_nM[peak_complex_idx]
    at2_at_peak = c_complex2_nM[peak_complex_idx]
    total_t2 = t2_at_peak + at2_at_peak
    target2_engagement_pct = (at2_at_peak / total_t2 * 100.0) if total_t2 > 0 else 0.0

    # Approximate terminal half-life: find where Cmax drops to 50%
    half_cmax = cmax_antibody / 2.0
    t_half_h = float("inf")
    for i in range(tmax_idx, len(c_antibody_mg_L) - 1):
        if c_antibody_mg_L[i] >= half_cmax > c_antibody_mg_L[i + 1]:
            # Linear interpolation
            dt_h = times_h[i + 1] - times_h[i]
            dc = c_antibody_mg_L[i + 1] - c_antibody_mg_L[i]
            if dc != 0:
                frac = (half_cmax - c_antibody_mg_L[i]) / dc
                t_half_h = times_h[i] + frac * dt_h - tmax_antibody_h
            else:
                t_half_h = times_h[i] - tmax_antibody_h
            break

    notes = (
        f"Route: {route}, Dose: {dose_mg:.1f} mg, "
        f"Target 1 engagement: {target1_engagement_pct:.1f}%, "
        f"Target 2 engagement: {target2_engagement_pct:.1f}%"
    )

    return BispecificAntibodyPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_antibody_mg_L=c_antibody_mg_L,
        c_target1_nM=c_target1_nM,
        c_target2_nM=c_target2_nM,
        c_complex1_nM=c_complex1_nM,
        c_complex2_nM=c_complex2_nM,
        cmax_antibody=cmax_antibody,
        tmax_antibody_h=tmax_antibody_h,
        auc_antibody=auc_antibody,
        target1_engagement_pct=target1_engagement_pct,
        target2_engagement_pct=target2_engagement_pct,
        t_half_h=t_half_h,
        notes=notes,
    )
