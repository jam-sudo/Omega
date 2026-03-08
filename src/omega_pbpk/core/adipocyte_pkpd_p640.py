"""Phase 640 — Drug Adipocyte PK/PD.

Models drug pharmacokinetics in adipocytes and pharmacodynamic effects
on adipocyte function (lipid metabolism / lipolysis).
"""

from dataclasses import dataclass


@dataclass
class AdipocytePKPDResult:
    """Result of adipocyte PK/PD simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_adipocyte_mg_g: list
    lipase_activity: list
    lipolysis_rate: list
    fatty_acid_release_mg_h: list
    cmax_plasma_mg_L: float
    auc_plasma_mg_h_per_L: float
    cumulative_lipolysis_mg: float
    lipase_inhibition_pct: float
    drug_effect_class: str
    notes: str


def _trapezoidal_auc(times, values):
    """Manual trapezoidal AUC."""
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def simulate_adipocyte_pkpd(
    drug_name,
    dose_mg,
    logP,
    mw_Da,
    cl_sys_L_per_h,
    vd_sys_L,
    ic50_lipase_mg_L=None,
    emax_lipase=0.9,
    baseline_lipolysis_mg_h=500.0,
    t_end_h=24.0,
    dt_h=0.1,
):
    """Simulate drug PK in adipocytes and PD effects on lipolysis.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Oral dose in mg.
    logP : float
        Octanol-water partition coefficient (log scale).
    mw_Da : float
        Molecular weight in Daltons (unused in core model but kept for API consistency).
    cl_sys_L_per_h : float
        Systemic clearance in L/h.
    vd_sys_L : float
        Volume of distribution (systemic) in L.
    ic50_lipase_mg_L : float or None
        Concentration inhibiting lipase by 50% (mg/L). If None, estimated from logP.
    emax_lipase : float
        Maximum fractional inhibition of lipase (0-1).
    baseline_lipolysis_mg_h : float
        Baseline lipolysis rate (mg triglyceride/h).
    t_end_h : float
        Simulation end time in hours.
    dt_h : float
        Time step in hours.

    Returns
    -------
    AdipocytePKPDResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if ic50_lipase_mg_L is not None and ic50_lipase_mg_L <= 0:
        raise ValueError("ic50_lipase_mg_L must be > 0 if provided")
    if not (0.0 <= emax_lipase <= 1.0):
        raise ValueError("emax_lipase must be in [0, 1]")
    if baseline_lipolysis_mg_h <= 0:
        raise ValueError("baseline_lipolysis_mg_h must be > 0")

    # Estimate IC50 from logP if not provided
    if ic50_lipase_mg_L is None:
        ic50_lipase_mg_L = max(0.01, 10.0 / (1.0 + max(0.0, logP) * 2.0))

    # Adipose partition coefficient
    kp_adip = max(0.5, logP * 2.0 + 0.5)
    kp_adip = max(0.5, min(20.0, kp_adip))

    # Adipose compartment
    fat_mass_kg = 15.0  # default: 20% of 75 kg body weight
    vd_adip_L = fat_mass_kg * kp_adip

    # Rate constants
    ka = 1.2  # oral absorption /h
    k_elim = cl_sys_L_per_h / vd_sys_L
    # Adipose blood flow ~0.5 L/h total
    q_adip = 0.5  # L/h
    k_in = q_adip / vd_sys_L  # plasma → adipocyte /h
    k_out = q_adip / vd_adip_L  # adipocyte → plasma /h

    # Initial conditions
    A_gut = dose_mg * 0.85
    C_plasma = 0.0
    C_adip = 0.0

    n_steps = int(t_end_h / dt_h) + 1
    times_h = []
    c_plasma_list = []
    c_adip_L_list = []  # adipocyte concentration in mg/L

    for step in range(n_steps):
        t = step * dt_h
        times_h.append(t)
        c_plasma_list.append(max(0.0, C_plasma))
        c_adip_L_list.append(max(0.0, C_adip))

        # ODEs
        dA_gut = -ka * A_gut
        dC_plasma = (
            ka * A_gut / vd_sys_L
            - k_elim * C_plasma
            - k_in * C_plasma
            + k_out * C_adip * vd_adip_L / vd_sys_L
        )
        dC_adip = k_in * C_plasma * vd_sys_L / vd_adip_L - k_out * C_adip

        # Forward Euler
        A_gut = max(0.0, A_gut + dA_gut * dt_h)
        C_plasma = max(0.0, C_plasma + dC_plasma * dt_h)
        C_adip = max(0.0, C_adip + dC_adip * dt_h)

    # Convert adipocyte concentration to mg/g fat
    # c_adip_mg_g = C_adip [mg/L] * Vd_adip_L [L] / (fat_mass_kg * 1000 [g])
    fat_mass_g = fat_mass_kg * 1000.0
    c_adipocyte_mg_g_list = [c * vd_adip_L / fat_mass_g for c in c_adip_L_list]

    # PD: lipase activity and lipolysis
    lipase_activity_list = []
    lipolysis_rate_list = []
    fatty_acid_release_list = []

    for c_adip in c_adip_L_list:
        la = 1.0 - emax_lipase * c_adip / (ic50_lipase_mg_L + c_adip)
        la = max(0.01, min(2.0, la))
        lysis = baseline_lipolysis_mg_h * la
        fa_release = lysis * 0.9

        lipase_activity_list.append(la)
        lipolysis_rate_list.append(lysis)
        fatty_acid_release_list.append(fa_release)

    # Summary metrics
    cmax_plasma = max(c_plasma_list)
    auc_plasma = _trapezoidal_auc(times_h, c_plasma_list)

    # Cumulative lipolysis via trapezoid
    cumulative_lipolysis_mg = _trapezoidal_auc(times_h, lipolysis_rate_list)

    # Lipase inhibition % at minimum lipase activity (corresponds to Cmax adipose)
    min_lipase_activity = min(lipase_activity_list)
    lipase_inhibition_pct = (1.0 - min_lipase_activity) * 100.0

    # Drug effect classification by mean lipase activity
    mean_lipase_activity = sum(lipase_activity_list) / len(lipase_activity_list)
    if mean_lipase_activity < 0.7:
        drug_effect_class = "antilipolytic"
    elif mean_lipase_activity < 0.95:
        drug_effect_class = "moderate_inhibition"
    else:
        drug_effect_class = "neutral"

    notes = (
        f"kp_adip={kp_adip:.2f}; "
        f"Vd_adip={vd_adip_L:.1f} L; "
        f"ic50_lipase={ic50_lipase_mg_L:.4f} mg/L; "
        f"mean_lipase_activity={mean_lipase_activity:.3f}"
    )

    return AdipocytePKPDResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_adipocyte_mg_g=c_adipocyte_mg_g_list,
        lipase_activity=lipase_activity_list,
        lipolysis_rate=lipolysis_rate_list,
        fatty_acid_release_mg_h=fatty_acid_release_list,
        cmax_plasma_mg_L=cmax_plasma,
        auc_plasma_mg_h_per_L=auc_plasma,
        cumulative_lipolysis_mg=cumulative_lipolysis_mg,
        lipase_inhibition_pct=lipase_inhibition_pct,
        drug_effect_class=drug_effect_class,
        notes=notes,
    )
