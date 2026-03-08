"""
Phase 404 — Hepatic Sinusoidal Drug Transport

Models drug uptake and efflux transport in hepatic sinusoids including
OATP-mediated uptake and MRP2/BSEP-mediated efflux.
"""

from dataclasses import dataclass


@dataclass
class HepaticSinusoidalResult:
    """Result of hepatic sinusoidal transport simulation. Contains list fields."""

    drug_name: str
    times_h: list
    c_sinusoidal_mg_L: list  # blood-side sinusoidal concentration
    c_hepatocyte_mg_L: list  # intracellular hepatocyte concentration
    c_bile_mg_L: list  # bile canalicular concentration
    hepatic_extraction_ratio: float
    biliary_excretion_fraction: float
    intracellular_accumulation_fold: float  # hepatocyte/blood ratio at SS
    oatp_uptake_rate: float  # effective uptake clearance (mL/min)
    efflux_mrp2_rate: float  # bile efflux rate
    auc_hepatocyte_mg_h_per_L: float
    auc_sinusoidal_mg_h_per_L: float
    notes: str


def _trapezoidal_auc(x: list, y: list) -> float:
    """Manual trapezoidal AUC integration."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_hepatic_sinusoidal_transport(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_L: float,
    oatp_vmax_uM_per_min: float = 10.0,
    oatp_km_uM: float = 5.0,
    mrp2_cl_mL_per_min: float = 2.0,
    passive_uptake_per_h: float = 0.5,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> HepaticSinusoidalResult:
    """
    Simulate hepatic sinusoidal drug transport.

    Models:
    - Systemic 1-compartment plasma kinetics
    - OATP-mediated Michaelis-Menten uptake into hepatocytes
    - Passive diffusion uptake into hepatocytes
    - MRP2-mediated efflux from hepatocytes to bile
    - Metabolic clearance in hepatocytes
    - Bile compartment kinetics

    Parameters
    ----------
    drug_name : str
    dose_mg : float
        IV dose in mg.
    cl_sys_L_per_h : float
        Systemic clearance in L/h.
    vd_L : float
        Volume of distribution in L.
    oatp_vmax_uM_per_min : float
        OATP maximum uptake rate in uM/min.
    oatp_km_uM : float
        OATP Michaelis constant in uM.
    mrp2_cl_mL_per_min : float
        MRP2 intrinsic clearance in mL/min.
    passive_uptake_per_h : float
        Passive uptake rate constant in 1/h.
    t_end_h : float
        Simulation end time in hours.
    dt_h : float
        Time step in hours.

    Returns
    -------
    HepaticSinusoidalResult
    """
    # --- Input validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be positive, got {cl_sys_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be positive, got {vd_L}")
    if oatp_vmax_uM_per_min <= 0:
        raise ValueError(f"oatp_vmax_uM_per_min must be positive, got {oatp_vmax_uM_per_min}")
    if oatp_km_uM <= 0:
        raise ValueError(f"oatp_km_uM must be positive, got {oatp_km_uM}")
    if mrp2_cl_mL_per_min <= 0:
        raise ValueError(f"mrp2_cl_mL_per_min must be positive, got {mrp2_cl_mL_per_min}")

    # --- Constants ---
    MW_DEFAULT = 400.0  # Da, default molecular weight for unit conversion
    HEPATIC_BLOOD_FLOW = 90.0  # L/h
    K_BILE_ELIM = 0.1  # h^-1, slow bile elimination

    # Hepatocyte volume fraction of Vd
    _v_hepatocyte = vd_L * 0.3  # noqa: F841

    # Elimination rate constant
    k_elim = cl_sys_L_per_h / vd_L  # h^-1

    # Metabolic clearance in hepatocytes (proxy)
    k_met = cl_sys_L_per_h / vd_L  # h^-1

    # MRP2 efflux: convert mL/min to L/h
    # mL/min * (1 L / 1000 mL) * (60 min / h) = L/h * (1/1000) * 60
    mrp2_cl_L_per_h = mrp2_cl_mL_per_min / 1000.0 * 60.0

    # OATP effective uptake clearance (representative value in mL/min)
    # For notes/reporting: use Vmax at half-saturation as proxy
    oatp_uptake_rate_mL_min = oatp_vmax_uM_per_min / 2.0  # at Km, rate = Vmax/2

    # MRP2 rate for reporting
    efflux_mrp2_rate = mrp2_cl_mL_per_min

    # --- Initial conditions ---
    c_sin = dose_mg / vd_L  # mg/L plasma at t=0
    c_hep = 0.0  # mg/L hepatocyte at t=0
    c_bile = 0.0  # mg/L bile at t=0

    # --- Time array ---
    n_steps = int(t_end_h / dt_h) + 1
    times = [i * dt_h for i in range(n_steps)]

    c_sin_arr = [0.0] * n_steps
    c_hep_arr = [0.0] * n_steps
    c_bile_arr = [0.0] * n_steps

    c_sin_arr[0] = c_sin
    c_hep_arr[0] = c_hep
    c_bile_arr[0] = c_bile

    # --- Forward Euler simulation ---
    for i in range(1, n_steps):
        # --- Sinusoidal / plasma (1-cpt) ---
        dc_sin_dt = -k_elim * c_sin
        c_sin_new = c_sin + dc_sin_dt * dt_h
        c_sin_new = max(0.0, c_sin_new)

        # --- OATP Michaelis-Menten uptake ---
        # Convert plasma mg/L to uM: (mg/L) / (MW g/mol) * 1000 = mmol/L = mM -> *1000 = uM
        # mg/L / (400 g/mol) * 1000 mg/g * 1000 = uM
        # mg/L * 1000 / MW_g_mol = umol/L = uM
        c_uM = c_sin * 1000.0 / MW_DEFAULT  # uM
        # Michaelis-Menten rate in uM/min
        uptake_uM_per_min = oatp_vmax_uM_per_min * c_uM / (oatp_km_uM + c_uM)
        # Convert to mg/L/h: uM/min * (MW Da / 1000) * 60 min/h
        # uM * MW_Da/1000 = mg/L, so uM/min * MW_Da/1000 * 60 = mg/L/h
        oatp_uptake_mg_L_per_h = uptake_uM_per_min * (MW_DEFAULT / 1000.0) * 60.0

        # --- Total influx to hepatocyte ---
        passive_influx = passive_uptake_per_h * c_sin
        total_influx = oatp_uptake_mg_L_per_h + passive_influx

        # --- MRP2 efflux from hepatocyte to bile ---
        efflux_to_bile = mrp2_cl_L_per_h * c_hep  # mg/L/h (L/h * mg/L = mg/h, need per unit volume)
        # Actually: efflux rate per unit hepatocyte volume = cl * concentration
        # mrp2_cl_L_per_h in L/h, c_hep in mg/L → efflux = cl * c_hep (mg/h per L of hepatocyte)

        # --- Metabolic clearance in hepatocyte ---
        met_clearance = k_met * c_hep  # mg/L/h

        # --- Hepatocyte ODE ---
        dc_hep_dt = total_influx - efflux_to_bile - met_clearance
        c_hep_new = c_hep + dc_hep_dt * dt_h
        c_hep_new = max(0.0, c_hep_new)

        # --- Bile ODE ---
        # Efflux from hepatocyte contributes to bile (scaled by volume ratio)
        # Bile volume is small; use proportional flux
        bile_inflow = efflux_to_bile  # mg/L/h (same units as hepatocyte)
        dc_bile_dt = bile_inflow - K_BILE_ELIM * c_bile
        c_bile_new = c_bile + dc_bile_dt * dt_h
        c_bile_new = max(0.0, c_bile_new)

        c_sin = c_sin_new
        c_hep = c_hep_new
        c_bile = c_bile_new

        c_sin_arr[i] = c_sin
        c_hep_arr[i] = c_hep
        c_bile_arr[i] = c_bile

    # --- AUC calculations ---
    auc_sin = _trapezoidal_auc(times, c_sin_arr)
    auc_hep = _trapezoidal_auc(times, c_hep_arr)
    auc_bile = _trapezoidal_auc(times, c_bile_arr)

    # --- Hepatic extraction ratio ---
    hepatic_extraction_ratio = min(0.99, cl_sys_L_per_h / HEPATIC_BLOOD_FLOW)

    # --- Intracellular accumulation fold ---
    c_sin_max = max(c_sin_arr) if c_sin_arr else 1e-12
    c_hep_max = max(c_hep_arr) if c_hep_arr else 0.0
    intracellular_accumulation_fold = c_hep_max / max(c_sin_max, 1e-12)

    # --- Biliary excretion fraction ---
    total_exposure = auc_bile + auc_hep
    if total_exposure > 0:
        biliary_excretion_fraction = auc_bile / total_exposure
    else:
        biliary_excretion_fraction = 0.0

    notes = (
        f"OATP Vmax={oatp_vmax_uM_per_min} uM/min, Km={oatp_km_uM} uM; "
        f"MRP2 CL={mrp2_cl_mL_per_min} mL/min; "
        f"passive k={passive_uptake_per_h} /h"
    )

    return HepaticSinusoidalResult(
        drug_name=drug_name,
        times_h=times,
        c_sinusoidal_mg_L=c_sin_arr,
        c_hepatocyte_mg_L=c_hep_arr,
        c_bile_mg_L=c_bile_arr,
        hepatic_extraction_ratio=hepatic_extraction_ratio,
        biliary_excretion_fraction=biliary_excretion_fraction,
        intracellular_accumulation_fold=intracellular_accumulation_fold,
        oatp_uptake_rate=oatp_uptake_rate_mL_min,
        efflux_mrp2_rate=efflux_mrp2_rate,
        auc_hepatocyte_mg_h_per_L=auc_hep,
        auc_sinusoidal_mg_h_per_L=auc_sin,
        notes=notes,
    )


def compare_oatp_inhibition(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_L: float,
    inhibition_factors: list = None,
) -> list:
    """
    Compare hepatic sinusoidal transport under different levels of OATP inhibition.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    cl_sys_L_per_h : float
    vd_L : float
    inhibition_factors : list of float
        Scaling factors on OATP Vmax (1.0 = no inhibition, 0.0 = complete inhibition).
        Default: [1.0, 0.5, 0.1, 0.01].

    Returns
    -------
    list of HepaticSinusoidalResult sorted by auc_hepatocyte descending.
    """
    if inhibition_factors is None:
        inhibition_factors = [1.0, 0.5, 0.1, 0.01]

    default_vmax = 10.0  # uM/min baseline

    results = []
    for factor in inhibition_factors:
        scaled_vmax = default_vmax * factor
        result = simulate_hepatic_sinusoidal_transport(
            drug_name=f"{drug_name}_oatp_factor_{factor:.2f}",
            dose_mg=dose_mg,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_L=vd_L,
            oatp_vmax_uM_per_min=max(scaled_vmax, 1e-6),  # avoid zero
        )
        results.append(result)

    # Sort by auc_hepatocyte descending (less inhibition → more uptake → higher AUC)
    results.sort(key=lambda r: r.auc_hepatocyte_mg_h_per_L, reverse=True)
    return results
