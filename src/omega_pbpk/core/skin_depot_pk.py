"""
Phase 447 — Transdermal drug delivery: depot formation in stratum corneum
and drug release kinetics through 3-layer skin model + systemic compartment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SkinDepotPKResult:
    """Result of transdermal skin depot PK simulation."""

    drug_name: str
    patch_dose_mg: float
    patch_area_cm2: float
    times_h: list
    c_stratum_corneum_mg_g: list  # SC depot concentration
    c_viable_epidermis_mg_mL: list  # VE concentration
    c_systemic_mg_L: list  # plasma concentration
    cmax_systemic_mg_L: float
    tmax_systemic_h: float
    auc_systemic_mg_h_per_L: float
    steady_state_conc_mg_L: float  # estimated plateau
    lag_time_h: float  # time to first detectable plasma conc
    flux_rate_mg_per_h: float  # drug flux through skin at peak
    patch_efficiency_pct: float  # % of patch dose absorbed
    t_half_systemic_h: float
    notes: str


def simulate_skin_depot_pk(
    drug_name: str,
    patch_dose_mg: float,
    patch_area_cm2: float,
    logp: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float = 50.0,
    patch_duration_h: float = 24.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> SkinDepotPKResult:
    """
    Simulate transdermal drug delivery using 3-layer skin + systemic model.

    Layers:
    - Stratum corneum (SC): patch depot
    - Viable epidermis (VE): permeation layer
    - Systemic: plasma compartment

    Parameters
    ----------
    drug_name : str
    patch_dose_mg : float
        Total dose in the patch (mg)
    patch_area_cm2 : float
        Active patch area (cm2)
    logp : float
        Octanol-water partition coefficient (log scale)
    cl_sys_L_per_h : float
        Systemic clearance (L/h)
    vd_sys_L : float
        Volume of distribution (L)
    patch_duration_h : float
        Duration over which patch releases drug (h)
    t_end_h : float
        Simulation end time (h)
    dt_h : float
        Integration time step (h)
    """
    # --- Validation ---
    if patch_dose_mg <= 0:
        raise ValueError("patch_dose_mg must be > 0")
    if patch_area_cm2 <= 0:
        raise ValueError("patch_area_cm2 must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if patch_duration_h <= 0:
        raise ValueError("patch_duration_h must be > 0")

    # --- Model parameters ---
    # SC → VE permeability coefficient (logP-dependent)
    # k_sc_to_ve = 0.05 * (1 + logp * 0.2) per h, clamped [0.01, 2.0]
    k_sc_to_ve = 0.05 * (1.0 + logp * 0.2)
    k_sc_to_ve = max(0.01, min(2.0, k_sc_to_ve))

    # VE → systemic transfer rate constant (1/h)
    k_ve_to_sys = 0.3  # /h

    # VE volume (mL): area * 0.02 mL/cm2
    V_ve_mL = patch_area_cm2 * 0.02

    # SC volume (g): approximated as area * 0.01 g/cm2 (thin SC layer ~0.1mm)
    V_sc_g = patch_area_cm2 * 0.01

    # Systemic elimination rate constant
    ke = cl_sys_L_per_h / vd_sys_L  # /h

    # Zero-order patch release rate
    k_release_mg_per_h = patch_dose_mg / patch_duration_h

    # --- Forward Euler integration ---
    n_steps = int(t_end_h / dt_h) + 1
    times = [0.0] * n_steps
    M_sc = [0.0] * n_steps  # mass in SC (mg)
    C_ve = [0.0] * n_steps  # concentration in VE (mg/mL)
    C_sys = [0.0] * n_steps  # plasma concentration (mg/L)

    # Track SC mass for concentration calculation
    m_sc_at_tmax = 0.0

    for i in range(1, n_steps):
        t = (i - 1) * dt_h
        times[i] = t + dt_h

        # Patch still releasing?
        k_rel = k_release_mg_per_h if t < patch_duration_h else 0.0

        # SC ODEs: dM_sc/dt = k_release - k_sc_to_ve * M_sc
        dM_sc = k_rel - k_sc_to_ve * M_sc[i - 1]

        # VE ODEs: dC_ve/dt = k_sc_to_ve * M_sc / V_ve - k_ve_to_sys * C_ve
        # (units: mg/h / mL = mg/mL/h)
        if V_ve_mL > 0:
            dC_ve = (k_sc_to_ve * M_sc[i - 1] / V_ve_mL) - k_ve_to_sys * C_ve[i - 1]
        else:
            dC_ve = 0.0

        # Systemic: dC_sys/dt = k_ve_to_sys * C_ve * V_ve / Vd_sys - ke * C_sys
        # (C_ve in mg/mL, V_ve in mL → mg/h; Vd_sys in L → mg/L/h)
        dC_sys = (k_ve_to_sys * C_ve[i - 1] * V_ve_mL / (vd_sys_L * 1000.0)) - ke * C_sys[i - 1]

        # Update states (clamp to >= 0)
        M_sc[i] = max(0.0, M_sc[i - 1] + dM_sc * dt_h)
        C_ve[i] = max(0.0, C_ve[i - 1] + dC_ve * dt_h)
        C_sys[i] = max(0.0, C_sys[i - 1] + dC_sys * dt_h)

    times[0] = 0.0  # ensure first time is 0

    # --- Derived PK metrics ---
    cmax = max(C_sys)
    tmax = times[C_sys.index(cmax)]

    # AUC via trapezoidal rule
    auc = sum(
        0.5 * (C_sys[i] + C_sys[i - 1]) * (times[i] - times[i - 1]) for i in range(1, n_steps)
    )

    # SC concentrations (mg/g): M_sc / V_sc_g
    C_sc = [m / V_sc_g if V_sc_g > 0 else 0.0 for m in M_sc]

    # Lag time: time until C_sys > 1% of cmax
    lag_time = 0.0
    threshold = 0.01 * cmax
    if cmax > 0:
        for i, c in enumerate(C_sys):
            if c > threshold:
                lag_time = times[i]
                break

    # Flux at tmax: k_sc_to_ve * M_sc_at_tmax
    tmax_idx = C_sys.index(cmax)
    m_sc_at_tmax = M_sc[tmax_idx]
    flux_rate = k_sc_to_ve * m_sc_at_tmax

    # Steady-state systemic concentration estimate
    # C_ss = (k_release / cl_sys) when patch is releasing
    steady_state_conc = k_release_mg_per_h / cl_sys_L_per_h  # mg/L

    # Half-life
    t_half = 0.693 / ke

    # Patch efficiency: % of patch dose absorbed systemically
    # Total absorbed = AUC * CL (in mg)
    total_absorbed_mg = auc * cl_sys_L_per_h
    patch_efficiency = min(100.0, (total_absorbed_mg / patch_dose_mg) * 100.0)

    notes = (
        f"Transdermal simulation: logP={logp:.2f}, k_sc_to_ve={k_sc_to_ve:.3f}/h, "
        f"patch release rate={k_release_mg_per_h:.3f} mg/h over {patch_duration_h:.1f}h. "
        f"Tmax={tmax:.1f}h, Cmax={cmax:.4f} mg/L."
    )

    return SkinDepotPKResult(
        drug_name=drug_name,
        patch_dose_mg=patch_dose_mg,
        patch_area_cm2=patch_area_cm2,
        times_h=times,
        c_stratum_corneum_mg_g=C_sc,
        c_viable_epidermis_mg_mL=list(C_ve),
        c_systemic_mg_L=list(C_sys),
        cmax_systemic_mg_L=cmax,
        tmax_systemic_h=tmax,
        auc_systemic_mg_h_per_L=auc,
        steady_state_conc_mg_L=steady_state_conc,
        lag_time_h=lag_time,
        flux_rate_mg_per_h=flux_rate,
        patch_efficiency_pct=patch_efficiency,
        t_half_systemic_h=t_half,
        notes=notes,
    )


def compare_patch_sizes(
    drug_name: str,
    logp: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    areas_cm2: list[float],
    dose_per_area_mg_per_cm2: float = 0.5,
) -> list[SkinDepotPKResult]:
    """
    Compare multiple patch sizes for the same drug.

    Parameters
    ----------
    drug_name : str
    logp : float
    cl_sys_L_per_h : float
    vd_sys_L : float
    areas_cm2 : list of float
        Patch areas to compare
    dose_per_area_mg_per_cm2 : float
        Dose density (mg per cm2); total dose = area * dose_per_area

    Returns
    -------
    List of SkinDepotPKResult sorted by steady_state_conc_mg_L descending.
    """
    results = []
    for area in areas_cm2:
        dose = area * dose_per_area_mg_per_cm2
        res = simulate_skin_depot_pk(
            drug_name=drug_name,
            patch_dose_mg=dose,
            patch_area_cm2=area,
            logp=logp,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_sys_L=vd_sys_L,
        )
        results.append(res)

    results.sort(key=lambda r: r.steady_state_conc_mg_L, reverse=True)
    return results
