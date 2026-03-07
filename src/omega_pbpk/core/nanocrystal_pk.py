"""Phase 769 — Nanocrystal Dissolution PK.

Models drug nanocrystal dissolution using Noyes-Whitney kinetics with
particle-size-dependent dissolution rate constants, GI absorption, and
systemic distribution.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NanocrystalPKResult:
    """Result of nanocrystal PK simulation."""

    drug_name: str
    dose_mg: float
    particle_radius_nm: float
    times_h: list
    c_plasma_mg_L: list
    c_solution_mg_L: list
    a_crystal_mg: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    k_diss_effective: float
    fraction_dissolved: float
    auc_enhancement_vs_coarse: float
    notes: str


def _compute_k_diss(particle_radius_nm: float) -> float:
    """Compute dissolution rate constant based on particle radius.

    Nano enhancement: k_diss_nano = k_diss_coarse * (r_coarse / r_nano)
    r_coarse = 50000 nm, k_diss_coarse = 0.1/h
    """
    r_coarse_nm = 50_000.0
    k_diss_coarse = 0.1  # /h
    k_diss = k_diss_coarse * (r_coarse_nm / particle_radius_nm)
    return k_diss


def _run_simulation(
    dose_mg: float,
    particle_radius_nm: float,
    c_sat_mg_L: float,
    k_abs_per_h: float,
    v_gi_L: float,
    s_area_m2_per_g: float,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[list, list, list, list]:
    """Run forward Euler simulation; returns (times, c_plasma, c_sol, a_cryst)."""
    k_diss = _compute_k_diss(particle_radius_nm)
    ke = cl_L_per_h / vd_L  # elimination rate constant

    # Initial conditions
    a_cryst = dose_mg  # mg in crystal form
    c_sol = 0.0  # mg/L in GI solution
    c_plasma = 0.0  # mg/L in plasma

    # Specific surface area: m2/g -> area scales with remaining crystal mass
    # s_area_initial in L (volume units for dissolution calculation)
    # We keep s_area_m2_per_g as a dimensionless scaling factor for k_diss
    # The effective surface area scales proportionally with remaining crystal mass.

    times = []
    c_plasma_list = []
    c_sol_list = []
    a_cryst_list = []

    n_steps = int(t_end_h / dt_h) + 1
    t = 0.0

    for _ in range(n_steps):
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_sol_list.append(c_sol)
        a_cryst_list.append(a_cryst)

        # Surface area scales with remaining crystal fraction
        if dose_mg > 0:
            area_fraction = a_cryst / dose_mg
        else:
            area_fraction = 0.0

        # Effective surface area factor (s_area_m2_per_g scales k_diss proportionally)
        # Normalize: at full dose, area_factor = s_area_m2_per_g / 10.0 (reference 10 m2/g)
        area_factor = area_fraction * (s_area_m2_per_g / 10.0)

        # Dissolution flux (mg/h)
        if a_cryst > 0 and c_sol < c_sat_mg_L:
            diss_flux = k_diss * area_factor * (c_sat_mg_L - c_sol) * v_gi_L
            # Prevent over-dissolution in a single step
            diss_flux = min(diss_flux, a_cryst / dt_h)
        else:
            diss_flux = 0.0

        # ODEs
        d_a_cryst = -diss_flux
        d_c_sol = (diss_flux / v_gi_L) - k_abs_per_h * c_sol
        d_c_plasma = (k_abs_per_h * c_sol * v_gi_L / vd_L) - ke * c_plasma

        # Forward Euler update
        a_cryst = max(0.0, a_cryst + d_a_cryst * dt_h)
        c_sol = max(0.0, c_sol + d_c_sol * dt_h)
        c_plasma = max(0.0, c_plasma + d_c_plasma * dt_h)

        t += dt_h

    return times, c_plasma_list, c_sol_list, a_cryst_list


def _trapezoidal_auc(times: list, values: list) -> float:
    """Compute AUC via trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (values[i - 1] + values[i]) * (times[i] - times[i - 1])
    return auc


def simulate_nanocrystal_pk(
    drug_name: str,
    dose_mg: float,
    particle_radius_nm: float = 500.0,
    c_sat_mg_L: float = 10.0,
    k_abs_per_h: float = 1.5,
    v_gi_L: float = 0.25,
    s_area_m2_per_g: float = 10.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.02,
) -> NanocrystalPKResult:
    """Simulate nanocrystal dissolution PK.

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    dose_mg : float
        Dose in mg.
    particle_radius_nm : float
        Nanocrystal particle radius in nm.
    c_sat_mg_L : float
        Saturation solubility in mg/L.
    k_abs_per_h : float
        GI absorption rate constant (1/h).
    v_gi_L : float
        GI volume in L.
    s_area_m2_per_g : float
        Specific surface area in m2/g.
    cl_L_per_h : float
        Systemic clearance in L/h.
    vd_L : float
        Volume of distribution in L.
    t_end_h : float
        Simulation end time in h.
    dt_h : float
        Forward Euler time step in h.

    Returns
    -------
    NanocrystalPKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if particle_radius_nm <= 0:
        raise ValueError("particle_radius_nm must be > 0")
    if c_sat_mg_L <= 0:
        raise ValueError("c_sat_mg_L must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")

    times, c_plasma, c_sol, a_cryst = _run_simulation(
        dose_mg=dose_mg,
        particle_radius_nm=particle_radius_nm,
        c_sat_mg_L=c_sat_mg_L,
        k_abs_per_h=k_abs_per_h,
        v_gi_L=v_gi_L,
        s_area_m2_per_g=s_area_m2_per_g,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    cmax = max(c_plasma)
    tmax = times[c_plasma.index(cmax)]
    auc = _trapezoidal_auc(times, c_plasma)
    k_diss_eff = _compute_k_diss(particle_radius_nm)
    fraction_dissolved = 1.0 - (a_cryst[-1] / dose_mg)

    # Compute coarse particle AUC for enhancement ratio
    _, c_plasma_coarse, _, _ = _run_simulation(
        dose_mg=dose_mg,
        particle_radius_nm=50_000.0,
        c_sat_mg_L=c_sat_mg_L,
        k_abs_per_h=k_abs_per_h,
        v_gi_L=v_gi_L,
        s_area_m2_per_g=s_area_m2_per_g,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )
    auc_coarse = _trapezoidal_auc(times, c_plasma_coarse)
    if auc_coarse > 0:
        auc_enhancement = auc / auc_coarse
    else:
        auc_enhancement = 1.0

    notes = (
        f"Particle radius {particle_radius_nm} nm; "
        f"k_diss={k_diss_eff:.4f}/h; "
        f"fraction dissolved={fraction_dissolved:.2%}; "
        f"AUC enhancement vs coarse (50 µm)={auc_enhancement:.2f}x"
    )

    return NanocrystalPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        particle_radius_nm=particle_radius_nm,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        c_solution_mg_L=c_sol,
        a_crystal_mg=a_cryst,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        k_diss_effective=k_diss_eff,
        fraction_dissolved=fraction_dissolved,
        auc_enhancement_vs_coarse=auc_enhancement,
        notes=notes,
    )


def compare_particle_sizes_nano(
    drug_name: str,
    dose_mg: float,
    radii_nm: list,
    c_sat_mg_L: float = 10.0,
    k_abs_per_h: float = 1.5,
    v_gi_L: float = 0.25,
    s_area_m2_per_g: float = 10.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.02,
) -> list:
    """Simulate nanocrystal PK for multiple particle sizes.

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    dose_mg : float
        Dose in mg.
    radii_nm : list
        List of particle radii in nm to compare.
    Other parameters follow simulate_nanocrystal_pk.

    Returns
    -------
    list[NanocrystalPKResult]
        Results sorted by AUC descending (smallest particles first).
    """
    results = []
    for r in radii_nm:
        res = simulate_nanocrystal_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            particle_radius_nm=r,
            c_sat_mg_L=c_sat_mg_L,
            k_abs_per_h=k_abs_per_h,
            v_gi_L=v_gi_L,
            s_area_m2_per_g=s_area_m2_per_g,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(res)

    results.sort(key=lambda r: r.auc_mg_h_per_L, reverse=True)
    return results
