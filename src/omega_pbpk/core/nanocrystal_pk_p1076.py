"""Phase 1076: Drug Nanocrystal PK Model (diameter-based, simplified two-step)."""

from __future__ import annotations

from dataclasses import dataclass

# Bulk crystal reference dissolution rate constant (1/h) at d_bulk_nm = 10 µm
K_DISS_BULK = 0.1  # /h  for 10 µm bulk crystals
D_BULK_NM = 10_000.0  # 10 µm in nm

# Absorption rate constant for dissolved drug in GI lumen (1/h)
KA = 0.5
# Oral bioavailability for dissolved drug
F_DISSOLVED = 0.8


def _trapz(times: list[float], values: list[float]) -> float:
    """Trapezoidal integration."""
    area = 0.0
    for i in range(1, len(times)):
        area += (times[i] - times[i - 1]) * (values[i] + values[i - 1]) / 2.0
    return area


@dataclass
class NanocrystalPKResult1076:
    """Result of nanocrystal PK simulation (Phase 1076)."""

    drug_name: str
    dose_mg: float
    particle_size_nm: float
    times_h: list
    c_plasma_mg_L: list
    a_crystal_mg: list
    a_dissolved_mg: list
    cmax: float
    tmax_h: float
    auc: float
    f_dissolved_at_4h: float
    dissolution_t50_h: float
    enhancement_vs_bulk: float
    notes: str


def _simulate_core(
    dose_mg: float,
    particle_size_nm: float,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float,
    dt: float,
) -> tuple[list, list, list, list]:
    """Run forward Euler ODE for nanocrystal PK; returns (times, c_plasma, a_crystal, a_dissolved)."""
    k_diss = K_DISS_BULK * (D_BULK_NM / particle_size_nm)
    ke = cl_L_per_h / vd_L

    a_crystal = dose_mg
    a_dissolved = 0.0
    c_plasma = 0.0

    times: list[float] = []
    c_plasma_list: list[float] = []
    a_crystal_list: list[float] = []
    a_dissolved_list: list[float] = []

    t = 0.0
    n_steps = max(1, round(t_end_h / dt))
    actual_dt = t_end_h / n_steps

    for step in range(n_steps + 1):
        times.append(round(t, 8))
        c_plasma_list.append(c_plasma)
        a_crystal_list.append(a_crystal)
        a_dissolved_list.append(a_dissolved)

        if step == n_steps:
            break

        # Forward Euler
        dA_crystal = -k_diss * a_crystal
        dA_dissolved = k_diss * a_crystal - KA * a_dissolved
        dC_plasma = KA * a_dissolved * F_DISSOLVED / vd_L - ke * c_plasma

        a_crystal = max(0.0, a_crystal + dA_crystal * actual_dt)
        a_dissolved = max(0.0, a_dissolved + dA_dissolved * actual_dt)
        c_plasma = max(0.0, c_plasma + dC_plasma * actual_dt)
        t += actual_dt

    return times, c_plasma_list, a_crystal_list, a_dissolved_list


def simulate_nanocrystal_pk(
    drug_name: str,
    dose_mg: float,
    particle_size_nm: float = 200.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt: float = 0.05,
    _compute_enhancement: bool = True,
) -> NanocrystalPKResult1076:
    """Simulate PK of a nanocrystal formulation.

    Two-step model:
      1. Dissolution: A_crystal → A_dissolved  (Noyes-Whitney, scaled by particle diameter)
         k_diss = K_DISS_BULK * (D_BULK_NM / particle_size_nm)
      2. Absorption + elimination:
         dA_crystal/dt  = -k_diss * A_crystal
         dA_dissolved/dt = k_diss * A_crystal - ka * A_dissolved
         dC_plasma/dt    = ka * A_dissolved * F / Vd  -  ke * C_plasma

    Args:
        drug_name: Drug name.
        dose_mg: Dose in mg. Must be > 0.
        particle_size_nm: Nanocrystal diameter in nm. Must be > 0 and <= 100000.
        cl_L_per_h: Plasma clearance (L/h). Must be > 0.
        vd_L: Volume of distribution (L). Must be > 0.
        t_end_h: Simulation duration (h).
        dt: ODE time step (h).
        _compute_enhancement: Internal flag; set False to skip bulk comparison recursion.

    Returns:
        NanocrystalPKResult1076 with full PK profile.

    Raises:
        ValueError: If inputs are invalid.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if particle_size_nm <= 0 or particle_size_nm > 100_000:
        raise ValueError(f"particle_size_nm must be in (0, 100000], got {particle_size_nm}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")

    times, c_plasma_list, a_crystal_list, a_dissolved_list = _simulate_core(
        dose_mg=dose_mg,
        particle_size_nm=particle_size_nm,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        t_end_h=t_end_h,
        dt=dt,
    )

    # PK metrics
    cmax = max(c_plasma_list)
    tmax_h = times[c_plasma_list.index(cmax)]
    auc = _trapz(times, c_plasma_list)

    # f_dissolved_at_4h: fraction of dose that has left crystal phase by t=4h
    f_dissolved_at_4h = 0.0
    for i, ti in enumerate(times):
        if ti >= 4.0:
            f_dissolved_at_4h = 1.0 - a_crystal_list[i] / dose_mg
            break
    else:
        f_dissolved_at_4h = 1.0 - a_crystal_list[-1] / dose_mg
    f_dissolved_at_4h = max(0.0, min(1.0, f_dissolved_at_4h))

    # dissolution_t50_h: time when 50% of crystals have dissolved
    half_crystal = 0.5 * dose_mg
    dissolution_t50_h = float("inf")
    for i, ac in enumerate(a_crystal_list):
        if ac <= half_crystal:
            dissolution_t50_h = times[i]
            break

    # Enhancement vs bulk reference (10 µm diameter = 10000 nm)
    if _compute_enhancement:
        _, c_bulk, _, _ = _simulate_core(
            dose_mg=dose_mg,
            particle_size_nm=D_BULK_NM,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            t_end_h=t_end_h,
            dt=dt,
        )
        bulk_auc = _trapz(times, c_bulk)
        enhancement_vs_bulk = auc / bulk_auc if bulk_auc > 0 else float("inf")
    else:
        enhancement_vs_bulk = 1.0

    k_diss = K_DISS_BULK * (D_BULK_NM / particle_size_nm)
    notes = (
        f"Nanocrystal PK (Phase 1076): d={particle_size_nm} nm, k_diss={k_diss:.4f}/h; "
        f"Cmax={cmax:.4f} mg/L, AUC={auc:.4f} mg·h/L, "
        f"enhancement={enhancement_vs_bulk:.2f}x vs 10 µm bulk."
    )

    return NanocrystalPKResult1076(
        drug_name=drug_name,
        dose_mg=dose_mg,
        particle_size_nm=particle_size_nm,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        a_crystal_mg=a_crystal_list,
        a_dissolved_mg=a_dissolved_list,
        cmax=cmax,
        tmax_h=tmax_h,
        auc=auc,
        f_dissolved_at_4h=f_dissolved_at_4h,
        dissolution_t50_h=dissolution_t50_h,
        enhancement_vs_bulk=enhancement_vs_bulk,
        notes=notes,
    )


def compare_particle_sizes(
    drug_name: str,
    dose_mg: float,
    sizes_nm: list[float],
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt: float = 0.05,
) -> list[NanocrystalPKResult1076]:
    """Compare nanocrystal PK across multiple particle sizes.

    Args:
        drug_name: Drug name.
        dose_mg: Dose in mg.
        sizes_nm: List of particle diameters (nm).
        cl_L_per_h: Clearance (L/h).
        vd_L: Volume of distribution (L).
        t_end_h: Simulation duration (h).
        dt: ODE time step (h).

    Returns:
        List of NanocrystalPKResult1076 sorted by auc descending.
    """
    results = [
        simulate_nanocrystal_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            particle_size_nm=s,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            t_end_h=t_end_h,
            dt=dt,
        )
        for s in sizes_nm
    ]
    results.sort(key=lambda r: r.auc, reverse=True)
    return results
