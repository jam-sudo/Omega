"""Phase 1141 — Spleen PK model.

Drug distribution and sequestration in the spleen, important for
nanoparticles and immunosuppressants.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpleenPKResult:
    drug_name: str
    dose_mg: float
    particle_size_nm: float
    times_h: list
    c_plasma_mg_L: list
    c_spleen_mg_L: list
    c_immune_mg_L: list
    cmax_spleen_mg_L: float
    tmax_spleen_h: float
    auc_spleen_mg_h_per_L: float
    auc_plasma_mg_h_per_L: float
    spleen_to_plasma_auc_ratio: float
    immune_cell_accumulation_pct: float
    splenic_targeting_class: str
    notes: str


def _trapz(xs: list, ys: list) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(xs)):
        total += 0.5 * (ys[i - 1] + ys[i]) * (xs[i] - xs[i - 1])
    return total


def simulate_spleen_pk(
    drug_name: str,
    dose_mg: float,
    particle_size_nm: float = 0.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> SpleenPKResult:
    """Simulate drug distribution and sequestration in the spleen.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose in mg.
    particle_size_nm:
        Nanoparticle diameter in nm; 0 means small molecule.
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward-Euler step size (h).

    Returns
    -------
    SpleenPKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if particle_size_nm < 0:
        raise ValueError("particle_size_nm must be >= 0")

    # Compartment volumes
    v_spleen = 0.2  # L
    v_immune = 0.5  # L

    # Rate constants
    kps_base = 0.3  # plasma -> spleen (1/h)
    ksp = 0.05  # spleen -> plasma (1/h)
    ksi = 0.02  # spleen -> immune cells (1/h)
    kic = 0.01  # immune cell clearance (1/h)
    ke = cl_L_per_h / vd_L

    # Nanoparticle effect on splenic uptake
    if particle_size_nm > 0:
        kps = kps_base * max(1.0, min(5.0, particle_size_nm / 100.0))
    else:
        kps = kps_base

    # Solve using mass (mg) for numerical stability, convert to concentration for output.
    # A_plasma = C_plasma * vd_L
    # A_spleen = C_spleen * v_spleen
    # A_immune = C_immune * v_immune
    #
    # dA_plasma/dt = -(ke + kps) * A_plasma + ksp * A_spleen
    # dA_spleen/dt = kps * A_plasma - (ksp + ksi) * A_spleen
    # dA_immune/dt = ksi * A_spleen - kic * A_immune
    #
    # This is equivalent to the concentration ODEs in the spec (via chain rule)
    # but avoids the Vd/V_spleen amplification term that causes numerical blow-up.

    a_plasma = dose_mg  # initial mass in plasma (mg)
    a_spleen = 0.0
    a_immune = 0.0

    c_plasma0 = a_plasma / vd_L
    c_spleen0 = 0.0
    c_immune0 = 0.0

    n_steps = max(1, int(round(t_end_h / dt_h)))
    times = [0.0]
    cp_list = [c_plasma0]
    cs_list = [c_spleen0]
    ci_list = [c_immune0]

    for _ in range(n_steps):
        da_plasma = -(ke + kps) * a_plasma + ksp * a_spleen
        da_spleen = kps * a_plasma - (ksp + ksi) * a_spleen
        da_immune = ksi * a_spleen - kic * a_immune

        a_plasma = max(0.0, a_plasma + da_plasma * dt_h)
        a_spleen = max(0.0, a_spleen + da_spleen * dt_h)
        a_immune = max(0.0, a_immune + da_immune * dt_h)

        times.append(times[-1] + dt_h)
        cp_list.append(a_plasma / vd_L)
        cs_list.append(a_spleen / v_spleen)
        ci_list.append(a_immune / v_immune)

    # Metrics
    cmax_spleen = max(cs_list)
    tmax_spleen = times[cs_list.index(cmax_spleen)]
    auc_spleen = _trapz(times, cs_list)
    auc_plasma = _trapz(times, cp_list)
    auc_immune = _trapz(times, ci_list)

    ratio = auc_spleen / auc_plasma if auc_plasma > 0 else 0.0

    denom = auc_spleen + auc_plasma
    immune_pct = (auc_immune / denom * 100.0) if denom > 0 else 0.0

    if ratio >= 2.0:
        targeting_class = "good"
    elif ratio >= 0.5:
        targeting_class = "moderate"
    else:
        targeting_class = "poor"

    notes_parts = []
    if particle_size_nm > 0:
        notes_parts.append(f"Nanoparticle size={particle_size_nm:.0f} nm applied kps boost")
    if targeting_class == "good":
        notes_parts.append("High splenic targeting")
    notes = "; ".join(notes_parts) if notes_parts else "Standard small-molecule distribution"

    return SpleenPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        particle_size_nm=particle_size_nm,
        times_h=times,
        c_plasma_mg_L=cp_list,
        c_spleen_mg_L=cs_list,
        c_immune_mg_L=ci_list,
        cmax_spleen_mg_L=cmax_spleen,
        tmax_spleen_h=tmax_spleen,
        auc_spleen_mg_h_per_L=auc_spleen,
        auc_plasma_mg_h_per_L=auc_plasma,
        spleen_to_plasma_auc_ratio=ratio,
        immune_cell_accumulation_pct=immune_pct,
        splenic_targeting_class=targeting_class,
        notes=notes,
    )
