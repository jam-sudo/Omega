"""Spleen PK model for nanoparticles and macrophage-targeted drugs (Phase 805).

Models drug distribution in the spleen as a 2-compartment model (plasma + spleen)
with phagocytosis as irreversible sink. Particle size modulates macrophage uptake.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SpleenPKResult:
    """Result of spleen PK simulation (Phase 805)."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_spleen_mg_L: list
    cmax_plasma: float
    cmax_spleen: float
    auc_plasma: float
    auc_spleen: float
    spleen_plasma_kp: float  # AUC spleen / AUC plasma
    red_pulp_uptake_pct: float  # fraction uptake in red pulp (reticuloendothelial)
    marginal_zone_pct: float  # fraction in marginal zone
    macrophage_phagocytosis_pct: float  # % phagocytosed
    t_half_spleen_h: float
    notes: str


def _particle_size_factor(particle_size_nm: float | None) -> float:
    """Return a multiplier for macrophage uptake based on particle size.

    100-500 nm: highest uptake (factor 2x).
    <20 nm or >1000 nm: lower uptake (factor 0.5x).
    Otherwise (20-100 or 500-1000 nm): factor 1.0x.
    """
    if particle_size_nm is None:
        return 1.0
    if 100.0 <= particle_size_nm <= 500.0:
        return 2.0
    if particle_size_nm < 20.0 or particle_size_nm > 1000.0:
        return 0.5
    return 1.0


def _trapz(y: list, x: list) -> float:
    """Manual trapezoidal integration."""
    result = 0.0
    for i in range(1, len(x)):
        result += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return result


def simulate_spleen_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_plasma_L: float = 5.0,
    q_spleen_L_per_h: float = 0.2,
    v_spleen_L: float = 0.15,
    kp_spleen: float = 3.0,
    macrophage_uptake_rate: float = 0.1,
    particle_size_nm: float | None = None,
    route: str = "iv",
    f_oral: float = 0.8,
    ka_per_h: float = 1.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> SpleenPKResult:
    """Simulate spleen PK with phagocytosis sink.

    2-compartment model: plasma + spleen with macrophage phagocytosis.
    - Plasma <-> spleen via Q_spleen * (C_plasma - C_spleen/Kp)
    - Phagocytosis: macrophage_uptake_rate * C_spleen (irreversible sink)
    - Particle size modulates macrophage_uptake_rate.

    Parameters
    ----------
    drug_name : str
        Name of the drug or compound.
    dose_mg : float
        Dose in mg.
    cl_L_per_h : float
        Systemic clearance (L/h).
    vd_plasma_L : float
        Plasma volume of distribution (L).
    q_spleen_L_per_h : float
        Splenic blood flow (L/h).
    v_spleen_L : float
        Spleen volume (L).
    kp_spleen : float
        Tissue:plasma partition coefficient for spleen.
    macrophage_uptake_rate : float
        First-order rate for phagocytosis (1/h).
    particle_size_nm : float or None
        If provided, modulates macrophage uptake rate by particle size.
    route : str
        Dosing route: "iv" or "oral".
    f_oral : float
        Oral bioavailability fraction (used only if route=="oral").
    ka_per_h : float
        Absorption rate constant (1/h) for oral route.
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Forward Euler step size (h).

    Returns
    -------
    SpleenPKResult
        Full PK time-course and summary metrics.
    """
    if not drug_name or not isinstance(drug_name, str):
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")
    if q_spleen_L_per_h <= 0:
        raise ValueError("q_spleen_L_per_h must be > 0")
    if v_spleen_L <= 0:
        raise ValueError("v_spleen_L must be > 0")
    if kp_spleen <= 0:
        raise ValueError("kp_spleen must be > 0")
    if macrophage_uptake_rate < 0:
        raise ValueError("macrophage_uptake_rate must be >= 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    # Apply particle size factor to macrophage uptake
    ps_factor = _particle_size_factor(particle_size_nm)
    effective_uptake_rate = macrophage_uptake_rate * ps_factor

    # Determine effective dose reaching systemic circulation
    if route == "oral":
        # Depot absorption
        a_gut_0 = dose_mg * f_oral
        a_plasma_0 = 0.0
    else:
        a_gut_0 = 0.0
        a_plasma_0 = dose_mg

    # Initial states: amount in plasma (mg), amount in spleen (mg), phagocytosed (mg)
    n_steps = max(int(math.ceil(t_end_h / dt_h)), 1)

    times_h = []
    c_plasma_list = []
    c_spleen_list = []

    # State variables
    a_gut = a_gut_0
    a_plasma = a_plasma_0
    a_spleen = 0.0
    a_phago = 0.0

    # Elimination rate constant from plasma
    ke_plasma = cl_L_per_h / vd_plasma_L

    for step in range(n_steps + 1):
        t = step * dt_h
        c_plasma = a_plasma / vd_plasma_L
        c_spleen = a_spleen / v_spleen_L

        times_h.append(t)
        c_plasma_list.append(c_plasma)
        c_spleen_list.append(c_spleen)

        if step == n_steps:
            break

        # Transfer between plasma and spleen (perfusion-limited)
        transfer = q_spleen_L_per_h * (c_plasma - c_spleen / kp_spleen)

        # Phagocytosis sink (irreversible)
        phago_rate = effective_uptake_rate * c_spleen * v_spleen_L

        # Oral absorption from gut
        if route == "oral":
            absorption = ka_per_h * a_gut
            da_gut = -absorption
        else:
            absorption = 0.0
            da_gut = 0.0

        # ODEs
        da_plasma = absorption - ke_plasma * a_plasma - transfer
        da_spleen = transfer - phago_rate
        da_phago = phago_rate

        # Forward Euler updates
        a_gut = max(a_gut + dt_h * da_gut, 0.0)
        a_plasma = max(a_plasma + dt_h * da_plasma, 0.0)
        a_spleen = max(a_spleen + dt_h * da_spleen, 0.0)
        a_phago = a_phago + dt_h * da_phago

    # PK metrics
    cmax_plasma = max(c_plasma_list) if c_plasma_list else 0.0
    cmax_spleen = max(c_spleen_list) if c_spleen_list else 0.0
    auc_plasma = _trapz(c_plasma_list, times_h)
    auc_spleen = _trapz(c_spleen_list, times_h)

    # Spleen:plasma Kp from AUC ratio
    if auc_plasma > 0:
        spleen_plasma_kp = auc_spleen / auc_plasma
    else:
        spleen_plasma_kp = 0.0

    # Macrophage phagocytosis percentage: a_phago / dose_mg * 100
    macro_phago_pct = (a_phago / dose_mg) * 100.0 if dose_mg > 0 else 0.0
    macro_phago_pct = min(macro_phago_pct, 100.0)

    # Red pulp uptake: larger particles (100-500 nm) go primarily to red pulp (~70%)
    # Marginal zone uptake: smaller particles more likely in marginal zone (~30%)
    if particle_size_nm is not None and 100.0 <= particle_size_nm <= 500.0:
        red_pulp_pct = 70.0
        marginal_zone_pct = 30.0
    elif particle_size_nm is not None and particle_size_nm < 100.0:
        red_pulp_pct = 40.0
        marginal_zone_pct = 60.0
    else:
        red_pulp_pct = 60.0
        marginal_zone_pct = 40.0

    # Estimate spleen t_half: from terminal phase of c_spleen
    # Find the last nonzero value and use ln(2)/effective_elimination
    # effective spleen elimination = (q/v_spleen + uptake_rate + q/(v_spleen*kp))
    effective_spleen_elim = (q_spleen_L_per_h / v_spleen_L) + effective_uptake_rate
    if effective_spleen_elim > 0:
        t_half_spleen = math.log(2.0) / effective_spleen_elim
    else:
        t_half_spleen = float("inf")

    # Notes
    notes_parts = []
    if particle_size_nm is not None:
        notes_parts.append(
            f"Particle size: {particle_size_nm} nm (uptake factor: {ps_factor:.1f}x)"
        )
    if spleen_plasma_kp > 5.0:
        notes_parts.append("High spleen accumulation")
    if macro_phago_pct > 50.0:
        notes_parts.append("Significant macrophage phagocytosis (>50%)")
    if route == "oral":
        notes_parts.append(f"Oral route (F={f_oral:.1%})")
    notes = "; ".join(notes_parts) if notes_parts else "Normal spleen PK"

    return SpleenPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_spleen_mg_L=c_spleen_list,
        cmax_plasma=cmax_plasma,
        cmax_spleen=cmax_spleen,
        auc_plasma=auc_plasma,
        auc_spleen=auc_spleen,
        spleen_plasma_kp=spleen_plasma_kp,
        red_pulp_uptake_pct=red_pulp_pct,
        marginal_zone_pct=marginal_zone_pct,
        macrophage_phagocytosis_pct=macro_phago_pct,
        t_half_spleen_h=t_half_spleen,
        notes=notes,
    )


__all__ = ["SpleenPKResult", "simulate_spleen_pk"]
