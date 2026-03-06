"""Biofilm pharmacokinetics — antibiotic penetration into bacterial biofilms.

Models the 2-region dynamics of drug distribution from systemic plasma into
biofilm surface and biofilm core, with bacterial kill kinetics. Useful for
simulating treatment of persistent infections caused by biofilm-forming
organisms (e.g., Pseudomonas aeruginosa, Staphylococcus aureus).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BiofilmPKResult:
    """Result of a biofilm PK simulation.

    Attributes
    ----------
    drug_name : str
        Name of the antibiotic.
    dose_mg : float
        Administered dose (mg).
    mic_mg_L : float
        Minimum inhibitory concentration (mg/L).
    route : str
        Administration route ("iv" or "oral").
    times_h : list[float]
        Simulation time points (h).
    c_plasma_mg_L : list[float]
        Plasma drug concentration over time (mg/L).
    c_biofilm_surface_mg_L : list[float]
        Biofilm surface layer drug concentration over time (mg/L).
    c_biofilm_core_mg_L : list[float]
        Biofilm core drug concentration over time (mg/L).
    bacteria_norm : list[float]
        Normalized bacterial count (N/N0, starts at 1.0).
    cmax_plasma : float
        Peak plasma concentration (mg/L).
    cmax_biofilm_core : float
        Peak biofilm core concentration (mg/L).
    t_above_mic_plasma_h : float
        Time plasma concentration exceeds MIC (h).
    t_above_mic_biofilm_h : float
        Time biofilm core concentration exceeds MIC (h).
    penetration_ratio : float
        Ratio of cmax_biofilm_core to cmax_plasma.
    log_kill_bacteria : float
        Maximum log10 reduction in bacterial count (log10(1/N_min)).
    eradication : bool
        True if bacteria_norm falls below 0.001 at any time point.
    notes : str
        Clinical interpretation notes.
    """

    drug_name: str
    dose_mg: float
    mic_mg_L: float
    route: str
    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_biofilm_surface_mg_L: list[float]
    c_biofilm_core_mg_L: list[float]
    bacteria_norm: list[float]
    cmax_plasma: float
    cmax_biofilm_core: float
    t_above_mic_plasma_h: float
    t_above_mic_biofilm_h: float
    penetration_ratio: float
    log_kill_bacteria: float
    eradication: bool
    notes: str


def simulate_biofilm_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    k_pen_per_h: float,
    mic_mg_L: float,
    route: str = "iv",
    ka_per_h: float = 1.0,
    k_bind_per_h: float = 0.5,
    k_unbind_per_h: float = 0.1,
    kill_rate_per_h: float = 0.3,
    kg_bacteria_per_h: float = 0.1,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> BiofilmPKResult:
    """Simulate antibiotic pharmacokinetics in a 2-region biofilm model.

    The model consists of:
    - Plasma: 1-compartment systemic PK
    - Biofilm surface: exchanges with plasma, drug binds to matrix
    - Biofilm core: receives drug by diffusion from surface; bacteria killed here

    Bacterial kill ODE:
        dN/dt = (kg - kill_rate * max(0, c_core - MIC) / (c_core + MIC + 0.001)) * N

    Parameters
    ----------
    drug_name : str
        Name of the antibiotic.
    dose_mg : float
        Administered dose (mg). Must be > 0.
    cl_L_per_h : float
        Systemic clearance (L/h). Must be > 0.
    vd_L : float
        Volume of distribution (L). Must be > 0.
    k_pen_per_h : float
        Biofilm penetration rate constant (surface→core, h⁻¹). Must be > 0.
    mic_mg_L : float
        Minimum inhibitory concentration (mg/L). Must be > 0.
    route : str
        Administration route: "iv" or "oral".
    ka_per_h : float
        Absorption rate constant for oral route (h⁻¹).
    k_bind_per_h : float
        Biofilm matrix binding rate (h⁻¹).
    k_unbind_per_h : float
        Biofilm matrix unbinding rate (h⁻¹).
    kill_rate_per_h : float
        Maximum bacterial kill rate at saturating drug concentrations (h⁻¹).
    kg_bacteria_per_h : float
        Bacterial growth rate constant (h⁻¹).
    t_end_h : float
        Simulation duration (h).
    dt_h : float
        Forward Euler integration time step (h).

    Returns
    -------
    BiofilmPKResult
        Simulation results including concentration-time profiles and PK/PD metrics.

    Raises
    ------
    ValueError
        If any critical parameter is out of valid range.
    """
    # Input validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if k_pen_per_h <= 0:
        raise ValueError(f"k_pen_per_h must be > 0, got {k_pen_per_h}")
    if mic_mg_L <= 0:
        raise ValueError(f"mic_mg_L must be > 0, got {mic_mg_L}")

    kel = cl_L_per_h / vd_L  # elimination rate constant (h⁻¹)

    n_steps = int(math.ceil(t_end_h / dt_h)) + 1
    times = [i * dt_h for i in range(n_steps)]

    # State variables
    # Plasma concentration (mg/L)
    c_plasma = 0.0
    # Biofilm surface free drug (mg/L)
    c_surf = 0.0
    # Biofilm surface bound drug (mg/L)
    c_bound = 0.0
    # Biofilm core drug (mg/L)
    c_core = 0.0
    # Bacteria normalized count
    bact = 1.0
    # Oral depot (mg/L equivalent via vd_L scaling)
    depot = 0.0

    # Initial conditions
    if route.lower() == "iv":
        c_plasma = dose_mg / vd_L
    elif route.lower() == "oral":
        depot = dose_mg / vd_L
    else:
        raise ValueError(f"route must be 'iv' or 'oral', got '{route}'")

    # Output arrays
    out_plasma = [c_plasma]
    out_surf = [c_surf]
    out_core = [c_core]
    out_bact = [bact]

    for _ in range(1, n_steps):
        # ODE right-hand sides (Forward Euler)

        # Oral absorption from depot
        if route.lower() == "oral":
            dDepot = -ka_per_h * depot
            dC_plasma_abs = ka_per_h * depot
        else:
            dDepot = 0.0
            dC_plasma_abs = 0.0

        # Plasma: elimination + exchange with biofilm surface
        # Transfer to surface modeled as first-order from plasma
        dC_plasma = dC_plasma_abs - kel * c_plasma - k_pen_per_h * c_plasma + k_pen_per_h * c_surf

        # Biofilm surface: exchange with plasma + binding/unbinding
        dC_surf = (
            k_pen_per_h * c_plasma
            - k_pen_per_h * c_surf
            - k_bind_per_h * c_surf
            + k_unbind_per_h * c_bound
            - k_pen_per_h * c_surf  # diffusion into core
            + k_pen_per_h * c_core  # back-diffusion from core
        )

        # Biofilm surface bound drug
        dC_bound = k_bind_per_h * c_surf - k_unbind_per_h * c_bound

        # Biofilm core: receives drug from surface layer
        dC_core = k_pen_per_h * c_surf - k_pen_per_h * c_core

        # Bacterial dynamics (Emax-type kill at concentrations above MIC)
        kill_eff = kill_rate_per_h * max(0.0, c_core - mic_mg_L) / (c_core + mic_mg_L + 0.001)
        dBact = (kg_bacteria_per_h - kill_eff) * bact

        # Forward Euler updates
        depot += dDepot * dt_h
        depot = max(depot, 0.0)
        c_plasma = max(c_plasma + dC_plasma * dt_h, 0.0)
        c_surf = max(c_surf + dC_surf * dt_h, 0.0)
        c_bound = max(c_bound + dC_bound * dt_h, 0.0)
        c_core = max(c_core + dC_core * dt_h, 0.0)
        bact = max(bact + dBact * dt_h, 1e-15)

        out_plasma.append(c_plasma)
        out_surf.append(c_surf)
        out_core.append(c_core)
        out_bact.append(bact)

    # PK/PD metrics
    cmax_plasma = float(max(out_plasma))
    cmax_core = float(max(out_core))

    # Time above MIC
    t_above_plasma = sum(1 for c in out_plasma if c > mic_mg_L) * dt_h
    t_above_core = sum(1 for c in out_core if c > mic_mg_L) * dt_h

    # Penetration ratio
    pen_ratio = cmax_core / cmax_plasma if cmax_plasma > 0 else 0.0

    # Log kill: log10(1 / min bacteria) = -log10(min bacteria)
    min_bact = min(out_bact)
    log_kill = -math.log10(max(min_bact, 1e-15))

    # Eradication check
    eradication = any(b < 0.001 for b in out_bact)

    # Clinical notes
    if eradication:
        notes = "Biofilm eradication achieved (bacteria < 0.1% of initial)."
    elif log_kill >= 3.0:
        notes = "Significant bactericidal activity (>3 log kill). Eradication not confirmed."
    elif log_kill >= 1.0:
        notes = "Moderate bactericidal activity (1-3 log kill). Consider dose escalation."
    elif pen_ratio < 0.1:
        notes = "Poor biofilm penetration (ratio < 0.1). Drug unlikely to be effective."
    else:
        notes = "Limited antimicrobial activity. Consider alternative therapy or combination."

    return BiofilmPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mic_mg_L=mic_mg_L,
        route=route,
        times_h=times,
        c_plasma_mg_L=out_plasma,
        c_biofilm_surface_mg_L=out_surf,
        c_biofilm_core_mg_L=out_core,
        bacteria_norm=out_bact,
        cmax_plasma=cmax_plasma,
        cmax_biofilm_core=cmax_core,
        t_above_mic_plasma_h=t_above_plasma,
        t_above_mic_biofilm_h=t_above_core,
        penetration_ratio=pen_ratio,
        log_kill_bacteria=log_kill,
        eradication=eradication,
        notes=notes,
    )


def assess_biofilm_eradication(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    mic_mg_L: float,
    doses_mg: list[float],
    **kwargs,
) -> list[BiofilmPKResult]:
    """Evaluate biofilm eradication potential across a range of doses.

    Simulates the drug at each dose level and ranks results by log kill
    in descending order (highest kill first).

    Parameters
    ----------
    drug_name : str
        Name of the antibiotic.
    dose_mg : float
        Reference dose (not used; individual doses come from doses_mg).
    cl_L_per_h : float
        Systemic clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    mic_mg_L : float
        Minimum inhibitory concentration (mg/L).
    doses_mg : list[float]
        List of doses to evaluate (mg).
    **kwargs
        Additional keyword arguments passed to simulate_biofilm_pk.

    Returns
    -------
    list[BiofilmPKResult]
        Results for each dose, sorted by log_kill_bacteria descending.
    """
    results = []
    for d in doses_mg:
        result = simulate_biofilm_pk(
            drug_name=drug_name,
            dose_mg=d,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            k_pen_per_h=kwargs.get("k_pen_per_h", 0.05),
            mic_mg_L=mic_mg_L,
            route=kwargs.get("route", "iv"),
            ka_per_h=kwargs.get("ka_per_h", 1.0),
            k_bind_per_h=kwargs.get("k_bind_per_h", 0.5),
            k_unbind_per_h=kwargs.get("k_unbind_per_h", 0.1),
            kill_rate_per_h=kwargs.get("kill_rate_per_h", 0.3),
            kg_bacteria_per_h=kwargs.get("kg_bacteria_per_h", 0.1),
            t_end_h=kwargs.get("t_end_h", 48.0),
            dt_h=kwargs.get("dt_h", 0.1),
        )
        results.append(result)

    results.sort(key=lambda r: r.log_kill_bacteria, reverse=True)
    return results
