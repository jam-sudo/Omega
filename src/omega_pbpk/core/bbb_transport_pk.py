"""Phase 705 — Blood-brain barrier transport pharmacokinetics with active transport."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["BBBTransportResult", "simulate_bbb_transport", "compare_pgp_effect"]


@dataclass
class BBBTransportResult:
    """Result from BBB transport PK simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    psa: float
    f_pgp: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_brain_mg_L: list = field(default_factory=list)
    cmax_plasma: float = 0.0
    cmax_brain: float = 0.0
    tmax_brain_h: float = 0.0
    auc_plasma: float = 0.0
    auc_brain: float = 0.0
    brain_to_plasma_ratio: float = 0.0
    ps_in_L_per_h: float = 0.0
    ps_out_L_per_h: float = 0.0
    bbb_penetration_class: str = "low"
    notes: str = ""


def _compute_ps_in(logP: float, psa: float, mw: float) -> float:
    """Compute permeability-surface area product for influx (L/h)."""
    log_ps = 0.5 * logP - 0.8 * (psa / 100.0) - 0.3 * max(0.0, mw - 300.0) / 100.0
    ps = 10.0**log_ps
    # Clamp to [0.001, 10.0]
    return max(0.001, min(10.0, ps))


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Manual trapezoidal AUC integration."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i] + concs[i - 1]) * dt
    return auc


def simulate_bbb_transport(
    drug_name: str,
    dose_mg: float,
    logP: float = 2.0,
    psa: float = 70.0,
    mw: float = 400.0,
    f_pgp: float = 0.0,
    ke_sys: float = 0.1,
    vd_plasma_L: float = 10.0,
    vd_brain_L: float = 1.5,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> BBBTransportResult:
    """Simulate BBB transport PK using 2-compartment (plasma + brain ECF) Forward Euler ODE.

    Args:
        drug_name: Name of the drug compound.
        dose_mg: IV bolus dose in mg.
        logP: Octanol-water partition coefficient.
        psa: Polar surface area (Angstrom^2).
        mw: Molecular weight (Da).
        f_pgp: P-gp efflux factor (0 = no efflux, >0 = increasing efflux).
        ke_sys: Systemic elimination rate constant (1/h).
        vd_plasma_L: Volume of distribution for plasma compartment (L).
        vd_brain_L: Volume of distribution for brain compartment (L).
        t_end_h: Simulation end time (h).
        dt_h: Forward Euler time step (h).

    Returns:
        BBBTransportResult with concentration-time profiles and PK parameters.

    Raises:
        ValueError: If input parameters are invalid.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if f_pgp < 0:
        raise ValueError(f"f_pgp must be >= 0, got {f_pgp}")
    if vd_plasma_L <= 0:
        raise ValueError(f"vd_plasma_L must be positive, got {vd_plasma_L}")
    if vd_brain_L <= 0:
        raise ValueError(f"vd_brain_L must be positive, got {vd_brain_L}")
    if not (0 <= psa <= 200):
        raise ValueError(f"psa must be in [0, 200], got {psa}")

    # Compute permeability parameters
    ps_in = _compute_ps_in(logP, psa, mw)
    # Baseline PS_out (no P-gp): equal to PS_in for passive diffusion
    # With P-gp: PS_out = PS_in * (1 + f_pgp) at low brain concentrations
    ps_out_baseline = ps_in * (1.0 + f_pgp)

    # Initial conditions: IV bolus into plasma
    # C_plasma(0) = dose_mg / vd_plasma_L (mg/L)
    c_plasma = dose_mg / vd_plasma_L
    c_brain = 0.0

    n_steps = int(t_end_h / dt_h) + 1
    times = []
    c_plasma_list = []
    c_brain_list = []

    km_pgp = 1.0  # mg/L reference for P-gp saturation

    for i in range(n_steps):
        t = i * dt_h
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_brain_list.append(c_brain)

        # P-gp efflux is concentration-dependent (Michaelis-Menten simplified)
        # PS_out = PS_in * (1 + f_pgp * C_plasma / (km_pgp + C_plasma))
        pgp_factor = f_pgp * c_plasma / (km_pgp + c_plasma) if (km_pgp + c_plasma) > 0 else 0.0
        ps_out_dynamic = ps_in * (1.0 + pgp_factor)

        # Net transfer rates (mg/h)
        influx_to_brain = ps_in * c_plasma  # PS_in * C_plasma [L/h * mg/L = mg/h]
        efflux_from_brain = ps_out_dynamic * c_brain  # PS_out * C_brain

        # ODEs:
        # dC_plasma/dt = -ke_sys * C_plasma - (influx - efflux) / vd_plasma
        # dC_brain/dt = (influx - efflux) / vd_brain - ke_brain * C_brain
        # ke_brain: brain clearance (elimination from brain, small)
        ke_brain = 0.05  # h^-1, slow brain elimination

        dc_plasma_dt = -ke_sys * c_plasma - (influx_to_brain - efflux_from_brain) / vd_plasma_L
        dc_brain_dt = (influx_to_brain - efflux_from_brain) / vd_brain_L - ke_brain * c_brain

        # Forward Euler update
        c_plasma = max(0.0, c_plasma + dc_plasma_dt * dt_h)
        c_brain = max(0.0, c_brain + dc_brain_dt * dt_h)

    # Compute PK parameters
    cmax_plasma = max(c_plasma_list)
    cmax_brain = max(c_brain_list)

    # tmax_brain
    tmax_brain_h = 0.0
    for i, c in enumerate(c_brain_list):
        if c == cmax_brain:
            tmax_brain_h = times[i]
            break

    auc_plasma = _trapezoidal_auc(times, c_plasma_list)
    auc_brain = _trapezoidal_auc(times, c_brain_list)

    # Brain-to-plasma ratio (Kp_brain = AUC_brain / AUC_plasma)
    kp_brain = auc_brain / auc_plasma if auc_plasma > 0 else 0.0

    # BBB penetration classification
    if kp_brain > 0.3:
        bbb_class = "high"
    elif kp_brain >= 0.1:
        bbb_class = "moderate"
    else:
        bbb_class = "low"

    notes_parts = [
        f"PS_in={ps_in:.4f} L/h",
        f"PS_out_baseline={ps_out_baseline:.4f} L/h",
        f"Kp_brain(AUC ratio)={kp_brain:.4f}",
    ]
    if f_pgp > 0:
        notes_parts.append(f"P-gp efflux active (f_pgp={f_pgp})")
    notes = "; ".join(notes_parts)

    return BBBTransportResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        psa=psa,
        f_pgp=f_pgp,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_brain_mg_L=c_brain_list,
        cmax_plasma=cmax_plasma,
        cmax_brain=cmax_brain,
        tmax_brain_h=tmax_brain_h,
        auc_plasma=auc_plasma,
        auc_brain=auc_brain,
        brain_to_plasma_ratio=kp_brain,
        ps_in_L_per_h=ps_in,
        ps_out_L_per_h=ps_out_baseline,
        bbb_penetration_class=bbb_class,
        notes=notes,
    )


def compare_pgp_effect(
    drug_name: str,
    dose_mg: float,
    logP: float = 2.0,
    psa: float = 70.0,
    f_pgp_values: list | None = None,
) -> list[BBBTransportResult]:
    """Compare P-gp efflux effect on BBB penetration across multiple f_pgp values.

    Args:
        drug_name: Name of the drug compound.
        dose_mg: IV bolus dose in mg.
        logP: Octanol-water partition coefficient.
        psa: Polar surface area (Angstrom^2).
        f_pgp_values: List of P-gp efflux factors to compare. Defaults to [0.0, 0.5, 1.0, 2.0].

    Returns:
        List of BBBTransportResult sorted by f_pgp ascending.
    """
    if f_pgp_values is None:
        f_pgp_values = [0.0, 0.5, 1.0, 2.0]

    results = []
    for f_pgp in sorted(f_pgp_values):
        result = simulate_bbb_transport(
            drug_name=drug_name,
            dose_mg=dose_mg,
            logP=logP,
            psa=psa,
            f_pgp=f_pgp,
        )
        results.append(result)

    return results
