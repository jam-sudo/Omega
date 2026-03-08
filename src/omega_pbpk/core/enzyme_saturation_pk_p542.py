"""Drug metabolizing enzyme saturation: 1-compartment PK with Michaelis-Menten elimination.

Models nonlinear (saturable) drug elimination via enzyme kinetics:
  dC/dt = -Vmax * C / (Km + C)

At low C (C << Km): first-order (linear) elimination
At high C (C >> Km): zero-order (saturable) elimination — dose-disproportional AUC

References
----------
- Gibaldi M, Perrier D. Pharmacokinetics. 2nd ed. 1982.
- Wagner JG. Fundamentals of Clinical Pharmacokinetics. 1975.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnzymeSaturationResult:
    """Result from enzyme saturation PK simulation (Phase 542).

    Attributes
    ----------
    drug_name : Name of the drug compound.
    dose_mg : Administered dose (mg).
    vmax_mg_per_h : Maximum elimination rate (mg/h).
    km_mg_per_L : Michaelis constant — concentration at half-maximal rate (mg/L).
    vd_L : Volume of distribution (L).
    cmax_mg_L : Peak plasma concentration (mg/L) — equals dose/Vd for IV bolus.
    times_h : Simulation time points (h).
    c_plasma_mg_L : Plasma concentration vs time (mg/L).
    auc_mg_h_per_L : Area under concentration-time curve (trapezoidal, mg*h/L).
    t_half_h : Terminal half-life at low concentration: 0.693*Vd/CL_intrinsic
        where CL_int = Vmax/Km.
    t_km_h : Time (h) when plasma concentration first drops to Km.
        0.0 if Cmax <= Km (never saturated).
    saturation_pct_at_t0 : Percent enzyme saturated at t=0:
        C[0] / (C[0] + Km) * 100.
    linearity_class : 'linear' if Cmax/Km < 1,
        'nonlinear' if Cmax/Km > 5, 'transition' otherwise.
    notes : Summary of key findings.
    """

    drug_name: str
    dose_mg: float
    vmax_mg_per_h: float
    km_mg_per_L: float
    vd_L: float
    cmax_mg_L: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    auc_mg_h_per_L: float = 0.0
    t_half_h: float = 0.0
    t_km_h: float = 0.0
    saturation_pct_at_t0: float = 0.0
    linearity_class: str = "linear"
    notes: str = ""


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Compute AUC using manual trapezoidal integration."""
    return sum(
        0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def simulate_enzyme_saturation(
    drug_name: str,
    dose_mg: float,
    vmax_mg_per_h: float,
    km_mg_per_L: float,
    vd_L: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> EnzymeSaturationResult:
    """Simulate IV bolus 1-compartment PK with Michaelis-Menten elimination.

    Parameters
    ----------
    drug_name : Name of the drug compound.
    dose_mg : IV bolus dose (mg), must be > 0.
    vmax_mg_per_h : Maximum elimination rate (mg/h), must be > 0.
    km_mg_per_L : Michaelis constant (mg/L), must be > 0.
    vd_L : Volume of distribution (L), must be > 0.
    t_end_h : Simulation end time (h).
    dt_h : Forward Euler time step (h).

    Returns
    -------
    EnzymeSaturationResult with simulated PK profile and derived metrics.

    Raises
    ------
    ValueError : If dose_mg, vmax_mg_per_h, km_mg_per_L, or vd_L <= 0.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if vmax_mg_per_h <= 0:
        raise ValueError("vmax_mg_per_h must be > 0")
    if km_mg_per_L <= 0:
        raise ValueError("km_mg_per_L must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    n_steps = max(int(round(t_end_h / dt_h)), 1)
    times: list[float] = []
    concs: list[float] = []

    # IV bolus: C(0) = dose / Vd
    c = dose_mg / vd_L
    cmax = c

    # Forward Euler ODE integration
    for i in range(n_steps + 1):
        t = i * dt_h
        times.append(t)
        concs.append(max(0.0, c))
        if i < n_steps:
            dc_dt = -vmax_mg_per_h * c / (km_mg_per_L + c) / vd_L
            c = max(0.0, c + dc_dt * dt_h)

    auc = _trapezoidal_auc(times, concs)

    # Terminal half-life at low concentration (first-order limit)
    # CL_intrinsic = Vmax/Km → ke = Vmax/(Km*Vd) → t_half = 0.693*Vd*Km/Vmax
    cl_intrinsic = vmax_mg_per_h / km_mg_per_L  # L/h
    t_half = 0.693 * vd_L / cl_intrinsic  # h

    # Time when C first drops to Km (transition from saturation to linear)
    t_km = 0.0
    if cmax > km_mg_per_L:
        for t_val, c_val in zip(times, concs):
            if c_val <= km_mg_per_L:
                t_km = t_val
                break

    # Saturation percent at t=0
    sat_pct = concs[0] / (concs[0] + km_mg_per_L) * 100.0 if concs else 0.0

    # Linearity class based on Cmax/Km ratio
    ratio = cmax / km_mg_per_L
    if ratio < 1.0:
        linearity_class = "linear"
    elif ratio > 5.0:
        linearity_class = "nonlinear"
    else:
        linearity_class = "transition"

    notes = (
        f"{drug_name}: dose={dose_mg} mg, Vd={vd_L} L. "
        f"Vmax={vmax_mg_per_h} mg/h, Km={km_mg_per_L} mg/L. "
        f"Cmax={cmax:.4f} mg/L, AUC={auc:.4f} mg*h/L. "
        f"Saturation at t=0: {sat_pct:.1f}%. "
        f"Linearity: {linearity_class} (Cmax/Km={ratio:.2f}). "
        f"t_half (linear limit)={t_half:.2f} h."
    )

    return EnzymeSaturationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        vmax_mg_per_h=vmax_mg_per_h,
        km_mg_per_L=km_mg_per_L,
        vd_L=vd_L,
        cmax_mg_L=cmax,
        times_h=times,
        c_plasma_mg_L=concs,
        auc_mg_h_per_L=auc,
        t_half_h=t_half,
        t_km_h=t_km,
        saturation_pct_at_t0=sat_pct,
        linearity_class=linearity_class,
        notes=notes,
    )


def dose_proportionality_test(
    drug_name: str,
    doses_mg: list,
    vmax_mg_per_h: float,
    km_mg_per_L: float,
    vd_L: float,
) -> list:
    """Test dose proportionality across multiple doses with MM elimination.

    At saturation, doubling the dose yields >2x AUC (superproportional).

    Parameters
    ----------
    drug_name : Name of the drug compound.
    doses_mg : List of doses to test (mg).
    vmax_mg_per_h : Maximum elimination rate (mg/h).
    km_mg_per_L : Michaelis constant (mg/L).
    vd_L : Volume of distribution (L).

    Returns
    -------
    List of dicts {'dose_mg': float, 'auc_mg_h_per_L': float} sorted by dose ascending.
    """
    results = []
    for dose in sorted(doses_mg):
        res = simulate_enzyme_saturation(
            drug_name=drug_name,
            dose_mg=dose,
            vmax_mg_per_h=vmax_mg_per_h,
            km_mg_per_L=km_mg_per_L,
            vd_L=vd_L,
        )
        results.append({"dose_mg": dose, "auc_mg_h_per_L": res.auc_mg_h_per_L})
    return results


__all__ = [
    "EnzymeSaturationResult",
    "simulate_enzyme_saturation",
    "dose_proportionality_test",
]
