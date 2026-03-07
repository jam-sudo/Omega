"""Liver zonation pharmacokinetics — Phase 494.

Models hepatic drug metabolism considering periportal-to-centrilobular
zonation using a tubes-in-series (conveyor belt) model.

Each zone is treated as a well-stirred compartment. Blood flows sequentially
from zone 1 (periportal) through zone n (centrilobular). Elimination in each
zone follows: CL_zone * C_zone * fu_plasma.

Overall hepatic extraction ratio:
    E = 1 - (C_outlet / C_inlet)

Hepatic clearance:
    CL_hepatic = Q_total * E

References
----------
- Roberts MS & Rowland M, J Pharmacokinet Biopharm. 1986;14(3):243-61.
- Pang KS & Rowland M, J Pharmacokinet Biopharm. 1977;5(6):625-53.
- Bass L et al., J Theor Biol. 1978;72(1):161-84.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LiverZonationResult:
    """Result of a liver zonation simulation.

    Attributes
    ----------
    drug_name : Drug identifier.
    n_zones : Number of hepatic zones modelled.
    times_h : Simulation time grid (hours).
    zone_concentrations : List of n_zones lists, each holding drug
        concentration (mg/L) at each time step.
    outlet_conc_mg_L : Drug concentration exiting the last zone over time.
    overall_extraction_ratio : Steady-state E = 1 - C_outlet / C_inlet.
    cl_hepatic_L_per_h : Hepatic clearance = Q_total * E (L/h).
    zonation_index : Ratio of periportal CLint to centrilobular CLint.
        1.0 means uniform zonation.
    notes : Diagnostic string.
    """

    drug_name: str
    n_zones: int
    times_h: list[float]
    zone_concentrations: list[list[float]]
    outlet_conc_mg_L: list[float]
    overall_extraction_ratio: float
    cl_hepatic_L_per_h: float
    zonation_index: float
    notes: str = field(default="")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def zone_extraction_ratio(
    zone_cl_int: float,
    flow_per_zone_L_per_h: float,
    fu_plasma: float,
) -> float:
    """Compute the extraction ratio for a single hepatic zone.

    Uses the well-stirred model within each zone:
        E_zone = fu * CLint_zone / (Q_zone + fu * CLint_zone)

    Parameters
    ----------
    zone_cl_int          : Intrinsic clearance of this zone (L/h).
    flow_per_zone_L_per_h: Blood flow through this zone (L/h, > 0).
    fu_plasma            : Unbound fraction in plasma (0 < fu ≤ 1).

    Returns
    -------
    float : Extraction ratio [0, 1).
    """
    if flow_per_zone_L_per_h <= 0:
        raise ValueError(f"flow_per_zone_L_per_h must be > 0, got {flow_per_zone_L_per_h}")
    if fu_plasma <= 0 or fu_plasma > 1:
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if zone_cl_int < 0:
        raise ValueError(f"zone_cl_int must be >= 0, got {zone_cl_int}")
    numerator = fu_plasma * zone_cl_int
    denominator = flow_per_zone_L_per_h + numerator
    return numerator / denominator if denominator > 0 else 0.0


def calculate_outlet_concentration(
    inlet_conc: float,
    n_zones: int,
    cl_int_per_zone: list[float],
    flow_L_per_h: float,
    fu_plasma: float,
) -> float:
    """Calculate steady-state outlet concentration after n_zones in series.

    Parameters
    ----------
    inlet_conc       : Drug concentration entering zone 1 (mg/L).
    n_zones          : Number of hepatic zones.
    cl_int_per_zone  : Intrinsic clearance for each zone (L/h), length n_zones.
    flow_L_per_h     : Total hepatic blood flow (L/h, > 0).
    fu_plasma        : Unbound plasma fraction (0 < fu ≤ 1).

    Returns
    -------
    float : Steady-state outlet concentration (mg/L).
    """
    if n_zones <= 0:
        raise ValueError(f"n_zones must be >= 1, got {n_zones}")
    if len(cl_int_per_zone) != n_zones:
        raise ValueError(
            f"cl_int_per_zone must have length n_zones={n_zones}, got {len(cl_int_per_zone)}"
        )
    if flow_L_per_h <= 0:
        raise ValueError(f"flow_L_per_h must be > 0, got {flow_L_per_h}")
    if fu_plasma <= 0 or fu_plasma > 1:
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")

    flow_zone = flow_L_per_h  # each zone sees the full flow (series model)
    c = inlet_conc
    for i in range(n_zones):
        e = zone_extraction_ratio(cl_int_per_zone[i], flow_zone, fu_plasma)
        c = c * (1.0 - e)
    return max(c, 0.0)


def periportal_vs_centrilobular(
    drug_name: str,
    cl_int_total: float,
    zonation_factor: float = 2.0,
    flow_L_per_h: float = 90.0,
    fu_plasma: float = 0.1,
) -> dict:
    """Compare periportal vs. centrilobular CLint distribution.

    Splits cl_int_total into two halves: periportal receives
    ``zonation_factor / (1 + zonation_factor)`` of total CLint, centrilobular
    receives the remainder.  Returns steady-state extraction ratios and
    hepatic CL for each scenario.

    Parameters
    ----------
    drug_name       : Drug identifier.
    cl_int_total    : Total intrinsic clearance (L/h, > 0).
    zonation_factor : Ratio of periportal CLint to centrilobular CLint (> 0).
                      1.0 → uniform; 2.0 → periportal has 2× centrilobular CLint.
    flow_L_per_h    : Total hepatic blood flow (L/h).
    fu_plasma       : Unbound plasma fraction.

    Returns
    -------
    dict with keys:
        drug_name, zonation_factor,
        cl_int_periportal, cl_int_centrilobular,
        e_periportal_first, cl_hepatic_periportal_first,
        e_centrilobular_first, cl_hepatic_centrilobular_first,
        e_uniform, cl_hepatic_uniform,
        notes
    """
    if cl_int_total <= 0:
        raise ValueError(f"cl_int_total must be > 0, got {cl_int_total}")
    if zonation_factor <= 0:
        raise ValueError(f"zonation_factor must be > 0, got {zonation_factor}")

    # Split CLint according to zonation factor
    cl_pp = cl_int_total * zonation_factor / (1.0 + zonation_factor)
    cl_cl = cl_int_total * 1.0 / (1.0 + zonation_factor)
    cl_uniform = cl_int_total / 2.0

    # Periportal first (zone1=periportal, zone2=centrilobular)
    c_out_pp_first = calculate_outlet_concentration(1.0, 2, [cl_pp, cl_cl], flow_L_per_h, fu_plasma)
    e_pp_first = 1.0 - c_out_pp_first
    cl_hep_pp_first = flow_L_per_h * e_pp_first

    # Centrilobular first (zone1=centrilobular, zone2=periportal)
    c_out_cl_first = calculate_outlet_concentration(1.0, 2, [cl_cl, cl_pp], flow_L_per_h, fu_plasma)
    e_cl_first = 1.0 - c_out_cl_first
    cl_hep_cl_first = flow_L_per_h * e_cl_first

    # Uniform (both zones equal)
    c_out_uniform = calculate_outlet_concentration(
        1.0, 2, [cl_uniform, cl_uniform], flow_L_per_h, fu_plasma
    )
    e_uniform = 1.0 - c_out_uniform
    cl_hep_uniform = flow_L_per_h * e_uniform

    return {
        "drug_name": drug_name,
        "zonation_factor": zonation_factor,
        "cl_int_periportal": cl_pp,
        "cl_int_centrilobular": cl_cl,
        "e_periportal_first": e_pp_first,
        "cl_hepatic_periportal_first": cl_hep_pp_first,
        "e_centrilobular_first": e_cl_first,
        "cl_hepatic_centrilobular_first": cl_hep_cl_first,
        "e_uniform": e_uniform,
        "cl_hepatic_uniform": cl_hep_uniform,
        "notes": (
            f"zonation_factor={zonation_factor}: CLint_pp={cl_pp:.3f} CLint_cl={cl_cl:.3f} L/h"
        ),
    }


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def simulate_liver_zonation(
    drug_name: str,
    inlet_conc_mg_L: float,
    flow_L_per_h: float = 90.0,
    n_zones: int = 5,
    cl_int_per_zone: list[float] | None = None,
    fu_plasma: float = 0.1,
    t_end_h: float = 2.0,
    dt_h: float = 0.05,
) -> LiverZonationResult:
    """Simulate drug concentration through hepatic zones over time.

    Each zone is a well-stirred compartment.  Blood flows from zone 1
    (periportal) to zone n (centrilobular) in series.  At each time step
    concentrations are computed for all zones using the steady-state
    well-stirred formula applied sequentially (instantaneous mixing within
    each zone, appropriate for hepatic transit time << PK time scale).

    Parameters
    ----------
    drug_name        : Drug identifier.
    inlet_conc_mg_L  : Concentration entering zone 1 (mg/L).  May be a
                       constant (scalar, > 0) representing portal input.
    flow_L_per_h     : Total hepatic blood flow (L/h, > 0).
    n_zones          : Number of hepatic zones (>= 1).
    cl_int_per_zone  : Intrinsic clearance per zone (L/h).  If None, total
                       CLint of 10 L/h is divided equally among zones.
    fu_plasma        : Unbound plasma fraction (0 < fu ≤ 1).
    t_end_h          : Simulation duration (hours).
    dt_h             : Time step (hours).

    Returns
    -------
    LiverZonationResult
    """
    # --- Validation ---
    if inlet_conc_mg_L <= 0:
        raise ValueError(f"inlet_conc_mg_L must be > 0, got {inlet_conc_mg_L}")
    if flow_L_per_h <= 0:
        raise ValueError(f"flow_L_per_h must be > 0, got {flow_L_per_h}")
    if n_zones < 1:
        raise ValueError(f"n_zones must be >= 1, got {n_zones}")
    if fu_plasma <= 0 or fu_plasma > 1:
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    if cl_int_per_zone is None:
        # Default: 10 L/h total CLint distributed uniformly
        default_total = 10.0
        cl_int_per_zone = [default_total / n_zones] * n_zones
    else:
        cl_int_per_zone = list(cl_int_per_zone)  # defensive copy

    if len(cl_int_per_zone) != n_zones:
        raise ValueError(
            f"cl_int_per_zone length ({len(cl_int_per_zone)}) must equal n_zones ({n_zones})"
        )
    for i, v in enumerate(cl_int_per_zone):
        if v < 0:
            raise ValueError(f"cl_int_per_zone[{i}] must be >= 0, got {v}")

    # --- Build time grid ---
    n_steps = max(1, round(t_end_h / dt_h))
    times_h: list[float] = []
    outlet_conc: list[float] = []
    # n_zones lists for zone concentrations
    zone_concs: list[list[float]] = [[] for _ in range(n_zones)]

    # --- Steady-state zone concentrations (hepatic transit << simulation dt) ---
    # At each time step, the inlet concentration is fixed (constant portal input).
    # We compute the steady-state outlet of each zone sequentially.
    t = 0.0
    for _step in range(n_steps + 1):
        times_h.append(t)
        c = inlet_conc_mg_L
        for z in range(n_zones):
            e = zone_extraction_ratio(cl_int_per_zone[z], flow_L_per_h, fu_plasma)
            c_out = c * (1.0 - e)
            zone_concs[z].append(c_out)
            c = c_out
        outlet_conc.append(c)
        t += dt_h

    # --- Steady-state summary (use last time point) ---
    c_in = inlet_conc_mg_L
    c_out_ss = outlet_conc[-1]
    e_overall = 1.0 - c_out_ss / c_in if c_in > 0 else 0.0
    e_overall = max(0.0, min(1.0, e_overall))
    cl_hepatic = flow_L_per_h * e_overall

    # Zonation index: CLint of zone 1 / CLint of last zone (periportal/centrilobular)
    if n_zones == 1:
        zonation_index = 1.0
    else:
        z_last = cl_int_per_zone[-1]
        z_first = cl_int_per_zone[0]
        zonation_index = z_first / z_last if z_last > 0 else float("inf")

    notes = (
        f"n_zones={n_zones}, Q={flow_L_per_h} L/h, fu={fu_plasma}, "
        f"CLint={[round(v, 3) for v in cl_int_per_zone]}, "
        f"E={e_overall:.4f}, CL_hep={cl_hepatic:.3f} L/h"
    )

    return LiverZonationResult(
        drug_name=drug_name,
        n_zones=n_zones,
        times_h=times_h,
        zone_concentrations=zone_concs,
        outlet_conc_mg_L=outlet_conc,
        overall_extraction_ratio=e_overall,
        cl_hepatic_L_per_h=cl_hepatic,
        zonation_index=zonation_index,
        notes=notes,
    )
