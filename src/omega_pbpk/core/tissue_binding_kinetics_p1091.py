"""Tissue binding kinetics simulation — Phase 1091.

Two-site binding model for drug binding to tissue proteins:
  Site 1 (non-specific, high capacity, low affinity): phospholipids
  Site 2 (specific, low capacity, high affinity): receptors

Models time-course of free/bound drug concentrations, apparent Kp,
and time-dependent fu_tissue. Useful for explaining PK hysteresis.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TissueBindingResult:
    """Results from a two-site tissue binding kinetics simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_free_mg_L: list
    b_site1_mg_L: list
    b_site2_mg_L: list
    c_total_mg_L: list
    fu_tissue_list: list
    fu_tissue_at_1h: float
    fu_tissue_at_24h: float
    kp_apparent_at_1h: float
    t_binding_equilibrium_h: float
    occupancy_site2_at_1h: float
    occupancy_site2_at_24h: float
    notes: str


def _find_at_time(times: list, values: list, target_h: float) -> float:
    """Return value at closest time point to target_h."""
    if not times:
        return 0.0
    best_idx = 0
    best_diff = abs(times[0] - target_h)
    for i, t in enumerate(times):
        diff = abs(t - target_h)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    return values[best_idx]


def simulate_tissue_binding(
    drug_name: str,
    dose_mg: float,
    v_tissue_L: float = 10.0,
    cl_tissue_L_per_h: float = 0.5,
    kon1: float = 0.1,
    koff1: float = 1.0,
    bmax1: float = 100.0,
    kon2: float = 1.0,
    koff2: float = 0.1,
    bmax2: float = 10.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> TissueBindingResult:
    """Simulate two-site tissue binding kinetics with Forward Euler.

    ODE system:
        C_free = free drug in tissue (mg/L)
        B1     = drug bound to site 1 (mg/L)
        B2     = drug bound to site 2 (mg/L)

        dC_free/dt = -kon1*(Bmax1-B1)*C_free + koff1*B1
                     -kon2*(Bmax2-B2)*C_free + koff2*B2
                     -ke*C_free
        dB1/dt = kon1*(Bmax1-B1)*C_free - koff1*B1
        dB2/dt = kon2*(Bmax2-B2)*C_free - koff2*B2

    IV input: C_free(0) = dose_mg / v_tissue_L

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        IV dose administered at t=0 (mg).
    v_tissue_L:
        Tissue volume (L).
    cl_tissue_L_per_h:
        Tissue elimination clearance (L/h). ke = cl_tissue / v_tissue.
    kon1:
        Association rate constant for site 1 (L/(mg*h)).
    koff1:
        Dissociation rate constant for site 1 (1/h).
    kon2:
        Association rate constant for site 2 (L/(mg*h)).
    koff2:
        Dissociation rate constant for site 2 (1/h).
    bmax1:
        Maximum binding capacity for site 1 (mg/L).
    bmax2:
        Maximum binding capacity for site 2 (mg/L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    TissueBindingResult
    """
    # Validation
    if dose_mg <= 0.0:
        raise ValueError("dose_mg must be > 0")
    if v_tissue_L <= 0.0:
        raise ValueError("v_tissue_L must be > 0")
    if cl_tissue_L_per_h <= 0.0:
        raise ValueError("cl_tissue_L_per_h must be > 0")
    if kon1 <= 0.0:
        raise ValueError("kon1 must be > 0")
    if koff1 <= 0.0:
        raise ValueError("koff1 must be > 0")
    if kon2 <= 0.0:
        raise ValueError("kon2 must be > 0")
    if koff2 <= 0.0:
        raise ValueError("koff2 must be > 0")
    if bmax1 <= 0.0:
        raise ValueError("bmax1 must be > 0")
    if bmax2 <= 0.0:
        raise ValueError("bmax2 must be > 0")

    ke = cl_tissue_L_per_h / v_tissue_L  # h^-1

    # Initial conditions: IV bolus → all drug as free
    c_free = dose_mg / v_tissue_L
    b1 = 0.0
    b2 = 0.0

    # Storage
    times_h: list = []
    c_free_list: list = []
    b1_list: list = []
    b2_list: list = []
    c_total_list: list = []
    fu_list: list = []

    n_steps = int(t_end_h / dt_h) + 1

    for step in range(n_steps):
        t = step * dt_h
        times_h.append(t)
        c_free_list.append(c_free)
        b1_list.append(b1)
        b2_list.append(b2)
        c_total = c_free + b1 + b2
        c_total_list.append(c_total)
        fu = c_free / c_total if c_total > 0.0 else 1.0
        fu_list.append(fu)

        if step == n_steps - 1:
            break

        # Forward Euler derivatives
        dc_free = (
            -kon1 * (bmax1 - b1) * c_free
            + koff1 * b1
            - kon2 * (bmax2 - b2) * c_free
            + koff2 * b2
            - ke * c_free
        )
        db1 = kon1 * (bmax1 - b1) * c_free - koff1 * b1
        db2 = kon2 * (bmax2 - b2) * c_free - koff2 * b2

        c_free = max(c_free + dc_free * dt_h, 0.0)
        b1 = max(b1 + db1 * dt_h, 0.0)
        b2 = max(b2 + db2 * dt_h, 0.0)

        # Enforce bound cannot exceed Bmax
        b1 = min(b1, bmax1)
        b2 = min(b2, bmax2)

    # Extract values at specific time points
    fu_at_1h = _find_at_time(times_h, fu_list, 1.0)
    fu_at_24h = _find_at_time(times_h, fu_list, 24.0)

    c_total_at_1h = _find_at_time(times_h, c_total_list, 1.0)
    # Simplified plasma reference: dose / Vd_plasma (use 5 L as typical plasma volume)
    vd_plasma_ref = 5.0
    c_plasma_ref = dose_mg / vd_plasma_ref
    kp_apparent_at_1h = c_total_at_1h / c_plasma_ref if c_plasma_ref > 0.0 else 1.0

    b2_at_1h = _find_at_time(times_h, b2_list, 1.0)
    b2_at_24h = _find_at_time(times_h, b2_list, 24.0)
    occupancy_site2_at_1h = b2_at_1h / bmax2
    occupancy_site2_at_24h = b2_at_24h / bmax2

    # Binding equilibrium: find time when |fu change| < 1% over 1h window
    t_binding_equilibrium_h = t_end_h  # default if not found
    window_steps = max(1, int(1.0 / dt_h))
    for i in range(window_steps, len(fu_list)):
        fu_now = fu_list[i]
        fu_prev = fu_list[i - window_steps]
        if fu_prev > 0.0:
            change = abs(fu_now - fu_prev) / fu_prev
            if change < 0.01:
                t_binding_equilibrium_h = times_h[i]
                break

    notes = (
        f"Two-site tissue binding kinetics; drug={drug_name}; "
        f"dose={dose_mg} mg; v_tissue={v_tissue_L} L; "
        f"kon1={kon1}, koff1={koff1}, Bmax1={bmax1}; "
        f"kon2={kon2}, koff2={koff2}, Bmax2={bmax2}."
    )

    return TissueBindingResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_free_mg_L=c_free_list,
        b_site1_mg_L=b1_list,
        b_site2_mg_L=b2_list,
        c_total_mg_L=c_total_list,
        fu_tissue_list=fu_list,
        fu_tissue_at_1h=fu_at_1h,
        fu_tissue_at_24h=fu_at_24h,
        kp_apparent_at_1h=kp_apparent_at_1h,
        t_binding_equilibrium_h=t_binding_equilibrium_h,
        occupancy_site2_at_1h=occupancy_site2_at_1h,
        occupancy_site2_at_24h=occupancy_site2_at_24h,
        notes=notes,
    )


def compare_affinities(
    drug_name: str,
    dose_mg: float,
    koff2_list: list,
) -> list:
    """Compare tissue binding across different site-2 affinities.

    Lower koff2 = higher affinity = higher receptor occupancy.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        IV dose (mg).
    koff2_list:
        List of koff2 values (1/h) to compare.

    Returns
    -------
    list[TissueBindingResult]
        Results sorted by occupancy_site2_at_24h descending.
    """
    results = []
    for koff2 in koff2_list:
        r = simulate_tissue_binding(
            drug_name=drug_name,
            dose_mg=dose_mg,
            koff2=koff2,
        )
        results.append(r)

    # Sort by occupancy_site2_at_24h descending (higher affinity = lower koff2 = higher occupancy)
    results.sort(key=lambda r: r.occupancy_site2_at_24h, reverse=True)
    return results
