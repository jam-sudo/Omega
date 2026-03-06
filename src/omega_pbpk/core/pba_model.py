"""Physiologically-Based Absorption (PBA) Model.

A simplified 4-segment GI absorption model combining transit, pH-dependent
dissolution, and segment-specific absorption (simplified ACAT-like model).

Segments: stomach, SI proximal, SI distal, colon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

__all__ = [
    "PBAResult",
    "simulate_pba",
    "compare_particle_sizes",
]

# ---------------------------------------------------------------------------
# Segment physiological parameters
# ---------------------------------------------------------------------------

_SEGMENTS = ["stomach", "si_proximal", "si_distal", "colon"]

_SEG_PH = {
    "stomach": 1.5,
    "si_proximal": 6.5,
    "si_distal": 7.0,
    "colon": 7.5,
}

_SEG_TRANSIT_H = {
    "stomach": 0.25,
    "si_proximal": 1.5,
    "si_distal": 1.5,
    "colon": 20.0,
}

_SEG_VOLUME_ML = {
    "stomach": 250.0,
    "si_proximal": 100.0,
    "si_distal": 150.0,
    "colon": 500.0,
}

# No absorption in stomach
_SEG_PEFF_SCALE = {
    "stomach": 0.0,
    "si_proximal": 1.0,
    "si_distal": 0.7,
    "colon": 0.1,
}

_INTESTINAL_RADIUS_CM = 0.5


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PBAResult:
    """Result of a physiologically-based absorption simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    pka: float
    compound_type: str
    solubility_mg_mL_pH7: float
    particle_radius_um: float

    times_h: list[float]
    c_plasma_mg_L: list[float]

    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float

    f_absorbed_total: float
    f_absorbed_per_segment: dict[str, float]

    notes: list[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _peff_cm_per_s(logP: float) -> float:
    """Empirical Peff (cm/s) from logP — simplified Sun 2002 relationship."""
    return 10.0 ** (-4.28 + 0.278 * logP)


def _ka_per_h(peff_cm_per_s: float) -> float:
    """Convert Peff (cm/s) to absorption rate constant (h^-1).

    ka = 2 * Peff * 3600 / radius_intestine_cm
    """
    return 2.0 * peff_cm_per_s * 3600.0 / _INTESTINAL_RADIUS_CM


def _solubility_at_ph(
    solubility_ph7: float,
    ph: float,
    pka: float,
    compound_type: str,
) -> float:
    """pH-adjusted solubility via Henderson-Hasselbalch.

    For acid: S(pH) = S0 * (1 + 10^(pH - pKa))
    For base: S(pH) = S0 * (1 + 10^(pKa - pH))
    For neutral: S(pH) = S_pH7 (unchanged)

    S0 is back-calculated from S_pH7.
    """
    if compound_type == "neutral":
        return solubility_ph7

    if compound_type == "acid":
        # S_pH7 = S0 * (1 + 10^(7 - pKa))
        factor_ph7 = 1.0 + 10.0 ** (7.0 - pka)
        s0 = solubility_ph7 / factor_ph7
        return s0 * (1.0 + 10.0 ** (ph - pka))

    # base
    factor_ph7 = 1.0 + 10.0 ** (pka - 7.0)
    s0 = solubility_ph7 / factor_ph7
    return s0 * (1.0 + 10.0 ** (pka - ph))


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_pba(
    drug_name: str,
    dose_mg: float,
    logP: float,
    pka: float,
    compound_type: str,
    solubility_mg_mL_pH7: float,
    particle_radius_um: float = 50.0,
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> PBAResult:
    """Simulate physiologically-based absorption with 4 GI segments.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Oral dose in mg.
    logP:
        Octanol-water partition coefficient (log scale).
    pka:
        Ionisation constant.
    compound_type:
        One of ``"acid"``, ``"base"``, or ``"neutral"``.
    solubility_mg_mL_pH7:
        Aqueous solubility at pH 7 (mg/mL).
    particle_radius_um:
        Initial particle radius in micrometres (affects dissolution rate).
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        ODE integration step size (h).

    Returns
    -------
    PBAResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if solubility_mg_mL_pH7 <= 0:
        raise ValueError("solubility_mg_mL_pH7 must be > 0")
    if particle_radius_um <= 0:
        raise ValueError("particle_radius_um must be > 0")
    if compound_type not in ("acid", "base", "neutral"):
        raise ValueError('compound_type must be "acid", "base", or "neutral"')

    notes: list[str] = []

    # --- Pre-compute segment Peff & ka ---
    peff_base = _peff_cm_per_s(logP)
    ka_base = _ka_per_h(peff_base)

    seg_ka: dict[str, float] = {}
    for seg in _SEGMENTS:
        seg_ka[seg] = ka_base * _SEG_PEFF_SCALE[seg]

    # --- Forward Euler simulation ---
    # States: C_plasma (mg/L), and for each segment: mass_dissolved (mg)
    # Drug moves sequentially through segments; each segment has a transit time.
    # Dissolution approximated by solubility-limited instantaneous equilibrium
    # within each segment (simplified: dissolved fraction = min(dose, solubility*vol)).

    kel = cl_L_per_h / vd_L  # elimination rate constant (h^-1)

    times = np.arange(0.0, t_end_h + dt_h, dt_h)
    n_steps = len(times)

    c_plasma = np.zeros(n_steps)

    # Segment transit rates (fraction leaving per hour)
    seg_transit_rate: dict[str, float] = {seg: 1.0 / _SEG_TRANSIT_H[seg] for seg in _SEGMENTS}

    # Dissolved mass in each segment (mg) — starts in stomach after dosing
    seg_dissolved = {seg: 0.0 for seg in _SEGMENTS}
    seg_undissolved = {seg: 0.0 for seg in _SEGMENTS}

    # At t=0: entire dose enters stomach
    sol_stomach = _solubility_at_ph(solubility_mg_mL_pH7, _SEG_PH["stomach"], pka, compound_type)
    max_dissolved_stomach = (
        sol_stomach * _SEG_VOLUME_ML["stomach"] / 1000.0
    )  # mg (vol in mL → L for mg/L×L)
    # Actually: solubility mg/mL × volume mL = mg
    max_dissolved_stomach = sol_stomach * _SEG_VOLUME_ML["stomach"]

    seg_dissolved["stomach"] = min(dose_mg, max_dissolved_stomach)
    seg_undissolved["stomach"] = max(0.0, dose_mg - max_dissolved_stomach)

    # Track absorbed mass per segment
    seg_absorbed_mg = {seg: 0.0 for seg in _SEGMENTS}

    for i in range(1, n_steps):
        cp = c_plasma[i - 1]

        # --- Dissolution: undissolved → dissolved within each segment ---
        for seg in _SEGMENTS:
            if seg_undissolved[seg] > 0.0:
                sol = _solubility_at_ph(solubility_mg_mL_pH7, _SEG_PH[seg], pka, compound_type)
                max_diss = sol * _SEG_VOLUME_ML[seg]
                space = max(0.0, max_diss - seg_dissolved[seg])
                # Dissolve up to available space; particle-size-based dissolution rate
                # Use Noyes-Whitney simplified: dM/dt ∝ 1/radius × (Cs - C_bulk)
                # Here simplified as fractional rate ~ 1/(radius_um/10) per h
                diss_rate = 1.0 / (particle_radius_um / 10.0)  # h^-1 (larger = slower)
                diss_amount = min(
                    seg_undissolved[seg] * diss_rate * dt_h,
                    space,
                )
                diss_amount = max(0.0, diss_amount)
                seg_undissolved[seg] -= diss_amount
                seg_dissolved[seg] += diss_amount

        # --- Absorption from each segment into plasma ---
        total_abs_this_step = 0.0
        for seg in _SEGMENTS:
            ka = seg_ka[seg]
            if ka > 0.0 and seg_dissolved[seg] > 0.0:
                abs_amount = ka * seg_dissolved[seg] * dt_h
                abs_amount = min(abs_amount, seg_dissolved[seg])
                seg_dissolved[seg] -= abs_amount
                seg_absorbed_mg[seg] += abs_amount
                total_abs_this_step += abs_amount

        # --- Transit: dissolved + undissolved move to next segment ---
        for idx, seg in enumerate(_SEGMENTS[:-1]):
            next_seg = _SEGMENTS[idx + 1]
            rate = seg_transit_rate[seg]

            # Dissolved transit
            transit_diss = seg_dissolved[seg] * rate * dt_h
            transit_diss = min(transit_diss, seg_dissolved[seg])
            seg_dissolved[seg] -= transit_diss

            # On entering next segment: re-equilibrate with next-segment solubility
            sol_next = _solubility_at_ph(
                solubility_mg_mL_pH7, _SEG_PH[next_seg], pka, compound_type
            )
            max_diss_next = sol_next * _SEG_VOLUME_ML[next_seg]
            space_next = max(0.0, max_diss_next - seg_dissolved[next_seg])
            new_dissolved_portion = min(transit_diss, space_next)
            new_undissolved_portion = transit_diss - new_dissolved_portion
            seg_dissolved[next_seg] += new_dissolved_portion
            seg_undissolved[next_seg] += new_undissolved_portion

            # Undissolved transit
            transit_undiss = seg_undissolved[seg] * rate * dt_h
            transit_undiss = min(transit_undiss, seg_undissolved[seg])
            seg_undissolved[seg] -= transit_undiss
            seg_undissolved[next_seg] += transit_undiss

        # --- Plasma PK (1-cpt elimination) ---
        absorbed_into_plasma = total_abs_this_step / vd_L  # mg → mg/L
        dc_dt = absorbed_into_plasma / dt_h - kel * cp
        cp_new = max(0.0, cp + dc_dt * dt_h)
        c_plasma[i] = cp_new

    # --- Compute PK metrics ---
    c_arr = c_plasma
    t_arr = times

    cmax = float(np.max(c_arr))
    tmax_idx = int(np.argmax(c_arr))
    tmax = float(t_arr[tmax_idx])
    auc = float(np_trapz(c_arr, t_arr))

    total_absorbed = sum(seg_absorbed_mg.values())
    f_total = min(1.0, total_absorbed / dose_mg)

    f_per_seg = {seg: seg_absorbed_mg[seg] / dose_mg for seg in _SEGMENTS}

    if logP < 0:
        notes.append("Low logP — limited passive absorption expected.")
    if f_total < 0.2:
        notes.append("Low oral bioavailability predicted (<20%).")

    return PBAResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        pka=pka,
        compound_type=compound_type,
        solubility_mg_mL_pH7=solubility_mg_mL_pH7,
        particle_radius_um=particle_radius_um,
        times_h=t_arr.tolist(),
        c_plasma_mg_L=c_arr.tolist(),
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        f_absorbed_total=f_total,
        f_absorbed_per_segment=f_per_seg,
        notes=notes,
    )


def compare_particle_sizes(
    drug_name: str,
    dose_mg: float,
    logP: float,
    pka: float,
    compound_type: str,
    solubility_mg_mL_pH7: float,
    particle_radii_um: list[float],
    **kwargs,
) -> list[PBAResult]:
    """Compare PBA simulation results for multiple particle sizes.

    Results are sorted by ``f_absorbed_total`` descending (highest first).

    Parameters
    ----------
    particle_radii_um:
        List of particle radii (micrometres) to evaluate.
    **kwargs:
        Additional keyword arguments forwarded to :func:`simulate_pba`.

    Returns
    -------
    list[PBAResult]
        One entry per particle size, sorted by f_absorbed_total descending.
    """
    if not particle_radii_um:
        raise ValueError("particle_radii_um must contain at least one value")

    results = [
        simulate_pba(
            drug_name=drug_name,
            dose_mg=dose_mg,
            logP=logP,
            pka=pka,
            compound_type=compound_type,
            solubility_mg_mL_pH7=solubility_mg_mL_pH7,
            particle_radius_um=r,
            **kwargs,
        )
        for r in particle_radii_um
    ]

    return sorted(results, key=lambda r: r.f_absorbed_total, reverse=True)
