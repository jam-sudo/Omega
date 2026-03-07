"""Phase 854 — Bile Acid Conjugation and Absorption PK model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BileAcidConjugationPKResult:
    drug_name: str
    dose_mg: float
    ka_conj_per_h: float
    k_cleavage_per_h: float
    times_h: list[float]
    c_plasma_drug_mg_L: list[float]
    a_plasma_conj_mg: list[float]
    a_gut_conjugate_mg: list[float]
    cmax_drug_mg_L: float
    tmax_drug_h: float
    auc_drug_mg_h_per_L: float
    f_absorbed_conj: float
    f_converted_to_drug: float
    notes: str


def _trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal integration."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (values[i - 1] + values[i]) * dt
    return auc


def simulate_bile_acid_conjugation_pk(
    drug_name: str,
    dose_mg: float,
    ka_conj_per_h: float = 0.8,
    k_cleavage_per_h: float = 0.2,
    k_elim_conj_per_h: float = 0.1,
    k_elim_drug_per_h: float = 0.5,
    vd_drug_L: float = 10.0,
    recycling_cycles: int = 2,
    recycled_fraction: float = 0.4,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> BileAcidConjugationPKResult:
    """Simulate bile acid conjugate oral PK with enterohepatic recycling.

    Three-compartment Forward Euler model:
      A_gut     — conjugate in GI lumen (mg)
      A_conj    — conjugate in plasma (mg)
      C_drug    — free drug plasma concentration (mg/L)

    ODEs:
      dA_gut/dt    = -ka_conj * A_gut  [+ recycling pulses]
      dA_conj/dt   = ka_conj * A_gut - (k_cleavage + k_elim_conj) * A_conj
      dC_drug/dt   = k_cleavage * A_conj / vd_drug - k_elim_drug * C_drug

    Parameters
    ----------
    drug_name : str
    dose_mg : float
        Oral dose of the conjugated prodrug (mg). Must be > 0.
    ka_conj_per_h : float
        ASBT-mediated absorption rate constant (h^-1). Must be > 0.
    k_cleavage_per_h : float
        Enzymatic cleavage rate of conjugate in plasma (h^-1). Must be >= 0.
    k_elim_conj_per_h : float
        Elimination rate of intact conjugate (h^-1).
    k_elim_drug_per_h : float
        Elimination rate of free drug (h^-1).
    vd_drug_L : float
        Volume of distribution of the free drug (L).
    recycling_cycles : int
        Number of enterohepatic recycling cycles. Must be >= 0.
    recycled_fraction : float
        Fraction of conjugate recycled back to gut each cycle.
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Forward Euler step size (h).

    Returns
    -------
    BileAcidConjugationPKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if ka_conj_per_h <= 0:
        raise ValueError("ka_conj_per_h must be > 0")
    if k_cleavage_per_h < 0:
        raise ValueError("k_cleavage_per_h must be >= 0")
    if recycling_cycles < 0:
        raise ValueError("recycling_cycles must be >= 0")

    # Precompute recycling pulse times (each cycle occurs at multiples of 6 h)
    recycle_times = set()
    for cyc in range(1, recycling_cycles + 1):
        recycle_time = cyc * 6.0
        if recycle_time < t_end_h:
            recycle_times.add(recycle_time)

    n_steps = max(int(round(t_end_h / dt_h)), 1)

    times: list[float] = []
    c_drug_traj: list[float] = []
    a_conj_traj: list[float] = []
    a_gut_traj: list[float] = []

    # Initial conditions
    A_gut = dose_mg
    A_conj = 0.0
    C_drug = 0.0
    t = 0.0

    # Track absorbed conjugate and converted drug for summary stats
    total_absorbed_conj = 0.0
    total_converted = 0.0

    times.append(t)
    c_drug_traj.append(C_drug)
    a_conj_traj.append(A_conj)
    a_gut_traj.append(A_gut)

    for _step in range(n_steps):
        t_next = t + dt_h

        # Recycling pulse: add recycled conjugate to gut at recycle time boundaries
        # Apply pulse at the start of the interval if the boundary falls within [t, t_next)
        recycle_added = 0.0
        for rt in list(recycle_times):
            if t <= rt < t_next:
                pulse = recycled_fraction * dose_mg
                recycle_added += pulse
                recycle_times.discard(rt)

        A_gut += recycle_added

        # ODE rates
        absorption_flux = ka_conj_per_h * A_gut  # mg/h leaving gut
        cleavage_flux = k_cleavage_per_h * A_conj  # mg/h cleaved from conj
        elim_conj_flux = k_elim_conj_per_h * A_conj  # mg/h eliminated as conj
        elim_drug_flux = k_elim_drug_per_h * C_drug  # (mg/L)/h eliminated as drug

        # Forward Euler update
        dA_gut = -absorption_flux
        dA_conj = absorption_flux - cleavage_flux - elim_conj_flux
        dC_drug = cleavage_flux / vd_drug_L - elim_drug_flux

        A_gut_new = max(A_gut + dA_gut * dt_h, 0.0)
        A_conj_new = max(A_conj + dA_conj * dt_h, 0.0)
        C_drug_new = max(C_drug + dC_drug * dt_h, 0.0)

        # Accumulate fluxes for summary (using current-step values)
        total_absorbed_conj += absorption_flux * dt_h
        total_converted += cleavage_flux * dt_h

        A_gut = A_gut_new
        A_conj = A_conj_new
        C_drug = C_drug_new
        t = t_next

        times.append(t)
        c_drug_traj.append(C_drug)
        a_conj_traj.append(A_conj)
        a_gut_traj.append(A_gut)

    # PK metrics for free drug
    cmax = max(c_drug_traj)
    tmax = times[c_drug_traj.index(cmax)]
    auc = _trapz(times, c_drug_traj)

    # Absorbed conjugate fraction (relative to total dose administered, including recycled)
    # Use the total dose absorbed divided by the initial dose
    f_absorbed = min(total_absorbed_conj / dose_mg, 1.0)
    f_converted = min(total_converted / dose_mg, 1.0)

    notes_parts = []
    if recycling_cycles > 0:
        notes_parts.append(
            f"Enterohepatic recycling: {recycling_cycles} cycle(s) at 6 h intervals."
        )
    if k_cleavage_per_h == 0.0:
        notes_parts.append("No enzymatic cleavage — no free drug released.")
    if not notes_parts:
        notes_parts.append("Standard bile acid conjugation PK simulation.")
    notes = " ".join(notes_parts)

    return BileAcidConjugationPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        ka_conj_per_h=ka_conj_per_h,
        k_cleavage_per_h=k_cleavage_per_h,
        times_h=times,
        c_plasma_drug_mg_L=c_drug_traj,
        a_plasma_conj_mg=a_conj_traj,
        a_gut_conjugate_mg=a_gut_traj,
        cmax_drug_mg_L=cmax,
        tmax_drug_h=tmax,
        auc_drug_mg_h_per_L=auc,
        f_absorbed_conj=f_absorbed,
        f_converted_to_drug=f_converted,
        notes=notes,
    )


def compare_cleavage_rates(
    drug_name: str,
    dose_mg: float,
    cleavage_rates: list[float],
) -> list[BileAcidConjugationPKResult]:
    """Compare outcomes across different enzymatic cleavage rates.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    cleavage_rates : list of float
        k_cleavage_per_h values to evaluate.

    Returns
    -------
    list of BileAcidConjugationPKResult sorted by cmax_drug_mg_L descending.
    """
    results = []
    for rate in cleavage_rates:
        result = simulate_bile_acid_conjugation_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            k_cleavage_per_h=rate,
        )
        results.append(result)

    results.sort(key=lambda r: r.cmax_drug_mg_L, reverse=True)
    return results
