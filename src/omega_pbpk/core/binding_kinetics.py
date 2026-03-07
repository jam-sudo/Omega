"""Drug-receptor binding kinetics — kon/koff, residence time, target occupancy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BindingKineticsResult:
    drug_name: str
    kon_per_M_per_s: float
    koff_per_s: float
    kd_nM: float  # koff/kon in nM
    residence_time_s: float  # 1/koff
    drug_conc_nM: float
    target_conc_nM: float
    times_s: list[float]
    occupancy_pct: list[float]  # drug-receptor occupancy over time
    equilibrium_occupancy_pct: float
    t_half_on_s: float  # time to half equilibrium occupancy
    t_half_off_s: float  # 0.693/koff
    notes: str


def simulate_binding_kinetics(
    drug_name: str,
    kon_per_M_per_s: float,
    koff_per_s: float,
    drug_conc_nM: float,
    target_conc_nM: float,
    t_end_s: float = 3600.0,
    dt_s: float = 1.0,
) -> BindingKineticsResult:
    """Simulate drug-receptor binding by law of mass action (forward Euler).

    Parameters
    ----------
    kon_per_M_per_s : float
        Association rate constant (M^-1 s^-1).
    koff_per_s : float
        Dissociation rate constant (s^-1).
    drug_conc_nM : float
        Free drug concentration (nM) — assumed constant (ligand excess approximation).
    target_conc_nM : float
        Total target concentration (nM).
    t_end_s : float
        Simulation duration (s).
    dt_s : float
        Time step (s).
    """
    if kon_per_M_per_s <= 0:
        raise ValueError("kon_per_M_per_s must be > 0")
    if koff_per_s <= 0:
        raise ValueError("koff_per_s must be > 0")
    if drug_conc_nM < 0:
        raise ValueError("drug_conc_nM must be >= 0")
    if target_conc_nM <= 0:
        raise ValueError("target_conc_nM must be > 0")

    # Convert nM → M
    drug_M = drug_conc_nM * 1e-9
    target_M = target_conc_nM * 1e-9

    # Derived constants
    kd_M = koff_per_s / kon_per_M_per_s
    kd_nM = kd_M * 1e9

    residence_time_s = 1.0 / koff_per_s
    t_half_off_s = 0.693 / koff_per_s

    k_obs = kon_per_M_per_s * drug_M + koff_per_s
    t_half_on_s = 0.693 / k_obs if k_obs > 0 else float("inf")

    # Analytical equilibrium occupancy
    eq_occ = drug_M / (kd_M + drug_M) * 100.0

    # Forward Euler simulation
    n_steps = max(int(t_end_s / dt_s), 1)
    times: list[float] = []
    occupancy: list[float] = []

    dr = 0.0  # drug-receptor complex (M)
    t = 0.0
    times.append(t)
    occ = dr / target_M * 100.0 if target_M > 0 else 0.0
    occupancy.append(occ)

    for _ in range(n_steps):
        free_target = target_M - dr
        ddr = (kon_per_M_per_s * drug_M * free_target - koff_per_s * dr) * dt_s
        dr = max(0.0, dr + ddr)
        t += dt_s
        times.append(t)
        occ = dr / target_M * 100.0
        occupancy.append(occ)

    notes = f"Kd = {kd_nM:.2f} nM; RT = {residence_time_s:.1f} s; eq_occ = {eq_occ:.1f}%"

    return BindingKineticsResult(
        drug_name=drug_name,
        kon_per_M_per_s=kon_per_M_per_s,
        koff_per_s=koff_per_s,
        kd_nM=kd_nM,
        residence_time_s=residence_time_s,
        drug_conc_nM=drug_conc_nM,
        target_conc_nM=target_conc_nM,
        times_s=times,
        occupancy_pct=occupancy,
        equilibrium_occupancy_pct=eq_occ,
        t_half_on_s=t_half_on_s,
        t_half_off_s=t_half_off_s,
        notes=notes,
    )


def residence_time_analysis(
    drug_name: str,
    kon_per_M_per_s: float,
    koff_range: list[float],
    drug_conc_nM: float = 100.0,
    target_conc_nM: float = 10.0,
) -> list[BindingKineticsResult]:
    """Compare binding kinetics across a range of koff values.

    Parameters
    ----------
    koff_range : list[float]
        List of koff values (s^-1) to compare.

    Returns
    -------
    list[BindingKineticsResult]
        Results sorted by residence_time_s descending (longest RT first).
    """
    results = [
        simulate_binding_kinetics(
            drug_name=drug_name,
            kon_per_M_per_s=kon_per_M_per_s,
            koff_per_s=koff,
            drug_conc_nM=drug_conc_nM,
            target_conc_nM=target_conc_nM,
        )
        for koff in koff_range
    ]
    return sorted(results, key=lambda r: r.residence_time_s, reverse=True)


__all__ = [
    "BindingKineticsResult",
    "simulate_binding_kinetics",
    "residence_time_analysis",
]
