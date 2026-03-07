"""Population PK simulation with inter-individual variability.

Simulates oral 1-compartment PK for a virtual population using LCG-based
random number generation and Box-Muller transforms for log-normal BSV.

Phase 835.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PopulationPKSimResult:
    """Summary statistics from a simulated population PK study."""

    drug_name: str
    n_subjects: int
    dose_mg: float
    tau_h: float
    # Population summary (from simulated subjects)
    mean_cmax_mg_L: float
    cv_cmax_pct: float
    p5_cmax_mg_L: float
    p95_cmax_mg_L: float
    mean_auc_mg_h_per_L: float
    cv_auc_pct: float
    p5_auc_mg_h_per_L: float
    p95_auc_mg_h_per_L: float
    mean_t_half_h: float
    fraction_above_target_pct: float  # % subjects with Cmax >= target_cmax
    target_cmax_mg_L: float
    n_doses_simulated: int
    notes: str


# ---------------------------------------------------------------------------
# LCG-based random number generation
# ---------------------------------------------------------------------------


def _lcg_next(state: int) -> tuple[int, float]:
    """Advance the LCG state and return (new_state, uniform_sample in [0, 1))."""
    new_state = (1664525 * state + 1013904223) % (2**32)
    return new_state, new_state / (2**32)


def _box_muller(state: int) -> tuple[int, float, float]:
    """Generate two standard normal samples from the LCG state using Box-Muller."""
    state, u1 = _lcg_next(state)
    state, u2 = _lcg_next(state)
    # Avoid log(0) by clamping u1 to a tiny minimum
    if u1 < 1e-15:
        u1 = 1e-15
    mag = math.sqrt(-2.0 * math.log(u1))
    two_pi_u2 = 2.0 * math.pi * u2
    z0 = mag * math.cos(two_pi_u2)
    z1 = mag * math.sin(two_pi_u2)
    return state, z0, z1


# ---------------------------------------------------------------------------
# Single-subject 1-compartment oral simulation (forward Euler)
# ---------------------------------------------------------------------------


def _simulate_subject_oral(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    tau_h: float,
    n_doses: int,
    dt_h: float = 0.1,
) -> tuple[list[float], list[float]]:
    """Simulate oral 1-cpt PK using forward Euler; return (times, concentrations).

    Dosing interval starts at t=0 for each dose. The simulation covers
    n_doses intervals, each of length tau_h.
    """
    ke = cl_L_per_h / vd_L
    total_time = n_doses * tau_h
    steps = int(round(total_time / dt_h)) + 1

    times: list[float] = []
    concs: list[float] = []

    # State: depot mass (mg), central amount (mg)
    depot = 0.0
    central = 0.0  # amount in mg

    dose_times = [i * tau_h for i in range(n_doses)]
    dose_idx = 0

    t = 0.0
    for _ in range(steps):
        # Apply dose if due (allow small tolerance)
        while dose_idx < len(dose_times) and t >= dose_times[dose_idx] - 1e-9:
            depot += dose_mg
            dose_idx += 1

        cp = central / vd_L  # mg/L
        times.append(t)
        concs.append(cp)

        # Forward Euler
        d_depot = -ka_per_h * depot
        d_central = ka_per_h * depot - ke * central

        depot += d_depot * dt_h
        central += d_central * dt_h
        if depot < 0.0:
            depot = 0.0
        if central < 0.0:
            central = 0.0

        t = round(t + dt_h, 10)

    return times, concs


# ---------------------------------------------------------------------------
# Population statistics helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _cv_pct(values: list[float]) -> float:
    """Coefficient of variation as a percentage."""
    m = _mean(values)
    if m == 0.0:
        return 0.0
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(variance) / m * 100.0


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Return the value at the given percentile (0–100) using nearest-rank."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    idx = int(math.ceil(pct / 100.0 * n)) - 1
    idx = max(0, min(idx, n - 1))
    return sorted_values[idx]


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Compute AUC via the trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1])
    return auc


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------


def simulate_population_pk(
    drug_name: str,
    dose_mg: float,
    n_subjects: int = 100,
    typical_cl_L_per_h: float = 5.0,
    typical_vd_L: float = 50.0,
    omega_cl_cv: float = 0.3,
    omega_vd_cv: float = 0.2,
    ka_per_h: float = 1.0,
    tau_h: float = 12.0,
    n_doses: int = 5,
    target_cmax_mg_L: float = 2.0,
    seed: int = 42,
) -> PopulationPKSimResult:
    """Simulate population PK with inter-individual variability.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose per administration in mg.
    n_subjects:
        Number of virtual subjects to simulate.
    typical_cl_L_per_h:
        Typical population clearance (L/h).
    typical_vd_L:
        Typical population volume of distribution (L).
    omega_cl_cv:
        BSV coefficient of variation for clearance (e.g., 0.3 = 30 %).
    omega_vd_cv:
        BSV coefficient of variation for volume of distribution.
    ka_per_h:
        First-order absorption rate constant (1/h), shared across subjects.
    tau_h:
        Dosing interval (h).
    n_doses:
        Number of doses to simulate.
    target_cmax_mg_L:
        Target Cmax threshold for fraction_above_target calculation.
    seed:
        LCG seed for deterministic generation.

    Returns
    -------
    PopulationPKSimResult
    """
    if n_subjects < 2:
        raise ValueError(f"n_subjects must be >= 2, got {n_subjects}")
    if n_doses < 1:
        raise ValueError(f"n_doses must be >= 1, got {n_doses}")
    if typical_cl_L_per_h <= 0.0:
        raise ValueError(f"typical_cl_L_per_h must be > 0, got {typical_cl_L_per_h}")
    if typical_vd_L <= 0.0:
        raise ValueError(f"typical_vd_L must be > 0, got {typical_vd_L}")
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")

    # Initialise LCG state
    lcg_state = (1664525 * seed + 1013904223) % (2**32)

    cmax_list: list[float] = []
    auc_list: list[float] = []
    t_half_list: list[float] = []

    # Omega parameters: omega = log(1 + cv^2)^0.5 for log-normal
    omega_cl = math.sqrt(math.log(1.0 + omega_cl_cv**2))
    omega_vd = math.sqrt(math.log(1.0 + omega_vd_cv**2))

    # Generate pairs of normal samples; we need 2 per subject (eta_cl, eta_vd)
    normals: list[float] = []
    while len(normals) < n_subjects * 2:
        lcg_state, z0, z1 = _box_muller(lcg_state)
        normals.append(z0)
        normals.append(z1)

    for i in range(n_subjects):
        eta_cl = normals[2 * i] * omega_cl
        eta_vd = normals[2 * i + 1] * omega_vd

        cl_i = typical_cl_L_per_h * math.exp(eta_cl)
        vd_i = typical_vd_L * math.exp(eta_vd)

        times, concs = _simulate_subject_oral(
            dose_mg=dose_mg,
            cl_L_per_h=cl_i,
            vd_L=vd_i,
            ka_per_h=ka_per_h,
            tau_h=tau_h,
            n_doses=n_doses,
            dt_h=0.1,
        )

        # Extract last dose interval
        last_dose_start = (n_doses - 1) * tau_h
        last_dose_end = n_doses * tau_h

        last_times: list[float] = []
        last_concs: list[float] = []
        for t_val, c_val in zip(times, concs):
            if t_val >= last_dose_start - 1e-9 and t_val <= last_dose_end + 1e-9:
                last_times.append(t_val)
                last_concs.append(c_val)

        if not last_concs:
            cmax_i = 0.0
            auc_i = 0.0
        else:
            cmax_i = max(last_concs)
            auc_i = _trapezoidal_auc(last_times, last_concs)

        ke_i = cl_i / vd_i
        t_half_i = math.log(2.0) / ke_i if ke_i > 0.0 else float("inf")

        cmax_list.append(cmax_i)
        auc_list.append(auc_i)
        t_half_list.append(t_half_i)

    # Population statistics
    sorted_cmax = sorted(cmax_list)
    sorted_auc = sorted(auc_list)

    mean_cmax = _mean(cmax_list)
    cv_cmax = _cv_pct(cmax_list)
    p5_cmax = _percentile(sorted_cmax, 5.0)
    p95_cmax = _percentile(sorted_cmax, 95.0)

    mean_auc = _mean(auc_list)
    cv_auc = _cv_pct(auc_list)
    p5_auc = _percentile(sorted_auc, 5.0)
    p95_auc = _percentile(sorted_auc, 95.0)

    mean_t_half = _mean(t_half_list)

    n_above = sum(1 for c in cmax_list if c >= target_cmax_mg_L)
    fraction_above = n_above / n_subjects * 100.0

    notes = (
        f"1-cpt oral model; n={n_subjects} subjects; "
        f"omega_cl_cv={omega_cl_cv:.2f}, omega_vd_cv={omega_vd_cv:.2f}; "
        f"seed={seed}"
    )

    return PopulationPKSimResult(
        drug_name=drug_name,
        n_subjects=n_subjects,
        dose_mg=dose_mg,
        tau_h=tau_h,
        mean_cmax_mg_L=mean_cmax,
        cv_cmax_pct=cv_cmax,
        p5_cmax_mg_L=p5_cmax,
        p95_cmax_mg_L=p95_cmax,
        mean_auc_mg_h_per_L=mean_auc,
        cv_auc_pct=cv_auc,
        p5_auc_mg_h_per_L=p5_auc,
        p95_auc_mg_h_per_L=p95_auc,
        mean_t_half_h=mean_t_half,
        fraction_above_target_pct=fraction_above,
        target_cmax_mg_L=target_cmax_mg_L,
        n_doses_simulated=n_doses,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Compare multiple populations
# ---------------------------------------------------------------------------


def compare_populations(
    drug_name: str,
    dose_mg: float,
    populations: list[dict[str, Any]],
) -> list[PopulationPKSimResult]:
    """Simulate and compare PK across multiple population parameter sets.

    Parameters
    ----------
    drug_name:
        Drug name shared across comparisons.
    dose_mg:
        Dose in mg shared across comparisons.
    populations:
        List of dicts; each dict may contain keys matching keyword arguments of
        ``simulate_population_pk`` (excluding drug_name and dose_mg).

    Returns
    -------
    list[PopulationPKSimResult]
        Results sorted by mean_cmax_mg_L in descending order.
    """
    results: list[PopulationPKSimResult] = []
    for pop_kwargs in populations:
        kwargs: dict[str, Any] = {k: v for k, v in pop_kwargs.items()}
        kwargs.pop("drug_name", None)
        kwargs.pop("dose_mg", None)
        result = simulate_population_pk(drug_name=drug_name, dose_mg=dose_mg, **kwargs)
        results.append(result)

    results.sort(key=lambda r: r.mean_cmax_mg_L, reverse=True)
    return results
